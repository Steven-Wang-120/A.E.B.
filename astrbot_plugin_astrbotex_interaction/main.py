from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import uuid
import os
import tempfile
from collections import defaultdict, deque
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mcp

from astrbot.api import FunctionTool, logger, star
from astrbot.api.event import AstrMessageEvent, MessageChain, MessageEventResult, filter
from astrbot.api.message_components import Plain
from astrbot.api.platform import (
    AstrBotMessage,
    MessageMember,
    MessageType,
    Platform,
    PlatformMetadata,
    register_platform_adapter,
)
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import ToolExecResult, ToolSet
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.platform.astr_message_event import MessageSesion

from .host_tasks import HostTaskContext, HostTaskOperations, execution_ready, host_message_key, parse_capabilities, task_projection
from .output_router import PublicOutputRouter
from .task_coordinator import TaskCoordinator
from .task_models import TaskAuthority, TaskError
from .task_store import TaskStore
from .zmq_transport import (
    ZmqReply,
    ZmqRouterChannel,
    ZmqTransportError,
)

DEFAULT_ZMQ_BIND_HOST = "0.0.0.0"
DEFAULT_TEXT_PORT = 8766
DEFAULT_AUDIO_PORT = 8767
DEFAULT_VISION_PORT = 8768
DEFAULT_SESSION_ID = "astrbotex_default"
DEFAULT_REQUEST_TIMEOUT_SEC = 10.0
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_VISION_BYTES = 64 * 1024 * 1024
MAX_VISION_CACHE_ITEMS = 8


def _endpoint(host: str, port: int) -> str:
    return f"tcp://{host}:{port}"


@register_platform_adapter("astrbotex", "AstrBotEX Interaction")
class AstrBotEXPlatformAdapter(Platform):
    """Expose AstrBotEX as a message platform backed by the text channel."""

    def __init__(
        self,
        platform_config: dict,
        platform_settings: dict,
        event_queue: asyncio.Queue,
    ) -> None:
        super().__init__(platform_config, event_queue)
        self.settings = platform_settings
        self.metadata = PlatformMetadata(
            name="astrbotex",
            description="AstrBotEX embodied robot interaction",
            id="astrbotex",
            support_proactive_message=True,
        )
        self._internal_queue: asyncio.Queue[tuple[dict[str, Any], bytes]] = (
            asyncio.Queue()
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._text_channel: ZmqRouterChannel | None = None
        self._session_id = DEFAULT_SESSION_ID
        self._request_timeout_sec = DEFAULT_REQUEST_TIMEOUT_SEC
        self._pending_routes: dict[str, deque[tuple[bytes, dict[str, Any]]]] = (
            defaultdict(deque)
        )

    def configure_transport(
        self,
        text_channel: ZmqRouterChannel,
        session_id: str,
        request_timeout_sec: float,
    ) -> None:
        self._text_channel = text_channel
        self._session_id = session_id
        self._request_timeout_sec = request_timeout_sec

    def inject_message(self, msg_data: dict[str, Any], peer: bytes) -> None:
        """Queue an EX message from the channel receive task."""
        if self._loop is None or self._loop.is_closed():
            logger.warning("AstrBotEX platform loop is not ready; message dropped.")
            return
        self._loop.call_soon_threadsafe(
            self._internal_queue.put_nowait, (msg_data, peer)
        )

    def run(self) -> Coroutine[Any, Any, None]:
        async def _loop() -> None:
            self._loop = asyncio.get_running_loop()
            try:
                while True:
                    msg_data, peer = await self._internal_queue.get()
                    text = str(msg_data.get("text", "")).strip()
                    session_id = str(msg_data.get("session_id", self._session_id))
                    if not text:
                        continue
                    metadata = msg_data.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    self._pending_routes[session_id].append((peer, dict(metadata)))

                    abm = AstrBotMessage()
                    abm.self_id = "astrbotex"
                    abm.sender = MessageMember(
                        user_id="astrbotex_robot",
                        nickname="AstrBotEX Robot",
                    )
                    abm.type = MessageType.FRIEND_MESSAGE
                    abm.session_id = session_id
                    abm.message_id = msg_data.get("message_id", "")
                    abm.message = [Plain(text=text)]
                    abm.message_str = text
                    abm.raw_message = msg_data

                    event = AstrBotEXMessageEvent(
                        message_str=text,
                        message_obj=abm,
                        platform_meta=self.meta(),
                        session_id=session_id,
                        adapter=self,
                        peer=peer,
                        route_metadata=metadata,
                    )
                    event.is_wake = True
                    self.commit_event(event)
            finally:
                self._loop = None

        return _loop()

    async def send_by_session(
        self,
        session: MessageSesion,
        message_chain: MessageChain,
    ) -> None:
        await super().send_by_session(session, message_chain)
        await self.forward_reply(session, message_chain)

    async def forward_reply(
        self,
        session: MessageSesion,
        message_chain: MessageChain,
        *,
        peer: bytes | None = None,
        route_metadata: dict[str, Any] | None = None,
        consume_pending: bool = True,
    ) -> None:
        """Send a plain-text AstrBot reply over the text channel."""
        text = message_chain.get_plain_text().strip()
        if not text or self._text_channel is None:
            return

        session_id = str(getattr(session, "session_id", self._session_id))
        if peer is not None:
            metadata = dict(route_metadata or {})
            if consume_pending:
                self._discard_pending_route(session_id, peer, metadata)
        else:
            route_and_metadata = self._pending_routes.get(session_id)
            if route_and_metadata:
                peer, metadata = route_and_metadata.popleft()
                if not route_and_metadata:
                    self._pending_routes.pop(session_id, None)
            else:
                peer = self._text_channel.default_peer
                metadata = {}
        if peer is None:
            logger.warning("AstrBotEX reply dropped: no text-channel peer.")
            return

        if not PublicOutputRouter.permits_automatic_output(
            metadata.get("source"), task_id=metadata.get("task_id")
        ) or metadata.get("visibility") == "private_planning":
            return
        payload: dict[str, Any] = {
            "text": text,
            "session_id": session_id,
            "type": "llm_reply",
        }
        for key in ("turn_id", "generation"):
            if key in metadata:
                payload[key] = metadata[key]
        try:
            await self._text_channel.request(
                "interaction.reply",
                payload,
                peer=peer,
                timeout_sec=self._request_timeout_sec,
            )
        except ZmqTransportError as exc:
            logger.warning(f"AstrBotEX reply request failed: {exc}")

    def _discard_pending_route(
        self,
        session_id: str,
        peer: bytes,
        metadata: dict[str, Any],
    ) -> None:
        """Remove the event-owned route while retaining later queued routes."""
        route_and_metadata = self._pending_routes.get(session_id)
        if not route_and_metadata:
            return
        try:
            route_and_metadata.remove((peer, metadata))
        except ValueError:
            return
        if not route_and_metadata:
            self._pending_routes.pop(session_id, None)

    def meta(self) -> PlatformMetadata:
        return self.metadata


class AstrBotEXMessageEvent(AstrMessageEvent):
    def __init__(
        self,
        message_str: str,
        message_obj: AstrBotMessage,
        platform_meta: PlatformMetadata,
        session_id: str,
        adapter: AstrBotEXPlatformAdapter,
        peer: bytes,
        route_metadata: dict[str, Any],
    ) -> None:
        super().__init__(message_str, message_obj, platform_meta, session_id)
        self._adapter = adapter
        self._peer = peer
        self._route_metadata = dict(route_metadata)
        self._route_consumed = False

    async def send(self, message: MessageChain) -> None:
        """Forward passive replies and preserve AstrBot event bookkeeping."""
        if (self._route_metadata.get("visibility") == "private_planning"
                or not PublicOutputRouter.permits_automatic_output(
                    self._route_metadata.get("source"), task_id=self._route_metadata.get("task_id"))):
            return
        await self._adapter.forward_reply(
            self.session,
            message,
            peer=self._peer,
            route_metadata=self._route_metadata,
            consume_pending=not self._route_consumed,
        )
        self._route_consumed = True
        await super().send(message)


@dataclass
class SubmitAstrBotEXProposalTool(FunctionTool[AstrAgentContext]):
    """Submit a validated high-level proposal to the EX text channel."""

    name: str = "submit_astrbotex_proposal"
    description: str = (
        "Submit a high-level AstrBotEX proposal. Use only action_id values listed "
        "in the latest AstrBotEX context affordances. Do not send motor speeds, "
        "CAN frames, wheel commands, or plugin method names."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "required": ["context_id", "commands"],
            "properties": {
                "context_id": {"type": "string"},
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["action_id", "params", "reason"],
                        "properties": {
                            "action_id": {"type": "string"},
                            "owner": {"type": "string"},
                            "uses_blocks": {
                                "type": "array",
                                "items": {"type": "object"},
                            },
                            "params": {"type": "object"},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        }
    )
    plugin: AstrBotEXInteractionPlugin | None = None

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ) -> ToolExecResult:
        if self.plugin is None:
            return "AstrBotEX ZeroMQ transport is not initialized."
        result = await self.plugin.request_text(
            "bridge.proposal.submit",
            {
                "context_id": str(kwargs.get("context_id", "")),
                "commands": kwargs.get("commands", []),
            },
        )
        return json_dumps(result)


class WokenHostTaskFilter(filter.CustomFilter):
    def filter(self, event: AstrMessageEvent, cfg) -> bool:
        # An all-message handler alone would wake otherwise-idle group traffic.
        return bool(event.is_wake or event.is_at_or_wake_command
                    or event.get_extra("provider_request"))


@dataclass
class ManageAstrBotEXTaskTool(FunctionTool[AstrAgentContext]):
    """One request's trusted Host binding; never registered in the global tool manager."""
    name: str = "manage_astrbotex_task"
    description: str = (
        "Create a robot task only for the user's requested robot work, or update/cancel/review your own active task. "
        "Supply business intent only, never task IDs, routes, users or physical commands. "
        "Normal conversation needs no task. Execution feedback, not admission, proves completion."
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object", "additionalProperties": False, "required": ["operation"],
        "properties": {"operation": {"type": "string", "enum": ["create", "update", "cancel", "review"]},
                       "text": {"type": "string", "minLength": 1, "maxLength": 8192}},
    })
    plugin: AstrBotEXInteractionPlugin | None = None
    host_context: HostTaskContext | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs: Any) -> ToolExecResult:
        try:
            if self.plugin is None or self.host_context is None:
                raise TaskError("task_entry_unavailable")
            operation = kwargs.get("operation")
            if operation not in {"create", "update", "cancel", "review"}:
                raise TaskError("invalid_host_operation")
            event = context.context.event
            if (event.unified_msg_origin != self.host_context.authority.session_id
                    or event.get_sender_id() != self.host_context.authority.user_id
                    or getattr(event.message_obj, "message_id", "") != self.host_context.message_id):
                raise TaskError("stale_host_context")
            current = await self.plugin._host_task_context(event, operation=operation)
            fields = ("authority", "peer", "message_id", "ex_session")
            if operation == "create":
                fields += ("provider_id",)
            if any(getattr(current, k) != getattr(self.host_context, k) for k in fields):
                raise TaskError("stale_host_context")
            result = await HostTaskOperations(self.plugin.task_coordinator).apply(current, kwargs)
            return json_dumps({"ok": True, "task_id": result["task_id"], "status": result["status"]})
        except Exception as exc:
            return json_dumps({"ok": False, "error": self.plugin._task_error_code(exc) if self.plugin else "task_entry_unavailable"})


@dataclass
class GetVisionJsonBufferTool(FunctionTool[AstrAgentContext]):
    """Return buffered vision JSON observations only when explicitly called."""

    name: str = "get_astrbotex_vision_json_buffer"
    description: str = (
        "Read buffered AstrBotEX YOLO/object-detection JSON on demand. Returns the "
        "latest JSON fields, or a bounded newest-first history. This tool must be "
        "called explicitly; vision JSON is not injected into prompts automatically."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "stream_id": {
                    "type": "string",
                    "description": "Optional vision stream ID. Omit to read the most recently updated stream.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of newest JSON records to return, up to the configured cache size.",
                    "default": 1,
                    "minimum": 1,
                },
            },
        }
    )
    plugin: AstrBotEXInteractionPlugin | None = None

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ) -> ToolExecResult:
        del context
        if self.plugin is None:
            return "AstrBotEX vision JSON buffer is not initialized."
        stream_id = str(kwargs.get("stream_id", "")).strip() or None
        try:
            limit = int(kwargs.get("limit", 1) or 1)
        except (TypeError, ValueError):
            limit = 1
        return self.plugin.get_vision_json_buffer_result(stream_id, limit)


@dataclass
class GetVisionJpegBufferTool(FunctionTool[AstrAgentContext]):
    """Return buffered vision JPEG frames only when explicitly called."""

    name: str = "get_astrbotex_vision_jpeg_buffer"
    description: str = (
        "Read buffered AstrBotEX JPEG frames on demand. Returns one selected JPEG "
        "as an MCP image content block plus routing metadata. This tool must be "
        "called explicitly; images are not injected into prompts automatically."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "stream_id": {
                    "type": "string",
                    "description": "Optional vision stream ID. Omit to read the most recently updated stream.",
                },
                "index": {
                    "type": "integer",
                    "description": "Newest-first image index. 0 returns the latest matching JPEG.",
                    "default": 0,
                    "minimum": 0,
                },
                "include_image": {
                    "type": "boolean",
                    "description": "Include the JPEG image content block in the result.",
                    "default": True,
                },
            },
        }
    )
    plugin: AstrBotEXInteractionPlugin | None = None

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ) -> ToolExecResult:
        del context
        if self.plugin is None:
            return "AstrBotEX vision JPEG buffer is not initialized."
        stream_id = str(kwargs.get("stream_id", "")).strip() or None
        try:
            index = int(kwargs.get("index", 0) or 0)
        except (TypeError, ValueError):
            index = 0
        include_image = bool(kwargs.get("include_image", True))
        return self.plugin.get_vision_jpeg_buffer_result(
            stream_id, index, include_image
        )


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


@star.register(
    "astrbot_plugin_astrbotex_interaction",
    "AstrBotEX Team",
    "Bridge AstrBot and AstrBotEX through independent ZeroMQ text, audio, and vision channels.",
    "0.5.0",
)
class AstrBotEXInteractionPlugin(star.Star):
    """Merged AstrBotEX platform, bridge, provider proxy, and ZMQ transport."""

    def __init__(self, context: star.Context) -> None:
        super().__init__(context)
        self.zmq_bind_host = os.environ.get(
            "ASTRBOTEX_ZMQ_BIND_HOST", DEFAULT_ZMQ_BIND_HOST
        )
        self.text_port = int(
            os.environ.get("ASTRBOTEX_ZMQ_TEXT_PORT", str(DEFAULT_TEXT_PORT))
        )
        self.audio_port = int(
            os.environ.get("ASTRBOTEX_ZMQ_AUDIO_PORT", str(DEFAULT_AUDIO_PORT))
        )
        self.vision_port = int(
            os.environ.get("ASTRBOTEX_ZMQ_VISION_PORT", str(DEFAULT_VISION_PORT))
        )
        self.session_id = os.environ.get("ASTRBOTEX_SESSION_ID", DEFAULT_SESSION_ID)
        self.request_timeout_sec = float(
            os.environ.get(
                "ASTRBOTEX_ZMQ_TIMEOUT_SEC", str(DEFAULT_REQUEST_TIMEOUT_SEC)
            )
        )
        self.vision_cache_items = int(
            os.environ.get(
                "ASTRBOTEX_ZMQ_VISION_CACHE_ITEMS",
                os.environ.get(
                    "ASTRBOTEX_ZMQ_VISION_CACHE_STREAMS",
                    str(MAX_VISION_CACHE_ITEMS),
                ),
            )
        )
        self.vision_cache_streams = self.vision_cache_items
        self.task_planning_enabled = os.environ.get("ASTRBOTEX_TASK_PLANNING", "auto").strip() != "0"
        self.task_provider_id = os.environ.get("ASTRBOTEX_TASK_PROVIDER_ID", "")
        self.task_store: TaskStore | None = None
        self.task_coordinator: TaskCoordinator | None = None
        self.task_robot_id = os.environ.get("ASTRBOTEX_TASK_ROBOT_ID", "")
        self.task_peer_id = os.environ.get("ASTRBOTEX_TASK_PEER_ID", "")
        self._task_routes: dict[str, dict[str, Any]] = {}
        self.text_channel: ZmqRouterChannel | None = None
        self.audio_channel: ZmqRouterChannel | None = None
        self.vision_channel: ZmqRouterChannel | None = None
        self._adapter: AstrBotEXPlatformAdapter | None = None
        self._vision_json_cache: deque[dict[str, Any]] = deque(
            maxlen=max(self.vision_cache_items, 1)
        )
        self._vision_jpeg_cache: deque[dict[str, Any]] = deque(
            maxlen=max(self.vision_cache_items, 1)
        )
        self.context.add_llm_tools(
            SubmitAstrBotEXProposalTool(plugin=self),
            GetVisionJsonBufferTool(plugin=self),
            GetVisionJpegBufferTool(plugin=self),
        )

    async def initialize(self) -> None:
        self.text_channel = ZmqRouterChannel(
            "text",
            _endpoint(self.zmq_bind_host, self.text_port),
            max_message_bytes=8 * 1024 * 1024,
            request_timeout_sec=self.request_timeout_sec,
        )
        self.audio_channel = ZmqRouterChannel(
            "audio",
            _endpoint(self.zmq_bind_host, self.audio_port),
            max_message_bytes=MAX_AUDIO_BYTES,
            request_timeout_sec=self.request_timeout_sec,
        )
        self.vision_channel = ZmqRouterChannel(
            "vision",
            _endpoint(self.zmq_bind_host, self.vision_port),
            max_message_bytes=MAX_VISION_BYTES,
            request_timeout_sec=self.request_timeout_sec,
        )

        self.text_channel.register_handler("interaction.message", self._handle_message)
        self.text_channel.register_handler("transport.status", self._handle_text_status)
        self.text_channel.register_handler("task.projection.get", self._handle_task_projection)
        self.text_channel.register_handler(
            "vision.json.publish", self._handle_vision_json_publish
        )
        self.text_channel.register_handler(
            "vision.json.status", self._handle_vision_json_status
        )
        self.audio_channel.register_handler("providers.status", self._handle_providers)
        self.audio_channel.register_handler("stt.transcribe", self._handle_stt)
        self.audio_channel.register_handler("tts.synthesize", self._handle_tts)
        self.vision_channel.register_handler(
            "vision.publish", self._handle_vision_publish
        )
        self.vision_channel.register_handler(
            "vision.jpeg.publish", self._handle_vision_jpeg_publish
        )
        self.vision_channel.register_handler(
            "vision.status", self._handle_vision_status
        )

        try:
            await self.text_channel.start()
            await self.audio_channel.start()
            await self.vision_channel.start()
        except Exception:
            await self.terminate()
            raise
        if self.task_planning_enabled and callable(getattr(self.context, "llm_generate", None)):
            try:
                db_path = os.environ.get("ASTRBOTEX_TASK_DB_PATH")
                if db_path is None:
                    data_dir = star.StarTools.get_data_dir("astrbot_plugin_astrbotex_interaction")
                    db_path = str(data_dir / "tasks.sqlite3")
                self.task_store = TaskStore(db_path)
                self._task_routes = {r["route_ref"]: r for r in self.task_store.routes()}
                router = PublicOutputRouter(self.task_store, self._send_task_message)
                self.task_coordinator = TaskCoordinator(
                    self.task_store, self._request_decision, host=self.context,
                    provider_id=self.task_provider_id, router=router, turn_sync=self._sync_task_turn,
                )
                self.task_coordinator.start()
                self.text_channel.register_handler("decision.feedback", self._handle_decision_feedback)
            except Exception:
                if self.task_store is not None:
                    self.task_store.close()
                    self.task_store = None
                self.task_coordinator = None
                logger.warning("AstrBotEX private planning dependencies unavailable; ordinary chat remains enabled.")
        await self._attach_platform(log_missing=True)
        logger.info(
            "AstrBotEX ZeroMQ channels started: text=%s, audio=%s, vision=%s",
            self.text_port,
            self.audio_port,
            self.vision_port,
        )

    @filter.on_platform_loaded()
    async def on_platform_loaded(self) -> None:
        await self._attach_platform(log_missing=False)

    async def _attach_platform(self, log_missing: bool) -> None:
        if self._adapter is None:
            for platform in self.context.platform_manager.platform_insts:
                if platform.meta().name == "astrbotex":
                    self._adapter = platform  # type: ignore[assignment]
                    break
        if self._adapter is None:
            if log_missing:
                logger.info(
                    "AstrBotEX platform adapter not found yet; waiting for platform load."
                )
            return
        if self.text_channel is not None:
            self._adapter.configure_transport(
                self.text_channel,
                self.session_id,
                self.request_timeout_sec,
            )

    async def request_text(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        timeout_sec: float | None = None,
        peer: bytes | None = None,
        business_rejection: bool = False,
    ) -> dict[str, Any]:
        if self.text_channel is None:
            raise ZmqTransportError("text channel is not initialized")
        if business_rejection and method == "decision.goal.submit":
            def validate_rejection(result):
                TaskCoordinator._validate_submit_result(payload, result)
                if result["ok"] is not False or result["phase"] != "rejected":
                    raise TaskError("invalid_submit_response")
            reply = await self.text_channel.request(
                method, payload, timeout_sec=timeout_sec, peer=peer,
                business_rejection_validator=validate_rejection,
            )
        else:
            reply = await self.text_channel.request(
                method, payload, timeout_sec=timeout_sec, peer=peer,
            )
        if method == "decision.capabilities.get" and reply.binary is not None:
            raise TaskError("invalid_capabilities")
        return reply.payload

    def admit_task_route(self, authority: TaskAuthority, peer: bytes, *, origin: str = "",
                         source_session: str = "") -> None:
        """Bind full trusted identity immutably; never consume a model/network claim."""
        authority.validate()
        data = {k: getattr(authority, k) for k in ("robot_id", "session_id", "user_id", "route_ref")}
        data.update(peer_hex=peer.hex(), origin=origin, source_session=source_session)
        prior = self._task_routes.get(authority.route_ref)
        if prior and prior != data:
            raise TaskError("route_owner_mismatch")
        if self.task_store is not None:
            self.task_store.bind_route(authority, peer, origin=origin, source_session=source_session)
        self._task_routes[authority.route_ref] = data

    def begin_robot_task(self, authority: TaskAuthority, text: str, request_id: str) -> dict:
        if self.task_coordinator is None:
            raise TaskError("task_planning_disabled")
        authority.validate()
        self._task_peer(authority.robot_id, authority.route_ref)
        route = self._task_routes[authority.route_ref]
        if any(route[k] != getattr(authority, k) for k in ("robot_id", "session_id", "user_id", "route_ref")):
            raise TaskError("route_owner_mismatch")
        return self.task_coordinator.create_task(authority, text, request_id)

    def _task_peer(self, robot_id: str, route_ref: str) -> bytes:
        route = self._task_routes.get(route_ref)
        if route is None and self.task_store is not None:
            route = self.task_store.route(route_ref)
        if route is None or route["robot_id"] != robot_id:
            raise TaskError("unbound_task_route")
        return bytes.fromhex(route["peer_hex"])

    def _resolve_task_binding(self, event: AstrMessageEvent) -> tuple[str, bytes]:
        bound_peer = None
        if isinstance(event, AstrBotEXMessageEvent):
            if event._adapter is not self._adapter:
                raise TaskError("untrusted_ex_event")
            bound_peer = event._peer
        if self.task_peer_id:
            configured = self.task_peer_id.encode("utf-8")
            if bound_peer is not None and bound_peer != configured:
                raise TaskError("task_peer_conflict")
            bound_peer = configured
        routes = self.task_store.routes() if self.task_store else []
        owned = [r for r in routes if r["session_id"] == event.unified_msg_origin
                 and r["user_id"] == event.get_sender_id()
                 and (not self.task_robot_id or r["robot_id"] == self.task_robot_id)]
        pairs = {(r["robot_id"], r["peer_hex"]) for r in owned}
        if len(pairs) > 1:
            raise TaskError("ambiguous_task_peer")
        if pairs:
            robot, peer_hex = next(iter(pairs))
            saved_peer = bytes.fromhex(peer_hex)
            if bound_peer is not None and bound_peer != saved_peer:
                raise TaskError("route_owner_mismatch")
            bound_peer = saved_peer
        else:
            robot = self.task_robot_id
        peers = self.text_channel.online_peers() if self.text_channel else ()
        if bound_peer is None:
            if len(peers) != 1:
                raise TaskError("ambiguous_task_peer" if peers else "task_peer_unavailable")
            bound_peer = peers[0]
        if bound_peer not in peers:
            raise TaskError("task_peer_unavailable")
        if not robot:
            robots = {r["robot_id"] for r in routes if r["peer_hex"] == bound_peer.hex()}
            if len(robots) > 1:
                raise TaskError("ambiguous_task_peer")
            # Transport peer identity is Host-owned, unlike robot claims in messages.
            robot = next(iter(robots)) if robots else "ex-peer-" + hashlib.sha256(bound_peer).hexdigest()
        return robot, bound_peer

    def _task_authority(self, event: AstrMessageEvent) -> TaskAuthority:
        if not self.task_planning_enabled or self.task_coordinator is None or self.task_store is None:
            raise TaskError("task_planning_disabled")
        from .task_contracts import require_id
        robot, peer = self._resolve_task_binding(event)
        origin, sender, session = event.unified_msg_origin, event.get_sender_id(), event.get_session_id()
        for key, value in (("origin", origin), ("sender", sender), ("session", session)):
            require_id(value, key)
        route_ref = hashlib.sha256(json.dumps([robot, origin, sender], ensure_ascii=False).encode()).hexdigest()
        auth = TaskAuthority(robot, origin, sender, route_ref, True)
        self.admit_task_route(auth, peer, origin=origin, source_session=session)
        return auth

    async def _task_provider(self, origin: str) -> str:
        from .task_contracts import require_id
        from astrbot.core.provider.provider import Provider
        provider_id = self.task_provider_id
        if not provider_id:
            resolve = getattr(self.context, "get_current_chat_provider_id", None)
            if not callable(resolve):
                raise TaskError("task_provider_unavailable")
            provider_id = await resolve(origin)
        require_id(provider_id, "provider_id")
        if not isinstance(self.context.get_provider_by_id(provider_id), Provider):
            raise TaskError("task_provider_unavailable")
        return provider_id

    async def _capabilities(self, peer: bytes) -> dict:
        # Always fresh, bounded and peer-specific; a prior message/session is not authority.
        raw = await self.request_text("decision.capabilities.get", {"schema_version": 1}, peer=peer,
                                      timeout_sec=min(self.request_timeout_sec, 3.0))
        return parse_capabilities(raw)

    async def _host_task_context(self, event: AstrMessageEvent, capabilities: dict | None = None,
                                 *, operation: str = "create") -> HostTaskContext:
        if not self.task_planning_enabled or self.task_coordinator is None or self.task_store is None:
            raise TaskError("task_planning_disabled")
        if operation not in {"create", "update", "cancel", "review", "offer"}:
            raise TaskError("invalid_host_operation")
        from .task_contracts import require_id
        message_id = getattr(event.message_obj, "message_id", "")
        require_id(message_id, "host_message_id")
        robot, peer = self._resolve_task_binding(event)
        capabilities = capabilities or await self._capabilities(peer)
        provider = ""
        ready = execution_ready(capabilities)
        if operation == "create" and not ready:
            raise TaskError("decision_execution_unavailable")
        if operation == "create" or (operation == "offer" and ready):
            try:
                provider = await self._task_provider(event.unified_msg_origin)
            except Exception:
                if operation == "create":
                    raise
        authority = self._task_authority(event)
        if authority.robot_id != robot or self._task_peer(robot, authority.route_ref) != peer:
            raise TaskError("route_owner_mismatch")
        binding = HostTaskContext(authority, peer, message_id, capabilities["ex_session"],
                                  capabilities["revision"], provider)
        if operation in {"update", "cancel", "review"} or (operation == "offer" and not provider):
            prior = self.task_store.host_message(binding.message_key)
            owned = (self.task_store.authorize(prior["task_id"], authority) if prior
                     else self.task_store.owned_active(authority))
            # Explicit review alone may migrate old EX authority, through Coordinator stop proof.
            if operation in {"update", "cancel"} and owned["ex_session"] != binding.ex_session:
                raise TaskError("stale_ex_session")
        return binding

    async def _handle_task_projection(self, peer: bytes, envelope: dict, binary: bytes | None) -> dict:
        if binary is not None:
            return {"ok": False, "error": "projection_binary_rejected"}
        if self.task_store is None or self.text_channel is None:
            return {"ok": False, "error": "projection_unavailable"}
        try:
            if peer not in self.text_channel.online_peers():
                raise TaskError("projection_peer_mismatch")
            routes = [r for r in self.task_store.routes() if r["peer_hex"] == peer.hex()]
            robots = {r["robot_id"] for r in routes}
            if len(robots) != 1:
                raise TaskError("projection_scope_unavailable")
            capabilities = await self._capabilities(peer)
            return task_projection(self.task_store, peer, envelope["payload"],
                                   ex_session=capabilities["ex_session"], robot_id=next(iter(robots)))
        except Exception as exc:
            return {"ok": False, "error": self._task_error_code(exc)}

    @staticmethod
    def _task_error_code(exc: Exception) -> str:
        code = getattr(exc, "code", "task_entry_unavailable")
        known = {"unauthorized", "task_planning_disabled", "robot_owned", "owner_mismatch", "unknown_task",
                 "ambiguous_task_peer", "task_peer_unavailable", "task_peer_conflict", "route_owner_mismatch",
                 "untrusted_ex_event", "unresolved_execution", "turn_sync_failed", "task_inactive", "empty_string",
                 "text_too_long", "invalid_type", "owned_task_unavailable", "duplicate_request_id_conflict",
                 "message_admission_unresolved", "task_provider_unavailable", "decision_execution_unavailable",
                 "business_intent_only", "invalid_host_operation", "stale_host_context", "stale_ex_session",
                 "projection_peer_mismatch", "projection_scope_unavailable", "invalid_projection_request",
                 "invalid_capabilities", "task_entry_unavailable"}
        return code if code in known else "task_entry_rejected"

    async def _sync_task_turn(self, payload: dict) -> None:
        # Fail closed on a non-acknowledging response; callback success gates tools.
        result = await self.request_text("interaction.task.turn", payload,
                               peer=self._task_peer(payload["robot_id"], payload["route_ref"]))
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise TaskError("turn_sync_failed")

    async def _request_decision(self, robot_id: str, route_ref: str, method: str, payload: dict) -> dict:
        from .task_contracts import parse_request
        parse_request(method, payload)
        return await self.request_text(method, payload, peer=self._task_peer(robot_id, route_ref),
                                       business_rejection=method == "decision.goal.submit")

    async def _send_task_message(self, payload: dict) -> dict:
        return await self.request_text("interaction.reply", payload,
            peer=self._task_peer(payload["robot_id"], payload["route_ref"]))

    async def _handle_decision_feedback(self, peer: bytes, envelope: dict, binary: bytes | None) -> dict:
        if binary is not None or self.task_coordinator is None or self.task_store is None:
            return {"ok": False, "error": "private_feedback_unavailable"}
        try:
            task = self.task_store.get(envelope["payload"].get("task_id"))
            if self._task_peer(task["robot_id"], task["route_ref"]) != peer:
                raise TaskError("owner_mismatch")
            return await self.task_coordinator.feedback(task["robot_id"], task["route_ref"], envelope["payload"])
        except Exception as exc:
            return {"ok": False, "error": getattr(exc, "code", "feedback_rejected")}

    def _host_reply_key(self, event: AstrMessageEvent) -> str | None:
        if self.task_store is None or event.get_extra("private_planning", False):
            return None
        if isinstance(event, AstrBotEXMessageEvent) and (
                event._adapter is not self._adapter
                or event._route_metadata.get("visibility") == "private_planning"
                or not PublicOutputRouter.permits_automatic_output(
                    event._route_metadata.get("source"), task_id=event._route_metadata.get("task_id"))):
            return None
        # Read only persisted Host routes; no transport/provider dependency during delivery/restart.
        message_id = getattr(event.message_obj, "message_id", "")
        if not message_id:
            return None
        for route in self.task_store.routes():
            if (route["session_id"] != event.unified_msg_origin
                    or route["user_id"] != event.get_sender_id()
                    or route["source_session"] != event.get_session_id()):
                continue
            if isinstance(event, AstrBotEXMessageEvent) and route["peer_hex"] != event._peer.hex():
                continue
            authority = TaskAuthority(*(route[k] for k in
                ("robot_id", "session_id", "user_id", "route_ref")), True)
            key = host_message_key(authority, message_id)
            if self.task_store.host_message(key):
                return key
        return None

    def _suppress_host_reply_replay(self, event: AstrMessageEvent) -> bool:
        key = self._host_reply_key(event)
        if key is None or not self.task_store.host_message(key)["public_claimed"]:
            return False
        original = event.get_extra("astrbotex_claimed_command_result")
        if (original and original[0] == key and original[1] is event.get_result()
                and event.get_extra("astrbotex_claimed_command_event") is event):
            return False  # Only this original command's still-pending first result.
        # No capabilities/provider IO: a persisted public receipt fences the entire
        # repeated default Host run, including run_agent's direct tool-status sends.
        event.should_call_llm(False)
        event.stop_event()
        event.clear_result()
        return True

    @filter.custom_filter(WokenHostTaskFilter)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def prepare_host_task_output(self, event: AstrMessageEvent) -> None:
        if self._suppress_host_reply_replay(event):
            return
        if event.get_extra("private_planning", False) or not self.task_planning_enabled:
            return
        if isinstance(event, AstrBotEXMessageEvent) and (
                event._route_metadata.get("visibility") == "private_planning"
                or not PublicOutputRouter.permits_automatic_output(
                    event._route_metadata.get("source"), task_id=event._route_metadata.get("task_id"))):
            return
        event.set_extra("astrbotex_public_output_prepared", False)
        try:
            if self._host_reply_key(event):
                event.set_extra("enable_streaming", False)
                event.set_extra("astrbotex_public_output_prepared", True)
                return
            _, peer = self._resolve_task_binding(event)
            capabilities = await self._capabilities(peer)
            if capabilities["control_mode"] == "decision":
                # Internal Host selects streaming before on_llm_request. Buffer before
                # that selection so later admission cannot bypass the public pre-send hook.
                event.set_extra("enable_streaming", False)
                event.set_extra("astrbotex_public_output_prepared", True)
        except Exception:
            return  # Unbound ordinary chat is untouched and never admitted here.

    @filter.on_decorating_result()
    async def guard_host_task_reply(self, event: AstrMessageEvent) -> None:
        from astrbot.core.message.message_event_result import ResultContentType
        result = event.get_result()
        if (result is None or not result.chain or result.result_content_type in
                {ResultContentType.STREAMING_RESULT, ResultContentType.STREAMING_FINISH}):
            return
        key = self._host_reply_key(event)
        if key is None:
            return  # Idle chat and private planning/feedback are not Host admissions.
        claimed = event.get_extra("astrbotex_claimed_command_result")
        if (claimed and claimed[0] == key and claimed[1] is result
                and event.get_extra("astrbotex_claimed_command_event") is event):
            event.set_extra("astrbotex_claimed_command_result", None)
            event.set_extra("astrbotex_claimed_command_event", None)
            return
        # Persist before send. Lost/crashed deliveries are not retried (at-most-once,
        # not exactly-once); the Host may split this one logical result into segments.
        if not self.task_store.claim_host_reply(key):
            event.clear_result()

    @filter.on_llm_request()
    async def inject_ex_context(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        if event is None or getattr(req, "private_planning", False) or event.get_extra("private_planning", False):
            return
        if self._suppress_host_reply_replay(event):
            return  # Also fences repeats if the early public message handler was bypassed.
        try:
            _, peer = self._resolve_task_binding(event)
            capabilities = await self._capabilities(peer)
        except Exception:
            # Unknown/ambiguous/offline EX does not grant legacy or task execution.
            original = req.func_tool
            req.func_tool = ToolSet(list(original.tools) if original else [])
            req.func_tool.remove_tool("submit_astrbotex_proposal")
            req.func_tool.remove_tool("manage_astrbotex_task")
            return
        if capabilities["control_mode"] == "decision":
            original = req.func_tool
            req.func_tool = ToolSet(list(original.tools) if original else [])
            req.func_tool.remove_tool("submit_astrbotex_proposal")
            req.func_tool.remove_tool("manage_astrbotex_task")
            if event.get_extra("astrbotex_public_output_prepared") is False:
                return  # Capability recovery after streaming selection is not admission authority.
            try:
                binding = await self._host_task_context(event, capabilities, operation="offer")
            except Exception:
                return  # Ordinary chat stays intact, without task admission.
            req.func_tool.add_tool(ManageAstrBotEXTaskTool(plugin=self, host_context=binding))
            summary = {"actions": [{"action_id": a["action_id"], "description": a["description"]}
                                   for a in capabilities["actions"]]}
            req.extra_user_content_parts.append(TextPart(text=(
                "AstrBotEX decision capabilities (not task admission). Use manage_astrbotex_task only "
                "for requested robot work; ordinary chat must not create tasks. " + json.dumps(summary, ensure_ascii=False)
            )).mark_as_temp())
            return
        try:
            context = await self.request_text("bridge.context.get", {}, peer=peer)
        except ZmqTransportError:
            logger.warning("Failed to fetch legacy AstrBotEX context over ZeroMQ.")
            return
        req.extra_user_content_parts.append(
            TextPart(text=self._format_context_for_llm(context)).mark_as_temp()
        )

    async def _task_command(self, event: AstrMessageEvent, operation: str, task_id: str = "", text: str = "") -> None:
        event.should_call_llm(False)
        event.stop_event()
        message_key = None
        try:
            binding = await self._host_task_context(event, operation=operation)
            message_key = binding.message_key
            if operation != "create" and task_id:
                # Compatibility ID is an ownership assertion, never an execution selector.
                prior = self.task_store.host_message(message_key)
                if prior:
                    owned = self.task_store.authorize(prior["task_id"], binding.authority)
                else:
                    try:
                        owned = self.task_store.owned_active(binding.authority)
                    except TaskError as exc:
                        raise TaskError("owner_mismatch") from exc
                if task_id != owned["task_id"]:
                    raise TaskError("owner_mismatch")
            intent = {"operation": operation}
            if operation in {"create", "update"}:
                intent["text"] = text
            await HostTaskOperations(self.task_coordinator).apply(binding, intent)
            # Successful compatibility commands are silent: no accepted placeholder.
        except Exception as exc:
            message_key = message_key or self._host_reply_key(event)
            if message_key and self.task_store.host_message(message_key):
                if not self.task_store.claim_host_reply(message_key):
                    event.set_result(MessageEventResult())
                    return
            code = self._task_error_code(exc)
            if task_id and code == "owned_task_unavailable":
                code = "owner_mismatch"
            result = MessageEventResult().message("AstrBotEX task: " + code)
            event.set_result(result)
            if message_key:
                event.set_extra("astrbotex_claimed_command_result", (message_key, result))
                event.set_extra("astrbotex_claimed_command_event", event)
            event.stop_event()

    @filter.command("ex_task")
    async def ex_task(self, event: AstrMessageEvent) -> None:
        parts = event.get_message_str().strip().split(maxsplit=1)
        await self._task_command(event, "create", text=parts[1] if len(parts) == 2 else "")

    @filter.command("ex_task_update")
    async def ex_task_update(self, event: AstrMessageEvent) -> None:
        parts = event.get_message_str().strip().split(maxsplit=2)
        await self._task_command(event, "update", parts[1] if len(parts) > 1 else "",
                                 parts[2] if len(parts) > 2 else "")

    @filter.command("ex_task_review")
    async def ex_task_review(self, event: AstrMessageEvent) -> None:
        parts = event.get_message_str().strip().split(maxsplit=1)
        await self._task_command(event, "review", parts[1] if len(parts) == 2 else "")

    @filter.command("ex_task_cancel")
    async def ex_task_cancel(self, event: AstrMessageEvent) -> None:
        parts = event.get_message_str().strip().split(maxsplit=1)
        await self._task_command(event, "cancel", parts[1] if len(parts) == 2 else "")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("ex_status")
    async def ex_status(self, event: AstrMessageEvent) -> None:
        """Show AstrBotEX ZeroMQ and runtime status."""
        try:
            status = await self.request_text("runtime.status", {})
        except ZmqTransportError as exc:
            event.set_result(MessageEventResult().message(f"AstrBotEX 连接失败: {exc}"))
            return
        event.set_result(
            MessageEventResult().message(
                f"AstrBotEX: {status.get('runtime_state', 'unknown')}, "
                f"text_peers={status.get('text_peers', 0)}, "
                f"vision_streams={status.get('vision_streams', 0)}"
            )
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("ex_context")
    async def ex_context(self, event: AstrMessageEvent) -> None:
        """Show the latest EX bridge context summary."""
        try:
            context = await self.request_text("bridge.context.get", {})
        except ZmqTransportError as exc:
            event.set_result(
                MessageEventResult().message(f"AstrBotEX context 获取失败: {exc}")
            )
            return
        event.set_result(
            MessageEventResult().message(self._format_context_summary(context))
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("ex_start")
    async def ex_start(self, event: AstrMessageEvent) -> None:
        """Start the AstrBotEX runtime through the text channel."""
        result = await self._submit_runtime_action("runtime.start.v1", {})
        event.set_result(MessageEventResult().message(json_dumps(result)))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("ex_stop")
    async def ex_stop(
        self,
        event: AstrMessageEvent,
        reason: str = "stopped from AstrBot",
    ) -> None:
        """Stop the AstrBotEX runtime through the text channel."""
        result = await self._submit_runtime_action(
            "runtime.stop.v1", {"reason": reason}
        )
        event.set_result(MessageEventResult().message(json_dumps(result)))

    async def _submit_runtime_action(
        self,
        action_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        context = await self.request_text("bridge.context.get", {})
        return await self.request_text(
            "bridge.proposal.submit",
            {
                "context_id": context.get("context_id", ""),
                "commands": [
                    {
                        "action_id": action_id,
                        "params": params,
                        "reason": "admin command from AstrBot",
                    }
                ],
            },
        )

    async def _handle_message(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        del binary
        data = envelope.get("payload", {})
        text = str(data.get("text", "")).strip()
        if not text:
            return {"ok": False, "error": "text is required"}
        if self._adapter is None:
            return {"ok": False, "error": "platform adapter unavailable"}
        session_id = str(data.get("session_id", self.session_id))
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if (metadata.get("visibility") == "private_planning"
                or not PublicOutputRouter.permits_automatic_output(
                    metadata.get("source"), task_id=metadata.get("task_id"))):
            return {"ok": False, "error": "private_task_requires_framework_admission"}
        self._adapter.inject_message(
            {"text": text, "session_id": session_id, "metadata": metadata,
             "message_id": envelope.get("id", "")},
            peer,
        )
        return {"ok": True, "session_id": session_id}

    async def _handle_text_status(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        del peer, envelope, binary
        return self._transport_status()

    async def _handle_providers(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        del peer, envelope, binary
        stt = self.context.get_using_stt_provider()
        tts = self._get_configured_tts_provider()
        return {
            "ok": True,
            "stt": type(stt).__name__ if stt is not None else None,
            "tts": type(tts).__name__ if tts is not None else None,
        }

    async def _handle_stt(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        del peer
        data = envelope.get("payload", {})
        audio_url = str(data.get("audio_url", ""))
        temporary_audio: Path | None = None
        if binary is not None:
            if len(binary) > MAX_AUDIO_BYTES:
                return {"ok": False, "error": "audio exceeds 25 MiB limit"}
            suffix = Path(str(data.get("filename", "audio.wav"))).suffix.lower()
            if not suffix or len(suffix) > 9 or not suffix[1:].isalnum():
                suffix = ".wav"
            with tempfile.NamedTemporaryFile(
                prefix="astrbotex_stt_",
                suffix=suffix,
                delete=False,
            ) as handle:
                handle.write(binary)
                temporary_audio = Path(handle.name)
            audio_url = str(temporary_audio)
        if not audio_url:
            return {"ok": False, "error": "audio or audio_url is required"}

        stt = self.context.get_using_stt_provider()
        if stt is None:
            if temporary_audio is not None:
                temporary_audio.unlink(missing_ok=True)
            return {"ok": False, "error": "provider not configured"}
        try:
            text = await stt.get_text(audio_url)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        finally:
            if temporary_audio is not None:
                temporary_audio.unlink(missing_ok=True)
        return {"ok": True, "text": text}

    async def _handle_tts(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> ZmqReply | dict[str, Any]:
        del peer, binary
        data = envelope.get("payload", {})
        text = str(data.get("text", "")).strip()
        if not text:
            return {"ok": False, "error": "text is required"}
        tts = self._get_configured_tts_provider()
        if tts is None:
            return {"ok": False, "error": "provider not configured"}
        try:
            audio_path = Path(str(await tts.get_audio(text)))
            audio_bytes = audio_path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            return {"ok": False, "error": "audio exceeds 25 MiB limit"}
        audio_format = audio_path.suffix.lower().lstrip(".") or "wav"
        return ZmqReply(
            {"ok": True, "audio_format": audio_format, "bytes": len(audio_bytes)},
            audio_bytes,
        )

    async def _handle_vision_publish(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        """Compatibility handler for older clients that sent JSON and JPEG together."""
        del peer
        if binary is not None and len(binary) > MAX_VISION_BYTES:
            return {"ok": False, "error": "vision frame exceeds 64 MiB limit"}
        data = dict(envelope.get("payload", {}))
        json_result = self._append_vision_json(data)
        if binary is not None:
            self._append_vision_jpeg(data, binary)
        return {
            "ok": True,
            "stream_id": json_result["stream_id"],
            "bytes": len(binary) if binary is not None else 0,
            "json_items": len(self._vision_json_cache),
            "jpeg_items": len(self._vision_jpeg_cache),
        }

    async def _handle_vision_json_publish(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        del peer
        if binary is not None:
            return {"ok": False, "error": "vision JSON channel does not accept binary frames"}
        data = dict(envelope.get("payload", {}))
        return {"ok": True, **self._append_vision_json(data)}

    async def _handle_vision_jpeg_publish(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        del peer
        if binary is None:
            return {"ok": False, "error": "JPEG binary frame is required"}
        if len(binary) > MAX_VISION_BYTES:
            return {"ok": False, "error": "vision frame exceeds 64 MiB limit"}
        data = dict(envelope.get("payload", {}))
        return {"ok": True, **self._append_vision_jpeg(data, binary)}

    def get_vision_json_buffer_result(
        self,
        stream_id: str | None = None,
        limit: int = 1,
    ) -> ToolExecResult:
        items = self._select_cache_items(self._vision_json_cache, stream_id, limit)
        if not items:
            return mcp.types.CallToolResult(
                content=[
                    mcp.types.TextContent(
                        type="text",
                        text=json_dumps(
                            {
                                "ok": False,
                                "error": self._empty_cache_message(
                                    "vision JSON buffer", stream_id
                                ),
                            }
                        ),
                    )
                ],
                isError=True,
            )

        result_payload = {
            "ok": True,
            "count": len(items),
            "cache_size": len(self._vision_json_cache),
            "items": [
                {
                    "stream_id": item["stream_id"],
                    "frame_id": item["frame_id"],
                    "received_at": item["received_at"],
                    "payload": item["payload"],
                }
                for item in items
            ],
        }
        return mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text=json_dumps(result_payload))]
        )

    def get_vision_jpeg_buffer_result(
        self,
        stream_id: str | None = None,
        index: int = 0,
        include_image: bool = True,
    ) -> ToolExecResult:
        safe_index = max(index, 0)
        items = self._select_cache_items(
            self._vision_jpeg_cache, stream_id, safe_index + 1
        )
        if len(items) <= safe_index:
            return mcp.types.CallToolResult(
                content=[
                    mcp.types.TextContent(
                        type="text",
                        text=json_dumps(
                            {
                                "ok": False,
                                "error": self._empty_cache_message(
                                    "vision JPEG buffer", stream_id
                                ),
                            }
                        ),
                    )
                ],
                isError=True,
            )

        item = items[safe_index]
        payload = dict(item["payload"])
        binary = item["binary"]
        mime_type = str(
            payload.get("frame_content_type")
            or payload.get("content_type")
            or "image/jpeg"
        ).lower()
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        image_returned = bool(
            include_image and binary is not None and mime_type.startswith("image/")
        )
        result_payload = {
            "ok": True,
            "stream_id": item["stream_id"],
            "frame_id": item["frame_id"],
            "index": safe_index,
            "bytes": len(binary) if binary is not None else 0,
            "mime_type": mime_type if binary is not None else None,
            "image_returned": image_returned,
            "payload": payload,
        }
        content: list[mcp.types.ContentBlock] = [
            mcp.types.TextContent(type="text", text=json_dumps(result_payload))
        ]
        if image_returned:
            content.append(
                mcp.types.ImageContent(
                    type="image",
                    data=base64.b64encode(binary).decode("ascii"),
                    mimeType=mime_type,
                )
            )
        return mcp.types.CallToolResult(content=content)

    async def _handle_vision_status(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        del peer, envelope, binary
        return {
            "ok": True,
            "json": self._cache_status(self._vision_json_cache, include_payload=True),
            "jpeg": self._cache_status(self._vision_jpeg_cache, include_payload=True),
        }

    async def _handle_vision_json_status(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        del peer, envelope, binary
        return {
            "ok": True,
            "json": self._cache_status(self._vision_json_cache, include_payload=True),
        }

    def _append_vision_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        stream_id = str(payload.get("stream_id", "default"))
        frame_id = payload.get("frame_id")
        self._vision_json_cache.append(
            {
                "stream_id": stream_id,
                "frame_id": frame_id,
                "payload": dict(payload),
                "received_at": asyncio.get_running_loop().time(),
            }
        )
        return {
            "stream_id": stream_id,
            "frame_id": frame_id,
            "items": len(self._vision_json_cache),
        }

    def _append_vision_jpeg(self, payload: dict[str, Any], binary: bytes) -> dict[str, Any]:
        stream_id = str(payload.get("stream_id", "default"))
        frame_id = payload.get("frame_id")
        image_payload = dict(payload)
        image_payload["frame_content_type"] = str(
            image_payload.get("frame_content_type") or "image/jpeg"
        )
        self._vision_jpeg_cache.append(
            {
                "stream_id": stream_id,
                "frame_id": frame_id,
                "payload": image_payload,
                "binary": binary,
                "received_at": asyncio.get_running_loop().time(),
            }
        )
        return {
            "stream_id": stream_id,
            "frame_id": frame_id,
            "bytes": len(binary),
            "items": len(self._vision_jpeg_cache),
        }

    @staticmethod
    def _select_cache_items(
        cache: deque[dict[str, Any]],
        stream_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        capped_limit = max(1, int(limit))
        matches: list[dict[str, Any]] = []
        for item in reversed(cache):
            if stream_id is not None and item["stream_id"] != stream_id:
                continue
            matches.append(item)
            if len(matches) >= capped_limit:
                break
        return matches

    @staticmethod
    def _empty_cache_message(cache_name: str, stream_id: str | None) -> str:
        if stream_id:
            return f"{cache_name} has no item for stream: {stream_id}"
        return f"{cache_name} is empty"

    @staticmethod
    def _cache_status(
        cache: deque[dict[str, Any]],
        *,
        include_payload: bool,
    ) -> dict[str, Any]:
        items = []
        for item in cache:
            value = {
                "stream_id": item["stream_id"],
                "frame_id": item["frame_id"],
                "received_at": item["received_at"],
                "bytes": len(item.get("binary") or b""),
            }
            if include_payload:
                value["payload"] = item["payload"]
            items.append(value)
        return {
            "items": items,
            "count": len(items),
        }

    def _transport_status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "channels": {
                "text": {
                    "port": self.text_port,
                    "peers": self.text_channel.peer_count if self.text_channel else 0,
                },
                "audio": {
                    "port": self.audio_port,
                    "peers": self.audio_channel.peer_count if self.audio_channel else 0,
                },
                "vision": {
                    "port": self.vision_port,
                    "peers": self.vision_channel.peer_count
                    if self.vision_channel
                    else 0,
                },
            },
            "vision_streams": max(
                len(self._vision_json_cache), len(self._vision_jpeg_cache)
            ),
            "vision_json_items": len(self._vision_json_cache),
            "vision_jpeg_items": len(self._vision_jpeg_cache),
        }

    def _get_configured_tts_provider(self) -> Any | None:
        config = self.context.get_config()
        provider_id = str(
            config.get("provider_tts_settings", {}).get("provider_id", "")
        ).strip()
        if not provider_id:
            return None
        provider = self.context.get_provider_by_id(provider_id)
        if provider not in self.context.get_all_tts_providers():
            logger.warning(
                f"Configured AstrBotEX TTS provider is unavailable: {provider_id}"
            )
            return None
        return provider

    @staticmethod
    def _format_context_for_llm(context: dict[str, Any]) -> str:
        payload = {
            "context_id": context.get("context_id"),
            "blocks": context.get("blocks", []),
            "affordances": context.get("affordances", []),
            "proposal_schema": context.get("proposal_schema", {}),
            "rules": context.get("rules", []),
        }
        return (
            "<astrbotex_context>\n"
            f"{json_dumps(payload)}\n"
            "</astrbotex_context>\n"
            "When controlling AstrBotEX, call submit_astrbotex_proposal with the context_id above."
        )

    @staticmethod
    def _format_context_summary(context: dict[str, Any]) -> str:
        blocks = context.get("blocks", [])
        actions = context.get("affordances", [])
        fresh_blocks = [block for block in blocks if block.get("fresh")]
        return (
            f"context_id={context.get('context_id')}\n"
            f"blocks={len(blocks)}, fresh={len(fresh_blocks)}, actions={len(actions)}\n"
            f"actions: {', '.join(str(item.get('action_id')) for item in actions) or '--'}"
        )

    async def terminate(self) -> None:
        if self.task_coordinator is not None:
            await self.task_coordinator.close()
            self.task_coordinator = None
        if self.task_store is not None:
            self.task_store.close()
            self.task_store = None
        self._task_routes.clear()
        for channel in (self.vision_channel, self.audio_channel, self.text_channel):
            if channel is not None:
                await channel.close()
        self.vision_channel = None
        self.audio_channel = None
        self.text_channel = None
        logger.info("AstrBotEX ZeroMQ channels stopped.")
