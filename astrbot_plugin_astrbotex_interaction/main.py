from __future__ import annotations

import asyncio
import os
import tempfile
from collections import OrderedDict, defaultdict, deque
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from astrbot.core.agent.tool import ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.platform.astr_message_event import MessageSesion

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
MAX_VISION_STREAMS = 8


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
                    if isinstance(metadata, dict):
                        self._pending_routes[session_id].append((peer, dict(metadata)))

                    abm = AstrBotMessage()
                    abm.self_id = "astrbotex"
                    abm.sender = MessageMember(
                        user_id="astrbotex_robot",
                        nickname="AstrBotEX Robot",
                    )
                    abm.type = MessageType.FRIEND_MESSAGE
                    abm.session_id = session_id
                    abm.message = [Plain(text=text)]
                    abm.message_str = text
                    abm.raw_message = msg_data

                    event = AstrBotEXMessageEvent(
                        message_str=text,
                        message_obj=abm,
                        platform_meta=self.meta(),
                        session_id=session_id,
                        adapter=self,
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
    ) -> None:
        """Send a plain-text AstrBot reply over the text channel."""
        text = message_chain.get_plain_text().strip()
        if not text or self._text_channel is None:
            return

        session_id = str(getattr(session, "session_id", self._session_id))
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
    ) -> None:
        super().__init__(message_str, message_obj, platform_meta, session_id)
        self._adapter = adapter

    async def send(self, message: MessageChain) -> None:
        """Forward passive replies and preserve AstrBot event bookkeeping."""
        await self._adapter.forward_reply(self.session, message)
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


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


@star.register(
    "astrbot_plugin_astrbotex_interaction",
    "AstrBotEX Team",
    "Bridge AstrBot and AstrBotEX through independent ZeroMQ text, audio, and vision channels.",
    "0.4.0",
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
        self.vision_cache_streams = int(
            os.environ.get(
                "ASTRBOTEX_ZMQ_VISION_CACHE_STREAMS", str(MAX_VISION_STREAMS)
            )
        )
        self.text_channel: ZmqRouterChannel | None = None
        self.audio_channel: ZmqRouterChannel | None = None
        self.vision_channel: ZmqRouterChannel | None = None
        self._adapter: AstrBotEXPlatformAdapter | None = None
        self._vision_latest: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.context.add_llm_tools(SubmitAstrBotEXProposalTool(plugin=self))

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
        self.audio_channel.register_handler("providers.status", self._handle_providers)
        self.audio_channel.register_handler("stt.transcribe", self._handle_stt)
        self.audio_channel.register_handler("tts.synthesize", self._handle_tts)
        self.vision_channel.register_handler(
            "vision.publish", self._handle_vision_publish
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
    ) -> dict[str, Any]:
        if self.text_channel is None:
            raise ZmqTransportError("text channel is not initialized")
        reply = await self.text_channel.request(
            method,
            payload,
            timeout_sec=timeout_sec,
        )
        return reply.payload

    @filter.on_llm_request()
    async def inject_ex_context(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        try:
            context = await self.request_text("bridge.context.get", {})
        except ZmqTransportError as exc:
            logger.warning(f"Failed to fetch AstrBotEX context over ZeroMQ: {exc}")
            return
        req.extra_user_content_parts.append(
            TextPart(text=self._format_context_for_llm(context)).mark_as_temp()
        )

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
        self._adapter.inject_message(
            {"text": text, "session_id": session_id, "metadata": metadata},
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
        del peer
        if binary is not None and len(binary) > MAX_VISION_BYTES:
            return {"ok": False, "error": "vision frame exceeds 64 MiB limit"}
        data = dict(envelope.get("payload", {}))
        stream_id = str(data.get("stream_id", "default"))
        self._vision_latest[stream_id] = {
            "payload": data,
            "binary": binary,
            "received_at": asyncio.get_running_loop().time(),
        }
        self._vision_latest.move_to_end(stream_id)
        while len(self._vision_latest) > max(self.vision_cache_streams, 1):
            self._vision_latest.popitem(last=False)
        return {
            "ok": True,
            "stream_id": stream_id,
            "bytes": len(binary) if binary is not None else 0,
        }

    async def _handle_vision_status(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> dict[str, Any]:
        del peer, envelope, binary
        streams = []
        for stream_id, item in self._vision_latest.items():
            streams.append(
                {
                    "stream_id": stream_id,
                    "bytes": len(item["binary"]) if item["binary"] is not None else 0,
                    "payload": item["payload"],
                }
            )
        return {"ok": True, "streams": streams}

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
            "vision_streams": len(self._vision_latest),
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
        for channel in (self.vision_channel, self.audio_channel, self.text_channel):
            if channel is not None:
                await channel.close()
        self.vision_channel = None
        self.audio_channel = None
        self.text_channel = None
        logger.info("AstrBotEX ZeroMQ channels stopped.")
