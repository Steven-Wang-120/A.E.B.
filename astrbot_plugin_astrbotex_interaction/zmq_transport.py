from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import zmq
import zmq.asyncio

PROTOCOL_NAME = "astrbotex-zmq"
PROTOCOL_VERSION = 1


class ZmqTransportError(RuntimeError):
    """Base error for the AstrBotEX ZeroMQ transport."""


class ZmqPeerUnavailable(ZmqTransportError):
    """Raised when no EX peer has completed the channel handshake."""


class ZmqRequestTimeout(ZmqTransportError, TimeoutError):
    """Raised when a request receives no response before its deadline."""


class ZmqRemoteError(ZmqTransportError):
    """Raised when the remote endpoint rejects a request."""


@dataclass(slots=True)
class ZmqReply:
    """Response data returned by a channel handler."""

    payload: dict[str, Any]
    binary: bytes | None = None


Handler = Callable[
    [bytes, dict[str, Any], bytes | None], Awaitable[ZmqReply | dict[str, Any] | None]
]
logger = logging.getLogger(__name__)


def _new_envelope(
    *,
    channel: str,
    kind: str,
    message_id: str | None = None,
    method: str,
    payload: dict[str, Any] | None = None,
    reply_to: str | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "channel": channel,
        "kind": kind,
        "id": message_id or uuid.uuid4().hex,
        "method": method,
        "timestamp": time.time(),
        "payload": payload or {},
    }
    if reply_to is not None:
        envelope["reply_to"] = reply_to
    return envelope


class ZmqRouterChannel:
    """Async ROUTER endpoint used by one independent EX data channel.

    The EX side should use one DEALER socket per channel and send a
    ``system.hello`` request before any application traffic. ROUTER/DEALER is
    used instead of REQ/REP so replies and unsolicited events can travel in
    either direction without a lock-step state machine.
    """

    def __init__(
        self,
        name: str,
        bind_endpoint: str,
        *,
        context: zmq.asyncio.Context | None = None,
        max_message_bytes: int = 64 * 1024 * 1024,
        request_timeout_sec: float = 10.0,
        high_water_mark: int = 100,
    ) -> None:
        self.name = name
        self.bind_endpoint = bind_endpoint
        self.max_message_bytes = max_message_bytes
        self.request_timeout_sec = request_timeout_sec
        self.high_water_mark = high_water_mark
        self.context = context or zmq.asyncio.Context.instance()
        self.socket: zmq.asyncio.Socket | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._handler_tasks: set[asyncio.Task[None]] = set()
        self._pending: dict[str, asyncio.Future[ZmqReply]] = {}
        self._pending_routes: dict[str, tuple[bytes, str]] = {}
        self._handlers: dict[str, Handler] = {}
        self._peer_last_seen: dict[bytes, float] = {}
        self._peer_hello_seen: dict[bytes, float] = {}
        self._default_peer: bytes | None = None
        self._send_lock = asyncio.Lock()
        self._closed = False

    @property
    def is_running(self) -> bool:
        return self.socket is not None and self._receive_task is not None

    @property
    def default_peer(self) -> bytes | None:
        return self._default_peer

    @property
    def peer_count(self) -> int:
        return len(self._peer_last_seen)

    def online_peers(self, *, max_age_sec: float = 30.0) -> tuple[bytes, ...]:
        now = time.monotonic()
        return tuple(peer for peer, seen in self._peer_hello_seen.items() if now - seen < max_age_sec)

    def register_handler(self, method: str, handler: Handler) -> None:
        if method in self._handlers:
            raise ValueError(f"duplicate ZeroMQ handler: {self.name}/{method}")
        self._handlers[method] = handler

    async def start(self) -> None:
        if self.is_running:
            return
        self._closed = False
        socket = self.context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        socket.setsockopt(zmq.RCVHWM, self.high_water_mark)
        socket.setsockopt(zmq.SNDHWM, self.high_water_mark)
        socket.setsockopt(zmq.MAXMSGSIZE, self.max_message_bytes)
        socket.bind(self.bind_endpoint)
        self.socket = socket
        self._receive_task = asyncio.create_task(
            self._receive_loop(),
            name=f"astrbotex-zmq-{self.name}-receive",
        )

    async def close(self) -> None:
        self._closed = True
        if self._receive_task is not None:
            self._receive_task.cancel()
            await asyncio.gather(self._receive_task, return_exceptions=True)
            self._receive_task = None

        tasks = list(self._handler_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._handler_tasks.clear()

        for future in self._pending.values():
            if not future.done():
                future.set_exception(ZmqTransportError("channel closed"))
        self._pending.clear()
        self._pending_routes.clear()

        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self._peer_last_seen.clear()
        self._peer_hello_seen.clear()
        self._default_peer = None

    async def request(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        peer: bytes | None = None,
        binary: bytes | None = None,
        timeout_sec: float | None = None,
        business_rejection_validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> ZmqReply:
        if business_rejection_validator is not None and method != "decision.goal.submit":
            raise ValueError("business rejection is only supported for goal submit")
        target = peer or self._default_peer
        if target is None:
            raise ZmqPeerUnavailable(f"no peer connected to {self.name}")

        message_id = uuid.uuid4().hex
        envelope = _new_envelope(
            channel=self.name,
            kind="request",
            message_id=message_id,
            method=method,
            payload=payload,
        )
        future: asyncio.Future[ZmqReply] = asyncio.get_running_loop().create_future()
        self._pending[message_id] = future
        self._pending_routes[message_id] = (target, method)
        try:
            await self._send(target, envelope, binary)
            timeout = self.request_timeout_sec if timeout_sec is None else timeout_sec
            try:
                reply = await asyncio.wait_for(future, timeout=timeout)
            except TimeoutError as exc:
                raise ZmqRequestTimeout(
                    f"{self.name}/{method} timed out after {timeout:.1f}s"
                ) from exc
            if not reply.payload.get("ok", True):
                if business_rejection_validator is not None and reply.binary is None:
                    try:
                        business_rejection_validator(reply.payload)
                    except (ValueError, TypeError):
                        pass
                    else:
                        return reply
                raise ZmqRemoteError(
                    str(reply.payload.get("error", "remote request failed"))
                )
            return reply
        finally:
            self._pending.pop(message_id, None)
            self._pending_routes.pop(message_id, None)

    async def send_event(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        peer: bytes | None = None,
        binary: bytes | None = None,
    ) -> None:
        target = peer or self._default_peer
        if target is None:
            raise ZmqPeerUnavailable(f"no peer connected to {self.name}")
        envelope = _new_envelope(
            channel=self.name,
            kind="event",
            method=method,
            payload=payload,
        )
        await self._send(target, envelope, binary)

    async def _receive_loop(self) -> None:
        assert self.socket is not None
        while not self._closed:
            try:
                frames = await self.socket.recv_multipart()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._closed:
                    raise ZmqTransportError(
                        f"{self.name} receive loop failed: {exc}"
                    ) from exc
                return

            if len(frames) < 2 or len(frames) > 3:
                logger.warning(
                    "%s dropped malformed frame count: %s",
                    self.name,
                    len(frames),
                )
                continue

            peer = frames[0]
            header = frames[1]
            binary = frames[2] if len(frames) == 3 else None
            if len(header) > self.max_message_bytes or (
                binary is not None and len(binary) > self.max_message_bytes
            ):
                logger.warning("%s dropped oversized frame", self.name)
                continue
            try:
                envelope = json.loads(header.decode("utf-8"))
                self._validate_envelope(envelope)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning("%s dropped invalid envelope: %s", self.name, exc)
                continue

            self._default_peer = peer
            self._peer_last_seen[peer] = time.time()
            if envelope["method"] == "system.hello" and envelope["kind"] == "request":
                self._peer_hello_seen[peer] = time.monotonic()
            elif peer in self._peer_hello_seen:
                self._peer_hello_seen[peer] = time.monotonic()
            kind = envelope["kind"]
            if kind == "response":
                reply_to = envelope.get("reply_to")
                future = self._pending.get(reply_to)
                if (future is not None and not future.done()
                        and self._pending_routes.get(reply_to) == (peer, envelope["method"])):
                    future.set_result(ZmqReply(envelope["payload"], binary))
                continue

            if kind not in {"request", "event"}:
                continue
            task = asyncio.create_task(
                self._handle_incoming(peer, envelope, binary),
                name=f"astrbotex-zmq-{self.name}-{envelope['method']}",
            )
            self._handler_tasks.add(task)
            task.add_done_callback(self._handler_tasks.discard)

    async def _handle_incoming(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> None:
        method = envelope["method"]
        handler = self._handlers.get(method)
        if handler is None and method == "system.hello":
            result: ZmqReply | dict[str, Any] | None = {
                "ok": True,
                "channel": self.name,
                "protocol": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
            }
        elif handler is None:
            result = {"ok": False, "error": f"unknown method: {method}"}
        else:
            try:
                result = await handler(peer, envelope, binary)
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": str(exc)}

        if envelope["kind"] != "request":
            return
        if isinstance(result, ZmqReply):
            reply = result
        elif result is None:
            reply = ZmqReply({"ok": True})
        else:
            reply = ZmqReply(result)
        response = _new_envelope(
            channel=self.name,
            kind="response",
            method=method,
            payload=reply.payload,
            reply_to=envelope["id"],
        )
        try:
            await self._send(peer, response, reply.binary)
        except Exception as exc:  # noqa: BLE001
            if not self._closed:
                logger.warning("%s response send failed: %s", self.name, exc)

    async def _send(
        self,
        peer: bytes,
        envelope: dict[str, Any],
        binary: bytes | None,
    ) -> None:
        if self.socket is None or self._closed:
            raise ZmqTransportError(f"channel {self.name} is not running")
        encoded = json.dumps(
            envelope,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > self.max_message_bytes:
            raise ValueError(f"{self.name} envelope exceeds message limit")
        if binary is not None and len(binary) > self.max_message_bytes:
            raise ValueError(f"{self.name} binary frame exceeds message limit")
        frames = [peer, encoded]
        if binary is not None:
            frames.append(binary)
        async with self._send_lock:
            await self.socket.send_multipart(frames)

    def _validate_envelope(self, envelope: Any) -> None:
        if not isinstance(envelope, dict):
            raise TypeError("envelope must be an object")
        if envelope.get("protocol") != PROTOCOL_NAME:
            raise ValueError("unsupported protocol")
        if envelope.get("version") != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol version")
        if envelope.get("channel") != self.name:
            raise ValueError("wrong channel")
        if envelope.get("kind") not in {"request", "response", "event"}:
            raise ValueError("invalid message kind")
        if not isinstance(envelope.get("id"), str):
            raise TypeError("message id is required")
        if not isinstance(envelope.get("method"), str):
            raise TypeError("method is required")
        if not isinstance(envelope.get("payload", {}), dict):
            raise TypeError("payload must be an object")
