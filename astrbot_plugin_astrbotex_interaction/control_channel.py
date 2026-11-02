"""AEB-side independent ZeroMQ ``control`` channel for upper step injection.

This is a fourth channel next to the existing ``text``/``audio``/``vision``
sockets.  It carries nothing but upper-step injection: it never falls back to
``interaction.message``, ``interaction.reply``, ``tts.synthesize`` or
``bridge.proposal.submit``, and the old three channels keep their sockets,
addresses and back-pressure domains untouched.

Transport reuse: :class:`~.zmq_transport.ZmqRouterChannel` already implements
ROUTER/DEALER framing, ``system.hello``, peer tracking, request correlation and
timeout handling, so this module wraps it instead of re-implementing a socket.

Two rules that shape everything below:

* The peer is always explicit.  ``default_peer`` is never consulted, because a
  control batch addressed to the wrong robot is worse than a refused batch.
* The outcome is biased towards ``delivery_unknown``.  Anything that could have
  reached the wire without a matching receipt is ambiguous, never "delivered",
  and never retried with a fresh id.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:  # pyzmq is already required by the plugin; keep the failure explicit anyway.
    import zmq as _zmq

    from .zmq_transport import (
        ZmqPeerUnavailable,
        ZmqRemoteError,
        ZmqRequestTimeout,
        ZmqRouterChannel,
        ZmqTransportError,
    )
    ZMQ_AVAILABLE = True
    ZMQ_ERROR = ""
    ZMQError: type[BaseException] = _zmq.ZMQError
except Exception as exc:  # pragma: no cover - exercised by the import guard test
    ZmqRouterChannel = None  # type: ignore[assignment]
    ZMQ_AVAILABLE = False
    ZMQ_ERROR = str(exc)

    class ZMQError(Exception):  # type: ignore[no-redef]
        pass

    class ZmqTransportError(RuntimeError):  # type: ignore[no-redef]
        pass

    class ZmqPeerUnavailable(ZmqTransportError):  # type: ignore[no-redef]
        pass

    class ZmqRequestTimeout(ZmqTransportError, TimeoutError):  # type: ignore[no-redef]
        pass

    class ZmqRemoteError(ZmqTransportError):  # type: ignore[no-redef]
        pass


SCHEMA_VERSION = "aeb.upper_steps.v1"
CHANNEL_NAME = "control"
DEFAULT_CONTROL_PORT = 8769
DEFAULT_BIND_HOST = "127.0.0.1"

CAPABILITIES_METHOD = "upper.capabilities.get"
INJECT_METHOD = "upper.steps.inject"

MAX_SEGMENTS = 64
MAX_SEGMENT_CHARS = 4096
MAX_TOTAL_BYTES = 256 * 1024
MAX_WIRE_ID_LEN = 128
DEFAULT_MAX_MESSAGE_BYTES = 512 * 1024
CAPABILITIES_TTL_SEC = 30.0

RECEIVED = "received"
INJECTED = "injected"
REJECTED = "rejected"
DELIVERY_UNKNOWN = "delivery_unknown"

# Codes a peer may legitimately answer with that prove the batch was refused
# before any model input was touched.  Anything else is treated as ambiguous.
DEFINITE_REJECTIONS = frozenset({
    "invalid_batch",
    "invalid_result",
    "invalid_probabilities",
    "payload_too_large",
    "route_mismatch",
    "session_mismatch",
    "id_conflict",
    "input_busy",
    "backend_unsupported",
})

PAYLOAD_KEYS = frozenset({
    "schema", "request_id", "plan_id", "plan_revision", "batch_id", "robot_id",
    "route_ref", "ex_session", "catalog_version", "prompt_version", "visibility",
    "segments",
})
SEGMENT_KEYS = frozenset({"segment_id", "ordinal", "text"})


class ControlError(RuntimeError):
    """A control-channel failure that is not a delivery outcome."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class DeliveryOutcome:
    """Result of one ``inject`` attempt.

    ``status`` is one of ``received``/``injected``/``rejected``/
    ``delivery_unknown``.  Only ``rejected`` is a definite "never reached the
    model"; only ``received``/``injected`` are transport acknowledgements.  None
    of them means the action ran.
    """

    status: str
    batch_id: str
    request_id: str
    payload_hash: str
    error_code: str | None = None
    sent: bool = False
    duplicate: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {RECEIVED, INJECTED}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "payload_hash": self.payload_hash,
            "error_code": self.error_code,
            "sent": self.sent,
            "duplicate": self.duplicate,
            "detail": dict(self.detail),
        }


def canonical_json(value: Any) -> str:
    """C01 canonical form: UTF-8, sorted keys, compact, non-finite rejected."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def payload_hash(payload: dict[str, Any], hasher: Callable[[dict], str] | None = None) -> str:
    if hasher is not None:
        return hasher(payload)
    import hashlib
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _wire_id(value: Any) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= MAX_WIRE_ID_LEN
            and value == value.strip()
            and not any(ch < " " or ch == "\x7f" for ch in value))


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class ControlChannel:
    """One ROUTER endpoint dedicated to upper-step injection."""

    def __init__(
        self,
        *,
        peer_id: str,
        robot_id: str,
        bind_host: str = DEFAULT_BIND_HOST,
        port: int = DEFAULT_CONTROL_PORT,
        request_timeout_sec: float = 10.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        channel_factory: Callable[..., Any] | None = None,
        hasher: Callable[[dict], str] | None = None,
        on_status: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not _wire_id(peer_id):
            raise ControlError("invalid_batch")
        if not _wire_id(robot_id):
            raise ControlError("invalid_batch")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ControlError("invalid_batch")
        if not isinstance(bind_host, str) or not bind_host.strip():
            raise ControlError("invalid_batch")
        self.peer_id = peer_id
        self.robot_id = robot_id
        self.bind_host = bind_host.strip()
        self.port = port
        self.request_timeout_sec = float(request_timeout_sec)
        self.max_message_bytes = int(max_message_bytes)
        self._factory = channel_factory
        self._hasher = hasher
        self._on_status = on_status
        self._clock = clock
        self._cache: dict[bytes, tuple[float, dict[str, Any]]] = {}
        # Every batch id this process has already put on the wire (or refused).
        # Re-injecting a registered batch is forbidden by C01 §5, so the guard
        # lives here rather than in each caller.
        self._attempted: dict[str, DeliveryOutcome] = {}
        self.channel: Any = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @property
    def endpoint(self) -> str:
        return f"tcp://{self.bind_host}:{self.port}"

    @property
    def available(self) -> bool:
        return ZMQ_AVAILABLE

    @property
    def is_running(self) -> bool:
        return self.channel is not None and bool(getattr(self.channel, "is_running", False))

    @property
    def peer(self) -> bytes:
        return self.peer_id.encode("utf-8")

    def _status(self, event: str, **detail: Any) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status({"channel": CHANNEL_NAME, "event": event, **detail})
        except Exception:  # noqa: BLE001 - a status sink must never break control
            pass

    async def start(self) -> None:
        """Bind the control socket.  Safe to call twice; the second call is a no-op."""
        if not ZMQ_AVAILABLE:
            raise ControlError("backend_unsupported")
        if self.is_running:
            return
        factory = self._factory or ZmqRouterChannel
        channel = factory(
            CHANNEL_NAME,
            self.endpoint,
            max_message_bytes=self.max_message_bytes,
            request_timeout_sec=self.request_timeout_sec,
        )
        await channel.start()
        self.channel = channel
        self._cache.clear()
        self._attempted.clear()
        self._status("started", endpoint=self.endpoint, peer_id=self.peer_id,
                     robot_id=self.robot_id)

    async def close(self) -> None:
        """Release the socket and every pending wait.  Safe to call repeatedly."""
        channel, self.channel = self.channel, None
        self._cache.clear()
        self._attempted.clear()
        if channel is not None:
            await channel.close()
        self._status("closed", endpoint=self.endpoint)

    # ------------------------------------------------------------------
    # peers
    # ------------------------------------------------------------------
    def online_peers(self) -> tuple[bytes, ...]:
        if self.channel is None:
            return ()
        try:
            return tuple(self.channel.online_peers())
        except Exception:  # noqa: BLE001
            return ()

    def resolve_peer(self, peer: bytes | None) -> bytes:
        """Return the explicit trusted peer, or refuse.

        There is no ``default_peer`` fallback: if the configured identity is not
        the one that completed ``system.hello``, control stays closed.
        """
        if not ZMQ_AVAILABLE:
            raise ControlError("backend_unsupported")
        if not self.is_running:
            raise ControlError("peer_unavailable")
        if peer is None:
            raise ControlError("peer_unavailable")
        if not isinstance(peer, bytes) or peer != self.peer:
            raise ControlError("peer_unavailable")
        if self.peer not in self.online_peers():
            raise ControlError("peer_unavailable")
        return peer

    def invalidate(self, peer: bytes | None = None) -> None:
        if peer is None:
            self._cache.clear()
        else:
            self._cache.pop(peer, None)

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    async def capabilities(self, peer: bytes | None = None, *, refresh: bool = False,
                           timeout_sec: float | None = None) -> dict[str, Any]:
        """Read the peer's read-only upper-control capability summary.

        Cached per peer + ``ex_session``.  The cache is dropped on close, on a
        ``session_mismatch`` answer, or when the peer stops being online, so a
        stale summary can never authorize a send.
        """
        target = self.resolve_peer(peer)
        if not refresh:
            cached = self._cache.get(target)
            if cached is not None:
                cached_at, summary = cached
                if (self._clock() - cached_at) < CAPABILITIES_TTL_SEC:
                    return summary
                self._cache.pop(target, None)

        request = {"schema": SCHEMA_VERSION}
        try:
            reply = await self.channel.request(
                CAPABILITIES_METHOD, request, peer=target,
                timeout_sec=self.request_timeout_sec if timeout_sec is None else timeout_sec,
            )
        except ZmqRequestTimeout as exc:
            raise ControlError("capabilities_unavailable") from exc
        except ZmqRemoteError as exc:
            code = str(exc)
            raise ControlError(code if code in DEFINITE_REJECTIONS else "capabilities_unavailable")
        except ZmqPeerUnavailable as exc:
            raise ControlError("peer_unavailable") from exc
        except ZMQError as exc:
            # ROUTER_MANDATORY raises this when the identity has no live pipe.
            raise ControlError("peer_unavailable") from exc
        except ZmqTransportError as exc:
            raise ControlError("capabilities_unavailable") from exc

        summary = self._validate_capabilities(reply.payload)
        self._cache[target] = (self._clock(), summary)
        return summary

    @staticmethod
    def _validate_capabilities(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ControlError("capabilities_unavailable")
        if payload.get("ok") is False:
            code = payload.get("error")
            raise ControlError(code if code in DEFINITE_REJECTIONS else "capabilities_unavailable")
        ex_session = payload.get("ex_session")
        if not _wire_id(ex_session):
            raise ControlError("capabilities_unavailable")
        input_ready = payload.get("input_ready")
        if not isinstance(input_ready, bool):
            raise ControlError("capabilities_unavailable")
        methods = payload.get("methods")
        if methods is not None:
            if not isinstance(methods, list) or not all(isinstance(m, str) for m in methods):
                raise ControlError("capabilities_unavailable")
            if INJECT_METHOD not in methods:
                raise ControlError("backend_unsupported")
        summary = dict(payload)
        summary["ex_session"] = ex_session
        summary["input_ready"] = input_ready
        return summary

    # ------------------------------------------------------------------
    # inject
    # ------------------------------------------------------------------
    async def inject(self, peer: bytes | None, batch: Any, *,
                     timeout_sec: float | None = None) -> DeliveryOutcome:
        """Send one frozen batch to the explicit peer and classify the outcome.

        ``batch`` is B04's frozen envelope projection (``payload``,
        ``payload_hash``, ``batch_id``, ``request_id``, ``ex_session``,
        ``robot_id``, ``route_ref``).  It is re-verified here, not trusted: the
        store is durable state, and a corrupted or edited row must not become a
        wire message.
        """
        batch_id = _field(batch, "batch_id", "") or ""
        request_id = _field(batch, "request_id", "") or ""
        frozen_hash = _field(batch, "payload_hash", "") or ""
        payload = _field(batch, "payload", None)

        def refuse(code: str) -> DeliveryOutcome:
            self._status("refused", batch_id=batch_id, error_code=code)
            return DeliveryOutcome(status=REJECTED, batch_id=batch_id,
                                   request_id=request_id, payload_hash=frozen_hash,
                                   error_code=code, sent=False)

        # A registered batch is never injected twice -- not after a rejection,
        # not after an unknown window, not after a reconnect.  The recorded
        # outcome is replayed so the caller records the same phase again.
        prior = self._attempted.get(batch_id) if batch_id else None
        if prior is not None:
            if prior.payload_hash != frozen_hash:
                return refuse("id_conflict")
            self._status("duplicate_injection", batch_id=batch_id, status=prior.status)
            return DeliveryOutcome(
                status=prior.status, batch_id=prior.batch_id, request_id=prior.request_id,
                payload_hash=prior.payload_hash, error_code=prior.error_code,
                sent=False, duplicate=True,
                detail={**prior.detail, "duplicate": True, "prior_status": prior.status})

        try:
            target = self.resolve_peer(peer)
        except ControlError as exc:
            return refuse(exc.code)

        code = self._verify_payload(payload, batch)
        if code is not None:
            return refuse(code)

        # Session binding: the frozen batch must target the session the peer
        # currently advertises.  A stale session is a refusal, not a retry.
        try:
            summary = await self.capabilities(target, timeout_sec=timeout_sec)
        except ControlError as exc:
            return refuse(exc.code)
        if summary["ex_session"] != _field(batch, "ex_session", None):
            self.invalidate(target)
            return refuse("session_mismatch")
        if not summary["input_ready"]:
            return refuse("input_busy")

        try:
            reply = await self.channel.request(
                INJECT_METHOD, payload, peer=target,
                timeout_sec=self.request_timeout_sec if timeout_sec is None else timeout_sec,
            )
        except ZmqRequestTimeout:
            # The frame may well have arrived.  Never claim otherwise.
            self._status("delivery_unknown", batch_id=batch_id, reason="timeout")
            return self._remember(batch_id, DeliveryOutcome(
                status=DELIVERY_UNKNOWN, batch_id=batch_id, request_id=request_id,
                payload_hash=frozen_hash, error_code="delivery_unknown", sent=True))
        except ZmqRemoteError as exc:
            answer = str(exc)
            if answer in DEFINITE_REJECTIONS:
                self._status("refused", batch_id=batch_id, error_code=answer)
                return self._remember(batch_id, DeliveryOutcome(
                    status=REJECTED, batch_id=batch_id, request_id=request_id,
                    payload_hash=frozen_hash, error_code=answer, sent=True))
            self._status("delivery_unknown", batch_id=batch_id, reason="remote_error")
            return self._remember(batch_id, DeliveryOutcome(
                status=DELIVERY_UNKNOWN, batch_id=batch_id, request_id=request_id,
                payload_hash=frozen_hash, error_code="delivery_unknown", sent=True))
        except ZmqPeerUnavailable:
            return refuse("peer_unavailable")
        except ZMQError:
            # ROUTER_MANDATORY raises before queueing, so the frame never left.
            return refuse("peer_unavailable")
        except ZmqTransportError:
            # A closed channel can mean "never left" or "left, no answer".
            # Ambiguity is recorded as unknown so the fence holds.
            self._status("delivery_unknown", batch_id=batch_id, reason="transport")
            return self._remember(batch_id, DeliveryOutcome(
                status=DELIVERY_UNKNOWN, batch_id=batch_id, request_id=request_id,
                payload_hash=frozen_hash, error_code="delivery_unknown", sent=True))

        outcome = self._classify_receipt(reply.payload, batch, target)
        self._status(outcome.status, batch_id=batch_id, error_code=outcome.error_code)
        return self._remember(batch_id, outcome)

    def _remember(self, batch_id: str, outcome: DeliveryOutcome) -> DeliveryOutcome:
        """Record a batch that reached the wire so it can never be re-sent."""
        if batch_id:
            self._attempted[batch_id] = outcome
        return outcome

    def _verify_payload(self, payload: Any, batch: Any) -> str | None:
        """Return an error code, or None when the envelope is safe to send."""
        if not isinstance(payload, dict):
            return "invalid_batch"
        if set(payload) != PAYLOAD_KEYS:
            return "invalid_batch"
        if payload.get("schema") != SCHEMA_VERSION:
            return "invalid_batch"
        if payload.get("visibility") != "decision_only":
            return "invalid_batch"
        for name in ("request_id", "plan_id", "batch_id", "robot_id", "route_ref",
                     "ex_session", "catalog_version", "prompt_version"):
            if not _wire_id(payload.get(name)):
                return "invalid_batch"
        revision = payload.get("plan_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            return "invalid_batch"
        if payload["batch_id"] != _field(batch, "batch_id", None):
            return "invalid_batch"
        if payload["request_id"] != _field(batch, "request_id", None):
            return "invalid_batch"
        if payload["ex_session"] != _field(batch, "ex_session", None):
            return "invalid_batch"
        if payload["robot_id"] != self.robot_id:
            return "session_mismatch"
        if payload["robot_id"] != _field(batch, "robot_id", None):
            return "invalid_batch"
        if payload["route_ref"] != _field(batch, "route_ref", None):
            return "invalid_batch"

        segments = payload.get("segments")
        if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_SEGMENTS:
            return "invalid_batch"
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict) or set(segment) != SEGMENT_KEYS:
                return "invalid_batch"
            if not _wire_id(segment.get("segment_id")):
                return "invalid_batch"
            ordinal = segment.get("ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
                return "invalid_batch"
            text = segment.get("text")
            if not isinstance(text, str) or not text.strip():
                return "invalid_batch"
            if len(text) > MAX_SEGMENT_CHARS:
                return "payload_too_large"

        expected_ordinal = _field(batch, "first_ordinal", None)
        if isinstance(expected_ordinal, int) and not isinstance(expected_ordinal, bool):
            if [s["ordinal"] for s in segments] != list(
                    range(expected_ordinal, expected_ordinal + len(segments))):
                return "invalid_batch"
        else:
            ordinals = [s["ordinal"] for s in segments]
            if ordinals != list(range(ordinals[0], ordinals[0] + len(ordinals))):
                return "invalid_batch"

        try:
            encoded = canonical_json(payload).encode("utf-8")
        except (TypeError, ValueError):
            return "invalid_batch"
        if len(encoded) > self.max_message_bytes or len(encoded) > MAX_TOTAL_BYTES:
            return "payload_too_large"
        if payload_hash(payload, self._hasher) != _field(batch, "payload_hash", None):
            return "invalid_batch"
        return None

    def _classify_receipt(self, receipt: Any, batch: Any, peer: bytes) -> DeliveryOutcome:
        """A receipt counts only when it answers this exact batch."""
        batch_id = _field(batch, "batch_id", "")
        request_id = _field(batch, "request_id", "")
        frozen_hash = _field(batch, "payload_hash", "")

        def unknown(reason: str) -> DeliveryOutcome:
            self._status("delivery_unknown", batch_id=batch_id, reason=reason)
            return DeliveryOutcome(status=DELIVERY_UNKNOWN, batch_id=batch_id,
                                   request_id=request_id, payload_hash=frozen_hash,
                                   error_code="delivery_unknown", sent=True)

        if not isinstance(receipt, dict):
            return unknown("receipt_shape")
        if receipt.get("ok") is False:
            code = receipt.get("error")
            if code in DEFINITE_REJECTIONS:
                return DeliveryOutcome(status=REJECTED, batch_id=batch_id,
                                       request_id=request_id, payload_hash=frozen_hash,
                                       error_code=code, sent=False)
            return unknown("remote_error")
        if receipt.get("batch_id") != batch_id:
            return unknown("receipt_batch_mismatch")
        if receipt.get("request_id") != request_id:
            return unknown("receipt_request_mismatch")
        if receipt.get("payload_hash") != frozen_hash:
            return unknown("receipt_hash_mismatch")
        phase = receipt.get("phase")
        if phase not in {RECEIVED, INJECTED, REJECTED}:
            return unknown("receipt_phase")
        if phase == REJECTED:
            code = receipt.get("error_code")
            return DeliveryOutcome(
                status=REJECTED, batch_id=batch_id, request_id=request_id,
                payload_hash=frozen_hash,
                error_code=code if code in DEFINITE_REJECTIONS else "invalid_batch",
                sent=False)
        return DeliveryOutcome(status=phase, batch_id=batch_id, request_id=request_id,
                               payload_hash=frozen_hash, error_code=None, sent=True,
                               detail={"ex_session": receipt.get("ex_session")})
