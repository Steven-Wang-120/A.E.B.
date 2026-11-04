"""B00 upper-control contracts, AEB side.

This module freezes the C01 upper-control payloads into one executable parser.
It is a *new namespace*: nothing here widens or renames the old
``decision.*`` / ``Goal`` / ``DecisionSnapshot`` contracts, and the frozen
transport envelope id is never reused as a business id.

Design rules enforced here (B00 §3):

* Strict key sets. A payload with a missing key, an extra key, a ``bool``
  standing in for an integer, a string probability, ``NaN``/``Inf``, or an
  out-of-range value is rejected. Nothing is silently dropped or coerced.
* Business identity (``robot_id``, ``route_ref``, ``ex_session``,
  ``plan_id``, ``batch_id``, segment ids) is framework-owned. A model result
  may only carry text, so ``UpperResult`` cannot mint an id and
  ``TrustedUpperInput`` cannot be built from a dict.
* Text is preserved verbatim, including the user's own numbers and units.
  Segment boundaries are never merged or truncated.
* Over-limit payloads raise ``payload_too_large``; a truncated payload is
  never reported as success.
* ``canonical``/``canonical_hash`` are the only accepted normalisation, so a
  byte-identical fixture yields a byte-identical hash on both ends.

The EX mirror is ``astrbot_ex/core/step_input_contracts.py``. The two files are
deliberately isomorphic but not shared at runtime: neither repository imports
the other.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Wire constants (C01 §1, §4)
# --------------------------------------------------------------------------

UPPER_STEPS_SCHEMA = "aeb.upper_steps.v1"

ROUTE_SPEECH_ONLY = "speech_only"
ROUTE_ACTION_ONLY = "action_only"
ROUTE_SPEECH_AND_ACTION = "speech_and_action"

#: Fixed tie-break order for equal probabilities. C01 §1: speech_only, then
#: speech_and_action, then action_only. No fourth class, no hidden threshold.
ROUTE_TIE_ORDER: tuple[str, ...] = (
    ROUTE_SPEECH_ONLY,
    ROUTE_SPEECH_AND_ACTION,
    ROUTE_ACTION_ONLY,
)
ROUTE_SET = frozenset(ROUTE_TIE_ORDER)

VISIBILITY_DECISION_ONLY = "decision_only"

PHASE_RECEIVED = "received"
PHASE_INJECTED = "injected"
PHASE_REJECTED = "rejected"
RECEIPT_PHASES = frozenset({PHASE_RECEIVED, PHASE_INJECTED, PHASE_REJECTED})

#: C01 §1 limits. Exceeding any of them is a hard rejection.
MAX_SEGMENTS = 64
MAX_SEGMENT_CHARS = 4096
MAX_PUBLIC_TEXT_CHARS = 4096
MAX_PAYLOAD_BYTES = 256 * 1024
MAX_WIRE_ID_LEN = 128

PROBABILITY_TOLERANCE = 0.001

#: A wire id is a restricted string, not an arbitrary path or shell token.
_WIRE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class ErrorCode:
    """Published rejection codes (B00 §3.6).

    The three groups are deliberately distinct so a consumer can never mistake
    a malformed payload for a state rejection or for an ambiguous delivery.
    """

    OK = "ok"

    # --- malformed payload: nothing was accepted, nothing was sent ---------
    INVALID_PROBABILITIES = "invalid_probabilities"
    INVALID_RESULT = "invalid_result"
    INVALID_BATCH = "invalid_batch"
    PAYLOAD_TOO_LARGE = "payload_too_large"

    # --- state rejection: well formed but must not proceed -----------------
    ROUTE_MISMATCH = "route_mismatch"
    SESSION_MISMATCH = "session_mismatch"
    ID_CONFLICT = "id_conflict"
    INPUT_BUSY = "input_busy"
    BACKEND_UNSUPPORTED = "backend_unsupported"

    # --- ambiguous delivery: outcome unknown, never retried as success -----
    DELIVERY_UNKNOWN = "delivery_unknown"


FORMAT_ERROR_CODES = frozenset(
    {
        ErrorCode.INVALID_PROBABILITIES,
        ErrorCode.INVALID_RESULT,
        ErrorCode.INVALID_BATCH,
        ErrorCode.PAYLOAD_TOO_LARGE,
    }
)
STATE_ERROR_CODES = frozenset(
    {
        ErrorCode.ROUTE_MISMATCH,
        ErrorCode.SESSION_MISMATCH,
        ErrorCode.ID_CONFLICT,
        ErrorCode.INPUT_BUSY,
        ErrorCode.BACKEND_UNSUPPORTED,
    }
)
UNCERTAIN_ERROR_CODES = frozenset({ErrorCode.DELIVERY_UNKNOWN})


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationError:
    """One rejection reason. ``path`` is a dotted/bracketed JSON pointer."""

    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "path": self.path, "message": self.message}


class UpperContractError(ValueError):
    """Raised when an upper-control payload must be rejected."""

    def __init__(self, error: ValidationError) -> None:
        super().__init__(f"{error.code}: {error.path}: {error.message}")
        self.error = error

    @property
    def code(self) -> str:
        return self.error.code


def err(code: str, path: str, message: str) -> ValidationError:
    return ValidationError(code=code, path=path, message=message)


def reject(code: str, path: str, message: str) -> None:
    raise UpperContractError(err(code, path, message))


# --------------------------------------------------------------------------
# Canonical JSON and hashing (B00 §3.4)
# --------------------------------------------------------------------------


def canonical(value: Any, *, code: str = ErrorCode.INVALID_BATCH) -> str:
    """Return the one canonical JSON spelling used for hashing.

    UTF-8, ``ensure_ascii=False``, sorted keys, compact separators, and no
    non-finite numbers. Anything else is rejected rather than normalised, so a
    hash can never hide a value the parser would refuse.
    """
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise UpperContractError(err(code, "$", "value is not canonical JSON")) from exc


def canonical_bytes(value: Any, *, code: str = ErrorCode.INVALID_BATCH) -> bytes:
    try:
        return canonical(value, code=code).encode("utf-8")
    except UnicodeEncodeError as exc:  # lone surrogates
        raise UpperContractError(err(code, "$", "text is not valid UTF-8")) from exc


def canonical_hash(value: Any, *, code: str = ErrorCode.INVALID_BATCH) -> str:
    """SHA-256 over the canonical JSON bytes."""
    return hashlib.sha256(canonical_bytes(value, code=code)).hexdigest()


def _require_budget(payload: Any, path: str = "$") -> None:
    """Reject an over-limit payload before it can be partially consumed."""
    try:
        size = len(canonical_bytes(payload))
    except UpperContractError:
        raise
    if size > MAX_PAYLOAD_BYTES:
        reject(
            ErrorCode.PAYLOAD_TOO_LARGE,
            path,
            f"payload is {size} bytes, limit is {MAX_PAYLOAD_BYTES}",
        )


# --------------------------------------------------------------------------
# Primitive guards. ``bool`` is never accepted as a number, anywhere.
# --------------------------------------------------------------------------


def _require_object(value: Any, path: str, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        reject(code, path, "must be an object")
    return value


def _require_exact_keys(
    value: dict[str, Any], expected: frozenset[str], path: str, code: str
) -> None:
    missing = sorted(expected - set(value))
    if missing:
        reject(code, f"{path}.{missing[0]}", "missing required field")
    unknown = sorted(set(value) - expected)
    if unknown:
        # An unknown field is a rejection, never a field we quietly drop.
        reject(code, f"{path}.{unknown[0]}", "unknown field is not accepted")


def _require_text(
    value: Any, path: str, code: str, *, max_len: int, preserve: bool = True
) -> str:
    if not isinstance(value, str):
        reject(code, path, "must be a string")
    text = value if preserve else value.strip()
    if not text.strip():
        reject(code, path, "must not be blank")
    if len(text) > max_len:
        reject(code, path, f"must not exceed {max_len} characters")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UpperContractError(err(code, path, "must be valid UTF-8")) from exc
    return text


def _require_optional_text(
    value: Any, path: str, code: str, *, max_len: int
) -> str | None:
    if value is None:
        return None
    return _require_text(value, path, code, max_len=max_len)


def _require_int(
    value: Any,
    path: str,
    code: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    # ``type(...) is int`` rejects bool, which ``isinstance(True, int)`` allows.
    if type(value) is not int:
        reject(code, path, "must be an integer")
    if minimum is not None and value < minimum:
        reject(code, path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        reject(code, path, f"must be <= {maximum}")
    return value


def _require_number(value: Any, path: str, code: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        reject(code, path, "must be a real number")
    number = float(value)
    if not math.isfinite(number):
        reject(code, path, "must be finite")
    return number


def _require_wire_id(value: Any, path: str, code: str) -> str:
    if not isinstance(value, str):
        reject(code, path, "must be a string")
    if not value:
        reject(code, path, "must not be empty")
    if len(value) > MAX_WIRE_ID_LEN:
        reject(code, path, f"must not exceed {MAX_WIRE_ID_LEN} characters")
    if not _WIRE_ID_RE.match(value):
        reject(code, path, "must match [A-Za-z0-9][A-Za-z0-9._:-]*")
    return value


def is_wire_id(value: Any) -> bool:
    """Non-raising form used by callers that only need a visibility decision."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_WIRE_ID_LEN
        and _WIRE_ID_RE.match(value) is not None
    )


# --------------------------------------------------------------------------
# Route probabilities (C01 §1)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteProbabilities:
    """The decision model's only permitted output: three named probabilities."""

    speech_only: float
    action_only: float
    speech_and_action: float

    _KEYS = frozenset({"speech_only", "action_only", "speech_and_action"})

    @classmethod
    def parse(cls, payload: Any) -> "RouteProbabilities":
        data = _require_object(payload, "$", ErrorCode.INVALID_PROBABILITIES)
        _require_exact_keys(
            data, cls._KEYS, "$", ErrorCode.INVALID_PROBABILITIES
        )
        values = {
            name: _require_number(
                data[name], f"$.{name}", ErrorCode.INVALID_PROBABILITIES
            )
            for name in ("speech_only", "action_only", "speech_and_action")
        }
        for name, number in values.items():
            if number < 0.0 or number > 1.0:
                reject(
                    ErrorCode.INVALID_PROBABILITIES,
                    f"$.{name}",
                    "must be within [0, 1]",
                )
        total = sum(values.values())
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            reject(
                ErrorCode.INVALID_PROBABILITIES,
                "$",
                f"probabilities sum to {total!r}, tolerance is {PROBABILITY_TOLERANCE}",
            )
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return {
            "speech_only": self.speech_only,
            "action_only": self.action_only,
            "speech_and_action": self.speech_and_action,
        }

    def route(self) -> str:
        """Framework-side argmax. The model never picks the route itself."""
        best = ROUTE_TIE_ORDER[0]
        best_value = self.to_dict()[best]
        for name in ROUTE_TIE_ORDER[1:]:
            value = self.to_dict()[name]
            # Strict ``>`` keeps the first route on an exact tie.
            if value > best_value:
                best, best_value = name, value
        return best


# --------------------------------------------------------------------------
# Generation result (C01 §1)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResultSegment:
    """One execution paragraph as the model emitted it.

    No id and no ordinal: those are framework-owned and are assigned only when
    the batch is built. Text is kept verbatim.
    """

    text: str

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text}


@dataclass(frozen=True, slots=True)
class UpperResult:
    """The single accepted result of one generation call.

    ``route`` is supplied by the framework after argmax; it is not part of the
    ``emit_upper_result`` payload, so a model cannot choose its own route.
    """

    route: str
    public_text: str | None
    segments: tuple[ResultSegment, ...]
    send_count: int

    _KEYS = frozenset({"public_text", "segments", "send_count"})

    @classmethod
    def parse(cls, payload: Any, route: str) -> "UpperResult":
        if route not in ROUTE_SET:
            reject(ErrorCode.INVALID_RESULT, "$.route", "unknown route")
        data = _require_object(payload, "$", ErrorCode.INVALID_RESULT)
        _require_exact_keys(data, cls._KEYS, "$", ErrorCode.INVALID_RESULT)
        _require_budget(data)

        public_text = _require_optional_text(
            data["public_text"], "$.public_text", ErrorCode.INVALID_RESULT,
            max_len=MAX_PUBLIC_TEXT_CHARS,
        )

        raw_segments = data["segments"]
        if not isinstance(raw_segments, list):
            reject(ErrorCode.INVALID_RESULT, "$.segments", "must be an array")
        if len(raw_segments) > MAX_SEGMENTS:
            reject(
                ErrorCode.INVALID_RESULT,
                "$.segments",
                f"must not exceed {MAX_SEGMENTS} segments",
            )
        segments: list[ResultSegment] = []
        for index, raw in enumerate(raw_segments):
            path = f"$.segments[{index}]"
            item = _require_object(raw, path, ErrorCode.INVALID_RESULT)
            _require_exact_keys(
                item, frozenset({"text"}), path, ErrorCode.INVALID_RESULT
            )
            segments.append(
                ResultSegment(
                    text=_require_text(
                        item["text"],
                        f"{path}.text",
                        ErrorCode.INVALID_RESULT,
                        max_len=MAX_SEGMENT_CHARS,
                    )
                )
            )

        send_count = _require_int(
            data["send_count"], "$.send_count", ErrorCode.INVALID_RESULT, minimum=0
        )

        result = cls(
            route=route,
            public_text=public_text,
            segments=tuple(segments),
            send_count=send_count,
        )
        result._check_route_shape()
        return result

    def _check_route_shape(self) -> None:
        """Reject any route/field combination C01 does not allow."""
        has_text = self.public_text is not None
        has_segments = bool(self.segments)

        if self.route == ROUTE_SPEECH_ONLY:
            # Speaking only: no execution paragraph may hide in the reply.
            if not has_text:
                reject(
                    ErrorCode.INVALID_RESULT,
                    "$.public_text",
                    "speech_only requires non-null public_text",
                )
            if has_segments:
                reject(
                    ErrorCode.INVALID_RESULT,
                    "$.segments",
                    "speech_only requires an empty segments array",
                )
            if self.send_count != 0:
                reject(
                    ErrorCode.INVALID_RESULT,
                    "$.send_count",
                    "speech_only requires send_count 0",
                )
            return

        if self.route == ROUTE_ACTION_ONLY:
            # Never read an action-only reply aloud: public_text must be null.
            if has_text:
                reject(
                    ErrorCode.INVALID_RESULT,
                    "$.public_text",
                    "action_only requires public_text null",
                )
        elif self.route == ROUTE_SPEECH_AND_ACTION:
            if not has_text:
                reject(
                    ErrorCode.INVALID_RESULT,
                    "$.public_text",
                    "speech_and_action requires non-null public_text",
                )
        else:  # pragma: no cover - guarded by parse()
            reject(ErrorCode.INVALID_RESULT, "$.route", "unknown route")

        if not has_segments:
            reject(
                ErrorCode.INVALID_RESULT,
                "$.segments",
                "action routes require at least one segment",
            )
        if self.send_count < 1:
            reject(
                ErrorCode.INVALID_RESULT,
                "$.send_count",
                "action routes require send_count >= 1",
            )
        if self.send_count > len(self.segments):
            reject(
                ErrorCode.INVALID_RESULT,
                "$.send_count",
                f"send_count {self.send_count} exceeds {len(self.segments)} segments",
            )

    # --- projections -------------------------------------------------------

    def sent_segments(self) -> tuple[ResultSegment, ...]:
        """The contiguous prefix actually sent now; the suffix stays in AEB."""
        return self.segments[: self.send_count]

    def deferred_segments(self) -> tuple[ResultSegment, ...]:
        return self.segments[self.send_count :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_text": self.public_text,
            "segments": [segment.to_dict() for segment in self.segments],
            "send_count": self.send_count,
        }


# --------------------------------------------------------------------------
# Trusted input (B00 §3.1)
# --------------------------------------------------------------------------

_TRUSTED_MARKER = object()


@dataclass(frozen=True, slots=True)
class TrustedUpperInput:
    """Identity resolved from the real Host event and deployment config.

    There is intentionally no ``parse``/``from_dict``. The private marker makes
    a hand-built instance fail, so a model result or a Page POST body can never
    be promoted to a trusted input.
    """

    robot_id: str
    session_id: str
    user_id: str
    route_ref: str
    ex_session: str
    input_text: str
    input_key: str
    origin: str = ""
    _marker: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _TRUSTED_MARKER:
            raise UpperContractError(
                err(
                    ErrorCode.INVALID_RESULT,
                    "$",
                    "TrustedUpperInput must be issued by framework admission",
                )
            )
        for name in ("robot_id", "session_id", "user_id", "route_ref", "ex_session", "input_key"):
            _require_wire_id(getattr(self, name), f"$.{name}", ErrorCode.INVALID_RESULT)
        _require_text(
            self.input_text, "$.input_text", ErrorCode.INVALID_RESULT,
            max_len=MAX_PUBLIC_TEXT_CHARS,
        )

    @classmethod
    def issue(
        cls,
        *,
        robot_id: str,
        session_id: str,
        user_id: str,
        route_ref: str,
        ex_session: str,
        input_text: str,
        input_key: str,
        origin: str = "",
    ) -> "TrustedUpperInput":
        """Framework admission only. Never call this from a model tool handler."""
        return cls(
            robot_id=robot_id,
            session_id=session_id,
            user_id=user_id,
            route_ref=route_ref,
            ex_session=ex_session,
            input_text=input_text,
            input_key=input_key,
            origin=origin,
            _marker=_TRUSTED_MARKER,
        )


# --------------------------------------------------------------------------
# Prompt snapshot metadata (B02 producer, B03/B05 consumers)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptSnapshotMeta:
    """Version identity of one frozen prompt assembly.

    B04 persists the *text actually used*; this object only carries the ids
    needed to explain which text that was.
    """

    routing_version: str
    planning_version: str
    catalog_version: str
    settings_version: str
    content_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "routing_version": self.routing_version,
            "planning_version": self.planning_version,
            "catalog_version": self.catalog_version,
            "settings_version": self.settings_version,
            "content_hash": self.content_hash,
        }

    @classmethod
    def parse(cls, payload: Any) -> "PromptSnapshotMeta":
        data = _require_object(payload, "$", ErrorCode.INVALID_RESULT)
        keys = frozenset(
            {
                "routing_version",
                "planning_version",
                "catalog_version",
                "settings_version",
                "content_hash",
            }
        )
        _require_exact_keys(data, keys, "$", ErrorCode.INVALID_RESULT)
        values = {
            name: _require_text(
                data[name], f"$.{name}", ErrorCode.INVALID_RESULT, max_len=MAX_WIRE_ID_LEN
            )
            for name in keys
        }
        return cls(**values)


# --------------------------------------------------------------------------
# Wire batch (C01 §4)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepSegment:
    """One paragraph on the wire. Identity is assigned by the framework."""

    segment_id: str
    ordinal: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "ordinal": self.ordinal,
            "text": self.text,
        }


_BATCH_KEYS = frozenset(
    {
        "schema",
        "request_id",
        "plan_id",
        "plan_revision",
        "batch_id",
        "robot_id",
        "route_ref",
        "ex_session",
        "catalog_version",
        "prompt_version",
        "visibility",
        "segments",
    }
)
_SEGMENT_KEYS = frozenset({"segment_id", "ordinal", "text"})


@dataclass(frozen=True, slots=True)
class InjectionBatch:
    """The only business payload that crosses the control channel."""

    schema: str
    request_id: str
    plan_id: str
    plan_revision: int
    batch_id: str
    robot_id: str
    route_ref: str
    ex_session: str
    catalog_version: str
    prompt_version: str
    visibility: str
    segments: tuple[StepSegment, ...]

    @classmethod
    def parse(cls, payload: Any) -> "InjectionBatch":
        data = _require_object(payload, "$", ErrorCode.INVALID_BATCH)
        _require_exact_keys(data, _BATCH_KEYS, "$", ErrorCode.INVALID_BATCH)
        _require_budget(data)

        if data["schema"] != UPPER_STEPS_SCHEMA:
            reject(
                ErrorCode.INVALID_BATCH,
                "$.schema",
                f"only {UPPER_STEPS_SCHEMA} is accepted",
            )
        if data["visibility"] != VISIBILITY_DECISION_ONLY:
            reject(
                ErrorCode.INVALID_BATCH,
                "$.visibility",
                f"only {VISIBILITY_DECISION_ONLY} is accepted",
            )

        ids = {
            name: _require_wire_id(data[name], f"$.{name}", ErrorCode.INVALID_BATCH)
            for name in (
                "request_id",
                "plan_id",
                "batch_id",
                "robot_id",
                "route_ref",
                "ex_session",
                "catalog_version",
                "prompt_version",
            )
        }
        plan_revision = _require_int(
            data["plan_revision"], "$.plan_revision", ErrorCode.INVALID_BATCH, minimum=1
        )

        raw_segments = data["segments"]
        if not isinstance(raw_segments, list):
            reject(ErrorCode.INVALID_BATCH, "$.segments", "must be an array")
        if not raw_segments:
            reject(ErrorCode.INVALID_BATCH, "$.segments", "must not be empty")
        if len(raw_segments) > MAX_SEGMENTS:
            reject(
                ErrorCode.INVALID_BATCH,
                "$.segments",
                f"must not exceed {MAX_SEGMENTS} segments",
            )

        segments: list[StepSegment] = []
        seen_ids: set[str] = set()
        previous_ordinal: int | None = None
        for index, raw in enumerate(raw_segments):
            path = f"$.segments[{index}]"
            item = _require_object(raw, path, ErrorCode.INVALID_BATCH)
            _require_exact_keys(item, _SEGMENT_KEYS, path, ErrorCode.INVALID_BATCH)
            segment_id = _require_wire_id(
                item["segment_id"], f"{path}.segment_id", ErrorCode.INVALID_BATCH
            )
            if segment_id in seen_ids:
                # Same id twice inside one batch is a conflict, not a dedupe.
                reject(
                    ErrorCode.ID_CONFLICT,
                    f"{path}.segment_id",
                    "duplicate segment_id within one batch",
                )
            seen_ids.add(segment_id)
            ordinal = _require_int(
                item["ordinal"], f"{path}.ordinal", ErrorCode.INVALID_BATCH, minimum=0
            )
            if previous_ordinal is not None and ordinal != previous_ordinal + 1:
                reject(
                    ErrorCode.INVALID_BATCH,
                    f"{path}.ordinal",
                    "ordinals must increase contiguously by one",
                )
            previous_ordinal = ordinal
            text = _require_text(
                item["text"], f"{path}.text", ErrorCode.INVALID_BATCH,
                max_len=MAX_SEGMENT_CHARS,
            )
            segments.append(
                StepSegment(segment_id=segment_id, ordinal=ordinal, text=text)
            )

        return cls(
            schema=data["schema"],
            request_id=ids["request_id"],
            plan_id=ids["plan_id"],
            plan_revision=plan_revision,
            batch_id=ids["batch_id"],
            robot_id=ids["robot_id"],
            route_ref=ids["route_ref"],
            ex_session=ids["ex_session"],
            catalog_version=ids["catalog_version"],
            prompt_version=ids["prompt_version"],
            visibility=data["visibility"],
            segments=tuple(segments),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "plan_id": self.plan_id,
            "plan_revision": self.plan_revision,
            "batch_id": self.batch_id,
            "robot_id": self.robot_id,
            "route_ref": self.route_ref,
            "ex_session": self.ex_session,
            "catalog_version": self.catalog_version,
            "prompt_version": self.prompt_version,
            "visibility": self.visibility,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    def payload_hash(self) -> str:
        """The hash both ends must agree on. Never includes the envelope id."""
        return canonical_hash(self.to_dict())

    def ordinals(self) -> tuple[int, ...]:
        return tuple(segment.ordinal for segment in self.segments)


def build_batch(
    *,
    trusted: TrustedUpperInput,
    result: UpperResult,
    plan_id: str,
    plan_revision: int,
    batch_id: str,
    request_id: str,
    catalog_version: str,
    prompt_version: str,
    segment_id_prefix: str,
    start_ordinal: int = 0,
) -> InjectionBatch:
    """Build the wire batch from a *trusted* input plus the accepted result.

    Every identity field comes from the trusted input or from framework
    arguments. Nothing is read back out of the model result except text.
    """
    if not isinstance(trusted, TrustedUpperInput):
        reject(ErrorCode.INVALID_BATCH, "$", "trusted input is required")
    segments = tuple(
        StepSegment(
            segment_id=f"{segment_id_prefix}-s{start_ordinal + offset}",
            ordinal=start_ordinal + offset,
            text=segment.text,
        )
        for offset, segment in enumerate(result.sent_segments())
    )
    batch = InjectionBatch(
        schema=UPPER_STEPS_SCHEMA,
        request_id=request_id,
        plan_id=plan_id,
        plan_revision=plan_revision,
        batch_id=batch_id,
        robot_id=trusted.robot_id,
        route_ref=trusted.route_ref,
        ex_session=trusted.ex_session,
        catalog_version=catalog_version,
        prompt_version=prompt_version,
        visibility=VISIBILITY_DECISION_ONLY,
        segments=segments,
    )
    # Re-parse so the builder cannot produce something the parser would refuse.
    return InjectionBatch.parse(batch.to_dict())


# --------------------------------------------------------------------------
# Receipt and process result (C01 §5)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """Delivery proof only. ``injected`` never means "executed"."""

    batch_id: str
    request_id: str
    payload_hash: str
    phase: str
    error_code: str = ""

    def __post_init__(self) -> None:
        # The invariant belongs to the object, so a direct construction cannot
        # produce a receipt that claims delivery and rejection at once.
        if self.phase not in RECEIPT_PHASES:
            reject(ErrorCode.INVALID_BATCH, "$.phase", "unknown receipt phase")
        if self.phase == PHASE_REJECTED and not self.error_code:
            reject(
                ErrorCode.INVALID_BATCH,
                "$.error_code",
                "a rejected receipt must name its error code",
            )
        if self.phase != PHASE_REJECTED and self.error_code:
            reject(
                ErrorCode.INVALID_BATCH,
                "$.error_code",
                "only a rejected receipt carries an error code",
            )

    @classmethod
    def parse(cls, payload: Any) -> "DeliveryReceipt":
        data = _require_object(payload, "$", ErrorCode.INVALID_BATCH)
        _require_exact_keys(
            data,
            frozenset({"batch_id", "request_id", "payload_hash", "phase", "error_code"}),
            "$",
            ErrorCode.INVALID_BATCH,
        )
        batch_id = _require_wire_id(
            data["batch_id"], "$.batch_id", ErrorCode.INVALID_BATCH
        )
        request_id = _require_wire_id(
            data["request_id"], "$.request_id", ErrorCode.INVALID_BATCH
        )
        payload_hash = _require_text(
            data["payload_hash"], "$.payload_hash", ErrorCode.INVALID_BATCH, max_len=128
        )
        phase = data["phase"]
        if phase not in RECEIPT_PHASES:
            reject(ErrorCode.INVALID_BATCH, "$.phase", "unknown receipt phase")
        error_code = data["error_code"]
        if not isinstance(error_code, str):
            reject(ErrorCode.INVALID_BATCH, "$.error_code", "must be a string")
        if phase == PHASE_REJECTED and not error_code:
            reject(
                ErrorCode.INVALID_BATCH,
                "$.error_code",
                "a rejected receipt must name its error code",
            )
        if phase != PHASE_REJECTED and error_code:
            reject(
                ErrorCode.INVALID_BATCH,
                "$.error_code",
                "only a rejected receipt carries an error code",
            )
        return cls(
            batch_id=batch_id,
            request_id=request_id,
            payload_hash=payload_hash,
            phase=phase,
            error_code=error_code,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "payload_hash": self.payload_hash,
            "phase": self.phase,
            "error_code": self.error_code,
        }

    @property
    def is_delivered(self) -> bool:
        return self.phase in {PHASE_RECEIVED, PHASE_INJECTED}


@dataclass(frozen=True, slots=True)
class UpperProcessResult:
    """What ``UpperPipeline.process`` hands back. No execution claim."""

    route: str
    result: UpperResult | None
    plan_id: str = ""
    batch_id: str = ""
    receipt: DeliveryReceipt | None = None
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return self.error_code == ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "result": self.result.to_dict() if self.result is not None else None,
            "plan_id": self.plan_id,
            "batch_id": self.batch_id,
            "receipt": self.receipt.to_dict() if self.receipt is not None else None,
            "error_code": self.error_code,
        }


# --------------------------------------------------------------------------
# Ingress identity checks (C01 §4, B00 §3.5)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrustedIngress:
    """The EX-side local connection record that actually grants identity.

    Built from deployment config, never from the payload. ``route_ref`` is the
    set of request sources registered for this connection.
    """

    robot_id: str
    ex_session: str
    route_refs: frozenset[str]


def check_ingress(batch: InjectionBatch, trusted: TrustedIngress) -> None:
    """Compare the wire claims against the local trusted record.

    A format-valid id is not an authorisation. The parser cannot make this
    call; only ingress can, because only ingress knows the local record.
    """
    if batch.robot_id != trusted.robot_id:
        reject(
            ErrorCode.SESSION_MISMATCH,
            "$.robot_id",
            "robot_id does not match the local connection record",
        )
    if batch.ex_session != trusted.ex_session:
        reject(
            ErrorCode.SESSION_MISMATCH,
            "$.ex_session",
            "ex_session is stale or unknown",
        )
    if batch.route_ref not in trusted.route_refs:
        reject(
            ErrorCode.ROUTE_MISMATCH,
            "$.route_ref",
            "route_ref is not registered for this connection",
        )


def check_replay(previous: InjectionBatch | None, incoming: InjectionBatch) -> bool:
    """Idempotency rule for one trusted source.

    Returns ``True`` when ``incoming`` is a byte-identical repeat that must
    return the stored record instead of acting again. Raises ``id_conflict``
    when the same ids carry different content.
    """
    if previous is None:
        return False
    if previous.batch_id != incoming.batch_id:
        return False
    if previous.request_id != incoming.request_id:
        reject(
            ErrorCode.ID_CONFLICT,
            "$.request_id",
            "batch_id reused with a different request_id",
        )
    if previous.payload_hash() != incoming.payload_hash():
        reject(
            ErrorCode.ID_CONFLICT,
            "$",
            "same ids but different payload content",
        )
    return True


# --------------------------------------------------------------------------
# Error classification helper
# --------------------------------------------------------------------------


def classify_error(code: str) -> str:
    """Map a published code onto its consumer contract.

    ``format`` -> nothing accepted, safe to report and drop.
    ``state``  -> payload was well formed, the caller must re-read state.
    ``uncertain`` -> delivery outcome unknown, never retry as success.
    """
    if code in FORMAT_ERROR_CODES:
        return "format"
    if code in STATE_ERROR_CODES:
        return "state"
    if code in UNCERTAIN_ERROR_CODES:
        return "uncertain"
    return "unknown"
