"""Durable upper-control plans, segments, injection batches and public replies.

The store reuses an already-open :class:`TaskStore` connection and its
``transaction()`` context manager.  It owns the ``upper_*`` tables only: the
legacy ``tasks``/``requests``/``receipts``/``public_messages``/``host_messages``
tables are never reinterpreted, renamed or deleted, so an older build that does
not know about these tables keeps working unchanged.

Contract objects (``UpperResult``/``InjectionBatch``/``DeliveryReceipt``) are
owned by B00 (``upper_contracts.py``).  This module never imports that file --
it consumes the documented C01 field surface through :func:`_field`, so B00's
dataclasses and plain dicts produced by the B00 parser both work.  Strict
schema rejection stays in B00; this module enforces only the invariants it is
responsible for (route/field combination, ordinal continuity, hashes, state
transitions).

Delivery semantics, deliberately conservative:

* ``received``/``injected`` only prove that EX recorded the batch and handed the
  text to its model input.  Neither means the action ran, and neither advances
  the pending suffix.
* ``delivery_unknown`` is a terminal state for this round.  Recovery never
  re-sends and never mints a new batch id.
* Nothing here claims exactly-once delivery.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
import uuid
from typing import Any, Callable, Iterable

from .task_models import TaskError, canonical

SCHEMA_VERSION = "aeb.upper_steps.v1"

# Tie-break order is fixed by C01 and is *not* alphabetical.
ROUTE_TIE_ORDER = ("speech_only", "speech_and_action", "action_only")
ROUTES = frozenset(ROUTE_TIE_ORDER)

MAX_SEGMENTS = 64
MAX_SEGMENT_CHARS = 4096
MAX_TOTAL_BYTES = 256 * 1024
MAX_WIRE_ID_LEN = 128
MAX_PROBABILITY_SUM_ERROR = 0.001

INPUT_CLAIMED = "claimed"
INPUT_PLANNED = "planned"
INPUT_REJECTED = "rejected"

PLAN_PLANNED = "planned"
PLAN_DISPATCHED = "dispatched"
PLAN_DELIVERY_UNKNOWN = "delivery_unknown"

SEGMENT_PENDING = "pending"
SEGMENT_SENDING = "sending"

# Phases that may be recorded against a batch.  ``received``/``injected`` are
# transport facts, not completion; ``rejected`` is a definite pre-injection
# refusal; ``delivery_unknown`` is the ambiguous window.
BATCH_PHASES = ("received", "injected", "rejected", "delivery_unknown")
BATCH_SENDING = "sending"

REPLY_CLAIMED = "claimed"
REPLY_SENDING = "sending"
REPLY_SENT = "sent"
REPLY_UNKNOWN = "delivery_unknown"


def _snapshot_value(value: Any) -> Any:
    """Convert frozen dataclasses and tuples into JSON-shaped values.

    This adapter belongs only to upper-control persistence. Legacy task JSON
    validation and its error codes retain their original behavior.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _snapshot_value(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_snapshot_value(item) for item in sorted(value, key=repr)]
    return value


def payload_hash(payload: dict[str, Any]) -> str:
    """SHA-256 over canonical UTF-8 JSON (C01: sorted keys, compact, no NaN)."""
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def derive_input_key(platform: str, origin: str, source_message_id: str) -> str:
    """Deterministic claim key for one trusted inbound message.

    Hashing the triple keeps the key path-safe and bounded while the component
    columns stay inspectable.  Duplicate detection never falls back to comparing
    input *text*: two distinct messages with identical text must stay distinct.

    B00's ``TrustedUpperInput`` already carries a framework-issued ``input_key``;
    in that case this helper is not used and the caller's key is stored verbatim.
    """
    material = "\x1f".join((platform, origin, source_message_id)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


# Historical name kept so callers that imported it keep working.
input_key = derive_input_key


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _require_wire_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TaskError("invalid_" + name)
    if len(value) > MAX_WIRE_ID_LEN:
        raise TaskError("invalid_" + name)
    if value != value.strip() or any(ch < " " or ch == "\x7f" for ch in value):
        raise TaskError("invalid_" + name)
    return value


def normalize_probabilities(result: Any) -> dict[str, float]:
    raw = _field(result, "probabilities", None)
    if raw is None:
        raw = {name: _field(result, name, None) for name in ROUTE_TIE_ORDER}
    if not isinstance(raw, dict) or set(raw) != ROUTES:
        raise TaskError("invalid_probabilities")
    parsed: dict[str, float] = {}
    for name in ROUTE_TIE_ORDER:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TaskError("invalid_probabilities")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise TaskError("invalid_probabilities")
        parsed[name] = number
    if abs(sum(parsed.values()) - 1.0) > MAX_PROBABILITY_SUM_ERROR:
        raise TaskError("invalid_probabilities")
    return parsed


def _segments(result: Any) -> list[str]:
    raw = _field(result, "segments", None)
    if raw is None or not isinstance(raw, (list, tuple)):
        raise TaskError("invalid_result")
    if len(raw) > MAX_SEGMENTS:
        raise TaskError("invalid_result")
    texts: list[str] = []
    for item in raw:
        text = item if isinstance(item, str) else _field(item, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise TaskError("invalid_result")
        if len(text) > MAX_SEGMENT_CHARS:
            raise TaskError("payload_too_large")
        texts.append(text)
    return texts


def normalize_result(result: Any) -> dict[str, Any]:
    """Validate the C01 route/field combination and return a plain value.

    B00 already parsed the model output; this second pass exists because the
    store is the last component before durable state and an external control
    send.  It rejects the combinations B00 would reject rather than trusting a
    caller to have done it.
    """
    route = _field(result, "route", None)
    if route not in ROUTES:
        raise TaskError("invalid_result")
    public_text = _field(result, "public_text", None)
    segments = _segments(result)
    send_count = _field(result, "send_count", None)
    if isinstance(send_count, bool) or not isinstance(send_count, int):
        raise TaskError("invalid_result")

    if public_text is not None:
        if not isinstance(public_text, str) or not public_text.strip():
            raise TaskError("invalid_result")
        if len(public_text) > MAX_SEGMENT_CHARS:
            raise TaskError("payload_too_large")
        public_text = public_text.strip()

    if route == "speech_only":
        if public_text is None or segments or send_count != 0:
            raise TaskError("route_mismatch")
    elif route == "action_only":
        if public_text is not None or not segments:
            raise TaskError("route_mismatch")
        if not 1 <= send_count <= len(segments):
            raise TaskError("route_mismatch")
    else:
        if public_text is None or not segments:
            raise TaskError("route_mismatch")
        if not 1 <= send_count <= len(segments):
            raise TaskError("route_mismatch")

    total = len(public_text.encode("utf-8")) if public_text else 0
    total += sum(len(text.encode("utf-8")) for text in segments)
    if total > MAX_TOTAL_BYTES:
        raise TaskError("payload_too_large")

    return {
        "route": route,
        "probabilities": normalize_probabilities(result),
        "public_text": public_text,
        "segments": segments,
        "send_count": send_count,
    }


def result_hash(normalized: dict[str, Any]) -> str:
    """Identity-free digest of a generated result, used for replay detection."""
    return payload_hash(normalized)


UPPER_SCHEMA = """
CREATE TABLE IF NOT EXISTS upper_inputs (
    input_key TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    origin TEXT NOT NULL,
    source_message_id TEXT NOT NULL,
    robot_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL,
    plan_id TEXT,
    error_code TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL);
-- Partial: B00's TrustedUpperInput carries its own framework-issued input_key
-- and has no platform/message id, so those rows must not collide on ''.
DROP INDEX IF EXISTS upper_inputs_identity;
CREATE UNIQUE INDEX IF NOT EXISTS upper_inputs_identity
    ON upper_inputs(platform, origin, source_message_id) WHERE source_message_id <> '';
CREATE TABLE IF NOT EXISTS upper_plans (
    plan_id TEXT PRIMARY KEY,
    input_key TEXT NOT NULL UNIQUE,
    request_id TEXT NOT NULL,
    route TEXT NOT NULL,
    probabilities TEXT NOT NULL,
    public_text TEXT,
    send_count INTEGER NOT NULL,
    segment_count INTEGER NOT NULL,
    result_hash TEXT NOT NULL,
    settings_hash TEXT NOT NULL DEFAULT '',
    catalog_hash TEXT NOT NULL DEFAULT '',
    prompt_hash TEXT NOT NULL DEFAULT '',
    settings_snapshot TEXT,
    catalog_snapshot TEXT,
    prompt_snapshot TEXT,
    plan_revision INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS upper_segments (
    segment_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS upper_segments_order
    ON upper_segments(plan_id, ordinal);
CREATE TABLE IF NOT EXISTS upper_batches (
    batch_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    peer TEXT NOT NULL,
    ex_session TEXT NOT NULL,
    robot_id TEXT NOT NULL,
    route_ref TEXT NOT NULL,
    first_ordinal INTEGER NOT NULL,
    last_ordinal INTEGER NOT NULL,
    segment_ids TEXT NOT NULL,
    status TEXT NOT NULL,
    phase TEXT,
    error_code TEXT,
    fence INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS upper_batches_plan
    ON upper_batches(plan_id);
CREATE TABLE IF NOT EXISTS upper_public_replies (
    reply_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL UNIQUE,
    input_key TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL);
"""


class UpperStore:
    """Upper-control persistence layered on one existing :class:`TaskStore`."""

    def __init__(self, store: Any, *, clock: Callable[[], float] = time.time,
                 recover: bool = True) -> None:
        self.store = store
        self.db = store.db
        self._clock = clock
        store.init_extension_tables(UPPER_SCHEMA)
        if recover:
            self.recover()

    # ------------------------------------------------------------------
    # internal row helpers -- each one assumes the caller already holds the
    # transaction; none of them opens a nested BEGIN.
    # ------------------------------------------------------------------
    def _row(self, table: str, key: str, column: str = "plan_id") -> dict | None:
        row = self.db.execute(
            f"SELECT * FROM {table} WHERE {column}=?", (key,)
        ).fetchone()
        return dict(row) if row else None

    def _input_row(self, key: str) -> dict | None:
        return self._row("upper_inputs", key, "input_key")

    def _plan_row(self, plan_id: str) -> dict | None:
        return self._row("upper_plans", plan_id)

    def _plan_by_input(self, key: str) -> dict | None:
        return self._row("upper_plans", key, "input_key")

    def _batch_row(self, batch_id: str) -> dict | None:
        return self._row("upper_batches", batch_id, "batch_id")

    def _batch_for_plan(self, plan_id: str) -> dict | None:
        return self._row("upper_batches", plan_id)

    def _reply_row(self, reply_id: str) -> dict | None:
        return self._row("upper_public_replies", reply_id, "reply_id")

    def _reply_for_plan(self, plan_id: str) -> dict | None:
        return self._row("upper_public_replies", plan_id)

    def _segment_rows(self, plan_id: str) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM upper_segments WHERE plan_id=? ORDER BY ordinal", (plan_id,))]

    def _touch(self, table: str, key: str, column: str, values: dict[str, Any]) -> None:
        assignments = ", ".join(f"{name}=?" for name in values)
        self.db.execute(
            f"UPDATE {table} SET {assignments} WHERE {column}=?",
            (*values.values(), key),
        )

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------
    def claim_input(self, *, robot_id: str, user_id: str, session_id: str,
                    input_key: str | None = None, platform: str = "",
                    origin: str = "", source_message_id: str = "") -> tuple[dict, bool]:
        """Atomically claim one trusted inbound message.

        Returns ``(record, claimed)``.  ``claimed`` is False when a concurrent or
        earlier handler already owns this input; the caller must then reuse the
        existing generation instead of generating and sending a second time.

        Pass B00's ``TrustedUpperInput.input_key`` when available; otherwise the
        key is derived from the ``platform``/``origin``/``source_message_id``
        triple, which must then all be present.
        """
        for value, name in ((robot_id, "robot_id"), (user_id, "user_id"),
                            (session_id, "session_id")):
            _require_wire_id(value, name)
        if input_key is not None:
            _require_wire_id(input_key, "input_key")
            key = input_key
            origin = origin or ""
        else:
            for value, name in ((platform, "platform"), (origin, "origin"),
                                (source_message_id, "source_message_id")):
                _require_wire_id(value, name)
            key = derive_input_key(platform, origin, source_message_id)
        now = self._clock()
        with self.store.transaction():
            row = self._input_row(key)
            if row is not None:
                if (row["robot_id"], row["user_id"], row["session_id"]) != (robot_id, user_id, session_id):
                    raise TaskError("id_conflict")
                return row, False
            self.db.execute(
                "INSERT INTO upper_inputs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, platform, origin, source_message_id, robot_id, user_id,
                 session_id, 1, INPUT_CLAIMED, None, None, now, now),
            )
            return self._input_row(key), True

    def reject_input(self, key: str, *, generation: int, error_code: str) -> dict:
        with self.store.transaction():
            row = self._input_row(key)
            if row is None:
                raise TaskError("unknown_input")
            if row["generation"] != generation:
                raise TaskError("stale_generation")
            if row["status"] != INPUT_CLAIMED:
                raise TaskError("invalid_transition")
            self._touch("upper_inputs", key, "input_key",
                        {"status": INPUT_REJECTED, "error_code": error_code,
                         "updated_at": self._clock()})
            return self._input_row(key)

    # ------------------------------------------------------------------
    # plans
    # ------------------------------------------------------------------
    def save_result(self, key: str, result: Any, *, generation: int,
                    settings_snapshot: Any = None, catalog_snapshot: Any = None,
                    prompt_snapshot: Any = None, settings_hash: str = "",
                    catalog_hash: str = "", prompt_hash: str = "") -> dict:
        """Persist the complete generated result before any external IO.

        A retried handler for the same input returns the plan that already
        exists (same ids, same segments) instead of creating a second one.  A
        different result for an already-planned input is ``id_conflict``.
        """
        normalized = normalize_result(result)
        digest = result_hash(normalized)
        now = self._clock()
        with self.store.transaction():
            inbound = self._input_row(key)
            if inbound is None:
                raise TaskError("unknown_input")
            if inbound["generation"] != generation:
                raise TaskError("stale_generation")
            existing = self._plan_by_input(key)
            if existing is not None:
                if existing["result_hash"] != digest:
                    raise TaskError("id_conflict")
                return self._plan_projection(existing["plan_id"])
            if inbound["status"] != INPUT_CLAIMED:
                raise TaskError("invalid_transition")

            plan_id = uuid.uuid4().hex
            request_id = uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO upper_plans(plan_id, input_key, request_id, route, probabilities,"
                " public_text, send_count, segment_count, result_hash, settings_hash,"
                " catalog_hash, prompt_hash, settings_snapshot, catalog_snapshot,"
                " prompt_snapshot, plan_revision, generation, status, created_at, updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (plan_id, key, request_id, normalized["route"],
                 canonical(normalized["probabilities"]), normalized["public_text"],
                 normalized["send_count"], len(normalized["segments"]), digest,
                 settings_hash, catalog_hash, prompt_hash,
                 canonical(_snapshot_value(settings_snapshot)) if settings_snapshot is not None else None,
                 canonical(_snapshot_value(catalog_snapshot)) if catalog_snapshot is not None else None,
                 canonical(_snapshot_value(prompt_snapshot)) if prompt_snapshot is not None else None,
                 1, generation, PLAN_PLANNED, now, now),
            )
            for ordinal, text in enumerate(normalized["segments"]):
                self.db.execute(
                    "INSERT INTO upper_segments VALUES(?,?,?,?,?,?)",
                    (uuid.uuid4().hex, plan_id, ordinal, text, SEGMENT_PENDING, now),
                )
            self._touch("upper_inputs", key, "input_key",
                        {"status": INPUT_PLANNED, "plan_id": plan_id, "updated_at": now})
            return self._plan_projection(plan_id)

    def get_plan(self, plan_id: str) -> dict:
        with self.store._lock:
            if self._plan_row(plan_id) is None:
                raise TaskError("unknown_plan")
            return self._plan_projection(plan_id)

    def list_plans(self, *, limit: int = 20, offset: int = 0) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise TaskError("invalid_limit")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise TaskError("invalid_limit")
        with self.store._lock:
            rows = self.db.execute(
                "SELECT plan_id FROM upper_plans ORDER BY created_at DESC, plan_id DESC "
                "LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            return [self._plan_projection(row["plan_id"]) for row in rows]

    def _plan_projection(self, plan_id: str) -> dict:
        """Read-only view for B07.

        Snapshot *bodies* (settings/catalog/prompt text) stay in the database:
        they can contain provider ids and full production context.  Only their
        hashes are exposed, and no hidden reasoning is ever stored.
        """
        plan = self._plan_row(plan_id)
        if plan is None:
            raise TaskError("unknown_plan")
        segments = [
            {"segment_id": row["segment_id"], "ordinal": row["ordinal"],
             "text": row["text"], "status": row["status"]}
            for row in self._segment_rows(plan_id)
        ]
        batch = self._batch_for_plan(plan_id)
        reply = self._reply_for_plan(plan_id)
        inbound = self._input_row(plan["input_key"])
        return {
            "plan_id": plan["plan_id"],
            "request_id": plan["request_id"],
            "input_key": plan["input_key"],
            "route": plan["route"],
            "probabilities": json.loads(plan["probabilities"]),
            "public_text": plan["public_text"],
            "send_count": plan["send_count"],
            "segment_count": plan["segment_count"],
            "result_hash": plan["result_hash"],
            "plan_revision": plan["plan_revision"],
            "generation": plan["generation"],
            "status": plan["status"],
            "settings_hash": plan["settings_hash"],
            "catalog_hash": plan["catalog_hash"],
            "prompt_hash": plan["prompt_hash"],
            "created_at": plan["created_at"],
            "updated_at": plan["updated_at"],
            "input_status": inbound["status"] if inbound else None,
            "segments": segments,
            "batch": self._batch_projection(batch) if batch else None,
            "public_reply": self._reply_projection(reply) if reply else None,
        }

    # ------------------------------------------------------------------
    # batches
    # ------------------------------------------------------------------
    def reserve_batch(self, plan_id: str, *, peer: bytes, ex_session: str,
                      robot_id: str, route_ref: str) -> dict:
        """Freeze the injection envelope for the first ``send_count`` segments.

        Reserve happens before the network call so a crash inside the send
        window is recoverable as ``delivery_unknown`` rather than a silent
        re-send.  One plan owns at most one batch (``upper_batches_plan``), so a
        repeated call returns the frozen envelope unchanged and the suffix
        segments stay ``pending``.
        """
        if not isinstance(peer, bytes) or not peer:
            raise TaskError("invalid_batch")
        _require_wire_id(ex_session, "ex_session")
        _require_wire_id(robot_id, "robot_id")
        _require_wire_id(route_ref, "route_ref")
        now = self._clock()
        with self.store.transaction():
            plan = self._plan_row(plan_id)
            if plan is None:
                raise TaskError("unknown_plan")
            existing = self._batch_for_plan(plan_id)
            if existing is not None:
                return self._batch_projection(existing)
            count = plan["send_count"]
            if count <= 0:
                raise TaskError("invalid_batch")
            segments = self._segment_rows(plan_id)[:count]
            if len(segments) != count:
                raise TaskError("invalid_batch")
            if any(row["status"] != SEGMENT_PENDING for row in segments):
                raise TaskError("invalid_transition")
            batch_id = uuid.uuid4().hex
            payload = {
                "schema": SCHEMA_VERSION,
                "request_id": plan["request_id"],
                "plan_id": plan_id,
                "plan_revision": plan["plan_revision"],
                "batch_id": batch_id,
                "robot_id": robot_id,
                "route_ref": route_ref,
                "ex_session": ex_session,
                "catalog_version": plan["catalog_hash"],
                "prompt_version": plan["prompt_hash"],
                "visibility": "decision_only",
                "segments": [
                    {"segment_id": row["segment_id"], "ordinal": row["ordinal"],
                     "text": row["text"]}
                    for row in segments
                ],
            }
            digest = payload_hash(payload)
            if len(canonical(payload).encode("utf-8")) > MAX_TOTAL_BYTES:
                raise TaskError("payload_too_large")
            self.db.execute(
                "INSERT INTO upper_batches VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (batch_id, plan_id, plan["request_id"], canonical(payload), digest,
                 peer.hex(), ex_session, robot_id, route_ref,
                 segments[0]["ordinal"], segments[-1]["ordinal"],
                 canonical([row["segment_id"] for row in segments]),
                 BATCH_SENDING, None, None, 0, now, now),
            )
            for row in segments:
                self._touch("upper_segments", row["segment_id"], "segment_id",
                            {"status": SEGMENT_SENDING})
            return self._batch_projection(self._batch_row(batch_id))

    def record_delivery(self, batch_id: str, phase: str, *, error_code: str | None = None,
                        receipt: Any = None) -> dict:
        """Record the transport outcome of one reserved batch.

        Only the segments inside this batch move.  The pending suffix is left
        untouched on purpose: this round has no automatic continuation, and
        ``received``/``injected`` are not completion.
        """
        if phase not in BATCH_PHASES:
            raise TaskError("invalid_transition")
        now = self._clock()
        with self.store.transaction():
            batch = self._batch_row(batch_id)
            if batch is None:
                raise TaskError("unknown_batch")
            if batch["status"] != BATCH_SENDING:
                if batch["status"] == phase:
                    return self._batch_projection(batch)
                raise TaskError("invalid_transition")
            if receipt is not None:
                self._verify_receipt(batch, receipt)
            self._touch("upper_batches", batch_id, "batch_id",
                        {"status": phase, "phase": phase, "error_code": error_code,
                         "updated_at": now})
            for segment_id in json.loads(batch["segment_ids"]):
                self._touch("upper_segments", segment_id, "segment_id", {"status": phase})
            plan_status = PLAN_DELIVERY_UNKNOWN if phase == "delivery_unknown" else PLAN_DISPATCHED
            self._touch("upper_plans", batch["plan_id"], "plan_id",
                        {"status": plan_status, "updated_at": now})
            return self._batch_projection(self._batch_row(batch_id))

    @staticmethod
    def _verify_receipt(batch: dict, receipt: Any) -> None:
        """A receipt only counts when it answers *this* batch, verbatim."""
        for field, expected in (("batch_id", batch["batch_id"]),
                                ("request_id", batch["request_id"]),
                                ("payload_hash", batch["payload_hash"])):
            if _field(receipt, field, None) != expected:
                raise TaskError("delivery_unknown")

    def _batch_projection(self, batch: dict | None) -> dict | None:
        if batch is None:
            return None
        payload = json.loads(batch["payload"])
        return {
            "batch_id": batch["batch_id"],
            "plan_id": batch["plan_id"],
            "request_id": batch["request_id"],
            "payload": payload,
            "payload_hash": batch["payload_hash"],
            "peer_hex": batch["peer"],
            "ex_session": batch["ex_session"],
            "robot_id": batch["robot_id"],
            "route_ref": batch["route_ref"],
            "first_ordinal": batch["first_ordinal"],
            "last_ordinal": batch["last_ordinal"],
            "segment_ids": json.loads(batch["segment_ids"]),
            "status": batch["status"],
            "phase": batch["phase"],
            "error_code": batch["error_code"],
            "fence": batch["fence"],
        }

    # ------------------------------------------------------------------
    # public replies
    # ------------------------------------------------------------------
    def claim_public_reply(self, plan_id: str) -> tuple[dict, bool]:
        """Claim the single public exit for a plan.

        The claim is committed before the send so an ambiguous timeout can never
        re-speak.  A plan without ``public_text`` has no public exit at all.
        """
        now = self._clock()
        with self.store.transaction():
            plan = self._plan_row(plan_id)
            if plan is None:
                raise TaskError("unknown_plan")
            text = plan["public_text"]
            if not text:
                raise TaskError("no_public_text")
            existing = self._reply_for_plan(plan_id)
            if existing is not None:
                return self._reply_projection(existing), False
            reply_id = uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO upper_public_replies VALUES(?,?,?,?,?,?,?,?)",
                (reply_id, plan_id, plan["input_key"], text, REPLY_CLAIMED, None, now, now),
            )
            return self._reply_projection(self._reply_row(reply_id)), True

    def record_public_delivery(self, reply_id: str, status: str, *,
                               error_code: str | None = None) -> dict:
        if status not in {REPLY_SENDING, REPLY_SENT, REPLY_UNKNOWN}:
            raise TaskError("invalid_transition")
        with self.store.transaction():
            reply = self._reply_row(reply_id)
            if reply is None:
                raise TaskError("unknown_reply")
            if reply["status"] in {REPLY_SENT, REPLY_UNKNOWN}:
                if reply["status"] == status:
                    return self._reply_projection(reply)
                raise TaskError("invalid_transition")
            if status == REPLY_SENDING and reply["status"] != REPLY_CLAIMED:
                raise TaskError("invalid_transition")
            self._touch("upper_public_replies", reply_id, "reply_id",
                        {"status": status, "error_code": error_code,
                         "updated_at": self._clock()})
            return self._reply_projection(self._reply_row(reply_id))

    def _reply_projection(self, reply: dict | None) -> dict | None:
        if reply is None:
            return None
        return {
            "reply_id": reply["reply_id"],
            "plan_id": reply["plan_id"],
            "input_key": reply["input_key"],
            "text": reply["text"],
            "status": reply["status"],
            "error_code": reply["error_code"],
        }

    # ------------------------------------------------------------------
    # recovery
    # ------------------------------------------------------------------
    def recover(self) -> dict[str, int]:
        """Downgrade every unresolved in-flight record to a conservative state.

        A crash between reserve and receipt, or between a public claim and its
        send, is genuinely ambiguous.  It becomes ``delivery_unknown`` and gains
        a fence so no later code path can auto-retry it.  ``pending`` segments are
        left alone and are not proactively dispatched.
        """
        now = self._clock()
        counts = {"batches": 0, "segments": 0, "replies": 0, "plans": 0}
        with self.store.transaction():
            for batch in self.db.execute(
                    "SELECT * FROM upper_batches WHERE status=?", (BATCH_SENDING,)).fetchall():
                self._touch("upper_batches", batch["batch_id"], "batch_id",
                            {"status": BATCH_PHASES[-1], "phase": BATCH_PHASES[-1],
                             "error_code": "restart_unresolved", "fence": batch["fence"] + 1,
                             "updated_at": now})
                for segment_id in json.loads(batch["segment_ids"]):
                    self._touch("upper_segments", segment_id, "segment_id",
                                {"status": BATCH_PHASES[-1]})
                    counts["segments"] += 1
                self._touch("upper_plans", batch["plan_id"], "plan_id",
                            {"status": PLAN_DELIVERY_UNKNOWN, "updated_at": now})
                counts["batches"] += 1
                counts["plans"] += 1
            # Orphaned ``sending`` segments (no batch row) are equally ambiguous.
            orphan = self.db.execute(
                "SELECT segment_id FROM upper_segments WHERE status=?", (SEGMENT_SENDING,)).fetchall()
            for row in orphan:
                self._touch("upper_segments", row["segment_id"], "segment_id",
                            {"status": BATCH_PHASES[-1]})
                counts["segments"] += 1
            for reply in self.db.execute(
                    "SELECT reply_id FROM upper_public_replies WHERE status IN (?,?)",
                    (REPLY_CLAIMED, REPLY_SENDING)).fetchall():
                self._touch("upper_public_replies", reply["reply_id"], "reply_id",
                            {"status": REPLY_UNKNOWN, "error_code": "restart_unresolved",
                             "updated_at": now})
                counts["replies"] += 1
        return counts
