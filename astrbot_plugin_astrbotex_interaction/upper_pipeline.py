"""One upper-control turn: trusted input -> classify -> generate -> two exits.

The pipeline is the only place that joins the pieces.  It owns no transport and
no SQL: it calls B03 for decisions, B04 for durable state and B06 for the
control send, and it never lets execution text touch a public channel.

Hard invariants, in the order they are enforced:

1. Identity comes from the real event, never from the model or a Page POST.  A
   :class:`TrustedUpperInput` without a real ``sender`` is refused before any
   model call or database write.
2. The input is claimed *first*, so a duplicated Host hook re-reads existing
   state instead of generating and sending twice.
3. The route is computed here by argmax over the three probabilities.  A route
   volunteered by the model is ignored.
4. The complete result is persisted before any external IO.  There is no path
   that sends control first and stores the plan afterwards.
5. The control payload is built from stored segments only.  ``public_text``
   never enters it, and the public sender only ever receives a bare string.
6. The two exits are independent network operations with independent records.
   Neither is retried on the other's failure.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .output_router import UPPER_PUBLIC, UpperOutputRouter
from .task_models import TaskError
from .upper_store import ROUTE_TIE_ORDER, UpperStore, normalize_probabilities

SPEECH_ONLY = "speech_only"
ACTION_ONLY = "action_only"
SPEECH_AND_ACTION = "speech_and_action"

MAX_INPUT_CHARS = 8192

# B00's published table.  Anything outside it is a pipeline-local diagnostic;
# format violations are folded back into the table so B07/B11 never have to
# switch on an unknown format code.
CONTRACT_CODES = frozenset({
    "invalid_probabilities", "invalid_result", "invalid_batch", "payload_too_large",
    "route_mismatch", "session_mismatch", "id_conflict", "input_busy",
    "backend_unsupported", "delivery_unknown",
})


def contract_code(code: str) -> str:
    """Fold store-level ``invalid_<field>`` codes into ``invalid_result``."""
    if code in CONTRACT_CODES:
        return code
    if code.startswith("invalid_"):
        return "invalid_result"
    return code


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def argmax_route(probabilities: dict[str, float]) -> str:
    """Framework-side route decision with the C01 tie order.

    ``speech_only`` wins a three-way tie, then ``speech_and_action``.  The model
    never decides the route itself, so a model that also emits a ``route`` field
    cannot override this.
    """
    best = ROUTE_TIE_ORDER[0]
    best_value = probabilities[best]
    for name in ROUTE_TIE_ORDER[1:]:
        if probabilities[name] > best_value:
            best, best_value = name, probabilities[name]
    return best


def default_result(**values: Any) -> dict[str, Any]:
    """Placeholder result object until B00's ``UpperProcessResult`` lands.

    B10 should inject ``upper_contracts.UpperProcessResult`` through
    ``result_factory``; this mapping carries exactly the four contracted keys
    plus diagnostics so nothing downstream depends on the placeholder type.
    """
    return values


class UpperPipeline:
    """Stateless-per-turn coordinator over injected collaborators."""

    def __init__(
        self,
        *,
        store: UpperStore,
        models: Any,
        prompts: Any,
        settings: Any,
        catalog: Any,
        control: Any,
        router: UpperOutputRouter,
        robot_id: str,
        peer_id: str,
        route_ref: str = "upper-control",
        routing_provider_id: str = "",
        generation_provider_id: str = "",
        active_control: Callable[[], bool] | None = None,
        alive: Callable[[], bool] | None = None,
        result_factory: Callable[..., Any] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.models = models
        self.prompts = prompts
        self.settings = settings
        self.catalog = catalog
        self.control = control
        self.router = router
        self.robot_id = robot_id
        self.peer_id = peer_id
        self.peer = peer_id.encode("utf-8")
        self.route_ref = route_ref
        self.routing_provider_id = routing_provider_id
        self.generation_provider_id = generation_provider_id
        self._active_control = active_control
        self._alive = alive
        self._result = result_factory or default_result
        self._on_event = on_event
        self._clock = clock

    # ------------------------------------------------------------------
    def _emit(self, event: str, **detail: Any) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event({"component": "upper_pipeline", "event": event, **detail})
        except Exception:  # noqa: BLE001 - telemetry must never break a turn
            pass

    def _is_alive(self) -> bool:
        if self._alive is None:
            return True
        try:
            return bool(self._alive())
        except Exception:  # noqa: BLE001
            return False

    def _control_occupied(self) -> bool:
        if self._active_control is None:
            return False
        try:
            return bool(self._active_control())
        except Exception:  # noqa: BLE001 - an unknown occupier is treated as busy
            return True

    def _finish(self, *, handled: bool, suppress_default: bool,
                error_code: str | None = None, **extra: Any) -> Any:
        return self._result(handled=handled, suppress_default=suppress_default,
                            error_code=error_code, **extra)

    # ------------------------------------------------------------------
    def _validate(self, trusted: Any) -> dict[str, Any]:
        """Refuse anything that did not come from a real platform event.

        Two accepted shapes:

        * B00's ``TrustedUpperInput`` -- carries a framework-issued ``input_key``
          and the private ``_marker`` that makes a hand-built instance fail.
        * The pre-B00 mapping used by the unit tests: ``trusted is True`` plus a
          real ``sender``/``event`` object.

        Both are checked for a live sender and for a robot id that matches this
        deployment.  A model result or a Page POST body can satisfy neither.
        """
        marker = _field(trusted, "_marker", None)
        issued = marker is not None and isinstance(_field(trusted, "input_key", None), str)
        if not issued:
            if _field(trusted, "trusted", False) is not True:
                raise TaskError("invalid_result")
            sender = _field(trusted, "sender", None)
            if sender is None:
                sender = _field(trusted, "event", None)
            if sender is None:
                raise TaskError("invalid_result")
        else:
            sender = marker

        values: dict[str, Any] = {}
        for name in ("robot_id", "user_id", "session_id"):
            value = _field(trusted, name, None)
            if not isinstance(value, str) or not value.strip():
                raise TaskError("invalid_result")
            values[name] = value
        if values["robot_id"] != self.robot_id:
            raise TaskError("invalid_result")

        values["input_key"] = _field(trusted, "input_key", None) if issued else None
        values["platform"] = _field(trusted, "platform", "") or ""
        values["origin"] = _field(trusted, "origin", "") or ""
        values["source_message_id"] = _field(trusted, "source_message_id", "") or ""
        if not issued and not (values["platform"] and values["source_message_id"]):
            raise TaskError("invalid_result")

        text = _field(trusted, "input_text", None)
        if not isinstance(text, str) or not text.strip():
            raise TaskError("invalid_result")
        if len(text) > MAX_INPUT_CHARS:
            raise TaskError("invalid_result")
        values["input_text"] = text
        values["route_ref"] = _field(trusted, "route_ref", None) or self.route_ref
        # B00 puts the session on the trusted input; prefer it over a live
        # capabilities read so the batch is bound to the session that admitted
        # the message rather than whatever the peer advertises later.
        values["ex_session"] = _field(trusted, "ex_session", None) or None
        values["sender"] = sender
        return values

    # ------------------------------------------------------------------
    async def process(self, trusted: Any, *, public_sender: Callable[[str], Any] | None = None,
                      peer: bytes | None = None) -> Any:
        """Run one complete upper-control turn.

        ``public_sender`` is bound by B10 to the original event's normal reply
        ability.  Exactly one public destination is used per turn.
        """
        try:
            values = self._validate(trusted)
        except TaskError as exc:
            self._emit("rejected", error_code=exc.code)
            return self._finish(handled=False, suppress_default=False, error_code=exc.code)

        if not self._is_alive():
            return self._finish(handled=False, suppress_default=False,
                                error_code="generation_stale")

        # Claim before any model work: a repeated Host hook must not generate.
        try:
            record, claimed = self.store.claim_input(
                input_key=values["input_key"], platform=values["platform"],
                origin=values["origin"],
                source_message_id=values["source_message_id"],
                robot_id=values["robot_id"], user_id=values["user_id"],
                session_id=values["session_id"])
        except TaskError as exc:
            code = contract_code(exc.code)
            self._emit("rejected", error_code=code)
            return self._finish(handled=False, suppress_default=False, error_code=code)

        if not claimed:
            # Already owned by an earlier or concurrent handler.  Read, do not act.
            self._emit("duplicate_input", input_key=record["input_key"],
                       plan_id=record["plan_id"])
            return self._finish(handled=True, suppress_default=True, error_code=None,
                                plan_id=record["plan_id"], duplicate=True, route=None,
                                control_status=None, public_sent=False)

        generation = record["generation"]

        # Freeze configuration before generation so the stored plan names the
        # exact prompt/catalog it was produced under.
        try:
            settings_snapshot = self.settings.load()
            catalog_snapshot = self.catalog.snapshot()
            prompt_snapshot = self.prompts.build(settings_snapshot, catalog_snapshot)
        except Exception as exc:  # noqa: BLE001
            self._reject(record["input_key"], generation, "config_invalid")
            self._emit("config_invalid", error=str(exc))
            return self._finish(handled=False, suppress_default=False,
                                error_code="config_invalid")

        # Decide, then generate.  Both are private: no streaming, no side effects.
        try:
            probabilities = await self.models.classify(
                values["input_text"], prompt_snapshot, self.routing_provider_id)
        except Exception as exc:  # noqa: BLE001
            self._reject(record["input_key"], generation, "decision_error")
            self._emit("decision_error", error=str(exc))
            return self._finish(handled=False, suppress_default=False,
                                error_code="decision_error")

        try:
            route = argmax_route(self._normalized_probabilities(probabilities))
        except TaskError as exc:
            code = contract_code(exc.code)
            self._reject(record["input_key"], generation, code)
            return self._finish(handled=False, suppress_default=False, error_code=code)

        if not self._is_alive():
            self._reject(record["input_key"], generation, "generation_stale")
            return self._finish(handled=False, suppress_default=False,
                                error_code="generation_stale")

        try:
            generated = await self.models.generate(
                values["input_text"], route, prompt_snapshot, self.generation_provider_id)
        except Exception as exc:  # noqa: BLE001
            self._reject(record["input_key"], generation, "generation_error")
            self._emit("generation_error", error=str(exc))
            return self._finish(handled=False, suppress_default=False,
                                error_code="generation_error")

        # The framework owns the route, never the model.
        composed = {
            "route": route,
            "probabilities": probabilities,
            "public_text": _field(generated, "public_text", None),
            "segments": _field(generated, "segments", None),
            "send_count": _field(generated, "send_count", None),
        }
        try:
            plan = self.store.save_result(
                record["input_key"], composed, generation=generation,
                settings_snapshot=settings_snapshot, catalog_snapshot=catalog_snapshot,
                prompt_snapshot=prompt_snapshot,
                settings_hash=_field(prompt_snapshot, "settings_hash", "") or "",
                catalog_hash=_field(prompt_snapshot, "catalog_version", "") or "",
                prompt_hash=_field(prompt_snapshot, "prompt_hash", "") or "")
        except TaskError as exc:
            code = contract_code(exc.code)
            self._reject(record["input_key"], generation, code)
            self._emit("result_rejected", error_code=code)
            return self._finish(handled=False, suppress_default=False, error_code=code)

        plan_id = plan["plan_id"]
        self._emit("planned", plan_id=plan_id, route=route,
                   segments=plan["segment_count"], send_count=plan["send_count"])

        # A late result after a config reload or plugin close must send nothing.
        if not self._is_alive():
            self._emit("stale_after_generation", plan_id=plan_id)
            return self._finish(handled=True, suppress_default=True,
                                error_code="generation_stale", plan_id=plan_id,
                                route=route, control_status=None, public_sent=False)

        control_status, control_error = await self._dispatch_control(
            plan, peer=peer, route_ref=values["route_ref"],
            ex_session=values["ex_session"])

        public_sent = False
        public_error = None
        if plan["public_text"]:
            if public_sender is None:
                public_error = "public_sender_missing"
            else:
                try:
                    emitted = await self.router.emit_public(plan_id, public_sender)
                except TaskError as exc:
                    emitted, public_error = None, exc.code
                if emitted is not None:
                    public_sent = bool(emitted["sent"])
                    if not public_sent:
                        public_error = emitted.get("reason")

        error_code = control_error or public_error
        self._emit("dispatched", plan_id=plan_id, route=route,
                   control_status=control_status, public_sent=public_sent,
                   error_code=error_code)
        return self._finish(handled=True, suppress_default=True, error_code=error_code,
                            plan_id=plan_id, route=route, control_status=control_status,
                            control_error=control_error, public_sent=public_sent,
                            public_error=public_error, duplicate=False)

    # ------------------------------------------------------------------
    async def _dispatch_control(self, plan: dict, *, peer: bytes | None,
                                route_ref: str,
                                ex_session: str | None = None) -> tuple[str | None, str | None]:
        """Reserve and inject.  Returns ``(status, error_code)``.

        Never raises: a control failure is a recorded outcome, and the already
        generated public text stays sendable.
        """
        if plan["send_count"] <= 0:
            return None, None
        if self._control_occupied():
            # Busy old task: refuse the new action, never cancel the old goal.
            self._emit("control_busy", plan_id=plan["plan_id"])
            return None, "input_busy"

        target = peer if peer is not None else self.peer
        if ex_session is None:
            # No session on the trusted input: ask the peer, then freeze the
            # answer into the batch so a later session change is a refusal.
            try:
                summary = await self.control.capabilities(target)
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", "backend_unsupported")
                self._emit("control_unavailable", plan_id=plan["plan_id"], error_code=code)
                return None, code
            ex_session = summary["ex_session"]

        try:
            batch = self.store.reserve_batch(
                plan["plan_id"], peer=target, ex_session=ex_session,
                robot_id=self.robot_id, route_ref=route_ref)
        except TaskError as exc:
            self._emit("reserve_failed", plan_id=plan["plan_id"], error_code=exc.code)
            return None, exc.code

        try:
            outcome = await self.control.inject(target, batch)
        except Exception as exc:  # noqa: BLE001 - unknown, never silently dropped
            self._emit("control_error", plan_id=plan["plan_id"], error=str(exc))
            self._record_delivery(batch["batch_id"], "delivery_unknown",
                                  error_code="delivery_unknown")
            return "delivery_unknown", "delivery_unknown"

        status = _field(outcome, "status", "delivery_unknown")
        error_code = _field(outcome, "error_code", None)
        self._record_delivery(batch["batch_id"], status, error_code=error_code)
        return status, (None if status in {"received", "injected"} else
                        error_code or "delivery_unknown")

    def _record_delivery(self, batch_id: str, status: str,
                         *, error_code: str | None = None) -> None:
        try:
            self.store.record_delivery(batch_id, status, error_code=error_code)
        except TaskError as exc:
            # A terminal batch answered differently: keep the stored truth.
            self._emit("delivery_not_recorded", batch_id=batch_id,
                       attempted=status, error_code=exc.code)

    def _reject(self, input_key: str, generation: int, error_code: str) -> None:
        try:
            self.store.reject_input(input_key, generation=generation,
                                    error_code=error_code)
        except TaskError:
            pass

    @staticmethod
    def _normalized_probabilities(raw: Any) -> dict[str, float]:
        return normalize_probabilities(raw)
