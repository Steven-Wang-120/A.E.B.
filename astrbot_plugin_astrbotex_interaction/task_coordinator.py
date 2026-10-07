"""One valid private turn per robot, durable feedback and independent leases."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any

from .task_contracts import (
    DecisionState, EventsReply, Feedback, GoalSubmit, can_transition,
    require_id, require_sequence, validate_params,
)
from .task_models import PlanningTurn, TaskAuthority, TaskError, json_value
from .task_store import TaskStore


class TaskCoordinator:
    def __init__(self, store: TaskStore, request, *, host=None, provider_id: str = "",
                 router=None, max_rounds: int = 8, max_failures: int = 3,
                 llm_timeout: float = 30.0, heartbeat_interval: float = 1.0,
                 clock=time.monotonic, turn_sync=None) -> None:
        self.store = store
        self.request = request
        self.host = host
        self.provider_id = provider_id
        self.router = router
        self.max_rounds = max_rounds
        self.max_failures = max_failures
        self.llm_timeout = llm_timeout
        self.heartbeat_interval = heartbeat_interval
        self.clock = clock
        self.turn_sync = turn_sync
        self._turn_sync_locks: dict[str, asyncio.Lock] = {}
        self._turn_sync_tasks: set[asyncio.Task] = set()
        self._workers: dict[str, asyncio.Task] = {}
        self._sync_locks: dict[str, asyncio.Lock] = {}
        self._submit_locks: dict[str, asyncio.Lock] = {}
        self._lease_deadlines: dict[str, float] = {}
        self._lease_identities: dict[str, tuple] = {}
        self._heartbeat: asyncio.Task | None = None
        self._reconciler: asyncio.Task | None = None
        self._closed = False

    def start(self) -> None:
        # Startup intentionally does NOT schedule persisted tasks or renew old leases.
        if self._heartbeat is None:
            self._heartbeat = asyncio.create_task(self._heartbeat_loop())
            self._reconciler = asyncio.create_task(self._reconcile_loop())

    async def close(self) -> None:
        self._closed = True
        for task in self.store.active_tasks():
            self.store.mutate(task["task_id"], lambda t: t.update(
                generation=t["generation"] + 1, turn_id=None, needs_planning=False, status="resume_review"))
            self._invalidate_public(task["task_id"])
        await self.drain_turn_sync()
        tasks = list(self._workers.values()) + [t for t in (self._heartbeat, self._reconciler) if t]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._heartbeat = None
        self._reconciler = None
        self._lease_deadlines.clear()
        self._lease_identities.clear()

    async def sync_turn(self, task_id: str, operation: str, *, turn: PlanningTurn | None = None,
                        expected_generation: int | None = None) -> None:
        if self.turn_sync is None:
            return
        if operation == "invalidate" and expected_generation is None:
            expected_generation = self.store.get(task_id)["generation"]
        lock = self._turn_sync_locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            task = self.store.get(task_id)
            if expected_generation is not None and task["generation"] != expected_generation:
                return
            if turn is not None:
                self.store.check_turn(turn, task)
            if not task["ex_session"]:
                return  # Bootstrap has no EX identity or previously bound public turn.
            payload = {"task_schema_version": 1, "operation": operation,
                       **{k: task[k] for k in ("ex_session", "task_id", "robot_id", "session_id", "user_id", "route_ref")},
                       "turn_id": task["turn_id"] if operation == "bind" else None,
                       "generation": task["generation"], "expected_revision": task["goal_revision"]}
            if operation == "bind" and not payload["turn_id"]:
                raise TaskError("stale_generation")
            try:
                await asyncio.wait_for(self.turn_sync(payload), timeout=5.0)
            except Exception as exc:
                raise TaskError("turn_sync_failed") from exc
            if turn is not None:
                self.store.check_turn(turn)

    def _invalidate_public(self, task_id: str) -> None:
        if self.turn_sync is None:
            return
        snapshot = self.store.get(task_id)
        async def invalidate():
            if self.store.get(task_id)["generation"] != snapshot["generation"]:
                return  # A newer bind/invalidate owns the watermark now.
            try:
                await self.sync_turn(task_id, "invalidate", expected_generation=snapshot["generation"])
            except TaskError:
                self.store.mutate(task_id, lambda t: t.update(error_code="turn_sync_failed"))
        pending = asyncio.create_task(invalidate())
        self._turn_sync_tasks.add(pending)
        pending.add_done_callback(self._turn_sync_tasks.discard)

    async def drain_turn_sync(self) -> None:
        if self._turn_sync_tasks:
            await asyncio.gather(*list(self._turn_sync_tasks))

    def _forget_lease(self, task_id: str) -> None:
        self._lease_deadlines.pop(task_id, None)
        self._lease_identities.pop(task_id, None)

    def _prune_lease(self, task_id: str) -> None:
        if self._lease_identities.get(task_id) != self.store.lease_identity(self.store.get(task_id)):
            self._forget_lease(task_id)

    def _expire_lease(self, task_id: str, identity: tuple, deadline: float) -> None:
        def expire(task):
            if (self._lease_identities.get(task_id) == identity
                    and self._lease_deadlines.get(task_id) == deadline
                    and self.store.lease_identity(task) == identity):
                task.update(status="lease_lost", needs_planning=False,
                            generation=task["generation"] + 1, turn_id=None)
                self._forget_lease(task_id)
        before = self.store.get(task_id)["generation"]
        self.store.mutate(task_id, expire)
        if self.store.get(task_id)["generation"] != before:
            self._invalidate_public(task_id)
        self._prune_lease(task_id)

    def _check_submit_lease(self, task_id: str, payload: dict) -> tuple:
        self._prune_lease(task_id)
        identity = self._lease_identities.get(task_id)
        goal = self.store.get(task_id)["current_goal"]
        if not identity or not goal or goal["payload"] != payload:
            raise TaskError("lease_unavailable")
        deadline = self._lease_deadlines[task_id]
        if self.clock() >= deadline:
            self._expire_lease(task_id, identity, deadline)
            raise TaskError("lease_lost")
        return identity

    async def _call(self, task: dict, method: str, payload: dict) -> dict:
        return await self.request(task["robot_id"], task["route_ref"], method, payload)

    def create_task(self, authority: TaskAuthority, text: str, request_id: str) -> dict:
        task = self.store.create(authority, text, request_id)
        self.wake(task["task_id"])
        return task

    def wake(self, task_id: str) -> None:
        if self._closed or self.host is None:
            return
        task = self.store.get(task_id)
        if (not task["active"] or not task["needs_planning"]
                or task["status"] in {"resume_review", "lease_lost", "canceling", "waiting_input"}):
            return
        old = self._workers.get(task["robot_id"])
        if old and not old.done():
            return
        worker = asyncio.create_task(self._planning_worker(task_id))
        self._workers[task["robot_id"]] = worker
        worker.add_done_callback(lambda done: self._worker_done(task["robot_id"], task_id, done))

    def _worker_done(self, robot_id: str, task_id: str, worker: asyncio.Task) -> None:
        if self._workers.get(robot_id) is worker:
            self._workers.pop(robot_id, None)
        if not worker.cancelled():
            worker.exception()
        if not self._closed:
            self.wake(task_id)

    async def _planning_worker(self, task_id: str) -> None:
        from .planning_tools import PlanningTools, run_planning_turn
        turn = self.store.begin_turn(task_id)
        task = self.store.get(task_id)
        try:
            if task["ex_session"]:
                await self.sync_turn(task_id, "bind", turn=turn)
            failures = task["planning_failures"]
            if failures:
                await asyncio.sleep(min(0.25 * 2 ** failures, 4.0))
            context = json_value(await self._call(task, "decision.context.get", {"schema_version": 1}))
            require_id(context.get("ex_session"), "ex_session")
            require_sequence(context.get("revision"), "revision")
            if task["ex_session"] and context["ex_session"] != task["ex_session"]:
                raise TaskError("stale_ex_session")
            def bind(current):
                current["ex_session"] = context["ex_session"]
                if not current["current_goal"]:
                    current["goal_revision"] = context["revision"]
            self.store.mutate(task_id, bind, turn=turn)
            await self.sync_turn(task_id, "bind", turn=turn)
            tools = PlanningTools(self, self.router, turn, context)
            # A bounded, on-demand JSON context, never public message pipeline/image fanout.
            prompt = json.dumps({"task": self.store.get(task_id), "decision_context": context}, ensure_ascii=False)
            if len(prompt.encode("utf-8")) > 262144:
                raise TaskError("planning_context_too_large")
            await run_planning_turn(self.host, task.get("provider_id") or self.provider_id, tools, prompt,
                                    max_rounds=self.max_rounds, timeout_sec=self.llm_timeout)
        except asyncio.CancelledError:
            def canceled(current):
                current.update(generation=current["generation"] + 1, turn_id=None,
                               needs_planning=False, status="resume_review")
            try:
                self.store.mutate(task_id, canceled, turn=turn)
                self._prune_lease(task_id)
                self._invalidate_public(task_id)
                await self.drain_turn_sync()
            except TaskError:
                pass
            raise
        except Exception as exc:
            code = getattr(exc, "code", "provider_timeout" if isinstance(exc, TimeoutError) else "planning_failed")
            def failed(current):
                current["turn_id"] = None
                current["generation"] += 1
                current["error_code"] = code
                current["planning_failures"] += 1
                # Once a submit has happened/been attempted, never blindly replan/replay.
                if current["current_goal"]:
                    current["status"] = "resume_review"
                    current["needs_planning"] = False
                elif current["planning_failures"] >= self.max_failures:
                    current["status"] = "waiting_input"
                    current["needs_planning"] = False
                else:
                    current["needs_planning"] = True
            try:
                self.store.mutate(task_id, failed, turn=turn)
                self._invalidate_public(task_id)
                await self.drain_turn_sync()
                self._prune_lease(task_id)
            except TaskError:
                pass  # Late errors cannot mutate a replacement generation.

    @staticmethod
    def _catalog(context: dict) -> dict[str, dict]:
        actions = context.get("actions")
        if not isinstance(actions, list) or len(actions) > 256:
            raise TaskError("invalid_capability_catalog")
        result = {}
        for action in actions:
            if (not isinstance(action, dict) or not isinstance(action.get("schema"), dict)
                    or not isinstance(action.get("action_id"), str) or action["action_id"] in result):
                raise TaskError("invalid_capability_catalog")
            result[action["action_id"]] = action
        return result

    async def submit(self, turn: PlanningTurn, args: dict, context: dict, call_id: str) -> dict:
        task = self.store.get(turn.task_id)
        self.store.check_turn(turn, task)
        catalog = self._catalog(context)
        for action_id in args["allowed_actions"]:
            if action_id not in catalog:
                raise TaskError("unknown_action")
            if validate_params(catalog[action_id]["schema"], args["parameters"].get(action_id, {})):
                raise TaskError("invalid_action_parameters")
        if not args["allowed_actions"]:
            raise TaskError("empty_allowed_actions")
        request_id = hashlib.sha256(f"{turn.turn_id}:{call_id}".encode()).hexdigest()
        # The framework owns all wire identity/CAS fields. Goal+params share one outbox row.
        existing = self.store.request(request_id)
        expected_revision = (existing["payload"].get("expected_revision") if existing
                             else task["goal_revision"])
        payload = GoalSubmit.parse({"schema_version": 1, "request_id": request_id,
                                   "ex_session": context["ex_session"], "task_id": turn.task_id,
                                   "goal_id": request_id, "expected_revision": expected_revision, **args}).to_dict()
        lock = self._submit_locks.setdefault(task["robot_id"], asyncio.Lock())
        async with lock:
            # Only a newly prepared request can create a deadline. An ambiguous retry
            # keeps the first deadline, and a revoked/missing lease cannot be recreated.
            new_request = self.store.request(request_id) is None
            prior = self.store.prepare_goal(turn, payload)
            if new_request:
                self._forget_lease(turn.task_id)
                identity = self.store.lease_identity(self.store.get(turn.task_id))
                if identity is not None:
                    self._lease_identities[turn.task_id] = identity
                    self._lease_deadlines[turn.task_id] = self.clock() + payload["lease_ms"] / 1000
            # Historical rejected/completed admissions remain readable, not executable.
            if prior["result"] and not self.store.get(turn.task_id)["current_goal"]:
                self._prune_lease(turn.task_id)
                return prior["result"]
            identity = self._check_submit_lease(turn.task_id, payload)
            if prior["result"]:
                return prior["result"]
            try:
                result = await self._call(task, "decision.goal.submit", payload)
            except TimeoutError:
                # state.get is read-only; replay admission only within the first lease.
                await self._call(task, "decision.state.get", {"schema_version": 1, "ex_session": payload["ex_session"]})
                self.store.check_turn(turn)
                self._check_submit_lease(turn.task_id, payload)
                result = await self._call(task, "decision.goal.submit", payload)
            self._validate_submit_result(payload, result)
            # Receipt of an old submit still matters for audit/cancel; it cannot issue a new goal.
            self.store.record_submit(turn.task_id, payload, result)
            current_identity = self.store.lease_identity(self.store.get(turn.task_id))
            if (self._lease_identities.get(turn.task_id) == identity
                    and current_identity is not None and current_identity[:-1] == identity[:-1]
                    and identity[-1] is None):
                # Admission assigns the revision, but never resets the local deadline.
                self._lease_identities[turn.task_id] = current_identity
            self._prune_lease(turn.task_id)
            self.store.check_turn(turn)
            if result["ok"]:
                self._check_submit_lease(turn.task_id, payload)
            await self.sync_turn(turn.task_id, "bind", turn=turn)
            return result

    @staticmethod
    def _validate_submit_result(payload: dict, result: dict) -> None:
        value = json_value(result)
        required = {"ok", "request_id", "ex_session", "goal_id", "revision", "phase"}
        if (not isinstance(value, dict) or not required.issubset(value)
                or set(value) - required - {"error"} or type(value.get("ok")) is not bool):
            raise TaskError("invalid_submit_response")
        if "error" in value:
            error = value["error"]
            if not isinstance(error, dict) or set(error) != {"code", "path", "message"}:
                raise TaskError("invalid_submit_response")
            if any(not isinstance(v, str) or len(v) > 4096 for v in error.values()):
                raise TaskError("invalid_submit_response")
        for key in ("request_id", "ex_session", "goal_id"):
            if result.get(key) != payload[key]:
                raise TaskError("invalid_submit_response")
        require_sequence(result.get("revision"), "revision")
        if result.get("phase") not in {"accepted", "pending_cancel", "active", "blocked", "rejected"}:
            raise TaskError("invalid_submit_response")
        if result["ok"] != (result["phase"] != "rejected"):
            raise TaskError("invalid_submit_response")

    def finish(self, turn: PlanningTurn, outcome: str) -> None:
        def finish(task):
            if outcome == "waiting_feedback" and not task["current_goal"]:
                raise TaskError("no_goal_to_wait_for")
            if outcome == "completed":
                if task["current_goal"] or not task["steps"] or any(s["status"] != "completed" for s in task["steps"]):
                    raise TaskError("completion_not_verified")
                task["status"] = "completed"
                task["active"] = False
            elif outcome == "waiting_input":
                task["status"] = "waiting_input"
            else:
                task["status"] = "executing"
            task["turn_id"] = None
            task["planning_failures"] = 0
        self.store.mutate(turn.task_id, finish, turn=turn)
        self._prune_lease(turn.task_id)

    @staticmethod
    def _apply_fact(task: dict, fact: Feedback) -> bool:
        goal = task["current_goal"]
        if (not goal or fact.task_id != task["task_id"] or fact.ex_session != task["ex_session"]
                or fact.goal_id != goal["payload"]["goal_id"]
                or fact.goal_revision != goal["revision"]):
            return False
        previous = goal.get("feedback_status", "admitted")
        last = goal.get("last_feedback")
        if last and (fact.event_seq <= last["event_seq"] or previous in {
                "succeeded", "failed", "rejected", "canceled", "timed_out", "unknown"}):
            return False
        if not can_transition(previous, fact.status):
            return False
        goal["feedback_status"] = fact.status
        goal["last_feedback"] = fact.to_dict()
        index = task["active_step"]
        if fact.status in {"admitted", "accepted", "running"}:
            task["steps"][index]["status"] = "executing" if fact.status == "running" else "dispatched"
            return False
        task["steps"][index]["status"] = "awaiting_review"
        return True  # State reconciliation, not LLM/step advance yet.

    async def feedback(self, robot_id: str, route_ref: str, raw: dict) -> dict:
        fact = Feedback.parse(raw)
        task = self.store.get(fact.task_id)
        if task["robot_id"] != robot_id or task["route_ref"] != route_ref:
            raise TaskError("owner_mismatch")
        if task["ex_session"] != fact.ex_session:
            raise TaskError("stale_ex_session")
        self.store.receive(robot_id, fact)
        cursor = self.store.cursor(robot_id, fact.ex_session)
        try:
            if fact.event_seq > cursor + 1 or fact.event_seq <= cursor:
                await self.sync(task["task_id"])
            else:
                wakes = self.store.apply_receipts(robot_id, fact.ex_session, self._apply_fact)
                self._prune_lease(task["task_id"])
                if wakes:
                    await self.sync(task["task_id"])
        except Exception:
            # Durable receipt still ACKs; no new planning is authorized by failed sync.
            self.store.mutate(task["task_id"], lambda t: t.update(error_code="feedback_sync_required"))
        return {"ok": True, "ex_session": fact.ex_session, "acked_event_seq": fact.event_seq}

    def _require_review(self, task_id: str) -> None:
        def review(task):
            if task["status"] != "resume_review":
                task["generation"] += 1
            task.update(status="resume_review", needs_planning=False, turn_id=None)
        self.store.mutate(task_id, review)
        self._prune_lease(task_id)
        self._invalidate_public(task_id)

    async def sync(self, task_id: str) -> dict:
        task = self.store.get(task_id)
        lock = self._sync_locks.setdefault(task["robot_id"], asyncio.Lock())
        async with lock:
            task = self.store.get(task_id)
            session = task["ex_session"]
            if not session:
                raise TaskError("missing_ex_session")
            horizon = None
            cropped = False
            state = None
            # Bound catch-up work; a moving ledger must not cause an endless poll.
            for _ in range(32):
                cursor = self.store.cursor(task["robot_id"], session)
                page = EventsReply.parse(await self._call(task, "decision.events.get", {
                    "schema_version": 1, "ex_session": session, "since_event_seq": cursor}))
                if page.ex_session != session:
                    self._require_review(task_id)
                    raise TaskError("stale_ex_session")
                if page.resync_required:
                    cropped = True
                else:
                    if page.latest_event_seq < cursor:
                        raise TaskError("stale_state")
                    for event in page.events:
                        fact = Feedback.parse({"schema_version": 1, **{k: v for k, v in event.items()
                            if k in {"ex_session", "task_id", "goal_id", "goal_revision", "event_seq", "status", "reason_code", "details"}}})
                        self.store.receive(task["robot_id"], fact)
                    self.store.apply_receipts(task["robot_id"], session, self._apply_fact)
                    self._prune_lease(task_id)
                    if self.store.cursor(task["robot_id"], session) < page.latest_event_seq:
                        if self.store.cursor(task["robot_id"], session) <= cursor:
                            raise TaskError("event_gap")
                        continue
                state = DecisionState.parse(await self._call(task, "decision.state.get", {
                    "schema_version": 1, "ex_session": session})).to_dict()
                if state["ex_session"] != session:
                    self._require_review(task_id)
                    raise TaskError("stale_ex_session")
                if state["execution"].get("previous_session_manual_review") is True:
                    current = self.store.get(task_id)
                    reviewed = {k: current[k] for k in ("task_id", "robot_id", "ex_session")}
                    if current.get("reviewed_session") != reviewed:
                        self._require_review(task_id)
                if state["event_seq"] < self.store.cursor(task["robot_id"], session):
                    raise TaskError("stale_state")
                if horizon is None:
                    horizon = state["event_seq"]
                if not cropped and self.store.cursor(task["robot_id"], session) < horizon:
                    if self.store.cursor(task["robot_id"], session) <= cursor:
                        raise TaskError("event_gap")
                    continue
                break
            else:
                raise TaskError("sync_page_limit")
            before_generation = self.store.get(task_id)["generation"]
            self.store.reconcile(task_id, state, self._apply_state, cropped=cropped)
            current = self.store.get(task_id)
            if current["generation"] != before_generation:
                self._invalidate_public(task_id)
            self._prune_lease(task_id)
        self.wake(task_id)
        return state

    @staticmethod
    def _terminal_verified(fact: dict, state: dict) -> bool:
        details = fact["details"]
        evidence = details.get("terminal_evidence")
        stop = details.get("stop_evidence")
        execution = state["execution"]
        if (not isinstance(evidence, dict) or not isinstance(stop, dict)
                or stop.get("stopped") is not True or evidence.get("verified") is not True
                or evidence.get("goal_id") != fact["goal_id"]
                or type(evidence.get("goal_revision")) is not int
                or evidence["goal_revision"] != fact["goal_revision"]
                or state["active_goal_id"] is not None or state["pending_goal_id"] is not None
                or execution.get("gate_open") is not False or execution.get("blocked") is not False
                or execution.get("internal_phase") != "idle" or execution.get("unresolved") != []
                or execution.get("stop_proven") is not True):
            return False
        dispatcher = evidence.get("dispatcher_epoch")
        proof = evidence.get("stop_proof_epoch")
        if (type(dispatcher) is not int or type(proof) is not int or not 0 <= dispatcher <= proof
                or type(execution.get("dispatcher_epoch")) is not int
                or type(execution.get("stop_proof_epoch")) is not int
                or execution["dispatcher_epoch"] != dispatcher or execution["stop_proof_epoch"] != proof):
            return False
        commands = evidence.get("commands")
        if not isinstance(commands, list):
            return False
        ids = set()
        for command in commands:
            if (not isinstance(command, dict) or not isinstance(command.get("command_id"), str)
                    or not command["command_id"] or command["command_id"] in ids
                    or command.get("ex_session") != fact["ex_session"]
                    or command.get("goal_id") != fact["goal_id"]
                    or type(command.get("goal_revision")) is not int
                    or command["goal_revision"] != fact["goal_revision"]
                    or type(command.get("event_seq")) is not int or command["event_seq"] <= 0
                    or not isinstance(command.get("status"), str)
                    or command["status"] not in {"succeeded", "rejected", "canceled", "failed"}):
                return False
            ids.add(command["command_id"])
            if command["status"] in {"canceled", "failed"}:
                command_stop = command.get("stop_evidence")
                if (not isinstance(command_stop, dict) or command_stop.get("stopped") is not True
                        or not isinstance(command_stop.get("source"), str) or not command_stop["source"]
                        or not isinstance(command_stop.get("reference"), str) or not command_stop["reference"]):
                    return False
        # Decision rejection can happen before dispatch: no physical FAILED row exists.
        # All other failed Goals still need the original, stopped Action failure facts.
        if fact["status"] == "failed" and (fact["reason_code"] not in {
                "low_confidence", "backend_requested_replan"}
                or any(command["status"] == "failed" for command in commands)):
            failure = details.get("failure_evidence")
            if (not isinstance(failure, dict) or failure.get("verified") is not True
                    or failure.get("goal_id") != fact["goal_id"]
                    or type(failure.get("goal_revision")) is not int
                    or failure["goal_revision"] != fact["goal_revision"]
                    or not isinstance(failure.get("commands"), list) or not failure["commands"]):
                return False
            if any(command not in commands or command.get("status") != "failed"
                   for command in failure["commands"]):
                return False
        return True

    @classmethod
    def _apply_state(cls, task: dict, state: dict) -> None:
        goal = task["current_goal"]
        if not goal or state["ex_session"] != task["ex_session"]:
            return
        execution = state["execution"]
        summaries = execution.get("goal_summaries", [])
        if not isinstance(summaries, list):
            raise TaskError("invalid_goal_summaries")
        raw_facts = list(execution.get("receipt_facts", [])) + summaries
        summary = execution.get("feedback")
        if isinstance(summary, dict):
            raw_facts.append(summary)
        facts = {}
        for raw in raw_facts:
            fact = Feedback.parse(raw)
            if (fact.task_id != task["task_id"] or fact.ex_session != task["ex_session"]
                    or fact.goal_id != goal["payload"]["goal_id"]
                    or fact.goal_revision != goal["revision"]
                    or fact.event_seq > state["event_seq"]):
                continue
            previous = facts.get(fact.event_seq)
            if previous and previous.to_dict() != fact.to_dict():
                raise TaskError("event_seq_conflict")
            facts[fact.event_seq] = fact
        for seq in sorted(facts):
            cls._apply_fact(task, facts[seq])
        fact = goal.get("last_feedback")
        if (fact and fact["ex_session"] == state["ex_session"]
                and fact["event_seq"] <= state["event_seq"]):
            task["last_event_seq"] = max(task["last_event_seq"], fact["event_seq"])
        # Normal running commands also lack stop proof; only blocked/uncertain
        # execution of the current Goal fences planning, without a terminal fact.
        unresolved = execution.get("unresolved", [])
        uncertain = (execution.get("stop_proven") is False and isinstance(unresolved, list)
            and any(isinstance(row, dict) and row.get("status") in {"unknown", "timed_out", "failed"}
                    for row in unresolved))
        if (goal["payload"]["goal_id"] in {state["active_goal_id"], state["pending_goal_id"]}
                and (execution.get("blocked") is True
                     or execution.get("internal_phase") == "blocked" or uncertain)):
            if task["status"] != "resume_review":
                task["generation"] += 1
            task.update(status="resume_review", needs_planning=False, turn_id=None)
            return
        if (not fact or fact["ex_session"] != state["ex_session"]
                or fact["event_seq"] > state["event_seq"]):
            return
        index = task["active_step"]
        status = fact["status"]
        if status in {"unknown", "timed_out"} or (status in {"failed", "rejected", "canceled"}
                and not cls._terminal_verified(fact, state)):
            if task["status"] != "resume_review":
                task["generation"] += 1
            task.update(status="resume_review", needs_planning=False, turn_id=None)
            return
        if goal["payload"]["goal_id"] in {state["active_goal_id"], state["pending_goal_id"]}:
            return
        if status == "succeeded":
            evidence = fact["details"].get("completion_evidence", {})
            required = goal["payload"]["completion"].get("required_success_actions", [])
            if (not isinstance(evidence, dict) or evidence.get("verified") is not True
                    or evidence.get("goal_id") != fact["goal_id"]
                    or type(evidence.get("goal_revision")) is not int
                    or evidence["goal_revision"] != fact["goal_revision"]
                    or not isinstance(evidence.get("succeeded_actions"), list)
                    or not set(required).issubset(evidence["succeeded_actions"])):
                return
            task["steps"][index]["status"] = "completed"
            task["active_step"] += 1
            task["completion_evidence_seq"] = fact["event_seq"]
        elif status in {"failed", "rejected", "canceled"}:
            task["steps"][index]["status"] = "canceled" if status == "canceled" else "failed"
        else:
            if task["status"] != "resume_review":
                task["generation"] += 1
                task["turn_id"] = None
            task["status"] = "resume_review"
            task["needs_planning"] = False
            return
        task["current_goal"] = None
        if task["turn_id"]:
            task["generation"] += 1
            task["turn_id"] = None
        if task.get("cancel_requested"):
            task.update(active=False, status="canceled", needs_planning=False)
            return
        if status == "canceled" and not task.get("replacement_requested"):
            task.update(status="waiting_input", needs_planning=False)
            return
        task.pop("replacement_requested", None)
        # A restarted task remains user-reviewed even after fetching the ledger.
        if task["status"] not in {"resume_review", "lease_lost", "waiting_input"}:
            task["status"] = "planning"
            task["needs_planning"] = True

    async def user_input(self, task_id: str, authority: TaskAuthority, text: str) -> None:
        self.store.invalidate(task_id, authority, text)
        self._prune_lease(task_id)
        await self.sync_turn(task_id, "invalidate")
        task = self.store.get(task_id)
        # Do not cancel in-flight transport IO: its late admission must be durably
        # recorded before cancel. Generation fences discard the old LLM output.
        goal = task["current_goal"]
        if goal:
            # Serialize behind an in-flight submit. Cancel accepted is not stop evidence.
            lock = self._submit_locks.setdefault(task["robot_id"], asyncio.Lock())
            async with lock:
                task = self.store.get(task_id)
                goal = task["current_goal"]
                if goal and goal["revision"] is not None:
                    await self._call(task, "decision.goal.cancel", {
                        "schema_version": 1, "request_id": uuid.uuid4().hex,
                        "ex_session": task["ex_session"], "goal_id": goal["payload"]["goal_id"],
                        "goal_revision": goal["revision"], "reason_code": "user_replacement"})
            await self.sync(task_id)
        self.wake(task_id)

    async def cancel_task(self, task_id: str, authority: TaskAuthority) -> None:
        task = self.store.authorize(task_id, authority)
        if not task["active"]:
            raise TaskError("task_inactive")
        self.store.invalidate(task_id, authority)
        self.store.mutate(task_id, lambda t: t.update(cancel_requested=True, needs_planning=False))
        self._prune_lease(task_id)
        await self.sync_turn(task_id, "invalidate")
        task = self.store.get(task_id)
        lock = self._submit_locks.setdefault(task["robot_id"], asyncio.Lock())
        async with lock:
            task = self.store.get(task_id)
            goal = task["current_goal"]
            if goal is None:
                self.store.mutate(task_id, lambda t: t.update(active=False, status="canceled"))
                return
            if goal["revision"] is None:
                raise TaskError("unresolved_execution")
            await self._call(task, "decision.goal.cancel", {
                "schema_version": 1, "request_id": uuid.uuid4().hex,
                "ex_session": task["ex_session"], "goal_id": goal["payload"]["goal_id"],
                "goal_revision": goal["revision"], "reason_code": "user_cancel"})
        await self.sync(task_id)

    async def review_resume(self, task_id: str, authority: TaskAuthority) -> None:
        task = self.store.authorize(task_id, authority)
        if not task["active"]:
            raise TaskError("task_inactive")
        self.store.mutate(task_id, lambda t: t.update(generation=t["generation"] + 1,
            turn_id=None, needs_planning=False, status="resume_review"))
        self._prune_lease(task_id)
        task = self.store.get(task_id)
        generation = task["generation"]
        context = json_value(await self._call(task, "decision.context.get", {"schema_version": 1}))
        require_id(context.get("ex_session"), "ex_session")
        require_sequence(context.get("revision"), "revision")
        if (task["ex_session"] and (context["ex_session"] != task["ex_session"]
                or task["current_goal"] or task.get("reviewed_goal", {}).get("review_required"))):
            lock = self._sync_locks.setdefault(task["robot_id"], asyncio.Lock())
            async with lock:
                state = DecisionState.parse(await self._call(task, "decision.state.get", {
                    "schema_version": 1, "ex_session": context["ex_session"]})).to_dict()
                if state["ex_session"] != context["ex_session"]:
                    raise TaskError("stale_ex_session")
                same_session = state["ex_session"] == task["ex_session"]
                if (state["revision"] < context["revision"] or state["event_seq"] <
                        self.store.cursor(task["robot_id"], state["ex_session"])
                        or (same_session and (state["event_seq"] < task["last_event_seq"]
                            or state["revision"] < task["goal_revision"]))):
                    raise TaskError("stale_state")
                execution = state["execution"]
                proof_epoch = execution.get("stop_proof_epoch")
                dispatcher_epoch = execution.get("dispatcher_epoch")
                # Empty active/pending pairs are not physical stop evidence.
                if (state["active_goal_id"] is not None or state["pending_goal_id"] is not None
                        or execution.get("blocked") is not False
                        or execution.get("gate_open") is not False
                        or execution.get("internal_phase") != "idle"
                        or execution.get("unresolved") != []
                        or execution.get("stop_proven") is not True
                        or type(proof_epoch) is not int or type(dispatcher_epoch) is not int
                        or dispatcher_epoch < 0 or proof_epoch < dispatcher_epoch):
                    raise TaskError("unresolved_execution")
                def migrate(current):
                    if (not current["active"] or current["generation"] != generation
                            or current["ex_session"] != task["ex_session"]
                            or current["current_goal"] != task["current_goal"]
                            or current["status"] != "resume_review"):
                        raise TaskError("stale_generation")
                    goal = current["current_goal"]
                    if goal:
                        # Retain the old admission for review, never claim completion.
                        current["reviewed_goal"] = {**goal, "feedback_status": "unknown",
                            "review_required": True,
                            "reason_code": "explicit_stop_review" if same_session else "ex_session_replaced",
                            "retirement_audit": {"ex_session": state["ex_session"],
                                "revision": state["revision"], "event_seq": state["event_seq"],
                                "generation": generation, "stop_proof_epoch": proof_epoch,
                                "dispatcher_epoch": dispatcher_epoch}}
                        current["steps"][current["active_step"]]["status"] = "awaiting_review"
                    applied_seq = self.store.cursor(current["robot_id"], state["ex_session"])
                    if same_session:
                        applied_seq = max(current["last_event_seq"], applied_seq)
                    current.update(ex_session=state["ex_session"], goal_revision=state["revision"],
                        last_event_seq=applied_seq,
                        reviewed_session={"task_id": current["task_id"], "robot_id": current["robot_id"],
                                          "ex_session": state["ex_session"]},
                        current_goal=None, status="resume_review", needs_planning=False)
                self.store.mutate(task_id, migrate)
                await self.sync_turn(task_id, "invalidate", expected_generation=generation)
                def wait_for_input(current):
                    if current["generation"] != generation or current["status"] != "resume_review":
                        raise TaskError("stale_generation")
                    current.update(status="waiting_input", needs_planning=False)
                self.store.mutate(task_id, wait_for_input)
            return  # Retired authority needs new user intent; never replay the old plan.
        if not task["ex_session"]:
            self.store.mutate(task_id, lambda t: t.update(ex_session=context["ex_session"], goal_revision=context["revision"]))
        await self.sync_turn(task_id, "invalidate")
        await self.sync(task_id)
        def review(task):
            if task["current_goal"]:
                raise TaskError("unresolved_execution")
            task["status"] = "planning"
            task["needs_planning"] = True
        self.store.mutate(task_id, review)
        self.wake(task_id)

    async def heartbeat_once(self) -> None:
        for task_id in list(self._lease_identities):
            self._prune_lease(task_id)
        for snapshot in self.store.active_tasks():
            task_id = snapshot["task_id"]
            # Earlier robots may await IO; do not pair their old snapshot with a
            # replacement goal's newly created local lease when this robot is reached.
            self._prune_lease(task_id)
            task = self.store.get(task_id)
            goal = task["current_goal"]
            identity = self._lease_identities.get(task_id)
            deadline = self._lease_deadlines.get(task_id)
            if not identity or deadline is None or self.store.lease_identity(task) != identity:
                continue
            # Expire ambiguous submits too; revision=None does not permit revival.
            if self.clock() >= deadline:
                self._expire_lease(task_id, identity, deadline)
                continue
            if not goal or goal["revision"] is None:
                continue
            lease_ms = goal["payload"]["lease_ms"]
            result = None
            try:
                result = await asyncio.wait_for(self._call(task, "decision.goal.renew", {
                    "schema_version": 1, "request_id": uuid.uuid4().hex,
                    "ex_session": task["ex_session"], "goal_id": goal["payload"]["goal_id"],
                    "goal_revision": goal["revision"], "lease_ms": lease_ms}),
                    timeout=max(0.001, min(self.heartbeat_interval, deadline - self.clock())))
            except Exception:
                pass  # Never extend locally on timeout. EX owns the physical lease stop.
            self._prune_lease(task_id)
            # Both the exact goal/generation and the captured lease must still exist.
            # A completed/canceled/recovered/expired lease cannot be resurrected by IO.
            def renewed(current):
                if (self.store.lease_identity(current) != identity
                        or self._lease_identities.get(task_id) != identity
                        or self._lease_deadlines.get(task_id) != deadline):
                    return
                now = self.clock()
                if now >= deadline:
                    current.update(status="lease_lost", needs_planning=False,
                                   generation=current["generation"] + 1, turn_id=None)
                    self._forget_lease(task_id)
                elif isinstance(result, dict) and result.get("ok") is True:
                    self._lease_deadlines[task_id] = now + lease_ms / 1000
            before_generation = self.store.get(task_id)["generation"]
            self.store.mutate(task_id, renewed)
            if self.store.get(task_id)["generation"] != before_generation:
                self._invalidate_public(task_id)
        await self.drain_turn_sync()

    async def _reconcile_loop(self) -> None:
        while not self._closed:
            # Receipt ACK survives reconnect/EX outbox removal. Polling is not a
            # planning wake: only a reconciled terminal fact can request planning.
            for task in self.store.active_tasks():
                if task["ex_session"] and task["current_goal"]:
                    try:
                        await asyncio.wait_for(self.sync(task["task_id"]), 5.0)
                    except Exception:
                        pass
            await asyncio.sleep(max(2.0, self.heartbeat_interval))

    async def _heartbeat_loop(self) -> None:
        while not self._closed:
            await self.heartbeat_once()
            await asyncio.sleep(self.heartbeat_interval)
