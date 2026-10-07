from __future__ import annotations

import copy

from astrbot_plugin_astrbotex_interaction.task_contracts import Feedback
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import CoordinatorFixture
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import steps


class DecisionReplanTests(CoordinatorFixture):
    async def prepared(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        self.coordinator.finish(turn, "waiting_feedback")

    def rejection(self, reason="low_confidence", *, commands=None):
        fact = self.ex.feedback(self.current(), "failed")
        fact["reason_code"] = reason
        fact["details"].pop("failure_evidence")
        fact["details"]["terminal_evidence"]["commands"] = commands or []
        fact["details"]["decision_rejection"] = {"choices": [{"action": "pick", "score": 0.12}], "truncated": False}
        self.ex.fact = copy.deepcopy(fact)
        self.ex.events[-1].update(reason_code=reason, details=copy.deepcopy(fact["details"]))
        return fact

    async def test_rejection_before_dispatch_needs_no_failed_action(self):
        for reason in ("low_confidence", "backend_requested_replan"):
            with self.subTest(reason=reason):
                if not self.current()["steps"]:
                    await self.prepared()
                else:
                    turn = self.turn()
                    self.store.save_plan(turn, steps(2, reason), self.current()["plan_revision"])
                    await self.submit(turn, f"{reason}-0", reason)
                    self.coordinator.finish(turn, "waiting_feedback")
                fact = self.rejection(reason)
                await self.coordinator.feedback("r1", "route-s1", fact)
                self.assertIsNone(self.current()["current_goal"])
                self.assertTrue(self.current()["needs_planning"])
                self.assertEqual(self.current()["status"], "planning")
                self.assertEqual(self.current()["active_step"], 0)
                self.assertEqual(self.current()["steps"][0]["status"], "failed")
                generation = self.current()["generation"]
                await self.coordinator.feedback("r1", "route-s1", fact)
                self.assertEqual(self.current()["generation"], generation)

    async def test_completed_prefix_preserved_and_only_rejected_goal_retired(self):
        await self.prepared()
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "succeeded"))
        completed = copy.deepcopy(self.current()["steps"][0])
        turn = self.turn()
        await self.submit(turn, "step-1", "second")
        self.coordinator.finish(turn, "waiting_feedback")
        await self.coordinator.feedback("r1", "route-s1", self.rejection("backend_requested_replan"))
        self.assertEqual(self.current()["active_step"], 1)
        self.assertEqual(self.current()["steps"][0], completed)
        self.assertEqual(self.current()["steps"][2]["status"], "planned")
        self.store.save_plan(self.turn(), steps(2, "repair"), 1)
        self.assertEqual(self.current()["steps"], [completed] + [dict(s, status="planned") for s in steps(2, "repair")])

    async def test_missing_mismatched_or_unresolved_stop_evidence_fails_closed(self):
        await self.prepared()
        fact = self.rejection()
        base = self.current()
        state = self.ex.state()
        variants = []
        for key in ("terminal_evidence", "stop_evidence"):
            bad = copy.deepcopy(fact)
            bad["details"].pop(key)
            variants.append((bad, state))
        for key, value in (("verified", False), ("goal_id", "foreign"), ("goal_revision", True),
                           ("goal_revision", 99), ("dispatcher_epoch", True), ("dispatcher_epoch", -1),
                           ("stop_proof_epoch", 1), ("stop_proof_epoch", 3), ("commands", None)):
            bad = copy.deepcopy(fact)
            bad["details"]["terminal_evidence"][key] = value
            variants.append((bad, state))
        for key, value in (("blocked", True), ("gate_open", True), ("internal_phase", "stopping"),
                           ("unresolved", [{"held_resources": ["arm"]}]), ("stop_proven", False),
                           ("dispatcher_epoch", 3), ("stop_proof_epoch", True)):
            bad_state = copy.deepcopy(state)
            bad_state["execution"][key] = value
            variants.append((fact, bad_state))
        for key in ("active_goal_id", "pending_goal_id"):
            bad_state = copy.deepcopy(state)
            bad_state[key] = fact["goal_id"]
            variants.append((fact, bad_state))
        for raw, snapshot in variants:
            with self.subTest(raw=raw["details"], execution=snapshot["execution"]):
                current = copy.deepcopy(base)
                snapshot = copy.deepcopy(snapshot)
                snapshot["execution"]["feedback"] = raw
                self.coordinator._apply_state(current, snapshot)
                self.assertIsNotNone(current["current_goal"])
                self.assertEqual(current["status"], "resume_review")
                self.assertFalse(current["needs_planning"])

    async def test_stop_commands_from_real_EX_include_canceled_not_failed(self):
        await self.prepared()
        goal = self.current()["current_goal"]
        command = {"command_id": "arm-1", "ex_session": "ex1", "goal_id": goal["payload"]["goal_id"],
                   "goal_revision": goal["revision"], "event_seq": 5, "status": "canceled",
                   "stop_evidence": {"stopped": True, "source": "arm", "reference": "arm-1"}}
        fact = self.rejection("backend_requested_replan", commands=[command])
        base = self.current()
        state = self.ex.state()
        for key, value in (("status", "running"), ("status", "unknown"), ("ex_session", "ex2"),
                           ("goal_revision", 99), ("stop_evidence", {"stopped": False})):
            bad_state = copy.deepcopy(state)
            bad_state["execution"]["feedback"]["details"]["terminal_evidence"]["commands"][0][key] = value
            current = copy.deepcopy(base)
            self.coordinator._apply_state(current, bad_state)
            self.assertEqual(current["status"], "resume_review")
            self.assertIsNotNone(current["current_goal"])
        await self.coordinator.feedback("r1", "route-s1", fact)
        self.assertTrue(self.current()["needs_planning"])

    async def test_physical_failure_still_requires_failed_stopped_action(self):
        await self.prepared()
        fact = self.ex.feedback(self.current(), "failed")
        base = self.current()
        state = self.ex.state()
        for change in ("missing", "empty", "foreign", "unverified"):
            bad_state = copy.deepcopy(state)
            details = bad_state["execution"]["feedback"]["details"]
            if change == "missing":
                details.pop("failure_evidence")
            elif change == "empty":
                details["failure_evidence"]["commands"] = []
            elif change == "foreign":
                details["failure_evidence"]["goal_id"] = "other"
            else:
                details["failure_evidence"]["verified"] = False
            current = copy.deepcopy(base)
            self.coordinator._apply_state(current, bad_state)
            self.assertIsNotNone(current["current_goal"])
            self.assertFalse(current["needs_planning"])
        await self.coordinator.feedback("r1", "route-s1", fact)
        self.assertIsNone(self.current()["current_goal"])
        self.assertTrue(self.current()["needs_planning"])

    async def test_replan_reason_never_bypasses_real_physical_failure_evidence(self):
        await self.prepared()
        fact = self.ex.feedback(self.current(), "failed")
        fact["reason_code"] = "low_confidence"
        fact["details"].pop("failure_evidence")
        self.ex.fact = copy.deepcopy(fact)
        self.ex.events[-1].update(reason_code="low_confidence", details=copy.deepcopy(fact["details"]))
        await self.coordinator.feedback("r1", "route-s1", fact)
        self.assertIsNotNone(self.current()["current_goal"])
        self.assertFalse(self.current()["needs_planning"])
        self.assertEqual(self.current()["status"], "resume_review")

    async def test_unknown_timeout_and_cancel_never_auto_replan(self):
        await self.prepared()
        base = self.current()
        fact = self.ex.feedback(base, "canceled")
        for status in ("unknown", "timed_out", "canceled"):
            current = copy.deepcopy(base)
            state = copy.deepcopy(self.ex.state())
            state["execution"]["feedback"]["status"] = status
            self.coordinator._apply_state(current, state)
            self.assertFalse(current["needs_planning"])
            if status != "canceled":
                self.assertEqual(current["status"], "resume_review")
                self.assertIsNotNone(current["current_goal"])
            else:
                self.assertEqual(current["status"], "waiting_input")
        self.store.mutate(base["task_id"], lambda t: t.update(cancel_requested=True, needs_planning=False))
        await self.coordinator.feedback("r1", "route-s1", fact)
        self.assertFalse(self.current()["active"])
        self.assertFalse(self.current()["needs_planning"])

    async def test_matching_blocked_or_uncertain_state_requires_review_without_terminal_fact(self):
        await self.prepared()
        base = self.current()
        for pair in ("active", "pending"):
            for execution in ({"blocked": True}, {"internal_phase": "blocked"}, *[
                    {"blocked": False, "internal_phase": "stopping", "stop_proven": False,
                     "unresolved": [{"command_id": "arm-1", "status": status, "held_resources": ["arm"]}]}
                    for status in ("unknown", "timed_out", "failed")]):
                with self.subTest(pair=pair, execution=execution):
                    state = self.ex.state()
                    state.update(active_goal_id=None, active_phase=None,
                                 pending_goal_id=None, pending_phase=None)
                    state[f"{pair}_goal_id"] = base["current_goal"]["payload"]["goal_id"]
                    state[f"{pair}_phase"] = "active" if pair == "active" else "blocked"
                    state["execution"] = execution
                    current = copy.deepcopy(base)
                    self.coordinator._apply_state(current, state)
                    self.assertEqual(current["status"], "resume_review")
                    self.assertFalse(current["needs_planning"])
                    self.assertIsNone(current["turn_id"])
                    self.assertEqual(current["generation"], base["generation"] + 1)
                    self.assertEqual(current["current_goal"], base["current_goal"])
                    self.assertEqual(current["steps"], base["steps"])
                    self.coordinator._apply_state(current, state)
                    self.assertEqual(current["generation"], base["generation"] + 1)

    async def test_blocked_sync_retains_prefix_goal_and_review_after_running_feedback(self):
        await self.prepared()
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "succeeded"))
        completed = copy.deepcopy(self.current()["steps"][0])
        turn = self.turn()
        await self.submit(turn, "step-1", "second")
        base = self.current()
        original = self.coordinator.request
        blocked = [True]
        async def execution_state(robot, route, method, payload):
            result = await original(robot, route, method, payload)
            if method == "decision.state.get":
                result["execution"].update(blocked=blocked[0], gate_open=not blocked[0],
                    internal_phase="blocked" if blocked[0] else "active", stop_proven=False,
                    unresolved=[{"command_id": "arm-1", "status": "unknown" if blocked[0] else "running",
                                 "held_resources": ["arm"]}])
                if blocked[0]:
                    result.update(pending_goal_id=result["active_goal_id"], pending_phase="blocked",
                                  active_goal_id=None, active_phase=None)
            return result
        self.coordinator.request = execution_state
        fact = self.ex.feedback(base, "running")
        fact.update(reason_code="ledger_unknown", details={"action_status": "unknown"})
        self.ex.fact = copy.deepcopy(fact)
        self.ex.events[-1].update(reason_code=fact["reason_code"], details=copy.deepcopy(fact["details"]))
        await self.coordinator.feedback("r1", "route-s1", fact)
        await self.coordinator.sync(self.task["task_id"])
        reviewed = self.current()
        self.assertEqual(reviewed["status"], "resume_review")
        self.assertEqual(reviewed["generation"], base["generation"] + 1)
        self.assertIsNone(reviewed["turn_id"])
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            self.coordinator.finish(turn, "waiting_feedback")
        for _ in range(2):
            await self.coordinator.feedback("r1", "route-s1", fact)
            await self.coordinator.sync(self.task["task_id"])
        blocked[0] = False
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "running"))
        await self.coordinator.sync(self.task["task_id"])
        current = self.current()
        self.assertEqual(current["status"], "resume_review")
        self.assertEqual(current["generation"], reviewed["generation"])
        self.assertFalse(current["needs_planning"])
        self.assertIsNone(current["turn_id"])
        self.assertEqual(current["current_goal"]["payload"], base["current_goal"]["payload"])
        self.assertEqual(current["current_goal"]["last_feedback"]["status"], "running")
        self.assertEqual(current["steps"][0], completed)
        self.assertEqual(current["active_step"], 1)
        self.assertEqual(len(self.ex.goals), 2)

    async def test_normal_running_stop_unproven_or_foreign_state_does_not_require_review(self):
        await self.prepared()
        prepared = self.current()
        running = self.ex.feedback(self.current(), "running")
        await self.coordinator.feedback("r1", "route-s1", running)
        base = self.current()
        variants = []
        for phase in ("active", "stopping"):
            state = self.ex.state()
            state["execution"].update(blocked=False, internal_phase=phase, stop_proven=False,
                unresolved=[{"command_id": "arm-1", "status": "running", "held_resources": ["arm"]}])
            variants.append((state, base))
        for foreign in ("session", "active", "pending"):
            state = self.ex.state()
            state["execution"].update(blocked=True, internal_phase="blocked", stop_proven=False,
                unresolved=[{"command_id": "arm-1", "status": "unknown"}])
            expected = base
            if foreign == "session":
                state["ex_session"] = "ex2"
            else:
                state.update(active_goal_id=None, active_phase=None,
                             pending_goal_id=None, pending_phase=None)
                state[f"{foreign}_goal_id"] = "foreign-goal"
                state[f"{foreign}_phase"] = "active" if foreign == "active" else "blocked"
                # Pure foreign-global block, not an orphaned current Goal running receipt.
                state["execution"].pop("feedback", None)
                state["execution"].pop("goal_summaries", None)
                state["execution"].pop("receipt_facts", None)
                expected = prepared
            variants.append((state, expected))
        for state, expected in variants:
            with self.subTest(state=state):
                current = copy.deepcopy(expected)
                self.coordinator._apply_state(current, state)
                self.assertEqual(current, expected)
        # Missing current Goal plus its running receipt is still unresolved, even
        # when a foreign Goal occupies the active slot; never silently resume.
        for foreign in (None, "foreign-goal"):
            state = self.ex.state()
            state.update(active_goal_id=foreign, active_phase="active" if foreign else None,
                         pending_goal_id=None, pending_phase=None)
            current = copy.deepcopy(base)
            self.coordinator._apply_state(current, state)
            self.assertEqual(current["status"], "resume_review")
            self.assertFalse(current["needs_planning"])
            self.assertIsNone(current["turn_id"])
            self.assertEqual(current["current_goal"], base["current_goal"])
            self.assertEqual(current["generation"], base["generation"] + 1)
        state = self.ex.state()
        state["event_seq"] = 0
        state["execution"].update(blocked=True, internal_phase="blocked")
        original = self.coordinator.request
        async def stale(robot, route, method, payload):
            if method == "decision.state.get":
                return state
            return await original(robot, route, method, payload)
        self.coordinator.request = stale
        with self.assertRaisesRegex(TaskError, "stale_state"):
            await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current(), base)

    async def test_cropped_history_recovers_durable_rejection_summary(self):
        await self.prepared()
        fact = self.rejection()
        self.ex.resync = True
        await self.coordinator.sync(self.task["task_id"])
        self.assertIsNone(self.current()["current_goal"])
        self.assertTrue(self.current()["needs_planning"])
        self.assertEqual(self.store.cursor("r1", "ex1"), fact["event_seq"])

    async def test_durable_receipt_is_not_replan_authority_when_state_unavailable(self):
        await self.prepared()
        fact = self.rejection()
        original = self.coordinator.request
        async def offline(*args):
            raise TimeoutError("offline")
        self.coordinator.request = offline
        ack = await self.coordinator.feedback("r1", "route-s1", fact)
        self.assertEqual(ack["acked_event_seq"], 1)
        self.assertFalse(self.current()["needs_planning"])
        self.assertIsNotNone(self.current()["current_goal"])
        self.coordinator.request = original
        await self.coordinator.sync(self.task["task_id"])
        self.assertTrue(self.current()["needs_planning"])
