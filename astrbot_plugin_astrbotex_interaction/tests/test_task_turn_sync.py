from __future__ import annotations

import asyncio
import copy
from pathlib import Path

from astrbot_plugin_astrbotex_interaction.task_store import TaskStore

from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import CoordinatorFixture, goal_args
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import authority, steps
from astrbot_plugin_astrbotex_interaction.tests.test_planning_tools import FakeHost, response

FIELDS = {"task_schema_version", "operation", "ex_session", "task_id", "robot_id", "session_id",
          "user_id", "route_ref", "turn_id", "generation", "expected_revision"}


class TurnSyncTests(CoordinatorFixture):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.turns = []
        async def callback(payload):
            self.turns.append(payload)
        self.coordinator.turn_sync = callback

    async def test_begin_context_submit_refresh_and_finish_preserves_public_turn(self):
        self.coordinator.host = FakeHost([response(["save_plan", "submit_current_goal", "emit_user_message", "finish_planning_turn"],
            [{"steps": steps(), "expected_revision": 0}, __import__(
                "astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator", fromlist=["goal_args"]).goal_args(),
             {"text": "Trying."}, {"outcome": "waiting_feedback"}])])
        await self.coordinator._planning_worker(self.task["task_id"])
        self.coordinator.host = None
        self.assertEqual([p["operation"] for p in self.turns], ["bind", "bind", "bind"])
        self.assertEqual([p["expected_revision"] for p in self.turns], [0, 0, 1])
        for payload in self.turns:
            self.assertEqual(set(payload), FIELDS)
            self.assertEqual(payload["task_schema_version"], 1)
            self.assertEqual(payload["user_id"], "user-s1")
            self.assertIsNotNone(payload["turn_id"])
        self.assertIsNone(self.current()["turn_id"])
        # EX last bind stays valid for already emitted queued text/audio after finish.
        self.assertEqual(self.public[0]["turn_id"], self.turns[-1]["turn_id"])
        self.assertEqual(self.public[0]["generation"], self.turns[-1]["generation"])

    async def test_target_update_invalidates_before_cancel_or_LLM_completes(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        self.turns.clear()
        original = self.coordinator.request
        async def request(*args):
            if args[2] == "decision.goal.cancel":
                self.assertEqual(self.turns[-1]["operation"], "invalidate")
                self.assertGreater(self.turns[-1]["generation"], turn.generation)
            return await original(*args)
        self.coordinator.request = request
        await self.coordinator.user_input(self.task["task_id"], authority(), "replacement")
        self.assertIsNone(self.turns[-1]["turn_id"])
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            await self.router.emit(turn, "old public text")

    async def test_late_planning_result_invalid_and_failure_invalidates(self):
        started, release = asyncio.Event(), asyncio.Event()
        class Host:
            async def llm_generate(self, **kwargs):
                started.set()
                await release.wait()
                return response(["emit_user_message", "finish_planning_turn"],
                    [{"text": "old turn"}, {"outcome": "waiting_input"}])
        self.coordinator.host = Host()
        old = asyncio.create_task(self.coordinator._planning_worker(self.task["task_id"]))
        await asyncio.wait_for(started.wait(), 2.0)
        old_generation = self.current()["generation"]
        await self.coordinator.user_input(self.task["task_id"], authority(), "new task")
        self.assertEqual(self.turns[-1]["operation"], "invalidate")
        self.assertGreater(self.turns[-1]["generation"], old_generation)
        self.coordinator.host = None
        release.set()
        await old
        self.assertEqual(self.public, [])
        self.coordinator.host = FakeHost([RuntimeError("secret provider fault")])
        await self.coordinator._planning_worker(self.task["task_id"])
        self.coordinator.host = None
        self.assertEqual(self.turns[-1]["operation"], "invalidate")

    async def test_sync_failure_cannot_run_tools_or_publish(self):
        async def reject(payload):
            raise ConnectionError("EX task gate unavailable")
        self.coordinator.turn_sync = reject
        host = FakeHost([response(["emit_user_message"], [{"text": "do not send"}])])
        self.coordinator.host = host
        self.coordinator.max_failures = 1
        await self.coordinator._planning_worker(self.task["task_id"])
        self.coordinator.host = None
        self.assertEqual(host.calls, [])
        self.assertEqual(self.current()["status"], "waiting_input")
        self.assertEqual(self.current()["error_code"], "turn_sync_failed")
        self.assertEqual(self.public, [])

    async def test_provider_cancellation_invalidates_immediately(self):
        started = asyncio.Event()
        class Host:
            async def llm_generate(self, **kwargs):
                started.set()
                await asyncio.Event().wait()
        self.coordinator.host = Host()
        worker = asyncio.create_task(self.coordinator._planning_worker(self.task["task_id"]))
        await asyncio.wait_for(started.wait(), 2.0)
        generation = self.current()["generation"]
        worker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await worker
        self.coordinator.host = None
        self.assertEqual(self.turns[-1]["operation"], "invalidate")
        self.assertGreater(self.turns[-1]["generation"], generation)
        self.assertEqual(self.current()["status"], "resume_review")

    async def test_scheduled_old_invalidation_cannot_revoke_new_bind(self):
        turn = self.turn()
        self.coordinator._invalidate_public(self.task["task_id"])
        self.store.invalidate(self.task["task_id"], authority())
        newer = self.turn()
        await self.coordinator.sync_turn(self.task["task_id"], "bind", turn=newer)
        await self.coordinator.drain_turn_sync()
        self.assertEqual([p["operation"] for p in self.turns], ["bind"])
        self.assertEqual(self.turns[0]["generation"], newer.generation)
        self.assertGreater(newer.generation, turn.generation)

    async def test_lease_expiry_immediately_syncs_invalidate(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        old_generation = self.current()["generation"]
        self.coordinator.clock = lambda: 100.0
        self.coordinator._lease_deadlines[self.task["task_id"]] = 99.0
        await self.coordinator.heartbeat_once()
        self.assertEqual(self.current()["status"], "lease_lost")
        self.assertEqual(self.turns[-1]["operation"], "invalidate")
        self.assertGreater(self.turns[-1]["generation"], old_generation)
        self.assertIsNone(self.turns[-1]["turn_id"])

    async def fresh_review_fixture(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        self.coordinator.finish(turn, "waiting_feedback")
        self.store.mutate(self.task["task_id"], lambda t: t.update(last_event_seq=8))
        self.store.db.execute("INSERT INTO cursors VALUES('r1','ex1',8)")
        self.turns.clear()
        self.ex.calls.clear()
        context = {**self.ex.context, "ex_session": "ex2", "revision": 0}
        state = {**self.ex.state(), "ex_session": "ex2", "revision": 0,
                 "active_goal_id": None, "active_phase": None,
                 "execution": {"blocked": False, "gate_open": False,
                               "internal_phase": "idle", "unresolved": [],
                               "stop_proven": True, "stop_proof_epoch": 7, "dispatcher_epoch": 7,
                               "previous_session_manual_review": True}}
        async def request(robot, route, method, payload):
            self.ex.calls.append((robot, route, method, copy.deepcopy(payload)))
            if method == "decision.context.get":
                return copy.deepcopy(context)
            if method == "decision.state.get":
                self.assertEqual(payload["ex_session"], "ex2")
                return copy.deepcopy(state)
            raise AssertionError("review must not execute: " + method)
        async def turn_sync(payload):
            self.assertEqual(payload["ex_session"], "ex2", "old session invalidate is illegal")
            self.assertEqual(payload["expected_revision"], 0)
            self.turns.append(payload)
        self.coordinator.request = request
        self.coordinator.turn_sync = turn_sync
        return state

    async def test_explicit_fresh_session_review_proven_stop_never_replays(self):
        await self.fresh_review_fixture()
        before = self.current()
        await self.coordinator.review_resume(self.task["task_id"], authority())
        current = self.current()
        self.assertEqual(current["ex_session"], "ex2")
        self.assertEqual(current["goal_revision"], 0)
        self.assertEqual(current["last_event_seq"], 0)
        self.assertEqual(self.store.cursor("r1", "ex1"), 8)
        self.assertIsNone(current["current_goal"])
        self.assertEqual(current["reviewed_goal"]["payload"], before["current_goal"]["payload"])
        self.assertEqual(current["reviewed_goal"]["feedback_status"], "unknown")
        self.assertTrue(current["reviewed_goal"]["review_required"])
        audit = current["reviewed_goal"]["retirement_audit"]
        self.assertEqual(audit["ex_session"], "ex2")
        self.assertEqual(audit["stop_proof_epoch"], 7)
        self.assertEqual(audit["dispatcher_epoch"], 7)
        self.assertEqual(audit["generation"], current["generation"])
        self.assertEqual(current["active_step"], 0)
        self.assertEqual(current["steps"][0]["status"], "awaiting_review")
        self.assertEqual(current["status"], "waiting_input")
        self.assertFalse(current["needs_planning"])
        self.assertEqual(current["reviewed_session"], {"task_id": self.task["task_id"],
            "robot_id": "r1", "ex_session": "ex2"})
        reopened = TaskStore(str(Path(self.tmp.name) / "tasks.db"), recover=False)
        try:
            self.assertEqual(reopened.get(self.task["task_id"])["reviewed_session"], current["reviewed_session"])
        finally:
            reopened.close()
        self.assertNotIn("completion_evidence_seq", current)
        self.assertEqual([p["operation"] for p in self.turns], ["invalidate"])
        await self.coordinator.heartbeat_once()
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_identities)
        self.assertEqual([c[2] for c in self.ex.calls], ["decision.context.get", "decision.state.get"])
        self.assertEqual(len(self.ex.goals), 1)
        await self.coordinator.review_resume(self.task["task_id"], authority())
        self.assertEqual(self.current()["status"], "waiting_input")
        self.assertFalse(self.current()["needs_planning"], "repeated review must not replay old intent")
        self.assertEqual(self.current()["active_step"], 0)
        self.assertEqual(len(self.ex.goals), 1)

    async def test_fresh_session_review_failed_invalidation_stays_review_without_lease(self):
        await self.fresh_review_fixture()
        async def failed(payload):
            raise ConnectionError("fresh EX unavailable")
        self.coordinator.turn_sync = failed
        with self.assertRaisesRegex(TaskError, "turn_sync_failed"):
            await self.coordinator.review_resume(self.task["task_id"], authority())
        self.assertEqual(self.current()["ex_session"], "ex2")
        self.assertIsNone(self.current()["current_goal"])
        self.assertTrue(self.current()["reviewed_goal"]["review_required"])
        self.assertEqual(self.current()["status"], "resume_review")
        self.assertFalse(self.current()["needs_planning"])
        await self.coordinator.heartbeat_once()
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_identities)
        self.assertFalse(any(c[2] == "decision.goal.renew" for c in self.ex.calls))

    async def test_fresh_session_review_absence_or_blocked_never_migrates_or_renews(self):
        state = await self.fresh_review_fixture()
        safe = copy.deepcopy(state)
        before = self.current()
        for patch in ({"stop_proven": False}, {"stop_proven": None}, {"stop_proven": 1},
                      {"blocked": True}, {"gate_open": True}, {"stop_proof_epoch": 6},
                      {"stop_proof_epoch": True}, {"dispatcher_epoch": -1},
                      {"dispatcher_epoch": None}, {"stop_proof_epoch": None},
                      {"unresolved": [{"command_id": "old-command"}]}, {"internal_phase": "stopping"}):
            with self.subTest(patch=patch):
                state["execution"] = {**safe["execution"], **patch}
                with self.assertRaisesRegex(TaskError, "unresolved_execution"):
                    await self.coordinator.review_resume(self.task["task_id"], authority())
                self.assertEqual(self.current()["ex_session"], "ex1")
                self.assertNotIn("reviewed_session", self.current())
                self.assertEqual(self.current()["goal_revision"], before["goal_revision"])
                self.assertEqual(self.current()["current_goal"], before["current_goal"])
                self.assertEqual(self.current()["status"], "resume_review")
                self.assertFalse(self.current()["needs_planning"])
                await self.coordinator.heartbeat_once()
        for missing in ("stop_proven", "stop_proof_epoch", "dispatcher_epoch", "gate_open"):
            with self.subTest(missing=missing):
                state["execution"] = {k: v for k, v in safe["execution"].items() if k != missing}
                with self.assertRaisesRegex(TaskError, "unresolved_execution"):
                    await self.coordinator.review_resume(self.task["task_id"], authority())
                self.assertEqual(self.current()["ex_session"], "ex1")
                self.assertEqual(self.current()["last_event_seq"], 8)
                self.assertNotIn("reviewed_session", self.current())
                self.assertEqual(self.current()["current_goal"], before["current_goal"])
                self.assertEqual(self.current()["status"], "resume_review")
                self.assertFalse(self.current()["needs_planning"])
        state["execution"] = safe["execution"]
        for key, phase in (("active_goal_id", "active_phase"), ("pending_goal_id", "pending_phase")):
            state[key], state[phase] = "foreign-goal", "active" if key == "active_goal_id" else "blocked"
            with self.assertRaisesRegex(TaskError, "unresolved_execution"):
                await self.coordinator.review_resume(self.task["task_id"], authority())
            state[key], state[phase] = None, None
        self.assertEqual(self.turns, [])
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_identities)
        self.assertFalse(any(c[2] in {"decision.goal.submit", "decision.goal.renew"} for c in self.ex.calls))

    async def test_fresh_review_context_state_session_mismatch_retains_old_records(self):
        state = await self.fresh_review_fixture()
        before = self.current()
        state["ex_session"] = "ex3"
        with self.assertRaisesRegex(TaskError, "stale_ex_session"):
            await self.coordinator.review_resume(self.task["task_id"], authority())
        current = self.current()
        for field in ("ex_session", "goal_revision", "last_event_seq", "current_goal", "steps"):
            self.assertEqual(current[field], before[field])
        self.assertEqual(current["status"], "resume_review")
        self.assertFalse(current["needs_planning"])
        self.assertNotIn("reviewed_goal", current)
        self.assertNotIn("reviewed_session", current)
        self.assertEqual(self.turns, [])
        await self.coordinator.heartbeat_once()
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_identities)

    async def test_repeated_same_session_review_preserves_applied_watermarks(self):
        state = await self.fresh_review_fixture()
        await self.coordinator.review_resume(self.task["task_id"], authority())
        self.store.db.execute("INSERT INTO cursors VALUES('r1','ex2',2)")
        self.store.mutate(self.task["task_id"], lambda t: t.update(last_event_seq=4))
        audit = self.current()["reviewed_goal"]
        state["event_seq"] = 3
        with self.assertRaisesRegex(TaskError, "stale_state"):
            await self.coordinator.review_resume(self.task["task_id"], authority())
        self.assertEqual(self.current()["last_event_seq"], 4)
        self.assertEqual(self.current()["status"], "resume_review")
        state["event_seq"] = 5
        await self.coordinator.review_resume(self.task["task_id"], authority())
        self.assertEqual(self.current()["last_event_seq"], 4)
        self.assertEqual(self.store.cursor("r1", "ex2"), 2)
        self.assertEqual(self.store.cursor("r1", "ex1"), 8)
        self.assertEqual(self.current()["reviewed_goal"], audit)
        self.assertEqual(self.current()["status"], "waiting_input")
        self.assertFalse(self.current()["needs_planning"])
        self.assertEqual(self.current()["active_step"], 0)
        self.assertEqual(len(self.ex.goals), 1)

    async def test_fresh_session_review_late_context_cannot_replace_new_intent(self):
        await self.fresh_review_fixture()
        original = self.coordinator.request
        async def replaced(*args):
            result = await original(*args)
            if args[2] == "decision.state.get":
                self.store.invalidate(self.task["task_id"], authority(), "replacement")
            return result
        self.coordinator.request = replaced
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            await self.coordinator.review_resume(self.task["task_id"], authority())
        self.assertEqual(self.current()["ex_session"], "ex1")
        self.assertEqual(self.current()["text"], "replacement")
        self.assertNotIn("reviewed_session", self.current())
        self.assertIsNotNone(self.current()["current_goal"])
        self.assertEqual(self.turns, [])

    async def test_reviewed_session_user_intent_feedback_does_not_reenter_review(self):
        await self.fresh_review_fixture()
        await self.coordinator.review_resume(self.task["task_id"], authority())
        retired = self.current()["reviewed_goal"]
        self.ex.context.update(ex_session="ex2", revision=0)
        self.ex.active, self.ex.fact = None, None
        self.ex.events.clear()
        async def request(*args):
            result = await self.ex(*args)
            result["ex_session"] = "ex2"
            if args[2] == "decision.state.get":
                result["execution"]["previous_session_manual_review"] = True
                if result["execution"].get("feedback"):
                    result["execution"]["feedback"]["ex_session"] = "ex2"
            if args[2] == "decision.events.get":
                for event in result["events"]:
                    event["ex_session"] = "ex2"
            return result
        self.coordinator.request = request
        async def turn_sync(payload):
            self.turns.append(payload)
        self.coordinator.turn_sync = turn_sync
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["status"], "waiting_input")
        self.assertFalse(self.current()["needs_planning"])
        await self.coordinator.user_input(self.task["task_id"], authority(), "explicit new target")
        self.coordinator.host = FakeHost([response(["save_plan", "submit_current_goal", "finish_planning_turn"],
            [{"steps": steps(1, "new-target"), "expected_revision": 1}, goal_args("new-target-0"),
             {"outcome": "waiting_feedback"}])])
        try:
            await self.coordinator._planning_worker(self.task["task_id"])
        finally:
            self.coordinator.host = None
        current = self.current()
        self.assertEqual(current["plan_revision"], 2)
        self.assertEqual(current["current_goal"]["payload"]["step_id"], "new-target-0")
        self.assertNotEqual(current["current_goal"]["payload"]["goal_id"], retired["payload"]["goal_id"])
        generation = current["generation"]
        running = {**self.ex.feedback(current, "running"), "ex_session": "ex2"}
        await self.coordinator.feedback("r1", "route-s1", running)
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["status"], "executing")
        self.assertEqual(self.current()["generation"], generation)
        self.assertEqual(self.current()["current_goal"]["last_feedback"]["event_seq"], 1)
        succeeded = {**self.ex.feedback(self.current(), "succeeded"), "ex_session": "ex2"}
        await self.coordinator.feedback("r1", "route-s1", succeeded)
        self.assertEqual(self.current()["active_step"], 1)
        self.assertEqual(self.current()["status"], "planning")
        self.assertEqual(self.current()["reviewed_goal"], retired)
        self.assertEqual(self.store.cursor("r1", "ex2"), 2)
        self.assertEqual(self.store.cursor("r1", "ex1"), 8)
        self.assertEqual(len(self.ex.goals), 2)
        self.assertEqual(self.public, [])

    async def test_review_marker_is_scoped_to_task_robot_and_session(self):
        state = await self.fresh_review_fixture()
        await self.coordinator.review_resume(self.task["task_id"], authority())
        marker = self.current()["reviewed_session"]
        original = self.coordinator.request
        async def request(*args):
            if args[2] == "decision.events.get":
                return {"schema_version": 1, "ex_session": "ex2", "events": [],
                    "oldest_available_seq": 0, "latest_event_seq": 0, "resync_required": False}
            return await original(*args)
        self.coordinator.request = request
        for key in ("task_id", "robot_id", "ex_session"):
            with self.subTest(key=key):
                self.store.mutate(self.task["task_id"], lambda t: t.update(
                    reviewed_session={**marker, key: "foreign"}, status="waiting_input"))
                await self.coordinator.sync(self.task["task_id"])
                self.assertEqual(self.current()["status"], "resume_review")
                self.assertFalse(self.current()["needs_planning"])
        self.store.mutate(self.task["task_id"], lambda t: t.update(reviewed_session=marker, status="waiting_input"))
        state["ex_session"] = "ex3"
        with self.assertRaisesRegex(TaskError, "stale_ex_session"):
            await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["status"], "resume_review")
        self.assertEqual(self.current()["reviewed_session"], marker)
        self.assertEqual(self.current()["ex_session"], "ex2")

    async def same_session_terminal_fixture(self, status="timed_out"):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        self.coordinator.finish(turn, "waiting_feedback")
        terminal = self.ex.feedback(self.current(), status)
        self.ex.active = None  # Deterministic EX fixture: released goal, not proof by itself.
        execution = {"blocked": False, "gate_open": False, "internal_phase": "idle",
                     "unresolved": [], "stop_proven": True, "stop_proof_epoch": 7,
                     "dispatcher_epoch": 7}
        overrides = {}
        async def request(*args):
            result = await self.ex(*args)
            if args[2] == "decision.state.get":
                result["execution"].update(execution)
                result.update(overrides)
            return result
        self.coordinator.request = request
        await self.coordinator.feedback("r1", "route-s1", terminal)
        self.assertEqual(self.current()["status"], "resume_review")
        self.assertEqual(self.current()["current_goal"]["feedback_status"], status)
        self.assertFalse(self.current()["needs_planning"])
        self.turns.clear()
        return execution, overrides

    async def test_same_session_proven_timed_out_review_waits_for_new_intent(self):
        await self.same_session_terminal_fixture()
        before = self.current()
        old_goal = before["current_goal"]
        await self.coordinator.review_resume(self.task["task_id"], authority())
        current = self.current()
        self.assertEqual(current["status"], "waiting_input")
        self.assertFalse(current["needs_planning"])
        self.assertEqual(current["ex_session"], "ex1")
        self.assertEqual(current["goal_revision"], before["goal_revision"])
        self.assertEqual(current["last_event_seq"], before["last_event_seq"])
        self.assertEqual(self.store.cursor("r1", "ex1"), 1)
        self.assertIsNone(current["current_goal"])
        self.assertEqual(current["reviewed_goal"]["payload"], old_goal["payload"])
        self.assertEqual(current["reviewed_goal"]["last_feedback"]["status"], "timed_out")
        self.assertEqual(current["reviewed_goal"]["feedback_status"], "unknown")
        self.assertTrue(current["reviewed_goal"]["review_required"])
        self.assertEqual(current["active_step"], 0)
        self.assertEqual(current["steps"][0]["status"], "awaiting_review")
        self.assertNotIn("completion_evidence_seq", current)
        self.assertEqual(current["reviewed_session"], {"task_id": self.task["task_id"],
            "robot_id": "r1", "ex_session": "ex1"})
        await self.coordinator.heartbeat_once()
        self.assertEqual(len(self.ex.goals), 1)
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_identities)
        await self.coordinator.user_input(self.task["task_id"], authority(), "explicit replacement after timeout")
        self.coordinator.host = FakeHost([response(["save_plan", "submit_current_goal", "finish_planning_turn"],
            [{"steps": steps(1, "after-timeout"), "expected_revision": 1}, goal_args("after-timeout-0"),
             {"outcome": "waiting_feedback"}])])
        try:
            await self.coordinator._planning_worker(self.task["task_id"])
        finally:
            self.coordinator.host = None
        self.assertEqual(self.current()["status"], "executing")
        self.assertEqual(self.current()["plan_revision"], 2)
        self.assertEqual(self.current()["current_goal"]["payload"]["step_id"], "after-timeout-0")
        self.assertNotEqual(self.current()["current_goal"]["payload"]["goal_id"], old_goal["payload"]["goal_id"])
        self.assertEqual(self.current()["goal_revision"], 2)
        self.assertEqual(self.store.cursor("r1", "ex1"), 1)
        self.assertEqual(len(self.ex.goals), 2)
        self.assertEqual(self.public, [])

    async def test_same_session_terminal_review_requires_full_proof_and_current_state(self):
        execution, overrides = await self.same_session_terminal_fixture("unknown")
        safe = dict(execution)
        before = self.current()
        for patch in ({"stop_proven": False}, {"stop_proven": None}, {"stop_proof_epoch": True},
                      {"stop_proof_epoch": 6}, {"dispatcher_epoch": "7"}, {"blocked": True},
                      {"gate_open": True}, {"unresolved": [{"command_id": "old"}]}):
            with self.subTest(patch=patch):
                execution.clear()
                execution.update({**safe, **patch})
                with self.assertRaisesRegex(TaskError, "unresolved_execution"):
                    await self.coordinator.review_resume(self.task["task_id"], authority())
                self.assertEqual(self.current()["current_goal"], before["current_goal"])
                self.assertEqual(self.current()["status"], "resume_review")
                self.assertFalse(self.current()["needs_planning"])
                self.assertNotIn("reviewed_session", self.current())
        execution.clear()
        execution.update(safe)
        del execution["stop_proven"]
        with self.assertRaisesRegex(TaskError, "unresolved_execution"):
            await self.coordinator.review_resume(self.task["task_id"], authority())
        execution.update(safe)
        for patch, code in (({"active_goal_id": self.ex.fact["goal_id"], "active_phase": "active"}, "unresolved_execution"),
                            ({"pending_goal_id": self.ex.fact["goal_id"], "pending_phase": "blocked"}, "unresolved_execution"),
                            ({"revision": 0}, "stale_state"), ({"event_seq": 0}, "stale_state"),
                            ({"ex_session": "ex2"}, "stale_ex_session")):
            with self.subTest(patch=patch):
                overrides.clear()
                overrides.update(patch)
                with self.assertRaisesRegex(TaskError, code):
                    await self.coordinator.review_resume(self.task["task_id"], authority())
                self.assertEqual(self.current()["current_goal"], before["current_goal"])
                self.assertEqual(self.current()["last_event_seq"], 1)
                self.assertNotIn("reviewed_session", self.current())
        self.assertEqual(self.turns, [])
        self.assertEqual(self.store.cursor("r1", "ex1"), 1)
        await self.coordinator.heartbeat_once()
        self.assertEqual(len(self.ex.goals), 1)
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_identities)

    async def test_same_session_terminal_review_late_CAS_cannot_retire_new_intent(self):
        await self.same_session_terminal_fixture()
        original = self.coordinator.request
        async def replaced(*args):
            result = await original(*args)
            if args[2] == "decision.state.get":
                self.store.invalidate(self.task["task_id"], authority(), "newer user intent")
            return result
        self.coordinator.request = replaced
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            await self.coordinator.review_resume(self.task["task_id"], authority())
        self.assertEqual(self.current()["text"], "newer user intent")
        self.assertIsNotNone(self.current()["current_goal"])
        self.assertNotIn("reviewed_session", self.current())
        self.assertNotIn("reviewed_goal", self.current())
        self.assertEqual(self.turns, [])
        self.assertEqual(self.store.cursor("r1", "ex1"), 1)

    async def test_lease_expiry_invalidate_and_cancel_waits_for_evidence(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        await self.coordinator.cancel_task(self.task["task_id"], authority())
        self.assertEqual(self.current()["status"], "canceling")
        self.assertTrue(self.current()["active"])
        self.assertEqual(self.turns[-1]["operation"], "invalidate")
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "canceled"))
        self.assertEqual(self.current()["status"], "canceled")
        self.assertFalse(self.current()["active"])
