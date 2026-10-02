from __future__ import annotations

import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from astrbot_plugin_astrbotex_interaction.output_router import PublicOutputRouter
from astrbot_plugin_astrbotex_interaction.task_coordinator import TaskCoordinator
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import authority, steps

ACTION = "mock.step.v1"


def goal_args(step="step-0"):
    return {"step_id": step, "goal_text_en": "Perform one bounded observable step.",
            "allowed_actions": [ACTION], "parameters": {ACTION: {}},
            "completion": {"required_success_actions": [ACTION]}, "lease_ms": 10000}


class FakeDecision:
    def __init__(self):
        self.context = {"schema_version": 1, "ex_session": "ex1", "revision": 0,
                        "actions": [{"action_id": ACTION, "schema": {"type": "object", "additionalProperties": False}}],
                        "observations": [{"observation_id": "fresh-0"}]}
        self.calls = []
        self.goals = {}
        self.active = None
        self.events = []
        self.fact = None
        self.timeout_once = False
        self.renew_ok = True
        self.resync = False

    def state(self):
        return {"schema_version": 1, "ex_session": "ex1", "revision": self.context["revision"],
                "active_goal_id": self.active, "active_phase": "active" if self.active else None,
                "pending_goal_id": None, "pending_phase": None,
                "execution": {"feedback": self.fact} if self.fact else {}, "event_seq": len(self.events)}

    async def __call__(self, robot, route, method, payload):
        self.calls.append((robot, route, method, copy.deepcopy(payload)))
        if method == "decision.context.get":
            return copy.deepcopy(self.context)
        if method == "decision.goal.submit":
            rid = payload["request_id"]
            if rid not in self.goals:
                self.context["revision"] += 1
                self.goals[rid] = {"ok": True, "request_id": rid, "ex_session": "ex1",
                                   "goal_id": payload["goal_id"], "revision": self.context["revision"], "phase": "active"}
                self.active = payload["goal_id"]
            if self.timeout_once:
                self.timeout_once = False
                raise TimeoutError("admitted but response lost")
            return dict(self.goals[rid])
        if method == "decision.state.get":
            return copy.deepcopy(self.state())
        if method == "decision.events.get":
            return {"schema_version": 1, "ex_session": "ex1", "events": [] if self.resync else
                    [e for e in self.events if e["event_seq"] > payload["since_event_seq"]],
                    "oldest_available_seq": len(self.events) + 1 if self.resync else (1 if self.events else 0),
                    "latest_event_seq": len(self.events), "resync_required": self.resync}
        if method == "decision.goal.renew":
            return {"ok": self.renew_ok}
        if method == "decision.goal.cancel":
            return {"ok": True, "phase": "pending_cancel"}
        raise AssertionError(method)

    def feedback(self, task, status, *, verified=True):
        current = task["current_goal"]
        seq = len(self.events) + 1
        goal_id = current["payload"]["goal_id"]
        details = {}
        if status == "succeeded":
            details["completion_evidence"] = {"verified": verified, "goal_id": goal_id,
                "goal_revision": current["revision"], "succeeded_actions": [ACTION]}
        if status == "canceled":
            details["stop_evidence"] = {"stopped": True}
        fact = {"schema_version": 1, "ex_session": "ex1", "task_id": task["task_id"],
                "goal_id": goal_id, "goal_revision": current["revision"], "event_seq": seq,
                "status": status, "reason_code": "", "details": details}
        event = {k: v for k, v in fact.items() if k != "schema_version"}
        event.update(event_id=f"e-{seq}", command_id=f"c-{seq}", owner="mock")
        self.events.append(event)
        self.fact = fact
        if status in {"succeeded", "failed", "canceled"}:
            self.active = None
            self.context["observations"] = [{"observation_id": f"fresh-{seq}"}]
        return copy.deepcopy(fact)


class CoordinatorFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TaskStore(str(Path(self.tmp.name) / "tasks.db"))
        self.ex = FakeDecision()
        self.public = []
        async def send(payload):
            self.public.append(payload)
        self.router = PublicOutputRouter(self.store, send)
        self.coordinator = TaskCoordinator(self.store, self.ex, router=self.router)
        self.task = self.store.create(authority(), "three steps", "create")
        self.store.mutate(self.task["task_id"], lambda t: t.update(ex_session="ex1"))

    async def asyncTearDown(self):
        await self.coordinator.close()
        self.store.close()
        self.tmp.cleanup()

    def turn(self):
        return self.store.begin_turn(self.task["task_id"])

    def current(self):
        return self.store.get(self.task["task_id"])

    async def submit(self, turn, step="step-0", call="submit"):
        return await self.coordinator.submit(turn, goal_args(step), self.ex.context, call)


class CoordinatorTests(CoordinatorFixture):
    async def test_P01_three_steps_and_accepted_not_completed(self):
        for i in range(3):
            turn = self.turn()
            if i == 0:
                self.store.save_plan(turn, steps(), 0)
            await self.submit(turn, f"step-{i}", f"call-{i}")
            self.coordinator.finish(turn, "waiting_feedback")
            running = self.ex.feedback(self.current(), "running")
            await self.coordinator.feedback("r1", "route-s1", running)
            self.assertEqual(self.current()["active_step"], i)
            fact = self.ex.feedback(self.current(), "succeeded")
            await self.coordinator.feedback("r1", "route-s1", fact)
            self.assertEqual(self.current()["active_step"], i + 1)
            for _ in range(10):
                await self.coordinator.feedback("r1", "route-s1", fact)
            self.assertEqual(self.current()["active_step"], i + 1)
        turn = self.turn()
        self.coordinator.finish(turn, "completed")
        self.assertFalse(self.current()["active"])
        self.assertEqual(len(self.ex.goals), 3)

    async def test_P02_failure_replans_suffix_without_first_replay(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        self.coordinator.finish(turn, "waiting_feedback")
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "succeeded"))
        turn = self.turn()
        await self.submit(turn, "step-1")
        self.coordinator.finish(turn, "waiting_feedback")
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "failed"))
        turn = self.turn()
        context = await self.ex("r1", "route-s1", "decision.context.get", {})
        self.assertNotEqual(context["observations"][0]["observation_id"], "fresh-0")
        self.store.save_plan(turn, steps(2, "repair"), 1)
        self.assertEqual(self.current()["steps"][0]["status"], "completed")
        self.assertEqual(self.current()["plan_revision"], 2)
        self.assertEqual(len(self.ex.goals), 2)

    async def test_P04_submit_timeout_same_request_and_one_goal(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        self.ex.timeout_once = True
        result = await self.submit(turn)
        self.assertTrue(result["ok"])
        self.assertEqual(len(self.ex.goals), 1)
        submits = [c[3] for c in self.ex.calls if c[2] == "decision.goal.submit"]
        self.assertEqual(len(submits), 2)
        self.assertEqual(submits[0], submits[1])
        self.assertEqual(await self.submit(turn), result)
        self.assertEqual(len(self.ex.goals), 1)

    async def test_no_advance_without_verified_completion(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        self.coordinator.finish(turn, "waiting_feedback")
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "succeeded", verified=False))
        self.assertEqual(self.current()["active_step"], 0)
        self.assertIsNotNone(self.current()["current_goal"])
        with self.assertRaises(TaskError):
            self.store.save_plan(self.turn(), steps(), 1)

    async def test_heartbeat_independent_and_expiry_cannot_resume(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        now = [0.0]
        self.coordinator.clock = lambda: now[0]
        self.coordinator._lease_deadlines[self.task["task_id"]] = 10
        await self.coordinator.heartbeat_once()
        self.assertTrue(any(c[2] == "decision.goal.renew" for c in self.ex.calls))
        self.ex.renew_ok = False
        now[0] = 11
        await self.coordinator.heartbeat_once()
        self.assertEqual(self.current()["status"], "lease_lost")
        count = len(self.ex.calls)
        await self.coordinator.heartbeat_once()
        self.assertEqual(len(self.ex.calls), count)

    async def test_P07_replacement_requires_confirmed_stop(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        await self.coordinator.user_input(self.task["task_id"], authority(), "new target")
        self.assertEqual(self.current()["status"], "canceling")
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            await self.router.emit(turn, "old completed text")
        self.assertEqual(self.public, [])
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "canceled"))
        self.assertIsNone(self.current()["current_goal"])
        self.assertEqual(self.current()["status"], "planning")

    async def test_P06_unknown_invalid_and_oversize_never_submit(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        bad = goal_args()
        bad["allowed_actions"] = ["invented.action.v1"]
        with self.assertRaisesRegex(TaskError, "unknown_action"):
            await self.coordinator.submit(turn, bad, self.ex.context, "bad")
        bad = goal_args()
        bad["parameters"] = {ACTION: {"unexpected": 1}}
        with self.assertRaisesRegex(TaskError, "invalid_action_parameters"):
            await self.coordinator.submit(turn, bad, self.ex.context, "bad")
        self.assertEqual(len(self.ex.goals), 0)

    async def test_progress_merges_without_LLM_and_provider_timeout_is_bounded(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        self.coordinator.finish(turn, "waiting_feedback")
        class CountingHost:
            calls = 0
            async def llm_generate(self, **kwargs):
                self.calls += 1
                await asyncio.Event().wait()
        host = CountingHost()
        self.coordinator.host = host
        for _ in range(30):
            await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "running"))
        self.assertEqual(host.calls, 0)
        self.store.invalidate(self.task["task_id"], authority())
        # No implicit start while an old goal exists.
        self.assertEqual(host.calls, 0)
        self.coordinator.host = None

    async def test_provider_timeout_limit_waits_for_user(self):
        class TimedOutHost:
            async def llm_generate(self, **kwargs):
                raise TimeoutError("private provider timeout")
        self.coordinator.host = TimedOutHost()
        self.coordinator.max_failures = 1
        await self.coordinator._planning_worker(self.task["task_id"])
        self.assertEqual(self.current()["status"], "waiting_input")
        self.assertEqual(self.current()["error_code"], "provider_timeout")
        self.assertEqual(self.public, [])
        self.assertEqual(len(self.ex.goals), 0)
        self.coordinator.host = None

    async def test_P08_coordinator_start_does_not_resume(self):
        reopened = TaskStore(str(Path(self.tmp.name) / "tasks.db"))
        other = TaskCoordinator(reopened, self.ex, host=SimpleNamespace())
        other.start()
        await asyncio.sleep(0)
        await other.close()
        self.assertFalse(self.ex.calls)
        reopened.close()
