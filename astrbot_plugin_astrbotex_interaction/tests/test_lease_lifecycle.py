from __future__ import annotations

import asyncio
from pathlib import Path

from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import CoordinatorFixture
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import authority, steps


class LeaseTests(CoordinatorFixture):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.now = 0.0
        self.coordinator.clock = lambda: self.now
        self.coordinator.heartbeat_interval = 60.0

    async def prepared(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        return turn

    async def complete(self, turn):
        self.coordinator.finish(turn, "waiting_feedback")
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "succeeded"))

    def latch_renew(self):
        started, release = asyncio.Event(), asyncio.Event()
        original = self.coordinator.request

        async def request(*args):
            if args[2] == "decision.goal.renew":
                started.set()
                await release.wait()
                return {"ok": True}
            return await original(*args)

        self.coordinator.request = request
        return started, release

    async def test_second_step_has_fresh_deadline_after_old_deadline(self):
        turn = await self.prepared()
        self.now = 8.0
        await self.complete(turn)
        next_turn = self.turn()
        await self.submit(next_turn, "step-1", "next")
        self.assertEqual(self.coordinator._lease_deadlines[self.task["task_id"]], 18.0)
        self.now = 11.0
        await self.coordinator.heartbeat_once()
        self.assertEqual(self.current()["status"], "executing")
        self.assertEqual(self.coordinator._lease_deadlines[self.task["task_id"]], 21.0)
        renews = [call[3] for call in self.ex.calls if call[2] == "decision.goal.renew"]
        self.assertEqual(renews[-1]["goal_id"], self.current()["current_goal"]["payload"]["goal_id"])

    async def test_completed_goal_cleans_local_deadline(self):
        await self.complete(await self.prepared())
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_rejected_submit_cleans_local_deadline(self):
        original = self.coordinator.request

        async def rejected(*args):
            result = await original(*args)
            if args[2] == "decision.goal.submit":
                return {**result, "ok": False, "phase": "rejected"}
            return result

        self.coordinator.request = rejected
        await self.prepared()
        self.assertIsNone(self.current()["current_goal"])
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_same_request_retry_preserves_first_deadline(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        original = self.coordinator.request

        async def unavailable(*args):
            if args[2] == "decision.goal.submit":
                raise ConnectionError("offline before admission")
            return await original(*args)

        self.coordinator.request = unavailable
        with self.assertRaises(ConnectionError):
            await self.submit(turn)
        self.assertEqual(self.coordinator._lease_deadlines[self.task["task_id"]], 10.0)
        self.now = 4.0
        self.coordinator.request = original
        await self.submit(turn)
        self.assertEqual(self.coordinator._lease_deadlines[self.task["task_id"]], 10.0)
        self.assertEqual(len(self.ex.goals), 1)

    async def test_expired_ambiguous_request_retry_never_resubmits(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        original = self.coordinator.request

        async def unavailable(*args):
            if args[2] == "decision.goal.submit":
                raise ConnectionError("offline before admission")
            return await original(*args)

        self.coordinator.request = unavailable
        with self.assertRaises(ConnectionError):
            await self.submit(turn)
        self.now = 10.0
        self.coordinator.request = original
        with self.assertRaises(TaskError):
            await self.submit(turn)
        self.assertEqual(self.current()["status"], "lease_lost")
        self.assertFalse(self.ex.goals)
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_timeout_internal_retry_cannot_cross_first_deadline(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        original = self.coordinator.request
        self.ex.timeout_once = True

        async def delayed_state(*args):
            result = await original(*args)
            if args[2] == "decision.state.get":
                self.now = 10.0
            return result

        self.coordinator.request = delayed_state
        with self.assertRaises(TaskError):
            await self.submit(turn)
        self.assertEqual(self.current()["status"], "lease_lost")
        submits = [call for call in self.ex.calls if call[2] == "decision.goal.submit"]
        self.assertEqual(len(submits), 1)
        self.assertEqual(len(self.ex.goals), 1)

    async def test_cached_submit_retry_after_expiry_does_not_authorize_goal(self):
        turn = await self.prepared()
        self.now = 10.0
        with self.assertRaises(TaskError):
            await self.submit(turn)
        self.assertEqual(self.current()["status"], "lease_lost")
        self.assertEqual(len(self.ex.goals), 1)

    async def test_late_old_renew_does_not_overwrite_new_goal_deadline(self):
        turn = await self.prepared()
        started, release = self.latch_renew()
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        self.now = 4.0
        await self.complete(turn)
        await self.submit(self.turn(), "step-1", "next")
        new_goal = self.current()["current_goal"]
        self.assertEqual(self.coordinator._lease_deadlines[self.task["task_id"]], 14.0)
        self.now = 5.0
        release.set()
        await heartbeat
        self.assertEqual(self.current()["current_goal"], new_goal)
        self.assertEqual(self.coordinator._lease_deadlines[self.task["task_id"]], 14.0)

    async def test_late_renew_after_cancel_does_not_restore_lease(self):
        await self.prepared()
        started, release = self.latch_renew()
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        await self.coordinator.user_input(self.task["task_id"], authority(), "replacement")
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)
        release.set()
        await heartbeat
        self.assertEqual(self.current()["status"], "canceling")
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "canceled"))
        self.assertIsNone(self.current()["current_goal"])
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_late_renew_after_real_store_recovery_is_discarded(self):
        await self.prepared()
        started, release = self.latch_renew()
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        reopened = TaskStore(str(Path(self.tmp.name) / "tasks.db"))
        try:
            self.assertEqual(reopened.get(self.task["task_id"])["status"], "resume_review")
            release.set()
            await heartbeat
            self.assertEqual(self.current()["status"], "resume_review")
            self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)
        finally:
            reopened.close()

    async def test_late_renew_after_generation_change_is_discarded(self):
        await self.prepared()
        started, release = self.latch_renew()
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        self.store.mutate(self.task["task_id"], lambda t: t.update(generation=t["generation"] + 1))
        release.set()
        await heartbeat
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_late_renew_after_revision_change_is_discarded(self):
        await self.prepared()
        started, release = self.latch_renew()
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        self.store.mutate(self.task["task_id"], lambda t: t["current_goal"].update(revision=2))
        release.set()
        await heartbeat
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_late_renew_at_deadline_revokes_instead_of_extending(self):
        turn = await self.prepared()
        started, release = self.latch_renew()
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        self.now = 10.0
        release.set()
        await heartbeat
        self.assertEqual(self.current()["status"], "lease_lost")
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)
        with self.assertRaises(TaskError):
            await self.submit(turn)
        self.assertEqual(len(self.ex.goals), 1)

    async def test_ambiguous_submit_without_revision_expires_on_heartbeat(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        original = self.coordinator.request

        async def unavailable(*args):
            if args[2] == "decision.goal.submit":
                raise ConnectionError("offline before admission")
            return await original(*args)

        self.coordinator.request = unavailable
        with self.assertRaises(ConnectionError):
            await self.submit(turn)
        self.assertIsNone(self.current()["current_goal"]["revision"])
        self.now = 10.0
        await self.coordinator.heartbeat_once()
        self.assertEqual(self.current()["status"], "lease_lost")
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)
        self.coordinator.request = original
        with self.assertRaises(TaskError):
            await self.submit(turn)
        self.assertFalse(self.ex.goals)

    async def test_submit_admission_after_deadline_is_recorded_not_revived(self):
        original = self.coordinator.request

        async def delayed_submit(*args):
            result = await original(*args)
            if args[2] == "decision.goal.submit":
                self.now = 10.0
            return result

        self.coordinator.request = delayed_submit
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        with self.assertRaises(TaskError):
            await self.submit(turn)
        self.assertEqual(self.current()["status"], "lease_lost")
        self.assertIsNotNone(self.current()["current_goal"]["revision"])
        request_id = self.current()["current_goal"]["payload"]["request_id"]
        self.assertTrue(self.store.request(request_id)["result"]["ok"])
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_late_renew_after_session_change_is_discarded(self):
        await self.prepared()
        started, release = self.latch_renew()
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        self.store.mutate(self.task["task_id"], lambda t: t.update(ex_session="ex2"))
        release.set()
        await heartbeat
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_late_renew_after_terminal_fact_cannot_resurrect_lease(self):
        turn = await self.prepared()
        self.coordinator.finish(turn, "waiting_feedback")
        started, release = self.latch_renew()
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        original = self.coordinator.request

        async def failed_sync(*args):
            if args[2] == "decision.events.get":
                raise TimeoutError("terminal fact durable, state temporarily unavailable")
            return await original(*args)

        self.coordinator.request = failed_sync
        await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(self.current(), "succeeded"))
        self.assertIsNotNone(self.current()["current_goal"])
        self.assertEqual(self.current()["active_step"], 0)
        release.set()
        await heartbeat
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_multiple_robots_renew_current_goal_not_snapshot_before_await(self):
        await self.prepared()
        second = self.store.create(authority("s2", "r2"), "two steps", "create-2")
        second_id = second["task_id"]
        self.store.mutate(second_id, lambda t: t.update(ex_session="ex1"))
        turn = self.store.begin_turn(second_id)
        self.store.save_plan(turn, steps(), 0)
        from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import goal_args
        await self.coordinator.submit(turn, goal_args(), self.ex.context, "second")
        self.coordinator.finish(turn, "waiting_feedback")
        started, release = asyncio.Event(), asyncio.Event()
        original = self.coordinator.request

        async def latched_first_robot(*args):
            if args[0] == "r1" and args[2] == "decision.goal.renew":
                started.set()
                await release.wait()
                return {"ok": True}
            return await original(*args)

        self.coordinator.request = latched_first_robot
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        await self.coordinator.feedback("r2", "route-s2", self.ex.feedback(self.store.get(second_id), "succeeded"))
        next_turn = self.store.begin_turn(second_id)
        self.now = 4.0
        await self.coordinator.submit(next_turn, goal_args("step-1"), self.ex.context, "next")
        new_goal_id = self.store.get(second_id)["current_goal"]["payload"]["goal_id"]
        release.set()
        await heartbeat
        renews = [call[3] for call in self.ex.calls if call[0] == "r2" and call[2] == "decision.goal.renew"]
        self.assertEqual([payload["goal_id"] for payload in renews], [new_goal_id])
        self.assertEqual(self.store.get(second_id)["status"], "executing")
        self.assertEqual(self.coordinator._lease_deadlines[second_id], 14.0)

    async def test_late_renew_after_other_heartbeat_revokes_is_discarded(self):
        await self.prepared()
        started, release = self.latch_renew()
        heartbeat = asyncio.create_task(self.coordinator.heartbeat_once())
        await started.wait()
        self.now = 10.0
        await self.coordinator.heartbeat_once()
        release.set()
        await heartbeat
        self.assertEqual(self.current()["status"], "lease_lost")
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)
