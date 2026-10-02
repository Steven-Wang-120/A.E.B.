from __future__ import annotations

from astrbot_plugin_astrbotex_interaction.task_contracts import Feedback
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import CoordinatorFixture
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import steps


class FeedbackTests(CoordinatorFixture):
    async def prepared(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        self.coordinator.finish(turn, "waiting_feedback")

    async def test_P03_out_of_order_gap_sync_and_duplicate(self):
        await self.prepared()
        first = self.ex.feedback(self.current(), "running")
        last = self.ex.feedback(self.current(), "succeeded")
        ack = await self.coordinator.feedback("r1", "route-s1", last)
        self.assertEqual(ack["acked_event_seq"], 2)
        self.assertEqual(self.current()["active_step"], 1)
        await self.coordinator.feedback("r1", "route-s1", first)
        await self.coordinator.feedback("r1", "route-s1", last)
        self.assertEqual(self.current()["active_step"], 1)
        self.assertEqual(self.store.cursor("r1", "ex1"), 2)

    async def test_durable_receipt_precedes_ack_even_sync_unavailable(self):
        await self.prepared()
        self.ex.feedback(self.current(), "running")
        last = self.ex.feedback(self.current(), "succeeded")
        original = self.coordinator.request
        async def failed(*args):
            raise TimeoutError("offline")
        self.coordinator.request = failed
        ack = await self.coordinator.feedback("r1", "route-s1", last)
        row = self.store.db.execute("SELECT payload FROM receipts WHERE seq=2").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(ack["acked_event_seq"], 2)
        self.assertEqual(self.current()["active_step"], 0)
        self.coordinator.request = original
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["active_step"], 1)

    async def test_P03_trimmed_history_state_full_restore(self):
        await self.prepared()
        self.ex.feedback(self.current(), "running")
        self.ex.feedback(self.current(), "succeeded")
        self.ex.resync = True
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["active_step"], 1)
        self.assertEqual(self.store.cursor("r1", "ex1"), 2)

    async def test_gap_without_state_evidence_never_advances(self):
        await self.prepared()
        self.ex.feedback(self.current(), "succeeded")
        self.ex.resync = True
        self.ex.fact = None
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["active_step"], 0)
        self.assertIsNotNone(self.current()["current_goal"])
        self.assertEqual(self.store.cursor("r1", "ex1"), 0)

    async def test_paginated_horizon_recovers_all_facts_before_advancing_cursor(self):
        await self.prepared()
        self.ex.feedback(self.current(), "running")
        self.ex.feedback(self.current(), "running")
        self.ex.feedback(self.current(), "succeeded")
        original = self.coordinator.request
        async def paginated(robot, route, method, payload):
            result = await original(robot, route, method, payload)
            if method == "decision.events.get":
                result["events"] = result["events"][:1]
                if result["events"]:
                    result["latest_event_seq"] = result["events"][-1]["event_seq"]
            return result
        self.coordinator.request = paginated
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.store.cursor("r1", "ex1"), 3)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
        self.assertEqual(self.current()["active_step"], 1)
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["active_step"], 1)

    async def test_crop_recovers_older_goal_summary_not_latest_foreign_feedback(self):
        await self.prepared()
        running = self.ex.feedback(self.current(), "running")
        terminal = self.ex.feedback(self.current(), "succeeded")
        foreign = {**terminal, "task_id": "foreign-task", "goal_id": "foreign-goal", "event_seq": 3}
        self.ex.events.append({**self.ex.events[-1], "event_seq": 3, "task_id": "foreign-task", "goal_id": "foreign-goal"})
        self.ex.fact = foreign
        self.ex.resync = True
        original = self.coordinator.request
        async def summaries(robot, route, method, payload):
            result = await original(robot, route, method, payload)
            if method == "decision.state.get":
                result["execution"]["goal_summaries"] = [foreign, terminal, running, terminal]
            return result
        self.coordinator.request = summaries
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["active_step"], 1)
        self.assertEqual(self.current()["completion_evidence_seq"], 2)
        self.assertEqual(self.store.cursor("r1", "ex1"), 3)
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["active_step"], 1)

    async def test_summary_identity_filtered_before_event_seq_dedup(self):
        await self.prepared()
        terminal = self.ex.feedback(self.current(), "succeeded")
        variants = [{**terminal, key: value} for key, value in (
            ("task_id", "foreign-task"), ("goal_id", "foreign-goal"),
            ("ex_session", "ex2"), ("goal_revision", terminal["goal_revision"] + 1))]
        self.ex.resync = True
        original = self.coordinator.request
        async def summaries(robot, route, method, payload):
            result = await original(robot, route, method, payload)
            if method == "decision.state.get":
                result["execution"] = {"goal_summaries": variants + [terminal, terminal]}
            return result
        self.coordinator.request = summaries
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["active_step"], 1)
        self.assertEqual(self.current()["completion_evidence_seq"], 1)
        self.assertIsNone(self.current()["current_goal"])
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["active_step"], 1)

    async def test_absent_active_with_only_foreign_summary_is_not_success(self):
        await self.prepared()
        terminal = self.ex.feedback(self.current(), "succeeded")
        self.ex.resync = True
        original = self.coordinator.request
        async def foreign_only(robot, route, method, payload):
            result = await original(robot, route, method, payload)
            if method == "decision.state.get":
                result["execution"] = {"goal_summaries": [
                    {**terminal, "goal_id": "foreign-goal"},
                    {**terminal, "goal_revision": terminal["goal_revision"] + 1}]}
            return result
        self.coordinator.request = foreign_only
        before = self.current()
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current(), before)
        self.assertEqual(self.store.cursor("r1", "ex1"), 0)

    async def test_old_progress_and_duplicate_terminal_cannot_replace_fact(self):
        await self.prepared()
        first = self.ex.feedback(self.current(), "running")
        terminal = self.ex.feedback(self.current(), "succeeded")
        task = self.current()
        apply = self.coordinator._apply_fact
        self.assertTrue(apply(task, Feedback.parse(terminal)))
        self.assertFalse(apply(task, Feedback.parse(first)))
        self.assertFalse(apply(task, Feedback.parse({**terminal, "event_seq": 3})))
        self.assertEqual(task["current_goal"]["last_feedback"], terminal)
        self.assertEqual(task["active_step"], 0)
        with_wrong_session = {**terminal, "ex_session": "ex2", "event_seq": 4}
        self.assertFalse(apply(task, Feedback.parse(with_wrong_session)))
        self.assertEqual(task["current_goal"]["last_feedback"], terminal)

    async def test_new_session_page_closes_planning_and_lease_without_rebinding(self):
        await self.prepared()
        generation = self.current()["generation"]
        original = self.coordinator.request
        async def restarted(robot, route, method, payload):
            result = await original(robot, route, method, payload)
            if method == "decision.events.get":
                result["ex_session"] = "ex2"
            return result
        self.coordinator.request = restarted
        with self.assertRaisesRegex(TaskError, "stale_ex_session"):
            await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["status"], "resume_review")
        self.assertEqual(self.current()["ex_session"], "ex1")
        self.assertGreater(self.current()["generation"], generation)
        self.assertFalse(self.current()["needs_planning"])
        self.assertNotIn(self.task["task_id"], self.coordinator._lease_deadlines)

    async def test_previous_session_manual_review_never_wakes_planning(self):
        await self.prepared()
        self.ex.feedback(self.current(), "succeeded")
        original = self.coordinator.request
        async def manual_review(robot, route, method, payload):
            result = await original(robot, route, method, payload)
            if method == "decision.state.get":
                result["execution"]["previous_session_manual_review"] = True
            return result
        self.coordinator.request = manual_review
        await self.coordinator.sync(self.task["task_id"])
        self.assertEqual(self.current()["status"], "resume_review")
        self.assertFalse(self.current()["needs_planning"])
        with self.assertRaisesRegex(TaskError, "review_required"):
            self.turn()

    async def test_conflict_and_foreign_route_session(self):
        await self.prepared()
        fact = self.ex.feedback(self.current(), "running")
        self.store.receive("r1", Feedback.parse(fact))
        with self.assertRaisesRegex(TaskError, "event_seq_conflict"):
            self.store.receive("r1", Feedback.parse({**fact, "reason_code": "different"}))
        with self.assertRaisesRegex(TaskError, "owner_mismatch"):
            await self.coordinator.feedback("r1", "other-route", fact)
        with self.assertRaisesRegex(TaskError, "stale_ex_session"):
            await self.coordinator.feedback("r1", "route-s1", {**fact, "ex_session": "foreign"})
