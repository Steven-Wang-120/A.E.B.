from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from astrbot_plugin_astrbotex_interaction.host_tasks import HostTaskContext, HostTaskOperations, task_projection
from astrbot_plugin_astrbotex_interaction.task_coordinator import TaskCoordinator
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import authority, steps


class HostFoundationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "tasks.db")
        self.store = TaskStore(self.path)
        self.calls = []
        async def offline(*args):
            self.calls.append(args)
            raise TimeoutError("offline")
        self.coordinator = TaskCoordinator(self.store, offline)
        self.ops = HostTaskOperations(self.coordinator)
        self.context = HostTaskContext(authority(), b"trusted-ex", "message-1", "ex1", 4, "private-provider")
        self.store.bind_route(authority(), self.context.peer)

    async def asyncTearDown(self):
        await self.coordinator.close()
        self.store.close()
        self.tmp.cleanup()

    async def create(self):
        return await self.ops.apply(self.context, {"operation": "create", "text": "three bounded steps"})

    def context_for(self, message):
        return replace(self.context, message_id=message)

    async def test_create_replay_conflict_restart_and_no_model_authority(self):
        first = await self.create()
        task = self.store.get(first["task_id"])
        self.assertEqual(task["goal_revision"], 4)
        self.assertEqual(task["ex_session"], "ex1")
        self.assertEqual(task["provider_id"], "private-provider")
        self.assertEqual((await self.create())["task_id"], first["task_id"])
        self.assertEqual(self.store.get(first["task_id"]), task)
        with self.assertRaisesRegex(TaskError, "duplicate_request_id_conflict"):
            await self.ops.apply(self.context, {"operation": "create", "text": "different"})
        for field in ("task_id", "user_id", "session_id", "robot_id", "authorized", "message_id", "route_ref"):
            with self.assertRaisesRegex(TaskError, "business_intent_only"):
                await self.ops.apply(self.context_for("spoof"), {"operation": "create", "text": "task", field: "spoof"})
        reopened = TaskStore(self.path)
        try:
            replay = HostTaskOperations(TaskCoordinator(reopened, self.coordinator.request))
            self.assertEqual((await replay.apply(self.context, {"operation": "create", "text": task["text"]}))["task_id"], task["task_id"])
            self.assertEqual(reopened.get(task["task_id"])["status"], "resume_review")
        finally:
            reopened.close()
        self.assertEqual(self.calls, [])

    async def test_update_and_cancel_owned_active_replay_without_new_mutation(self):
        first = await self.create()
        updated = await self.ops.apply(self.context_for("update"), {"operation": "update", "text": "new target"})
        snapshot = self.store.get(first["task_id"])
        self.assertEqual(snapshot["text"], "new target")
        self.assertEqual((await self.ops.apply(self.context_for("update"), {"operation": "update", "text": "new target"})), updated)
        self.assertEqual(self.store.get(first["task_id"]), snapshot)
        with self.assertRaisesRegex(TaskError, "duplicate_request_id_conflict"):
            await self.ops.apply(self.context_for("update"), {"operation": "update", "text": "altered"})
        with self.assertRaisesRegex(TaskError, "duplicate_request_id_conflict"):
            await self.ops.apply(self.context_for("update"), {"operation": "cancel"})
        canceled = await self.ops.apply(self.context_for("cancel"), {"operation": "cancel"})
        self.assertEqual(canceled["status"], "canceled")
        self.assertFalse(self.store.get(first["task_id"])["active"])
        self.assertEqual(await self.ops.apply(self.context_for("cancel"), {"operation": "cancel"}), canceled)
        # A retry must target its historical receipt, not cancel a newer active task.
        new = await self.ops.apply(self.context_for("new"), {"operation": "create", "text": "new task"})
        await self.ops.apply(self.context_for("cancel"), {"operation": "cancel"})
        self.assertTrue(self.store.get(new["task_id"])["active"])
        self.assertTrue(self.store.claim_host_reply(canceled["message_key"]))
        self.assertFalse(self.store.claim_host_reply(canceled["message_key"]))
        other = TaskStore(self.path, recover=False)
        try:
            self.assertFalse(other.claim_host_reply(canceled["message_key"]))
        finally:
            other.close()

    async def test_conflicting_operation_and_trusted_peer_owner_session_required(self):
        await self.create()
        with self.assertRaisesRegex(TaskError, "duplicate_request_id_conflict"):
            await self.ops.apply(self.context, {"operation": "cancel"})
        with self.assertRaisesRegex(TaskError, "route_owner_mismatch"):
            await self.ops.apply(replace(self.context_for("x"), peer=b"foreign"), {"operation": "cancel"})
        foreign = authority("s2")
        self.store.bind_route(foreign, self.context.peer)
        with self.assertRaisesRegex(TaskError, "owned_task_unavailable"):
            await self.ops.apply(replace(self.context_for("x"), authority=foreign), {"operation": "cancel"})
        with self.assertRaisesRegex(TaskError, "stale_ex_session"):
            await self.ops.apply(replace(self.context_for("x"), ex_session="ex2"), {"operation": "cancel"})
        with self.assertRaisesRegex(TaskError, "unauthorized"):
            await self.ops.apply(replace(self.context, authority=replace(authority(), authorized=False)),
                                 {"operation": "create", "text": "task"})
        self.assertEqual(self.calls, [])

    async def test_claimed_operation_does_not_reexecute_after_failure_or_restart(self):
        first = await self.create()
        context = self.context_for("update")
        async def failed(*args):
            raise TimeoutError("invalidation unavailable")
        self.coordinator.user_input = failed
        with self.assertRaises(TimeoutError):
            await self.ops.apply(context, {"operation": "update", "text": "replacement"})
        with self.assertRaisesRegex(TaskError, "message_admission_unresolved"):
            await self.ops.apply(context, {"operation": "update", "text": "replacement"})
        other = TaskStore(self.path, recover=False)
        try:
            ops = HostTaskOperations(TaskCoordinator(other, self.coordinator.request))
            with self.assertRaisesRegex(TaskError, "message_admission_unresolved"):
                await ops.apply(context, {"operation": "update", "text": "replacement"})
            self.assertEqual(other.host_message(context.message_key)["task_id"], first["task_id"])
        finally:
            other.close()

    async def test_concurrent_same_message_update_reserves_before_IO(self):
        await self.create()
        started, release = asyncio.Event(), asyncio.Event()
        original = self.coordinator.user_input
        calls = []
        async def held(*args):
            calls.append(args)
            started.set()
            await asyncio.wait_for(release.wait(), 2)
            await original(*args)
        self.coordinator.user_input = held
        context = self.context_for("update")
        intent = {"operation": "update", "text": "replacement"}
        pending = asyncio.create_task(self.ops.apply(context, intent))
        try:
            await asyncio.wait_for(started.wait(), 2)
            with self.assertRaisesRegex(TaskError, "message_admission_unresolved"):
                await self.ops.apply(context, intent)
        finally:
            release.set()
            await asyncio.wait_for(pending, 2)
        self.assertEqual(len(calls), 1)

    async def test_projection_exact_scope_keys_bounded_text_and_no_private_fields(self):
        first = await self.create()
        turn = self.store.begin_turn(first["task_id"])
        self.store.save_plan(turn, steps(), 0)
        def state(task):
            task["steps"][0]["status"] = "completed"
            task["active_step"] = 1
            task["text"] = " title\n" * 1000
            task["current_goal"] = {"payload": {"goal_text_en": "Bounded Goal\n" * 1000,
                "parameters": {"secret": "private"}}, "revision": 5}
            task["status"] = "executing"
        self.store.mutate(first["task_id"], state)
        result = self.project()
        self.assertEqual(set(result), {"schema_version", "ex_session", "robot_id", "task_id", "generation", "turn_id",
            "available", "phase", "title", "current_goal", "completed", "total", "can_cancel", "updated_at", "message"})
        self.assertEqual((result["completed"], result["total"]), (1, 3))
        self.assertTrue(result["available"])
        self.assertFalse(result["can_cancel"])
        self.assertEqual(result["phase"], "executing")
        self.assertLessEqual(len(result["title"]), 160)
        self.assertLessEqual(len(result["current_goal"]), 160)
        self.assertNotIn("\n", result["title"])
        self.assertNotIn("private", str(result))
        self.assertEqual(result["updated_at"], self.store.get(first["task_id"])["updated_at"])
        before = self.store.get(first["task_id"])
        self.project()
        self.store.mutate(first["task_id"], lambda t: None)
        self.assertEqual(self.store.get(first["task_id"]), before)

    def project(self, request=None, peer=None):
        return task_projection(self.store, peer or self.context.peer,
            request or {"schema_version": 1, "ex_session": "ex1"}, ex_session="ex1", robot_id="r1")

    async def test_projection_previous_session_unbound_or_foreign_route_not_idle(self):
        first = await self.create()
        for session in ("old", None):
            self.store.mutate(first["task_id"], lambda t: t.update(ex_session=session, status="resume_review"))
            result = self.project()
            self.assertFalse(result["available"])
            self.assertEqual(result["phase"], "unavailable")
            self.assertIsNone(result["task_id"])
        self.store.mutate(first["task_id"], lambda t: t.update(ex_session="ex1", route_ref="foreign"))
        self.assertFalse(self.project()["available"])
        self.store.mutate(first["task_id"], lambda t: t.update(active=False))
        idle = self.project()
        self.assertTrue(idle["available"])
        self.assertEqual(idle["phase"], "idle")
        self.assertIsNone(idle["task_id"])

    async def test_projection_peer_session_extra_keys_and_ambiguity_fail_closed(self):
        with self.assertRaisesRegex(TaskError, "projection_peer_mismatch"):
            self.project(peer=b"foreign")
        with self.assertRaisesRegex(TaskError, "stale_ex_session"):
            self.project({"schema_version": 1, "ex_session": "wrong"})
        for request in ({"schema_version": True, "ex_session": "ex1"},
                        {"schema_version": 1, "ex_session": "ex1", "task_id": "selected"}):
            with self.assertRaisesRegex(TaskError, "invalid_projection_request"):
                self.project(request)
        self.store.bind_route(authority("s2", "other-robot"), self.context.peer)
        self.assertFalse(self.project()["available"])
        self.assertIsNone(self.project()["task_id"])

    async def test_projection_multiple_active_scope_returns_unavailable(self):
        first = await self.create()
        task = self.store.get(first["task_id"])
        with patch.object(self.store, "active_tasks", return_value=[task, task]):
            self.assertFalse(self.project()["available"])
