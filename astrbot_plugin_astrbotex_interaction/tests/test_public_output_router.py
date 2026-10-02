from __future__ import annotations

from astrbot_plugin_astrbotex_interaction.output_router import PublicOutputRouter
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import CoordinatorFixture
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import authority, steps


class PublicRouterTests(CoordinatorFixture):
    async def test_O02_message_identity_and_dedup(self):
        turn = self.turn()
        await self.router.emit(turn, "准备尝试", message_id="notice")
        self.assertFalse(await self.router.emit(turn, "准备尝试", message_id="notice"))
        self.assertEqual(len(self.public), 1)
        message = self.public[0]
        self.assertEqual(message["route_ref"], "route-s1")
        self.assertEqual(message["visibility"], "user")
        self.assertEqual(message["task_id"], self.task["task_id"])
        with self.assertRaisesRegex(TaskError, "duplicate_message_id_conflict"):
            await self.router.emit(turn, "changed", message_id="notice")

    async def test_O06_TTS_failure_does_not_retry_goal(self):
        turn = self.turn()
        self.store.save_plan(turn, steps(), 0)
        await self.submit(turn)
        async def failed(payload):
            raise TimeoutError("tts secret error")
        router = PublicOutputRouter(self.store, failed, delivery="tts")
        self.assertFalse(await router.emit(turn, "尝试中", message_id="tts-1"))
        self.assertFalse(await router.emit(turn, "尝试中", message_id="tts-1"))
        self.assertEqual(len(self.ex.goals), 1)
        self.assertEqual(self.public, [])

    async def test_O08_old_generation_and_unverified_completed(self):
        turn = self.turn()
        with self.assertRaisesRegex(TaskError, "completion_not_verified"):
            await self.router.emit(turn, "已完成", claim="completed", evidence_seq=1)
        self.store.invalidate(self.task["task_id"], authority())
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            await self.router.emit(turn, "旧文字音频", message_id="old")
        self.assertFalse(self.public)

    async def test_O07_two_session_routes(self):
        turn = self.turn()
        other = self.store.create(authority("s2", "r2"), "second task", "create-2")
        other_turn = self.store.begin_turn(other["task_id"])
        await self.router.emit(other_turn, "second")
        await self.router.emit(turn, "first")
        self.assertEqual([m["route_ref"] for m in self.public], ["route-s2", "route-s1"])

    async def test_automatic_output_gate_preserves_public_chat(self):
        self.assertTrue(PublicOutputRouter.permits_automatic_output("public_chat"))
        self.assertTrue(PublicOutputRouter.permits_automatic_output(None))
        self.assertFalse(PublicOutputRouter.permits_automatic_output("private_planning"))
        self.assertFalse(PublicOutputRouter.permits_automatic_output(None, task_id="task"))
