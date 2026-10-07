from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from astrbot.core.provider.provider import Provider

from astrbot_plugin_astrbotex_interaction import main as plugin_module
from astrbot_plugin_astrbotex_interaction.output_router import PublicOutputRouter
from astrbot_plugin_astrbotex_interaction.task_coordinator import TaskCoordinator
from astrbot_plugin_astrbotex_interaction.task_models import TaskAuthority, TaskError
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
from astrbot_plugin_astrbotex_interaction.tests.test_plugin_channels import FakeContext
from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import FakeDecision


def event(text, *, session="session-a", user="admin-a", role="admin"):
    message = plugin_module.AstrBotMessage()
    message.type = plugin_module.MessageType.FRIEND_MESSAGE
    message.session_id = session
    message.sender = plugin_module.MessageMember(user_id=user, nickname="offline admin")
    message.message_id = f"message-{text}"
    message.message_str = text
    message.message = [plugin_module.Plain(text=text)]
    item = plugin_module.AstrMessageEvent(text, message,
        plugin_module.PlatformMetadata(name="offline", description="offline", id="offline-1"), session)
    item.role = role
    item.is_at_or_wake_command = True
    return item


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plugin = plugin_module.AstrBotEXInteractionPlugin(FakeContext())
        self.plugin.task_planning_enabled = True
        self.plugin.task_robot_id = "robot-from-config"
        self.plugin.task_peer_id = ""
        self.plugin.task_store = TaskStore(str(Path(self.tmp.name) / "task.db"))
        self.plugin.text_channel = SimpleNamespace(online_peers=lambda: (b"ex-trusted",))
        self.ex = FakeDecision()
        from astrbot_plugin_astrbotex_interaction.tests.test_chat_tasks import capabilities, provider
        self.plugin.task_provider_id = "offline"
        self.offline_provider = provider()
        self.plugin.context.get_provider_by_id = lambda name: self.offline_provider
        async def request(method, payload, **kwargs):
            if method == "decision.capabilities.get":
                return capabilities(self.ex)
            return await self.ex("robot-from-config", "route", method, payload)
        self.plugin.request_text = request
        self.coordinator = TaskCoordinator(self.plugin.task_store, self.ex,
            router=PublicOutputRouter(self.plugin.task_store, self.send), turn_sync=self.sync_turn)
        self.plugin.task_coordinator = self.coordinator
        self.turns = []
        self.public = []

    async def send(self, payload):
        self.public.append(payload)

    async def sync_turn(self, payload):
        self.turns.append(payload)

    async def asyncTearDown(self):
        await self.coordinator.close()
        self.plugin.task_store.close()
        self.tmp.cleanup()

    def create_event(self):
        return event("ex_task do three bounded steps quietly")

    async def create(self):
        item = self.create_event()
        await self.plugin.ex_task(item)
        return self.plugin.task_store.active_tasks()[0], item

    async def test_real_admin_metadata_and_Host_command_filter(self):
        from astrbot.core.star.star_handler import star_handlers_registry
        from astrbot.core.star.filter.permission import PermissionType, PermissionTypeFilter
        from astrbot.core.star.filter.command import CommandFilter
        for name in ("ex_task", "ex_task_update", "ex_task_review", "ex_task_cancel"):
            md = star_handlers_registry.get_handler_by_full_name(f"{plugin_module.__name__}_{name}")
            self.assertIsNotNone(md)
            permissions = [f for f in md.event_filters if isinstance(f, PermissionTypeFilter)]
            self.assertEqual(permissions, [])
            command = next(f for f in md.event_filters if isinstance(f, CommandFilter))
            request = event(name + " task-id all remaining words")
            self.assertTrue(command.filter(request, {}))
            self.assertEqual(request.get_extra("parsed_params"), {})
        task, item = await self.create()
        self.assertEqual(task["text"], "do three bounded steps quietly")
        self.assertEqual(task["robot_id"], "robot-from-config")
        self.assertEqual(task["session_id"], item.unified_msg_origin)
        self.assertEqual(task["user_id"], item.get_sender_id())
        self.assertEqual(item.get_result().get_plain_text(), "")
        self.assertFalse(item.call_llm)
        self.assertTrue(item.is_stopped())
        route = self.plugin.task_store.route(task["route_ref"])
        self.assertEqual(route["peer_hex"], b"ex-trusted".hex())
        self.assertEqual(route["source_session"], "session-a")
        self.assertEqual(route["origin"], item.unified_msg_origin)
        self.assertEqual(self.public, [])

    async def test_explicit_off_and_nonadmin_admission_ignores_payload_identity(self):
        self.plugin.task_planning_enabled = False
        item = self.create_event()
        await self.plugin.ex_task(item)
        self.assertIn("task_planning_disabled", item.get_result().get_plain_text())
        self.plugin.task_planning_enabled = True
        item = event("ex_task spoof robot session user", role="member")
        item.message_obj.raw_message = {"authorized": True, "user_id": "admin", "robot_id": "spoof"}
        await self.plugin.ex_task(item)
        task = self.plugin.task_store.active_tasks()[0]
        self.assertEqual(task["user_id"], item.get_sender_id())
        self.assertEqual(task["robot_id"], "robot-from-config")
        self.assertEqual(task["session_id"], item.unified_msg_origin)
        self.assertEqual(item.get_result().get_plain_text(), "")

    async def test_other_session_user_cannot_update_review_cancel_or_rebind(self):
        task, item = await self.create()
        for name in ("ex_task_update", "ex_task_review", "ex_task_cancel"):
            suffix = " replacement" if name == "ex_task_update" else ""
            foreign = event(name + " " + task["task_id"] + suffix, session="session-b")
            await getattr(self.plugin, name)(foreign)
            self.assertIn("owner_mismatch", foreign.get_result().get_plain_text())
        auth = TaskAuthority(task["robot_id"], task["session_id"], "different-user", task["route_ref"], True)
        with self.assertRaisesRegex(TaskError, "route_owner_mismatch"):
            self.plugin.admit_task_route(auth, b"ex-trusted")
        original = self.plugin._task_authority(item)
        with self.assertRaisesRegex(TaskError, "route_owner_mismatch"):
            self.plugin.admit_task_route(original, b"different-peer")
        self.assertEqual(self.plugin._task_peer(task["robot_id"], task["route_ref"]), b"ex-trusted")

    async def test_ambiguous_peer_refuses_and_explicit_config_works(self):
        self.plugin.text_channel = SimpleNamespace(online_peers=lambda: (b"one", b"two"), default_peer=b"two")
        item = self.create_event()
        await self.plugin.ex_task(item)
        self.assertIn("ambiguous_task_peer", item.get_result().get_plain_text())
        self.assertEqual(self.plugin.task_store.active_tasks(), [])
        self.plugin.task_peer_id = "one"
        task, item = await self.create()
        self.assertEqual(self.plugin._task_peer(task["robot_id"], task["route_ref"]), b"one")
        self.plugin.task_peer_id = "two"
        with self.assertRaisesRegex(TaskError, "route_owner_mismatch"):
            self.plugin._task_authority(item)

    async def test_framework_EX_event_peer_beats_default_and_ordinary_payload_does_not_bind(self):
        adapter = plugin_module.AstrBotEXPlatformAdapter({}, {}, asyncio.Queue())
        self.plugin._adapter = adapter
        self.plugin.text_channel = SimpleNamespace(online_peers=lambda: (b"a", b"b"), default_peer=b"b")
        ordinary = event("normal text")
        request = plugin_module.AstrBotEXMessageEvent("ex_task bound event", ordinary.message_obj,
            adapter.meta(), "ex-session", adapter, b"a", {})
        request.role = "admin"
        await self.plugin.ex_task(request)
        task = self.plugin.task_store.active_tasks()[0]
        self.assertEqual(self.plugin._task_peer(task["robot_id"], task["route_ref"]), b"a")
        captured = []
        adapter.inject_message = lambda data, peer: captured.append((data, peer))
        result = await self.plugin._handle_message(b"b", {"payload": {"text": "normal", "authorized": True,
            "robot_id": "spoof", "user_id": "spoof", "session_id": "session-b"}}, None)
        self.assertTrue(result["ok"])
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(self.plugin.task_store.active_tasks()), 1)
        self.assertEqual(self.plugin._task_peer(task["robot_id"], task["route_ref"]), b"a")

    async def test_real_update_review_cancel_silent_and_restart_route_is_immutable(self):
        task, _ = await self.create()
        self.plugin.task_store.mutate(task["task_id"], lambda t: t.update(ex_session="ex1", generation=1))
        update = event("ex_task_update " + task["task_id"] + " changed target with spaces")
        await self.plugin.ex_task_update(update)
        self.assertEqual(self.plugin.task_store.get(task["task_id"])["text"], "changed target with spaces")
        self.assertEqual(self.turns[-1]["operation"], "invalidate")
        self.assertEqual(update.get_result().get_plain_text(), "")
        review = event("ex_task_review " + task["task_id"], role="member")
        await self.plugin.ex_task_review(review)
        self.assertEqual(review.get_result().get_plain_text(), "")
        canceled = event("ex_task_cancel " + task["task_id"])
        await self.plugin.ex_task_cancel(canceled)
        final = self.plugin.task_store.get(task["task_id"])
        self.assertEqual(final["status"], "canceled")
        self.assertFalse(final["active"])
        self.assertEqual(canceled.get_result().get_plain_text(), "")
        saved = self.plugin.task_store.route(task["route_ref"])
        reopened = TaskStore(str(Path(self.tmp.name) / "task.db"))
        try:
            self.assertEqual(reopened.route(task["route_ref"]), saved)
            with self.assertRaises(TaskError):
                reopened.bind_route(TaskAuthority(task["robot_id"], task["session_id"], "other", task["route_ref"], True), b"ex-trusted")
        finally:
            reopened.close()

    async def test_owner_commands_disabled_and_provider_unavailable_keep_controls(self):
        from unittest.mock import patch
        task, _ = await self.create()
        from astrbot_plugin_astrbotex_interaction.tests.test_chat_tasks import capabilities
        disabled = capabilities(self.ex)
        disabled["execution"].update(mode="disabled", execution_allowed=False, runtime_state="paused")
        self.plugin.context.get_provider_by_id = lambda name: None
        with patch.object(self.plugin, "_capabilities", return_value=disabled):
            create = event("ex_task blocked new work")
            await self.plugin.ex_task(create)
            self.assertIn("decision_execution_unavailable", create.get_result().get_plain_text())
            update = event("ex_task_update " + task["task_id"] + " owner replacement")
            await self.plugin.ex_task_update(update)
            self.assertEqual(update.get_result().get_plain_text(), "")
            self.assertEqual(self.plugin.task_store.get(task["task_id"])["text"], "owner replacement")
            foreign = event("ex_task_cancel " + task["task_id"], user="foreign-member", role="member")
            await self.plugin.ex_task_cancel(foreign)
            self.assertIn("owner_mismatch", foreign.get_result().get_plain_text())
            review = event("ex_task_review " + task["task_id"])
            await self.plugin.ex_task_review(review)
            self.assertEqual(review.get_result().get_plain_text(), "")
            cancel = event("ex_task_cancel " + task["task_id"])
            await self.plugin.ex_task_cancel(cancel)
            self.assertEqual(cancel.get_result().get_plain_text(), "")
            self.assertFalse(self.plugin.task_store.get(task["task_id"])["active"])
            generation = self.plugin.task_store.get(task["task_id"])["generation"]
            await self.plugin.ex_task_cancel(event("ex_task_cancel " + task["task_id"]))
            self.assertEqual(self.plugin.task_store.get(task["task_id"])["generation"], generation)
        self.assertEqual(self.ex.goals, {})
        self.assertFalse(disabled["execution"]["execution_allowed"])

    async def test_provider_unavailable_denies_new_create_but_owner_controls_remain(self):
        task, _ = await self.create()
        self.plugin.context.get_provider_by_id = lambda name: None
        create = event("ex_task another task")
        await self.plugin.ex_task(create)
        self.assertIn("task_provider_unavailable", create.get_result().get_plain_text())
        cancel = event("ex_task_cancel " + task["task_id"])
        await self.plugin.ex_task_cancel(cancel)
        self.assertEqual(cancel.get_result().get_plain_text(), "")
        self.assertFalse(self.plugin.task_store.get(task["task_id"])["active"])

    async def test_real_public_Host_entry_context_tools_and_silent_finish(self):
        from astrbot.api.star import Context
        from astrbot.core.provider.provider import Provider
        from astrbot.core.provider.entities import LLMResponse
        class OfflineProvider(Provider):
            def get_current_key(self):
                return "offline"
            def set_key(self, key):
                pass
            async def get_models(self):
                return []
            async def text_chat(self, **kwargs):
                self.seen = kwargs
                return LLMResponse(role="assistant", tools_call_name=["finish_planning_turn"],
                                   tools_call_args=[{"outcome": "waiting_input"}], tools_call_ids=["finish"])
        provider = object.__new__(OfflineProvider)
        class Manager:
            async def get_provider_by_id(self, provider_id):
                return provider
        host = object.__new__(Context)
        host.provider_manager = Manager()
        self.coordinator.host = host
        self.coordinator.provider_id = "offline"
        task, item = await self.create()
        for _ in range(100):
            if self.plugin.task_store.get(task["task_id"])["status"] == "waiting_input":
                break
            await asyncio.sleep(0.001)
        self.assertEqual(self.plugin.task_store.get(task["task_id"])["status"], "waiting_input")
        self.assertFalse(provider.seen["stream"])
        self.assertEqual(len(provider.seen["func_tool"].tools), 4)
        self.assertEqual(self.turns[-1]["operation"], "bind")
        self.assertIsNotNone(self.turns[-1]["turn_id"])
        self.assertEqual(self.public, [])
        self.assertEqual(item.get_result().get_plain_text(), "")
