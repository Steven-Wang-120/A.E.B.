from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from astrbot.api.star import Context
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import ToolSet
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.provider import Provider

from astrbot_plugin_astrbotex_interaction import main as plugin_module
from astrbot_plugin_astrbotex_interaction.output_router import PublicOutputRouter
from astrbot_plugin_astrbotex_interaction.task_coordinator import TaskCoordinator
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
from astrbot_plugin_astrbotex_interaction.tests.test_plugin_channels import FakeContext, free_port
from astrbot_plugin_astrbotex_interaction.tests.test_task_admission import event
from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import ACTION, FakeDecision, goal_args
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import steps


def capabilities(ex):
    return {"schema_version": 1, "ex_session": "ex1", "revision": ex.context["revision"],
            "catalog_revision": 1, "control_mode": "decision",
            "execution": {"mode": "execute", "execution_allowed": True, "runtime_state": "running"},
            "actions": [{"owner": "mock", "plugin_generation": 1, "action_id": ACTION,
                         "description": "A bounded mock software step", "schema": {"type": "object", "additionalProperties": False},
                         "operations": ["start"], "resources": []}]}


class OfflineProvider(Provider):
    def get_current_key(self):
        return "offline-not-a-key"

    def set_key(self, key):
        pass

    async def get_models(self):
        return []

    def meta(self):
        return SimpleNamespace(id="session-provider")

    async def text_chat(self, **kwargs):
        self.seen.append(kwargs)
        task = json.loads(kwargs["contexts"][0].content)["task"]
        if task["current_goal"]:
            return LLMResponse(role="assistant", tools_call_name=["finish_planning_turn"],
                tools_call_args=[{"outcome": "waiting_feedback"}], tools_call_ids=["finish"])
        return LLMResponse(role="assistant", tools_call_name=["save_plan", "submit_current_goal", "finish_planning_turn"],
            tools_call_args=[{"steps": steps(), "expected_revision": task["plan_revision"]}, goal_args(),
                             {"outcome": "waiting_feedback"}], tools_call_ids=["plan", "goal", "finish"])


def provider():
    result = object.__new__(OfflineProvider)
    result.seen = []
    return result


class ChatTaskTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "tasks.db")
        self.context = FakeContext()
        self.provider = provider()
        self.umos = []
        async def current(umo):
            self.umos.append(umo)
            return "session-provider"
        self.context.get_current_chat_provider_id = current
        self.context.get_provider_by_id = lambda name: self.provider if name == "session-provider" else None
        self.host = object.__new__(Context)
        class Manager:
            async def get_provider_by_id(inner, name):
                self.provider_ids.append(name)
                return self.provider
        self.provider_ids = []
        self.host.provider_manager = Manager()
        self.context.llm_generate = self.host.llm_generate
        self.plugin = plugin_module.AstrBotEXInteractionPlugin(self.context)
        self.plugin.task_robot_id = ""
        self.plugin.task_peer_id = ""
        self.plugin.task_store = TaskStore(self.path)
        self.plugin.text_channel = SimpleNamespace(online_peers=lambda: (b"trusted",), default_peer=b"wrong")
        self.ex = FakeDecision()
        self.cap = capabilities(self.ex)
        self.calls = []
        self.public = []
        async def request(method, payload, *, peer=None, **kwargs):
            self.calls.append((method, copy.deepcopy(payload), peer))
            if method == "decision.capabilities.get":
                self.assertEqual(payload, {"schema_version": 1})
                value = copy.deepcopy(self.cap)
                value["revision"] = self.ex.context["revision"]
                return value
            if method == "bridge.context.get":
                return {"context_id": "legacy-context", "blocks": [], "affordances": [{"action_id": "legacy.v1"}]}
            if method == "bridge.proposal.submit":
                return {"ok": False, "error": "control_mode_decision"}
            if method in {"interaction.task.turn", "interaction.reply"}:
                if method == "interaction.reply":
                    self.public.append(payload)
                return {"ok": True}
            return await self.ex("r", "route", method, payload)
        self.plugin.request_text = request
        router = PublicOutputRouter(self.plugin.task_store, self.plugin._send_task_message)
        self.plugin.task_coordinator = TaskCoordinator(self.plugin.task_store, self.plugin._request_decision,
            host=self.context, provider_id="wrong-global-provider", router=router, turn_sync=self.plugin._sync_task_turn)
        self.shared = ToolSet(list(self.context.tools))

    async def asyncTearDown(self):
        await self.plugin.task_coordinator.close()
        self.plugin.task_store.close()
        self.tmp.cleanup()

    def item(self, text="please do bounded work", user="member-a", message="message-1"):
        item = event(text, role="member", user=user)
        item.message_obj.message_id = message
        return item

    async def tool(self, item):
        request = ProviderRequest(prompt=item.get_message_str(), func_tool=self.shared)
        await self.plugin.inject_ex_context(item, request)
        return request, request.func_tool.get_tool("manage_astrbotex_task")

    def wrapper(self, item):
        return ContextWrapper(context=AstrAgentContext(context=self.host, event=item))

    async def settle(self):
        workers = list(self.plugin.task_coordinator._workers.values())
        if workers:
            await asyncio.wait_for(asyncio.gather(*workers), 3)

    async def test_nonadmin_ordinary_tool_intent_runs_real_Host_and_current_goal(self):
        item = self.item()
        request, tool = await self.tool(item)
        self.assertIsInstance(tool, plugin_module.ManageAstrBotEXTaskTool)
        self.assertIsNot(request.func_tool, self.shared)
        self.assertIsNot(request.func_tool.tools, self.shared.tools)
        self.assertIsNone(request.func_tool.get_tool("submit_astrbotex_proposal"))
        self.assertIsNotNone(self.shared.get_tool("submit_astrbotex_proposal"))
        self.assertNotIn("manage_astrbotex_task", [t.name for t in self.context.tools])
        self.assertEqual(self.plugin.task_store.active_tasks(), [])
        result = json.loads(await tool.call(self.wrapper(item), operation="create", text="three steps"))
        self.assertTrue(result["ok"])
        await self.settle()
        task = self.plugin.task_store.get(result["task_id"])
        self.assertEqual(task["provider_id"], "session-provider")
        self.assertEqual(task["status"], "executing")
        self.assertIsNotNone(task["current_goal"])
        self.assertEqual(task["user_id"], "member-a")
        self.assertEqual(task["session_id"], item.unified_msg_origin)
        self.assertTrue(task["robot_id"].startswith("ex-peer-"))
        self.assertEqual(len(self.ex.goals), 1)
        self.assertEqual(self.provider_ids, ["session-provider"])
        self.assertFalse(self.provider.seen[0]["stream"])
        self.assertEqual(len(self.provider.seen[0]["func_tool"].tools), 4)
        self.assertEqual(self.public, [])
        self.assertNotIn("bridge.context.get", [c[0] for c in self.calls])

    async def test_idle_chat_never_creates_task_goal_or_public_accepted(self):
        for text in ("hello", "can you help", "go to kitchen maybe later", "cancel?", "ex_task is a command name"):
            request, tool = await self.tool(self.item(text))
            self.assertIsNotNone(tool)
        self.assertEqual(self.plugin.task_store.active_tasks(), [])
        self.assertEqual(self.ex.goals, {})
        self.assertEqual(self.provider.seen, [])
        self.assertEqual(self.public, [])

    async def test_private_skip_has_no_IO_or_tool_mutation(self):
        item = self.item()
        request = ProviderRequest(func_tool=self.shared)
        request.private_planning = True
        await self.plugin.inject_ex_context(item, request)
        self.assertIs(request.func_tool, self.shared)
        self.assertEqual(self.calls, [])
        self.assertEqual(request.extra_user_content_parts, [])

    async def test_legacy_context_retains_shared_tools_and_delayed_proposal_rejected(self):
        self.cap["control_mode"] = "legacy"
        request, tool = await self.tool(self.item())
        self.assertIsNone(tool)
        self.assertIs(request.func_tool, self.shared)
        self.assertIn("legacy-context", request.extra_user_content_parts[0].text)
        self.cap["control_mode"] = "decision"
        stale = self.shared.get_tool("submit_astrbotex_proposal")
        result = json.loads(await stale.call(None, context_id="legacy-context", commands=[]))
        self.assertFalse(result["ok"])
        self.assertEqual(self.ex.goals, {})

    async def test_disabled_missing_ambiguous_provider_or_execution_ready_safe(self):
        variants = ("explicit0", "no_peer", "ambiguous", "provider", "off", "runtime", "gate", "actions", "message")
        for variant in variants:
            with self.subTest(variant=variant):
                original = (self.plugin.task_planning_enabled, self.plugin.text_channel, self.context.get_provider_by_id, copy.deepcopy(self.cap))
                item = self.item()
                if variant == "explicit0":
                    self.plugin.task_planning_enabled = False
                elif variant in {"no_peer", "ambiguous"}:
                    self.plugin.text_channel = SimpleNamespace(online_peers=lambda: () if variant == "no_peer" else (b"a", b"b"))
                elif variant == "provider":
                    self.context.get_provider_by_id = lambda name: None
                elif variant == "off":
                    self.cap["execution"]["mode"] = "disabled"
                elif variant == "runtime":
                    self.cap["execution"]["runtime_state"] = "idle"
                elif variant == "gate":
                    self.cap["execution"]["execution_allowed"] = False
                elif variant == "actions":
                    self.cap["actions"] = []
                else:
                    item.message_obj.message_id = ""
                _, tool = await self.tool(item)
                self.assertIsNone(tool)
                self.assertEqual(self.plugin.task_store.active_tasks(), [])
                self.assertEqual(self.ex.goals, {})
                self.plugin.task_planning_enabled, self.plugin.text_channel, self.context.get_provider_by_id, self.cap = original

    async def test_strict_capabilities_and_identity_arguments_never_authorize(self):
        item = self.item()
        for field in ("user_id", "task_id", "message_id", "robot_id", "route_ref", "ex_session"):
            _, tool = await self.tool(item)
            result = json.loads(await tool.call(self.wrapper(item), operation="create", text="work", **{field: "spoof"}))
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "business_intent_only")
        original = copy.deepcopy(self.cap)
        for change in ("extra", "bool_schema", "bad_action", "bool_generation"):
            self.cap = copy.deepcopy(original)
            if change == "extra":
                self.cap["private_task"] = "secret"
            elif change == "bool_schema":
                self.cap["schema_version"] = True
            elif change == "bool_generation":
                self.cap["actions"][0]["plugin_generation"] = True
            else:
                self.cap["actions"][0]["owner"] = "other"
            _, tool = await self.tool(item)
            self.assertIsNone(tool)
        self.assertEqual(self.plugin.task_store.active_tasks(), [])

    async def test_foreign_or_late_tool_binding_is_rejected(self):
        item = self.item()
        _, tool = await self.tool(item)
        result = json.loads(await tool.call(self.wrapper(self.item(user="member-b")), operation="create", text="steal"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "stale_host_context")
        self.cap["ex_session"] = "ex2"
        result = json.loads(await tool.call(self.wrapper(item), operation="create", text="late"))
        self.assertFalse(result["ok"])
        self.assertEqual(self.plugin.task_store.active_tasks(), [])

    async def test_cross_user_update_cancel_tool_denied_server_owned_selection(self):
        item = self.item()
        _, tool = await self.tool(item)
        result = json.loads(await tool.call(self.wrapper(item), operation="create", text="own work"))
        await self.settle()
        before = self.plugin.task_store.get(result["task_id"])
        for operation in ("update", "cancel"):
            foreign = self.item(user="member-b", message="foreign")
            _, foreign_tool = await self.tool(foreign)
            args = {"operation": operation}
            if operation == "update":
                args["text"] = "steal"
            denied = json.loads(await foreign_tool.call(self.wrapper(foreign), **args))
            self.assertEqual(denied["error"], "owned_task_unavailable")
        self.assertEqual(self.plugin.task_store.get(result["task_id"]), before)

    async def test_same_event_tool_command_and_restart_have_one_admission(self):
        item = self.item("ex_task three steps")
        _, tool = await self.tool(item)
        first = json.loads(await tool.call(self.wrapper(item), operation="create", text="three steps"))
        await self.settle()
        snapshot = self.plugin.task_store.get(first["task_id"])
        await self.plugin.ex_task(item)
        self.assertEqual(item.get_result().get_plain_text(), "")
        self.assertEqual(self.plugin.task_store.get(first["task_id"]), snapshot)
        self.assertEqual(len(self.ex.goals), 1)
        await self.plugin.task_coordinator.close()
        self.plugin.task_store.close()
        self.plugin.task_store = TaskStore(self.path)
        self.plugin._task_routes.clear()
        self.plugin.task_coordinator = TaskCoordinator(self.plugin.task_store, self.plugin._request_decision,
            host=self.context, turn_sync=self.plugin._sync_task_turn)
        _, replay_tool = await self.tool(item)
        replay = json.loads(await replay_tool.call(self.wrapper(item), operation="create", text="three steps"))
        self.assertEqual(replay["task_id"], first["task_id"])
        self.assertEqual(self.plugin.task_store.get(first["task_id"])["status"], "resume_review")
        self.assertEqual(len(self.ex.goals), 1)
        conflict = self.item("ex_task different", message="message-1")
        await self.plugin.ex_task(conflict)
        self.assertIn("duplicate_request_id_conflict", conflict.get_result().get_plain_text())
        second = self.item("ex_task different", message="message-1")
        await self.plugin.ex_task(second)
        self.assertEqual(second.get_result().get_plain_text(), "")

    async def test_unknown_capabilities_hide_robot_tools_without_global_mutation(self):
        from astrbot.core.agent.tool import FunctionTool
        unrelated = FunctionTool(name="ordinary_chat_tool", description="Ordinary chat", parameters={"type": "object"})
        self.shared.add_tool(unrelated)
        snapshot = list(self.shared.tools)
        global_snapshot = list(self.context.tools)
        for peers in ((), (b"one", b"two"), (b"trusted",)):
            self.plugin.text_channel.online_peers = lambda: peers
            with patch.object(self.plugin, "_capabilities", side_effect=TaskError("invalid_capabilities")):
                request, tool = await self.tool(self.item())
            self.assertIsNone(tool)
            self.assertIsNone(request.func_tool.get_tool("submit_astrbotex_proposal"))
            self.assertIs(request.func_tool.get_tool("ordinary_chat_tool"), unrelated)
            self.assertIsNot(request.func_tool, self.shared)
            self.assertEqual(self.shared.tools, snapshot)
            self.assertEqual(self.context.tools, global_snapshot)
        self.assertEqual(self.ex.goals, {})
        self.assertEqual(self.calls, [])

    async def test_capabilities_actual_reply_rejects_any_binary_only_for_control(self):
        from astrbot_plugin_astrbotex_interaction.zmq_transport import ZmqReply
        from unittest.mock import AsyncMock
        self.plugin.text_channel.request = AsyncMock()
        for binary in (b"", b"binary"):
            self.plugin.text_channel.request.return_value = ZmqReply(copy.deepcopy(self.cap), binary)
            with self.assertRaisesRegex(TaskError, "invalid_capabilities"):
                await plugin_module.AstrBotEXInteractionPlugin.request_text(
                    self.plugin, "decision.capabilities.get", {"schema_version": 1}, peer=b"trusted")
            result = await plugin_module.AstrBotEXInteractionPlugin.request_text(
                self.plugin, "unrelated.text.method", {}, peer=b"trusted")
            self.assertEqual(result, self.cap)
        self.plugin.text_channel.request.return_value = ZmqReply(copy.deepcopy(self.cap), None)
        self.assertEqual(await plugin_module.AstrBotEXInteractionPlugin.request_text(
            self.plugin, "decision.capabilities.get", {"schema_version": 1}, peer=b"trusted"), self.cap)

    async def test_projection_binary_peer_current_session_late_response_and_exact_summary(self):
        item = self.item()
        _, tool = await self.tool(item)
        first = json.loads(await tool.call(self.wrapper(item), operation="create", text="three steps"))
        await self.settle()
        payload = {"payload": {"schema_version": 1, "ex_session": "ex1"}}
        result = await self.plugin._handle_task_projection(b"trusted", payload, None)
        self.assertEqual(result["task_id"], first["task_id"])
        self.assertFalse(result["can_cancel"])
        self.assertEqual(set(result), {"schema_version", "ex_session", "robot_id", "task_id", "generation", "turn_id",
            "available", "phase", "title", "current_goal", "completed", "total", "can_cancel", "updated_at", "message"})
        calls = len(self.calls)
        for binary in (b"", b"binary"):
            self.assertEqual((await self.plugin._handle_task_projection(b"trusted", payload, binary))["error"], "projection_binary_rejected")
        self.assertEqual(len(self.calls), calls)
        self.assertFalse((await self.plugin._handle_task_projection(b"foreign", payload, None))["ok"])
        self.cap["ex_session"] = "ex2"
        old = await self.plugin._handle_task_projection(b"trusted", payload, None)
        self.assertEqual(old["error"], "stale_ex_session")
        current = await self.plugin._handle_task_projection(b"trusted", {"payload": {"schema_version": 1, "ex_session": "ex2"}}, None)
        self.assertFalse(current["available"])
        self.assertEqual(current["phase"], "unavailable")
        self.assertIsNone(current["task_id"])


class AutoInitializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_initialization_data_dir_and_public_provider_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = object.__new__(Context)
            prov = provider()
            using = []
            class Manager:
                inst_map = {"session-provider": prov}
                def get_using_provider(self, **kwargs):
                    using.append(kwargs)
                    return prov
            host.provider_manager = Manager()
            host.platform_manager = SimpleNamespace(platform_insts=[])
            host.add_llm_tools = lambda *args: None
            plugin = plugin_module.AstrBotEXInteractionPlugin(host)
            plugin.zmq_bind_host = "127.0.0.1"
            plugin.text_port, plugin.audio_port, plugin.vision_port = free_port(), free_port(), free_port()
            self.assertTrue(plugin.task_planning_enabled)
            with patch.object(plugin_module.star.StarTools, "get_data_dir", return_value=Path(tmp)) as data_dir:
                await plugin.initialize()
                try:
                    self.assertIsNotNone(plugin.task_coordinator)
                    self.assertTrue((Path(tmp) / "tasks.sqlite3").exists())
                    data_dir.assert_called_once_with("astrbot_plugin_astrbotex_interaction")
                    self.assertEqual(await plugin._task_provider("host:session"), "session-provider")
                    self.assertEqual(using[0]["umo"], "host:session")
                    self.assertIn("task.projection.get", plugin.text_channel._handlers)
                finally:
                    await plugin.terminate()

    async def test_explicit_zero_no_database_no_planner(self):
        with patch.dict(os.environ, {"ASTRBOTEX_TASK_PLANNING": "0"}):
            context = FakeContext()
            context.llm_generate = lambda **kwargs: None
            plugin = plugin_module.AstrBotEXInteractionPlugin(context)
        plugin.zmq_bind_host = "127.0.0.1"
        plugin.text_port, plugin.audio_port, plugin.vision_port = free_port(), free_port(), free_port()
        with patch.object(plugin_module.star.StarTools, "get_data_dir") as data_dir:
            await plugin.initialize()
            try:
                self.assertFalse(plugin.task_planning_enabled)
                self.assertIsNone(plugin.task_coordinator)
                data_dir.assert_not_called()
            finally:
                await plugin.terminate()
