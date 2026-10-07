from __future__ import annotations

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.message.message_event_result import MessageEventResult, ResultContentType
from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.star.star_handler import EventType, star_handlers_registry

from astrbot_plugin_astrbotex_interaction import main as plugin_module
from astrbot_plugin_astrbotex_interaction.task_coordinator import TaskCoordinator
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
from astrbot_plugin_astrbotex_interaction.tests import test_chat_tasks as chat
from astrbot_plugin_astrbotex_interaction.tests.test_task_admission import event


class HostChatRoutingTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = chat.ChatTaskTests.asyncSetUp
    asyncTearDown = chat.ChatTaskTests.asyncTearDown
    item = chat.ChatTaskTests.item
    tool = chat.ChatTaskTests.tool
    wrapper = chat.ChatTaskTests.wrapper
    settle = chat.ChatTaskTests.settle

    async def admitted(self):
        item = self.item()
        _, tool = await self.tool(item)
        result = json.loads(await tool.call(self.wrapper(item), operation="create", text="bounded work"))
        self.assertTrue(result["ok"])
        await self.settle()
        return item, self.plugin.task_store.get(result["task_id"])

    async def stages(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["provider_tts_settings"]["enable"] = False
        config["t2i"] = False
        config["platform_settings"]["segmented_reply"]["enable"] = False
        config["platform_settings"]["reply_prefix"] = ""
        config["content_safety"]["also_use_in_response"] = False
        context = SimpleNamespace(astrbot_config=config, plugin_manager=SimpleNamespace(
            context=SimpleNamespace(get_using_tts_provider=lambda origin: None)))
        decorate, respond = ResultDecorateStage(), RespondStage()
        await decorate.initialize(context)
        await respond.initialize(context)
        return decorate, respond

    def registered(self, name):
        metadata = star_handlers_registry.get_handler_by_full_name(f"{plugin_module.__name__}_{name}")
        self.assertIsNotNone(metadata)
        return SimpleNamespace(handler=getattr(self.plugin, name), handler_name=metadata.handler_name,
                               handler_module_path=metadata.handler_module_path,
                               handler_full_name=metadata.handler_full_name, event_filters=metadata.event_filters)

    async def deliver(self, item, text, *, decorate=None, respond=None):
        if decorate is None:
            decorate, respond = await self.stages()
        item.set_result(MessageEventResult().message(text))
        handler = self.registered("guard_host_task_reply")
        def handlers(kind, **kwargs):
            return [handler] if kind == EventType.OnDecoratingResultEvent else []
        with patch.object(star_handlers_registry, "get_handlers_by_event_type", side_effect=handlers), patch(
                "astrbot.core.pipeline.result_decorate.stage.star_map",
                {handler.handler_module_path: SimpleNamespace(name="offline-interaction")}):
            async for _ in decorate.process(item):
                pass
            await respond.process(item)

    async def public_pipeline(self, item, *, early=True):
        from astrbot.core.message.components import Json
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
        from astrbot.core.pipeline.process_stage.stage import ProcessStage
        from astrbot.core.pipeline.process_stage.method.agent_request import AgentRequestSubStage
        from astrbot.core.pipeline.process_stage.method.star_request import StarRequestSubStage
        from astrbot.core.pipeline.scheduler import PipelineScheduler
        from astrbot.core.pipeline.waking_check.stage import WakingCheckStage

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["wake_prefix"] = []
        config["provider_settings"]["enable"] = True
        context = SimpleNamespace(astrbot_config=config, plugin_manager=SimpleNamespace(context=self.context))
        stage = internal.InternalAgentSubStage()
        stage.streaming_response = True
        stage.main_agent_cfg = internal.MainAgentBuildConfig(tool_call_timeout=1)
        stage.ctx = context
        stage.unsupported_streaming_strategy = "realtime_segmenting"
        stage.max_step = 3
        stage.show_tool_use = True  # Real run_agent direct tool-status path, not a fake send.
        stage.show_tool_call_result = stage.show_reasoning = stage.buffer_intermediate_messages = False
        offline_provider = SimpleNamespace(provider_config={"id": "offline", "api_base": ""},
            get_model=lambda: "offline", meta=lambda: SimpleNamespace(type="offline"))
        request = ProviderRequest(func_tool=self.shared)
        tool_calls = []
        class OfflineRunner:
            streaming = False
            finished = False
            step_count = 0
            req = request
            provider = offline_provider
            stats = SimpleNamespace(to_dict=lambda: {})
            run_context = SimpleNamespace(context=SimpleNamespace(event=item), messages=[])
            def done(inner):
                return inner.finished
            def was_aborted(inner):
                return False
            def get_final_llm_resp(inner):
                return None
            async def step(inner):
                inner.step_count += 1
                yield SimpleNamespace(type="tool_call", data={"chain": plugin_module.MessageChain([
                    Json(data={"id": "task-call", "name": "manage_astrbotex_task"})])})
                tool = request.func_tool.get_tool("manage_astrbotex_task")
                if tool:
                    tool_calls.append(json.loads(await tool.call(self.wrapper(item), operation="create", text="bounded work")))
                yield SimpleNamespace(type="llm_result", data={"chain": plugin_module.MessageChain().message("normal final reply")})
                inner.finished = True
        runner = OfflineRunner()
        build = AsyncMock(return_value=internal.MainAgentBuildResult(runner, request, offline_provider))
        agent = AgentRequestSubStage()
        agent.ctx = context
        agent.prov_wake_prefix = ""
        agent.agent_sub_stage = stage
        process = ProcessStage()
        process.ctx = context
        process.agent_sub_stage = agent
        process.star_request_sub_stage = StarRequestSubStage()
        await process.star_request_sub_stage.initialize(context)
        waking = WakingCheckStage()
        await waking.initialize(context)
        decorate, respond = await self.stages()
        scheduler = object.__new__(PipelineScheduler)
        scheduler.stages = [waking, process, decorate, respond]
        names = ("prepare_host_task_output", "inject_ex_context", "guard_host_task_reply")
        registered = {name: self.registered(name) for name in names}
        def handlers(kind, **kwargs):
            if kind == EventType.AdapterMessageEvent:
                return [registered["prepare_host_task_output"]] if early else []
            if kind == EventType.OnLLMRequestEvent:
                return [registered["inject_ex_context"]]
            if kind == EventType.OnDecoratingResultEvent:
                return [registered["guard_host_task_reply"]]
            return []
        host_plugins = {registered[names[0]].handler_module_path: SimpleNamespace(name="offline-interaction")}
        with patch.object(internal, "build_main_agent", build), patch.object(internal, "try_capture_follow_up", return_value=None), patch.object(
                internal, "register_active_runner"), patch.object(internal, "unregister_active_runner"), patch.object(
                internal, "_record_internal_agent_stats", new=AsyncMock()), patch.object(internal.Metric, "upload", new=AsyncMock()), patch(
                "astrbot.core.pipeline.process_stage.method.agent_request.SessionServiceManager.should_process_llm_request", new=AsyncMock(return_value=True)), patch(
                "astrbot.core.pipeline.waking_check.stage.SessionPluginManager.filter_handlers_by_session", new=AsyncMock(side_effect=lambda e, h: h)), patch.object(
                star_handlers_registry, "get_handlers_by_event_type", side_effect=handlers), patch(
                "astrbot.core.pipeline.waking_check.stage.star_map", host_plugins), patch(
                "astrbot.core.pipeline.process_stage.method.star_request.star_map", host_plugins), patch(
                "astrbot.core.pipeline.context_utils.star_map", host_plugins), patch(
                "astrbot.core.pipeline.result_decorate.stage.star_map", host_plugins):
            await scheduler._process_stages(item)
        return build, runner, tool_calls

    async def test_real_pipeline_restart_replay_stops_before_agent_and_direct_tool_status_offline(self):
        first = self.item()
        sent = []
        first.send = AsyncMock(side_effect=lambda chain: sent.append((chain.type, chain.get_plain_text())))
        build, runner, tools = await self.public_pipeline(first)
        await self.settle()
        build.assert_awaited_once()
        self.assertEqual(runner.step_count, 1)
        self.assertTrue(tools[0]["ok"])
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0][0], "tool_call")
        self.assertEqual(sent[1][1], "normal final reply")
        self.assertFalse(first.is_stopped())
        task = self.plugin.task_store.active_tasks()[0]
        key = task["host_message_key"]
        self.assertEqual(self.plugin.task_store.host_message(key)["public_claimed"], 1)
        await self.plugin.task_coordinator.close()
        self.plugin.task_store.close()
        self.plugin.task_store = TaskStore(self.path)
        self.plugin._task_routes.clear()
        self.plugin.task_coordinator = TaskCoordinator(self.plugin.task_store, self.plugin._request_decision,
            turn_sync=self.plugin._sync_task_turn)
        self.plugin.text_channel.online_peers = lambda: ()
        self.context.get_provider_by_id = lambda name: None
        self.plugin.task_planning_enabled = False
        before = self.plugin.task_store.get(task["task_id"])
        calls = len(self.calls)
        for _ in range(2):
            replay = self.item()
            replay.set_result(MessageEventResult().message("stale replay result"))
            replay.send = AsyncMock()
            build, runner, tools = await self.public_pipeline(replay)
            build.assert_not_awaited()
            self.assertEqual(runner.step_count, 0)
            self.assertEqual(tools, [])
            replay.send.assert_not_awaited()
            self.assertTrue(replay.is_stopped())
            self.assertFalse(replay.call_llm)
            self.assertIsNone(replay.get_result())
            self.assertEqual(self.plugin.task_store.get(task["task_id"]), before)
        self.assertEqual(len(self.calls), calls)
        self.assertEqual(len(self.ex.goals), 1)

    async def test_real_internal_request_fallback_prevents_replay_agent_tools_and_status(self):
        item, task = await self.admitted()
        await self.deliver(item, "first reply")
        replay = self.item()
        replay.send = AsyncMock()
        calls = len(self.calls)
        build, runner, tools = await self.public_pipeline(replay, early=False)
        build.assert_awaited_once()  # Internal Host builds before on_llm_request.
        self.assertEqual(runner.step_count, 0)
        self.assertEqual(tools, [])
        replay.send.assert_not_awaited()
        self.assertTrue(replay.is_stopped())
        self.assertFalse(replay.call_llm)
        self.assertIsNone(replay.get_result())
        self.assertEqual(len(self.calls), calls)
        self.assertEqual(self.plugin.task_store.host_message(task["host_message_key"])["public_claimed"], 1)

    async def test_replay_guard_only_matching_trusted_public_identity_and_idle_pipeline_runs(self):
        item, task = await self.admitted()
        await self.deliver(item, "first reply")
        private = self.item()
        private.set_extra("private_planning", True)
        other_session = event("hello", session="another-session", user="member-a", role="member")
        other_session.message_obj.message_id = item.message_obj.message_id
        adapter = plugin_module.AstrBotEXPlatformAdapter({}, {}, asyncio.Queue())
        self.plugin._adapter = adapter
        def ex_event(peer, metadata, *, bound_adapter=adapter):
            return plugin_module.AstrBotEXMessageEvent(item.message_str, item.message_obj, item.platform_meta,
                item.get_session_id(), bound_adapter, peer, metadata)
        variants = [private, self.item(user="foreign"), other_session, self.item(message="new-message"),
            ex_event(b"foreign-peer", {}), ex_event(b"trusted", {"visibility": "private_planning"}),
            ex_event(b"trusted", {"source": "execution_feedback", "task_id": task["task_id"]}),
            ex_event(b"trusted", {}, bound_adapter=object())]
        for other in variants:
            other.set_result(MessageEventResult().message("keep other path"))
            self.assertFalse(self.plugin._suppress_host_reply_replay(other))
            self.assertFalse(other.is_stopped())
            self.assertEqual(other.get_result().get_plain_text(), "keep other path")
        trusted_ex_repeat = ex_event(b"trusted", {})
        self.assertTrue(self.plugin._suppress_host_reply_replay(trusted_ex_repeat))
        self.assertTrue(trusted_ex_repeat.is_stopped())
        self.assertIsNone(trusted_ex_repeat.get_result())
        request = ProviderRequest(func_tool=self.shared)
        request.private_planning = True
        private_request = self.item()
        await self.plugin.inject_ex_context(private_request, request)
        self.assertFalse(private_request.is_stopped())
        self.assertIs(request.func_tool, self.shared)
        # Actual Host pipeline, not only a helper assertion: a new idle message still replies.
        idle = self.item("hello", message="idle-new")
        idle.set_extra("enable_streaming", False)  # Offline base event has no streaming implementation.
        idle.send = AsyncMock()
        self.plugin.task_planning_enabled = False
        build, runner, tools = await self.public_pipeline(idle)
        build.assert_awaited_once()
        self.assertEqual(runner.step_count, 1)
        self.assertEqual(tools, [])
        self.assertFalse(idle.is_stopped())
        self.assertEqual(idle.send.await_count, 2)  # Normal Host status + final reply.
        self.assertEqual(len(self.plugin.task_store.active_tasks()), 1)

    async def test_real_public_hook_one_reply_retry_and_SQLite_restart(self):
        item, task = await self.admitted()
        sent = []
        item.send = AsyncMock(side_effect=lambda chain: sent.append(chain.get_plain_text()))
        await self.deliver(item, "I will work on that.")
        self.assertEqual(sent, ["I will work on that."])
        receipt = self.plugin.task_store.host_message(task["host_message_key"])
        self.assertEqual(receipt["public_claimed"], 1)
        await self.deliver(item, "retry duplicate")
        self.assertEqual(len(sent), 1)
        await self.plugin.task_coordinator.close()
        self.plugin.task_store.close()
        self.plugin.task_store = TaskStore(self.path)
        self.plugin._task_routes.clear()
        self.plugin.task_coordinator = TaskCoordinator(self.plugin.task_store, self.plugin._request_decision,
            turn_sync=self.plugin._sync_task_turn)
        self.plugin.text_channel.online_peers = lambda: ()
        self.context.get_provider_by_id = lambda name: None
        replay = self.item()
        replay.send = AsyncMock(side_effect=lambda chain: sent.append(chain.get_plain_text()))
        await self.deliver(replay, "restart duplicate")
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.plugin.task_store.host_message(task["host_message_key"])["public_claimed"], 1)
        self.assertEqual(len(self.ex.goals), 1)

    async def test_idle_chat_zero_tasks_and_real_Host_reply_not_suppressed(self):
        item = self.item("hello")
        await self.plugin.prepare_host_task_output(item)
        _, tool = await self.tool(item)
        self.assertIsNotNone(tool)
        item.send = AsyncMock()
        for _ in range(2):
            await self.deliver(item, "hello back")
        self.assertEqual(item.send.await_count, 2)
        self.assertEqual(self.plugin.task_store.active_tasks(), [])
        self.assertEqual(self.ex.goals, {})
        self.assertEqual(self.public, [])

    async def test_owner_disabled_provider_unavailable_update_review_cancel_tool(self):
        _, task = await self.admitted()
        before_goals = len(self.ex.goals)
        self.cap["execution"].update(mode="disabled", execution_allowed=False, runtime_state="paused")
        self.context.get_provider_by_id = lambda name: None
        update = self.item(message="update")
        request, tool = await self.tool(update)
        self.assertIsNotNone(tool)
        self.assertIsNot(request.func_tool, self.shared)
        denied = json.loads(await tool.call(self.wrapper(update), operation="create", text="new"))
        self.assertEqual(denied["error"], "decision_execution_unavailable")
        result = json.loads(await tool.call(self.wrapper(update), operation="update", text="changed intent"))
        self.assertTrue(result["ok"])
        self.assertEqual(self.plugin.task_store.get(task["task_id"])["text"], "changed intent")
        review = self.item(message="review-unproven")
        _, tool = await self.tool(review)
        result = json.loads(await tool.call(self.wrapper(review), operation="review"))
        self.assertEqual(result["error"], "unresolved_execution")
        current = self.plugin.task_store.get(task["task_id"])
        self.assertEqual(current["status"], "resume_review")
        self.assertFalse(current["needs_planning"])
        self.assertIsNotNone(current["current_goal"])
        self.ex.feedback(current, "canceled")
        review = self.item(message="review-proven")
        _, tool = await self.tool(review)
        result = json.loads(await tool.call(self.wrapper(review), operation="review"))
        self.assertTrue(result["ok"])
        await self.settle()
        self.assertEqual(len(self.ex.goals), before_goals)
        self.assertFalse(self.cap["execution"]["execution_allowed"])
        cancel = self.item(message="cancel")
        _, tool = await self.tool(cancel)
        result = json.loads(await tool.call(self.wrapper(cancel), operation="cancel"))
        self.assertTrue(result["ok"])
        self.assertFalse(self.plugin.task_store.get(task["task_id"])["active"])

    async def test_controls_provider_loss_after_offer_and_stale_owner_event_peer_session_denied(self):
        _, task = await self.admitted()
        item = self.item(message="control")
        _, tool = await self.tool(item)
        self.context.get_provider_by_id = lambda name: None
        self.cap["execution"]["mode"] = "disabled"
        foreign = self.item(user="other", message="foreign")
        _, foreign_tool = await self.tool(foreign)
        self.assertIsNone(foreign_tool)
        for other in (foreign, self.item(message="new-message")):
            denied = json.loads(await tool.call(self.wrapper(other), operation="cancel"))
            self.assertEqual(denied["error"], "stale_host_context")
        self.plugin.text_channel.online_peers = lambda: ()
        denied = json.loads(await tool.call(self.wrapper(item), operation="cancel"))
        self.assertEqual(denied["error"], "task_peer_unavailable")
        self.plugin.text_channel.online_peers = lambda: (b"trusted",)
        self.cap["ex_session"] = "ex2"
        denied = json.loads(await tool.call(self.wrapper(item), operation="cancel"))
        self.assertEqual(denied["error"], "stale_ex_session")
        self.cap["ex_session"] = "ex1"
        result = json.loads(await tool.call(self.wrapper(item), operation="update", text="owner intent"))
        self.assertTrue(result["ok"])
        self.assertEqual(self.plugin.task_store.get(task["task_id"])["text"], "owner intent")

    async def test_trusted_identity_not_LLM_ids_private_and_feedback_do_not_claim(self):
        item, task = await self.admitted()
        key = task["host_message_key"]
        variants = [self.item(user="other"), self.item(message="unadmitted")]
        private = self.item()
        private.set_extra("private_planning", True)
        variants.append(private)
        adapter = plugin_module.AstrBotEXPlatformAdapter({}, {}, asyncio.Queue())
        self.plugin._adapter = adapter
        feedback = plugin_module.AstrBotEXMessageEvent(item.message_str, item.message_obj,
            item.platform_meta, item.get_session_id(), adapter, b"trusted",
            {"source": "execution_feedback", "task_id": task["task_id"]})
        variants.append(feedback)
        for other in variants:
            other.set_extra("message_key", key)
            other.message_obj.raw_message = {"task_id": task["task_id"], "message_key": key}
            other.set_result(MessageEventResult().message("not this admission"))
            await self.plugin.guard_host_task_reply(other)
            self.assertEqual(other.get_result().get_plain_text(), "not this admission")
            self.assertEqual(self.plugin.task_store.host_message(key)["public_claimed"], 0)
        item.set_result(MessageEventResult().message("real owner reply"))
        await self.plugin.guard_host_task_reply(item)
        self.assertEqual(self.plugin.task_store.host_message(key)["public_claimed"], 1)

    async def test_empty_result_and_lost_send_do_not_allow_retry(self):
        item, task = await self.admitted()
        item.set_result(MessageEventResult())
        await self.plugin.guard_host_task_reply(item)
        self.assertEqual(self.plugin.task_store.host_message(task["host_message_key"])["public_claimed"], 0)
        item.send = AsyncMock(side_effect=RuntimeError("offline send failure"))
        await self.deliver(item, "first attempted reply")
        await self.deliver(item, "retry")
        self.assertEqual(item.send.await_count, 1)

    async def test_installed_internal_Host_selects_buffered_before_tool_admission(self):
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
        item = self.item()
        await self.plugin.prepare_host_task_output(item)
        self.assertIs(item.get_extra("enable_streaming"), False)
        stage = internal.InternalAgentSubStage()
        stage.streaming_response = True
        stage.main_agent_cfg = internal.MainAgentBuildConfig(tool_call_timeout=1)
        stage.ctx = SimpleNamespace(plugin_manager=SimpleNamespace(context=self.context))
        stage.unsupported_streaming_strategy = "realtime_segmenting"
        stage.max_step = 3
        stage.show_tool_use = stage.show_tool_call_result = stage.show_reasoning = False
        stage.buffer_intermediate_messages = False
        offline_provider = SimpleNamespace(provider_config={"id": "offline", "api_base": ""},
            get_model=lambda: "offline", meta=lambda: SimpleNamespace(type="offline"))
        runner = SimpleNamespace(done=lambda: False, was_aborted=lambda: False, get_final_llm_resp=lambda: None,
            stats=SimpleNamespace(to_dict=lambda: {}), run_context=SimpleNamespace(messages=[]), provider=offline_provider)
        built = internal.MainAgentBuildResult(runner, ProviderRequest(func_tool=self.shared), offline_provider)
        selected = []
        async def build(**kwargs):
            selected.append(kwargs["config"].streaming_response)
            return built
        async def run(*args, **kwargs):
            tool = built.provider_request.func_tool.get_tool("manage_astrbotex_task")
            result = json.loads(await tool.call(self.wrapper(item), operation="create", text="bounded work"))
            self.assertTrue(result["ok"])
            item.set_result(MessageEventResult().message("one ordinary reply"))
            yield item.get_result()
        handler = self.registered("inject_ex_context")
        def handlers(kind, **kwargs):
            return [handler] if kind == EventType.OnLLMRequestEvent else []
        decorate, respond = await self.stages()
        item.send = AsyncMock()
        with patch.object(internal, "build_main_agent", side_effect=build), patch.object(internal, "run_agent", run), patch.object(
                internal, "try_capture_follow_up", return_value=None), patch.object(internal, "register_active_runner"), patch.object(
                internal, "unregister_active_runner"), patch.object(internal, "_record_internal_agent_stats", new=AsyncMock()), patch.object(
                internal.Metric, "upload", new=AsyncMock()), patch.object(star_handlers_registry,
                "get_handlers_by_event_type", side_effect=handlers), patch(
                "astrbot.core.pipeline.context_utils.star_map", {handler.handler_module_path: SimpleNamespace(name="offline")}):
            async for _ in stage.process(item, ""):
                await self.deliver(item, item.get_result().get_plain_text(), decorate=decorate, respond=respond)
        await self.settle()
        self.assertEqual(selected, [False])
        item.send.assert_awaited_once()
        task = self.plugin.task_store.active_tasks()[0]
        self.assertEqual(self.plugin.task_store.host_message(task["host_message_key"])["public_claimed"], 1)

    async def test_command_error_claim_does_not_swallow_first_public_result(self):
        item, task = await self.admitted()
        conflict = self.item("ex_task different work")
        conflict.send = AsyncMock()
        await self.plugin.ex_task(conflict)
        text = conflict.get_result().get_plain_text()
        self.assertIn("duplicate_request_id_conflict", text)
        original_result = conflict.get_result()
        await self.plugin.prepare_host_task_output(conflict)
        self.assertIs(conflict.get_result(), original_result)
        self.assertFalse(self.plugin._suppress_host_reply_replay(conflict))
        copied = self.item("ex_task different work")
        copied.set_result(original_result)
        copied.set_extra("astrbotex_claimed_command_result", conflict.get_extra("astrbotex_claimed_command_result"))
        copied.set_extra("astrbotex_claimed_command_event", conflict)
        self.assertTrue(self.plugin._suppress_host_reply_replay(copied))
        self.assertTrue(copied.is_stopped())
        self.assertIsNone(copied.get_result())
        decorate, respond = await self.stages()
        handler = self.registered("guard_host_task_reply")
        def handlers(kind, **kwargs):
            return [handler] if kind == EventType.OnDecoratingResultEvent else []
        with patch.object(star_handlers_registry, "get_handlers_by_event_type", side_effect=handlers), patch(
                "astrbot.core.pipeline.result_decorate.stage.star_map",
                {handler.handler_module_path: SimpleNamespace(name="offline-interaction")}):
            async for _ in decorate.process(conflict):
                pass
            await respond.process(conflict)
        conflict.send.assert_awaited_once()
        await self.deliver(conflict, "second result")
        conflict.send.assert_awaited_once()
        self.assertEqual(self.plugin.task_store.host_message(task["host_message_key"])["public_claimed"], 1)

    async def test_buffer_handler_registered_public_filter_and_unbound_idle_stream_untouched(self):
        from astrbot.core.star.filter.event_message_type import EventMessageTypeFilter
        metadata = star_handlers_registry.get_handler_by_full_name(
            f"{plugin_module.__name__}_prepare_host_task_output")
        message_filter = next(f for f in metadata.event_filters if isinstance(f, EventMessageTypeFilter))
        item = self.item("hello")
        self.assertTrue(message_filter.filter(item, {}))
        wake_filter = next(f for f in metadata.event_filters if isinstance(f, plugin_module.WokenHostTaskFilter))
        item.is_wake = False
        item.is_at_or_wake_command = False
        self.assertFalse(wake_filter.filter(item, {}))
        item.is_at_or_wake_command = True
        self.assertTrue(wake_filter.filter(item, {}))
        self.plugin.text_channel.online_peers = lambda: ()
        await self.plugin.prepare_host_task_output(item)
        self.assertIsNone(item.get_extra("enable_streaming"))
        self.assertEqual(self.plugin.task_store.active_tasks(), [])
        self.plugin.text_channel.online_peers = lambda: (b"trusted",)
        _, tool = await self.tool(item)
        self.assertIsNone(tool)  # Do not admit after late capability recovery.

    async def test_streaming_bypass_limit_explicit_and_private_no_mutation(self):
        item, task = await self.admitted()
        private = self.item()
        private.set_extra("private_planning", True)
        await self.plugin.prepare_host_task_output(private)
        self.assertIsNone(private.get_extra("enable_streaming"))
        item.set_result(MessageEventResult().message("already-streamed").set_result_content_type(ResultContentType.STREAMING_FINISH))
        await self.plugin.guard_host_task_reply(item)
        self.assertEqual(self.plugin.task_store.host_message(task["host_message_key"])["public_claimed"], 0)
