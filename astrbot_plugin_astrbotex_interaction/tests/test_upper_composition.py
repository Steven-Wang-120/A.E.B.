"""B10 composition tests: lifecycle, takeover, suppression and teardown.

Two layers are proved here:

* ``UpperCompositionTests`` injects substitutes for B01-B09 and drives the real
  Host entry points, so what is under test is the wiring -- ordering, failure
  containment, single processing, and the one public exit.
* ``CompositionContractTests`` exercises the real ``UpperModuleFactory`` with
  recording stand-ins for the modules. It pins the exact keyword arguments B10
  passes to each collaborator, which is the contract the other tasks build to.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

WORK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK_ROOT))

from astrbot_plugin_astrbotex_interaction import main as plugin_module
from astrbot_plugin_astrbotex_interaction.tests.test_plugin_channels import FakeContext


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def platform_event(text: str, *, session: str = "session-upper", user: str = "user-upper",
                   message_id: str | None = None):
    message = plugin_module.AstrBotMessage()
    message.type = plugin_module.MessageType.FRIEND_MESSAGE
    message.session_id = session
    message.sender = plugin_module.MessageMember(user_id=user, nickname="offline user")
    message.message_id = message_id if message_id is not None else f"message-{text}"
    message.message_str = text
    message.message = [plugin_module.Plain(text=text)]
    item = plugin_module.AstrMessageEvent(
        text,
        message,
        plugin_module.PlatformMetadata(name="offline", description="offline", id="offline-1"),
        session,
    )
    item.role = "admin"
    item.is_at_or_wake_command = True
    item.is_wake = True
    return item


def ex_event(text: str, *, peer: bytes = b"ex-trusted", adapter: Any = None):
    message = plugin_module.AstrBotMessage()
    message.type = plugin_module.MessageType.FRIEND_MESSAGE
    message.session_id = "session-ex"
    message.sender = plugin_module.MessageMember(user_id="astrbotex_robot", nickname="EX")
    message.message_id = f"ex-{text}"
    message.message_str = text
    message.message = [plugin_module.Plain(text=text)]
    item = plugin_module.AstrBotEXMessageEvent(
        text,
        message,
        plugin_module.PlatformMetadata(name="astrbotex", description="ex", id="astrbotex"),
        "session-ex",
        adapter,
        peer,
        {},
    )
    item.is_at_or_wake_command = True
    item.is_wake = True
    return item


# --------------------------------------------------------------------------
# Substitutes: same constructor shape and same observable behaviour as B01-B09
# --------------------------------------------------------------------------


class FakeTrustedInput:
    """Mirrors ``upper_contracts.TrustedUpperInput`` including ``issue``."""

    def __init__(self, **kwargs: Any) -> None:
        self.fields = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def issue(cls, **kwargs: Any) -> "FakeTrustedInput":
        return cls(**kwargs)


class FakeSettingsRepository:
    def __init__(self, *, config: Any = None) -> None:
        self.config = config
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeCatalog:
    def __init__(self, *, data_root: Path) -> None:
        self.data_root = data_root
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePrompts:
    def __init__(self) -> None:
        self.built = 0


class FakeStore:
    def __init__(self, task_store: Any = None) -> None:
        self.task_store = task_store
        self.closed = 0

    def get_plan(self, plan_id: str) -> Any:
        raise plugin_module.TaskError("unknown_plan")

    def close(self) -> None:
        self.closed += 1


class FakeModels:
    def __init__(self, host: Any, *, timeout_s: float = 30.0) -> None:
        self.host = host
        self.timeout_s = timeout_s
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class FakeControl:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.start_calls = 0
        self.close_calls = 0
        self.started = False
        self.start_error: Exception | None = None

    async def start(self) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    async def close(self) -> None:
        self.close_calls += 1
        self.started = False


class FakeRouter:
    def __init__(self, store: Any, *, delivery: str = "text") -> None:
        self.store = store
        self.delivery = delivery


class FakePipeline:
    """Reproduces the B05 surface B10 depends on: async process with per-call peer."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.inputs: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self.script: dict[str, tuple] = {}
        self.fail: Exception | None = None
        self.error_code = ""
        self.close_calls = 0

    async def process(self, trusted: Any, *, public_sender=None, peer=None) -> Any:
        self.inputs.append(trusted)
        self.calls.append({"public_sender": public_sender, "peer": peer})
        if self.fail is not None:
            raise self.fail
        route, public_text = self.script.get(
            trusted.input_text, ("speech_only", "收到。")
        )
        public_sent = None
        if public_text is not None and public_sender is not None:
            public_sent = await public_sender(public_text)
        return SimpleNamespace(
            route=route, public_sent=public_sent, error_code=self.error_code
        )

    async def close(self) -> None:
        self.close_calls += 1


class FakeUpperFactory:
    """Substitute for UpperModuleFactory; records what main.py asked for."""

    def __init__(self) -> None:
        self.loads: list[Any] = []
        self.components: list[Any] = []
        self.fail_with: Exception | None = None
        self.api_calls: list[Any] = []
        self.api_error: Exception | None = None

    def load(self, plugin, settings):  # noqa: ANN001 - mirrors the real factory
        self.loads.append(settings)
        if self.fail_with is not None:
            raise self.fail_with
        store = FakeStore(plugin.task_store)
        components = plugin_module.UpperComponents(
            settings=settings,
            data_root=plugin.upper_data_root(),
            settings_repository=FakeSettingsRepository(config=plugin.plugin_config),
            catalog=FakeCatalog(data_root=plugin.upper_data_root()),
            prompts=FakePrompts(),
            store=store,
            models=FakeModels(plugin.upper_host_adapter(), timeout_s=settings.model_timeout_s),
            control=FakeControl(
                peer_id=settings.control_peer_id,
                robot_id=settings.robot_id,
                bind_host=settings.control_bind_host,
                port=settings.control_port,
            ),
            router=FakeRouter(store),
            pipeline=FakePipeline(
                robot_id=settings.robot_id,
                peer_id=settings.control_peer_id,
                route_ref=settings.control_route_ref,
                alive=plugin.upper_alive,
            ),
            trusted_input=FakeTrustedInput,
            api_registrar=self._register_api,
        )
        self.components.append(components)
        return components

    def _register_api(self, context, dependencies) -> None:
        self.api_calls.append(dependencies)
        if self.api_error is not None:
            raise self.api_error


class UpperTestContext(FakeContext):
    """A Host that can plan tasks, so the real TaskStore path runs.

    Upper control layers on the legacy store, so a context without
    ``llm_generate`` would silently disable the feature and the composition
    tests would pass for the wrong reason.
    """

    async def llm_generate(self, **kwargs: Any) -> Any:
        return SimpleNamespace(is_chunk=False, role="assistant", completion_text="")

    def get_using_provider(self, umo=None):
        return SimpleNamespace(meta=lambda: SimpleNamespace(id="provider-chat"))


class RecordingContext(UpperTestContext):
    """UpperTestContext plus the Host web-API surface."""

    def __init__(self) -> None:
        super().__init__()
        self.web_apis: list[tuple] = []

    def register_web_api(self, route, view_handler, methods, desc) -> None:
        self.web_apis.append((route, view_handler, tuple(methods), desc))


class FakeTextChannel:
    """Records interaction.reply calls without a live socket."""

    def __init__(self) -> None:
        self.requests: list[tuple] = []
        self.default_peer = b"ex-trusted"
        self.peer_count = 1
        self.closed = 0

    async def request(self, method, payload, **kwargs):
        self.requests.append((method, payload, kwargs))
        return {"ok": True}

    async def close(self) -> None:
        self.closed += 1


async def _noop_send(chain) -> None:
    return None


class UpperCompositionTests(unittest.IsolatedAsyncioTestCase):
    def build(self, *, config: dict | None = None, context: Any = None) -> Any:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Keep the legacy TaskStore out of the Host data directory.
        previous = os.environ.get("ASTRBOTEX_TASK_DB_PATH")
        os.environ["ASTRBOTEX_TASK_DB_PATH"] = str(Path(self.tmp.name) / "tasks.sqlite3")
        if previous is None:
            self.addCleanup(os.environ.pop, "ASTRBOTEX_TASK_DB_PATH", None)
        else:
            self.addCleanup(os.environ.__setitem__, "ASTRBOTEX_TASK_DB_PATH", previous)
        self.context = context if context is not None else UpperTestContext()
        self.plugin = plugin_module.AstrBotEXInteractionPlugin(self.context, config)
        self.plugin._upper_data_root_override = Path(self.tmp.name)
        self.plugin.zmq_bind_host = "127.0.0.1"
        self.plugin.text_port = free_port()
        self.plugin.audio_port = free_port()
        self.plugin.vision_port = free_port()
        try:
            self.plugin.upper_settings = plugin_module.UpperSettings.from_config(config, environ={})
        except plugin_module.UpperUnavailable:
            # The plugin keeps its own disabled fallback for a rejected setting.
            self.plugin.upper_settings = plugin_module.UpperSettings()
        self.factory = FakeUpperFactory()
        self.plugin.upper_factory = self.factory
        return self.plugin

    def enabled_config(self, **overrides: Any) -> dict:
        config = {
            "upper_enabled": True,
            "control_enabled": True,
            "control_port": free_port(),
            "robot_id": "robot-upper",
            "control_peer_id": "ex-control-peer",
            "routing_provider_id": "provider-routing",
            "generation_provider_id": "provider-generation",
        }
        config.update(overrides)
        return config

    async def start(self) -> Any:
        await self.plugin.initialize()
        self.addAsyncCleanup(self.plugin.terminate)
        return self.plugin

    # -- lifecycle ---------------------------------------------------------

    async def test_feature_off_builds_nothing_and_keeps_legacy_channels(self) -> None:
        plugin = self.build(config={"upper_enabled": False})
        await self.start()

        self.assertEqual(self.factory.loads, [])
        self.assertIsNone(plugin._upper)
        self.assertFalse(plugin.upper_ready)
        self.assertEqual(plugin.upper_status["reason"], "upper_disabled")
        self.assertIsNotNone(plugin.text_channel)
        self.assertIsNotNone(plugin.audio_channel)
        self.assertIsNotNone(plugin.vision_channel)
        self.assertNotIn("control", plugin._transport_status()["channels"])

    async def test_upper_without_control_channel_is_refused(self) -> None:
        # Every upper turn ends at the control channel. Enabling the feature
        # without it would classify an input and have nowhere to deliver.
        plugin = self.build(config=self.enabled_config(control_enabled=False))
        await self.start()

        self.assertEqual(self.factory.loads, [])
        self.assertFalse(plugin.upper_ready)
        self.assertEqual(plugin.upper_status["reason"], "control_disabled")
        self.assertIsNotNone(plugin.text_channel)

    async def test_enabled_composition_builds_in_dependency_order(self) -> None:
        plugin = self.build(config=self.enabled_config(), context=RecordingContext())
        await self.start()

        self.assertEqual(len(self.factory.loads), 1)
        self.assertTrue(plugin.upper_ready)
        components = self.factory.components[0]
        self.assertIs(components.settings, plugin.upper_settings)
        self.assertEqual(components.data_root, Path(self.tmp.name))
        self.assertEqual(len(self.factory.api_calls), 1)
        self.assertTrue(components.api_registered)
        self.assertEqual(plugin.upper_status["reason"], None)

    async def test_control_enabled_starts_fourth_channel_and_reports_it(self) -> None:
        plugin = self.build(config=self.enabled_config(control_enabled=True, control_port=free_port()))
        await self.start()

        components = self.factory.components[0]
        self.assertEqual(components.control.start_calls, 1)
        self.assertTrue(components.control.started)
        status = plugin._transport_status()
        self.assertIn("control", status["channels"])
        self.assertEqual(status["channels"]["control"]["port"], plugin.upper_settings.control_port)
        self.assertEqual(status["upper"]["control_started"], True)

    async def test_missing_modules_degrade_without_touching_legacy(self) -> None:
        plugin = self.build(config=self.enabled_config())
        self.factory.fail_with = plugin_module.UpperUnavailable("module_missing", "upper_pipeline")
        await self.start()

        self.assertIsNone(plugin._upper)
        self.assertFalse(plugin.upper_ready)
        self.assertEqual(plugin.upper_status["reason"], "module_missing")
        self.assertIsNotNone(plugin.text_channel)
        await plugin.terminate()  # idempotent teardown after a failed composition

    async def test_control_start_failure_keeps_takeover_off(self) -> None:
        plugin = self.build(config=self.enabled_config(control_enabled=True, control_port=free_port()))
        original_load = self.factory.load

        def load(plugin_arg, settings):  # noqa: ANN001
            components = original_load(plugin_arg, settings)
            components.control.start_error = RuntimeError("bind failed")
            return components

        self.factory.load = load
        await self.start()

        components = self.factory.components[0]
        self.assertFalse(components.control_started)
        self.assertFalse(plugin.upper_ready)
        self.assertEqual(plugin.upper_status["reason"], "control_unavailable")

    async def test_invalid_settings_keep_legacy_and_record_reason(self) -> None:
        plugin = self.build(config={"upper_enabled": True, "control_port": "not-a-port"})
        self.assertEqual(plugin._upper_config_error, "invalid_upper_settings")
        self.factory.loads.clear()
        await self.start()

        self.assertEqual(self.factory.loads, [])
        self.assertFalse(plugin.upper_ready)
        self.assertEqual(plugin.upper_status["reason"], "invalid_upper_settings")
        self.assertIsNotNone(plugin.text_channel)

    async def test_host_without_page_api_records_unavailable(self) -> None:
        # FakeContext exposes no register_web_api: the page is unavailable, not fatal.
        plugin = self.build(config=self.enabled_config())
        await self.start()

        self.assertEqual(self.factory.api_calls, [])
        self.assertEqual(plugin._upper_api_error, "host_page_api_unavailable")
        self.assertTrue(plugin.upper_ready)

    async def test_page_api_registration_failure_does_not_block_input(self) -> None:
        plugin = self.build(config=self.enabled_config(), context=RecordingContext())
        self.factory.api_error = RuntimeError("route taken")
        await self.start()

        self.assertEqual(plugin._upper_api_error, "upper_api_registration_failed")
        self.assertTrue(plugin.upper_ready)

    async def test_page_api_registered_on_real_host_surface(self) -> None:
        context = RecordingContext()
        plugin = self.build(config=self.enabled_config(), context=context)
        await self.start()

        self.assertEqual(len(self.factory.api_calls), 1)
        self.assertTrue(plugin.upper_status["api_registered"])
        dependencies = self.factory.api_calls[0]
        # B07's dependency object: the seams B10 assembled, plus the read-only
        # projections the page calls back into.
        self.assertIs(dependencies.settings, self.factory.components[0].settings_repository)
        self.assertIsInstance(dependencies.actions, plugin_module._UpperActionRepository)
        self.assertIs(dependencies.actions._catalog, self.factory.components[0].catalog)
        self.assertEqual(dependencies.plans.get_plan("missing"), None)
        self.assertIn(
            dependencies.ex_availability()["state"], {"online", "offline", "unknown"}
        )
        self.assertIn("control_state", dependencies.runtime_status())
        self.assertEqual(dependencies.notes["control_route_ref"], "upper-control")

    async def test_reload_three_times_does_not_accumulate_resources(self) -> None:
        plugin = self.build(config=self.enabled_config(control_enabled=True, control_port=free_port()))
        generations = []
        for _ in range(3):
            await plugin.initialize()
            generations.append(plugin.upper_generation)
            self.assertEqual(len(plugin._upper_live_events), 0)
            self.assertTrue(plugin.upper_ready)
        await plugin.terminate()

        self.assertEqual(len(self.factory.components), 3)
        self.assertEqual(len(set(generations)), 3)
        for components in self.factory.components:
            # Every reload closed the previous control channel exactly once.
            self.assertEqual(components.control.close_calls, 1)
            self.assertFalse(components.control.started)
            self.assertEqual(components.pipeline.close_calls, 1)
        # Distinct control objects, never a shared/duplicated listener.
        self.assertEqual(len({id(c.control) for c in self.factory.components}), 3)

    async def test_terminate_stops_new_input_before_legacy_store(self) -> None:
        plugin = self.build(config=self.enabled_config(control_enabled=True, control_port=free_port()))
        await plugin.initialize()
        components = self.factory.components[0]
        plugin.task_store = None

        await plugin.terminate()

        self.assertEqual(components.control.close_calls, 1)
        self.assertEqual(components.store.closed, 1)
        self.assertIsNone(plugin._upper)
        self.assertIsNone(plugin.text_channel)
        self.assertFalse(plugin.upper_ready)

    # -- takeover ----------------------------------------------------------

    async def test_takeover_processes_input_once_and_suppresses_default_llm(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        item = platform_event("帮我看一下前面的杯子")
        sent: list[str] = []
        llm_calls: list[bool] = []

        async def send(chain):
            sent.append(chain.get_plain_text())

        item.send = send
        original = item.should_call_llm
        item.should_call_llm = lambda value: (llm_calls.append(value), original(value))[1]

        await plugin.take_over_upper_input(item)

        self.assertEqual(len(pipeline.inputs), 1)
        self.assertEqual(sent, ["收到。"])
        self.assertEqual(llm_calls, [False])
        self.assertTrue(item.is_stopped())
        self.assertTrue(item.get_extra("upper_handled"))

    async def test_trusted_input_carries_only_framework_identity(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        item = platform_event("身份字段", message_id="host-message-identity")
        item.send = _noop_send

        await plugin.take_over_upper_input(item)

        fields = pipeline.inputs[0].fields
        self.assertEqual(
            set(fields),
            {"robot_id", "session_id", "user_id", "route_ref", "ex_session",
             "input_text", "input_key", "origin"},
        )
        self.assertEqual(fields["robot_id"], "robot-upper")
        self.assertEqual(fields["user_id"], item.get_sender_id())
        self.assertEqual(fields["route_ref"], "upper-control")
        self.assertEqual(fields["origin"], item.unified_msg_origin)
        # The contract requires a wire id and EX compares it with the session it
        # advertises, so a platform input carries this deployment's EX session.
        self.assertEqual(fields["ex_session"], self.plugin.session_id)

    async def test_replayed_host_event_is_processed_once(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        item = platform_event("同一个事件重放二十次")
        item.send = _noop_send

        for _ in range(20):
            await plugin.take_over_upper_input(item)

        self.assertEqual(len(pipeline.inputs), 1)

    async def test_replayed_message_id_yields_stable_input_key(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        first = platform_event("同一条消息的新事件", message_id="host-message-1")
        second = platform_event("同一条消息的新事件", message_id="host-message-1")
        first.send = _noop_send
        second.send = _noop_send

        await plugin.take_over_upper_input(first)
        await plugin.take_over_upper_input(second)

        self.assertEqual(len(pipeline.inputs), 2)
        # B04 owns the store-level dedupe; B10 owes it a stable key.
        self.assertEqual(pipeline.inputs[0].input_key, pipeline.inputs[1].input_key)

    async def test_message_without_id_never_collides(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        first = platform_event("同样的文字")
        second = platform_event("同样的文字")
        first.message_obj.message_id = ""
        second.message_obj.message_id = ""
        first.send = _noop_send
        second.send = _noop_send

        await plugin.take_over_upper_input(first)
        await plugin.take_over_upper_input(second)

        self.assertEqual(len(pipeline.inputs), 2)
        self.assertNotEqual(pipeline.inputs[0].input_key, pipeline.inputs[1].input_key)

    async def test_management_command_keeps_priority(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        item = platform_event("/ex_task 做一个任务")
        from astrbot.core.star.filter.command import CommandFilter

        item.set_extra(
            "activated_handlers",
            [SimpleNamespace(event_filters=[CommandFilter("ex_task")])],
        )
        await plugin.take_over_upper_input(item)

        self.assertEqual(pipeline.inputs, [])
        self.assertFalse(item.is_stopped())

    async def test_active_legacy_task_refuses_upper_input(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        from astrbot_plugin_astrbotex_interaction.task_store import TaskStore

        store = TaskStore(str(Path(self.tmp.name) / "legacy.db"))
        self.addCleanup(store.close)
        item = platform_event("这条被旧任务占用")
        from astrbot_plugin_astrbotex_interaction.task_models import TaskAuthority

        store.create(
            TaskAuthority("robot-upper", item.unified_msg_origin, item.get_sender_id(), "route-1", True),
            "legacy work",
            "request-1",
        )
        plugin.task_store = store

        await plugin.take_over_upper_input(item)

        self.assertEqual(pipeline.inputs, [])
        self.assertFalse(item.is_stopped())

    async def test_pipeline_failure_still_suppresses_default_llm(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        pipeline.fail = plugin_module.UpperUnavailable("decision_error")
        item = platform_event("分类失败也不能走默认回复")
        sent: list[str] = []

        async def send(chain):
            sent.append(chain.get_plain_text())

        item.send = send
        await plugin.take_over_upper_input(item)

        self.assertEqual(sent, [])
        self.assertTrue(item.is_stopped())
        self.assertEqual(item.get_extra("upper_error"), "decision_error")

    async def test_recorded_error_code_suppresses_default_llm(self) -> None:
        # B05 reports a refused turn through the result, not by raising. The
        # default reply must still stay off: a second interpretation is a
        # second reply.
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        pipeline.error_code = "control_unavailable"
        item = platform_event("注入不可用")
        item.send = _noop_send

        await plugin.take_over_upper_input(item)

        self.assertTrue(item.is_stopped())
        self.assertEqual(item.get_extra("upper_error"), "control_unavailable")

    async def test_late_result_after_reload_produces_no_output(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        item = platform_event("重载后迟到的模型结果")
        sent: list[str] = []

        async def send(chain):
            sent.append(chain.get_plain_text())

        item.send = send
        await plugin.take_over_upper_input(item)
        await plugin.terminate()

        sender = plugin.upper_public_sender_for(item, "platform")
        self.assertFalse(await sender("迟到的话"))
        self.assertEqual(sent, ["收到。"])

    async def test_ex_voice_input_replies_on_text_peer_without_task_fields(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        channel = FakeTextChannel()
        plugin.text_channel = channel
        item = ex_event("语音转出来的文字", adapter=plugin._adapter)

        await plugin.take_over_upper_input(item)

        self.assertEqual(len(pipeline.inputs), 1)
        self.assertEqual(pipeline.inputs[0].ex_session, "session-ex")
        self.assertIsNone(pipeline.calls[0]["peer"])
        self.assertEqual(len(channel.requests), 1)
        method, payload, kwargs = channel.requests[0]
        self.assertEqual(method, "interaction.reply")
        self.assertEqual(payload["text"], "收到。")
        self.assertEqual(kwargs["peer"], b"ex-trusted")
        for forbidden in ("task_id", "route_ref", "generation", "visibility"):
            self.assertNotIn(forbidden, payload)
        self.assertTrue(item.is_stopped())

    async def test_platform_input_passes_no_peer_override(self) -> None:
        # A platform event has no EX peer; B05 must fall back to the configured one.
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        item = platform_event("平台消息")
        item.send = _noop_send

        await plugin.take_over_upper_input(item)

        self.assertIsNone(pipeline.calls[0]["peer"])

    async def test_untrusted_ex_event_is_rejected(self) -> None:
        plugin = self.build(config=self.enabled_config())
        await self.start()
        pipeline = self.factory.components[0].pipeline
        item = ex_event("伪造的 EX 事件", adapter=SimpleNamespace())

        await plugin.take_over_upper_input(item)

        self.assertEqual(pipeline.inputs, [])
        self.assertFalse(item.is_stopped())


# --------------------------------------------------------------------------
# Real factory: the keyword contract with B01-B09
# --------------------------------------------------------------------------


class Recorder:
    """Records the keyword arguments one constructor received."""

    def __init__(self, name: str, *, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.error is not None:
            raise self.error
        self.calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(name=self.name, args=args, kwargs=kwargs, close=self._close)

    def _close(self) -> None:
        self.closed += 1


class CompositionContractTests(unittest.IsolatedAsyncioTestCase):
    """Pins the exact arguments B10 hands to each B01-B09 collaborator."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_root = Path(self.tmp.name)
        self.recorders: dict[str, Recorder] = {}
        self.settings = plugin_module.UpperSettings(
            enabled=True,
            control_enabled=True,
            control_port=free_port(),
            control_peer_id="ex-control-peer",
            robot_id="robot-upper",
            routing_provider_id="provider-routing",
            generation_provider_id="provider-generation",
            control_route_ref="upper-control",
            model_timeout_s=12.5,
        )
        self.plugin = SimpleNamespace(
            plugin_config={"upper_enabled": True},
            task_store=SimpleNamespace(name="task-store"),
            upper_data_root=lambda: self.data_root,
            upper_control_occupied=lambda: False,
            upper_alive=lambda: True,
            upper_event=lambda record: None,
            upper_control_status=lambda record: None,
            upper_host_adapter=lambda: self.host,
        )
        self.host = SimpleNamespace(
            llm_generate=self._generate,
            get_provider_by_id=lambda pid: None,
            get_current_chat_provider_id=lambda origin=None: "provider-x",
        )
        self.factory = plugin_module.UpperModuleFactory(
            "astrbot_plugin_astrbotex_interaction"
        )

    async def _generate(self, **kwargs: Any) -> Any:
        return SimpleNamespace(is_chunk=False, role="assistant")

    def install(self, *, skip: str | None = None,
                error: dict[str, Exception] | None = None) -> None:
        errors = error or {}
        recorders: dict[str, Recorder] = {}
        for key in plugin_module.UpperModuleFactory.SYMBOLS:
            if key == skip:
                continue
            recorders[key] = Recorder(key, error=errors.get(key))
        self.recorders = recorders

        def resolve(key: str) -> Any:
            if key not in recorders:
                raise plugin_module.UpperUnavailable("module_missing", key)
            return recorders[key]

        self.factory.resolve = resolve  # type: ignore[assignment]

    async def test_pipeline_receives_the_published_keyword_contract(self) -> None:
        self.install()
        components = await self.factory.load(self.plugin, self.settings)

        pipeline = self.recorders["pipeline"].calls[0]["kwargs"]
        self.assertEqual(
            set(pipeline),
            {"store", "models", "prompts", "settings", "catalog", "control",
             "router", "robot_id", "peer_id", "routing_provider_id",
             "generation_provider_id", "route_ref", "active_control", "alive",
             "on_event"},
        )
        self.assertEqual(pipeline["robot_id"], "robot-upper")
        self.assertEqual(pipeline["peer_id"], "ex-control-peer")
        self.assertEqual(pipeline["route_ref"], "upper-control")
        self.assertEqual(pipeline["routing_provider_id"], "provider-routing")
        self.assertEqual(pipeline["generation_provider_id"], "provider-generation")
        self.assertIs(pipeline["store"], components.store)
        self.assertIs(pipeline["router"], components.router)
        self.assertIs(pipeline["control"], components.control)
        # The two callables are the plugin's, not copies: a reload flips them.
        self.assertTrue(pipeline["alive"]())
        self.assertFalse(pipeline["active_control"]())

    async def test_each_collaborator_gets_its_documented_arguments(self) -> None:
        self.install()
        await self.factory.load(self.plugin, self.settings)

        settings_repo = self.recorders["settings_repository"].calls[0]["kwargs"]
        self.assertEqual(settings_repo, {"config": self.plugin.plugin_config})
        catalog = self.recorders["catalog"].calls[0]["kwargs"]
        self.assertEqual(catalog, {"data_root": self.data_root})
        self.assertEqual(self.recorders["prompts"].calls[0]["kwargs"], {})
        store_args = self.recorders["store"].calls[0]["args"]
        # UpperStore layers on the one legacy connection; it is not a new file.
        self.assertEqual(store_args, (self.plugin.task_store,))
        models_args = self.recorders["models"].calls[0]["args"]
        self.assertEqual(models_args, (self.host,))
        self.assertEqual(
            self.recorders["models"].calls[0]["kwargs"], {"timeout_s": 12.5}
        )
        control = self.recorders["control"].calls[0]["kwargs"]
        self.assertEqual(
            set(control),
            {"peer_id", "robot_id", "bind_host", "port",
             "request_timeout_sec", "on_status"},
        )
        self.assertEqual(control["peer_id"], "ex-control-peer")
        self.assertEqual(control["bind_host"], "127.0.0.1")
        self.assertEqual(control["port"], self.settings.control_port)

    async def test_router_wraps_the_upper_store(self) -> None:
        self.install()
        components = await self.factory.load(self.plugin, self.settings)
        router_args = self.recorders["router"].calls[0]["args"]
        self.assertEqual(len(router_args), 1)
        self.assertEqual(router_args[0].name, "store")
        self.assertIs(router_args[0], components.store)

    async def test_missing_api_module_is_a_partial_feature_not_a_failure(self) -> None:
        self.install(skip="api")
        components = await self.factory.load(self.plugin, self.settings)
        self.assertIsNone(components.api_registrar)
        self.assertFalse(components.api_registered)

    async def test_missing_required_module_raises_a_named_gap(self) -> None:
        self.install(skip="pipeline")
        with self.assertRaises(plugin_module.UpperUnavailable) as caught:
            await self.factory.load(self.plugin, self.settings)
        self.assertEqual(caught.exception.code, "module_missing")

    async def test_partial_assembly_closes_what_it_already_opened(self) -> None:
        self.install(error={"pipeline": RuntimeError("bad signature")})
        with self.assertRaises(plugin_module.UpperUnavailable) as caught:
            await self.factory.load(self.plugin, self.settings)
        self.assertEqual(caught.exception.code, "upper_composition_failed")
        for key in ("settings_repository", "catalog", "models", "control"):
            self.assertEqual(self.recorders[key].closed, 1, key)

    async def test_preflight_failures_name_the_missing_setting(self) -> None:
        self.install()
        for attribute, code in (
            ("robot_id", "robot_id_unconfigured"),
            ("control_peer_id", "control_peer_unconfigured"),
        ):
            broken = plugin_module.UpperSettings(
                enabled=True, control_enabled=True,
                robot_id="robot-upper", control_peer_id="ex-control-peer",
            )
            object.__setattr__(broken, attribute, "")
            with self.assertRaises(plugin_module.UpperUnavailable) as caught:
                await self.factory.load(self.plugin, broken)
            self.assertEqual(caught.exception.code, code)

    async def test_incomplete_host_adapter_is_refused_before_construction(self) -> None:
        self.install()
        self.plugin.upper_host_adapter = lambda: SimpleNamespace()
        with self.assertRaises(plugin_module.UpperUnavailable) as caught:
            await self.factory.load(self.plugin, self.settings)
        self.assertEqual(caught.exception.code, "host_adapter_incomplete")
        self.assertEqual(self.recorders["settings_repository"].calls, [])


class HostAdapterTests(unittest.IsolatedAsyncioTestCase):
    """``_UpperHostAdapter`` is the only Host surface B03 may touch."""

    def test_provider_id_is_resolved_synchronously(self) -> None:
        # B03 calls this without await, so a coroutine here would silently
        # never resolve and every turn would fail provider resolution.
        provider = SimpleNamespace(meta=lambda: SimpleNamespace(id="provider-real"))
        adapter = plugin_module._UpperHostAdapter(
            SimpleNamespace(get_using_provider=lambda origin: provider)
        )
        result = adapter.get_current_chat_provider_id("umo")
        self.assertEqual(result, "provider-real")
        self.assertFalse(inspect.isawaitable(result))

    def test_missing_provider_is_a_named_failure(self) -> None:
        adapter = plugin_module._UpperHostAdapter(
            SimpleNamespace(get_using_provider=lambda origin: None)
        )
        with self.assertRaises(plugin_module.UpperUnavailable) as caught:
            adapter.get_current_chat_provider_id("umo")
        self.assertEqual(caught.exception.code, "provider_unavailable")

    def test_unknown_provider_id_is_none_not_an_exception(self) -> None:
        adapter = plugin_module._UpperHostAdapter(SimpleNamespace())
        self.assertIsNone(adapter.get_provider_by_id("missing"))

    async def test_llm_generate_delegates_and_never_streams(self) -> None:
        seen: list[dict] = []

        async def generate(**kwargs):
            seen.append(kwargs)
            return SimpleNamespace(is_chunk=False, role="assistant")

        adapter = plugin_module._UpperHostAdapter(SimpleNamespace(llm_generate=generate))
        response = await adapter.llm_generate(
            chat_provider_id="p", system_prompt="s", contexts=[], tools=None, stream=False
        )
        self.assertEqual(response.role, "assistant")
        self.assertFalse(seen[0]["stream"])


if __name__ == "__main__":
    unittest.main()
