"""B11 AEB-side cross-end acceptance.

The whole AEB stack runs for real here: the real ``UpperModuleFactory`` builds
the real ``UpperStore``, ``UpperOutputRouter``, ``ControlChannel`` and
``UpperPipeline``; the real B10 takeover hook owns the Host event; a real
``zmq.asyncio`` ROUTER binds an ephemeral port and a real DEALER plays EX.

Only the two model calls are scripted, because they are the one part of the
turn that would otherwise reach a network and a paid provider. Everything the
acceptance actually claims -- one classification per input, one public reply,
one frozen batch, exact segment boundaries, sentinel isolation, and no second
delivery after a replay -- is produced by production code.

``VerifyRunnerTests`` checks the acceptance CLI itself: absent dependencies and
an existing evidence directory must fail loudly rather than report success.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

WORK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK_ROOT))

from astrbot_plugin_astrbotex_interaction import main as plugin_module
from astrbot_plugin_astrbotex_interaction.tests.test_control_channel import LoopbackExPeer
from astrbot_plugin_astrbotex_interaction.tests.test_upper_composition import UpperTestContext

RUN_ID = "b11e2e"
PUBLIC_SENTINEL = f"UPPER_PUBLIC_{RUN_ID}"
EX_SESSION = "ex-session-b11"
CONTROL_PEER = "ex-control-b11"


def exec_sentinel(ordinal: int) -> str:
    return f"UPPER_EXEC_SENTINEL_{RUN_ID}_{ordinal}"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def reserve_ports(count: int) -> list[int]:
    """Reserve several distinct ports at once.

    Calling ``free_port()`` repeatedly can hand back the same number twice
    because each socket is released immediately, which shows up as an
    occasional bind failure in a full-suite run.
    """
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def make_event(text: str, *, session: str = "session-e2e", user: str = "user-e2e",
               message_id: str | None = None):
    message = plugin_module.AstrBotMessage()
    message.type = plugin_module.MessageType.FRIEND_MESSAGE
    message.session_id = session
    message.sender = plugin_module.MessageMember(user_id=user, nickname="offline user")
    message.message_id = message_id or f"msg-{uuid.uuid4().hex}"
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


class HostConfig(dict):
    """``AstrBotConfig`` is a mutable mapping with ``save_config()``.

    The settings repository persists through the Host's own config object, so a
    plain dict would make upper control unavailable instead of testable.
    """

    def __init__(self, initial: dict | None = None) -> None:
        super().__init__(initial or {})
        self.saves = 0

    def save_config(self) -> None:
        self.saves += 1


class ScriptedModels:
    """Stands in for B03's two private model calls.

    The script is keyed by input text, so a test states the route and the
    content and everything downstream is real.
    """

    def __init__(self, script: dict[str, dict]) -> None:
        self.script = script
        self.classify_calls: list[str] = []
        self.generate_calls: list[tuple[str, str]] = []

    async def classify(self, text: str, prompt_snapshot: Any, provider_id: str = "") -> dict:
        self.classify_calls.append(text)
        return dict(self.script[text]["probabilities"])

    async def generate(self, text: str, route: str, prompt_snapshot: Any,
                       provider_id: str = "") -> Any:
        self.generate_calls.append((text, route))
        plan = self.script[text]
        return SimpleNamespace(
            public_text=plan["public_text"],
            segments=list(plan["segments"]),
            send_count=plan["send_count"],
        )


class ScriptedFactory(plugin_module.UpperModuleFactory):
    """The real factory with only the model adapter swapped out."""

    def __init__(self, package: str, models: ScriptedModels) -> None:
        super().__init__(package)
        self.models = models

    def resolve(self, key: str) -> Any:
        if key == "models":
            return lambda host, *, timeout_s=30.0: self.models
        return super().resolve(key)


def plan_speech(public_text: str) -> dict:
    return {
        "probabilities": {"speech_only": 0.9, "action_only": 0.05, "speech_and_action": 0.05},
        "public_text": public_text,
        "segments": [],
        "send_count": 0,
    }


def plan_action(segments: list[str], send_count: int | None = None) -> dict:
    return {
        "probabilities": {"speech_only": 0.05, "action_only": 0.9, "speech_and_action": 0.05},
        "public_text": None,
        "segments": segments,
        "send_count": len(segments) if send_count is None else send_count,
    }


def plan_both(public_text: str, segments: list[str], send_count: int | None = None) -> dict:
    return {
        "probabilities": {"speech_only": 0.05, "action_only": 0.05, "speech_and_action": 0.9},
        "public_text": public_text,
        "segments": segments,
        "send_count": len(segments) if send_count is None else send_count,
    }


class UpperStackAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    """Full AEB stack, real sockets, scripted models."""

    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmp.name)
        previous = os.environ.get("ASTRBOTEX_TASK_DB_PATH")
        os.environ["ASTRBOTEX_TASK_DB_PATH"] = str(self.data_root / "tasks.sqlite3")
        if previous is None:
            self.addCleanup(os.environ.pop, "ASTRBOTEX_TASK_DB_PATH", None)
        else:
            self.addCleanup(os.environ.__setitem__, "ASTRBOTEX_TASK_DB_PATH", previous)
        # AEB and EX must agree on the session id; the plugin reads it from config.
        previous_session = os.environ.get("ASTRBOTEX_SESSION_ID")
        os.environ["ASTRBOTEX_SESSION_ID"] = EX_SESSION
        if previous_session is None:
            self.addCleanup(os.environ.pop, "ASTRBOTEX_SESSION_ID", None)
        else:
            self.addCleanup(os.environ.__setitem__, "ASTRBOTEX_SESSION_ID", previous_session)
        self.control_port, self.text_port, self.audio_port, self.vision_port = reserve_ports(4)
        self.models = ScriptedModels({})
        self.public_sink: list[str] = []
        self.peer: LoopbackExPeer | None = None
        await self.boot()

    async def boot(self, config: dict | None = None) -> None:
        # A published catalog is a real dependency: the frozen batch carries its
        # version, and an empty version is not a wire id. Seeding it here keeps
        # the acceptance run honest instead of stubbing the version string.
        self.seed_catalog()
        self.plugin = plugin_module.AstrBotEXInteractionPlugin(
            UpperTestContext(),
            HostConfig(config) if config is not None else HostConfig({
                "upper_enabled": True,
                "control_enabled": True,
                "control_port": self.control_port,
                "control_bind_host": "127.0.0.1",
                "control_peer_id": CONTROL_PEER,
                "robot_id": "robot-e2e",
                "control_route_ref": "upper-control",
                "model_timeout_s": 5.0,
            }),
        )
        self.plugin._upper_data_root_override = self.data_root
        self.plugin.zmq_bind_host = "127.0.0.1"
        self.plugin.text_port = self.text_port
        self.plugin.audio_port = self.audio_port
        self.plugin.vision_port = self.vision_port
        self.plugin.upper_factory = ScriptedFactory(
            "astrbot_plugin_astrbotex_interaction", self.models
        )
        await self.plugin.initialize()

    def seed_catalog(self) -> None:
        from astrbot_plugin_astrbotex_interaction.action_catalog import ActionCatalog

        catalog = ActionCatalog(self.data_root)
        if catalog.refs():
            return
        catalog.save(
            {
                "action_ref": "pick_cup",
                "level": "basic",
                "name": "拿杯子",
                "description": "拿起目标杯子",
                "action_id": "ex.pick",
                "parameters": [],
                "usage_stages": [],
                "skill_markdown": "## 拿杯子\n靠近并抓取目标杯子。",
                "enabled": True,
            },
            expected_revision=0,
        )

    async def attach_peer(self, **kwargs: Any) -> LoopbackExPeer:
        self.peer = LoopbackExPeer(
            self.control_port, identity=CONTROL_PEER, ex_session=EX_SESSION, **kwargs
        )
        await self.peer.connect()
        for _ in range(200):
            if self.plugin._upper is not None and self.plugin._upper.control.peer in (
                self.plugin._upper.control.online_peers()
            ):
                return self.peer
            await asyncio.sleep(0.01)
        self.fail("loopback EX peer never completed hello")

    async def asyncTearDown(self) -> None:
        if self.peer is not None:
            await self.peer.close()
        await self.plugin.terminate()
        self.tmp.cleanup()

    @property
    def components(self):
        return self.plugin._upper

    def injects(self) -> list[dict]:
        return [] if self.peer is None else [item["payload"] for item in self.peer.injects()]

    async def deliver(self, text: str, plan: dict, *, message_id: str | None = None):
        self.models.script[text] = plan
        item = make_event(text, message_id=message_id)

        async def send(chain):
            self.public_sink.append(chain.get_plain_text())

        item.send = send
        await self.plugin.take_over_upper_input(item)
        return item

    def assert_no_cross_leak(self) -> None:
        control_payload = json.dumps(self.injects(), ensure_ascii=False)
        public_payload = "\n".join(self.public_sink)
        for ordinal in range(8):
            sentinel = exec_sentinel(ordinal)
            self.assertNotIn(sentinel, public_payload,
                             "execution sentinel leaked to a chat/TTS surface")
        if self.public_sink:
            self.assertNotIn(PUBLIC_SENTINEL, control_payload,
                             "public sentinel leaked into control")
        self.assertNotIn("public_text", control_payload)

    # -- route matrix ------------------------------------------------------

    async def test_speech_only_sends_public_once_and_never_injects(self) -> None:
        await self.attach_peer()
        await self.deliver("只说话", plan_speech(f"{PUBLIC_SENTINEL} 收到。"))

        self.assertEqual(self.public_sink, [f"{PUBLIC_SENTINEL} 收到。"])
        self.assertEqual(self.injects(), [])
        self.assert_no_cross_leak()

    async def test_action_only_injects_without_public_output(self) -> None:
        await self.attach_peer()
        await self.deliver("只执行", plan_action([f"寻找目标 {exec_sentinel(0)}。"]))

        self.assertEqual(self.public_sink, [])
        sent = self.injects()
        self.assertEqual(len(sent), 1)
        self.assertEqual([s["text"] for s in sent[0]["segments"]],
                         [f"寻找目标 {exec_sentinel(0)}。"])
        self.assert_no_cross_leak()

    async def test_speech_and_action_uses_both_surfaces_once(self) -> None:
        await self.attach_peer()
        await self.deliver(
            "又说又做",
            plan_both(f"{PUBLIC_SENTINEL} 我来处理。", [f"接近目标 {exec_sentinel(0)}。"]),
        )

        self.assertEqual(len(self.public_sink), 1)
        self.assertEqual(len(self.injects()), 1)
        self.assert_no_cross_leak()

    # -- segment matrix ----------------------------------------------------

    async def test_s1_single_segment_with_three_actions_is_not_split(self) -> None:
        await self.attach_peer()
        segment = f"寻找 {exec_sentinel(0)}，对齐 {exec_sentinel(1)}，抓取 {exec_sentinel(2)}。"
        await self.deliver("S1", plan_action([segment]))

        segments = self.injects()[0]["segments"]
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], segment)
        self.assert_no_cross_leak()

    async def test_s2_three_segments_travel_in_one_batch(self) -> None:
        await self.attach_peer()
        segments = [f"第{index}段 {exec_sentinel(index)}。" for index in range(3)]
        await self.deliver("S2", plan_action(segments))

        self.assertEqual(len(self.injects()), 1)
        sent = self.injects()[0]["segments"]
        self.assertEqual([s["text"] for s in sent], segments)
        self.assertEqual([s["ordinal"] for s in sent], [0, 1, 2])

    async def test_s3_prefix_only_keeps_suffix_pending(self) -> None:
        await self.attach_peer()
        segments = [f"第{index}段 {exec_sentinel(index)}。" for index in range(3)]
        await self.deliver("S3", plan_action(segments, send_count=1))

        sent = self.injects()[0]["segments"]
        self.assertEqual([s["text"] for s in sent], segments[:1])
        plan = self.components.store.list_plans(limit=1)[0]
        self.assertEqual(plan["send_count"], 1)
        # The full plan stays local: a later batch continues the same plan.
        self.assertEqual(plan["segment_count"], 3)
        self.assert_no_cross_leak()

    async def test_s4_two_of_three_segments(self) -> None:
        await self.attach_peer()
        segments = [f"第{index}段 {exec_sentinel(index)}。" for index in range(3)]
        await self.deliver("S4", plan_action(segments, send_count=2))

        self.assertEqual([s["text"] for s in self.injects()[0]["segments"]], segments[:2])
        self.assertEqual([s["ordinal"] for s in self.injects()[0]["segments"]], [0, 1])

    # -- idempotency and lifecycle ----------------------------------------

    async def test_replayed_event_produces_one_plan_and_one_delivery(self) -> None:
        await self.attach_peer()
        item = await self.deliver(
            "重放",
            plan_both(f"{PUBLIC_SENTINEL} 收到。", [f"执行 {exec_sentinel(0)}。"]),
        )
        for _ in range(19):
            await self.plugin.take_over_upper_input(item)

        self.assertEqual(self.models.classify_calls, ["重放"])
        self.assertEqual(len(self.models.generate_calls), 1)
        self.assertEqual(len(self.components.store.list_plans()), 1)
        self.assertEqual(len(self.injects()), 1)
        self.assertEqual(len(self.public_sink), 1)

    async def test_same_message_id_from_a_new_event_is_not_regenerated(self) -> None:
        await self.attach_peer()
        plan = plan_speech(f"{PUBLIC_SENTINEL} 收到。")
        await self.deliver("同一条消息", plan, message_id="host-message-fixed")
        await self.deliver("同一条消息", plan, message_id="host-message-fixed")

        self.assertEqual(len(self.models.classify_calls), 1)
        self.assertEqual(len(self.public_sink), 1)

    async def test_feature_off_produces_zero_classification_and_zero_injection(self) -> None:
        await self.plugin.terminate()
        await self.boot(config={"upper_enabled": False})
        self.plugin._upper_data_root_override = self.data_root / "off"
        item = make_event("关闭功能后仍应走旧链路")
        sent: list[str] = []

        async def send(chain):
            sent.append(chain.get_plain_text())

        item.send = send
        await self.plugin.take_over_upper_input(item)

        self.assertEqual(sent, [])
        self.assertEqual(self.models.classify_calls, [])
        self.assertFalse(item.is_stopped())
        self.assertEqual(self.plugin.upper_status["reason"], "upper_disabled")
        self.assertNotIn("control", self.plugin._transport_status()["channels"])

    async def test_terminate_releases_control_and_invalidates_generation(self) -> None:
        await self.attach_peer()
        item = await self.deliver("重载前", plan_speech(f"{PUBLIC_SENTINEL} 收到。"))
        sender = self.plugin.upper_public_sender_for(item, "platform")
        control = self.components.control

        await self.plugin.terminate()

        self.assertFalse(control.is_running)
        self.assertFalse(await sender("迟到输出"))
        self.assertEqual(len(self.public_sink), 1)
        self.assertFalse(self.plugin.upper_ready)


    async def test_management_api_is_available_before_enabling_or_configuring_robot(self):
        from astrbot_plugin_astrbotex_interaction.tests.test_upper_composition import RecordingContext
        await self.plugin.terminate()
        self.plugin.plugin_config.update(upper_enabled=False, control_enabled=False,
                                         robot_id="", control_peer_id="")
        self.plugin.upper_settings = plugin_module.UpperSettings.from_config(
            self.plugin.plugin_config, environ={})
        context = RecordingContext()
        self.plugin.context = context
        await self.plugin.initialize()
        self.assertEqual(len(context.web_apis), 10)
        self.assertTrue(self.plugin.upper_status["api_registered"])
        self.assertFalse(self.plugin.upper_ready)
        self.assertNotIn("control", self.plugin._transport_status()["channels"])
        self.assertEqual(self.models.classify_calls, [])
        defaults = self.plugin.upper_prompt_defaults()
        self.assertTrue(defaults["routing"])
        self.assertTrue(defaults["planning"])

    async def test_provider_save_applies_to_next_input_but_not_inflight_input(self):
        classify_started, release = asyncio.Event(), asyncio.Event()
        calls = []
        classify, generate = self.models.classify, self.models.generate

        async def record_classify(text, prompt, provider_id=""):
            calls.append((text, "routing", provider_id))
            if text == "inflight":
                classify_started.set()
                await release.wait()
            return await classify(text, prompt, provider_id)

        async def record_generate(text, route, prompt, provider_id=""):
            calls.append((text, "generation", provider_id))
            return await generate(text, route, prompt, provider_id)

        self.models.classify, self.models.generate = record_classify, record_generate
        first = asyncio.create_task(self.deliver("inflight", plan_speech("first reply")))
        await asyncio.wait_for(classify_started.wait(), 2)
        repository = self.components.settings_repository
        repository.save({"routing_provider_id": "new-routing", "generation_provider_id": "new-generation"},
                        repository.load().revision)
        release.set()
        await first
        await self.deliver("next-input", plan_speech("next reply"))
        self.assertEqual(calls, [("inflight", "routing", ""), ("inflight", "generation", ""),
                                 ("next-input", "routing", "new-routing"),
                                 ("next-input", "generation", "new-generation")])

    async def test_voice_route_matrix_keeps_text_peer_separate_from_control_peer(self):
        from astrbot_plugin_astrbotex_interaction.tests.test_upper_composition import ex_event
        await self.attach_peer()
        replies = []
        request = self.plugin.text_channel.request

        async def capture_reply(method, payload, **kwargs):
            if method == "interaction.reply":
                replies.append((payload, kwargs["peer"]))
                return SimpleNamespace(payload={"ok": True})
            return await request(method, payload, **kwargs)

        self.plugin.text_channel.request = capture_reply
        cases = [plan_speech("voice-public-1"), plan_action(["voice-exec-2"]),
                 plan_both("voice-public-3", ["voice-exec-3"])]
        for index, plan in enumerate(cases):
            with self.subTest(route=index):
                text = f"voice-input-{index}"
                self.models.script[text] = plan
                event = ex_event(text, peer=b"separate-text-peer", adapter=self.plugin._adapter)
                event.session_id = EX_SESSION
                await self.plugin.take_over_upper_input(event)
        self.assertEqual(len(self.injects()), 2)
        self.assertEqual([item[0]["text"] for item in replies], ["voice-public-1", "voice-public-3"])
        self.assertTrue(all(peer == b"separate-text-peer" for _, peer in replies))
        self.assertTrue(all("voice-exec" not in item[0]["text"] for item in replies))
        self.assertTrue(all("public" not in json.dumps(item) for item in self.injects()))

    async def test_saved_enable_flag_admits_next_input_without_rebinding_connection_settings(self):
        await self.plugin.terminate()
        self.plugin.plugin_config["upper_enabled"] = False
        self.plugin.upper_settings = plugin_module.UpperSettings.from_config(
            self.plugin.plugin_config, environ={})
        await self.plugin.initialize()
        self.assertFalse(self.plugin.upper_ready)
        self.plugin.plugin_config["upper_enabled"] = True
        await self.deliver("enabled-next-input", plan_speech("enabled reply"))
        self.assertTrue(self.plugin.upper_ready)
        self.assertEqual(self.public_sink, ["enabled reply"])
        original_port = self.plugin.upper_settings.control_port
        self.components.settings_repository.save(
            {"control_port": original_port + 1}, self.components.settings_repository.load().revision)
        await self.deliver("same-listener", plan_speech("same listener"))
        self.assertEqual(self.plugin.upper_settings.control_port, original_port)

    async def test_disabling_suppresses_late_model_output_and_restores_normal_admission(self):
        entered, release = asyncio.Event(), asyncio.Event()
        generate = self.models.generate

        async def slow_generate(*args, **kwargs):
            entered.set()
            await release.wait()
            return await generate(*args, **kwargs)

        self.models.generate = slow_generate
        task = asyncio.create_task(self.deliver("late-disabled", plan_speech("must stay private")))
        await asyncio.wait_for(entered.wait(), 2)
        self.components.settings_repository.save({"upper_enabled": False},
                                                 self.components.settings_repository.load().revision)
        release.set()
        await task
        self.assertEqual(self.public_sink, [])
        item = make_event("normal-chat-again")
        await self.plugin.take_over_upper_input(item)
        self.assertFalse(item.get_extra("upper_handled", False))
        self.assertFalse(item.is_stopped())


class RealManagementCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_catalog_create_reload_edit_conflict_delete(self):
        from astrbot_plugin_astrbotex_interaction.action_catalog import ActionCatalog
        from astrbot_plugin_astrbotex_interaction.tests.test_upper_api import (
            FakeRequest, RecordingResponder, valid_record)
        from astrbot_plugin_astrbotex_interaction.upper_api import UpperApi, UpperApiDependencies
        with tempfile.TemporaryDirectory() as tmp:
            def api(body=None):
                return UpperApi(UpperApiDependencies(actions=plugin_module._UpperActionRepository(
                    ActionCatalog(tmp))), request_proxy=FakeRequest(body or {}, {}),
                    responder=RecordingResponder())
            created = await api({"record": valid_record(), "expected_revision": 0}).save_action()
            self.assertEqual(created.status_code, 200, created.body)
            ref = created.body["action_ref"]
            self.assertTrue(ref.startswith("aeb-"))
            listed = await api().get_actions()
            self.assertEqual(listed.body["actions"][0]["action_ref"], ref)
            record = dict(listed.body["actions"][0], name="edited", enabled=False)
            edited = await api({"record": record, "expected_revision": 1}).save_action()
            self.assertEqual(edited.status_code, 200, edited.body)
            self.assertEqual(edited.body["action_ref"], ref)
            self.assertEqual(edited.body["action"]["revision"], 2)
            conflict = await api({"record": record, "expected_revision": 1}).save_action()
            self.assertEqual(conflict.status_code, 409)
            deleted = await api({"action_ref": ref, "expected_revision": 2}).delete_action()
            self.assertEqual(deleted.status_code, 200, deleted.body)
            self.assertEqual((await api().get_actions()).body["actions"], [])



class VerifyRunnerTests(unittest.TestCase):
    """The acceptance runner must fail loudly, never default to success."""

    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_upper_control.py"

    def invoke(self, *args: str):
        import subprocess

        return subprocess.run(
            [sys.executable, "-B", str(self.script), *args],
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_absent_dependencies_exit_nonzero_and_report_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_aeb = Path(tmp) / "aeb"
            empty_ex = Path(tmp) / "ex"
            empty_aeb.mkdir()
            empty_ex.mkdir()
            evidence = Path(tmp) / "evidence"
            completed = self.invoke(
                "--aeb-root", str(empty_aeb),
                "--ex-root", str(empty_ex),
                "--evidence-dir", str(evidence),
                "--scope", "unit",
                "--python", sys.executable,
            )

            self.assertNotEqual(completed.returncode, 0)
            results = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(results["verdict"], "partial")
            self.assertEqual(results["stages"][0]["status"], "NOT_RUN")
            self.assertIn("missing_dependencies", results["stages"][0]["reason"])
            self.assertIn("missing_dependencies",
                          (evidence / "report.md").read_text(encoding="utf-8"))

    def test_existing_evidence_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "evidence"
            evidence.mkdir()
            (evidence / "keep.txt").write_text("previous run", encoding="utf-8")
            completed = self.invoke(
                "--aeb-root", tmp,
                "--ex-root", tmp,
                "--evidence-dir", str(evidence),
                "--scope", "unit",
            )

            self.assertEqual(completed.returncode, 2)
            self.assertEqual((evidence / "keep.txt").read_text(encoding="utf-8"), "previous run")


if __name__ == "__main__":
    unittest.main()
