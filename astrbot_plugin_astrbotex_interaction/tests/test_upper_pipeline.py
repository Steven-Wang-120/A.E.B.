"""B05 pipeline unit tests with substituted providers and senders.

Everything downstream of the pipeline is a spy here: models, prompts, settings,
catalog, control sender and public sender.  That is deliberate -- B05's job is
the wiring contract, and the acceptance text says spies are ``unit_fake`` only.
The real ROUTER/DEALER and real Host link are verified by B11 / A.md.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from astrbot_plugin_astrbotex_interaction.output_router import UpperOutputRouter
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
from astrbot_plugin_astrbotex_interaction.upper_pipeline import UpperPipeline, argmax_route
from astrbot_plugin_astrbotex_interaction.upper_store import UpperStore

SPEECH = {"speech_only": 0.9, "speech_and_action": 0.05, "action_only": 0.05}
ACTION = {"speech_only": 0.05, "speech_and_action": 0.05, "action_only": 0.9}
BOTH = {"speech_only": 0.05, "speech_and_action": 0.9, "action_only": 0.05}

PUBLIC_TEXT = "收到，我来处理。"
SEGMENTS = ("寻找红色杯子。", "接近并对齐同一个目标。", "按已配置方案抓取同一个目标。")


class FakeModels:
    def __init__(self, probabilities=None, public_text=PUBLIC_TEXT, segments=SEGMENTS,
                 send_count=2, classify_error=None, generate_error=None,
                 extra_route=None):
        self.probabilities = probabilities or BOTH
        self.public_text = public_text
        self.segments = list(segments)
        self.send_count = send_count
        self.classify_error = classify_error
        self.generate_error = generate_error
        self.extra_route = extra_route
        self.classify_calls: list[dict] = []
        self.generate_calls: list[dict] = []

    async def classify(self, input_text, prompt_snapshot, provider_id):
        self.classify_calls.append({"input_text": input_text, "provider_id": provider_id,
                                    "stream": False, "prompt": prompt_snapshot})
        if self.classify_error:
            raise self.classify_error
        return dict(self.probabilities)

    async def generate(self, input_text, route, prompt_snapshot, provider_id):
        self.generate_calls.append({"input_text": input_text, "route": route,
                                    "provider_id": provider_id, "stream": False,
                                    "prompt": prompt_snapshot})
        if self.generate_error:
            raise self.generate_error
        payload = {"public_text": self.public_text, "segments": self.segments,
                   "send_count": self.send_count}
        if self.extra_route:
            # A model that tries to pick its own route must be ignored.
            payload["route"] = self.extra_route
        return payload


class FakePrompts:
    def build(self, settings_snapshot, catalog_snapshot):
        return {"routing_prompt": "routing", "planning_prompt": "planning",
                "catalog_version": "catalog-1", "prompt_hash": "prompt-1",
                "settings_hash": "settings-1"}


class FakeSettings:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def load(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("config missing")
        return {"upper_enabled": True, "control_port": 8769}


class FakeCatalog:
    def snapshot(self):
        return {"version": "catalog-1", "actions": []}


class SpyControl:
    def __init__(self, status="injected", capabilities_error=None, inject_error=None,
                 ex_session="ex-1", input_ready=True):
        self.status = status
        self.capabilities_error = capabilities_error
        self.inject_error = inject_error
        self.ex_session = ex_session
        self.input_ready = input_ready
        self.capabilities_calls: list[bytes] = []
        self.inject_calls: list[tuple[bytes, dict]] = []

    async def capabilities(self, peer):
        self.capabilities_calls.append(peer)
        if self.capabilities_error:
            raise self.capabilities_error
        return {"ex_session": self.ex_session, "input_ready": self.input_ready,
                "methods": ["upper.capabilities.get", "upper.steps.inject"]}

    async def inject(self, peer, batch):
        self.inject_calls.append((peer, batch))
        if self.inject_error:
            raise self.inject_error
        return {"status": self.status,
                "error_code": None if self.status in {"received", "injected"} else self.status,
                "batch_id": batch["batch_id"], "sent": True}


class SpyPublicSender:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls: list[str] = []

    async def __call__(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("platform send failed")


class SpySurface:
    """Every other place execution text could leak out of."""

    def __init__(self):
        self.onebot = []
        self.host_auto = []
        self.tts = []
        self.interaction_reply = []


def trusted(message_id="m1", platform="onebot", origin="group:1", text="把红杯子拿过来",
            robot_id="r1", trusted_flag=True, sender="event-object"):
    return {
        "trusted": trusted_flag,
        "platform": platform,
        "origin": origin,
        "source_message_id": message_id,
        "robot_id": robot_id,
        "user_id": "u1",
        "session_id": "s1",
        "input_text": text,
        "route_ref": "route-1",
        "sender": sender,
    }


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fresh()

    def fresh(self):
        """Isolated database for one turn (the matrix test needs several)."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = TaskStore(str(Path(self.tmp.name) / "tasks.db"))
        self.addCleanup(self.store.close)
        self.upper = UpperStore(self.store)

    def pipeline(self, models=None, control=None, settings=None, **kwargs):
        self.models = models or FakeModels()
        self.control = control or SpyControl()
        self.router = UpperOutputRouter(self.upper)
        self.pipeline_obj = UpperPipeline(
            store=self.upper, models=self.models, prompts=FakePrompts(),
            settings=settings or FakeSettings(), catalog=FakeCatalog(),
            control=self.control, router=self.router, robot_id="r1",
            peer_id="ex-control", route_ref="route-1", **kwargs)
        return self.pipeline_obj

    async def run_turn(self, pipeline, sender, message_id="m1", **kwargs):
        return await pipeline.process(trusted(message_id=message_id, **kwargs),
                                      public_sender=sender)

    # ------------------------------------------------------------------
    async def test_three_routes_produce_the_documented_exit_counts(self):
        cases = {
            "speech_only": (SPEECH, dict(public_text=PUBLIC_TEXT, segments=(), send_count=0),
                            1, 0),
            "action_only": (ACTION, dict(public_text=None, segments=SEGMENTS, send_count=3),
                            0, 1),
            "speech_and_action": (BOTH, dict(public_text=PUBLIC_TEXT, segments=SEGMENTS,
                                             send_count=2), 1, 1),
        }
        for index, (route, (probabilities, generated, public_count, control_count)) in enumerate(
                cases.items()):
            with self.subTest(route=route):
                self.fresh()
                pipeline = self.pipeline(FakeModels(probabilities=probabilities, **generated))
                sender = SpyPublicSender()
                result = await self.run_turn(pipeline, sender, message_id=f"m-{index}")
                self.assertTrue(result["handled"])
                self.assertTrue(result["suppress_default"])
                self.assertEqual(result["route"], route)
                self.assertEqual(len(sender.calls), public_count)
                self.assertEqual(len(self.control.inject_calls), control_count)
                self.assertEqual(result["public_sent"], bool(public_count))
                self.assertIsNone(result["error_code"])
                plan = self.upper.get_plan(result["plan_id"])
                self.assertEqual(plan["route"], route)
                self.assertEqual([s["status"] for s in plan["segments"]][:control_count],
                                 ["injected"] * control_count)

    async def test_send_count_two_of_three_leaves_the_suffix_pending(self):
        pipeline = self.pipeline(FakeModels(probabilities=BOTH, send_count=2))
        result = await self.run_turn(pipeline, SpyPublicSender())
        plan = self.upper.get_plan(result["plan_id"])
        self.assertEqual([s["status"] for s in plan["segments"]],
                         ["injected", "injected", "pending"])
        _, batch = self.control.inject_calls[0]
        self.assertEqual([s["ordinal"] for s in batch["payload"]["segments"]], [0, 1])
        self.assertEqual(batch["first_ordinal"], 0)
        self.assertEqual(batch["last_ordinal"], 1)

    async def test_model_supplied_route_is_ignored(self):
        pipeline = self.pipeline(FakeModels(probabilities=SPEECH, public_text=PUBLIC_TEXT,
                                            segments=(), send_count=0,
                                            extra_route="action_only"))
        result = await self.run_turn(pipeline, SpyPublicSender())
        self.assertEqual(result["route"], "speech_only")
        self.assertEqual(self.control.inject_calls, [])
        self.assertEqual(self.upper.get_plan(result["plan_id"])["route"], "speech_only")

    async def test_tie_break_follows_c01_order(self):
        tie = {"speech_only": 1 / 3, "speech_and_action": 1 / 3, "action_only": 1 / 3}
        self.assertEqual(argmax_route(tie), "speech_only")
        self.assertEqual(argmax_route({"speech_only": 0.1, "speech_and_action": 0.45,
                                       "action_only": 0.45}), "speech_and_action")

    async def test_duplicate_host_hook_never_generates_or_sends_twice(self):
        pipeline = self.pipeline()
        sender = SpyPublicSender()
        first = await self.run_turn(pipeline, sender)
        second = await self.run_turn(pipeline, sender)
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["plan_id"], first["plan_id"])
        self.assertEqual(len(self.models.classify_calls), 1)
        self.assertEqual(len(self.models.generate_calls), 1)
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(len(self.control.inject_calls), 1)

    async def test_untrusted_identity_is_refused_before_any_io(self):
        pipeline = self.pipeline()
        for broken in (
            trusted(trusted_flag=False),
            trusted(sender=None),
            trusted(robot_id="r2"),
            trusted(text="   "),
            trusted(platform=""),
            trusted(message_id=""),
        ):
            result = await pipeline.process(broken, public_sender=SpyPublicSender())
            self.assertFalse(result["handled"])
            self.assertFalse(result["suppress_default"])
            self.assertEqual(result["error_code"], "invalid_result")
        self.assertEqual(self.models.classify_calls, [])
        self.assertEqual(self.control.inject_calls, [])
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM upper_inputs").fetchone()[0], 0)

    async def test_b00_trusted_input_shape_is_accepted(self):
        """B00's TrustedUpperInput carries input_key/_marker/ex_session, no platform."""
        from types import SimpleNamespace

        marker = object()
        b00_input = SimpleNamespace(
            _marker=marker, input_key="b00-input-key-1", origin="group:1",
            robot_id="r1", user_id="u1", session_id="s1", route_ref="route-1",
            ex_session="ex-from-trusted", input_text="把红杯子拿过来",
        )
        pipeline = self.pipeline()
        sender = SpyPublicSender()
        result = await pipeline.process(b00_input, public_sender=sender)
        self.assertTrue(result["handled"], result)
        self.assertEqual(result["route"], "speech_and_action")
        self.assertEqual(len(sender.calls), 1)
        plan = self.upper.get_plan(result["plan_id"])
        self.assertEqual(plan["input_key"], "b00-input-key-1")
        # The session comes from the trusted input, not from a capabilities read.
        self.assertEqual(self.control.capabilities_calls, [])
        self.assertEqual(self.control.inject_calls[0][1]["ex_session"], "ex-from-trusted")

    async def test_hand_built_b00_input_without_marker_is_refused(self):
        from types import SimpleNamespace

        forged = SimpleNamespace(
            _marker=None, input_key="forged", origin="group:1", robot_id="r1",
            user_id="u1", session_id="s1", route_ref="route-1", ex_session="ex-1",
            input_text="do something",
        )
        pipeline = self.pipeline()
        result = await pipeline.process(forged, public_sender=SpyPublicSender())
        self.assertEqual(result["error_code"], "invalid_result")
        self.assertFalse(result["handled"])
        self.assertEqual(self.models.classify_calls, [])

    async def test_decision_failure_sends_nothing(self):
        pipeline = self.pipeline(FakeModels(classify_error=RuntimeError("provider down")))
        sender = SpyPublicSender()
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["error_code"], "decision_error")
        self.assertFalse(result["handled"])
        self.assertEqual(sender.calls, [])
        self.assertEqual(self.control.inject_calls, [])
        self.assertEqual(self.models.generate_calls, [])

    async def test_generation_failure_sends_nothing(self):
        pipeline = self.pipeline(FakeModels(generate_error=RuntimeError("timeout")))
        sender = SpyPublicSender()
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["error_code"], "generation_error")
        self.assertEqual(sender.calls, [])
        self.assertEqual(self.control.inject_calls, [])

    async def test_invalid_probabilities_send_nothing(self):
        pipeline = self.pipeline(FakeModels(probabilities={"speech_only": 0.9,
                                                           "speech_and_action": 0.9,
                                                           "action_only": 0.9}))
        sender = SpyPublicSender()
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["error_code"], "invalid_probabilities")
        self.assertEqual(sender.calls, [])
        self.assertEqual(self.models.generate_calls, [])

    async def test_route_mismatch_from_the_model_sends_nothing(self):
        # action_only with a public_text is a contract violation, not a fallback.
        pipeline = self.pipeline(FakeModels(probabilities=ACTION, public_text="我做了",
                                            segments=SEGMENTS, send_count=2))
        sender = SpyPublicSender()
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["error_code"], "route_mismatch")
        self.assertEqual(sender.calls, [])
        self.assertEqual(self.control.inject_calls, [])

    async def test_config_failure_happens_before_the_model(self):
        pipeline = self.pipeline(settings=FakeSettings(fail=True))
        sender = SpyPublicSender()
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["error_code"], "config_invalid")
        self.assertEqual(self.models.classify_calls, [])
        self.assertEqual(sender.calls, [])

    async def test_stale_generation_blocks_late_sends(self):
        alive = {"value": True}
        pipeline = self.pipeline(alive=lambda: alive["value"])

        original_generate = self.models.generate

        async def generate_then_die(*args, **kwargs):
            alive["value"] = False
            return await original_generate(*args, **kwargs)

        self.models.generate = generate_then_die
        sender = SpyPublicSender()
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["error_code"], "generation_stale")
        self.assertEqual(sender.calls, [])
        self.assertEqual(self.control.inject_calls, [])

    async def test_dead_plugin_before_the_turn_sends_nothing(self):
        pipeline = self.pipeline(alive=lambda: False)
        result = await self.run_turn(pipeline, SpyPublicSender())
        self.assertEqual(result["error_code"], "generation_stale")
        self.assertFalse(result["handled"])
        self.assertEqual(self.models.classify_calls, [])

    async def test_busy_old_task_refuses_the_action_but_still_speaks(self):
        pipeline = self.pipeline(active_control=lambda: True)
        sender = SpyPublicSender()
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["error_code"], "input_busy")
        self.assertEqual(self.control.inject_calls, [])
        self.assertEqual(sender.calls, [PUBLIC_TEXT])
        self.assertTrue(result["handled"])

    async def test_control_failure_still_sends_the_public_text_once(self):
        pipeline = self.pipeline(control=SpyControl(status="delivery_unknown"))
        sender = SpyPublicSender()
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["control_status"], "delivery_unknown")
        self.assertEqual(result["error_code"], "delivery_unknown")
        self.assertEqual(sender.calls, [PUBLIC_TEXT])
        plan = self.upper.get_plan(result["plan_id"])
        self.assertEqual(plan["batch"]["status"], "delivery_unknown")

    async def test_control_unavailable_does_not_retry_and_keeps_pending(self):
        pipeline = self.pipeline(control=SpyControl(capabilities_error=TaskError("peer_unavailable")))
        sender = SpyPublicSender()
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["error_code"], "peer_unavailable")
        self.assertEqual(self.control.inject_calls, [])
        self.assertEqual(sender.calls, [PUBLIC_TEXT])
        plan = self.upper.get_plan(result["plan_id"])
        self.assertIsNone(plan["batch"])
        self.assertEqual([s["status"] for s in plan["segments"]],
                         ["pending", "pending", "pending"])

    async def test_public_failure_never_retries_the_control_batch(self):
        pipeline = self.pipeline()
        sender = SpyPublicSender(fail=True)
        result = await self.run_turn(pipeline, sender)
        self.assertEqual(result["control_status"], "injected")
        self.assertFalse(result["public_sent"])
        self.assertEqual(result["public_error"], "delivery_unknown")
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(len(self.control.inject_calls), 1)
        # A second hook for the same message must not re-send anything.
        again = await self.run_turn(pipeline, sender)
        self.assertTrue(again["duplicate"])
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(len(self.control.inject_calls), 1)

    async def test_control_payload_never_contains_public_text(self):
        pipeline = self.pipeline()
        await self.run_turn(pipeline, SpyPublicSender())
        _, batch = self.control.inject_calls[0]
        blob = repr(batch["payload"])
        self.assertNotIn(PUBLIC_TEXT, blob)
        self.assertNotIn("public_text", blob)
        self.assertEqual(batch["payload"]["visibility"], "decision_only")

    async def test_ex_voice_source_uses_the_same_single_public_exit(self):
        pipeline = self.pipeline()
        sender = SpyPublicSender()
        result = await pipeline.process(
            trusted(platform="astrbotex", origin="voice:front", message_id="voice-1"),
            public_sender=sender)
        self.assertTrue(result["handled"])
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(len(self.control.inject_calls), 1)
        self.assertEqual(self.control.inject_calls[0][1]["payload"]["ex_session"], "ex-1")

    async def test_prompt_and_catalog_hashes_are_frozen_into_the_plan(self):
        pipeline = self.pipeline()
        result = await self.run_turn(pipeline, SpyPublicSender())
        plan = self.upper.get_plan(result["plan_id"])
        self.assertEqual(plan["catalog_hash"], "catalog-1")
        self.assertEqual(plan["prompt_hash"], "prompt-1")
        self.assertEqual(plan["settings_hash"], "settings-1")
        self.assertEqual(self.models.classify_calls[0]["prompt"]["routing_prompt"], "routing")


if __name__ == "__main__":
    unittest.main()
