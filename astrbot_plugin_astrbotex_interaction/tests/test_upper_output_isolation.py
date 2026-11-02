"""B05 output isolation: execution text must never reach a public surface.

Each execution segment carries a unique sentinel.  The four surfaces a leak
would show up on -- OneBot platform send, the Host's automatic reply, TTS
request/playback, and ``interaction.reply`` -- are asserted at zero.  The
public text is asserted exactly once on the one public exit, and zero times in
the control payload.

These are unit-level spies; the real Host link is B11's acceptance run.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from astrbot_plugin_astrbotex_interaction.output_router import (
    UPPER_CONTROL, UPPER_PRIVATE, UPPER_PUBLIC, PublicOutputRouter, UpperOutputRouter,
)
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
from astrbot_plugin_astrbotex_interaction.upper_pipeline import UpperPipeline
from astrbot_plugin_astrbotex_interaction.upper_store import UpperStore

SENTINELS = (
    "SEG-A7f3-寻找红色杯子",
    "SEG-B91c-接近并对齐",
    "SEG-C2d0-抓取同一个目标",
)
PUBLIC_TEXT = "收到，我来处理。"

SPEECH = {"speech_only": 0.9, "speech_and_action": 0.05, "action_only": 0.05}
ACTION = {"speech_only": 0.05, "speech_and_action": 0.05, "action_only": 0.9}
BOTH = {"speech_only": 0.05, "speech_and_action": 0.9, "action_only": 0.05}


class HostSurface:
    """The four places an execution segment could escape to."""

    def __init__(self, fail_public=False):
        self.onebot: list[str] = []
        self.host_auto: list[str] = []
        self.tts_requests: list[str] = []
        self.tts_played: list[str] = []
        self.interaction_reply: list[str] = []
        self.fail_public = fail_public

    async def public_sender(self, text: str) -> None:
        self.onebot.append(text)
        if self.fail_public:
            raise RuntimeError("platform send failed")

    def total(self, needle: str) -> int:
        return sum(blob.count(needle) for blob in (
            *self.onebot, *self.host_auto, *self.tts_requests,
            *self.tts_played, *self.interaction_reply))


class FakeModels:
    def __init__(self, probabilities, public_text, segments, send_count):
        self.probabilities = probabilities
        self.public_text = public_text
        self.segments = list(segments)
        self.send_count = send_count
        self.classify_calls = 0
        self.generate_calls = 0

    async def classify(self, input_text, prompt_snapshot, provider_id):
        self.classify_calls += 1
        return dict(self.probabilities)

    async def generate(self, input_text, route, prompt_snapshot, provider_id):
        self.generate_calls += 1
        return {"public_text": self.public_text, "segments": self.segments,
                "send_count": self.send_count}


class FakePrompts:
    def build(self, settings_snapshot, catalog_snapshot):
        return {"routing_prompt": "r", "planning_prompt": "p",
                "catalog_version": "c", "prompt_hash": "ph", "settings_hash": "sh"}


class FakeSettings:
    def load(self):
        return {"upper_enabled": True}


class FakeCatalog:
    def snapshot(self):
        return {"version": "c"}


class SpyControl:
    def __init__(self, status="injected", error_code=None):
        self.status = status
        self.error_code = error_code
        self.inject_calls: list[dict] = []

    async def capabilities(self, peer):
        return {"ex_session": "ex-1", "input_ready": True}

    async def inject(self, peer, batch):
        self.inject_calls.append(batch)
        return {"status": self.status, "error_code": self.error_code,
                "batch_id": batch["batch_id"], "sent": True}


def trusted(message_id="m1", platform="onebot"):
    return {"trusted": True, "platform": platform, "origin": "group:1",
            "source_message_id": message_id, "robot_id": "r1", "user_id": "u1",
            "session_id": "s1", "input_text": "把红杯子拿过来",
            "route_ref": "route-1", "sender": object()}


class OutputIsolationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = TaskStore(str(Path(self.tmp.name) / "tasks.db"))
        self.addCleanup(self.store.close)
        self.upper = UpperStore(self.store)
        self.router = UpperOutputRouter(self.upper)

    def pipeline(self, probabilities, public_text, send_count, control=None,
                 segments=SENTINELS, **kwargs):
        self.control = control or SpyControl()
        return UpperPipeline(
            store=self.upper, models=FakeModels(probabilities, public_text, segments, send_count),
            prompts=FakePrompts(), settings=FakeSettings(), catalog=FakeCatalog(),
            control=self.control, router=self.router, robot_id="r1",
            peer_id="ex-control", route_ref="route-1", **kwargs)

    async def test_no_route_leaks_a_segment_to_a_public_surface(self):
        matrix = (
            ("speech_only", SPEECH, PUBLIC_TEXT, 0, 0, ()),
            ("action_only", ACTION, None, 3, 1, SENTINELS),
            ("speech_and_action", BOTH, PUBLIC_TEXT, 2, 1, SENTINELS),
        )
        for index, (route, probabilities, public_text, send_count, control_count,
                    segments) in enumerate(matrix):
            with self.subTest(route=route):
                surface = HostSurface()
                pipeline = self.pipeline(probabilities, public_text, send_count,
                                         segments=segments)
                result = await pipeline.process(trusted(f"m-{index}"),
                                                public_sender=surface.public_sender)
                self.assertEqual(result["route"], route)
                for sentinel in SENTINELS:
                    self.assertEqual(surface.total(sentinel), 0,
                                     f"{route} leaked {sentinel}")
                self.assertEqual(len(surface.tts_requests), 0)
                self.assertEqual(len(surface.tts_played), 0)
                self.assertEqual(len(surface.interaction_reply), 0)
                self.assertEqual(surface.host_auto, [])

                expected_public = 1 if public_text else 0
                self.assertEqual(surface.onebot.count(PUBLIC_TEXT), expected_public)
                self.assertEqual(len(self.control.inject_calls), control_count)
                for batch in self.control.inject_calls:
                    blob = repr(batch["payload"])
                    self.assertNotIn(PUBLIC_TEXT, blob)
                    self.assertNotIn("public_text", blob)

    async def test_segments_reach_the_control_payload_verbatim_and_ordered(self):
        pipeline = self.pipeline(BOTH, PUBLIC_TEXT, 2)
        await pipeline.process(trusted(), public_sender=HostSurface().public_sender)
        payload = self.control.inject_calls[0]["payload"]
        self.assertEqual([s["text"] for s in payload["segments"]], list(SENTINELS[:2]))
        self.assertEqual([s["ordinal"] for s in payload["segments"]], [0, 1])
        # The suffix never leaves the database this round.
        self.assertNotIn(SENTINELS[2], repr(payload))

    async def test_duplicate_callback_speaks_once(self):
        surface = HostSurface()
        pipeline = self.pipeline(BOTH, PUBLIC_TEXT, 2)
        first = await pipeline.process(trusted(), public_sender=surface.public_sender)
        for _ in range(3):
            await pipeline.process(trusted(), public_sender=surface.public_sender)
        self.assertEqual(surface.onebot, [PUBLIC_TEXT])
        self.assertEqual(len(self.control.inject_calls), 1)
        self.assertEqual(self.upper.get_plan(first["plan_id"])["public_reply"]["status"], "sent")

    async def test_public_timeout_is_recorded_once_and_never_re_sent(self):
        surface = HostSurface(fail_public=True)
        pipeline = self.pipeline(BOTH, PUBLIC_TEXT, 2)
        result = await pipeline.process(trusted(), public_sender=surface.public_sender)
        self.assertFalse(result["public_sent"])
        self.assertEqual(len(surface.onebot), 1)
        reply = self.upper.get_plan(result["plan_id"])["public_reply"]
        self.assertEqual(reply["status"], "delivery_unknown")
        await pipeline.process(trusted(), public_sender=surface.public_sender)
        self.assertEqual(len(surface.onebot), 1)

    async def test_stale_session_is_rejected_without_a_retry(self):
        surface = HostSurface()
        control = SpyControl(status="rejected", error_code="session_mismatch")
        pipeline = self.pipeline(BOTH, PUBLIC_TEXT, 2, control=control)
        result = await pipeline.process(trusted(), public_sender=surface.public_sender)
        self.assertEqual(result["control_status"], "rejected")
        self.assertEqual(result["error_code"], "session_mismatch")
        self.assertEqual(surface.onebot, [PUBLIC_TEXT])
        plan = self.upper.get_plan(result["plan_id"])
        self.assertEqual([s["status"] for s in plan["segments"]],
                         ["rejected", "rejected", "pending"])
        await pipeline.process(trusted(), public_sender=surface.public_sender)
        self.assertEqual(len(control.inject_calls), 1)
        self.assertEqual(len(surface.onebot), 1)

    async def test_late_result_after_plugin_close_sends_nothing(self):
        surface = HostSurface()
        alive = {"value": True}
        pipeline = self.pipeline(BOTH, PUBLIC_TEXT, 2, alive=lambda: alive["value"])
        models = pipeline.models

        original = models.generate

        async def generate_then_close(*args, **kwargs):
            result = await original(*args, **kwargs)
            alive["value"] = False
            return result

        models.generate = generate_then_close
        result = await pipeline.process(trusted(), public_sender=surface.public_sender)
        self.assertEqual(result["error_code"], "generation_stale")
        self.assertEqual(surface.onebot, [])
        self.assertEqual(self.control.inject_calls, [])

    async def test_busy_old_task_still_never_leaks_a_segment(self):
        surface = HostSurface()
        pipeline = self.pipeline(BOTH, PUBLIC_TEXT, 2, active_control=lambda: True)
        result = await pipeline.process(trusted(), public_sender=surface.public_sender)
        self.assertEqual(result["error_code"], "input_busy")
        self.assertEqual(self.control.inject_calls, [])
        self.assertEqual(surface.onebot, [PUBLIC_TEXT])
        for sentinel in SENTINELS:
            self.assertEqual(surface.total(sentinel), 0)


class UpperOutputRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = TaskStore(str(Path(self.tmp.name) / "tasks.db"))
        self.addCleanup(self.store.close)
        self.upper = UpperStore(self.store)
        self.router = UpperOutputRouter(self.upper)

    def planned(self, public_text="收到。"):
        record, _ = self.upper.claim_input(
            platform="onebot", origin="group:1", source_message_id="m1",
            robot_id="r1", user_id="u1", session_id="s1")
        return self.upper.save_result(
            record["input_key"],
            {"route": "speech_only", "public_text": public_text, "segments": [],
             "send_count": 0,
             "probabilities": {"speech_only": 1.0, "speech_and_action": 0.0,
                               "action_only": 0.0}},
            generation=1)

    def test_only_upper_public_is_permitted(self):
        self.assertTrue(UpperOutputRouter.permits_public_output(UPPER_PUBLIC))
        self.assertFalse(UpperOutputRouter.permits_public_output(UPPER_PRIVATE))
        self.assertFalse(UpperOutputRouter.permits_public_output(UPPER_CONTROL))
        self.assertFalse(UpperOutputRouter.permits_public_output(None))
        self.assertFalse(UpperOutputRouter.permits_public_output("public_chat"))
        self.assertFalse(UpperOutputRouter.permits_public_output(UPPER_PUBLIC,
                                                                 plan_id="plan-1"))

    def test_automatic_output_is_suppressed_for_every_upper_source(self):
        for source in (UPPER_PRIVATE, UPPER_PUBLIC, UPPER_CONTROL):
            self.assertFalse(UpperOutputRouter.permits_automatic_output(source))
        self.assertFalse(UpperOutputRouter.permits_automatic_output(None, plan_id="p"))
        self.assertTrue(UpperOutputRouter.permits_automatic_output("public_chat"))
        self.assertTrue(UpperOutputRouter.permits_automatic_output(None))

    def test_non_public_sources_are_refused_loudly(self):
        plan = self.planned()
        for source in (UPPER_PRIVATE, UPPER_CONTROL):
            with self.assertRaisesRegex(TaskError, "source_not_public"):
                self.router.claim(source=source, plan_id=plan["plan_id"])
        with self.assertRaisesRegex(TaskError, "invalid_source"):
            self.router.claim(source="public_chat", plan_id=plan["plan_id"])

    def test_claim_is_idempotent_per_plan(self):
        plan = self.planned()
        first = self.router.claim(source=UPPER_PUBLIC, plan_id=plan["plan_id"])
        second = self.router.claim(source=UPPER_PUBLIC, plan_id=plan["plan_id"])
        self.assertTrue(first["claimed"])
        self.assertFalse(second["claimed"])
        self.assertEqual(first["reply"]["reply_id"], second["reply"]["reply_id"])

    def test_legacy_public_router_behaviour_is_unchanged(self):
        self.assertTrue(PublicOutputRouter.permits_automatic_output(None))
        self.assertTrue(PublicOutputRouter.permits_automatic_output("public_chat"))
        self.assertFalse(PublicOutputRouter.permits_automatic_output("private_planning"))
        self.assertFalse(PublicOutputRouter.permits_automatic_output(None, task_id="t1"))
        with self.assertRaises(ValueError):
            PublicOutputRouter(self.store, None, delivery="carrier-pigeon")


if __name__ == "__main__":
    unittest.main()
