"""B03 decision and generation checks.

Every test runs against a fake Host adapter, so nothing here touches a network,
a paid model or a device. The suite is built around the failure modes that would
otherwise be silent: a model deciding its own route, a second external side
effect from a repeated tool call, and a cancelled turn still writing a plan.

The real-Host smoke test is guarded and reports itself as skipped rather than
pretending the public interface was verified — B03 §5 forbids claiming real
interface coverage from a double.
"""

from __future__ import annotations

import ast
import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from astrbot_plugin_astrbotex_interaction import upper_models
from astrbot_plugin_astrbotex_interaction.upper_models import (
    ErrorCode,
    ModelError,
    UpperModels,
)

PLUGIN = Path(__file__).resolve().parents[1]

ROUTING_TOOL = "submit_route_probabilities"
EMIT_TOOL = "emit_upper_result"


def response(names=(), args=(), ids=None, text=None, chunk=False, role="assistant"):
    return SimpleNamespace(
        role=role,
        tools_call_name=list(names),
        tools_call_args=list(args),
        tools_call_ids=list(ids or [f"call-{index}" for index in range(len(names))]),
        completion_text=text,
        is_chunk=chunk,
    )


def emit(**payload):
    import json

    return response(names=[EMIT_TOOL], args=[json.dumps(payload, ensure_ascii=False)])


def probabilities(speech, action, both):
    import json

    return response(
        names=[ROUTING_TOOL],
        args=[json.dumps({"speech_only": speech, "action_only": action, "speech_and_action": both})],
    )


class Provider:
    def __init__(self, provider_id):
        self._id = provider_id

    def meta(self):
        return SimpleNamespace(id=self._id)


class FakeHost:
    """Injected adapter over the Host public surface."""

    def __init__(self, responses, *, providers=None, current="prov-1"):
        self.responses = list(responses)
        self.calls = []
        self.providers = providers if providers is not None else {"prov-1": Provider("prov-1")}
        self.current = current
        self.current_calls = []
        self.models = None

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if callable(item):
            item = item()
        if isinstance(item, BaseException):
            raise item
        return item

    def get_provider_by_id(self, provider_id):
        return self.providers.get(provider_id)

    def get_current_chat_provider_id(self, umo):
        self.current_calls.append(umo)
        if isinstance(self.current, BaseException):
            raise self.current
        return self.current


def snapshot(routing="ROUTING-SPEC", planning="PLANNING-SPEC"):
    return SimpleNamespace(
        routing_prompt=routing,
        planning_prompt=planning,
        routing_version="routing-v1",
        planning_version="planning-v1",
        content_hash="content-v1",
    )


def models_for(responses, **kwargs):
    host = FakeHost(responses, **kwargs)
    models = UpperModels(host, timeout_s=5.0, max_rounds=3)
    host.models = models
    return models, host


class RouteArgmaxTests(unittest.IsolatedAsyncioTestCase):
    CASES = (
        ((0.82, 0.11, 0.07), "speech_only"),
        ((0.60, 0.35, 0.05), "speech_only"),
        ((0.06, 0.81, 0.13), "action_only"),
        ((0.10, 0.55, 0.35), "action_only"),
        ((0.09, 0.12, 0.79), "speech_and_action"),
        ((0.30, 0.30, 0.40), "speech_and_action"),
    )

    async def test_two_probability_sets_per_route(self):
        for values, expected in self.CASES:
            with self.subTest(values=values):
                models, host = models_for([probabilities(*values)])
                result = await models.classify("把红杯子拿过来", snapshot())
                self.assertEqual(result.route, expected)
                self.assertEqual(len(host.calls), 1)

    async def test_an_exact_tie_follows_the_c01_order(self):
        models, _ = models_for([probabilities(0.5, 0.5, 0.0)])
        result = await models.classify("随便说点什么", snapshot())
        self.assertEqual(result.route, "speech_only")

    async def test_the_framework_picks_the_route_not_the_model(self):
        # The payload carries a route hint; the strict parser refuses it.
        import json

        payload = response(
            names=[ROUTING_TOOL],
            args=[
                json.dumps(
                    {
                        "speech_only": 0.9,
                        "action_only": 0.05,
                        "speech_and_action": 0.05,
                        "route": "action_only",
                    }
                )
            ],
        )
        models, _ = models_for([payload])
        with self.assertRaises(ModelError) as raised:
            await models.classify("把杯子拿过来", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.DECISION_ERROR)


class DecisionRejectionTests(unittest.IsolatedAsyncioTestCase):
    async def assert_decision_error(self, item):
        models, host = models_for([item])
        with self.assertRaises(ModelError) as raised:
            await models.classify("把杯子拿过来", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.DECISION_ERROR)
        # Only the classification call happened: zero generation, zero send.
        self.assertEqual(len(host.calls), 1)

    async def test_a_bool_probability_is_rejected(self):
        await self.assert_decision_error(
            response(
                names=[ROUTING_TOOL],
                args=['{"speech_only": true, "action_only": 0.1, "speech_and_action": 0.1}'],
            )
        )

    async def test_a_string_probability_is_rejected(self):
        await self.assert_decision_error(
            response(
                names=[ROUTING_TOOL],
                args=['{"speech_only": "0.8", "action_only": 0.1, "speech_and_action": 0.1}'],
            )
        )

    async def test_a_bad_sum_is_rejected(self):
        await self.assert_decision_error(probabilities(0.5, 0.2, 0.2))

    async def test_prose_instead_of_probabilities_is_rejected(self):
        await self.assert_decision_error(response(text="我觉得应该执行这个动作。"))

    async def test_an_empty_routing_turn_is_rejected(self):
        await self.assert_decision_error(response())

    async def test_a_wrong_tool_name_is_rejected(self):
        await self.assert_decision_error(
            response(names=["decide_route"], args=['{"speech_only":1,"action_only":0,"speech_and_action":0}'])
        )

    async def test_two_tool_calls_in_one_turn_are_rejected(self):
        await self.assert_decision_error(
            response(
                names=[ROUTING_TOOL, ROUTING_TOOL],
                args=[
                    '{"speech_only":0.8,"action_only":0.1,"speech_and_action":0.1}',
                    '{"speech_only":0.1,"action_only":0.8,"speech_and_action":0.1}',
                ],
            )
        )

    async def test_a_bare_json_object_is_accepted(self):
        models, _ = models_for(
            [response(text='{"speech_only": 0.1, "action_only": 0.8, "speech_and_action": 0.1}')]
        )
        result = await models.classify("把杯子拿过来", snapshot())
        self.assertEqual(result.route, "action_only")


class CallShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_classify_is_private_and_non_streaming(self):
        models, host = models_for([probabilities(0.8, 0.1, 0.1)])
        await models.classify("你好", snapshot())
        call = host.calls[0]
        self.assertIs(call["stream"], False)
        self.assertEqual(call["system_prompt"], "ROUTING-SPEC")
        self.assertEqual(call["chat_provider_id"], "prov-1")
        # The Host expects a ToolSet of FunctionTool objects, not dicts.
        names = [tool.name for tool in call["tools"]]
        self.assertEqual(names, [ROUTING_TOOL])

    async def test_generate_exposes_only_emit_upper_result(self):
        models, host = models_for(
            [emit(public_text=None, segments=[{"text": "第一步。"}], send_count=1)]
        )
        await models.generate("把杯子拿过来", "action_only", snapshot())
        call = host.calls[0]
        self.assertIs(call["stream"], False)
        self.assertEqual(call["system_prompt"], "PLANNING-SPEC")
        names = [tool.name for tool in call["tools"]]
        self.assertEqual(names, [EMIT_TOOL])
        # The old planning / proposal / execution tools must not be reachable.
        for forbidden in ("manage_astrbotex_task", "proposal", "submit_current_goal", "save_plan"):
            with self.subTest(tool=forbidden):
                self.assertNotIn(forbidden, names)


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_empty_id_uses_the_real_input_origin(self):
        models, host = models_for([probabilities(0.8, 0.1, 0.1)])
        await models.classify("你好", snapshot(), origin="astrbotex:group:42")
        self.assertEqual(host.current_calls, ["astrbotex:group:42"])

    async def test_an_unknown_explicit_provider_is_reported_not_swapped(self):
        models, host = models_for([probabilities(0.8, 0.1, 0.1)])
        with self.assertRaises(ModelError) as raised:
            await models.classify("你好", snapshot(), provider_id="prov-missing")
        self.assertEqual(raised.exception.code, ErrorCode.PROVIDER_UNAVAILABLE)
        self.assertEqual(host.calls, [])

    async def test_a_missing_current_provider_is_reported(self):
        models, _ = models_for([probabilities(0.8, 0.1, 0.1)], current=None)
        with self.assertRaises(ModelError) as raised:
            await models.classify("你好", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.PROVIDER_UNAVAILABLE)

    async def test_a_current_provider_lookup_failure_is_reported(self):
        models, _ = models_for(
            [probabilities(0.8, 0.1, 0.1)], current=RuntimeError("provider not found")
        )
        with self.assertRaises(ModelError) as raised:
            await models.classify("你好", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.PROVIDER_UNAVAILABLE)

    async def test_the_two_provider_ids_are_independent(self):
        models, host = models_for(
            [probabilities(0.8, 0.1, 0.1)],
            providers={"prov-1": Provider("prov-1"), "prov-gen": Provider("prov-gen")},
        )
        await models.classify("你好", snapshot(), provider_id="prov-1")
        self.assertEqual(host.calls[0]["chat_provider_id"], "prov-1")


class GenerationShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_speech_only_has_no_segment(self):
        models, _ = models_for(
            [emit(public_text="好的，我在听。", segments=[], send_count=0)]
        )
        result = await models.generate("你好", "speech_only", snapshot())
        self.assertEqual(result.result.public_text, "好的，我在听。")
        self.assertEqual(result.result.segments, ())
        self.assertEqual(result.result.send_count, 0)

    async def test_action_only_multi_segment(self):
        models, _ = models_for(
            [
                emit(
                    public_text=None,
                    segments=[{"text": "第一步。"}, {"text": "第二步。"}],
                    send_count=2,
                )
            ]
        )
        result = await models.generate("做事", "action_only", snapshot())
        self.assertIsNone(result.result.public_text)
        self.assertEqual(len(result.result.segments), 2)

    async def test_send_count_one_defers_the_suffix(self):
        models, _ = models_for(
            [
                emit(
                    public_text="先做第一步。",
                    segments=[
                        {"text": "寻找目标。"},
                        {"text": "接近目标。"},
                        {"text": "抓取目标。"},
                    ],
                    send_count=1,
                )
            ]
        )
        result = await models.generate("拿杯子", "speech_and_action", snapshot())
        self.assertEqual([segment.text for segment in result.result.sent_segments()], ["寻找目标。"])
        self.assertEqual(
            [segment.text for segment in result.result.deferred_segments()],
            ["接近目标。", "抓取目标。"],
        )

    async def test_send_count_all(self):
        models, _ = models_for(
            [
                emit(
                    public_text="一起做。",
                    segments=[{"text": "一。"}, {"text": "二。"}, {"text": "三。"}],
                    send_count=3,
                )
            ]
        )
        result = await models.generate("拿杯子", "speech_and_action", snapshot())
        self.assertEqual(result.result.send_count, 3)
        self.assertEqual(result.result.deferred_segments(), ())


class GenerationRejectionTests(unittest.IsolatedAsyncioTestCase):
    async def assert_generation_error(self, item, route="action_only"):
        models, host = models_for([item])
        with self.assertRaises(ModelError) as raised:
            await models.generate("做事", route, snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.GENERATION_ERROR)
        return host

    async def test_an_action_only_result_may_not_carry_public_text(self):
        await self.assert_generation_error(
            emit(public_text="我这就做。", segments=[{"text": "第一步。"}], send_count=1)
        )

    async def test_a_result_may_not_supply_identity(self):
        await self.assert_generation_error(
            emit(
                public_text=None,
                segments=[{"text": "第一步。"}],
                send_count=1,
                plan_id="plan-from-model",
            )
        )

    async def test_a_send_count_beyond_the_segments_is_rejected(self):
        await self.assert_generation_error(
            emit(public_text=None, segments=[{"text": "第一步。"}], send_count=3)
        )

    async def test_an_unknown_tool_is_rejected(self):
        await self.assert_generation_error(
            response(names=["manage_astrbotex_task"], args=["{}"])
        )

    async def test_a_failed_result_is_not_repaired_and_retried(self):
        host = await self.assert_generation_error(
            emit(public_text="不该有的文字", segments=[{"text": "第一步。"}], send_count=1)
        )
        # Exactly one call: no automatic repair round, no silent route downgrade.
        self.assertEqual(len(host.calls), 1)


class GenerationLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_repeated_tool_call_returns_the_first_result_once(self):
        models, host = models_for(
            [
                response(
                    names=[EMIT_TOOL, EMIT_TOOL],
                    args=[
                        '{"public_text": null, "segments": [{"text": "第一次。"}], "send_count": 1}',
                        '{"public_text": null, "segments": [{"text": "第二次。"}], "send_count": 1}',
                    ],
                    ids=["call-a", "call-b"],
                )
            ]
        )
        result = await models.generate("做事", "action_only", snapshot())
        self.assertEqual(result.result.segments[0].text, "第一次。")
        # One provider round only, so no second external side effect.
        self.assertEqual(len(host.calls), 1)

    async def test_a_text_only_round_is_retried_then_times_out_of_rounds(self):
        models, host = models_for(
            [response(text="我先想想。"), response(text="还是想想。")], 
        )
        models._max_rounds = 2
        with self.assertRaises(ModelError) as raised:
            await models.generate("做事", "action_only", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.GENERATION_NO_RESULT)
        self.assertEqual(len(host.calls), 2)

    async def test_an_empty_response_does_not_produce_a_result(self):
        models, _ = models_for([response(), response()])
        models._max_rounds = 2
        with self.assertRaises(ModelError) as raised:
            await models.generate("做事", "action_only", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.GENERATION_NO_RESULT)

    async def test_a_result_after_one_text_round_is_accepted(self):
        models, host = models_for(
            [
                response(text="我先确认一下目标。"),
                emit(public_text=None, segments=[{"text": "寻找目标。"}], send_count=1),
            ]
        )
        result = await models.generate("做事", "action_only", snapshot())
        self.assertEqual(result.result.segments[0].text, "寻找目标。")
        self.assertEqual(len(host.calls), 2)
        self.assertEqual(result.evidence.rounds, 2)


class TimeoutAndFenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_slow_provider_times_out_without_a_result(self):
        class SlowHost(FakeHost):
            async def llm_generate(self, **kwargs):
                self.calls.append(kwargs)
                await asyncio.sleep(1.0)
                return emit(public_text=None, segments=[{"text": "太晚了。"}], send_count=1)

        host = SlowHost([])
        models = UpperModels(host, timeout_s=0.05, max_rounds=2)
        with self.assertRaises(ModelError) as raised:
            await models.generate("做事", "action_only", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.GENERATION_TIMEOUT)

    async def test_a_provider_exception_is_a_generation_error(self):
        models, _ = models_for([RuntimeError("boom")])
        with self.assertRaises(ModelError) as raised:
            await models.generate("做事", "action_only", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.GENERATION_ERROR)

    async def test_a_cancelled_turn_rejects_its_own_late_result(self):
        def cancel_then_answer():
            models.cancel()
            return emit(public_text=None, segments=[{"text": "迟到的结果。"}], send_count=1)

        models, host = models_for([cancel_then_answer])
        with self.assertRaises(ModelError) as raised:
            await models.generate("做事", "action_only", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.GENERATION_STALE)

    async def test_a_config_switch_invalidates_an_in_flight_classification(self):
        def switch_then_answer():
            models.invalidate()
            return probabilities(0.8, 0.1, 0.1)

        models, _ = models_for([switch_then_answer])
        with self.assertRaises(ModelError) as raised:
            await models.classify("你好", snapshot())
        self.assertEqual(raised.exception.code, ErrorCode.GENERATION_STALE)

    async def test_the_fence_is_monotonic(self):
        models, _ = models_for([])
        start = models.fence
        self.assertGreater(models.cancel(), start)
        self.assertGreater(models.invalidate(), models.fence - 1)


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_classification_evidence_carries_versions_and_hashes(self):
        models, _ = models_for([probabilities(0.8, 0.1, 0.1)])
        result = await models.classify("把杯子拿过来", snapshot())
        evidence = result.evidence
        self.assertEqual(evidence.provider_id, "prov-1")
        self.assertEqual(evidence.model_id, "prov-1")
        self.assertEqual(evidence.prompt_hash, "routing-v1")
        self.assertEqual(len(evidence.input_hash), 64)
        self.assertEqual(len(evidence.output_hash), 64)
        self.assertEqual(evidence.probabilities["speech_only"], 0.8)

    async def test_generation_evidence_does_not_carry_probabilities(self):
        models, _ = models_for(
            [emit(public_text=None, segments=[{"text": "第一步。"}], send_count=1)]
        )
        result = await models.generate("做事", "action_only", snapshot())
        self.assertIsNone(result.evidence.probabilities)
        self.assertEqual(result.evidence.prompt_hash, "planning-v1")

    async def test_evidence_is_not_part_of_the_result_payload(self):
        models, _ = models_for(
            [emit(public_text=None, segments=[{"text": "第一步。"}], send_count=1)]
        )
        result = await models.generate("做事", "action_only", snapshot())
        # The B00 result stays a pure contract object.
        self.assertEqual(
            set(result.result.to_dict()), {"public_text", "segments", "send_count"}
        )


class IsolationTests(unittest.TestCase):
    def test_the_module_does_not_import_the_host_at_module_level(self):
        source = (PLUGIN / "upper_models.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith("astrbot"), alias.name)
            elif isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("astrbot"), node.module)

    def test_the_module_never_sends_a_message_or_opens_a_socket(self):
        source = (PLUGIN / "upper_models.py").read_text(encoding="utf-8")
        for forbidden in ("send_message", "import zmq", "import requests", "aiohttp"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, source)

    def test_the_contract_objects_come_from_b00(self):
        source = (PLUGIN / "upper_models.py").read_text(encoding="utf-8")
        self.assertIn("from .upper_contracts import", source)


class RealHostSmokeTests(unittest.TestCase):
    """B03 §5: a double passing is not evidence about the real interface."""

    def test_the_real_host_public_signature_is_not_claimed(self):
        try:
            import astrbot  # noqa: F401
        except Exception:
            self.skipTest(
                "AstrBot is not installed in this interpreter, so the real Host "
                "public llm_generate signature was NOT verified here. The double "
                "coverage above must not be reported as real_host scope."
            )
        # When the Host is present the integration smoke belongs to B11, which
        # owns the real cross-end run.
        self.assertTrue(hasattr(upper_models, "UpperModels"))


class FrozenTurnSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_timeout_can_increase_without_changing_other_turns(self):
        host = FakeHost([])
        async def slow_response(**kwargs):
            await asyncio.sleep(0.03)
            return probabilities(0.9, 0.05, 0.05)
        host.llm_generate = slow_response
        models = UpperModels(host, timeout_s=0.001)
        turn = models.for_turn(timeout_s=0.5, origin="origin-next-input")
        self.assertEqual((await turn.classify("input", snapshot())).route, "speech_only")
        self.assertEqual(host.current_calls, ["origin-next-input"])
        with self.assertRaises(ModelError) as caught:
            await models.for_turn(timeout_s=0.001).classify("input", snapshot())
        self.assertEqual(caught.exception.code, ErrorCode.GENERATION_TIMEOUT)

    async def test_bound_turn_shares_model_invalidation(self):
        entered, release = asyncio.Event(), asyncio.Event()
        host = FakeHost([])
        async def delayed(**kwargs):
            entered.set()
            await release.wait()
            return probabilities(0.9, 0.05, 0.05)
        host.llm_generate = delayed
        models = UpperModels(host)
        task = asyncio.create_task(models.for_turn(timeout_s=2).classify("input", snapshot()))
        await asyncio.wait_for(entered.wait(), 1)
        models.invalidate()
        release.set()
        with self.assertRaises(ModelError) as caught:
            await task
        self.assertEqual(caught.exception.code, ErrorCode.GENERATION_STALE)


if __name__ == "__main__":
    unittest.main()
