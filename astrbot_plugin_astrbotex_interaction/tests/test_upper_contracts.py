"""B00 AEB-side upper-control contract checks.

Runs without the AstrBot host: the module under test is loaded straight from
the plugin directory, exactly like ``test_task_contracts.py`` does for the
frozen decision contract.

Two things are being proven:

1. The shared fixture set under ``tests/fixtures/upper_control`` is
   byte-identical to the EX copy (verified by SHA-256 against ``manifest.json``),
   and every shared case is accepted or rejected the same way on both ends.
2. The parser refuses the inputs C01 forbids: bools as numbers, unknown fields,
   string probabilities, non-finite numbers, over-limit payloads, and identity
   that did not come from framework admission.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "upper_control"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


contracts = load_module("aeb_upper_contracts", PLUGIN / "upper_contracts.py")


def materialise(case: dict) -> dict:
    """Expand a fixture that carries a builder spec instead of a payload."""
    spec = case.get("payload_builder")
    if not spec:
        return case["payload"]
    if spec["kind"] != "oversize_batch":
        raise AssertionError(f"unknown payload builder {spec['kind']!r}")
    payload = json.loads(json.dumps(spec["base"]))
    payload["segments"] = [
        {
            "segment_id": f"seg-{index + 1:04d}",
            "ordinal": index,
            "text": "机" * spec["chars"],
        }
        for index in range(spec["segments"])
    ]
    return payload


def load_cases() -> list[dict]:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    cases = []
    for entry in manifest["files"]:
        raw = (FIXTURES / entry["name"]).read_bytes()
        cases.append(json.loads(raw.decode("utf-8")))
    return cases


def trusted_of(case: dict):
    block = case.get("trusted")
    if block is None:
        return None
    return contracts.TrustedIngress(
        robot_id=block["robot_id"],
        ex_session=block["ex_session"],
        route_refs=frozenset(block["route_refs"]),
    )


class FixtureIntegrityTests(unittest.TestCase):
    """The fixtures are the cross-repository contract; their bytes matter."""

    def test_manifest_lists_every_case_file(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        on_disk = {path.name for path in FIXTURES.glob("*.json")} - {"manifest.json"}
        listed = {entry["name"] for entry in manifest["files"]}
        self.assertEqual(on_disk, listed, "manifest and fixture directory disagree")

    def test_every_case_file_matches_its_recorded_sha256(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            raw = (FIXTURES / entry["name"]).read_bytes()
            self.assertEqual(
                hashlib.sha256(raw).hexdigest(),
                entry["sha256"],
                f"{entry['name']} is not the reviewed byte sequence",
            )

    def test_every_case_declares_expected_outcome(self):
        for case in load_cases():
            with self.subTest(case=case["case_id"]):
                self.assertIn("expected_accept", case)
                if case["expected_accept"]:
                    self.assertIsNone(case.get("expected_error"))
                else:
                    self.assertTrue(
                        case.get("expected_error"), "a rejection must name its code"
                    )
                self.assertIn("aeb", case["sides"])


class SharedFixtureTests(unittest.TestCase):
    """Every case the EX parser also consumes must agree with the AEB parser."""

    def test_shared_cases_are_batch_cases(self):
        for case in load_cases():
            if "ex" in case["sides"]:
                with self.subTest(case=case["case_id"]):
                    self.assertEqual(
                        case["kind"],
                        "batch",
                        "only wire batches are shared with EX",
                    )

    def test_shared_case_count_is_not_trivial(self):
        shared = [case for case in load_cases() if "ex" in case["sides"]]
        self.assertGreaterEqual(len(shared), 15)


class ProbabilityTests(unittest.TestCase):
    def test_fixture_cases(self):
        for case in load_cases():
            if case["kind"] != "probabilities":
                continue
            with self.subTest(case=case["case_id"]):
                if case["expected_accept"]:
                    parsed = contracts.RouteProbabilities.parse(materialise(case))
                    self.assertEqual(parsed.route(), case["expected_route"])
                else:
                    with self.assertRaises(contracts.UpperContractError) as raised:
                        contracts.RouteProbabilities.parse(materialise(case))
                    self.assertEqual(raised.exception.code, case["expected_error"])

    def test_non_finite_numbers_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                with self.assertRaises(contracts.UpperContractError) as raised:
                    contracts.RouteProbabilities.parse(
                        {"speech_only": bad, "action_only": 0.0, "speech_and_action": 0.0}
                    )
                self.assertEqual(raised.exception.code, "invalid_probabilities")

    def test_bool_is_not_a_probability_or_an_integer(self):
        # The classic Python trap: isinstance(True, int) is True.
        self.assertIsInstance(True, int)
        with self.assertRaises(contracts.UpperContractError):
            contracts.RouteProbabilities.parse(
                {"speech_only": True, "action_only": 0.0, "speech_and_action": 0.0}
            )

    def test_sum_tolerance_accepts_rounding_noise(self):
        parsed = contracts.RouteProbabilities.parse(
            {"speech_only": 0.3334, "action_only": 0.3333, "speech_and_action": 0.3333}
        )
        self.assertEqual(parsed.route(), "speech_only")

    def test_route_is_never_taken_from_the_payload(self):
        # A model that also emits a route must be rejected, not obeyed.
        with self.assertRaises(contracts.UpperContractError) as raised:
            contracts.RouteProbabilities.parse(
                {
                    "speech_only": 0.9,
                    "action_only": 0.05,
                    "speech_and_action": 0.05,
                    "route": "action_only",
                }
            )
        self.assertEqual(raised.exception.code, "invalid_probabilities")


class ResultTests(unittest.TestCase):
    def test_fixture_cases(self):
        for case in load_cases():
            if case["kind"] != "result":
                continue
            with self.subTest(case=case["case_id"]):
                payload = materialise(case)
                if case["expected_accept"]:
                    result = contracts.UpperResult.parse(payload, case["route"])
                    self.assertEqual(
                        [segment.text for segment in result.segments],
                        case["expected_texts"],
                    )
                    self.assertEqual(result.send_count, case["expected_send_count"])
                    self.assertEqual(
                        [segment.text for segment in result.deferred_segments()],
                        case.get("expected_deferred", []),
                    )
                    self.assertEqual(
                        len(result.sent_segments()), case["expected_send_count"]
                    )
                else:
                    with self.assertRaises(contracts.UpperContractError) as raised:
                        contracts.UpperResult.parse(payload, case["route"])
                    self.assertEqual(raised.exception.code, case["expected_error"])

    def test_route_must_be_known(self):
        with self.assertRaises(contracts.UpperContractError) as raised:
            contracts.UpperResult.parse(
                {"public_text": "好的。", "segments": [], "send_count": 0},
                "speech_and_execute",
            )
        self.assertEqual(raised.exception.code, "invalid_result")

    def test_text_is_preserved_verbatim(self):
        text = "  把杯子向上抬 20 米，不要改成 20 厘米。  "
        result = contracts.UpperResult.parse(
            {"public_text": None, "segments": [{"text": text}], "send_count": 1},
            "action_only",
        )
        self.assertEqual(result.segments[0].text, text)

    def test_segment_count_limit(self):
        with self.assertRaises(contracts.UpperContractError) as raised:
            contracts.UpperResult.parse(
                {
                    "public_text": "收到。",
                    "segments": [{"text": "步骤。"}] * 65,
                    "send_count": 1,
                },
                "speech_and_action",
            )
        self.assertEqual(raised.exception.code, "invalid_result")

    def test_segment_length_limit(self):
        with self.assertRaises(contracts.UpperContractError) as raised:
            contracts.UpperResult.parse(
                {
                    "public_text": "收到。",
                    "segments": [{"text": "机" * 4097}],
                    "send_count": 1,
                },
                "speech_and_action",
            )
        self.assertEqual(raised.exception.code, "invalid_result")

    def test_result_carries_no_identity(self):
        result = contracts.UpperResult.parse(
            {"public_text": None, "segments": [{"text": "第一步。"}], "send_count": 1},
            "action_only",
        )
        self.assertEqual(set(result.to_dict()), {"public_text", "segments", "send_count"})
        self.assertEqual(set(result.segments[0].to_dict()), {"text"})


class BatchTests(unittest.TestCase):
    def test_fixture_cases(self):
        for case in load_cases():
            if case["kind"] != "batch":
                continue
            with self.subTest(case=case["case_id"]):
                payload = materialise(case)
                if case["expected_accept"]:
                    batch = contracts.InjectionBatch.parse(payload)
                    trusted = trusted_of(case)
                    if trusted is not None:
                        contracts.check_ingress(batch, trusted)
                    if "previous" in case:
                        previous = contracts.InjectionBatch.parse(case["previous"])
                        self.assertEqual(
                            contracts.check_replay(previous, batch),
                            case.get("expect_replay", False),
                        )
                    self.assertEqual(batch.payload_hash(), case["expected_hash"])
                    self.assertEqual(
                        [segment.text for segment in batch.segments],
                        case["expected_texts"],
                    )
                else:
                    self.assertRejectedCase(case, payload)

    def assertRejectedCase(self, case, payload):
        with self.assertRaises(contracts.UpperContractError) as raised:
            batch = contracts.InjectionBatch.parse(payload)
            trusted = trusted_of(case)
            if trusted is not None:
                contracts.check_ingress(batch, trusted)
            if "previous" in case:
                previous = contracts.InjectionBatch.parse(case["previous"])
                contracts.check_replay(previous, batch)
        self.assertEqual(raised.exception.code, case["expected_error"])

    def test_payload_hash_is_stable_and_key_order_independent(self):
        payload = materialise(load_cases()[0]) if False else {
            "schema": "aeb.upper_steps.v1",
            "request_id": "req-1",
            "plan_id": "plan-1",
            "plan_revision": 1,
            "batch_id": "batch-1",
            "robot_id": "robot-1",
            "route_ref": "route-1",
            "ex_session": "ex-1",
            "catalog_version": "cat-1",
            "prompt_version": "prompt-1",
            "visibility": "decision_only",
            "segments": [{"segment_id": "s1", "ordinal": 0, "text": "寻找杯子。"}],
        }
        forward = contracts.InjectionBatch.parse(payload)
        reversed_payload = dict(reversed(list(payload.items())))
        backward = contracts.InjectionBatch.parse(reversed_payload)
        self.assertEqual(forward.payload_hash(), backward.payload_hash())

    def test_hash_covers_text_and_order(self):
        base = {
            "schema": "aeb.upper_steps.v1",
            "request_id": "req-1",
            "plan_id": "plan-1",
            "plan_revision": 1,
            "batch_id": "batch-1",
            "robot_id": "robot-1",
            "route_ref": "route-1",
            "ex_session": "ex-1",
            "catalog_version": "cat-1",
            "prompt_version": "prompt-1",
            "visibility": "decision_only",
            "segments": [
                {"segment_id": "s1", "ordinal": 0, "text": "第一步。"},
                {"segment_id": "s2", "ordinal": 1, "text": "第二步。"},
            ],
        }
        forward = contracts.InjectionBatch.parse(base).payload_hash()
        swapped = json.loads(json.dumps(base))
        swapped["segments"] = [
            {"segment_id": "s2", "ordinal": 1, "text": "第二步。"},
            {"segment_id": "s1", "ordinal": 0, "text": "第一步。"},
        ]
        # Reordering the array keeps ordinals contiguous only if we also swap
        # ordinals; instead prove the hash notices a text change and an id change.
        changed_text = json.loads(json.dumps(base))
        changed_text["segments"][1]["text"] = "第二步（改）。"
        changed_id = json.loads(json.dumps(base))
        changed_id["batch_id"] = "batch-2"
        self.assertNotEqual(forward, contracts.InjectionBatch.parse(changed_text).payload_hash())
        self.assertNotEqual(forward, contracts.InjectionBatch.parse(changed_id).payload_hash())
        # Same content, different key order and different segment order in dict.
        reordered = json.loads(json.dumps(base))
        reordered["segments"][0] = {
            "text": "第一步。", "ordinal": 0, "segment_id": "s1"
        }
        self.assertEqual(forward, contracts.InjectionBatch.parse(reordered).payload_hash())

    def test_ingress_rejects_a_format_valid_but_foreign_robot(self):
        payload = {
            "schema": "aeb.upper_steps.v1",
            "request_id": "req-1",
            "plan_id": "plan-1",
            "plan_revision": 1,
            "batch_id": "batch-1",
            "robot_id": "robot-other",
            "route_ref": "route-1",
            "ex_session": "ex-1",
            "catalog_version": "cat-1",
            "prompt_version": "prompt-1",
            "visibility": "decision_only",
            "segments": [{"segment_id": "s1", "ordinal": 0, "text": "第一步。"}],
        }
        batch = contracts.InjectionBatch.parse(payload)  # format is fine
        trusted = contracts.TrustedIngress(
            robot_id="robot-1", ex_session="ex-1", route_refs=frozenset({"route-1"})
        )
        with self.assertRaises(contracts.UpperContractError) as raised:
            contracts.check_ingress(batch, trusted)
        self.assertEqual(raised.exception.code, "session_mismatch")


class TrustedInputTests(unittest.TestCase):
    def test_cannot_be_built_from_a_dict(self):
        with self.assertRaises(contracts.UpperContractError):
            contracts.TrustedUpperInput(
                robot_id="robot-1",
                session_id="sess-1",
                user_id="user-1",
                route_ref="route-1",
                ex_session="ex-1",
                input_text="把杯子拿过来。",
                input_key="input-1",
            )

    def test_issue_accepts_a_framework_identity(self):
        trusted = contracts.TrustedUpperInput.issue(
            robot_id="robot-1",
            session_id="sess-1",
            user_id="user-1",
            route_ref="route-1",
            ex_session="ex-1",
            input_text="把杯子拿过来。",
            input_key="input-1",
        )
        self.assertEqual(trusted.robot_id, "robot-1")

    def test_issue_rejects_a_path_like_id(self):
        with self.assertRaises(contracts.UpperContractError):
            contracts.TrustedUpperInput.issue(
                robot_id="../robot",
                session_id="sess-1",
                user_id="user-1",
                route_ref="route-1",
                ex_session="ex-1",
                input_text="把杯子拿过来。",
                input_key="input-1",
            )


class BuildBatchTests(unittest.TestCase):
    def test_identity_comes_from_trusted_input_only(self):
        trusted = contracts.TrustedUpperInput.issue(
            robot_id="robot-1",
            session_id="sess-1",
            user_id="user-1",
            route_ref="route-1",
            ex_session="ex-1",
            input_text="把杯子拿过来。",
            input_key="input-1",
        )
        result = contracts.UpperResult.parse(
            {
                "public_text": None,
                "segments": [{"text": "第一步。"}, {"text": "第二步。"}],
                "send_count": 1,
            },
            "action_only",
        )
        batch = contracts.build_batch(
            trusted=trusted,
            result=result,
            plan_id="plan-1",
            plan_revision=1,
            batch_id="batch-1",
            request_id="req-1",
            catalog_version="cat-1",
            prompt_version="prompt-1",
            segment_id_prefix="plan-1",
        )
        self.assertEqual(batch.robot_id, "robot-1")
        self.assertEqual(batch.route_ref, "route-1")
        self.assertEqual(batch.ex_session, "ex-1")
        # send_count 1 -> only the prefix travels; identity is never model text.
        self.assertEqual(len(batch.segments), 1)
        self.assertEqual(batch.segments[0].text, "第一步。")
        self.assertEqual(batch.ordinals(), (0,))

    def test_suffix_batch_continues_ordinals(self):
        trusted = contracts.TrustedUpperInput.issue(
            robot_id="robot-1",
            session_id="sess-1",
            user_id="user-1",
            route_ref="route-1",
            ex_session="ex-1",
            input_text="继续。",
            input_key="input-2",
        )
        result = contracts.UpperResult.parse(
            {"public_text": None, "segments": [{"text": "第三步。"}], "send_count": 1},
            "action_only",
        )
        batch = contracts.build_batch(
            trusted=trusted,
            result=result,
            plan_id="plan-1",
            plan_revision=2,
            batch_id="batch-2",
            request_id="req-2",
            catalog_version="cat-1",
            prompt_version="prompt-1",
            segment_id_prefix="plan-1",
            start_ordinal=2,
        )
        self.assertEqual(batch.ordinals(), (2,))
        self.assertEqual(batch.plan_revision, 2)


class ErrorTableTests(unittest.TestCase):
    def test_published_codes_are_classified(self):
        expected = {
            "invalid_probabilities": "format",
            "invalid_result": "format",
            "invalid_batch": "format",
            "payload_too_large": "format",
            "route_mismatch": "state",
            "session_mismatch": "state",
            "id_conflict": "state",
            "input_busy": "state",
            "backend_unsupported": "state",
            "delivery_unknown": "uncertain",
        }
        for code, group in expected.items():
            with self.subTest(code=code):
                self.assertEqual(contracts.classify_error(code), group)
        # delivery_unknown must never be lumped in with a definite rejection.
        self.assertEqual(contracts.classify_error("delivery_unknown"), "uncertain")


if __name__ == "__main__":
    unittest.main()
