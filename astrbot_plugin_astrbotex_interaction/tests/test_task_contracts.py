"""Independent AEB-side B00 contract checks; no AstrBot host import required."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


PLUGIN = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
ACTION = "new_arm.pick_selected.v1"
MAX_SEQ = 2**53 - 1


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


contracts = load_module("aeb_standalone_contracts", PLUGIN / "task_contracts.py")


def goal():
    return {
        "schema_version": 1,
        "request_id": "req-1",
        "ex_session": "ex-1",
        "task_id": "task-1",
        "step_id": "step-1",
        "goal_id": "goal-1",
        "goal_text_en": "Pick the selected object.",
        "allowed_actions": [ACTION],
        "parameters": {ACTION: {"target": {"object_id": "track-1"}}},
        "completion": {"required_success_actions": [ACTION]},
        "lease_ms": 1000,
    }


def feedback(seq=1):
    return {
        "schema_version": 1,
        "ex_session": "ex-1",
        "task_id": "task-1",
        "goal_id": "goal-1",
        "goal_revision": 1,
        "event_seq": seq,
        "status": "running",
        "details": {"target": {"object_id": "track-1"}},
    }


def event(seq):
    return {
        "event_id": f"evt-{seq}",
        "event_seq": seq,
        "ex_session": "ex-1",
        "task_id": "task-1",
        "goal_id": "goal-1",
        "goal_revision": 1,
        "command_id": "cmd-1",
        "owner": "new_arm",
        "status": "running",
        "reason_code": "",
        "details": {},
    }


class ContractTests(unittest.TestCase):
    def assert_contract_error(self, code, path, call):
        with self.assertRaises(contracts.ContractError) as raised:
            call()
        self.assertEqual(raised.exception.error.code, code)
        self.assertEqual(raised.exception.error.path, path)
        self.assertIsInstance(raised.exception.error.message, str)
        self.assertTrue(raised.exception.error.message)

    def test_shared_golden_cases(self):
        runner = load_module("aeb_contract_fixture_runner", TESTS / "contract_fixture_runner.py")
        fixture = runner.load_fixture(TESTS / "fixtures" / "decision_contracts" / "golden.json")
        cases = fixture["cases"]
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 60)
        for required in ("S-GOAL-OK", "S-GOAL-COMPLETION", "S-METHOD-UNKNOWN",
                         "S-ST-TERMINAL-WINS", "S-CANCEL-TIMEOUT"):
            self.assertIn(required, ids)
        results = runner.run_fixtures(contracts, fixture)
        self.assertEqual([result.case_id for result in results], ids)
        self.assertFalse([(r.case_id, r.detail) for r in results if not r.ok])

    def test_standalone_subprocess_import_and_wire_roundtrip(self):
        script = """import importlib.util, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('task_contracts_standalone', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
payload = json.loads(sys.argv[2])
parsed = module.parse_request('decision.goal.submit', payload)
assert json.loads(json.dumps(parsed.to_dict())) == payload
assert not any(name == 'astrbot' or name.startswith('astrbot.') for name in sys.modules)
assert not any(name == 'astrbot_ex' or name.startswith('astrbot_ex.') for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(PLUGIN / "task_contracts.py"), json.dumps(goal())],
            capture_output=True, text=True, timeout=15, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_methods_and_unknown_method_never_falls_back(self):
        expected = {
            "decision.context.get", "decision.goal.submit", "decision.goal.cancel",
            "decision.goal.renew", "decision.state.get", "decision.events.get",
            "decision.feedback",
        }
        self.assertEqual(contracts.DECISION_METHODS, expected)
        for method in ("decision.proposal.submit", "bridge.proposal.submit", "decision.goal.unknown"):
            with self.subTest(method=method):
                self.assert_contract_error("unsupported_method", "method",
                                           lambda: contracts.parse_request(method, goal()))
        for method in ([], {}, None, "", "x" * 257):
            with self.subTest(invalid_method=method):
                self.assert_contract_error("invalid_type" if not isinstance(method, str) else
                                           "empty_string" if not method else "text_too_long", "method",
                                           lambda: contracts.parse_request(method, goal()))

    def test_strict_versions_ids_and_sequence_bounds(self):
        for value in (True, False, 1.0, 2):
            with self.subTest(version=value):
                payload = goal()
                payload["schema_version"] = value
                self.assert_contract_error("unsupported_schema_version", "schema_version",
                                           lambda: contracts.GoalSubmit.parse(payload))
        for key in ("request_id", "ex_session", "task_id", "step_id", "goal_id"):
            for value, code in ((True, "invalid_type"), ("", "empty_string"),
                                ("x" * 257, "text_too_long")):
                with self.subTest(field=key, value=value):
                    payload = goal()
                    payload[key] = value
                    self.assert_contract_error(code, key, lambda: contracts.GoalSubmit.parse(payload))
        for value, code in ((True, "invalid_type"), (-1, "range_violation"),
                            (MAX_SEQ + 1, "range_violation")):
            with self.subTest(sequence=value):
                payload = {"schema_version": 1, "ex_session": "ex-1", "since_event_seq": value}
                self.assert_contract_error(code, "since_event_seq",
                                           lambda: contracts.EventsRequest.parse(payload))
        self.assertEqual(contracts.EventsRequest.parse(
            {"schema_version": 1, "ex_session": "ex-1", "since_event_seq": 0}
        ).since_event_seq, 0)
        self.assert_contract_error("range_violation", "event_seq",
                                   lambda: contracts.Feedback.parse(feedback(0)))

    def test_revision_compare_and_advance_with_optional_cas(self):
        for expected in (None, 8):
            with self.subTest(expected=expected):
                self.assertEqual(contracts.check_revision(expected, 8), 9)
        self.assert_contract_error("revision_conflict", "expected_revision",
                                   lambda: contracts.check_revision(7, 8))
        self.assert_contract_error("invalid_type", "expected_revision",
                                   lambda: contracts.check_revision(True, 8))
        parsed = contracts.GoalSubmit.parse(goal())
        self.assertIsNone(parsed.expected_revision)
        self.assertNotIn("expected_revision", parsed.to_dict())

    def test_goal_parameters_and_completion_are_authorized(self):
        for field, value, code, path in (
            ("parameters", {"other.move.v1": {}}, "unknown_action", "parameters.other.move.v1"),
            ("parameters", {ACTION: []}, "invalid_type", f"parameters.{ACTION}"),
            ("completion", {"required_success_actions": ["other.move.v1"]},
             "unknown_action", "completion.required_success_actions[0]"),
            ("completion", {"required_success_actions": [ACTION], "ignore_failures": True},
             "unknown_field", "completion.ignore_failures"),
        ):
            with self.subTest(field=field, value=value):
                payload = goal()
                payload[field] = value
                self.assert_contract_error(code, path, lambda: contracts.GoalSubmit.parse(payload))
        for lease, code in ((True, "invalid_type"), (0, "non_positive_lease"),
                            (600001, "range_violation")):
            with self.subTest(lease=lease):
                payload = goal()
                payload["lease_ms"] = lease
                self.assert_contract_error(code, "lease_ms", lambda: contracts.GoalSubmit.parse(payload))

    def test_parsed_nested_values_do_not_alias_input_or_output(self):
        payload = goal()
        parsed = contracts.GoalSubmit.parse(payload)
        payload["parameters"][ACTION]["target"]["object_id"] = "input-edited"
        payload["completion"]["required_success_actions"].clear()
        self.assertEqual(parsed.parameters[ACTION]["target"]["object_id"], "track-1")
        self.assertEqual(parsed.completion["required_success_actions"], [ACTION])
        output = parsed.to_dict()
        output["parameters"][ACTION]["target"]["object_id"] = "output-edited"
        output["completion"]["required_success_actions"].clear()
        self.assertEqual(parsed.to_dict()["parameters"][ACTION]["target"]["object_id"], "track-1")
        self.assertEqual(parsed.to_dict()["completion"]["required_success_actions"], [ACTION])
        wire = json.loads(json.dumps(feedback()))
        parsed_feedback = contracts.Feedback.parse(wire)
        wire["details"]["target"]["object_id"] = "input-edited"
        self.assertEqual(parsed_feedback.details["target"]["object_id"], "track-1")
        parsed_feedback.to_dict()["details"]["target"]["object_id"] = "output-edited"
        self.assertEqual(parsed_feedback.to_dict()["details"]["target"]["object_id"], "track-1")

    def test_event_reply_does_not_alias_buffer_or_output(self):
        buffered = [event(1)]
        page = contracts.events_reply(ex_session="ex-1", buffered=buffered,
                                      oldest_available_seq=1, latest_event_seq=1,
                                      since_event_seq=0)
        buffered[0]["details"]["external"] = "changed"
        self.assertEqual(page.events[0]["details"], {})
        page.to_dict()["events"][0]["details"]["external"] = "changed again"
        self.assertEqual(page.to_dict()["events"][0]["details"], {})

    def test_bootstrap_reads_and_control_session(self):
        for method in ("decision.context.get", "decision.state.get"):
            with self.subTest(method=method):
                self.assertEqual(contracts.parse_request(method, {"schema_version": 1}),
                                 {"schema_version": 1})
                self.assert_contract_error("invalid_type", "ex_session",
                                           lambda: contracts.parse_request(
                                               method, {"schema_version": 1, "ex_session": True}))
        self.assert_contract_error("missing_field", "ex_session",
                                   lambda: contracts.parse_request("decision.events.get",
                                                                   {"schema_version": 1, "since_event_seq": 0}))

    def test_events_validate_history_before_resync(self):
        page = contracts.events_reply(
            ex_session="ex-1", buffered=[event(4), event(5)],
            oldest_available_seq=4, latest_event_seq=5, since_event_seq=3,
        )
        self.assertFalse(page.resync_required)
        self.assertEqual([item["event_seq"] for item in page.events], [4, 5])
        self.assertEqual(page.to_dict()["latest_event_seq"], 5)
        gap = contracts.events_reply(
            ex_session="ex-1", buffered=[event(4)], oldest_available_seq=4,
            latest_event_seq=9, since_event_seq=2,
        )
        self.assertTrue(gap.resync_required)
        self.assertEqual(gap.events, [])
        self.assertEqual(gap.latest_event_seq, 9)

    def test_events_reject_invalid_buffer_even_when_resyncing(self):
        for buffered, code, path in (
            ([event(0)], "range_violation", "events[0].event_seq"),
            ([event(5), event(4)], "range_violation", "events[1].event_seq"),
            ([event(4), event(4)], "range_violation", "events[1].event_seq"),
            ([{**event(4), "ex_session": "other"}], "stale_ex_session", "events[0].ex_session"),
            ([{**event(3)}], "range_violation", "events[0].event_seq"),
        ):
            with self.subTest(buffered=buffered):
                self.assert_contract_error(code, path, lambda: contracts.events_reply(
                    ex_session="ex-1", buffered=buffered, oldest_available_seq=4,
                    latest_event_seq=9, since_event_seq=0,
                ))

    def test_state_pairs_feedback_status_and_terminal_rules(self):
        state = {"schema_version": 1, "ex_session": "ex-1", "revision": 0,
                 "active_goal_id": "goal-1", "active_phase": "active", "execution": {}, "event_seq": 0}
        self.assertEqual(contracts.DecisionState.parse(state).active_phase, "active")
        self.assert_contract_error("enum_violation", "active_phase",
                                   lambda: contracts.DecisionState.parse({**state, "active_phase": "blocked"}))
        self.assert_contract_error("invalid_type", "active_phase",
                                   lambda: contracts.DecisionState.parse({**state, "active_phase": None}))
        self.assert_contract_error("enum_violation", "pending_phase",
                                   lambda: contracts.DecisionState.parse({
                                       **state, "pending_goal_id": "goal-2", "pending_phase": "active",
                                   }))
        self.assert_contract_error("invalid_type", "pending_phase",
                                   lambda: contracts.DecisionState.parse({**state, "pending_phase": "accepted"}))
        for status in ("succeeded", "failed", "canceled", "timed_out", "unknown", "rejected"):
            with self.subTest(terminal=status):
                self.assertTrue(contracts.is_terminal(status))
                self.assertEqual(contracts.apply_status(status, status), status)
                self.assert_contract_error("illegal_transition", "status",
                                           lambda: contracts.apply_status(status, "running"))
        self.assertEqual(contracts.cancel_outcome(has_stop_evidence=False, timed_out=False), None)
        self.assertEqual(contracts.cancel_outcome(has_stop_evidence=False, timed_out=True), "timed_out")
        self.assertEqual(contracts.cancel_outcome(has_stop_evidence=True, timed_out=False), "canceled")
        for status, code in (("done", "enum_violation"), ("", "empty_string"),
                             ("blocked", "enum_violation")):
            with self.subTest(feedback_status=status):
                self.assert_contract_error(code, "status",
                                           lambda: contracts.Feedback.parse({**feedback(), "status": status}))

    def test_action_event_requires_positive_stop_evidence(self):
        canceled = {**event(1), "status": "canceled"}
        self.assert_contract_error("missing_field", "details.stop_evidence",
                                   lambda: contracts.ActionEvent.parse(canceled))
        for stopped, code in ((False, "enum_violation"), (1, "invalid_type")):
            with self.subTest(stopped=stopped):
                self.assert_contract_error(code, "details.stop_evidence.stopped",
                                           lambda: contracts.ActionEvent.parse({
                                               **canceled, "details": {"stop_evidence": {"stopped": stopped}},
                                           }))
        valid = {**canceled, "details": {"stop_evidence": {
            "stopped": True, "source": "plugin", "reference": "stop-1",
        }}}
        self.assertEqual(contracts.ActionEvent.parse(valid).to_dict(), valid)
        for key in ("source", "reference"):
            with self.subTest(overlong_evidence=key):
                bad = {**canceled, "details": {"stop_evidence": {
                    "stopped": True, key: "x" * 257,
                }}}
                self.assert_contract_error("text_too_long", f"details.stop_evidence.{key}",
                                           lambda: contracts.ActionEvent.parse(bad))
        self.assert_contract_error("range_violation", "event_seq",
                                   lambda: contracts.ActionEvent.parse(event(0)))

    def test_canceled_feedback_requires_positive_stop_evidence(self):
        canceled = {**feedback(), "status": "canceled", "details": {}}
        self.assert_contract_error("missing_field", "details.stop_evidence",
                                   lambda: contracts.Feedback.parse(canceled))
        for stopped, code in ((False, "enum_violation"), (1, "invalid_type")):
            with self.subTest(stopped=stopped):
                self.assert_contract_error(code, "details.stop_evidence.stopped",
                                           lambda: contracts.Feedback.parse({
                                               **canceled, "details": {"stop_evidence": {"stopped": stopped}},
                                           }))
        valid = {**canceled, "reason_code": "", "details": {"stop_evidence": {
            "stopped": True, "source": "plugin", "reference": "stop-1",
        }}}
        self.assertEqual(contracts.Feedback.parse(valid).to_dict(), valid)
        omitted_reason = {**valid}
        del omitted_reason["reason_code"]
        parsed_default = contracts.Feedback.parse(omitted_reason)
        self.assertEqual(parsed_default.reason_code, "")
        self.assertEqual(parsed_default.to_dict(), {**omitted_reason, "reason_code": ""})
        for key in ("source", "reference"):
            with self.subTest(overlong_evidence=key):
                bad = {**canceled, "details": {"stop_evidence": {
                    "stopped": True, key: "x" * 257,
                }}}
                self.assert_contract_error("text_too_long", f"details.stop_evidence.{key}",
                                           lambda: contracts.Feedback.parse(bad))

    def test_oversized_json_integer_is_a_contract_error(self):
        payload = goal()
        payload["parameters"][ACTION]["counter"] = 10**4096
        self.assert_contract_error("value_budget_exceeded", "goal_submit",
                                   lambda: contracts.GoalSubmit.parse(payload))

    def test_nested_enum_rejects_boolean_as_integer(self):
        schema = {"type": "object", "properties": {"target": {"type": "object",
                  "properties": {"selected": {"enum": [1]}}, "required": ["selected"],
                  "additionalProperties": False}}, "required": ["target"]}
        errors = contracts.validate_params(schema, {"target": {"selected": True}})
        self.assertEqual([(e.code, e.path) for e in errors], [("enum_violation", "params.target.selected")])
        self.assertFalse(contracts.validate_params(schema, {"target": {"selected": 1}}))

    def test_nested_nonfinite_and_cycle_rejected(self):
        payload = goal()
        payload["parameters"][ACTION]["target"]["depth_m"] = float("inf")
        self.assert_contract_error("non_finite_number", "parameters." + ACTION + ".target.depth_m",
                                   lambda: contracts.GoalSubmit.parse(payload))
        cycle = {}
        cycle["self"] = cycle
        payload = goal()
        payload["parameters"][ACTION]["target"] = cycle
        self.assert_contract_error("invalid_type", "parameters." + ACTION + ".target.self",
                                   lambda: contracts.GoalSubmit.parse(payload))

    def test_nested_payload_budget_limits(self):
        payload = goal()
        payload["parameters"][ACTION]["blob"] = "x" * 1_048_576
        self.assert_contract_error("value_budget_exceeded", "goal_submit",
                                   lambda: contracts.GoalSubmit.parse(payload))
        too_deep = {}
        current = too_deep
        for _ in range(34):
            current["child"] = {}
            current = current["child"]
        payload = goal()
        payload["parameters"][ACTION] = too_deep
        self.assert_contract_error("schema_depth_exceeded", "goal_submit",
                                   lambda: contracts.GoalSubmit.parse(payload))


if __name__ == "__main__":
    unittest.main()
