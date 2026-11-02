from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from astrbot_plugin_astrbotex_interaction.task_models import TaskAuthority, TaskError
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
from astrbot_plugin_astrbotex_interaction.upper_store import (
    UpperStore, input_key, normalize_result, payload_hash,
)

WORK_ROOT = Path(__file__).resolve().parents[2]


def authority(session="s1", robot="r1"):
    return TaskAuthority(robot, session, "user-" + session, "route-" + session, True)


def result(route="speech_and_action", public_text="收到，我来处理。",
           segments=("寻找红色杯子。", "接近并对齐同一个目标。", "按已配置方案抓取同一个目标。"),
           send_count=2, probabilities=None):
    return {
        "route": route,
        "public_text": public_text,
        "segments": list(segments),
        "send_count": send_count,
        "probabilities": probabilities or {
            "speech_only": 0.1, "speech_and_action": 0.75, "action_only": 0.15,
        },
    }


class UpperStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "tasks.db")
        self.store = TaskStore(self.path)
        self.addCleanup(self.store.close)
        self.upper = UpperStore(self.store)

    def claim(self, message_id="m1", platform="onebot", origin="group:1"):
        return self.upper.claim_input(
            platform=platform, origin=origin, source_message_id=message_id,
            robot_id="r1", user_id="u1", session_id="s1",
        )

    def planned(self, message_id="m1"):
        record, _ = self.claim(message_id)
        plan = self.upper.save_result(record["input_key"], result(), generation=record["generation"])
        return record, plan

    def reserved(self, message_id="m1"):
        record, plan = self.planned(message_id)
        batch = self.upper.reserve_batch(
            plan["plan_id"], peer=b"ex-control", ex_session="ex-1",
            robot_id="r1", route_ref="route-1",
        )
        return record, plan, batch

    # ------------------------------------------------------------------
    def test_three_segments_send_two_survive_restart_byte_identical(self):
        _, plan, batch = self.reserved()
        plan = self.upper.get_plan(plan["plan_id"])
        self.assertEqual(len(plan["segments"]), 3)
        self.assertEqual([s["status"] for s in plan["segments"]],
                         ["sending", "sending", "pending"])
        self.assertEqual(batch["first_ordinal"], 0)
        self.assertEqual(batch["last_ordinal"], 1)
        self.assertEqual([s["ordinal"] for s in batch["payload"]["segments"]], [0, 1])

        frozen = {
            "plan_id": plan["plan_id"],
            "segment_ids": [s["segment_id"] for s in plan["segments"]],
            "payload": batch["payload"],
            "payload_hash": batch["payload_hash"],
        }
        self.upper.store.close()

        reopened = TaskStore(self.path, recover=False)
        self.addCleanup(reopened.close)
        upper = UpperStore(reopened)
        after = upper.get_plan(plan["plan_id"])
        self.assertEqual([s["segment_id"] for s in after["segments"]], frozen["segment_ids"])
        self.assertEqual(after["batch"]["payload"], frozen["payload"])
        self.assertEqual(after["batch"]["payload_hash"], frozen["payload_hash"])
        # Restart downgrades the unresolved window instead of replaying it.
        self.assertEqual(after["batch"]["status"], "delivery_unknown")
        self.assertEqual(after["batch"]["fence"], 1)
        self.assertEqual([s["status"] for s in after["segments"]],
                         ["delivery_unknown", "delivery_unknown", "pending"])
        self.assertEqual(reopened.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_twenty_concurrent_duplicate_claims_yield_one_plan_and_one_reservation(self):
        def worker(index):
            record, claimed = self.claim("same-message")
            if not claimed:
                # A duplicate handler must reuse the existing generation; it must
                # never mint a second plan or a second batch.
                plan = self.upper.get_plan(record["plan_id"]) if record["plan_id"] else None
                return ("duplicate", plan["plan_id"] if plan else None)
            plan = self.upper.save_result(record["input_key"], result(),
                                          generation=record["generation"])
            batch = self.upper.reserve_batch(plan["plan_id"], peer=b"ex-control",
                                             ex_session="ex-1", robot_id="r1",
                                             route_ref="route-1")
            return ("owner", plan["plan_id"], batch["batch_id"])

        with ThreadPoolExecutor(20) as pool:
            outcomes = list(pool.map(worker, range(20)))

        owners = [o for o in outcomes if o[0] == "owner"]
        self.assertEqual(len(owners), 1)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM upper_plans").fetchone()[0], 1)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM upper_batches").fetchone()[0], 1)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM upper_segments").fetchone()[0], 3)

    def test_identical_text_from_distinct_messages_stays_distinct(self):
        first, _ = self.claim("m-a")
        second, _ = self.claim("m-b")
        self.assertNotEqual(first["input_key"], second["input_key"])
        plan_a = self.upper.save_result(first["input_key"], result(), generation=1)
        plan_b = self.upper.save_result(second["input_key"], result(), generation=1)
        self.assertNotEqual(plan_a["plan_id"], plan_b["plan_id"])
        # Same content, different identity: the store must not collapse them.
        self.assertEqual(plan_a["result_hash"], plan_b["result_hash"])

    def test_same_message_never_produces_a_second_plan(self):
        record, plan = self.planned()
        again = self.upper.save_result(record["input_key"], result(), generation=1)
        self.assertEqual(again["plan_id"], plan["plan_id"])
        self.assertEqual(again["segments"], plan["segments"])
        with self.assertRaisesRegex(TaskError, "id_conflict"):
            self.upper.save_result(record["input_key"], result(public_text="改口了"), generation=1)

    def test_stale_generation_is_rejected(self):
        record, _ = self.claim()
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            self.upper.save_result(record["input_key"], result(), generation=2)
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            self.upper.reject_input(record["input_key"], generation=9, error_code="x")

    def test_delivery_unknown_is_terminal_and_never_resent(self):
        _, plan, batch = self.reserved()
        self.upper.record_delivery(batch["batch_id"], "delivery_unknown",
                                   error_code="send_timeout")
        # The suffix stays pending; nothing advances it.
        after = self.upper.get_plan(plan["plan_id"])
        self.assertEqual([s["status"] for s in after["segments"]],
                         ["delivery_unknown", "delivery_unknown", "pending"])
        # Re-reserving returns the frozen envelope, it does not mint a new id.
        again = self.upper.reserve_batch(plan["plan_id"], peer=b"ex-control",
                                         ex_session="ex-1", robot_id="r1",
                                         route_ref="route-1")
        self.assertEqual(again["batch_id"], batch["batch_id"])
        self.assertEqual(again["payload_hash"], batch["payload_hash"])
        with self.assertRaisesRegex(TaskError, "invalid_transition"):
            self.upper.record_delivery(batch["batch_id"], "injected")
        # Recovery does not resurrect a terminal batch either.
        self.upper.recover()
        self.assertEqual(self.upper.get_plan(plan["plan_id"])["batch"]["status"],
                         "delivery_unknown")

    def test_received_does_not_advance_the_suffix(self):
        _, plan, batch = self.reserved()
        self.upper.record_delivery(batch["batch_id"], "received",
                                   receipt={"batch_id": batch["batch_id"],
                                            "request_id": batch["request_id"],
                                            "payload_hash": batch["payload_hash"]})
        after = self.upper.get_plan(plan["plan_id"])
        self.assertEqual([s["status"] for s in after["segments"]],
                         ["received", "received", "pending"])
        self.assertEqual(after["status"], "dispatched")

    def test_mismatched_receipt_is_refused_and_leaves_state_intact(self):
        _, plan, batch = self.reserved()
        for broken in ({"batch_id": "other", "request_id": batch["request_id"],
                        "payload_hash": batch["payload_hash"]},
                       {"batch_id": batch["batch_id"], "request_id": "other",
                        "payload_hash": batch["payload_hash"]},
                       {"batch_id": batch["batch_id"], "request_id": batch["request_id"],
                        "payload_hash": "0" * 64}):
            with self.assertRaisesRegex(TaskError, "delivery_unknown"):
                self.upper.record_delivery(batch["batch_id"], "injected", receipt=broken)
            self.assertEqual(self.upper.get_plan(plan["plan_id"])["batch"]["status"], "sending")
            self.assertEqual([s["status"] for s in self.upper.get_plan(plan["plan_id"])["segments"]],
                             ["sending", "sending", "pending"])

    def test_public_reply_claims_once(self):
        _, plan = self.planned()
        reply, claimed = self.upper.claim_public_reply(plan["plan_id"])
        self.assertTrue(claimed)
        self.assertEqual(reply["text"], "收到，我来处理。")
        again, second = self.upper.claim_public_reply(plan["plan_id"])
        self.assertFalse(second)
        self.assertEqual(again["reply_id"], reply["reply_id"])
        self.upper.record_public_delivery(reply["reply_id"], "sent")
        with self.assertRaisesRegex(TaskError, "invalid_transition"):
            self.upper.record_public_delivery(reply["reply_id"], "delivery_unknown")

    def test_action_only_has_no_public_exit(self):
        record, _ = self.claim()
        plan = self.upper.save_result(
            record["input_key"],
            result(route="action_only", public_text=None, send_count=1),
            generation=1)
        self.assertIsNone(plan["public_text"])
        with self.assertRaisesRegex(TaskError, "no_public_text"):
            self.upper.claim_public_reply(plan["plan_id"])

    def test_speech_only_reserves_nothing(self):
        record, _ = self.claim()
        plan = self.upper.save_result(
            record["input_key"], result(route="speech_only", segments=(), send_count=0),
            generation=1)
        with self.assertRaisesRegex(TaskError, "invalid_batch"):
            self.upper.reserve_batch(plan["plan_id"], peer=b"p", ex_session="ex-1",
                                     robot_id="r1", route_ref="route-1")

    def test_projection_hides_production_context(self):
        record, _ = self.claim()
        plan = self.upper.save_result(
            record["input_key"], result(), generation=1,
            settings_snapshot={"provider_api_key": "sk-secret", "port": 8769},
            prompt_snapshot={"planning_prompt": "internal rules"},
            settings_hash="sh", prompt_hash="ph", catalog_hash="ch")
        self.assertNotIn("sk-secret", json.dumps(plan, ensure_ascii=False))
        self.assertNotIn("internal rules", json.dumps(plan, ensure_ascii=False))
        self.assertEqual((plan["settings_hash"], plan["catalog_hash"], plan["prompt_hash"]),
                         ("sh", "ch", "ph"))
        listed = self.upper.list_plans()
        self.assertEqual([item["plan_id"] for item in listed], [plan["plan_id"]])

    def test_legacy_tables_are_untouched_by_the_extension(self):
        legacy = self.store.create(authority(), "three steps", "create-1")
        before = self.store.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        before_data = self.store.get(legacy["task_id"])
        self.reserved("legacy-migration")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], before)
        self.assertEqual(self.store.get(legacy["task_id"])["steps"], before_data["steps"])
        self.assertEqual(self.store.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        # Re-opening the store must not alter legacy rows or their recovery path.
        self.upper.store.close()
        reopened = TaskStore(self.path)
        self.addCleanup(reopened.close)
        UpperStore(reopened)
        restored = reopened.get(legacy["task_id"])
        self.assertEqual(restored["status"], "resume_review")
        self.assertEqual(restored["steps"], before_data["steps"])

    def test_crash_between_reserve_and_receipt_is_unresolved_not_retried(self):
        record, _ = self.claim()
        plan = self.upper.save_result(record["input_key"], result(), generation=1)
        script = textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {str(WORK_ROOT)!r})
            from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
            from astrbot_plugin_astrbotex_interaction.upper_store import UpperStore
            store = TaskStore({self.path!r}, recover=False)
            upper = UpperStore(store)
            upper.reserve_batch({plan["plan_id"]!r}, peer=b"ex-control",
                                ex_session="ex-1", robot_id="r1", route_ref="route-1")
            sys.stdout.write("reserved")
            sys.stdout.flush()
            os._exit(9)
        """)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(WORK_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True,
                              text=True, env=env, timeout=120)
        self.assertEqual(proc.returncode, 9, proc.stderr)
        self.assertEqual(proc.stdout, "reserved")

        reopened = TaskStore(self.path, recover=False)
        self.addCleanup(reopened.close)
        upper = UpperStore(reopened)
        after = upper.get_plan(plan["plan_id"])
        self.assertEqual(after["batch"]["status"], "delivery_unknown")
        self.assertEqual(after["batch"]["fence"], 1)
        self.assertEqual(after["batch"]["error_code"], "restart_unresolved")
        self.assertEqual([s["status"] for s in after["segments"]],
                         ["delivery_unknown", "delivery_unknown", "pending"])
        # A second restart must not escalate the fence again.
        reopened.close()
        third = TaskStore(self.path, recover=False)
        self.addCleanup(third.close)
        self.assertEqual(UpperStore(third).get_plan(plan["plan_id"])["batch"]["fence"], 1)


class UpperContractGuardTests(unittest.TestCase):
    """Store-side invariants that B00 also owns, asserted independently here."""

    def test_route_field_combinations(self):
        for broken, code in (
            (result(route="speech_only", segments=("x",), send_count=1), "route_mismatch"),
            (result(route="speech_only", segments=(), send_count=0, public_text=None),
             "route_mismatch"),
            (result(route="action_only", public_text="hi", send_count=1), "route_mismatch"),
            (result(route="action_only", public_text=None, send_count=0), "route_mismatch"),
            (result(send_count=4), "route_mismatch"),
            (result(send_count=0), "route_mismatch"),
            (result(route="both"), "invalid_result"),
        ):
            with self.assertRaisesRegex(TaskError, code):
                normalize_result(broken)

    def test_probabilities_are_strict(self):
        base = result()["probabilities"]
        for probabilities in (
            {**base, "speech_only": True},
            {**base, "speech_only": "0.1"},
            {**base, "speech_only": float("nan")},
            {**base, "speech_only": 1.5},
            {"speech_only": 0.5, "speech_and_action": 0.5, "action_only": 0.5},
            {"speech_only": 0.5, "speech_and_action": 0.5},
        ):
            with self.assertRaisesRegex(TaskError, "invalid_probabilities"):
                normalize_result(result(probabilities=probabilities))

    def test_payload_hash_is_stable_and_identity_free(self):
        left = {"b": 1, "a": "中文", "c": [1, 2]}
        right = {"c": [1, 2], "a": "中文", "b": 1}
        self.assertEqual(payload_hash(left), payload_hash(right))
        self.assertEqual(payload_hash(left), payload_hash(dict(left)))
        self.assertNotEqual(payload_hash(left), payload_hash({"b": 2, "a": "中文", "c": [1, 2]}))

    def test_input_key_never_collides_across_identity(self):
        keys = {
            input_key("onebot", "group:1", "m1"),
            input_key("onebot", "group:2", "m1"),
            input_key("astrbotex", "group:1", "m1"),
            input_key("onebot", "group:1", "m2"),
        }
        self.assertEqual(len(keys), 4)

    def test_payload_too_large_is_refused_not_truncated(self):
        huge = "字" * 4097
        with self.assertRaisesRegex(TaskError, "payload_too_large"):
            normalize_result(result(segments=(huge,), send_count=1))
        many = tuple("s" for _ in range(65))
        with self.assertRaisesRegex(TaskError, "invalid_result"):
            normalize_result(result(segments=many, send_count=1))


if __name__ == "__main__":
    unittest.main()
