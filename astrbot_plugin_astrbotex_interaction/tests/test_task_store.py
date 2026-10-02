from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from astrbot_plugin_astrbotex_interaction.task_models import TaskAuthority, TaskError
from astrbot_plugin_astrbotex_interaction.task_store import TaskStore


def authority(session="s1", robot="r1"):
    return TaskAuthority(robot, session, "user-" + session, "route-" + session, True)


def steps(count=3, prefix="step"):
    return [{"step_id": f"{prefix}-{i}", "intent": f"Do bounded step {i}",
             "completion_condition": "Observed completion"} for i in range(count)]


class TaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "tasks.db")
        self.store = TaskStore(self.path)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.store.close)

    def create(self):
        return self.store.create(authority(), "three steps", "create-1")

    def test_P05_owner_and_creation_idempotency(self):
        task = self.create()
        self.assertEqual(task["task_id"], self.create()["task_id"])
        with self.assertRaisesRegex(TaskError, "robot_owned"):
            self.store.create(authority("s2"), "steal", "create-2")
        with self.assertRaisesRegex(TaskError, "owner_mismatch"):
            self.store.invalidate(task["task_id"], authority("s2"))
        with self.assertRaisesRegex(TaskError, "unauthorized"):
            self.store.create(TaskAuthority("r2", "s", "u", "route", False), "task", "x")
        with self.assertRaisesRegex(TaskError, "duplicate_request_id_conflict"):
            self.store.create(authority(), "changed", "create-1")

    def test_atomic_creation_across_connections(self):
        other = TaskStore(self.path, recover=False)
        self.addCleanup(other.close)
        def create(pair):
            db, session = pair
            try:
                return db.create(authority(session), "task", session)["task_id"]
            except TaskError:
                return None
        with ThreadPoolExecutor(2) as pool:
            ids = list(pool.map(create, [(self.store, "a"), (other, "b")]))
        self.assertEqual(sum(x is not None for x in ids), 1)
        self.assertEqual(len(self.store.active_tasks()), 1)

    def test_CAS_intents_and_P07_generation(self):
        task = self.create()
        turn = self.store.begin_turn(task["task_id"])
        self.assertEqual(self.store.save_plan(turn, steps(), 0), 1)
        with self.assertRaisesRegex(TaskError, "revision_conflict"):
            self.store.save_plan(turn, steps(), 0)
        with self.assertRaisesRegex(TaskError, "intent_only_steps_required"):
            self.store.save_plan(turn, [{**steps()[0], "bbox": [1, 2, 3, 4]}], 1)
        with self.assertRaisesRegex(TaskError, "planning_in_progress"):
            self.store.begin_turn(task["task_id"])
        self.store.invalidate(task["task_id"], authority(), "replacement")
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            self.store.save_plan(turn, steps(), 1)

    def test_P08_restart_review_no_replay(self):
        task = self.create()
        turn = self.store.begin_turn(task["task_id"])
        self.store.save_plan(turn, steps(), 0)
        reopened = TaskStore(self.path)
        self.addCleanup(reopened.close)
        restored = reopened.get(task["task_id"])
        self.assertEqual(restored["status"], "resume_review")
        self.assertFalse(restored["needs_planning"])
        self.assertGreater(restored["generation"], turn.generation)
        with self.assertRaisesRegex(TaskError, "review_required"):
            reopened.begin_turn(task["task_id"])

    def test_reconcile_cannot_regress_or_jump_unapplied_cursor(self):
        task = self.create()
        self.store.mutate(task["task_id"], lambda t: t.update(ex_session="ex1"))
        state = {"ex_session": "ex1", "revision": 1, "event_seq": 8, "execution": {}}
        self.store.reconcile(task["task_id"], state, lambda *args: None)
        self.assertEqual(self.store.cursor("r1", "ex1"), 0)
        self.store.db.execute("UPDATE cursors SET seq=5 WHERE robot_id='r1' AND ex_session='ex1'")
        self.store.reconcile(task["task_id"], state, lambda *args: None)
        self.assertEqual(self.store.cursor("r1", "ex1"), 5)
        before = self.store.get(task["task_id"])
        applied = []
        def clobber(current, restored):
            applied.append(restored)
            current.update(text="stale overwrite", goal_revision=99)
        for stale in ({**state, "event_seq": 4, "revision": 99},
                      {**state, "event_seq": 9, "revision": 0}):
            with self.assertRaisesRegex(TaskError, "stale_state"):
                self.store.reconcile(task["task_id"], stale, clobber, cropped=True)
            self.assertEqual(self.store.cursor("r1", "ex1"), 5)
            self.assertEqual(self.store.get(task["task_id"]), before)
        self.assertEqual(applied, [])
        self.assertEqual(self.store.cursor("r1", "ex2"), 0)
        with self.assertRaisesRegex(TaskError, "stale_ex_session"):
            self.store.reconcile(task["task_id"], {**state, "ex_session": "foreign"}, lambda *args: None)
        self.assertEqual(self.store.get(task["task_id"])["ex_session"], "ex1")

    def test_transaction_rollback_and_defensive_copy(self):
        task = self.create()
        def bad(t):
            t["text"] = "clobbered"
            raise TaskError("rollback")
        with self.assertRaises(TaskError):
            self.store.mutate(task["task_id"], bad)
        self.assertEqual(self.store.get(task["task_id"])["text"], "three steps")
        task["text"] = "local"
        self.assertEqual(self.store.get(task["task_id"])["text"], "three steps")
