"""SQLite durable receipt, request outbox and per-robot control ownership."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Callable

from .task_contracts import Feedback
from .task_models import (
    PlanningTurn, TaskAuthority, TaskError, canonical, json_value, validate_steps,
)


class TaskStore:
    def __init__(self, path: str, *, recover: bool = True) -> None:
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY, robot_id TEXT NOT NULL,
                active INTEGER NOT NULL, data TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_robot
                ON tasks(robot_id) WHERE active=1;
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                payload TEXT NOT NULL, result TEXT);
            CREATE TABLE IF NOT EXISTS receipts (
                robot_id TEXT NOT NULL, ex_session TEXT NOT NULL,
                seq INTEGER NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(robot_id, ex_session, seq));
            CREATE TABLE IF NOT EXISTS cursors (
                robot_id TEXT NOT NULL, ex_session TEXT NOT NULL,
                seq INTEGER NOT NULL, PRIMARY KEY(robot_id, ex_session));
            CREATE TABLE IF NOT EXISTS task_routes (
                route_ref TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS public_messages (
                message_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                payload TEXT NOT NULL, status TEXT NOT NULL);
        """)
        if recover:
            with self.transaction():
                for row in self.db.execute("SELECT data FROM tasks WHERE active=1").fetchall():
                    task = json.loads(row["data"])
                    task["status"] = "resume_review"
                    task["generation"] += 1
                    task["turn_id"] = None
                    task["needs_planning"] = False
                    self._put(task)

    @contextmanager
    def transaction(self):
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            else:
                self.db.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def bind_route(self, authority: TaskAuthority, peer: bytes, *, origin: str = "",
                   source_session: str = "") -> None:
        authority.validate()
        if not isinstance(peer, bytes) or not peer:
            raise TaskError("invalid_task_peer")
        data = {k: getattr(authority, k) for k in ("robot_id", "session_id", "user_id", "route_ref")}
        data.update(peer_hex=peer.hex(), origin=origin, source_session=source_session)
        with self.transaction():
            row = self.db.execute("SELECT data FROM task_routes WHERE route_ref=?", (authority.route_ref,)).fetchone()
            if row and json.loads(row[0]) != data:
                raise TaskError("route_owner_mismatch")
            self.db.execute("INSERT OR IGNORE INTO task_routes VALUES(?,?)", (authority.route_ref, canonical(data)))

    def route(self, route_ref: str) -> dict | None:
        with self._lock:
            row = self.db.execute("SELECT data FROM task_routes WHERE route_ref=?", (route_ref,)).fetchone()
            return json.loads(row[0]) if row else None

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute("SELECT data FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise TaskError("unknown_task")
            return json.loads(row["data"])

    def active_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            return [json.loads(r[0]) for r in self.db.execute("SELECT data FROM tasks WHERE active=1")]

    def _put(self, task: dict[str, Any]) -> None:
        self.db.execute("UPDATE tasks SET active=?, data=? WHERE task_id=?",
                        (int(task["active"]), canonical(task), task["task_id"]))

    def create(self, authority: TaskAuthority, text: str, request_id: str) -> dict[str, Any]:
        authority.validate()
        from .task_contracts import require_id, require_text
        require_id(request_id, "request_id")
        require_text(text, "task_text", max_len=8192)
        payload = {"robot_id": authority.robot_id, "session_id": authority.session_id,
                   "user_id": authority.user_id, "route_ref": authority.route_ref, "text": text}
        with self.transaction():
            old = self.request(request_id, payload)
            if old is not None:
                return self.get(old["task_id"])
            occupied = self.db.execute("SELECT task_id FROM tasks WHERE robot_id=? AND active=1",
                                       (authority.robot_id,)).fetchone()
            if occupied:
                raise TaskError("robot_owned")
            task_id = uuid.uuid4().hex
            task = {**payload, "task_id": task_id, "active": True, "status": "planning",
                    "plan_revision": 0, "steps": [], "active_step": 0,
                    "current_goal": None, "goal_revision": 0, "ex_session": None,
                    "last_event_seq": 0, "generation": 0, "turn_id": None,
                    "public_message_id": None, "needs_planning": True,
                    "planning_failures": 0, "error_code": None}
            self.db.execute("INSERT INTO tasks VALUES(?,?,1,?)", (task_id, authority.robot_id, canonical(task)))
            self.db.execute("INSERT INTO requests VALUES(?,?,?,?)",
                            (request_id, task_id, canonical(payload), canonical({"task_id": task_id})))
            return task

    def authorize(self, task_id: str, authority: TaskAuthority) -> dict[str, Any]:
        authority.validate()
        task = self.get(task_id)
        if any(task[k] != getattr(authority, k) for k in ("robot_id", "session_id", "user_id", "route_ref")):
            raise TaskError("owner_mismatch")
        return task

    def mutate(self, task_id: str, fn: Callable, *, turn: PlanningTurn | None = None) -> Any:
        with self.transaction():
            task = self.get(task_id)
            if turn is not None:
                self.check_turn(turn, task)
            result = fn(task)
            self._put(task)
            return result

    def check_turn(self, turn: PlanningTurn, task: dict | None = None) -> None:
        task = task or self.get(turn.task_id)
        if (not task["active"] or task["generation"] != turn.generation
                or task["turn_id"] != turn.turn_id or task["route_ref"] != turn.route_ref
                or task["robot_id"] != turn.robot_id or turn.visibility != "private_planning"
                or task["status"] in {"resume_review", "lease_lost", "canceling"}):
            raise TaskError("stale_generation")

    def begin_turn(self, task_id: str) -> PlanningTurn:
        def begin(task):
            if not task["active"] or task["status"] in {"resume_review", "lease_lost", "canceling"}:
                raise TaskError("review_required")
            if task["turn_id"]:
                raise TaskError("planning_in_progress")
            task["generation"] += 1
            task["turn_id"] = uuid.uuid4().hex
            task["needs_planning"] = False
            return PlanningTurn(task_id, task["robot_id"], task["turn_id"], task["generation"], task["route_ref"])
        return self.mutate(task_id, begin)

    def invalidate(self, task_id: str, authority: TaskAuthority, text: str | None = None) -> None:
        self.authorize(task_id, authority)
        def invalidate(task):
            if not task["active"]:
                raise TaskError("task_inactive")
            task["generation"] += 1
            task["turn_id"] = None
            if text is not None:
                from .task_contracts import require_text
                require_text(text, "task_text", max_len=8192)
                task["text"] = text
                task["cancel_requested"] = False
            task["status"] = "canceling" if task["current_goal"] else "planning"
            task["needs_planning"] = not bool(task["current_goal"])
            task["planning_failures"] = 0
        self.mutate(task_id, invalidate)

    def save_plan(self, turn: PlanningTurn, steps: Any, expected_revision: int) -> int:
        planned = validate_steps(steps)
        def save(task):
            if type(expected_revision) is not int or task["plan_revision"] != expected_revision:
                raise TaskError("revision_conflict")
            if task["current_goal"]:
                raise TaskError("active_goal_requires_cancel")
            completed = [s for s in task["steps"] if s["status"] == "completed"]
            if {s["step_id"] for s in completed} & {s["step_id"] for s in planned}:
                raise TaskError("completed_step_replay")
            task["steps"] = completed + planned
            task["active_step"] = len(completed)
            task["plan_revision"] += 1
            task["status"] = "planning"
            return task["plan_revision"]
        return self.mutate(turn.task_id, save, turn=turn)

    def request(self, request_id: str, payload: dict | None = None) -> dict | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                return None
            if payload is not None and row["payload"] != canonical(payload):
                raise TaskError("duplicate_request_id_conflict")
            return {"task_id": row["task_id"], "payload": json.loads(row["payload"]),
                    "result": json.loads(row["result"]) if row["result"] else None}

    @staticmethod
    def lease_identity(task: dict) -> tuple | None:
        goal = task["current_goal"]
        if (not task["active"] or not goal
                or task["status"] in {"resume_review", "lease_lost", "canceling"}
                or goal["phase"] == "rejected"
                or goal.get("feedback_status") in {"succeeded", "failed", "rejected", "canceled", "timed_out", "unknown"}):
            return None
        payload = goal["payload"]
        return (task["task_id"], task["robot_id"], task["route_ref"], task["generation"],
                task["ex_session"], payload["ex_session"], payload["request_id"],
                payload["goal_id"], goal["revision"])

    def prepare_goal(self, turn: PlanningTurn, payload: dict) -> dict:
        def prepare(task):
            prior = self.request(payload["request_id"], payload)
            if prior:
                if prior["task_id"] != task["task_id"]:
                    raise TaskError("owner_mismatch")
                return prior
            if task["current_goal"]:
                raise TaskError("active_goal_exists")
            index = task["active_step"]
            if index >= len(task["steps"]) or task["steps"][index]["step_id"] != payload["step_id"]:
                raise TaskError("not_current_step")
            task["steps"][index]["status"] = "preparing"
            task["current_goal"] = {"payload": json_value(payload), "phase": "submitting", "revision": None}
            task["status"] = "executing"
            self.db.execute("INSERT INTO requests VALUES(?,?,?,NULL)",
                            (payload["request_id"], task["task_id"], canonical(payload)))
            return {"task_id": task["task_id"], "payload": payload, "result": None}
        return self.mutate(turn.task_id, prepare, turn=turn)

    def record_submit(self, task_id: str, payload: dict, result: dict) -> None:
        def record(task):
            self.request(payload["request_id"], payload)
            self.db.execute("UPDATE requests SET result=? WHERE request_id=?", (canonical(result), payload["request_id"]))
            goal = task["current_goal"]
            if not goal or goal["payload"]["request_id"] != payload["request_id"]:
                return
            goal["phase"] = result["phase"]
            goal["revision"] = result["revision"]
            goal.setdefault("feedback_status", "accepted" if result["ok"] else "rejected")
            task["goal_revision"] = result["revision"]
            task["steps"][task["active_step"]]["status"] = "dispatched"
            if result["phase"] == "rejected":
                task["steps"][task["active_step"]]["status"] = "failed"
                task["current_goal"] = None
                task["needs_planning"] = True
        self.mutate(task_id, record)

    def cursor(self, robot_id: str, ex_session: str) -> int:
        with self._lock:
            row = self.db.execute("SELECT seq FROM cursors WHERE robot_id=? AND ex_session=?", (robot_id, ex_session)).fetchone()
            return row[0] if row else 0

    def receive(self, robot_id: str, feedback: Feedback) -> bool:
        """Commit before caller ACKs. Cursor represents applied contiguous facts, not max receipt."""
        payload = canonical(feedback.to_dict())
        with self.transaction():
            row = self.db.execute("SELECT payload FROM receipts WHERE robot_id=? AND ex_session=? AND seq=?",
                                  (robot_id, feedback.ex_session, feedback.event_seq)).fetchone()
            if row:
                if row[0] != payload:
                    raise TaskError("event_seq_conflict")
                return False
            self.db.execute("INSERT INTO receipts VALUES(?,?,?,?)",
                            (robot_id, feedback.ex_session, feedback.event_seq, payload))
            return True

    def apply_receipts(self, robot_id: str, ex_session: str, apply: Callable) -> list[str]:
        wakes = []
        with self.transaction():
            cursor = self.cursor(robot_id, ex_session)
            while True:
                row = self.db.execute("SELECT payload FROM receipts WHERE robot_id=? AND ex_session=? AND seq=?",
                                      (robot_id, ex_session, cursor + 1)).fetchone()
                if row is None:
                    break
                fact = Feedback.parse(json.loads(row[0]))
                try:
                    task = self.get(fact.task_id)
                except TaskError:
                    task = None
                if task and task["robot_id"] == robot_id and task["ex_session"] == ex_session:
                    if apply(task, fact):
                        wakes.append(task["task_id"])
                    task["last_event_seq"] = max(task["last_event_seq"], fact.event_seq)
                    self._put(task)
                cursor += 1
            self.db.execute("INSERT INTO cursors VALUES(?,?,?) ON CONFLICT(robot_id,ex_session) DO UPDATE SET seq=excluded.seq",
                            (robot_id, ex_session, cursor))
        return wakes

    def reconcile(self, task_id: str, state: dict, apply_state: Callable, *,
                  cropped: bool = False) -> None:
        with self.transaction():
            task = self.get(task_id)
            if task["ex_session"] != state["ex_session"]:
                raise TaskError("stale_ex_session")
            cursor = self.cursor(task["robot_id"], state["ex_session"])
            if (state["event_seq"] < max(cursor, task["last_event_seq"])
                    or state["revision"] < task["goal_revision"]):
                raise TaskError("stale_state")
            # The ledger head is not proof that unreturned pages were applied.
            # Cropped recovery also consumes our durable receipts, not just the
            # latest goal summary (which may belong to another task/goal).
            restored = dict(state)
            restored["execution"] = dict(state["execution"])
            restored["execution"]["receipt_facts"] = [json.loads(row[0]) for row in self.db.execute(
                "SELECT payload FROM receipts WHERE robot_id=? AND ex_session=? AND seq<=? ORDER BY seq",
                (task["robot_id"], state["ex_session"], state["event_seq"]))]
            apply_state(task, restored)
            task["goal_revision"] = max(task["goal_revision"], state["revision"])
            task["last_event_seq"] = max(task["last_event_seq"], cursor)
            # Missing evidence cannot authorize dropping the missing history.
            if cropped and task["current_goal"] is None:
                cursor = max(cursor, state["event_seq"])
                task["last_event_seq"] = max(task["last_event_seq"], cursor)
            self._put(task)
            self.db.execute("INSERT INTO cursors VALUES(?,?,?) ON CONFLICT(robot_id,ex_session) DO UPDATE SET seq=MAX(cursors.seq,excluded.seq)",
                            (task["robot_id"], state["ex_session"], cursor))

    def claim_message(self, turn: PlanningTurn, message_id: str, payload: dict) -> bool:
        with self.transaction():
            task = self.get(turn.task_id)
            self.check_turn(turn, task)
            row = self.db.execute("SELECT payload FROM public_messages WHERE message_id=?", (message_id,)).fetchone()
            if row:
                if row[0] != canonical(payload):
                    raise TaskError("duplicate_message_id_conflict")
                return False
            self.db.execute("INSERT INTO public_messages VALUES(?,?,?,?)",
                            (message_id, turn.task_id, canonical(payload), "claimed"))
            task["public_message_id"] = message_id
            self._put(task)
            return True

    def message_result(self, message_id: str, status: str) -> None:
        with self.transaction():
            self.db.execute("UPDATE public_messages SET status=? WHERE message_id=?", (status, message_id))
