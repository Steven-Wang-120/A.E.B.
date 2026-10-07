"""Host-bound task operations and read-only peer projection; no chat hooks here."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from .task_contracts import parse_action_manifest, require_id, require_sequence, require_text
from .task_models import TaskAuthority, TaskError, canonical, json_value


def host_message_key(authority: TaskAuthority, message_id: str) -> str:
    authority.validate()
    require_id(message_id, "host_message_id")
    identity = [getattr(authority, key) for key in
                ("robot_id", "session_id", "user_id", "route_ref")]
    return hashlib.sha256(canonical([identity, message_id]).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class HostTaskContext:
    """Built only by Host admission from trusted event/config/EX context, not tool args."""
    authority: TaskAuthority
    peer: bytes
    message_id: str
    ex_session: str
    revision: int
    provider_id: str = ""

    def validate(self) -> None:
        self.authority.validate()
        require_id(self.message_id, "host_message_id")
        require_id(self.ex_session, "ex_session")
        require_sequence(self.revision, "revision")
        if not isinstance(self.peer, bytes) or not self.peer:
            raise TaskError("invalid_task_peer")

    @property
    def message_key(self) -> str:
        self.validate()
        return host_message_key(self.authority, self.message_id)


class HostTaskOperations:
    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator
        self.store = coordinator.store

    async def apply(self, context: HostTaskContext, intent: dict) -> dict:
        context.validate()
        value = json_value(intent)
        if (not isinstance(value, dict) or not isinstance(value.get("operation"), str)
                or value["operation"] not in {"create", "update", "cancel", "review"}):
            raise TaskError("invalid_host_operation")
        operation = value["operation"]
        if set(value) != ({"operation"} if operation in {"cancel", "review"} else {"operation", "text"}):
            raise TaskError("business_intent_only")
        text = value.get("text")
        if operation in {"create", "update"}:
            require_text(text, "task_text", max_len=8192)
        route = self.store.route(context.authority.route_ref)
        if (not route or route.get("peer_hex") != context.peer.hex()
                or any(route.get(k) != getattr(context.authority, k)
                       for k in ("robot_id", "session_id", "user_id", "route_ref"))):
            raise TaskError("route_owner_mismatch")
        key = context.message_key
        if operation == "create":
            prior = self.store.host_message(key)
            task = self.store.create(context.authority, text, key,
                ex_session=context.ex_session, revision=context.revision, provider_id=context.provider_id)
            if prior is None:
                self.coordinator.wake(task["task_id"])
        else:
            task, claimed = self.store.reserve_owned_host_operation(context.authority, key, operation, text,
                                                                  ex_session=None if operation == "review" else context.ex_session)
            if claimed:
                if operation == "update":
                    await self.coordinator.user_input(task["task_id"], context.authority, text)
                elif operation == "review":
                    await self.coordinator.review_resume(task["task_id"], context.authority)
                else:
                    await self.coordinator.cancel_task(task["task_id"], context.authority)
                self.store.finish_host_operation(key)
            task = self.store.get(task["task_id"])
        return {"task_id": task["task_id"], "status": task["status"], "message_key": key}


def parse_capabilities(raw: dict) -> dict:
    value = json_value(raw)
    if (not isinstance(value, dict) or set(value) != {"schema_version", "ex_session", "revision",
            "catalog_revision", "control_mode", "execution", "actions"}
            or type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("control_mode") not in ("legacy", "decision")):
        raise TaskError("invalid_capabilities")
    require_id(value["ex_session"], "ex_session")
    for key in ("revision", "catalog_revision"):
        require_sequence(value[key], key)
    execution = value["execution"]
    if (not isinstance(execution, dict) or set(execution) != {"mode", "execution_allowed", "runtime_state"}
            or execution["mode"] not in ("disabled", "shadow", "execute")
            or type(execution["execution_allowed"]) is not bool
            or execution["runtime_state"] not in ("idle", "ready", "running", "paused", "fault", "finished")):
        raise TaskError("invalid_capabilities")
    actions = value["actions"]
    if not isinstance(actions, list) or len(actions) > 256:
        raise TaskError("invalid_capabilities")
    ids = set()
    for action in actions:
        if not isinstance(action, dict):
            raise TaskError("invalid_capabilities")
        require_id(action.get("owner"), "owner")
        require_sequence(action.get("plugin_generation"), "plugin_generation")
        declaration = {k: v for k, v in action.items() if k not in {"owner", "plugin_generation"}}
        parsed = parse_action_manifest({"action_api_version": 2, "actions": [declaration]}, owner=action["owner"])
        action_id = parsed.actions[0].action_id
        if action_id in ids:
            raise TaskError("invalid_capabilities")
        ids.add(action_id)
    return value


def execution_ready(capabilities: dict) -> bool:
    execution = capabilities["execution"]
    return (capabilities["control_mode"] == "decision" and execution["mode"] == "execute"
            and execution["execution_allowed"] is True and execution["runtime_state"] == "running"
            and bool(capabilities["actions"]))


def _short(text: str, limit: int = 160) -> str:
    return " ".join(text.split())[:limit]


def task_projection(store, peer: bytes, request: dict, *, ex_session: str, robot_id: str) -> dict:
    """Trusted transport/config owns peer/session/robot; no caller-selected task or user."""
    require_id(ex_session, "ex_session")
    require_id(robot_id, "robot_id")
    if (not isinstance(request, dict) or set(request) != {"schema_version", "ex_session"}
            or type(request.get("schema_version")) is not int or request["schema_version"] != 1):
        raise TaskError("invalid_projection_request")
    if request["ex_session"] != ex_session:
        raise TaskError("stale_ex_session")
    if not isinstance(peer, bytes) or not peer:
        raise TaskError("invalid_task_peer")
    with store._lock:
        routes = [r for r in store.routes() if r.get("peer_hex") == peer.hex()]
        if not routes:
            raise TaskError("projection_peer_mismatch")
        result = {"schema_version": 1, "ex_session": ex_session, "robot_id": robot_id,
                  "task_id": None, "generation": None, "turn_id": None,
                  "available": False, "phase": "unavailable", "title": "", "current_goal": None,
                  "completed": 0, "total": 0, "can_cancel": False,
                  "updated_at": time.time(), "message": "Task scope unavailable"}
        if {r.get("robot_id") for r in routes} != {robot_id}:
            return result
        active = [t for t in store.active_tasks() if t["robot_id"] == robot_id]
        scoped = []
        for task in active:
            route = next((r for r in routes if r["route_ref"] == task["route_ref"]), None)
            if (not route or any(route.get(k) != task[k] for k in
                    ("robot_id", "session_id", "user_id", "route_ref"))
                    or task["ex_session"] != ex_session):
                return result  # Unresolved old-session/unbound task must never appear idle.
            scoped.append(task)
        if len(scoped) > 1:
            return result
        if not scoped:
            result.update(available=True, phase="idle", message="No active task")
            return result
        task = scoped[0]
        goal = task["current_goal"]
        current = _short(goal["payload"]["goal_text_en"]) if goal else None
        result.update(task_id=task["task_id"], generation=task["generation"], turn_id=task["turn_id"],
            available=True, phase=task["status"], title=_short(task["text"]), current_goal=current,
            completed=sum(s["status"] == "completed" for s in task["steps"]), total=len(task["steps"]),
            updated_at=task.get("updated_at", 0.0), message="Current task")
        return result
