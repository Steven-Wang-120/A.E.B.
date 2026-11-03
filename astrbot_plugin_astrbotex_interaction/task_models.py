"""AEB-owned task state. Frozen decision wire models live in task_contracts."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

from .task_contracts import measure_json_budget, require_id, require_text


class TaskError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def plain(value: Any) -> Any:
    """Convert frozen dataclasses and tuples into JSON-shaped values.

    The budget walker in ``task_contracts`` only understands dict/list/scalar,
    so a ``SettingsSnapshot`` or ``CatalogSnapshot`` handed to ``canonical``
    used to be rejected as ``invalid_type``. Every snapshot B02/B04 produces is
    a frozen dataclass, which made a plan impossible to persist at all.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: plain(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [plain(item) for item in sorted(value, key=repr)]
    return value


def json_value(value: Any) -> Any:
    value = plain(value)
    violation = measure_json_budget(value)
    if violation:
        # Report the walker's own code: "invalid_type" and "value_budget_exceeded"
        # are different failures and callers branch on them.
        raise TaskError(violation)
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (ValueError, TypeError, RecursionError) as exc:
        raise TaskError("invalid_json") from exc


def canonical(value: Any) -> str:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class TaskAuthority:
    """Created by framework admission, never reconstructed from model arguments."""
    robot_id: str
    session_id: str
    user_id: str
    route_ref: str
    authorized: bool = False

    def validate(self) -> None:
        for key in ("robot_id", "session_id", "user_id", "route_ref"):
            require_id(getattr(self, key), key)
        if self.authorized is not True:
            raise TaskError("unauthorized")


@dataclass(frozen=True, slots=True)
class PlanningTurn:
    task_id: str
    robot_id: str
    turn_id: str
    generation: int
    route_ref: str
    visibility: str = "private_planning"


def validate_steps(steps: Any) -> list[dict[str, Any]]:
    value = json_value(steps)
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise TaskError("invalid_steps")
    ids = set()
    result = []
    for step in value:
        if not isinstance(step, dict) or set(step) != {"step_id", "intent", "completion_condition"}:
            raise TaskError("intent_only_steps_required")
        require_id(step["step_id"], "step_id")
        require_text(step["intent"], "intent", max_len=4096)
        require_text(step["completion_condition"], "completion_condition", max_len=4096)
        if step["step_id"] in ids:
            raise TaskError("duplicate_step_id")
        ids.add(step["step_id"])
        result.append({**step, "status": "planned"})
    return result
