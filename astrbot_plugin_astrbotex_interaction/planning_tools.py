"""Bounded, non-streaming planning via the public Host llm_generate API."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .task_contracts import require_id, validate_params
from .task_models import PlanningTurn, TaskError, json_value


TOOL_SCHEMAS = {
    "save_plan": {
        "type": "object", "required": ["steps", "expected_revision"],
        "additionalProperties": False,
        "properties": {
            "expected_revision": {"type": "integer", "minimum": 0},
            "steps": {"type": "array", "minItems": 1, "maxItems": 32, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["step_id", "intent", "completion_condition"],
                "properties": {k: {"type": "string", "minLength": 1, "maxLength": 4096}
                               for k in ("step_id", "intent", "completion_condition")}}},
        },
    },
    "submit_current_goal": {
        "type": "object", "additionalProperties": False,
        "required": ["step_id", "goal_text_en", "allowed_actions", "parameters", "completion", "lease_ms"],
        "properties": {
            "step_id": {"type": "string", "minLength": 1, "maxLength": 256},
            "goal_text_en": {"type": "string", "minLength": 1, "maxLength": 4096},
            "allowed_actions": {"type": "array", "minItems": 1, "uniqueItems": True,
                                "items": {"type": "string", "maxLength": 256}},
            "parameters": {"type": "object"}, "completion": {"type": "object"},
            "lease_ms": {"type": "integer", "minimum": 1, "maximum": 600000},
        },
    },
    "emit_user_message": {
        "type": "object", "additionalProperties": False, "required": ["text"],
        "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 4096},
                       "claim": {"type": "string", "enum": ["question", "progress", "completed"]},
                       "evidence_seq": {"type": "integer", "minimum": 1}},
    },
    "finish_planning_turn": {
        "type": "object", "additionalProperties": False, "required": ["outcome"],
        "properties": {"outcome": {"type": "string", "enum": ["waiting_feedback", "waiting_input", "completed"]}},
    },
}


def validate_call(name: str, arguments: Any) -> dict:
    if name not in TOOL_SCHEMAS:
        raise TaskError("unsupported_tool")
    if isinstance(arguments, str):
        if len(arguments.encode("utf-8")) > 65536:
            raise TaskError("tool_arguments_too_long")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise TaskError("duplicate_json_key")
                result[key] = value
            return result
        try:
            arguments = json.loads(arguments, object_pairs_hook=pairs)
        except (ValueError, RecursionError) as exc:
            raise TaskError("invalid_tool_json") from exc
    value = json_value(arguments)
    if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 65536:
        raise TaskError("tool_arguments_too_long")
    if validate_params(TOOL_SCHEMAS[name], value):
        raise TaskError("invalid_tool_arguments")
    return value


class PlanningTools:
    def __init__(self, coordinator, router, turn: PlanningTurn, decision_context: dict) -> None:
        self.coordinator = coordinator
        self.router = router
        self.turn = turn
        self.decision_context = decision_context
        self.finished = False

    async def execute(self, name: str, arguments: Any, call_id: str) -> dict:
        args = validate_call(name, arguments)
        require_id(call_id, "tool_call_id")
        if self.finished:
            raise TaskError("turn_already_finished")
        self.coordinator.store.check_turn(self.turn)
        if name == "save_plan":
            revision = self.coordinator.store.save_plan(self.turn, args["steps"], args["expected_revision"])
            return {"ok": True, "plan_revision": revision}
        if name == "submit_current_goal":
            return await self.coordinator.submit(self.turn, args, self.decision_context, call_id)
        if name == "emit_user_message":
            # Identity, route, generation and message IDs are framework-owned.
            message_id = f"{self.turn.turn_id}:{call_id}"
            return {"ok": await self.router.emit(self.turn, message_id=message_id, **args)}
        self.coordinator.finish(self.turn, args["outcome"])
        self.finished = True
        return {"ok": True, "finished": True}


def host_toolset():
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    return ToolSet([FunctionTool(name=name, description={
        "save_plan": "Save future step intents without scene-bound parameters; completed steps cannot replay.",
        "submit_current_goal": "Atomically submit only the current step, using the fresh advertised catalog.",
        "emit_user_message": "The only public exit. Optional; use for a question or factual progress. No recipients.",
        "finish_planning_turn": "Finish privately without a final message or public_text.",
    }[name], parameters=schema) for name, schema in TOOL_SCHEMAS.items()])


async def run_planning_turn(host, provider_id: str, tools: PlanningTools, prompt: str,
                            *, max_rounds: int = 8, timeout_sec: float = 30.0) -> None:
    from astrbot.core.agent.message import Message, ToolCall
    skill = (Path(__file__).parent / "skills" / "robot_task_planning" / "SKILL.md").read_text(encoding="utf-8")
    contexts = [Message(role="user", content=prompt)]
    toolset = host_toolset()
    for _ in range(max_rounds):
        tools.coordinator.store.check_turn(tools.turn)
        response = await asyncio.wait_for(host.llm_generate(
            chat_provider_id=provider_id, tools=toolset, system_prompt=skill,
            contexts=contexts, stream=False,
        ), timeout_sec)
        tools.coordinator.store.check_turn(tools.turn)
        if (response is None or getattr(response, "is_chunk", False)
                or getattr(response, "role", None) != "assistant"):
            raise TaskError("invalid_provider_response")
        names, args, ids = response.tools_call_name, response.tools_call_args, response.tools_call_ids
        if not names:
            raise TaskError("missing_finish_planning_turn")
        if len(names) > 16 or len(names) != len(args) or len(names) != len(ids) or len(set(ids)) != len(ids):
            raise TaskError("invalid_tool_batch")
        validated = [validate_call(n, a) for n, a in zip(names, args)]
        for call_id in ids:
            require_id(call_id, "tool_call_id")
        if "finish_planning_turn" in names and names[-1] != "finish_planning_turn":
            raise TaskError("finish_must_be_last")
        contexts.append(Message(role="assistant", content=response.completion_text or None,
                                tool_calls=[ToolCall(id=i, function=ToolCall.FunctionBody(
                                    name=n, arguments=json.dumps(a, ensure_ascii=False)))
                                            for n, a, i in zip(names, validated, ids)]))
        for name, arguments, call_id in zip(names, validated, ids):
            result = await tools.execute(name, arguments, call_id)
            contexts.append(Message(role="tool", tool_call_id=call_id, content=json.dumps(result)))
        if tools.finished:
            return
    raise TaskError("planning_round_limit")
