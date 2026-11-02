"""B03 three-way decision and private content generation.

Two model calls, no side effects:

* :meth:`UpperModels.classify` asks a routing model for three probabilities and
  lets the *framework* pick the route by argmax. The model never decides.
* :meth:`UpperModels.generate` asks a content model for one
  ``emit_upper_result`` payload under the frozen prompt snapshot.

Both run as private tool loops. Neither sends a chat message, opens a socket or
touches the database — ``UpperPipeline`` (B05) owns those. The only thing that
leaves this module is a validated B00 object plus internal evidence.

Fencing: every call captures the current generation token. :meth:`invalidate`
(config switch, plugin shutdown, user cancel) bumps it, so an in-flight call or
a late provider response can never be consumed by a newer plan.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from .upper_contracts import (
    ROUTE_SET,
    ROUTE_TIE_ORDER,
    RouteProbabilities,
    UpperContractError,
    UpperResult,
    canonical_hash,
)

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorCode:
    OK = "ok"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    DECISION_ERROR = "decision_error"
    GENERATION_ERROR = "generation_error"
    GENERATION_TIMEOUT = "generation_timeout"
    GENERATION_CANCELLED = "generation_cancelled"
    GENERATION_STALE = "generation_stale"
    GENERATION_NO_RESULT = "generation_no_result"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"


class ModelError(ValueError):
    """A model call that must be reported, never silently repaired."""

    def __init__(self, code: str, message: str = "", *, detail: Any = None) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


def _fail(code: str, message: str, detail: Any = None) -> None:
    raise ModelError(code, message, detail=detail)


# --------------------------------------------------------------------------
# Tool schemas: the only surface either model may touch
# --------------------------------------------------------------------------

ROUTING_TOOL_NAME = "submit_route_probabilities"
EMIT_TOOL_NAME = "emit_upper_result"

ROUTING_TOOL_SPEC: dict[str, Any] = {
    "name": ROUTING_TOOL_NAME,
    "description": (
        "Submit the three routing probabilities for this input. "
        "No other field, no route name, no explanation."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["speech_only", "action_only", "speech_and_action"],
        "properties": {
            "speech_only": {"type": "number", "minimum": 0, "maximum": 1},
            "action_only": {"type": "number", "minimum": 0, "maximum": 1},
            "speech_and_action": {"type": "number", "minimum": 0, "maximum": 1},
        },
    },
}

EMIT_TOOL_SPEC: dict[str, Any] = {
    "name": EMIT_TOOL_NAME,
    "description": (
        "Submit exactly one complete upper result. Collects the result only; "
        "it does not execute anything."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["public_text", "segments", "send_count"],
        "properties": {
            "public_text": {"type": ["string", "null"], "maxLength": 4096},
            "segments": {
                "type": "array",
                "maxItems": 64,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text"],
                    "properties": {"text": {"type": "string", "maxLength": 4096}},
                },
            },
            "send_count": {"type": "integer", "minimum": 0, "maximum": 64},
        },
    },
}


def _toolset(specs: tuple[dict[str, Any], ...]):
    """Build the Host tool set, falling back to plain specs off-Host.

    The fallback exists so the offline tests can run without AstrBot installed;
    it is never used when the real Host is present.
    """
    try:
        from astrbot.core.agent.tool import FunctionTool, ToolSet
    except Exception:  # pragma: no cover - depends on the deployment
        return [dict(spec) for spec in specs]
    return ToolSet(
        [
            FunctionTool(
                name=spec["name"],
                description=spec["description"],
                parameters=spec["parameters"],
            )
            for spec in specs
        ]
    )


def _message(
    role: str,
    content: str | None,
    tool_calls: Any = None,
    tool_call_id: str | None = None,
):
    """Build one context message, with a dict fallback off-Host."""
    try:
        from astrbot.core.agent.message import Message, ToolCall
    except Exception:  # pragma: no cover - depends on the deployment
        payload: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            payload["tool_calls"] = tool_calls
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
        return payload
    if tool_calls:
        return Message(
            role=role,
            content=content,
            tool_calls=[
                ToolCall(
                    id=call_id,
                    function=ToolCall.FunctionBody(name=name, arguments=arguments),
                )
                for call_id, name, arguments in tool_calls
            ],
        )
    if tool_call_id:
        return Message(role=role, tool_call_id=tool_call_id, content=content)
    return Message(role=role, content=content)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    """Internal evidence for one call. Never part of a public reply.

    Carries the probability vector so an operator can audit a routing decision,
    but the consumer must not echo it: B03 §6 forbids probabilities and model
    exceptions from reaching ``public_text``.
    """

    provider_id: str
    model_id: str
    prompt_hash: str
    input_hash: str
    output_hash: str
    elapsed_ms: int
    rounds: int
    probabilities: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "elapsed_ms": self.elapsed_ms,
            "rounds": self.rounds,
        }
        if self.probabilities is not None:
            payload["probabilities"] = dict(self.probabilities)
        return payload


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    probabilities: RouteProbabilities
    route: str
    evidence: ModelEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "probabilities": self.probabilities.to_dict(),
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    result: UpperResult
    evidence: ModelEvidence

    def to_dict(self) -> dict[str, Any]:
        return {"result": self.result.to_dict(), "evidence": self.evidence.to_dict()}


# --------------------------------------------------------------------------
# UpperModels
# --------------------------------------------------------------------------


class UpperModels:
    """Private model calls with a hard deadline, a round cap and a fence.

    ``host`` is an injected adapter over the Host's public surface. It must
    provide ``llm_generate(**kwargs)``, ``get_provider_by_id(provider_id)`` and
    ``get_current_chat_provider_id(umo)``. This class never reads a global
    platform or peer.
    """

    def __init__(
        self,
        host: Any,
        *,
        timeout_s: float = 30.0,
        max_rounds: int = 6,
        clock=time.monotonic,
    ) -> None:
        if timeout_s <= 0:
            _fail(ErrorCode.GENERATION_ERROR, "timeout_s must be positive")
        if max_rounds < 1:
            _fail(ErrorCode.GENERATION_ERROR, "max_rounds must be at least 1")
        for attribute in ("llm_generate", "get_provider_by_id", "get_current_chat_provider_id"):
            if not hasattr(host, attribute):
                _fail(
                    ErrorCode.PROVIDER_UNAVAILABLE,
                    f"host adapter must expose {attribute}()",
                )
        self._host = host
        self._timeout_s = float(timeout_s)
        self._max_rounds = int(max_rounds)
        self._clock = clock
        self._fence = 0

    # -- fence ------------------------------------------------------------

    @property
    def fence(self) -> int:
        return self._fence

    def invalidate(self) -> int:
        """Discard every in-flight call (config switch, plugin shutdown)."""
        self._fence += 1
        return self._fence

    def cancel(self) -> int:
        """Cancel the current turn. A late result becomes ``generation_stale``."""
        return self.invalidate()

    def _check_fence(self, token: int) -> None:
        if token != self._fence:
            _fail(
                ErrorCode.GENERATION_STALE,
                "the generation was superseded or cancelled",
                detail={"token": token, "current": self._fence},
            )

    # -- provider resolution ---------------------------------------------

    @staticmethod
    def _model_id_of(provider: Any) -> str:
        meta = getattr(provider, "meta", None)
        if callable(meta):
            try:
                value = meta()
                return str(getattr(value, "id", "") or "")
            except Exception:  # pragma: no cover - defensive
                return ""
        return str(getattr(provider, "model", "") or "")

    def _resolve_provider(self, provider_id: str | None, origin: str) -> tuple[str, str]:
        """Return ``(provider_id, model_id)``.

        An explicitly configured id that no longer exists is a hard error: a
        silent fallback to another model would change behaviour behind the
        operator's back.
        """
        if provider_id:
            provider = self._host.get_provider_by_id(provider_id)
            if provider is None:
                _fail(
                    ErrorCode.PROVIDER_UNAVAILABLE,
                    f"configured provider {provider_id!r} does not exist",
                    detail={"provider_id": provider_id},
                )
            return provider_id, self._model_id_of(provider)
        try:
            resolved = self._host.get_current_chat_provider_id(origin)
        except Exception as exc:
            _fail(
                ErrorCode.PROVIDER_UNAVAILABLE,
                f"no current chat provider for this origin ({type(exc).__name__})",
            )
        if not resolved:
            _fail(ErrorCode.PROVIDER_UNAVAILABLE, "no current chat provider for this origin")
        provider = self._host.get_provider_by_id(resolved)
        if provider is None:
            _fail(
                ErrorCode.PROVIDER_UNAVAILABLE,
                f"current provider {resolved!r} does not exist",
            )
        return str(resolved), self._model_id_of(provider)

    # -- the one call primitive ------------------------------------------

    async def _call(
        self,
        *,
        provider_id: str,
        system_prompt: str,
        contexts: list[Any],
        tools: Any,
        deadline: float,
        token: int,
    ) -> Any:
        self._check_fence(token)
        remaining = deadline - self._clock()
        if remaining <= 0:
            _fail(ErrorCode.GENERATION_TIMEOUT, "the turn deadline elapsed before the call")
        try:
            response = await asyncio.wait_for(
                self._host.llm_generate(
                    chat_provider_id=provider_id,
                    system_prompt=system_prompt,
                    contexts=contexts,
                    tools=tools,
                    # Never stream: a chunked reply must not reach the Host's
                    # automatic chat output.
                    stream=False,
                ),
                remaining,
            )
        except (asyncio.TimeoutError, TimeoutError):
            _fail(ErrorCode.GENERATION_TIMEOUT, f"the call exceeded {self._timeout_s}s")
        except ModelError:
            raise
        except Exception as exc:
            _fail(
                ErrorCode.GENERATION_ERROR,
                f"provider raised {type(exc).__name__}",
            )
        # A result that arrives after a cancel or a config switch is unusable.
        self._check_fence(token)
        if response is None or getattr(response, "is_chunk", False):
            _fail(ErrorCode.INVALID_PROVIDER_RESPONSE, "expected one complete response")
        if getattr(response, "role", None) != "assistant":
            _fail(ErrorCode.INVALID_PROVIDER_RESPONSE, "response is not an assistant turn")
        return response

    @staticmethod
    def _calls_of(response: Any) -> list[tuple[str, str, Any]]:
        """Return ``[(call_id, name, raw_arguments)]`` for one response."""
        names = getattr(response, "tools_call_name", None) or []
        arguments = getattr(response, "tools_call_args", None) or []
        ids = getattr(response, "tools_call_ids", None) or []
        if not names:
            return []
        if not (len(names) == len(arguments) == len(ids)):
            _fail(ErrorCode.INVALID_PROVIDER_RESPONSE, "malformed tool call batch")
        if len(set(ids)) != len(ids):
            _fail(ErrorCode.INVALID_PROVIDER_RESPONSE, "duplicate tool call id")
        return [
            (str(call_id), str(name), argument)
            for call_id, name, argument in zip(ids, names, arguments)
        ]

    @staticmethod
    def _decode_arguments(raw: Any) -> Any:
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (ValueError, RecursionError):
                _fail(ErrorCode.INVALID_PROVIDER_RESPONSE, "tool arguments are not JSON")
        return raw

    # -- classify ---------------------------------------------------------

    async def classify(
        self,
        input_text: str,
        prompt_snapshot: Any,
        provider_id: str | None = None,
        *,
        origin: str = "",
    ) -> ClassificationResult:
        """Return the framework's route decision, never the model's."""
        self._require_input(input_text)
        resolved, model_id = self._resolve_provider(provider_id, origin)
        token = self._fence
        started = self._clock()
        deadline = started + self._timeout_s

        response = await self._call(
            provider_id=resolved,
            system_prompt=prompt_snapshot.routing_prompt,
            contexts=[_message("user", input_text)],
            tools=_toolset((ROUTING_TOOL_SPEC,)),
            deadline=deadline,
            token=token,
        )
        payload = self._routing_payload(response)
        try:
            probabilities = RouteProbabilities.parse(payload)
        except UpperContractError as exc:
            # Zero generation, zero send. The caller decides what to tell the user.
            _fail(
                ErrorCode.DECISION_ERROR,
                f"routing output rejected: {exc.code}",
                detail=exc.error.to_dict(),
            )
        route = probabilities.route()  # framework argmax, not model output
        evidence = ModelEvidence(
            provider_id=resolved,
            model_id=model_id,
            prompt_hash=prompt_snapshot.routing_version,
            input_hash=_hash_text(input_text),
            output_hash=canonical_hash(probabilities.to_dict()),
            elapsed_ms=int((self._clock() - started) * 1000),
            rounds=1,
            probabilities=probabilities.to_dict(),
        )
        return ClassificationResult(probabilities=probabilities, route=route, evidence=evidence)

    def _routing_payload(self, response: Any) -> Any:
        calls = self._calls_of(response)
        if calls:
            if len(calls) != 1 or calls[0][1] != ROUTING_TOOL_NAME:
                _fail(
                    ErrorCode.DECISION_ERROR,
                    "the routing turn must submit exactly one probability tool call",
                )
            return self._decode_arguments(calls[0][2])
        # A provider that ignores the tool may still answer with the bare object.
        # Anything else (prose, a route name, extra keys) is a decision error.
        text = (getattr(response, "completion_text", None) or "").strip()
        if not text:
            _fail(ErrorCode.DECISION_ERROR, "the routing turn produced no result")
        try:
            payload = json.loads(text)
        except (ValueError, RecursionError):
            _fail(ErrorCode.DECISION_ERROR, "the routing turn produced prose, not probabilities")
        if not isinstance(payload, dict):
            _fail(ErrorCode.DECISION_ERROR, "the routing turn produced a non-object")
        return payload

    # -- generate ---------------------------------------------------------

    async def generate(
        self,
        input_text: str,
        route: str,
        prompt_snapshot: Any,
        provider_id: str | None = None,
        *,
        origin: str = "",
    ) -> GenerationResult:
        """Collect exactly one ``emit_upper_result`` payload."""
        self._require_input(input_text)
        if route not in ROUTE_SET:
            _fail(ErrorCode.GENERATION_ERROR, f"unknown route {route!r}")
        resolved, model_id = self._resolve_provider(provider_id, origin)
        token = self._fence
        started = self._clock()
        deadline = started + self._timeout_s

        contexts: list[Any] = [_message("user", input_text)]
        tools = _toolset((EMIT_TOOL_SPEC,))
        collected: UpperResult | None = None
        rounds = 0

        for _ in range(self._max_rounds):
            rounds += 1
            response = await self._call(
                provider_id=resolved,
                system_prompt=prompt_snapshot.planning_prompt,
                contexts=contexts,
                tools=tools,
                deadline=deadline,
                token=token,
            )
            calls = self._calls_of(response)
            if not calls:
                # No result yet. Keep the assistant text in context and ask again;
                # an empty response simply burns a round.
                text = getattr(response, "completion_text", None)
                if text:
                    contexts.append(_message("assistant", text))
                continue

            for call_id, name, raw in calls:
                if name != EMIT_TOOL_NAME:
                    # The old planning/proposal/execution tools must not be
                    # reachable from this turn.
                    _fail(
                        ErrorCode.GENERATION_ERROR,
                        f"tool {name!r} is not available in this turn",
                    )
                if collected is not None:
                    # A repeat call returns the already collected state. It can
                    # never produce a second external side effect.
                    return GenerationResult(
                        result=collected,
                        evidence=self._evidence(
                            resolved, model_id, prompt_snapshot, input_text,
                            collected, started, rounds,
                        ),
                    )
                try:
                    collected = UpperResult.parse(self._decode_arguments(raw), route)
                except UpperContractError as exc:
                    # No auto-repair, no silent downgrade to another route.
                    _fail(
                        ErrorCode.GENERATION_ERROR,
                        f"generated result rejected: {exc.code}",
                        detail=exc.error.to_dict(),
                    )
                contexts.append(
                    _message(
                        "tool",
                        json.dumps(
                            {"ok": True, "collected": True, "route": route},
                            ensure_ascii=False,
                        ),
                        tool_call_id=call_id,
                    )
                )
            if collected is not None:
                return GenerationResult(
                    result=collected,
                    evidence=self._evidence(
                        resolved, model_id, prompt_snapshot, input_text,
                        collected, started, rounds,
                    ),
                )

        _fail(
            ErrorCode.GENERATION_NO_RESULT,
            f"no emit_upper_result after {self._max_rounds} rounds",
        )

    def _evidence(
        self,
        provider_id: str,
        model_id: str,
        prompt_snapshot: Any,
        input_text: str,
        result: UpperResult,
        started: float,
        rounds: int,
    ) -> ModelEvidence:
        return ModelEvidence(
            provider_id=provider_id,
            model_id=model_id,
            prompt_hash=prompt_snapshot.planning_version,
            input_hash=_hash_text(input_text),
            output_hash=canonical_hash(result.to_dict()),
            elapsed_ms=int((self._clock() - started) * 1000),
            rounds=rounds,
        )

    # -- shared -----------------------------------------------------------

    @staticmethod
    def _require_input(input_text: Any) -> None:
        if not isinstance(input_text, str) or not input_text.strip():
            _fail(ErrorCode.GENERATION_ERROR, "input text must be a non-empty string")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def route_order() -> tuple[str, ...]:
    """Expose the fixed tie order so callers do not re-derive it."""
    return ROUTE_TIE_ORDER
