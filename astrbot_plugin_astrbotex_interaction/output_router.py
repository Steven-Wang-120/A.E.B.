"""The only public exits from AEB-owned private planning turns.

Two routers live here and they never share state:

* :class:`PublicOutputRouter` -- the legacy private-planning exit.  Unchanged.
* :class:`UpperOutputRouter`  -- the upper-control exit (B05).  It exists so a
  single place decides whether upper-origin text may reach the Host's default
  output.  Only ``upper_public`` is ever allowed; private classification text,
  tool arguments, exceptions, execution segments and stream chunks are not.
"""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from .task_contracts import require_text
from .task_models import PlanningTurn, TaskError
from .task_store import TaskStore

UPPER_PRIVATE = "upper_private"
UPPER_PUBLIC = "upper_public"
UPPER_CONTROL = "upper_control"
UPPER_SOURCES = frozenset({UPPER_PRIVATE, UPPER_PUBLIC, UPPER_CONTROL})


class PublicOutputRouter:
    def __init__(self, store: TaskStore, sender: Callable[[dict[str, Any]], Awaitable[Any]],
                 *, delivery: str = "text") -> None:
        if delivery not in {"text", "tts"}:
            raise ValueError("choose one public delivery")
        self.store = store
        self.sender = sender
        self.delivery = delivery

    async def emit(self, turn: PlanningTurn, text: str, *, message_id: str | None = None,
                   claim: str = "progress", evidence_seq: int | None = None) -> bool:
        require_text(text, "text", max_len=4096)
        task = self.store.get(turn.task_id)
        self.store.check_turn(turn, task)
        if claim not in {"question", "progress", "completed"}:
            raise TaskError("invalid_public_claim")
        if claim == "completed":
            if (task["current_goal"] is not None or type(evidence_seq) is not int
                    or evidence_seq != task.get("completion_evidence_seq") or not evidence_seq):
                raise TaskError("completion_not_verified")
        message_id = message_id or uuid.uuid4().hex
        payload = {"visibility": "user", "source": "private_planning", "type": "task_reply",
                   "task_id": turn.task_id, "turn_id": turn.turn_id,
                   "generation": turn.generation, "route_ref": turn.route_ref,
                   "session_id": task["session_id"], "user_id": task["user_id"], "robot_id": task["robot_id"],
                   "ex_session": task["ex_session"], "message_id": message_id,
                   "text": text.strip(), "delivery": self.delivery, "claim": claim}
        if claim == "completed":
            payload["evidence_seq"] = evidence_seq
        if not self.store.claim_message(turn, message_id, payload):
            return False
        # Reserve before IO: an ambiguous text/TTS timeout must never retry a goal or
        # duplicate speech. EX must also gate generation immediately before playback.
        try:
            self.store.check_turn(turn)
            await self.sender(payload)
        except TaskError:
            self.store.message_result(message_id, "stale")
            raise
        except Exception:
            self.store.message_result(message_id, "delivery_unknown")
            return False
        self.store.message_result(message_id, "sent")
        return True

    @staticmethod
    def permits_automatic_output(source: str | None, *, task_id: str | None = None) -> bool:
        return task_id is None and source in {None, "public_chat"}


class UpperOutputRouter:
    """Single public exit for an upper-control plan.

    The legacy router above is untouched: it still owns private planning turns.
    This one owns ``upper_*`` content and refuses to speak anything except the
    plan's own ``public_text``.
    """

    def __init__(self, store: Any, *, delivery: str = "text") -> None:
        if delivery not in {"text", "tts"}:
            raise ValueError("choose one public delivery")
        self.store = store
        self.delivery = delivery

    @staticmethod
    def permits_public_output(source: str | None, *, plan_id: str | None = None) -> bool:
        """True only for the plan's own public reply.

        B10 calls this from the Host output hook: anything carrying a
        ``plan_id`` or any other ``upper_*`` source must be suppressed so the
        execution segments can never be read aloud or echoed into chat.
        """
        return source == UPPER_PUBLIC and plan_id is None

    @staticmethod
    def permits_automatic_output(source: str | None, *, plan_id: str | None = None) -> bool:
        """Host-side gate: default output is allowed only for ordinary chat."""
        if source in UPPER_SOURCES:
            return False
        return plan_id is None and source in {None, "public_chat"}

    def claim(self, *, source: str, plan_id: str, reply_id: str | None = None,
              request_id: str | None = None) -> dict:
        """Claim the public exit for one plan, or refuse loudly.

        A non-public source is a wiring bug, not a no-op: silently dropping it
        would hide leaked execution text inside the Host's reply path.
        """
        if source not in UPPER_SOURCES:
            raise TaskError("invalid_source")
        if source != UPPER_PUBLIC:
            raise TaskError("source_not_public")
        reply, claimed = self.store.claim_public_reply(plan_id)
        if reply_id is not None and reply["reply_id"] != reply_id:
            raise TaskError("id_conflict")
        return {"reply": reply, "claimed": claimed, "request_id": request_id}

    async def emit_public(self, plan_id: str,
                          sender: Callable[[str], Awaitable[Any]]) -> dict:
        """Claim, send and record the plan's public text exactly once.

        The claim is committed before the send.  An ambiguous timeout becomes
        ``delivery_unknown`` and is never retried, so a caller cannot re-speak
        by trying again.
        """
        claim = self.claim(source=UPPER_PUBLIC, plan_id=plan_id)
        reply = claim["reply"]
        if not claim["claimed"]:
            return {"sent": False, "reply": reply, "reason": "already_claimed"}
        self.store.record_public_delivery(reply["reply_id"], "sending")
        try:
            await sender(reply["text"])
        except Exception:  # noqa: BLE001 - a failed send is not a retry permit
            self.store.record_public_delivery(reply["reply_id"], "delivery_unknown",
                                              error_code="public_send_failed")
            return {"sent": False, "reply": self.store._reply_projection(
                self.store._reply_row(reply["reply_id"])), "reason": "delivery_unknown"}
        self.store.record_public_delivery(reply["reply_id"], "sent")
        return {"sent": True, "reply": self.store._reply_projection(
            self.store._reply_row(reply["reply_id"])), "reason": "sent"}
