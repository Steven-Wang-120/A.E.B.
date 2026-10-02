"""The only public exit from AEB-owned private planning turns."""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from .task_contracts import require_text
from .task_models import PlanningTurn, TaskError
from .task_store import TaskStore


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
