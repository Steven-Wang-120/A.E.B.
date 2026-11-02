"""Real loopback ROUTER/DEALER tests for the AEB ``control`` channel.

Nothing here is mocked at the socket level: a real ``zmq.asyncio`` ROUTER binds
an ephemeral port and a real DEALER peer connects with an explicit IDENTITY, so
framing, hello tracking, correlation ids and timeouts are all exercised.

:class:`LoopbackExPeer` is written to be reusable: B09 (EX side) and B11 (e2e)
need the same peer behaviours -- accept, definite rejection, silence, wrong
receipt, wrong hash, wrong session.
"""
from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
import unittest
import uuid

import zmq
import zmq.asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from astrbot_plugin_astrbotex_interaction.control_channel import (  # noqa: E402
    ControlChannel, ControlError, DeliveryOutcome, payload_hash,
)
from astrbot_plugin_astrbotex_interaction.zmq_transport import ZmqRouterChannel  # noqa: E402

SCHEMA = "aeb.upper_steps.v1"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def frozen_batch(*, batch_id="batch-1", request_id="request-1", ex_session="ex-1",
                 robot_id="r1", route_ref="route-1", first_ordinal=0,
                 texts=("寻找红色杯子。", "接近并对齐同一个目标。"), tamper_hash=False):
    """Build the C01 envelope the way B04 freezes it."""
    payload = {
        "schema": SCHEMA,
        "request_id": request_id,
        "plan_id": "plan-1",
        "plan_revision": 1,
        "batch_id": batch_id,
        "robot_id": robot_id,
        "route_ref": route_ref,
        "ex_session": ex_session,
        "catalog_version": "catalog-hash",
        "prompt_version": "prompt-hash",
        "visibility": "decision_only",
        "segments": [
            {"segment_id": f"segment-{first_ordinal + index}",
             "ordinal": first_ordinal + index, "text": text}
            for index, text in enumerate(texts)
        ],
    }
    digest = payload_hash(payload)
    if tamper_hash:
        digest = "0" * 64
    return {
        "batch_id": batch_id,
        "plan_id": "plan-1",
        "request_id": request_id,
        "payload": payload,
        "payload_hash": digest,
        "peer_hex": b"ex-control".hex(),
        "ex_session": ex_session,
        "robot_id": robot_id,
        "route_ref": route_ref,
        "first_ordinal": first_ordinal,
        "last_ordinal": first_ordinal + len(texts) - 1,
        "segment_ids": [segment["segment_id"] for segment in payload["segments"]],
        "status": "sending",
    }


def envelope(method: str, payload: dict, *, kind="request", reply_to=None,
             channel="control") -> dict:
    message = {
        "protocol": "astrbotex-zmq",
        "version": 1,
        "channel": channel,
        "kind": kind,
        "id": uuid.uuid4().hex,
        "method": method,
        "timestamp": time.time(),
        "payload": payload,
    }
    if reply_to:
        message["reply_to"] = reply_to
    return message


class LoopbackExPeer:
    """A DEALER that plays the EX side of the control channel."""

    def __init__(self, port: int, *, identity: str = "ex-control",
                 ex_session: str = "ex-1", input_ready: bool = True,
                 accept_phase: str = "injected", reject_with: str | None = None,
                 silent: bool = False, capabilities: dict | None = None,
                 receipt_override: dict | None = None) -> None:
        self.port = port
        self.identity = identity
        self.ex_session = ex_session
        self.input_ready = input_ready
        self.accept_phase = accept_phase
        self.reject_with = reject_with
        self.silent = silent
        self.capabilities = capabilities
        self.receipt_override = receipt_override
        self.socket: zmq.asyncio.Socket | None = None
        self.task: asyncio.Task | None = None
        self.requests: list[dict] = []

    async def connect(self) -> None:
        self.socket = zmq.asyncio.Context.instance().socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.IDENTITY, self.identity.encode())
        self.socket.connect(f"tcp://127.0.0.1:{self.port}")
        self.task = asyncio.create_task(self._loop())
        await self.send("system.hello", {})

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None

    async def send(self, method: str, payload: dict) -> None:
        await self.socket.send(json.dumps(envelope(method, payload)).encode())

    async def _loop(self) -> None:
        while True:
            frames = await self.socket.recv_multipart()
            message = json.loads(frames[0])
            if message["kind"] != "request":
                continue
            self.requests.append(message)
            reply = self._reply_for(message)
            if reply is None:
                continue
            await self.socket.send(json.dumps(
                envelope(message["method"], reply, kind="response",
                         reply_to=message["id"])).encode())

    def _reply_for(self, message: dict) -> dict | None:
        method = message["method"]
        if method == "system.hello":
            return {"ok": True, "channel": "control"}
        if method == "upper.capabilities.get":
            if self.capabilities is not None:
                return self.capabilities
            return {
                "ok": True,
                "ex_session": self.ex_session,
                "input_ready": self.input_ready,
                "methods": ["upper.capabilities.get", "upper.steps.inject"],
                "max_bytes": 262144,
                "max_segments": 64,
                "schema": SCHEMA,
            }
        if method == "upper.steps.inject":
            payload = message["payload"]
            if self.silent:
                # Swallow the frame: the caller must record an unknown window
                # rather than a delivery, and must not retry.
                return None
            if self.reject_with:
                return {"ok": False, "error": self.reject_with}
            receipt = {
                "ok": True,
                "batch_id": payload["batch_id"],
                "request_id": payload["request_id"],
                "payload_hash": payload_hash(payload),
                "phase": self.accept_phase,
                "ex_session": self.ex_session,
            }
            if self.receipt_override:
                receipt.update(self.receipt_override)
            return receipt
        return {"ok": False, "error": "unknown_method"}

    def injects(self) -> list[dict]:
        return [item for item in self.requests if item["method"] == "upper.steps.inject"]


class ControlChannelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.port = free_port()
        self.channel = ControlChannel(peer_id="ex-control", robot_id="r1", port=self.port,
                                      request_timeout_sec=1.0)
        await self.channel.start()
        self.peer: LoopbackExPeer | None = None

    async def asyncTearDown(self) -> None:
        if self.peer is not None:
            await self.peer.close()
        await self.channel.close()

    async def peer_with(self, **kwargs) -> LoopbackExPeer:
        self.peer = LoopbackExPeer(self.port, **kwargs)
        await self.peer.connect()
        for _ in range(100):
            if self.channel.peer in self.channel.online_peers():
                return self.peer
            await asyncio.sleep(0.01)
        self.fail("loopback peer never completed hello")

    # ------------------------------------------------------------------
    async def test_single_and_multi_segment_payload_arrive_intact_and_ordered(self):
        peer = await self.peer_with()
        batch = frozen_batch()
        outcome = await self.channel.inject(b"ex-control", batch)
        self.assertEqual(outcome.status, "injected")
        self.assertTrue(outcome.sent)
        self.assertEqual(outcome.batch_id, "batch-1")
        self.assertEqual(outcome.payload_hash, batch["payload_hash"])

        sent = peer.injects()[0]["payload"]
        self.assertEqual(sent, batch["payload"])
        self.assertEqual([s["ordinal"] for s in sent["segments"]], [0, 1])
        self.assertEqual([s["text"] for s in sent["segments"]],
                         ["寻找红色杯子。", "接近并对齐同一个目标。"])
        self.assertEqual(sent["visibility"], "decision_only")
        self.assertNotIn("public_text", json.dumps(sent, ensure_ascii=False))

        single = frozen_batch(batch_id="batch-2", request_id="request-2",
                              texts=("只发这一段。",), first_ordinal=2)
        self.assertEqual((await self.channel.inject(b"ex-control", single)).status, "injected")
        self.assertEqual(len(peer.injects()), 2)
        self.assertEqual(peer.injects()[1]["payload"]["segments"][0]["ordinal"], 2)

    async def test_same_batch_is_never_injected_twice(self):
        peer = await self.peer_with()
        batch = frozen_batch()
        first = await self.channel.inject(b"ex-control", batch)
        second = await self.channel.inject(b"ex-control", batch)
        self.assertEqual(first.status, "injected")
        self.assertEqual(second.status, "injected")
        self.assertTrue(second.duplicate)
        self.assertFalse(second.sent)
        self.assertEqual(len(peer.injects()), 1)

        with self.assertRaisesRegex(ControlError, "invalid_batch"):
            ControlChannel(peer_id="", robot_id="r1", port=self.port)

    async def test_duplicate_with_different_hash_is_a_conflict(self):
        peer = await self.peer_with()
        await self.channel.inject(b"ex-control", frozen_batch())
        tampered = frozen_batch(tamper_hash=True)
        outcome = await self.channel.inject(b"ex-control", tampered)
        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.error_code, "id_conflict")
        self.assertEqual(len(peer.injects()), 1)

    async def test_wrong_peer_and_missing_peer_are_refused_without_sending(self):
        peer = await self.peer_with()
        for bad in (None, b"someone-else", "ex-control"):
            outcome = await self.channel.inject(bad, frozen_batch())
            self.assertEqual(outcome.status, "rejected")
            self.assertEqual(outcome.error_code, "peer_unavailable")
            self.assertFalse(outcome.sent)
        self.assertEqual(peer.injects(), [])
        # default_peer is never consulted, even though the transport has one.
        self.assertIsNotNone(self.channel.channel.default_peer)

    async def test_session_mismatch_is_refused_and_invalidates_the_cache(self):
        peer = await self.peer_with(ex_session="ex-2")
        outcome = await self.channel.inject(b"ex-control", frozen_batch(ex_session="ex-1"))
        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.error_code, "session_mismatch")
        self.assertFalse(outcome.sent)
        self.assertEqual(peer.injects(), [])
        self.assertEqual(self.channel._cache, {})

    async def test_input_not_ready_is_input_busy(self):
        peer = await self.peer_with(input_ready=False)
        outcome = await self.channel.inject(b"ex-control", frozen_batch())
        self.assertEqual(outcome.error_code, "input_busy")
        self.assertEqual(peer.injects(), [])

    async def test_timeout_marks_unknown_and_never_sends_a_second_time(self):
        peer = await self.peer_with(silent=True)
        outcome = await self.channel.inject(b"ex-control", frozen_batch(), timeout_sec=0.25)
        self.assertEqual(outcome.status, "delivery_unknown")
        self.assertEqual(outcome.error_code, "delivery_unknown")
        self.assertTrue(outcome.sent)
        # A retry of the same batch must not put a second frame on the wire.
        again = await self.channel.inject(b"ex-control", frozen_batch(), timeout_sec=0.25)
        self.assertEqual(again.status, "delivery_unknown")
        self.assertTrue(again.duplicate)
        self.assertEqual(len(peer.injects()), 1)

    async def test_wrong_receipt_fields_are_unknown_not_success(self):
        peer = await self.peer_with()
        for override, reason in (
            ({"batch_id": "other-batch"}, "receipt_batch_mismatch"),
            ({"request_id": "other-request"}, "receipt_request_mismatch"),
            ({"payload_hash": "0" * 64}, "receipt_hash_mismatch"),
            ({"phase": "queued"}, "receipt_phase"),
        ):
            peer.receipt_override = override
            batch = frozen_batch(batch_id=f"batch-{reason}")
            outcome = await self.channel.inject(b"ex-control", batch)
            self.assertEqual(outcome.status, "delivery_unknown", reason)
            self.assertTrue(outcome.sent)
            self.assertFalse(outcome.ok)
            # A registered batch stays final, so the next case needs a new id.
            self.assertIn(batch["batch_id"], self.channel._attempted)
        self.assertEqual(len(peer.injects()), 4)

    async def test_remote_rejection_is_definite_and_not_retried(self):
        peer = await self.peer_with(reject_with="session_mismatch")
        outcome = await self.channel.inject(b"ex-control", frozen_batch())
        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.error_code, "session_mismatch")
        again = await self.channel.inject(b"ex-control", frozen_batch())
        self.assertEqual(again.status, "rejected")
        self.assertTrue(again.duplicate)
        self.assertEqual(len(peer.injects()), 1)

    async def test_unknown_remote_error_is_not_treated_as_a_refusal(self):
        await self.peer_with(reject_with="model_exploded")
        outcome = await self.channel.inject(b"ex-control", frozen_batch())
        self.assertEqual(outcome.status, "delivery_unknown")

    async def test_receipt_phase_rejected_is_reported_as_rejected(self):
        await self.peer_with(accept_phase="rejected",
                             receipt_override={"error_code": "route_mismatch"})
        outcome = await self.channel.inject(b"ex-control", frozen_batch())
        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.error_code, "route_mismatch")

    async def test_tampered_payload_is_refused_before_any_io(self):
        peer = await self.peer_with()
        cases = []
        batch = frozen_batch(tamper_hash=True)
        cases.append((batch, "invalid_batch"))
        batch = frozen_batch()
        batch["payload"]["public_text"] = "泄漏"
        cases.append((batch, "invalid_batch"))
        batch = frozen_batch()
        batch["payload"]["schema"] = "aeb.upper_steps.v0"
        cases.append((batch, "invalid_batch"))
        batch = frozen_batch()
        batch["payload"]["segments"][1]["ordinal"] = 5
        cases.append((batch, "invalid_batch"))
        batch = frozen_batch()
        batch["payload"]["robot_id"] = "r2"
        cases.append((batch, "session_mismatch"))
        batch = frozen_batch()
        batch["payload"]["segments"][0]["text"] = "字" * 4097
        cases.append((batch, "payload_too_large"))
        for broken, code in cases:
            outcome = await self.channel.inject(b"ex-control", broken)
            self.assertEqual(outcome.error_code, code)
            self.assertFalse(outcome.sent)
        self.assertEqual(peer.injects(), [])

    async def test_capabilities_are_cached_and_refreshed(self):
        peer = await self.peer_with()
        summary = await self.channel.capabilities(b"ex-control")
        self.assertEqual(summary["ex_session"], "ex-1")
        self.assertTrue(summary["input_ready"])
        self.assertEqual(summary["max_segments"], 64)
        await self.channel.capabilities(b"ex-control")
        calls = [m for m in peer.requests if m["method"] == "upper.capabilities.get"]
        self.assertEqual(len(calls), 1)
        await self.channel.capabilities(b"ex-control", refresh=True)
        calls = [m for m in peer.requests if m["method"] == "upper.capabilities.get"]
        self.assertEqual(len(calls), 2)

    async def test_capabilities_reject_unsupported_backend(self):
        await self.peer_with(capabilities={"ok": True, "ex_session": "ex-1",
                                           "input_ready": True,
                                           "methods": ["decision.goal.submit"]})
        with self.assertRaisesRegex(ControlError, "backend_unsupported"):
            await self.channel.capabilities(b"ex-control")

    async def test_capabilities_without_peer_fail_explicitly(self):
        with self.assertRaisesRegex(ControlError, "peer_unavailable"):
            await self.channel.capabilities(None)

    async def test_control_congestion_does_not_block_the_text_channel(self):
        text_port = free_port()
        text_channel = ZmqRouterChannel("text", f"tcp://127.0.0.1:{text_port}",
                                        request_timeout_sec=1.0)

        async def handler(peer, message, binary):
            return {"ok": True, "echo": message["payload"].get("text")}

        text_channel.register_handler("interaction.message", handler)
        await text_channel.start()
        self.addAsyncCleanup(text_channel.close)

        client = zmq.asyncio.Context.instance().socket(zmq.DEALER)
        client.setsockopt(zmq.LINGER, 0)
        client.setsockopt(zmq.IDENTITY, b"ex-text")
        client.connect(f"tcp://127.0.0.1:{text_port}")
        self.addCleanup(client.close, 0)
        await client.send(json.dumps(envelope("system.hello", {}, channel="text")).encode())
        await asyncio.wait_for(client.recv_multipart(), 2.0)

        peer = await self.peer_with(silent=True)
        blocked = asyncio.create_task(
            self.channel.inject(b"ex-control", frozen_batch(), timeout_sec=3.0))
        await asyncio.sleep(0.1)
        self.assertFalse(blocked.done())

        request_id = uuid.uuid4().hex
        await client.send(json.dumps({**envelope("interaction.message", {"text": "hello"},
                                                 channel="text"),
                                      "id": request_id}).encode())
        reply = json.loads((await asyncio.wait_for(client.recv_multipart(), 1.0))[0])
        self.assertEqual(reply["reply_to"], request_id)
        self.assertEqual(reply["payload"]["echo"], "hello")

        outcome = await blocked
        self.assertEqual(outcome.status, "delivery_unknown")
        self.assertEqual(len(peer.injects()), 1)

    async def test_close_releases_the_port_and_pending_waits(self):
        await self.peer_with(silent=True)
        pending = asyncio.create_task(
            self.channel.inject(b"ex-control", frozen_batch(), timeout_sec=5.0))
        await asyncio.sleep(0.1)
        await self.channel.close()
        outcome = await asyncio.wait_for(pending, 2.0)
        self.assertIn(outcome.status, {"delivery_unknown", "rejected"})
        self.assertFalse(self.channel.is_running)
        # start/close must be repeatable, and the port must be free again.
        await self.channel.start()
        self.assertTrue(self.channel.is_running)
        await self.channel.close()
        probe = zmq.asyncio.Context.instance().socket(zmq.ROUTER)
        probe.setsockopt(zmq.LINGER, 0)
        try:
            probe.bind(f"tcp://127.0.0.1:{self.port}")
        finally:
            probe.close(linger=0)

    async def test_outcome_serialises_without_leaking_payload(self):
        await self.peer_with()
        outcome = await self.channel.inject(b"ex-control", frozen_batch())
        wire = json.dumps(outcome.to_dict(), ensure_ascii=False)
        self.assertNotIn("寻找红色杯子", wire)
        self.assertIn("payload_hash", wire)
        self.assertIsInstance(outcome, DeliveryOutcome)


if __name__ == "__main__":
    unittest.main()
