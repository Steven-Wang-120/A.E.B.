"""Forced-process termination at durable delivery boundaries (A07).

Children use production stores, pipeline/router and real ZMQ. The parent kills
them only after a committed boundary marker, then starts a fresh interpreter
over the unchanged database. Evidence is opt-in via UPPER_ACCEPTANCE_EVIDENCE.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def gate(root, value):
    write(root / "ready.json", {**value, "gate_time": time.time()})
    while True:
        time.sleep(1)


def projection(path):
    with sqlite3.connect(str(path)) as db:
        db.row_factory = sqlite3.Row
        return {name: [dict(row) for row in db.execute(f"SELECT * FROM {name}")]
                for name in ("upper_inputs", "upper_plans", "upper_segments",
                             "upper_batches", "upper_public_replies")}


async def aeb_worker(mode, root):
    from astrbot_plugin_astrbotex_interaction.tests.test_upper_e2e import (
        UpperStackAcceptanceTests, plan_action, plan_speech,
    )
    fixture = UpperStackAcceptanceTests("test_speech_only_sends_public_once_and_never_injects")
    await fixture.asyncSetUp()
    db = os.environ["ASTRBOTEX_TASK_DB_PATH"]
    def reached(**extra):
        gate(root, {"db": db, "before": projection(db), "mode": mode, **extra})
    if mode == "reserve":
        original = fixture.components.store.reserve_batch
        def reserve(*args, **kwargs):
            batch = original(*args, **kwargs)
            reached(batch=batch, wire_count=0)
            return batch
        fixture.components.store.reserve_batch = reserve
        await fixture.attach_peer()
    elif mode == "sent":
        peer = await fixture.attach_peer(silent=True)
        original = peer._reply_for
        def reply(message):
            if message["method"] == "upper.steps.inject":
                reached(wire=message, wire_count=1)
            return original(message)
        peer._reply_for = reply
    if mode == "public":
        from astrbot_plugin_astrbotex_interaction.tests.test_upper_e2e import make_event
        fixture.models.script["crash-input"] = plan_speech("UPPER_PUBLIC_crash")
        event = make_event("crash-input", message_id="crash-id")
        async def send(chain):
            write(root / "external.json", {"text": chain.get_plain_text(), "calls": 1})
            reached(public_count=1)
        event.send = send
        await fixture.plugin.take_over_upper_input(event)
    else:
        await fixture.deliver("crash-input", plan_action([
            "UPPER_EXEC_SENTINEL_crash_0", "UPPER_EXEC_SENTINEL_crash_1",
            "UPPER_EXEC_SENTINEL_crash_2"], send_count=2), message_id="crash-id")
    raise AssertionError("crash gate was not reached")


def ex_worker(root):
    from tests.test_upper_control_e2e import UpperControlCrossEndTests, upper_batch, jev
    class CrashFixture(UpperControlCrossEndTests):
        def _make_backend(self):
            def transport(body, headers, deadline, cancel, limit):
                write(root / "request.json", json.loads(body))
                gate(root, {"mode": "consume", "log": str(self.log_path),
                            "before": self.log_path.read_text(encoding="utf-8"),
                            "payload": payload, "model_calls": 1})
            return jev.JevBackend(jev.JevConfig(mode="shadow", min_interval_ms=0),
                                  transport=transport, secret_provider=lambda: "local-test")
    payload = upper_batch(["UPPER_EXEC_SENTINEL_consume_0"])
    fixture = CrashFixture("test_three_segments_reach_the_real_backend_request_in_order")
    fixture.setUp()
    fixture.call(fixture.start_control(), "upper.steps.inject", payload, timeout=30)
    # request_budget may expire while backend is gated; keep the process alive.
    while True:
        time.sleep(1)


def recover(mode, root):
    info = json.loads((root / "ready.json").read_text(encoding="utf-8"))
    if mode == "consume":
        from astrbot_ex.core.step_input import StepInputLog, StepInputIngress, build_trusted_context
        calls = []
        class Adapter:
            def supports(self): return True, ""
            def accept(self, *args, **kwargs):
                calls.append(args)
                raise AssertionError("recovery consumed again")
        log = StepInputLog(Path(info["log"]))
        context = build_trusted_context(robot_id="robot-e2e",
            ex_session=info["payload"]["ex_session"], route_refs=(info["payload"]["route_ref"],))
        ingress = StepInputIngress(log=log, adapter=Adapter(), trusted_context=lambda _: context)
        ingress.start()
        try:
            reply, _ = ingress.handle("control", "upper.steps.inject", info["payload"], None, feature="control")
            assert reply["duplicate"] and reply["phase"] == "unknown", reply
            time.sleep(.15)
            assert calls == []
        finally:
            ingress.close()
        write(root / "after.json", {"log": log.entries(), "replay": reply, "new_model_calls": 0})
        return
    from astrbot_plugin_astrbotex_interaction.task_store import TaskStore
    from astrbot_plugin_astrbotex_interaction.upper_store import UpperStore
    from astrbot_plugin_astrbotex_interaction.output_router import UpperOutputRouter
    store = TaskStore(info["db"])
    upper = UpperStore(store)
    plans = upper.list_plans()
    assert len(plans) == 1
    plan = upper.get_plan(plans[0]["plan_id"])
    before = info["before"]
    assert plan["plan_id"] == before["upper_plans"][0]["plan_id"]
    # Same event cannot be admitted after a restart.
    inbound = before["upper_inputs"][0]
    _, claimed = upper.claim_input(input_key=inbound["input_key"], robot_id=inbound["robot_id"],
        user_id=inbound["user_id"], session_id=inbound["session_id"])
    assert not claimed
    if mode == "public":
        async def send(text):
            raise AssertionError("recovery re-spoke public text")
        result = asyncio.run(UpperOutputRouter(upper).emit_public(plan["plan_id"], send))
        assert not result["sent"]
        assert plan["public_reply"]["status"] == "delivery_unknown", plan
    else:
        assert plan["batch"]["status"] == "delivery_unknown", plan
        assert [s["status"] for s in plan["segments"]] == ["delivery_unknown", "delivery_unknown", "pending"], plan
        assert [s["segment_id"] for s in plan["segments"]] == [s["segment_id"] for s in before["upper_segments"]]
        assert [s["text"] for s in plan["segments"]] == [s["text"] for s in before["upper_segments"]]
        assert plan["result_hash"] == before["upper_plans"][0]["result_hash"]
    time.sleep(.15)
    write(root / "after.json", {"database": projection(info["db"]), "new_io": 0, "replay_claimed": claimed})
    store.close()


class ForcedCrashRecoveryTests(unittest.TestCase):
    def crash(self, mode):
        with tempfile.TemporaryDirectory(prefix="upper-crash-parent-") as tmp:
            evidence = os.environ.get("UPPER_ACCEPTANCE_EVIDENCE")
            root = Path(evidence) / ("crash-" + mode) if evidence else Path(tmp)
            root.mkdir(parents=True, exist_ok=True)
            self.assertFalse((root / "ready.json").exists(), "use a new evidence directory")
            with (root / "child.log").open("w", encoding="utf-8") as output:
                child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                    "--worker", mode, str(root)], stdout=output, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 25
                    while not (root / "ready.json").exists():
                        self.assertIsNone(child.poll(), (root / "child.log").read_text(encoding="utf-8"))
                        self.assertLess(time.monotonic(), deadline, "child did not reach committed gate")
                        time.sleep(.02)
                    # No graceful shutdown/finally handlers can finalize the receipt.
                    child.kill()
                    child.wait(timeout=5)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=5)
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                "--recover", mode, str(root)], capture_output=True, text=True, timeout=20)
            write(root / "process.json", {"child_exit": child.returncode, "kill_time": time.time(),
                "recover_exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / "after.json").exists())

    def test_kill_after_reserve_before_send(self): self.crash("reserve")
    def test_kill_after_send_before_receipt(self): self.crash("sent")
    def test_kill_after_public_send_before_confirmation(self): self.crash("public")
    def test_kill_after_ex_consumption_starts(self): self.crash("consume")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        mode, root = sys.argv[2], Path(sys.argv[3])
        ex_worker(root) if mode == "consume" else asyncio.run(aeb_worker(mode, root))
    elif len(sys.argv) == 4 and sys.argv[1] == "--recover":
        recover(sys.argv[2], Path(sys.argv[3]))
    else:
        unittest.main()
