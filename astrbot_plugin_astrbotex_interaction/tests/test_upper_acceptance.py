"""A03-A06/A08: production Host adapters, sockets and local HTTP backend.

Only the paid LLM responses, OneBot API peer and audio providers/devices are
local substitutes. AiocqhttpMessageEvent, upper modules, EX connection manager,
ingress, decision service, JevBackend HTTP transport and InteractionCore run.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

from astrbot_plugin_astrbotex_interaction.tests import test_upper_e2e as e2e
from astrbot_plugin_astrbotex_interaction.tests.test_upper_e2e import (
    CONTROL_PEER, EX_SESSION, make_event,
    plan_action, plan_both, plugin_module,
)
from astrbot_plugin_astrbotex_interaction.tests.test_upper_composition import ex_event
from astrbot_plugin_astrbotex_interaction.tests.test_upper_models import (
    FakeHost, probabilities, emit,
)
from astrbot_plugin_astrbotex_interaction.upper_models import UpperModels


class ContextHost(FakeHost):
    """Run real Context.llm_generate with concrete offline Host Providers."""
    def __init__(self, responses):
        from astrbot.core.provider.provider import Provider
        super().__init__(responses)
        owner = self
        class LocalProvider(Provider):
            def get_current_key(self): return "local"
            def set_key(self, key): pass
            async def get_models(self): return ["local"]
            def meta(self): return SimpleNamespace(id=self.provider_config["id"])
            async def text_chat(self, **kwargs):
                item = owner.responses.pop(0)
                if callable(item): item = item()
                if isinstance(item, BaseException): raise item
                return item
        self.providers = {key: LocalProvider({"id":key}, {}) for key in ("prov-router", "prov-generator")}
        async def get_provider(key): return self.providers.get(key)
        self.context = SimpleNamespace(provider_manager=SimpleNamespace(get_provider_by_id=get_provider))
    async def llm_generate(self, **kwargs):
        from astrbot.core.star.context import Context
        self.calls.append(kwargs)
        return await Context.llm_generate(self.context, **kwargs)


def record(name, data):
    root = os.environ.get("UPPER_ACCEPTANCE_EVIDENCE")
    if root:
        target = Path(root) / (name + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class SystemAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.aeb = e2e.UpperStackAcceptanceTests("test_speech_only_sends_public_once_and_never_injects")
        await self.aeb.asyncSetUp()
        self.addAsyncCleanup(self.aeb.asyncTearDown)
        self.addCleanup(self.aeb.doCleanups)
        self.ex = None

    async def real_ex(self):
        from tests.test_upper_control_e2e import UpperControlCrossEndTests, instruction_ack, jev
        class HTTPFixture(UpperControlCrossEndTests):
            def _make_backend(f):
                captured = f.captured
                class Handler(BaseHTTPRequestHandler):
                    def do_POST(handler):
                        captured.append(handler.rfile.read(int(handler.headers["Content-Length"])))
                        reply = instruction_ack()
                        handler.send_response(200)
                        handler.send_header("Content-Length", str(len(reply)))
                        handler.end_headers()
                        handler.wfile.write(reply)
                    def log_message(*args): pass
                server = HTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                f.addCleanup(server.server_close)
                f.addCleanup(thread.join, 3)
                f.addCleanup(server.shutdown)
                backend = jev.JevBackend(jev.JevConfig(base_url=f"http://127.0.0.1:{server.server_port}",
                    allow_live_http=True, mode="shadow", min_interval_ms=0),
                    secret_provider=lambda: "local-only")
                f.addCleanup(backend.close)
                return backend
        f = HTTPFixture("test_three_segments_reach_the_real_backend_request_in_order")
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.manager.update("control", {"config": {"endpoint": f"tcp://127.0.0.1:{self.aeb.control_port}",
            "identity": CONTROL_PEER, "protocol_profile": "astrbotex", "channel": "control",
            "upper_robot_id": "robot-e2e", "upper_route_refs": "upper-control", "upper_ex_session": EX_SESSION}})
        f.manager.set_control_session_provider(lambda: EX_SESSION)
        f.manager.start("control")
        await self.online(self.aeb.components.control, CONTROL_PEER.encode())
        self.ex = f
        return f

    async def online(self, channel, peer):
        for _ in range(300):
            if peer in channel.online_peers(): return
            await asyncio.sleep(.01)
        self.fail("peer did not announce hello")

    def onebot_event(self, text, identity, captures):
        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
        class Bot:
            async def send_private_msg(bot, **kwargs):
                captures.append(kwargs)
        raw = make_event(text, message_id=identity, user="700001", session="700001")
        event = AiocqhttpMessageEvent(text, raw.message_obj,
            plugin_module.PlatformMetadata(name="aiocqhttp", description="OneBot", id="acceptance-onebot"),
            "700001", Bot())
        event.role = "admin"
        event.is_at_or_wake_command = True
        event.is_wake = True
        return event

    async def text_core(self):
        from astrbot_ex.core.connection_manager import ConnectionManager
        from astrbot_ex.core.interaction_core import InteractionCore
        from astrbot_ex.core.plugin_registry import PluginRegistry
        from astrbot_ex.core.event_bus import EventBus
        from astrbot_ex.core.topic_bus import TopicBus
        captures = {"reply": [], "tts": [], "stt": [], "play": [], "outgoing": []}
        class TTS:
            async def get_audio(self, text):
                captures["tts"].append(text)
                return "/tmp/acceptance-local-audio.wav"
        class STT:
            async def get_text(self, audio_url):
                captures["stt"].append(audio_url)
                return "用户原始语音输入"
        topics, events = TopicBus(), EventBus()
        core = InteractionCore(registry=PluginRegistry(), topic_bus=topics, event_bus=events,
                               tts_provider=TTS(), stt_provider=STT())
        self.addCleanup(core.on_runtime_stop, "acceptance cleanup")
        for key, topic in (("play", "interaction_core.audio.play"),
                           ("outgoing", "interaction_core.message.outgoing")):
            self.addCleanup(topics.subscribe(topic, lambda message, key=key: captures[key].append(message.payload)))
        manager = ConnectionManager(self.aeb.data_root / "text-connections.json", event_bus=events, topic_bus=topics)
        self.addCleanup(manager.close)
        def handle(*args, **kwargs):
            # Production manager decoded and routed the real text ZMQ frames.
            method, payload = args[1], args[2]
            if method == "interaction.reply":
                captures["reply"].append(payload)
                core.handle_astrbot_reply(payload)
                return {"ok": True}, None
            return {"ok": False}, None
        manager.set_business_handler(handle)
        manager.create({"id": "text", "name": "public text", "type": "zmq_client", "enabled": False,
            "config": {"endpoint": f"tcp://127.0.0.1:{self.aeb.text_port}", "identity": "acceptance-text",
                       "protocol_profile": "astrbotex", "channel": "text"}})
        manager.start("text")
        await self.online(self.aeb.plugin.text_channel, b"acceptance-text")
        return captures

    async def test_ten_routes_real_onebot_voice_tts_and_http_no_leaks(self):
        ex = await self.real_ex()
        audio = await self.text_core()
        settings = self.aeb.components.settings_repository
        settings.save({"routing_provider_id":"prov-router", "generation_provider_id":"prov-generator"},
                      settings.load().revision)
        platform = []
        matrix, all_calls = [], []
        cases = [((.90,.04,.06),"speech_only"), ((.05,.90,.05),"action_only"),
                 ((.05,.05,.90),"speech_and_action"), ((.50,0,.50),"speech_only"),
                 ((0,.50,.50),"speech_and_action")]
        for source in ("onebot", "voice"):
            for index, (probs, route) in enumerate(cases,1):
                key = f"{source}-R{index}"
                public = "UPPER_PUBLIC_" + key
                segments = ["UPPER_EXEC_SENTINEL_"+key+"_0"] if route != "speech_only" else []
                payload = {"public_text": public if route != "action_only" else None,
                           "segments": [{"text": s} for s in segments], "send_count": len(segments)}
                host = ContextHost([probabilities(*probs), emit(**payload)])
                # Real UpperModels uses the Host public llm_generate contract.
                models = UpperModels(host, timeout_s=2)
                self.aeb.components.pipeline.models = models
                event = (self.onebot_event(key,key,platform) if source == "onebot" else
                         ex_event(key, peer=b"acceptance-text",adapter=self.aeb.plugin._adapter))
                event.session_id = EX_SESSION if source == "voice" else event.session_id
                before = [len(platform),len(audio["reply"]),len(audio["tts"]),len(audio["play"]),len(ex.captured)]
                await self.aeb.plugin.take_over_upper_input(event)
                # Replay a fresh Host event with the same framework message identity.
                replay = (self.onebot_event(key,key,platform) if source == "onebot" else
                          ex_event(key,peer=b"acceptance-text",adapter=self.aeb.plugin._adapter))
                replay.session_id = event.session_id
                await self.aeb.plugin.take_over_upper_input(replay)
                delta = [n-b for n,b in zip([len(platform),len(audio["reply"]),len(audio["tts"]),len(audio["play"]),len(ex.captured)],before)]
                speak = int(route != "action_only")
                self.assertEqual(delta, [speak if source=="onebot" else 0,
                    speak if source=="voice" else 0,speak if source=="voice" else 0,
                    speak if source=="voice" else 0,int(bool(segments))],
                    (key,event.get_extra("upper_error"),host.calls,self.aeb.plugin._upper_last_delivery))
                self.assertFalse(event.call_llm)
                self.assertTrue(event.is_stopped())
                self.assertEqual(len(host.calls),2,key)
                self.assertEqual([c["chat_provider_id"] for c in host.calls],["prov-router","prov-generator"])
                for call in host.calls:
                    self.assertFalse(call["stream"])
                    tools=call.get("func_tool") or call.get("tools")
                    self.assertIsNotNone(tools)
                all_calls.extend([{k:str(v) for k,v in call.items() if k != "contexts"} for call in host.calls])
                plans=self.aeb.components.store.list_plans(limit=100)
                detail=next(self.aeb.components.store.get_plan(p["plan_id"]) for p in plans
                            if p["input_key"]==event.get_extra("upper_input_key"))
                self.assertEqual(detail["route"],route)
                if segments:
                    body=json.loads(ex.captured[-1])
                    wire=detail["batch"]["payload"]
                    self.assertEqual(body["state"]["instruction_input"],wire)
                matrix.append({"id":key,"route":route,"counts":delta,"plan":detail})
        public_surfaces=json.dumps([platform,audio],ensure_ascii=False)
        self.assertNotIn("UPPER_EXEC_SENTINEL_",public_surfaces)
        wire=json.dumps([row["plan"]["batch"] for row in matrix],ensure_ascii=False)
        self.assertNotIn("UPPER_PUBLIC_",wire)
        self.assertEqual(audio["stt"],[])  # Downlink never re-enters STT.
        ex.assert_no_execution()
        record("route-output-matrix",{"matrix":matrix,"onebot":platform,"voice":audio,
            "host_default_calls":0,"private_model_calls":all_calls,"backend_requests":[json.loads(b) for b in ex.captured],
            "request_hashes":[hashlib.sha256(b).hexdigest() for b in ex.captured],"execution_leaks":0,"public_control_leaks":0})

    async def test_concurrent_twenty_replays_and_other_origin_not_muted(self):
        await self.aeb.attach_peer()
        captures=[]
        script=plan_both("UPPER_PUBLIC_twenty",["UPPER_EXEC_SENTINEL_twenty_0"])
        self.aeb.models.script["twenty"]=script
        self.aeb.models.script["silent"]=plan_action(["UPPER_EXEC_SENTINEL_other_0"])
        events=[self.onebot_event("twenty","same-host-id",captures) for _ in range(20)]
        events.append(self.onebot_event("silent","other-host-id",captures))
        await asyncio.gather(*(self.aeb.plugin.take_over_upper_input(e) for e in events))
        self.assertEqual(self.aeb.models.classify_calls.count("twenty"),1)
        self.assertEqual(len(captures),1)
        self.assertEqual(len(self.aeb.injects()),2)
        plans=self.aeb.components.store.list_plans()
        self.assertEqual(len(plans),2)
        record("concurrent-replay",{"replays":20,"plans":plans,"onebot":captures,"wire":self.aeb.injects()})

    async def test_S1_S2_S3_S4_real_http_wait_and_restart(self):
        ex=await self.real_ex()
        settings=self.aeb.components.settings_repository
        settings.save({"model_timeout_s":1},settings.load().revision)
        rows=[]
        for case,count,total in (("S1",1,1),("S2",3,3),("S3",1,3),("S4",2,3)):
            texts=(["寻找→对齐→抓取 UPPER_EXEC_SENTINEL_S1_0"] if total==1 else
                   [f"第{i}段 UPPER_EXEC_SENTINEL_{case}_{i}" for i in range(total)])
            await self.aeb.deliver(case,plan_both("UPPER_PUBLIC_"+case,texts,count))
            plan=self.aeb.components.store.get_plan(self.aeb.components.store.list_plans()[0]["plan_id"])
            body=json.loads(ex.captured[-1])
            self.assertEqual(body["state"]["instruction_input"],plan["batch"]["payload"])
            self.assertEqual([s["text"] for s in plan["segments"]],texts)
            self.assertEqual([s["text"] for s in body["state"]["instruction_input"]["segments"]],texts[:count])
            self.assertEqual([s["status"] for s in plan["segments"]][count:],["pending"]*(total-count))
            db=self.aeb.components.store.db
            raw=dict(db.execute("SELECT * FROM upper_plans WHERE plan_id=?",(plan["plan_id"],)).fetchone())
            for name in ("settings","catalog","prompt"):
                self.assertTrue(raw[name+"_hash"])
                self.assertIsNotNone(raw[name+"_snapshot"])
            rows.append({"case":case,"plan":plan,"raw_snapshot":raw,"request":body,
                         "request_hash":hashlib.sha256(ex.captured[-1]).hexdigest()})
        before=[r["plan"] for r in rows]
        count=len(ex.captured)
        from astrbot_plugin_astrbotex_interaction.zmq_transport import ZmqRemoteError
        wrong=dict(before[-1]["batch"]["payload"],ex_session="stale-acceptance-session")
        with self.assertRaises(ZmqRemoteError) as rejected:
            await self.aeb.components.control.channel.request("upper.steps.inject",wrong,
                peer=CONTROL_PEER.encode(),timeout_sec=2)
        self.assertIn("session_mismatch",str(rejected.exception))
        self.assertEqual(len(ex.captured),count)
        await asyncio.sleep(2.1)  # At least two saved model_timeout windows.
        self.assertEqual(len(ex.captured),count)
        await self.aeb.plugin.terminate()
        await self.aeb.boot()
        await asyncio.sleep(2.1)
        self.assertEqual(len(ex.captured),count)
        after=[self.aeb.components.store.get_plan(p["plan_id"]) for p in before]
        for a,b in zip(before,after):
            self.assertEqual(a["segments"],b["segments"])
            self.assertEqual(a["batch"],b["batch"])
            self.assertEqual(a["result_hash"],b["result_hash"])
        ex.assert_no_execution()
        record("segment-http-restart",{"cases":rows,"after":after,"wait_s":4.2,"timeout_s":1,
            "extra_model_calls":0,"wrong_session":str(rejected.exception)})

    async def test_external_skill_changes_only_future_frozen_plan(self):
        from astrbot_plugin_astrbotex_interaction.action_catalog import ActionCatalog
        await self.aeb.attach_peer()
        await self.aeb.deliver("old-skill",plan_action(["UPPER_EXEC_SENTINEL_old_skill"]))
        first=self.aeb.components.store.list_plans()[0]
        db=self.aeb.components.store.db
        def raw(plan):
            return dict(db.execute("SELECT * FROM upper_plans WHERE plan_id=?",(plan["plan_id"],)).fetchone())
        frozen=raw(first)
        catalog=ActionCatalog(self.aeb.data_root)
        snapshot=catalog.snapshot()
        file=catalog.root/"versions"/snapshot.version/"actions"/"pick_cup"/"SKILL.md"
        file.write_text("# 外部受控编辑\n仅新任务读取 EXTERNAL_SKILL_V2。",encoding="utf-8")
        await self.aeb.deliver("new-skill",plan_action(["UPPER_EXEC_SENTINEL_new_skill"]))
        second=raw(self.aeb.components.store.list_plans()[0])
        self.assertNotEqual(frozen["catalog_hash"],second["catalog_hash"])
        self.assertNotEqual(frozen["prompt_hash"],second["prompt_hash"])
        self.assertIn("EXTERNAL_SKILL_V2",second["catalog_snapshot"])
        self.assertNotIn("EXTERNAL_SKILL_V2",frozen["catalog_snapshot"])
        self.assertEqual(frozen,raw(first))
        record("external-skill-frozen-plans",{"before":frozen,"new":second,"old_after":raw(first)})

    async def test_real_model_failures_and_epoch_change_emit_nothing(self):
        from astrbot_plugin_astrbotex_interaction.tests.test_upper_models import response
        await self.aeb.attach_peer()
        settings=self.aeb.components.settings_repository
        settings.save({"model_timeout_s":1},settings.load().revision)
        public=[]
        class Slow(FakeHost):
            async def llm_generate(self,**kwargs):
                self.calls.append(kwargs)
                await asyncio.sleep(5)
        class SlowGeneration(Slow):
            async def llm_generate(self,**kwargs):
                if not self.calls:
                    self.calls.append(kwargs)
                    return probabilities(.05,.90,.05)
                return await super().llm_generate(**kwargs)
        cases=[("classify-timeout",Slow([]),1),
               ("generation-timeout",SlowGeneration([]),1),
               ("no-tool",FakeHost([probabilities(.05,.90,.05),response(text="private prose")]),1),
               ("invalid-provider",FakeHost([]),1)]
        results=[]
        for key,host,rounds in cases:
            if key=="invalid-provider":
                settings.save({"routing_provider_id":"does-not-exist"},settings.load().revision)
            self.aeb.components.pipeline.models=UpperModels(host,max_rounds=rounds)
            event=self.onebot_event(key,key,public)
            await self.aeb.plugin.take_over_upper_input(event)
            self.assertTrue(event.get_extra("upper_error"),key)
            self.assertFalse(event.call_llm)
            results.append({"case":key,"error":event.get_extra("upper_error"),"calls":len(host.calls)})
        settings.save({"routing_provider_id":""},settings.load().revision)
        entered,release=asyncio.Event(),asyncio.Event()
        class Late(FakeHost):
            async def llm_generate(self,**kwargs):
                self.calls.append(kwargs)
                entered.set()
                await release.wait()
                return probabilities(.05,.90,.05)
        host=Late([])
        models=UpperModels(host)
        self.aeb.components.pipeline.models=models
        event=self.onebot_event("late-epoch","late-epoch",public)
        pending=asyncio.create_task(self.aeb.plugin.take_over_upper_input(event))
        await asyncio.wait_for(entered.wait(),2)
        models.invalidate()
        release.set()
        await pending
        self.assertEqual(public,[])
        self.assertEqual(self.aeb.injects(),[])
        self.assertEqual(self.aeb.components.store.list_plans(),[])
        results.append({"case":"epoch-change","error":event.get_extra("upper_error"),"calls":len(host.calls)})
        record("model-failures-and-epoch",{"cases":results,"public_calls":0,"control_calls":0,"plans":0})

    async def test_feature_off_three_reloads_release_control_listener(self):
        import socket
        await self.aeb.plugin.terminate()
        config=dict(self.aeb.plugin.plugin_config,upper_enabled=False,control_enabled=False)
        rounds=[]
        for index in range(3):
            await self.aeb.boot(config)
            self.assertFalse(self.aeb.plugin.upper_ready)
            self.assertIsNotNone(self.aeb.plugin.text_channel)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",self.aeb.control_port))
            event=make_event("feature-off-"+str(index))
            await self.aeb.plugin.take_over_upper_input(event)
            self.assertFalse(event.get_extra("upper_handled",False))
            await self.aeb.plugin.terminate()
            rounds.append({"reload":index+1,"control_listener":False,"upper_handled":False})
        self.assertEqual(self.aeb.models.classify_calls,[])
        record("feature-off-three-reloads",{"rounds":rounds,"classifier_calls":0,"control_calls":0})

    async def test_onebot_send_failure_is_unknown_and_never_repeated(self):
        attempts=[]
        event=self.onebot_event("failed-platform","failed-platform",[])
        async def fail(**kwargs):
            attempts.append(kwargs)
            raise ConnectionError("local OneBot fixture disconnected")
        event.bot.send_private_msg=fail
        self.aeb.models.script["failed-platform"]={
            "probabilities":{"speech_only":.9,"action_only":.05,"speech_and_action":.05},
            "public_text":"UPPER_PUBLIC_failed_platform","segments":[],"send_count":0}
        await self.aeb.plugin.take_over_upper_input(event)
        plan=self.aeb.components.store.get_plan(self.aeb.components.store.list_plans()[0]["plan_id"])
        self.assertEqual(plan["public_reply"]["status"],"delivery_unknown")
        replay=self.onebot_event("failed-platform","failed-platform",[])
        replay.bot.send_private_msg=fail
        await self.aeb.plugin.take_over_upper_input(replay)
        self.assertEqual(len(attempts),1)
        record("public-send-failure",{"plan":plan,"send_attempts":len(attempts),"automatic_retry":False})
