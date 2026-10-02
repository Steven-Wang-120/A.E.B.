from __future__ import annotations

import unittest
from types import SimpleNamespace

from astrbot_plugin_astrbotex_interaction.planning_tools import PlanningTools, run_planning_turn, validate_call
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import CoordinatorFixture, goal_args
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import steps


def response(names, arguments, *, text="INTERNAL-SECRET", chunk=False):
    return SimpleNamespace(role="assistant", tools_call_name=names, tools_call_args=arguments,
                           tools_call_ids=[f"call-{i}" for i in range(len(names))],
                           completion_text=text, is_chunk=chunk)


class FakeHost:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class PlanningTests(CoordinatorFixture):
    async def test_private_goal_plus_emit_and_silent_finish(self):
        turn = self.turn()
        tools = PlanningTools(self.coordinator, self.router, turn, self.ex.context)
        host = FakeHost([response(["save_plan", "submit_current_goal", "emit_user_message", "finish_planning_turn"],
            [{"steps": steps(), "expected_revision": 0}, goal_args(),
             {"text": "我会尝试第一步。"}, {"outcome": "waiting_feedback"}])])
        await run_planning_turn(host, "configured-provider", tools, "private task")
        self.assertEqual(len(self.ex.goals), 1)
        self.assertEqual(len(self.public), 1)
        self.assertEqual(self.public[0]["text"], "我会尝试第一步。")
        self.assertNotIn("INTERNAL-SECRET", str(self.public))
        self.assertFalse(host.calls[0]["stream"])
        self.assertIn("CAN", host.calls[0]["system_prompt"])
        self.assertEqual({t.name for t in host.calls[0]["tools"].tools},
                         {"save_plan", "submit_current_goal", "emit_user_message", "finish_planning_turn"})

    async def test_O03_question_only_no_action(self):
        tools = PlanningTools(self.coordinator, self.router, self.turn(), self.ex.context)
        host = FakeHost([response(["emit_user_message", "finish_planning_turn"],
                                 [{"text": "请确认目标？", "claim": "question"}, {"outcome": "waiting_input"}])])
        await run_planning_turn(host, "provider", tools, "task")
        self.assertEqual(len(self.ex.goals), 0)
        self.assertEqual(len(self.public), 1)
        self.assertEqual(self.current()["status"], "waiting_input")

    async def test_P09_raw_text_empty_final_errors_chunks_not_success(self):
        for output in [response([], [], text="{\"submit_current_goal\": {}}"),
                       response([], [], text=""), RuntimeError("secret provider traceback"),
                       response(["emit_user_message"], [{"text": "secret chunk"}], chunk=True)]:
            with self.subTest(output=output):
                self.store.invalidate(self.task["task_id"], __import__(
                    "astrbot_plugin_astrbotex_interaction.tests.test_task_store", fromlist=["authority"]).authority())
                tools = PlanningTools(self.coordinator, self.router, self.turn(), self.ex.context)
                with self.assertRaises((TaskError, RuntimeError)):
                    await run_planning_turn(FakeHost([output]), "provider", tools, "task")
        self.assertEqual(self.public, [])
        self.assertEqual(len(self.ex.goals), 0)

    async def test_P06_batch_validation_before_side_effect(self):
        tools = PlanningTools(self.coordinator, self.router, self.turn(), self.ex.context)
        host = FakeHost([response(["emit_user_message", "finish_planning_turn"],
                         [{"text": "should not publish"}, {"outcome": "waiting_input", "public_text": "forbidden"}])])
        with self.assertRaises(TaskError):
            await run_planning_turn(host, "provider", tools, "task")
        self.assertEqual(self.public, [])
        for value in ['{"text":', '{"text":"a","text":"b"}', {"text": "x" * 70000},
                      {"text": "hi", "recipient": "other-session"}, {"text": float("nan")}]:
            with self.assertRaises(Exception):
                validate_call("emit_user_message", value)

    async def test_P09_round_limit(self):
        tools = PlanningTools(self.coordinator, self.router, self.turn(), self.ex.context)
        host = FakeHost([response(["save_plan"], [{"steps": steps(), "expected_revision": 0}])])
        with self.assertRaisesRegex(TaskError, "planning_round_limit"):
            await run_planning_turn(host, "provider", tools, "task", max_rounds=1)
        self.assertEqual(self.public, [])

    async def test_P01_fixed_fake_LLM_coordinator_completes_three_steps(self):
        import asyncio
        import json
        class ScriptHost:
            calls = 0
            async def llm_generate(inner, **kwargs):
                inner.calls += 1
                task = json.loads(kwargs["contexts"][0].content)["task"]
                if not task["steps"]:
                    return response(["save_plan", "submit_current_goal", "finish_planning_turn"],
                        [{"steps": steps(), "expected_revision": 0}, goal_args(), {"outcome": "waiting_feedback"}])
                if task["active_step"] == 3:
                    return response(["finish_planning_turn"], [{"outcome": "completed"}])
                return response(["submit_current_goal", "finish_planning_turn"],
                    [goal_args(f"step-{task['active_step']}"), {"outcome": "waiting_feedback"}])
        host = ScriptHost()
        self.coordinator.host = host
        self.coordinator.provider_id = "offline"
        self.coordinator.wake(self.task["task_id"])
        for i in range(3):
            for _ in range(100):
                task = self.current()
                if task["current_goal"] and task["current_goal"]["revision"] is not None and task["turn_id"] is None:
                    break
                await asyncio.sleep(0.001)
            else:
                self.fail("private planner did not dispatch")
            self.assertEqual(task["active_step"], i)
            await self.coordinator.feedback("r1", "route-s1", self.ex.feedback(task, "succeeded"))
        for _ in range(100):
            if not self.current()["active"]:
                break
            await asyncio.sleep(0.001)
        self.assertEqual(self.current()["status"], "completed")
        self.assertEqual(len(self.ex.goals), 3)
        self.assertEqual(host.calls, 4)
        self.assertEqual(self.public, [])

    async def test_actual_public_Context_llm_generate_with_fake_provider(self):
        # Imports the real Host and runs its public method, not a mocked import.
        # Provider is deterministic and offline; this is not a live LLM/platform test.
        from astrbot.api.star import Context
        from astrbot.core.provider.provider import Provider
        from astrbot.core.provider.entities import LLMResponse
        class OfflineProvider(Provider):
            def get_current_key(self):
                return "offline"
            def set_key(self, key):
                pass
            async def get_models(self):
                return []
            async def text_chat(self, **kwargs):
                self.kwargs = kwargs
                return LLMResponse(role="assistant", tools_call_name=["finish_planning_turn"],
                                   tools_call_args=[{"outcome": "waiting_input"}], tools_call_ids=["finish"])
        provider = object.__new__(OfflineProvider)
        class Manager:
            async def get_provider_by_id(self, provider_id):
                return provider
        host = object.__new__(Context)
        host.provider_manager = Manager()
        tools = PlanningTools(self.coordinator, self.router, self.turn(), self.ex.context)
        await run_planning_turn(host, "offline", tools, "test public method")
        self.assertIn("func_tool", provider.kwargs)
        self.assertFalse(provider.kwargs["stream"])
        self.assertEqual(self.public, [])
