from __future__ import annotations

import asyncio
import random

from astrbot_plugin_astrbotex_interaction.planning_tools import PlanningTools, run_planning_turn
from astrbot_plugin_astrbotex_interaction.task_models import TaskError
from astrbot_plugin_astrbotex_interaction.tests.test_task_coordinator import CoordinatorFixture, goal_args
from astrbot_plugin_astrbotex_interaction.tests.test_task_store import authority, steps
from astrbot_plugin_astrbotex_interaction.tests.test_planning_tools import FakeHost, response


class IsolationTests(CoordinatorFixture):
    async def test_O01_goal_without_emit_has_no_public_exit(self):
        turn = self.turn()
        tools = PlanningTools(self.coordinator, self.router, turn, self.ex.context)
        host = FakeHost([response(["save_plan", "submit_current_goal", "finish_planning_turn"],
                   [{"steps": steps(), "expected_revision": 0}, goal_args(), {"outcome": "waiting_feedback"}])])
        await run_planning_turn(host, "provider", tools, "task")
        self.assertEqual(len(self.ex.goals), 1)
        self.assertEqual(self.public, [])

    async def test_100_seeded_private_marker_combinations_no_leak(self):
        rng = random.Random(906)
        for i in range(100):
            marker = f"PRIVATE-MARKER-{i}"
            self.store.invalidate(self.task["task_id"], authority())
            tools = PlanningTools(self.coordinator, self.router, self.turn(), self.ex.context)
            mode = rng.randrange(4)
            if mode == 0:
                output = response(["finish_planning_turn"], [{"outcome": "waiting_input"}], text=marker)
            elif mode == 1:
                output = response(["emit_user_message", "finish_planning_turn"],
                                  [{"text": "请确认？"}, {"outcome": "waiting_input"}], text=marker)
            elif mode == 2:
                output = response(["emit_user_message"], [{"text": marker}], chunk=True)
            else:
                output = RuntimeError(marker)
            try:
                await run_planning_turn(FakeHost([output]), "provider", tools, marker)
            except (TaskError, RuntimeError):
                pass
            self.assertNotIn(marker, str(self.public))
        self.assertFalse(self.ex.goals)

    async def test_P07_late_LLM_result_has_no_send_or_submit(self):
        started, release = asyncio.Event(), asyncio.Event()
        class LateHost:
            async def llm_generate(self, **kwargs):
                started.set()
                await release.wait()
                return response(["emit_user_message", "finish_planning_turn"],
                                [{"text": "stale public"}, {"outcome": "waiting_input"}])
        tools = PlanningTools(self.coordinator, self.router, self.turn(), self.ex.context)
        future = asyncio.create_task(run_planning_turn(LateHost(), "provider", tools, "task"))
        # A setup/import failure must surface, not strand the test on an unset Event.
        await asyncio.wait_for(started.wait(), 2.0)
        self.store.invalidate(self.task["task_id"], authority(), "changed")
        release.set()
        with self.assertRaisesRegex(TaskError, "stale_generation"):
            await future
        self.assertEqual(self.public, [])
        self.assertEqual(len(self.ex.goals), 0)
