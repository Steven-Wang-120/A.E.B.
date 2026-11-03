"""Contract tests for the B07 Dashboard management API.

Scope: unit_fake.  The tests drive the ten handlers through a recording
responder that mirrors the Host envelope, so the assertions cover HTTP status,
the stable ``[UPPER:code]`` message prefix and the response shape without
needing a live AstrBot process.  ``real_host`` evidence is produced separately
by B11 against a running Dashboard.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[2]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from astrbot_plugin_astrbotex_interaction.upper_api import (  # noqa: E402
    PLUGIN_NAME,
    ROUTE_SPECS,
    ApiResult,
    UpperApi,
    UpperApiDependencies,
    register_upper_api,
)


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class RecordingResponder:
    """Same envelope semantics as ``astrbot.api.web`` without importing it."""

    def __init__(self) -> None:
        self.calls: list[ApiResult] = []

    def ok(self, payload: Any, status_code: int = 200) -> ApiResult:
        result = ApiResult(status_code=status_code, body=payload)
        self.calls.append(result)
        return result

    def error(self, message: str, status_code: int, data: Any = None) -> ApiResult:
        result = ApiResult(
            status_code=status_code,
            body={"status": "error", "message": message, "data": data},
        )
        self.calls.append(result)
        return result


class FakeQuery:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = dict(params or {})

    def get(self, key: str, default: Any = None, type: Any = None) -> Any:  # noqa: A002
        raw = self._params.get(key, default)
        if type is not None and raw is not None and raw is not default:
            try:
                return type(raw)
            except (TypeError, ValueError):
                return raw
        return raw


class FakeRequest:
    def __init__(self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None):
        self._body = body if body is not None else {}
        self.query = FakeQuery(params)

    async def json(self, default: Any = None) -> Any:
        return self._body if self._body is not None else default


class RepositoryError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class FakeSettingsRepository:
    """Revision-CAS settings store over a shared backing dict (survives rebuild)."""

    def __init__(self, backing: dict[str, Any] | None = None) -> None:
        self.backing = backing if backing is not None else {"values": {}, "revision": 1}
        self.save_calls: list[dict[str, Any]] = []

    def load(self) -> dict[str, Any]:
        return {
            "values": copy.deepcopy(self.backing["values"]),
            "revision": self.backing["revision"],
        }

    def save(self, data: dict[str, Any], expected_revision: int) -> dict[str, Any]:
        self.save_calls.append(dict(data))
        if expected_revision != self.backing["revision"]:
            raise RepositoryError("revision_conflict")
        self.backing["values"].update(copy.deepcopy(data))
        self.backing["revision"] += 1
        return {"revision": self.backing["revision"]}


class FakeActionCatalog:
    """Versioned catalog over a shared backing dict."""

    def __init__(self, backing: dict[str, Any] | None = None) -> None:
        self.backing = backing if backing is not None else {
            "records": {},
            "revision": 1,
            "version": "cat-1",
            "next": 1,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "records": [copy.deepcopy(row) for row in self.backing["records"].values()],
            "revision": self.backing["revision"],
            "catalog_version": self.backing["version"],
        }

    def save(self, record: dict[str, Any], expected_revision: int) -> dict[str, Any]:
        if expected_revision != self.backing["revision"]:
            raise RepositoryError("revision_conflict")
        action_ref = record.get("action_ref")
        if not action_ref:
            action_ref = f"aeb-{self.backing['next']:03d}"
            self.backing["next"] += 1
        elif action_ref not in self.backing["records"]:
            raise RepositoryError("not_found")
        self.backing["records"][action_ref] = {
            **copy.deepcopy(record),
            "action_ref": action_ref,
            "revision": expected_revision + 1,
        }
        self.backing["revision"] = expected_revision + 1
        self.backing["version"] = f"cat-{self.backing['revision']}"
        return {
            "action_ref": action_ref,
            "revision": self.backing["revision"],
            "catalog_version": self.backing["version"],
        }

    def delete(self, action_ref: str, expected_revision: int) -> dict[str, Any]:
        if expected_revision != self.backing["revision"]:
            raise RepositoryError("revision_conflict")
        if action_ref not in self.backing["records"]:
            raise RepositoryError("not_found")
        del self.backing["records"][action_ref]
        self.backing["revision"] = expected_revision + 1
        self.backing["version"] = f"cat-{self.backing['revision']}"
        return {
            "action_ref": action_ref,
            "revision": self.backing["revision"],
            "catalog_version": self.backing["version"],
        }


class FakePlanRepository:
    def __init__(self, plans: list[dict[str, Any]] | None = None) -> None:
        self.plans = plans if plans is not None else []

    def list_plans(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], bool]:
        window = self.plans[offset : offset + limit]
        return [copy.deepcopy(item) for item in window], offset + limit < len(self.plans)

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        for plan in self.plans:
            if plan.get("plan_id") == plan_id:
                return copy.deepcopy(plan)
        return None


class FakeContext:
    """Mirrors ``Context.register_web_api`` replace-on-route+methods semantics."""

    def __init__(self) -> None:
        self.registered_web_apis: list[tuple[str, Any, list[str], str]] = []

    def register_web_api(self, route, view_handler, methods, desc) -> None:
        for index, entry in enumerate(self.registered_web_apis):
            if entry[0] == route and entry[2] == methods:
                self.registered_web_apis[index] = (route, view_handler, methods, desc)
                return
        self.registered_web_apis.append((route, view_handler, methods, desc))


def plan_fixture(plan_id: str = "plan-1", segments: int = 3, sent: int = 1) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "route": "speech_and_action",
        "input_key": "input-" + plan_id,
        "plan_revision": 1,
        "sent_prefix": sent,
        "prompt_version": "prompt-hash",
        "catalog_version": "cat-1",
        "created_at": "2026-11-02T06:00:00Z",
        "delivery_phase": "injected",
        "segments": [
            {"segment_id": f"s{i}", "ordinal": i, "text": f"第 {i} 段：寻找红色杯子。"}
            for i in range(segments)
        ],
        "deliveries": [
            {
                "batch_id": "batch-1",
                "phase": "injected",
                "payload_hash": "abc123",
                "segment_ids": ["s0"],
                "attempted_at": "2026-11-02T06:00:01Z",
            }
        ],
    }


# --------------------------------------------------------------------------
# Base case
# --------------------------------------------------------------------------


class UpperApiTestCase(unittest.IsolatedAsyncioTestCase):
    def build(
        self,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        settings: FakeSettingsRepository | None = None,
        catalog: FakeActionCatalog | None = None,
        plans: FakePlanRepository | None = None,
        ex_availability: Any = None,
        runtime_status: Any = None,
        prompt_defaults: Any = None,
    ) -> tuple[UpperApi, RecordingResponder]:
        self.responder = RecordingResponder()
        self.settings_backing = getattr(settings, "backing", {"values": {}, "revision": 1})
        dependencies = UpperApiDependencies(
            settings=settings if settings is not None else FakeSettingsRepository(),
            actions=catalog if catalog is not None else FakeActionCatalog(),
            plans=plans if plans is not None else FakePlanRepository(),
            ex_availability=ex_availability,
            runtime_status=runtime_status,
            prompt_defaults=prompt_defaults,
        )
        api = UpperApi(
            dependencies,
            responder=self.responder,
            request_proxy=FakeRequest(body or {}, params or {}),
        )
        return api, self.responder

    def assert_ok(self, result: ApiResult, status: int = 200) -> dict[str, Any]:
        self.assertEqual(result.status_code, status, result.body)
        self.assertNotIn("status", result.body)
        self.assertNotIn("data", result.body)
        return result.body

    def assert_error(self, result: ApiResult, code: str, status: int) -> dict[str, Any]:
        self.assertEqual(result.status_code, status, result.body)
        self.assertEqual(result.body["status"], "error")
        self.assertEqual(result.body["data"]["error_code"], code)
        self.assertTrue(
            result.body["message"].startswith(f"[UPPER:{code}]"), result.body["message"]
        )
        return result.body


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


class SettingsEndpointTests(UpperApiTestCase):
    async def test_get_settings_returns_values_revision_scope_and_runtime(self) -> None:
        api, _ = self.build(
            runtime_status=lambda: {
                "control_state": "running",
                "control_enabled": True,
                "control_peer": "ex-1",
                "ex_session": "session-a",
                "input_ready": True,
                "last_delivery": None,
            }
        )
        body = self.assert_ok(await api.get_settings())
        self.assertEqual(body["revision"], 1)
        self.assertFalse(body["settings"]["upper_enabled"])
        self.assertEqual(body["settings"]["control_port"], 8769)
        self.assertEqual(body["apply_scope"]["control_port"], "reload_required")
        self.assertEqual(body["runtime"]["control_state"], "running")
        self.assertTrue(body["runtime"]["input_ready"])
        self.assertEqual(len(body["fields"]), len(body["settings"]))

    async def test_get_settings_without_repository_is_503(self) -> None:
        responder = RecordingResponder()
        api = UpperApi(
            UpperApiDependencies(),
            responder=responder,
            request_proxy=FakeRequest(),
        )
        self.assert_error(await api.get_settings(), "dependency_unavailable", 503)

    async def test_get_settings_never_returns_secrets(self) -> None:
        api, _ = self.build()
        body = self.assert_ok(await api.get_settings())
        text = repr(body).lower()
        for forbidden in ("api_key", "base_url", "sk-", "token", "password"):
            self.assertNotIn(forbidden, text)

    async def test_save_settings_updates_revision_and_reports_scope(self) -> None:
        api, _ = self.build(body={"settings": {"upper_enabled": True, "control_port": 8790}, "expected_revision": 1})
        body = self.assert_ok(await api.save_settings())
        self.assertEqual(body["revision"], 2)
        self.assertEqual(body["changed"], ["upper_enabled", "control_port"])
        self.assertEqual(body["reload_required"], ["control_port"])
        self.assertEqual(body["next_input"], ["upper_enabled"])

    async def test_save_settings_rejects_unknown_field(self) -> None:
        api, _ = self.build(body={"settings": {"api_key": "sk-x"}, "expected_revision": 1})
        self.assert_error(await api.save_settings(), "field_not_allowed", 400)

    async def test_save_settings_rejects_wrong_types(self) -> None:
        cases = [
            {"upper_enabled": "yes"},
            {"upper_enabled": 1},
            {"control_port": True},
            {"control_port": 70000},
            {"model_timeout_s": 0},
            {"control_bind_host": "http://127.0.0.1/x"},
        ]
        for case in cases:
            with self.subTest(case=case):
                api, _ = self.build(body={"settings": case, "expected_revision": 1})
                self.assert_error(await api.save_settings(), "invalid_request", 400)

    async def test_save_settings_revision_conflict_is_409(self) -> None:
        api, _ = self.build(body={"settings": {"upper_enabled": True}, "expected_revision": 7})
        self.assert_error(await api.save_settings(), "revision_conflict", 409)

    async def test_save_settings_rejects_bool_revision(self) -> None:
        api, _ = self.build(body={"settings": {"upper_enabled": True}, "expected_revision": True})
        self.assert_error(await api.save_settings(), "invalid_request", 400)

    async def test_save_settings_persists_across_repository_rebuild(self) -> None:
        backing = {"values": {}, "revision": 1}
        api, _ = self.build(
            body={"settings": {"routing_provider_id": "prov-a"}, "expected_revision": 1},
            settings=FakeSettingsRepository(backing),
        )
        self.assert_ok(await api.save_settings())

        rebuilt, _ = self.build(settings=FakeSettingsRepository(backing))
        body = self.assert_ok(await rebuilt.get_settings())
        self.assertEqual(body["settings"]["routing_provider_id"], "prov-a")
        self.assertEqual(body["revision"], 2)


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


class RealSettingsEndpointTests(UpperApiTestCase):
    async def test_real_settings_snapshot_read_save_reload_and_prompts(self):
        from astrbot_plugin_astrbotex_interaction.upper_settings import SettingsRepository
        class Config(dict):
            def save_config(self):
                self.saved = dict(self)
        config = Config(routing_provider_id="router-real", model_timeout_s=45.0)
        repository = SettingsRepository(config)
        api, _ = self.build(settings=repository)
        initial = self.assert_ok(await api.get_settings())
        self.assertEqual(initial["settings"]["routing_provider_id"], "router-real")
        self.assertEqual(initial["settings"]["model_timeout_s"], 45.0)
        save, _ = self.build(settings=repository, body={"expected_revision": initial["revision"],
            "settings": {"model_timeout_s": 46.0, "generation_provider_id": "generator-real"}})
        saved = self.assert_ok(await save.save_settings())
        rebuilt, _ = self.build(settings=SettingsRepository(config))
        loaded = self.assert_ok(await rebuilt.get_settings())
        self.assertEqual(loaded["settings"]["model_timeout_s"], 46.0)
        self.assertEqual(loaded["settings"]["generation_provider_id"], "generator-real")
        prompt, _ = self.build(settings=SettingsRepository(config), body={
            "expected_revision": saved["revision"], "kind": "routing", "content": "saved prompt"})
        self.assert_ok(await prompt.save_prompt())
        read_prompt, _ = self.build(settings=SettingsRepository(config))
        prompts = self.assert_ok(await read_prompt.get_prompts())
        self.assertEqual(prompts["prompts"]["routing"]["content"], "saved prompt")
        self.assertEqual(config.saved["model_timeout_s"], 46.0)



class ProviderEndpointTests(UpperApiTestCase):
    async def test_providers_are_listed_without_secrets(self) -> None:
        api, _ = self.build(
            settings=FakeSettingsRepository(
                {"values": {"routing_provider_id": "prov-a"}, "revision": 1}
            )
        )
        api.dependencies.providers = lambda: [
            {"id": "prov-a", "name": "主模型", "model": "gpt-x", "type": "openai"}
        ]
        body = self.assert_ok(await api.get_providers())
        self.assertEqual([row["id"] for row in body["providers"]], ["prov-a"])
        self.assertEqual(body["selected"]["routing_provider_id"], "prov-a")
        self.assertEqual(body["stale"], [])

    async def test_unknown_selected_provider_is_reported_as_stale(self) -> None:
        api, _ = self.build(
            settings=FakeSettingsRepository(
                {"values": {"generation_provider_id": "gone"}, "revision": 1}
            )
        )
        api.dependencies.providers = lambda: []
        body = self.assert_ok(await api.get_providers())
        self.assertEqual(body["stale"], ["gone"])
        self.assertFalse(body["installed"])

    async def test_provider_api_unavailable_is_explicit(self) -> None:
        api, _ = self.build()
        body = self.assert_ok(await api.get_providers())
        self.assertEqual(body["providers"], [])
        self.assertEqual(body["unavailable_reason"], "host_provider_api_unavailable")


# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------


def valid_record(**overrides: Any) -> dict[str, Any]:
    record = {
        "level": "primitive",
        "action_id": "ex.pick_object",
        "name": "抓取",
        "description": "按已配置方案抓取目标。",
        "parameters": [
            {"name": "target", "description": "目标引用", "unit": "", "required": True}
        ],
        "usage_stages": ["grasp"],
        "skill_markdown": "",
        "enabled": True,
    }
    record.update(overrides)
    return record


class ActionEndpointTests(UpperApiTestCase):
    async def test_get_actions_includes_catalog_version_and_ex_availability(self) -> None:
        api, _ = self.build(
            ex_availability=lambda: {
                "state": "offline",
                "catalog_version": None,
                "action_ids": [],
                "checked_at": "2026-11-02T06:00:00Z",
            }
        )
        body = self.assert_ok(await api.get_actions())
        self.assertEqual(body["catalog_version"], "cat-1")
        self.assertEqual(body["revision"], 1)
        self.assertEqual(body["ex_availability"]["state"], "offline")

    async def test_offline_ex_still_allows_local_editing(self) -> None:
        api, _ = self.build(
            body={"record": valid_record(), "expected_revision": 1},
            ex_availability=lambda: {"state": "offline", "action_ids": []},
        )
        body = self.assert_ok(await api.save_action())
        self.assertTrue(body["action_ref"].startswith("aeb-"))
        self.assertEqual(body["revision"], 2)

    async def test_create_then_read_back_from_rebuilt_catalog(self) -> None:
        backing = {"records": {}, "revision": 1, "version": "cat-1", "next": 1}
        api, _ = self.build(
            body={"record": valid_record(), "expected_revision": 1},
            catalog=FakeActionCatalog(backing),
        )
        created = self.assert_ok(await api.save_action())
        self.assertEqual(created["action_ref"], "aeb-001")

        rebuilt, _ = self.build(catalog=FakeActionCatalog(backing))
        listed = self.assert_ok(await rebuilt.get_actions())
        self.assertEqual(len(listed["actions"]), 1)
        self.assertEqual(listed["actions"][0]["action_id"], "ex.pick_object")
        self.assertEqual(listed["revision"], 2)

    async def test_skill_level_rejects_action_id(self) -> None:
        api, _ = self.build(
            body={
                "record": valid_record(level="skill", action_id="ex.pick_object"),
                "expected_revision": 1,
            }
        )
        self.assert_error(await api.save_action(), "invalid_request", 400)

    async def test_primitive_requires_action_id(self) -> None:
        api, _ = self.build(
            body={"record": valid_record(action_id=None), "expected_revision": 1}
        )
        self.assert_error(await api.save_action(), "invalid_request", 400)

    async def test_absolute_path_action_id_is_rejected(self) -> None:
        for value in ("C:\\tools\\run.exe", "/opt/run.sh", "..\\escape"):
            with self.subTest(value=value):
                api, _ = self.build(
                    body={"record": valid_record(action_id=value), "expected_revision": 1}
                )
                self.assert_error(await api.save_action(), "path_not_allowed", 400)

    async def test_file_path_field_is_rejected(self) -> None:
        api, _ = self.build(
            body={"record": valid_record(file_path="C:\\x.md"), "expected_revision": 1}
        )
        self.assert_error(await api.save_action(), "path_not_allowed", 400)

    async def test_machine_schema_and_exec_permission_are_rejected(self) -> None:
        for key in ("machine_schema", "exec_permission", "params", "shell"):
            with self.subTest(key=key):
                api, _ = self.build(
                    body={"record": valid_record(**{key: {"x": 1}}), "expected_revision": 1}
                )
                self.assert_error(await api.save_action(), "field_not_allowed", 400)

    async def test_parameter_execution_keys_are_rejected(self) -> None:
        api, _ = self.build(
            body={
                "record": valid_record(
                    parameters=[{"name": "t", "description": "d", "unit": "", "required": True, "params": {}}]
                ),
                "expected_revision": 1,
            }
        )
        self.assert_error(await api.save_action(), "field_not_allowed", 400)

    async def test_save_action_revision_conflict_is_409(self) -> None:
        api, _ = self.build(body={"record": valid_record(), "expected_revision": 9})
        self.assert_error(await api.save_action(), "revision_conflict", 409)

    async def test_delete_requires_known_reference_and_cas(self) -> None:
        backing = {"records": {}, "revision": 1, "version": "cat-1", "next": 1}
        api, _ = self.build(
            body={"record": valid_record(), "expected_revision": 1},
            catalog=FakeActionCatalog(backing),
        )
        created = self.assert_ok(await api.save_action())

        stale, _ = self.build(
            body={"action_ref": created["action_ref"], "expected_revision": 1},
            catalog=FakeActionCatalog(backing),
        )
        self.assert_error(await stale.delete_action(), "revision_conflict", 409)

        api2, _ = self.build(
            body={"action_ref": "aeb-999", "expected_revision": 2},
            catalog=FakeActionCatalog(backing),
        )
        self.assert_error(await api2.delete_action(), "not_found", 404)

        api3, _ = self.build(
            body={"action_ref": created["action_ref"], "expected_revision": 2},
            catalog=FakeActionCatalog(backing),
        )
        body = self.assert_ok(await api3.delete_action())
        self.assertTrue(body["deleted"])
        self.assertTrue(body["ex_action_untouched"])
        self.assertEqual(body["revision"], 3)

    async def test_markdown_is_stored_as_plain_data(self) -> None:
        markdown = "# 标题\n<script>alert(1)</script>\n| a | b |\n| - | - |"
        api, _ = self.build(
            body={
                "record": valid_record(level="skill", action_id=None, skill_markdown=markdown),
                "expected_revision": 1,
            }
        )
        body = self.assert_ok(await api.save_action())
        self.assertEqual(body["action"]["skill_markdown"], markdown)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


class PromptEndpointTests(UpperApiTestCase):
    async def test_prompts_report_builtin_source_and_hashes(self) -> None:
        api, _ = self.build(prompt_defaults=lambda: {"routing": "内置路由", "planning": "内置编排"})
        body = self.assert_ok(await api.get_prompts())
        routing = body["prompts"]["routing"]
        self.assertEqual(routing["source"], "builtin")
        self.assertEqual(routing["content"], "")
        self.assertEqual(routing["effective_content"], "内置路由")
        self.assertEqual(len(routing["hash"]), 64)
        # With no override the effective content *is* the builtin text.
        self.assertEqual(routing["hash"], routing["default_hash"])

    async def test_prompt_override_switches_source(self) -> None:
        api, _ = self.build(
            settings=FakeSettingsRepository(
                {"values": {"routing_prompt": "自定义路由"}, "revision": 3}
            ),
            prompt_defaults=lambda: {"routing": "内置路由", "planning": "内置编排"},
        )
        body = self.assert_ok(await api.get_prompts())
        self.assertEqual(body["prompts"]["routing"]["source"], "override")
        self.assertEqual(body["prompts"]["routing"]["effective_content"], "自定义路由")
        self.assertEqual(body["prompts"]["planning"]["source"], "builtin")

    async def test_save_prompt_only_writes_its_own_field(self) -> None:
        settings = FakeSettingsRepository(
            {"values": {"planning_prompt": "保留我", "upper_enabled": True}, "revision": 1}
        )
        api, _ = self.build(
            body={"kind": "routing", "content": "新路由", "expected_revision": 1},
            settings=settings,
        )
        body = self.assert_ok(await api.save_prompt())
        self.assertEqual(body["revision"], 2)
        self.assertEqual(body["source"], "override")
        self.assertEqual(settings.save_calls, [{"routing_prompt": "新路由"}])
        self.assertEqual(settings.backing["values"]["planning_prompt"], "保留我")
        self.assertTrue(settings.backing["values"]["upper_enabled"])

    async def test_save_prompt_rejects_other_kinds(self) -> None:
        for kind in ("system", "generation", ""):
            with self.subTest(kind=kind):
                api, _ = self.build(
                    body={"kind": kind, "content": "x", "expected_revision": 1}
                )
                self.assert_error(await api.save_prompt(), "invalid_request", 400)

    async def test_save_prompt_empty_restores_default(self) -> None:
        settings = FakeSettingsRepository(
            {"values": {"routing_prompt": "旧覆盖"}, "revision": 1}
        )
        api, _ = self.build(
            body={"kind": "routing", "content": "", "expected_revision": 1},
            settings=settings,
            prompt_defaults=lambda: {"routing": "内置路由"},
        )
        body = self.assert_ok(await api.save_prompt())
        self.assertEqual(body["source"], "builtin")
        self.assertEqual(settings.backing["values"]["routing_prompt"], "")

    async def test_save_prompt_conflict_is_409(self) -> None:
        api, _ = self.build(body={"kind": "planning", "content": "x", "expected_revision": 4})
        self.assert_error(await api.save_prompt(), "revision_conflict", 409)


# --------------------------------------------------------------------------
# plans
# --------------------------------------------------------------------------


class PlanEndpointTests(UpperApiTestCase):
    async def test_plans_are_paged_with_bounded_limit(self) -> None:
        plans = FakePlanRepository([plan_fixture(f"plan-{i}") for i in range(25)])
        api, _ = self.build(params={"limit": "10"}, plans=plans)
        body = self.assert_ok(await api.get_plans())
        self.assertEqual(body["count"], 10)
        self.assertEqual(body["limit"], 10)
        self.assertIsNotNone(body["next_cursor"])

        api2, _ = self.build(params={"limit": "10", "cursor": body["next_cursor"]}, plans=plans)
        page2 = self.assert_ok(await api2.get_plans())
        self.assertEqual(page2["plans"][0]["plan_id"], "plan-10")

    async def test_plan_limit_bounds(self) -> None:
        for limit in ("0", "101", "-3"):
            with self.subTest(limit=limit):
                api, _ = self.build(params={"limit": limit})
                self.assert_error(await api.get_plans(), "invalid_request", 400)

    async def test_plan_limit_default_and_non_numeric(self) -> None:
        api, _ = self.build()
        body = self.assert_ok(await api.get_plans())
        self.assertEqual(body["limit"], 20)

        bad, _ = self.build(params={"limit": "abc"})
        self.assert_error(await bad.get_plans(), "invalid_request", 400)

    async def test_invalid_cursor_is_rejected(self) -> None:
        for cursor in ("***", "!!!!", "x" * 80, "LTE"):
            with self.subTest(cursor=cursor):
                api, _ = self.build(params={"cursor": cursor})
                self.assert_error(await api.get_plans(), "cursor_invalid", 400)

    async def test_plan_detail_keeps_segment_boundaries_and_no_success_label(self) -> None:
        plans = FakePlanRepository([plan_fixture(segments=3, sent=1)])
        api, _ = self.build(params={"plan_id": "plan-1"}, plans=plans)
        body = self.assert_ok(await api.get_plan())
        plan = body["plan"]
        self.assertEqual(plan["segment_total"], 3)
        self.assertEqual(plan["sent_prefix"], 1)
        self.assertEqual(plan["pending_suffix"], 2)
        self.assertEqual([s["ordinal"] for s in plan["segments"]], [0, 1, 2])
        self.assertEqual([s["segment_id"] for s in plan["pending_segments"]], ["s1", "s2"])
        self.assertEqual(plan["delivery_phase"], "injected")
        self.assertIn(plan["delivery_phase"], ("unsent", "sending", "received", "injected", "rejected", "unknown"))
        text = repr(body).lower()
        for forbidden in ("执行完成", "completed", "success", "retry", "resume"):
            self.assertNotIn(forbidden, text)

    async def test_unknown_plan_is_404(self) -> None:
        api, _ = self.build(params={"plan_id": "missing"})
        self.assert_error(await api.get_plan(), "not_found", 404)

    async def test_missing_plan_id_is_400(self) -> None:
        api, _ = self.build()
        self.assert_error(await api.get_plan(), "invalid_request", 400)

    async def test_unknown_delivery_phase_maps_to_unknown(self) -> None:
        plan = plan_fixture()
        plan["delivery_phase"] = "delivery_unknown"
        api, _ = self.build(params={"plan_id": "plan-1"}, plans=FakePlanRepository([plan]))
        body = self.assert_ok(await api.get_plan())
        self.assertEqual(body["plan"]["delivery_phase"], "unknown")


# --------------------------------------------------------------------------
# registration, isolation, error hygiene
# --------------------------------------------------------------------------


class RegistrationTests(UpperApiTestCase):
    def test_registers_ten_prefixed_routes(self) -> None:
        context = FakeContext()
        registration = register_upper_api(context, UpperApiDependencies(), responder=RecordingResponder())
        self.assertEqual(len(context.registered_web_apis), 10)
        routes = {entry[0] for entry in context.registered_web_apis}
        for route, _methods in ROUTE_SPECS:
            self.assertIn(f"/{PLUGIN_NAME}/{route}", routes)
        self.assertEqual(len(registration.routes), 10)
        self.assertEqual(registration.plugin_name, PLUGIN_NAME)

    def test_reregistration_replaces_instead_of_stacking(self) -> None:
        context = FakeContext()
        register_upper_api(context, UpperApiDependencies(), responder=RecordingResponder())
        first = dict((entry[0], entry[1]) for entry in context.registered_web_apis)
        registration = register_upper_api(context, UpperApiDependencies(), responder=RecordingResponder())
        self.assertEqual(len(context.registered_web_apis), 10)
        second = dict((entry[0], entry[1]) for entry in context.registered_web_apis)
        # Every handler must be re-bound to the newest plugin instance.
        for route, handler in second.items():
            self.assertIsNot(handler, first[route])
            self.assertIs(getattr(handler, "__self__", None), registration.api)

    def test_methods_are_upper_case_and_stable(self) -> None:
        context = FakeContext()
        register_upper_api(context, UpperApiDependencies(), responder=RecordingResponder())
        for _route, _handler, methods, _desc in context.registered_web_apis:
            self.assertTrue(all(method == method.upper() for method in methods))


class SchemaTests(unittest.TestCase):
    """``_conf_schema.json`` is the single declared source for defaults/types."""

    def test_schema_matches_setting_fields(self) -> None:
        import json

        from astrbot_plugin_astrbotex_interaction import upper_api

        path = Path(upper_api.__file__).with_name("_conf_schema.json")
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(sorted(schema), sorted(upper_api.SETTING_FIELDS))
        for name, entry in schema.items():
            self.assertIn(entry.get("type"), ("bool", "string", "text", "int", "float"), name)
            self.assertIn("default", entry, name)
            self.assertTrue(entry.get("description"), name)

    def test_new_features_default_to_off(self) -> None:
        from astrbot_plugin_astrbotex_interaction.upper_api import schema_defaults

        defaults = schema_defaults()
        self.assertIs(defaults["upper_enabled"], False)
        self.assertIs(defaults["control_enabled"], False)
        self.assertEqual(defaults["control_bind_host"], "127.0.0.1")
        self.assertEqual(defaults["control_port"], 8769)
        self.assertEqual(defaults["routing_prompt"], "")
        self.assertEqual(defaults["planning_prompt"], "")

    def test_control_port_does_not_collide_with_existing_channels(self) -> None:
        from astrbot_plugin_astrbotex_interaction.upper_api import schema_defaults

        self.assertNotIn(schema_defaults()["control_port"], (8766, 8767, 8768))


class HostAdapterTests(unittest.TestCase):
    """The responder must call the Host helpers with their documented signature.

    ``astrbot.api.web`` is not importable in a Host-less unit process, so the
    helpers are replaced by recorders.  Real Host wiring is B11 evidence.
    """

    def test_responder_delegates_to_host_helpers(self) -> None:
        import astrbot_plugin_astrbotex_interaction.upper_api as module

        calls: list[tuple[Any, ...]] = []
        original_ok = module._host_json_response
        original_err = module._host_error_response
        module._host_json_response = lambda data=None, **kwargs: calls.append(
            ("ok", data, kwargs)
        ) or "OK"
        module._host_error_response = lambda message, **kwargs: calls.append(
            ("err", message, kwargs)
        ) or "ERR"
        try:
            responder = module.HostResponder()
            self.assertTrue(responder.available())
            self.assertEqual(responder.ok({"a": 1}), "OK")
            self.assertEqual(
                responder.error("m", status_code=409, data={"error_code": "x"}), "ERR"
            )
        finally:
            module._host_json_response = original_ok
            module._host_error_response = original_err

        self.assertEqual(calls[0], ("ok", {"a": 1}, {"status_code": 200}))
        self.assertEqual(
            calls[1], ("err", "m", {"status_code": 409, "data": {"error_code": "x"}})
        )


class IsolationTests(UpperApiTestCase):
    async def test_api_never_imports_pipeline_or_control_modules(self) -> None:
        watched = {
            f"astrbot_plugin_astrbotex_interaction.{module}"
            for module in ("upper_pipeline", "control_channel", "upper_models", "upper_prompts")
        }
        # Snapshot first: another suite in the same process may already have
        # imported these for its own reasons. What matters is that calling the
        # API handlers adds none of them.
        before = set(sys.modules)
        api, _ = self.build(
            body={"record": valid_record(), "expected_revision": 1},
            params={"limit": "5"},
        )
        await api.get_settings()
        await api.get_providers()
        await api.get_actions()
        await api.save_action()
        await api.get_prompts()
        await api.get_plans()
        imported = watched & (set(sys.modules) - before)
        self.assertEqual(
            imported, set(), "the API layer must not import the pipeline or control modules"
        )

    async def test_responses_carry_no_stack_trace(self) -> None:
        class BrokenCatalog:
            def snapshot(self):
                raise RuntimeError("boom at C:\\secret\\path.py line 1")

            def save(self, record, expected_revision):
                raise RuntimeError("boom")

            def delete(self, action_ref, expected_revision):
                raise RuntimeError("boom")

        responder = RecordingResponder()
        api = UpperApi(
            UpperApiDependencies(actions=BrokenCatalog()),
            responder=responder,
            request_proxy=FakeRequest(),
        )
        result = await api.get_actions()
        self.assert_error(result, "dependency_unavailable", 503)
        self.assertNotIn("boom", repr(result.body))
        self.assertNotIn("Traceback", repr(result.body))

    async def test_repository_conflict_code_maps_to_409(self) -> None:
        class ConflictingSettings(FakeSettingsRepository):
            def save(self, data, expected_revision):
                raise RepositoryError("revision_conflict")

        api, _ = self.build(
            body={"settings": {"upper_enabled": True}, "expected_revision": 1},
            settings=ConflictingSettings(),
        )
        self.assert_error(await api.save_settings(), "revision_conflict", 409)

    async def test_flat_settings_body_is_accepted(self) -> None:
        api, _ = self.build(body={"upper_enabled": True, "expected_revision": 1})
        body = self.assert_ok(await api.save_settings())
        self.assertEqual(body["changed"], ["upper_enabled"])


if __name__ == "__main__":
    unittest.main()
