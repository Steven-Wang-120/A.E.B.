"""B01 settings repository checks.

No AstrBot host is needed: the repository is duck-typed against a mutable
mapping that exposes ``save_config()``, which is exactly the surface B10 injects.

The properties under test are the ones that cause real damage if they break:
a stale Page must not overwrite a newer value, a failed Host write must stay
visible, and a save must never drop a field the caller did not edit.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


settings = load_module("aeb_upper_settings", PLUGIN / "upper_settings.py")


class FakeConfig(dict):
    """Stand-in for AstrBotConfig: a mapping plus a visible save hook."""

    def __init__(self, initial=None):
        super().__init__(initial or {})
        self.save_calls = 0
        self.fail_next = False

    def save_config(self, replace_config=None, *, indent=2):
        if self.fail_next:
            self.fail_next = False
            raise OSError("simulated disk full")
        if replace_config:
            self.update(replace_config)
        self.save_calls += 1


class DefaultTests(unittest.TestCase):
    def test_feature_ships_disabled_on_loopback(self):
        snapshot = settings.SettingsRepository(FakeConfig()).load()
        self.assertFalse(snapshot.settings["upper_enabled"])
        self.assertFalse(snapshot.settings["control_enabled"])
        self.assertEqual(snapshot.settings["control_bind_host"], "127.0.0.1")
        self.assertEqual(snapshot.settings["control_port"], 8769)
        self.assertEqual(snapshot.revision, 1)

    def test_defaults_match_the_documented_schema(self):
        self.assertEqual(set(settings.DEFAULT_SETTINGS), {
            "upper_enabled",
            "routing_provider_id",
            "generation_provider_id",
            "routing_prompt",
            "planning_prompt",
            "control_enabled",
            "control_bind_host",
            "control_port",
            "control_peer_id",
            "robot_id",
            "model_timeout_s",
        })

    def test_old_channel_keys_are_left_alone(self):
        config = FakeConfig({"text_port": 8766, "audio_port": 8767, "vision_port": 8768})
        repo = settings.SettingsRepository(config)
        repo.save({"upper_enabled": True}, repo.load().revision)
        self.assertEqual(config["text_port"], 8766)
        self.assertEqual(config["audio_port"], 8767)
        self.assertEqual(config["vision_port"], 8768)


class PartialSaveTests(unittest.TestCase):
    def test_a_partial_patch_preserves_untouched_fields(self):
        config = FakeConfig()
        repo = settings.SettingsRepository(config)
        first = repo.save(
            {"routing_provider_id": "prov-a", "planning_prompt": "keep me"},
            repo.load().revision,
        )
        second = repo.save({"control_port": 9100}, first.revision)
        self.assertEqual(second.settings["routing_provider_id"], "prov-a")
        self.assertEqual(second.settings["planning_prompt"], "keep me")
        self.assertEqual(second.settings["control_port"], 9100)

    def test_unknown_setting_is_rejected(self):
        repo = settings.SettingsRepository(FakeConfig())
        with self.assertRaises(settings.UpperSettingsError) as raised:
            repo.save({"not_a_setting": 1}, repo.load().revision)
        self.assertEqual(raised.exception.code, "invalid_settings")

    def test_invalid_values_are_rejected(self):
        cases = {
            "control_port": 70000,
            "upper_enabled": "yes",
            "model_timeout_s": 0.0,
            "control_bind_host": "",
            "routing_provider_id": "bad id with spaces",
            "robot_id": "../../etc",
        }
        for name, value in cases.items():
            with self.subTest(field=name):
                repo = settings.SettingsRepository(FakeConfig())
                with self.assertRaises(settings.UpperSettingsError):
                    repo.save({name: value}, repo.load().revision)

    def test_bool_is_not_a_port(self):
        repo = settings.SettingsRepository(FakeConfig())
        with self.assertRaises(settings.UpperSettingsError):
            repo.save({"control_port": True}, repo.load().revision)

    def test_oversized_prompt_uses_the_prompt_error_code(self):
        repo = settings.SettingsRepository(FakeConfig())
        with self.assertRaises(settings.UpperSettingsError) as raised:
            repo.save(
                {"routing_prompt": "x" * (settings.MAX_PROMPT_CHARS + 1)},
                repo.load().revision,
            )
        self.assertEqual(raised.exception.code, "prompt_too_large")

    def test_empty_prompt_is_legal_and_means_use_the_builtin(self):
        repo = settings.SettingsRepository(FakeConfig())
        snapshot = repo.save({"planning_prompt": ""}, repo.load().revision)
        self.assertEqual(snapshot.settings["planning_prompt"], "")

    def test_no_op_save_does_not_move_the_revision(self):
        repo = settings.SettingsRepository(FakeConfig())
        first = repo.save({"control_port": 9100}, repo.load().revision)
        second = repo.save({"control_port": 9100}, first.revision)
        self.assertEqual(second.revision, first.revision)


class RevisionTests(unittest.TestCase):
    def test_two_saves_from_the_same_revision_only_one_wins(self):
        repo = settings.SettingsRepository(FakeConfig())
        shared = repo.load().revision
        repo.save({"control_port": 9100}, shared)
        with self.assertRaises(settings.UpperSettingsError) as raised:
            repo.save({"control_port": 9200}, shared)
        self.assertEqual(raised.exception.code, "revision_conflict")

    def test_an_external_host_edit_invalidates_an_older_page_save(self):
        config = FakeConfig()
        repo = settings.SettingsRepository(config)
        page_view = repo.load()

        # The Host's own configuration page edits the same object.
        config["routing_prompt"] = "edited through the host page"
        repo.load()  # detects the drift and bumps the revision

        with self.assertRaises(settings.UpperSettingsError) as raised:
            repo.save({"control_port": 9100}, page_view.revision)
        self.assertEqual(raised.exception.code, "revision_conflict")
        # The external value survived: the stale Page did not overwrite it.
        self.assertEqual(config["routing_prompt"], "edited through the host page")

    def test_revision_survives_a_restart(self):
        config = FakeConfig()
        first = settings.SettingsRepository(config)
        saved = first.save({"control_port": 9100}, first.load().revision)

        restarted = settings.SettingsRepository(config)
        reloaded = restarted.load()
        self.assertEqual(reloaded.settings["control_port"], 9100)
        self.assertEqual(reloaded.revision, saved.revision)
        # A stale revision from before the restart is still refused.
        with self.assertRaises(settings.UpperSettingsError):
            restarted.save({"control_port": 9200}, saved.revision - 1)

    def test_a_broken_external_value_falls_back_visibly(self):
        config = FakeConfig({"control_port": "not-a-port"})
        snapshot = settings.SettingsRepository(config).load()
        self.assertEqual(snapshot.settings["control_port"], 8769)


class WriteFailureTests(unittest.TestCase):
    def test_a_failed_host_write_is_reported_and_rolled_back(self):
        config = FakeConfig()
        repo = settings.SettingsRepository(config)
        before = repo.load()
        config.fail_next = True
        with self.assertRaises(settings.UpperSettingsError) as raised:
            repo.save({"control_port": 9100}, before.revision)
        self.assertEqual(raised.exception.code, "settings_write_failed")
        # The old value is still there and still readable.
        self.assertEqual(config.get("control_port", 8769), 8769)
        self.assertEqual(repo.load().settings["control_port"], 8769)

    def test_no_second_settings_file_is_created(self):
        source = inspect.getsource(settings)
        # The repository must go through the Host's own persistence, not a
        # private file that the Host configuration page cannot see.
        self.assertNotIn("open(", source)
        self.assertNotIn("Path(", source)
        self.assertIn("save_config()", source)


class PromptSharingTests(unittest.TestCase):
    def test_prompts_and_settings_share_one_revision(self):
        repo = settings.SettingsRepository(FakeConfig())
        start = repo.load()
        prompts = repo.save_prompts(
            routing_prompt="route here", expected_revision=start.revision
        )
        self.assertEqual(prompts.settings["routing_prompt"], "route here")
        # A settings save holding the pre-prompt revision must be refused.
        with self.assertRaises(settings.UpperSettingsError) as raised:
            repo.save({"control_port": 9100}, start.revision)
        self.assertEqual(raised.exception.code, "revision_conflict")
        self.assertEqual(repo.load().settings["routing_prompt"], "route here")

    def test_saving_one_prompt_leaves_the_other_alone(self):
        repo = settings.SettingsRepository(FakeConfig())
        repo.save_prompts(
            routing_prompt="r1", planning_prompt="p1", expected_revision=repo.load().revision
        )
        snapshot = repo.save_prompts(
            routing_prompt="r2", expected_revision=repo.load().revision
        )
        self.assertEqual(snapshot.settings["routing_prompt"], "r2")
        self.assertEqual(snapshot.settings["planning_prompt"], "p1")

    def test_load_prompts_returns_the_override_and_revision(self):
        repo = settings.SettingsRepository(FakeConfig())
        repo.save_prompts(
            routing_prompt="r", planning_prompt="p", expected_revision=repo.load().revision
        )
        routing, planning, revision = repo.load_prompts()
        self.assertEqual((routing, planning), ("r", "p"))
        self.assertEqual(revision, repo.load().revision)


class ApplyScopeTests(unittest.TestCase):
    def test_connection_fields_require_a_reload(self):
        repo = settings.SettingsRepository(FakeConfig())
        for name, value in (
            ("control_port", 9100),
            ("control_bind_host", "0.0.0.0"),
            ("control_peer_id", "peer-2"),
            ("robot_id", "robot-2"),
            ("control_enabled", True),
        ):
            with self.subTest(field=name):
                snapshot = repo.save({name: value}, repo.load().revision)
                self.assertEqual(snapshot.apply_scope, "reload_required")

    def test_prompt_and_model_fields_apply_to_the_next_input(self):
        repo = settings.SettingsRepository(FakeConfig())
        for name, value in (
            ("routing_prompt", "x"),
            ("planning_prompt", "y"),
            ("routing_provider_id", "prov-b"),
            ("model_timeout_s", 12.0),
            ("upper_enabled", True),
        ):
            with self.subTest(field=name):
                snapshot = repo.save({name: value}, repo.load().revision)
                self.assertEqual(snapshot.apply_scope, "next_input")

    def test_apply_scope_never_reports_a_reload_for_a_prompt_only_change(self):
        repo = settings.SettingsRepository(FakeConfig())
        snapshot = repo.save_prompts(
            planning_prompt="new plan", expected_revision=repo.load().revision
        )
        self.assertEqual(snapshot.apply_scope, "next_input")


class IsolationTests(unittest.TestCase):
    def test_the_module_cannot_call_a_model_or_a_socket(self):
        source = inspect.getsource(settings)
        for forbidden in ("astrbot", "zmq", "requests", "aiohttp", "send_message", "llm_generate"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(f"import {forbidden}", source)
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
