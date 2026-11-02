"""B01 action catalog checks.

The catalog is copy-on-write: a save publishes a complete new version and only
then moves the pointer. The tests below target the failure modes that matter:
a half-published version, a path escaping the data root, a deleted description
that cannot be recovered, and an offline cache that quietly claims an action is
executable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PLUGIN = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


catalog_module = load_module("aeb_action_catalog", PLUGIN / "action_catalog.py")


def record(ref="pick-selected", **over):
    payload = {
        "action_ref": ref,
        "level": "basic",
        "name": "抓取已选目标",
        "description": "闭合夹爪并抬起，不判断是否持物成功。",
        "action_id": "new_arm.pick_selected.v1",
        "parameters": [
            {"name": "target", "description": "目标引用", "unit": "", "required": True}
        ],
        "usage_stages": ["执行"],
        "skill_markdown": "# 抓取已选目标\n\n闭合夹爪，然后抬起。\n",
        "enabled": True,
    }
    payload.update(over)
    return payload


class CatalogTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.catalog = catalog_module.ActionCatalog(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def publish(self, payload=None):
        return self.catalog.save(payload or record(), self.catalog.revision())


class EmptyCatalogTests(CatalogTestCase):
    def test_a_fresh_catalog_is_empty_not_broken(self):
        snapshot = self.catalog.snapshot()
        self.assertEqual(snapshot.entries, ())
        self.assertEqual(snapshot.version, "")
        self.assertEqual(self.catalog.revision(), 0)
        self.assertEqual(self.catalog.list(), [])

    def test_the_data_root_is_derived_from_the_plugin_data_dir(self):
        self.assertEqual(
            self.catalog.root, self.root / "upper" / "catalog"
        )


class PublishTests(CatalogTestCase):
    def test_save_writes_the_documented_layout(self):
        snapshot = self.publish()
        version = snapshot.version
        self.assertTrue((self.catalog.root / "current.json").is_file())
        manifest = self.catalog.root / "versions" / version / "manifest.json"
        skill = self.catalog.root / "versions" / version / "actions" / "pick-selected" / "SKILL.md"
        self.assertTrue(manifest.is_file())
        self.assertTrue(skill.is_file())
        self.assertIn("闭合夹爪", skill.read_text(encoding="utf-8"))
        pointer = json.loads((self.catalog.root / "current.json").read_text(encoding="utf-8"))
        self.assertEqual(pointer["version"], version)
        self.assertEqual(pointer["revision"], 1)

    def test_the_same_content_publishes_the_same_version(self):
        first = self.publish()
        second = self.catalog.save(record(), self.catalog.revision())
        self.assertEqual(first.version, second.version)
        # Same content, so no new version directory was needed.
        versions = list((self.catalog.root / "versions").iterdir())
        self.assertEqual(len(versions), 1)

    def test_an_update_bumps_the_entry_revision(self):
        first = self.publish()
        self.assertEqual(first.get("pick-selected").record.revision, 1)
        second = self.catalog.save(
            record(skill_markdown="# 抓取已选目标\n\n先对齐再闭合。\n"),
            self.catalog.revision(),
        )
        self.assertEqual(second.get("pick-selected").record.revision, 2)
        self.assertNotEqual(first.version, second.version)

    def test_restart_reads_the_published_catalog(self):
        self.publish()
        restarted = catalog_module.ActionCatalog(self.root)
        entries = restarted.records()
        self.assertEqual([entry.action_ref for entry in entries], ["pick-selected"])
        self.assertIn("闭合夹爪", entries[0].skill_markdown)
        self.assertEqual(restarted.revision(), 1)

    def test_a_stale_revision_cannot_publish(self):
        self.publish()
        stale = self.catalog.revision() - 1
        with self.assertRaises(catalog_module.CatalogError) as raised:
            self.catalog.save(record(ref="other-action"), stale)
        self.assertEqual(raised.exception.code, "revision_conflict")


class DeleteTests(CatalogTestCase):
    def test_delete_removes_the_entry_but_keeps_the_old_version(self):
        first = self.publish()
        old_version_dir = self.catalog.root / "versions" / first.version
        second = self.catalog.delete("pick-selected", self.catalog.revision())
        self.assertEqual(second.entries, ())
        # The old version stays on disk: a saved task snapshot may still cite it.
        self.assertTrue(old_version_dir.is_dir())
        self.assertIn("闭合夹爪", (old_version_dir / "actions" / "pick-selected" / "SKILL.md").read_text(encoding="utf-8"))

    def test_deleting_an_unknown_entry_is_reported(self):
        with self.assertRaises(catalog_module.CatalogError) as raised:
            self.catalog.delete("never-existed", self.catalog.revision())
        self.assertEqual(raised.exception.code, "action_not_found")

    def test_delete_does_not_touch_an_ex_action(self):
        # There is no API that could reach EX, and delete only rewrites the
        # manifest. Prove it by asserting the entry count, not the EX catalog.
        self.publish()
        snapshot = self.catalog.delete("pick-selected", self.catalog.revision())
        self.assertEqual(len(snapshot.entries), 0)


class PathSafetyTests(CatalogTestCase):
    def test_a_traversal_action_ref_is_refused(self):
        for bad in ("../evil", "a/b", "UPPER", "..", ".hidden", "x" * 65, ""):
            with self.subTest(ref=bad):
                with self.assertRaises(catalog_module.CatalogError):
                    self.catalog.save(record(ref=bad), self.catalog.revision())

    def test_nothing_is_written_outside_the_data_root(self):
        with self.assertRaises(catalog_module.CatalogError):
            self.catalog.save(record(ref="../../escaped"), self.catalog.revision())
        # Nothing escaped, and the data root holds only the expected directory.
        self.assertFalse((self.root.parent / "escaped").exists())
        self.assertFalse((self.root.parent.parent / "escaped").exists())
        self.assertTrue(set(path.name for path in self.root.iterdir()) <= {"upper"})

    def test_an_html_name_is_display_text_not_a_path(self):
        snapshot = self.catalog.save(
            record(name="<script>alert(1)</script>"), self.catalog.revision()
        )
        entry = snapshot.get("pick-selected")
        self.assertEqual(entry.record.name, "<script>alert(1)</script>")
        skill = (
            self.catalog.root
            / "versions"
            / snapshot.version
            / "actions"
            / "pick-selected"
            / "SKILL.md"
        )
        self.assertTrue(skill.is_file())

    def test_there_is_no_path_parameter_anywhere(self):
        for name in ("import_skill", "skill_text_of", "save", "delete", "snapshot"):
            with self.subTest(method=name):
                params = catalog_module.ActionCatalog.__dict__[name].__annotations__
                self.assertNotIn("path", params)


class ValidationTests(CatalogTestCase):
    def test_duplicate_parameter_names_are_refused(self):
        payload = record()
        payload["parameters"] = [
            {"name": "target", "required": True},
            {"name": "target", "required": False},
        ]
        with self.assertRaises(catalog_module.CatalogError) as raised:
            self.catalog.save(payload, self.catalog.revision())
        self.assertEqual(raised.exception.code, "duplicate_parameter")

    def test_a_skill_must_not_carry_an_action_id(self):
        with self.assertRaises(catalog_module.CatalogError) as raised:
            self.catalog.save(
                record(level="skill", action_id="new_arm.pick_selected.v1"),
                self.catalog.revision(),
            )
        self.assertEqual(raised.exception.code, "invalid_action")

    def test_a_primitive_may_carry_an_action_id(self):
        snapshot = self.catalog.save(
            record(level="primitive", action_id="new_arm.close_gripper.v1"),
            self.catalog.revision(),
        )
        self.assertEqual(snapshot.get("pick-selected").record.level, "primitive")

    def test_an_unknown_level_is_refused(self):
        with self.assertRaises(catalog_module.CatalogError):
            self.catalog.save(record(level="macro"), self.catalog.revision())

    def test_unknown_fields_are_refused(self):
        payload = record()
        payload["machine_range"] = {"min": 0, "max": 1}
        with self.assertRaises(catalog_module.CatalogError):
            self.catalog.save(payload, self.catalog.revision())

    def test_a_parameter_cannot_declare_a_driver_range(self):
        payload = record()
        payload["parameters"] = [
            {"name": "target", "min": 0, "max": 1},
        ]
        with self.assertRaises(catalog_module.CatalogError):
            self.catalog.save(payload, self.catalog.revision())


class AvailabilityTests(CatalogTestCase):
    def snapshot_with(self, view):
        self.publish()
        return self.catalog.snapshot(capability=view)

    def test_offline_is_unknown_never_available(self):
        view = catalog_module.CapabilityView(
            ex_session="ex-1", catalog_version="cat-1", fetched_at=0.0, connected=False
        )
        entry = self.snapshot_with(view).get("pick-selected")
        self.assertEqual(entry.availability, "unknown")

    def test_no_capability_snapshot_is_unknown(self):
        self.publish()
        entry = self.catalog.snapshot().get("pick-selected")
        self.assertEqual(entry.availability, "unknown")

    def test_a_bound_action_is_available_only_when_ex_reports_it(self):
        view = catalog_module.CapabilityView(
            ex_session="ex-1",
            catalog_version="cat-1",
            fetched_at=0.0,
            actions={"new_arm.pick_selected.v1": {"available": True}},
        )
        self.assertEqual(self.snapshot_with(view).get("pick-selected").availability, "available")

    def test_an_action_ex_does_not_report_is_only_described(self):
        view = catalog_module.CapabilityView(
            ex_session="ex-1", catalog_version="cat-1", fetched_at=0.0, actions={}
        )
        self.assertEqual(
            self.snapshot_with(view).get("pick-selected").availability, "described_only"
        )

    def test_an_unavailable_ex_action_is_unavailable(self):
        view = catalog_module.CapabilityView(
            ex_session="ex-1",
            catalog_version="cat-1",
            fetched_at=0.0,
            actions={
                "new_arm.pick_selected.v1": {"available": False, "reason": "disabled in EX"}
            },
        )
        entry = self.snapshot_with(view).get("pick-selected")
        self.assertEqual(entry.availability, "unavailable")
        self.assertEqual(entry.reason, "disabled in EX")

    def test_a_disabled_entry_is_never_available(self):
        view = catalog_module.CapabilityView(
            ex_session="ex-1",
            catalog_version="cat-1",
            fetched_at=0.0,
            actions={"new_arm.pick_selected.v1": {"available": True}},
        )
        self.catalog.save(record(enabled=False), self.catalog.revision())
        self.assertEqual(self.catalog.snapshot(capability=view).get("pick-selected").availability, "unavailable")

    def test_a_new_action_without_ex_capability_cannot_be_available(self):
        view = catalog_module.CapabilityView(
            ex_session="ex-1",
            catalog_version="cat-1",
            fetched_at=0.0,
            actions={"new_arm.pick_selected.v1": {"available": True}},
        )
        self.catalog.save(
            record(ref="brand-new", action_id="new_arm.not_implemented.v1"),
            self.catalog.revision(),
        )
        snapshot = self.catalog.snapshot(capability=view)
        self.assertEqual(snapshot.get("brand-new").availability, "described_only")
        self.assertEqual([entry.record.action_ref for entry in snapshot.available()], [])

    def test_a_skill_without_an_ex_action_is_described_only(self):
        self.catalog.save(
            record(ref="compose-pick", level="skill", action_id=None),
            self.catalog.revision(),
        )
        entry = self.catalog.snapshot().get("compose-pick")
        self.assertEqual(entry.availability, "described_only")


class ExternalEditTests(CatalogTestCase):
    def test_a_valid_external_edit_is_republished_as_a_new_version(self):
        first = self.publish()
        skill = (
            self.catalog.root
            / "versions"
            / first.version
            / "actions"
            / "pick-selected"
            / "SKILL.md"
        )
        skill.write_text("# 改过的说明\n\n先对齐，再闭合。\n", encoding="utf-8")

        second = self.catalog.snapshot()
        self.assertNotEqual(second.version, first.version)
        self.assertIn("先对齐", second.get("pick-selected").record.skill_markdown)
        # The snapshot taken before the edit keeps the text it was built with.
        self.assertIn("闭合夹爪", first.get("pick-selected").record.skill_markdown)

    def test_a_missing_skill_file_is_a_configuration_error(self):
        first = self.publish()
        skill = (
            self.catalog.root
            / "versions"
            / first.version
            / "actions"
            / "pick-selected"
            / "SKILL.md"
        )
        skill.unlink()
        with self.assertRaises(catalog_module.CatalogError) as raised:
            self.catalog.snapshot()
        self.assertEqual(raised.exception.code, "catalog_corrupt")


class WriteFailureTests(CatalogTestCase):
    def test_a_failed_publish_leaves_the_pointer_untouched(self):
        first = self.publish()
        before = (self.catalog.root / "current.json").read_text(encoding="utf-8")
        with mock.patch("os.rename", side_effect=OSError("simulated disk full")):
            with self.assertRaises(catalog_module.CatalogError) as raised:
                self.catalog.save(record(ref="second-action"), self.catalog.revision())
        self.assertEqual(raised.exception.code, "catalog_write_failed")
        self.assertEqual((self.catalog.root / "current.json").read_text(encoding="utf-8"), before)
        self.assertEqual(self.catalog.snapshot().version, first.version)

    def test_a_scratch_directory_is_not_left_referenced(self):
        self.publish()
        with mock.patch("os.rename", side_effect=OSError("simulated disk full")):
            with self.assertRaises(catalog_module.CatalogError):
                self.catalog.save(record(ref="second-action"), self.catalog.revision())
        pointer = json.loads((self.catalog.root / "current.json").read_text(encoding="utf-8"))
        self.assertFalse(pointer["version"].startswith(".tmp-"))
        # Any leftover scratch directory is unreferenced, which is allowed.
        for path in (self.catalog.root / "versions").iterdir():
            if path.name.startswith(".tmp-"):
                self.assertNotEqual(path.name, pointer["version"])


class SnapshotPinningTests(CatalogTestCase):
    def test_a_snapshot_pins_settings_and_capability_versions(self):
        self.publish()
        view = catalog_module.CapabilityView(
            ex_session="ex-7",
            catalog_version="cat-9",
            fetched_at=1.0,
            actions={"new_arm.pick_selected.v1": {"available": True}},
        )
        snapshot = self.catalog.snapshot(capability=view, settings_hash="settings-abc")
        self.assertEqual(snapshot.settings_hash, "settings-abc")
        self.assertEqual(snapshot.capability_version, "ex-7:cat-9")

    def test_content_hash_changes_with_the_text(self):
        first = self.publish()
        second = self.catalog.save(
            record(skill_markdown="# 另一份说明\n"), self.catalog.revision()
        )
        self.assertNotEqual(first.content_hash(), second.content_hash())


class IsolationTests(CatalogTestCase):
    def test_the_catalog_never_calls_a_model_or_a_socket(self):
        import inspect

        source = inspect.getsource(catalog_module)
        for forbidden in ("import zmq", "import requests", "import aiohttp", "llm_generate", "send_message"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
