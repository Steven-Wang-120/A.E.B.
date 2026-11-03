"""Regression coverage for truthful acceptance and portable interpreter argv."""
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_upper_control.py"
spec = importlib.util.spec_from_file_location("upper_verify_runner_regressions", SCRIPT)
verify = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = verify
spec.loader.exec_module(verify)


class AcceptanceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = verify.build_parser().parse_args([
            "--aeb-root", self.temp.name, "--ex-root", self.temp.name,
            "--evidence-dir", str(Path(self.temp.name) / "evidence"), "--scope", "all"])
        self.runner = verify.Runner(self.args)
        self.addCleanup(shutil.rmtree, self.runner.temp_root, ignore_errors=True)

    def complete_matrix(self):
        for name in ["unit_aeb", "unit_ex", "loopback_aeb", "loopback_ex",
                     "host", "page_spec", "legacy_regression"]:
            self.runner.stage(name).status = verify.PASS

    def test_page_failure_is_required_even_if_marked_non_gating(self):
        self.complete_matrix()
        self.runner.stages[-2].status = verify.FAIL
        self.runner.stages[-2].gating = False
        self.assertEqual(self.runner.verdict(), "partial")

    def test_missing_page_prevents_complete_acceptance(self):
        self.complete_matrix()
        self.runner.stages[-2].status = verify.NOT_RUN
        self.assertEqual(self.runner.verdict(), "partial")

    def test_all_scope_rejects_an_incomplete_matrix(self):
        self.runner.stage("unit_aeb").status = verify.PASS
        self.assertEqual(self.runner.verdict(), "partial")

    def test_complete_matrix_passes(self):
        self.complete_matrix()
        self.assertEqual(self.runner.verdict(), "wiring_pass")

    def test_real_host_url_is_required_before_browser_probe(self):
        with patch.dict(verify.os.environ, {"UPPER_HOST_URL": ""}):
            result = self.runner.run_page()
        self.assertEqual(result.status, verify.NOT_RUN)
        self.assertIn("real_host_url_missing", result.reason)

    def test_fake_page_cannot_replace_real_host_in_all_scope(self):
        self.args.ui_mode = "fake_bridge"
        result = self.runner.run_page()
        self.assertEqual(result.status, verify.NOT_RUN)
        self.assertIn("real_host_page_required", result.reason)

    def test_default_windows_executable_keeps_backslashes_and_spaces(self):
        expected = r"C:\Program Files\Python\python.exe"
        with patch.object(verify.sys, "executable", expected):
            self.assertEqual(self.runner.python_for(host=False), [expected])

    def test_explicit_interpreter_file_with_spaces_is_one_argument(self):
        path = Path(self.temp.name) / "python with spaces.exe"
        path.write_text("", encoding="utf-8")
        self.args.python = str(path)
        self.assertEqual(self.runner.python_for(host=False), [str(path)])

    def test_docker_keeps_source_read_only_and_data_outside_source(self):
        self.args.host_image = "local/offline-host"
        argv = self.runner.docker_argv(Path(self.temp.name),
            ["-B", "-m", "unittest", "discover", "-s",
             "astrbot_plugin_astrbotex_interaction/tests", "-p", "test_upper_api.py"])
        self.assertIn(f"{Path(self.temp.name)}:/aeb-src:ro", argv)
        self.assertEqual(argv[argv.index("-w") + 1], "/tmp")
        self.assertIn("/aeb-src/astrbot_plugin_astrbotex_interaction/tests", argv)
        self.assertEqual(argv[argv.index("--pull") + 1], "never")

    def test_failed_git_probe_is_unavailable(self):
        (Path(self.temp.name) / ".git").write_text("gitdir: missing", encoding="utf-8")
        failed = verify.subprocess.CompletedProcess([], 128, "", "not a git repository")
        with patch.object(verify.subprocess, "run", return_value=failed):
            state = verify.git_state(Path(self.temp.name))
        self.assertFalse(state["available"])
        self.assertIn("exit_128", state["error"])
        self.assertNotIn("head", state)

    def test_failure_and_error_counts_are_independent(self):
        self.assertEqual(verify.summarize("Ran 3 tests in 0.2s\nFAILED (failures=1)"),
                         {"tests": 3, "skipped": 0, "failures": 1, "errors": 0})
        self.assertEqual(verify.summarize("Ran 3 tests in 0.2s\nFAILED (errors=2)"),
                         {"tests": 3, "skipped": 0, "failures": 0, "errors": 2})
