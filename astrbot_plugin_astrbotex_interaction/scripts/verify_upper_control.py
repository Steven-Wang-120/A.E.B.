#!/usr/bin/env python3
"""B11 cross-repository acceptance runner for AstrBot/AEB upper control.

The runner is deliberately pessimistic: a stage that cannot be executed is
reported as NOT_RUN or BLOCKED and makes the process exit non-zero. Nothing here
downloads packages, pulls images, contacts a remote model or touches hardware.

Scopes
------
unit      Module/contract tests on both sides (AEB ``tests/test_upper_*``,
          ``test_control_channel.py``, ``test_action_catalog.py``; EX
          ``test_step_input*``).
loopback  B11's own end-to-end suites, which drive a real ROUTER/DEALER pair
          and a local backend request capture. A dependency-gated skip is
          converted to NOT_RUN, never to PASS.
host      Host Python integration and the mandatory real Host page spec.
page      Only the B08 page spec; useful where a browser exists but the Host
          image does not.
all       unit + loopback + host + page.

Usage
-----
    python -B scripts/verify_upper_control.py \
        --aeb-root D:/Code/A.E.B-work/upper-b10-b11 \
        --ex-root  D:/Code/AstrBotEX-work/upper-b10-b11 \
        --host-root D:/Code/AstrBot-latest \
        --evidence-dir ./port_artifacts/upper/B11/run-001 \
        --scope all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "upper_control.verify.v1"
PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
BLOCKED = "BLOCKED"

PLUGIN_DIR = "astrbot_plugin_astrbotex_interaction"
UPPER_SCHEMA = "aeb.upper_steps.v1"

#: New files the other tasks owe. A missing file is a dependency gap, never a pass.
#: Split by owner: the core pipeline (B00-B06) is what the upper-control turn
#: needs, and the page (B07/B08) is a separate surface. Conflating them would
#: report "nothing ran" while the whole turn is in fact verifiable.
AEB_REQUIRED_CORE = {
    "upper_contracts.py": "B00",
    "upper_settings.py": "B01",
    "action_catalog.py": "B01",
    "upper_prompts.py": "B02",
    "upper_models.py": "B03",
    "upper_store.py": "B04",
    "upper_pipeline.py": "B05",
    "control_channel.py": "B06",
    "output_router.py": "B06",
}

AEB_REQUIRED_PAGE = {
    "upper_api.py": "B07",
    "_conf_schema.json": "B07",
    "pages/upper-control/index.html": "B08",
}

EX_REQUIRED = {
    "astrbot_ex/core/step_input_contracts.py": "B00",
    "astrbot_ex/core/step_input.py": "B09",
    "astrbot_ex/core/decision/instruction_input.py": "B09",
}

AEB_UNIT_TESTS = [
    "test_upper_contracts",
    "test_upper_settings",
    "test_action_catalog",
    "test_upper_prompts",
    "test_upper_models",
    "test_upper_store",
    "test_upper_pipeline",
    "test_upper_output_isolation",
    "test_control_channel",
    "test_upper_composition",
    "test_upper_verify_runner",
]

#: Owned by B07; kept out of the core list so an absent management API does not
#: hide a green pipeline.
AEB_API_TESTS = [
    "test_upper_api",
    "test_upper_acceptance",
    "test_upper_crash_recovery",
]

EX_UNIT_TESTS = [
    "test_step_input_contracts",
    "test_step_input",
    "test_step_input_integration",
]

AEB_LEGACY_TESTS = [
    "test_plugin_import",
    "test_plugin_channels",
    "test_zmq_transport",
    "test_task_contracts",
    "test_task_store",
    "test_public_output_router",
    "test_output_isolation",
    "test_host_foundation",
    "test_host_chat_routing",
]

EX_LEGACY_TESTS = [
    "test_connection_manager",
    "test_jev_backend",
    "test_decision_service",
    "test_decision_backend_contract",
    "test_decision_transport",
    "test_decision_management_interleaving",
]

PAGE_SPEC = Path("tests") / "ui" / "upper_control.spec.mjs"

SUMMARY_RE = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)
SKIP_RE = re.compile(r"skipped=(\d+)")
FAIL_RE = re.compile(r"failures=(\d+)")
ERROR_RE = re.compile(r"errors=(\d+)")
#: ``node --test`` reports TAP, not unittest. Without these a spec that skipped
#: every case would look like "no tests collected".
TAP_TESTS_RE = re.compile(r"^# tests (\d+)$", re.MULTILINE)
TAP_FAIL_RE = re.compile(r"^# fail (\d+)$", re.MULTILINE)
TAP_SKIP_RE = re.compile(r"^# skipped (\d+)$", re.MULTILINE)
DEPENDENCY_SKIP = "upper_dependency_missing"


@dataclass
class CommandResult:
    name: str
    argv: list[str]
    cwd: str
    exit_code: int
    duration_s: float
    tests: int = 0
    skipped: int = 0
    failures: int = 0
    errors: int = 0
    log: str = ""
    status: str = PASS
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": self.argv,
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 3),
            "tests": self.tests,
            "skipped": self.skipped,
            "failures": self.failures,
            "errors": self.errors,
            "status": self.status,
            "reason": self.reason,
            "log": self.log,
        }


@dataclass
class Stage:
    name: str
    status: str = NOT_RUN
    reason: str = ""
    #: Kept in the evidence schema for older readers. Every recorded stage
    #: participates in the verdict, including the mandatory real Host page.
    gating: bool = True
    commands: list[CommandResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "status": self.status,
            "reason": self.reason,
            "gating": self.gating,
            "commands": [c.as_dict() for c in self.commands],
        }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    """HEAD, branch and a dirty fingerprint; never mutates the work tree."""
    state: dict[str, Any] = {"root": str(root), "available": False}
    if not (root / ".git").exists():
        return state
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        state["error"] = str(exc)
        return state
    failed = [(name, result) for name, result in
              (("head", head), ("branch", branch), ("status", status))
              if result.returncode != 0]
    if failed:
        state["error"] = "; ".join(
            f"{name}: exit_{result.returncode}: {result.stderr.strip()}"
            for name, result in failed)
        return state
    porcelain = status.stdout.strip()
    state.update(
        {
            "available": True,
            "head": head.stdout.strip(),
            "branch": branch.stdout.strip(),
            "dirty": bool(porcelain),
            "dirty_fingerprint": sha256_text(porcelain),
            "dirty_entries": len(porcelain.splitlines()),
        }
    )
    return state


def probe_modules(root: Path, expected: dict[str, str], prefix: str = "") -> dict[str, Any]:
    found: dict[str, str] = {}
    missing: dict[str, str] = {}
    for relative, owner in expected.items():
        target = root / prefix / relative
        if target.exists():
            found[relative] = owner
        else:
            missing[relative] = owner
    return {"found": found, "missing": missing}


def summarize(output: str) -> dict[str, int]:
    tests = 0
    match = SUMMARY_RE.search(output)
    if match:
        tests = int(match.group(1))
    if not match:
        tap = TAP_TESTS_RE.search(output)
        if tap:
            fail = TAP_FAIL_RE.search(output)
            skip = TAP_SKIP_RE.search(output)
            return {
                "tests": int(tap.group(1)),
                "skipped": int(skip.group(1)) if skip else 0,
                "failures": int(fail.group(1)) if fail else 0,
                "errors": 0,
            }
    skipped = 0
    skip_match = SKIP_RE.search(output)
    if skip_match:
        skipped = int(skip_match.group(1))
    failures = errors = 0
    fail_match = FAIL_RE.search(output)
    if fail_match:
        failures = int(fail_match.group(1))
    error_match = ERROR_RE.search(output)
    if error_match:
        errors = int(error_match.group(1))
    return {"tests": tests, "skipped": skipped, "failures": failures, "errors": errors}


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.aeb_root = Path(args.aeb_root).resolve()
        self.ex_root = Path(args.ex_root).resolve()
        self.host_root = Path(args.host_root).resolve() if args.host_root else None
        self.evidence = Path(args.evidence_dir).resolve()
        self.logs = self.evidence / "logs"
        self.started_at = time.time()
        self.commands: list[CommandResult] = []
        self.stages: list[Stage] = []
        self.temp_root = Path(tempfile.mkdtemp(prefix="upper-b11-"))
        self.notes: list[str] = []

    # -- plumbing ----------------------------------------------------------

    def prepare(self) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        (self.evidence / "commands.log").write_text("", encoding="utf-8")

    def python_for(self, *, host: bool) -> list[str]:
        if host and self.args.host_image:
            return []
        configured = self.args.host_python if host and self.args.host_python else self.args.python
        if not configured:
            return [sys.executable]
        # Executable paths are one argv item, including Windows backslashes and spaces.
        if Path(configured).is_file():
            return [configured]
        if os.name == "nt":
            return [part.strip('"') for part in shlex.split(configured, posix=False)]
        return shlex.split(configured)

    def docker_argv(self, root: Path, inner: list[str]) -> list[str]:
        """Run an interpreter command inside a local image; the image is never pulled."""
        # The Host creates data relative to cwd during import. Keep that data
        # inside the disposable container and mount candidate source read-only.
        inner = [f"/aeb-src/{arg}" if arg.startswith(f"{PLUGIN_DIR}/") else arg
                 for arg in inner]
        cross_test = next((name for name in AEB_API_TESTS[1:] if any(name in arg for arg in inner)), None)
        extra = []
        pythonpath = "/aeb-src"
        if cross_test:
            output = self.evidence / "system" / cross_test
            output.mkdir(parents=True, exist_ok=True)
            extra = ["-v", f"{self.ex_root}:/ex-src:ro",
                     "-v", f"{output}:/acceptance-output",
                     "-e", "UPPER_ACCEPTANCE_EVIDENCE=/acceptance-output"]
            pythonpath += ":/ex-src"
        return [
            "docker", "run", "--rm", "--pull", "never", "--network", "none",
            "-v", f"{root}:/aeb-src:ro", *extra,
            "-e", f"PYTHONPATH={pythonpath}", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "ASTRBOTEX_UPPER_DATA_ROOT=/tmp/upper-data",
            "-e", "ASTRBOTEX_TASK_DB_PATH=/tmp/upper-data/tasks.sqlite3",
            "-w", "/tmp", self.args.host_image, "python", *inner,
        ]

    def run(
        self,
        name: str,
        argv: list[str],
        cwd: Path,
        *,
        extra_env: dict[str, str] | None = None,
        timeout: float = 1800.0,
    ) -> CommandResult:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # Isolated data roots: never a production database or a real account.
        env["ASTRBOTEX_UPPER_DATA_ROOT"] = str(self.temp_root / "aeb-data")
        env["ASTRBOTEX_TASK_DB_PATH"] = str(self.temp_root / "aeb-data" / "tasks.sqlite3")
        if extra_env:
            env.update(extra_env)
        started = time.time()
        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            code = completed.returncode
        except FileNotFoundError as exc:
            output = f"command not found: {exc}"
            code = 127
        except subprocess.TimeoutExpired as exc:
            output = f"timeout after {timeout}s\n{exc}"
            code = 124
        duration = time.time() - started
        counts = summarize(output)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        log_name = f"{len(self.commands) + 1:03d}-{safe}.log"
        log_path = self.logs / log_name
        log_path.write_text(output, encoding="utf-8")
        result = CommandResult(
            name=name,
            argv=argv,
            cwd=str(cwd),
            exit_code=code,
            duration_s=duration,
            log=str(log_path.relative_to(self.evidence)),
            **counts,
        )
        self.classify(result, output)
        self.commands.append(result)
        with (self.evidence / "commands.log").open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{time.strftime('%H:%M:%S')}] {name} cwd={cwd} exit={code} "
                f"tests={result.tests} skipped={result.skipped} status={result.status}\n"
                f"    {' '.join(argv)}\n"
            )
        return result

    def classify(self, result: CommandResult, output: str) -> None:
        if result.exit_code == 127:
            result.status = BLOCKED
            result.reason = "command_not_found"
            return
        if result.exit_code == 124:
            result.status = FAIL
            result.reason = "timeout"
            return
        if result.exit_code != 0:
            result.status = FAIL
            result.reason = f"exit_{result.exit_code}"
            return
        if result.tests == 0:
            result.status = NOT_RUN
            result.reason = "no_tests_collected"
            return
        if result.skipped:
            # A dependency-gated skip is not evidence of success.
            reason = DEPENDENCY_SKIP if DEPENDENCY_SKIP in output else "skipped"
            result.status = NOT_RUN
            result.reason = f"{reason}:{result.skipped}"
            return
        result.status = PASS

    def stage(self, name: str, *, gating: bool = True) -> Stage:
        stage = Stage(name, gating=gating)
        self.stages.append(stage)
        return stage

    def close_stage(self, stage: Stage, *, required: bool, missing: dict[str, str] | None = None) -> None:
        if missing:
            stage.status = NOT_RUN
            owners = sorted(set(missing.values()))
            stage.reason = f"missing_dependencies:{','.join(owners)}"
            return
        if not stage.commands:
            stage.status = NOT_RUN
            stage.reason = "no_commands"
            return
        failed = [c for c in stage.commands if c.status == FAIL]
        if failed:
            stage.status = FAIL
            stage.reason = ",".join(c.name for c in failed)
            return
        not_run = [c for c in stage.commands if c.status in (NOT_RUN, BLOCKED)]
        if not_run:
            stage.status = NOT_RUN if required else BLOCKED
            stage.reason = ",".join(f"{c.name}:{c.reason}" for c in not_run)
            return
        stage.status = PASS

    # -- environment -------------------------------------------------------

    def environment(self) -> dict[str, Any]:
        def version_of(argv: list[str]) -> str:
            try:
                completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)
                return (completed.stdout or completed.stderr).strip().splitlines()[0]
            except (OSError, subprocess.SubprocessError, IndexError):
                return "unavailable"

        host_version = "not_importable"
        if self.host_root is not None or self.args.host_image:
            probe_argv = ["-B", "-c", "from astrbot.core.config.default import VERSION;print(VERSION)"]
            if self.args.host_image:
                probe_argv = self.docker_argv(self.aeb_root, probe_argv)
            else:
                probe_argv = [*self.python_for(host=True), *probe_argv]
            host_version = version_of(probe_argv)
        return {
            "schema": SCHEMA,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started_at)),
            "os": f"{platform.system()} {platform.release()} {platform.machine()}",
            "runner_python": version_of([*self.python_for(host=False), "-V"]),
            "host_python": (
                version_of([*self.python_for(host=True), "-V"])
                if self.args.host_python
                else None
            ),
            "host_image": self.args.host_image or None,
            "host_source": "local_image" if self.args.host_image else "host_python",

            "host_version": host_version,
            "scope": self.args.scope,
            "aeb": git_state(self.aeb_root),
            "ex": git_state(self.ex_root),
            "host": git_state(self.host_root) if self.host_root else None,
            "upper_schema": UPPER_SCHEMA,
            "isolation": {
                "temp_root": str(self.temp_root),
                "network": "none (no downloads, no image pulls, no remote model)",
            },
        }

    # -- stages ------------------------------------------------------------

    def run_unit(self) -> Stage:
        aeb_missing = probe_modules(self.aeb_root, AEB_REQUIRED_CORE, PLUGIN_DIR)["missing"]
        ex_missing = probe_modules(self.ex_root, EX_REQUIRED)["missing"]
        # One stage per repository: the AEB modules are all present, so its unit
        # result is real evidence even while the EX ingress is still missing.
        aeb_stage = self.stage("unit_aeb")
        tests_dir = self.aeb_root / PLUGIN_DIR / "tests"
        for name in AEB_UNIT_TESTS:
            if not (tests_dir / f"{name}.py").exists():
                result = CommandResult(
                    name=f"aeb:{name}",
                    argv=[],
                    cwd=str(self.aeb_root),
                    exit_code=0,
                    duration_s=0.0,
                    status=NOT_RUN,
                    reason="test_missing",
                )
                self.commands.append(result)
                aeb_stage.commands.append(result)
                continue
            argv = [
                *self.python_for(host=True),
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                f"{PLUGIN_DIR}/tests",
                "-p",
                f"{name}.py",
                "-v",
            ]
            if self.args.host_image:
                argv = self.docker_argv(self.aeb_root, argv[len(self.python_for(host=True)) :])
            aeb_stage.commands.append(self.run(f"aeb:{name}", argv, self.aeb_root))
        self.close_stage(aeb_stage, required=True, missing=aeb_missing)

        ex_stage = self.stage("unit_ex")
        for name in EX_UNIT_TESTS:
            if not (self.ex_root / "tests" / f"{name}.py").exists():
                result = CommandResult(
                    name=f"ex:{name}",
                    argv=[],
                    cwd=str(self.ex_root),
                    exit_code=0,
                    duration_s=0.0,
                    status=NOT_RUN,
                    reason="test_missing",
                )
                self.commands.append(result)
                ex_stage.commands.append(result)
                continue
            argv = [
                *self.python_for(host=False),
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-t",
                ".",
                "-p",
                f"{name}.py",
                "-v",
            ]
            ex_stage.commands.append(self.run(f"ex:{name}", argv, self.ex_root))
        self.close_stage(ex_stage, required=True, missing=ex_missing)
        return aeb_stage

    def run_loopback(self) -> Stage:
        aeb_missing = probe_modules(self.aeb_root, AEB_REQUIRED_CORE, PLUGIN_DIR)["missing"]
        ex_missing = probe_modules(self.ex_root, EX_REQUIRED)["missing"]
        if not (self.aeb_root / PLUGIN_DIR / "tests" / "test_upper_e2e.py").exists():
            aeb_missing["tests/test_upper_e2e.py"] = "B11"
        if not (self.ex_root / "tests" / "test_upper_control_e2e.py").exists():
            ex_missing["tests/test_upper_control_e2e.py"] = "B11"
        # Two stages, one per repository. A missing EX ingress module must not
        # hide the AEB-side result, and vice versa.
        aeb_stage = self.stage("loopback_aeb")
        if not aeb_missing:
            argv = [
                *self.python_for(host=True),
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                f"{PLUGIN_DIR}/tests",
                "-p",
                "test_upper_e2e.py",
                "-v",
            ]
            if self.args.host_image:
                argv = self.docker_argv(self.aeb_root, argv[len(self.python_for(host=True)) :])
            aeb_stage.commands.append(
                self.run(
                    "aeb:test_upper_e2e",
                    argv,
                    self.aeb_root,
                    extra_env={"ASTRBOTEX_UPPER_E2E": "1"},
                )
            )
        self.close_stage(aeb_stage, required=True, missing=aeb_missing)

        ex_stage = self.stage("loopback_ex")
        if not ex_missing:
            argv = [
                *self.python_for(host=False),
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-t",
                ".",
                "-p",
                "test_upper_control_e2e.py",
                "-v",
            ]
            ex_stage.commands.append(
                self.run(
                    "ex:test_upper_control_e2e",
                    argv,
                    self.ex_root,
                    extra_env={"ASTRBOTEX_UPPER_E2E": "1"},
                )
            )
        self.close_stage(ex_stage, required=True, missing=ex_missing)
        return aeb_stage

    def run_host(self) -> Stage:
        stage = self.stage("host")
        problems: dict[str, str] = {}
        if self.host_root is None or not self.host_root.exists():
            problems["host_root"] = "B10"
        if self.host_root is not None and not self.args.host_image and not self.args.host_python:
            problems["host_python"] = "environment"
        if self.args.host_image and shutil.which("docker") is None:
            problems["docker"] = "environment"
        page_missing = probe_modules(self.aeb_root, AEB_REQUIRED_PAGE, PLUGIN_DIR)["missing"]
        if page_missing:
            problems.update(page_missing)
        if not problems:
            argv = [
                *self.python_for(host=True),
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                f"{PLUGIN_DIR}/tests",
                "-p",
                "test_upper_composition.py",
                "-v",
            ]
            if self.args.host_image:
                argv = self.docker_argv(self.aeb_root, argv[len(self.python_for(host=True)) :])
            stage.commands.append(
                self.run("host:composition", argv, self.aeb_root, extra_env={"ASTRBOTEX_HOST_ROOT": str(self.host_root)})
            )
            for name in AEB_API_TESTS:
                if not (self.aeb_root / PLUGIN_DIR / "tests" / f"{name}.py").exists():
                    problems[f"tests/{name}.py"] = "B11"
                    continue
                api_argv = [
                    *self.python_for(host=True),
                    "-B",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    f"{PLUGIN_DIR}/tests",
                    "-p",
                    f"{name}.py",
                    "-v",
                ]
                if self.args.host_image:
                    api_argv = self.docker_argv(
                        self.aeb_root, api_argv[len(self.python_for(host=True)) :]
                    )
                extra_env = None
                if name in {"test_upper_acceptance", "test_upper_crash_recovery"} and not self.args.host_image:
                    output = self.evidence / "system" / name
                    output.mkdir(parents=True, exist_ok=True)
                    roots = [self.aeb_root, self.ex_root]
                    if self.host_root:
                        roots.append(self.host_root)
                    extra_env = {"PYTHONPATH": os.pathsep.join(map(str, roots)),
                                 "UPPER_ACCEPTANCE_EVIDENCE": str(output)}
                stage.commands.append(self.run(f"host:{name}", api_argv, self.aeb_root, extra_env=extra_env))
        self.close_stage(stage, required=True, missing=problems)
        return stage

    def run_page(self) -> Stage:
        """Real Host page verification is mandatory for the complete acceptance."""

        stage = self.stage("page_spec")
        ui_mode = self.args.ui_mode
        host_url = self.args.host_url or os.environ.get("UPPER_HOST_URL", "")
        if self.args.scope in {"all", "host"} and ui_mode != "real_host":
            stage.status = NOT_RUN
            stage.reason = "real_host_page_required: fake_bridge is only a page-scope unit check"
            return stage
        if ui_mode == "real_host" and not host_url:
            stage.status = NOT_RUN
            stage.reason = "environment:real_host_url_missing (set --host-url or UPPER_HOST_URL)"
            return stage
        self.notes.append(f"page_mode={ui_mode}")
        spec = self.aeb_root / PLUGIN_DIR / PAGE_SPEC
        if not spec.exists():
            stage.status = NOT_RUN
            stage.reason = "missing_dependencies:B08"
            return stage
        node = shutil.which(self.args.node)
        if node is None:
            stage.status = NOT_RUN
            stage.reason = f"environment:node_missing ({self.args.node})"
            return stage
        # The spec accepts any of these, plus PLAYWRIGHT_MODULE (which may be a
        # path). Probe the same list, with the same path normalisation, so the
        # runner never claims a browser it cannot actually reach.
        probe_script = (
            "const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');"
            "const c=[process.env.PLAYWRIGHT_MODULE,'playwright','playwright-core',"
            "'@playwright/test'].filter(Boolean);"
            "const n=(s)=>{if(s.startsWith('file://'))return s;"
            "if(!s.includes('/')&&!s.includes('\\\\'))return s;const r=path.resolve(s);"
            "try{if(fs.statSync(r).isDirectory())return pathToFileURL(path.join(r,'index.js')).href}"
            "catch{return pathToFileURL(r).href}return pathToFileURL(r).href};"
            "(async()=>{for(const s of c){try{const m=await import(n(s));"
            "const x=(m&&m.chromium)?m:m.default;if(x&&x.chromium){console.log('ok');return}}"
            "catch(e){}}process.exit(1)})()"
        )
        probe = subprocess.run(
            [node, "-e", probe_script],
            cwd=self.aeb_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if probe.returncode != 0:
            stage.status = NOT_RUN
            stage.reason = "environment:playwright_missing (set PLAYWRIGHT_MODULE to override)"
            return stage
        stage.commands.append(
            self.run(
                "page:upper_control",
                [node, "--test", "--test-reporter=tap", str(spec)],
                self.aeb_root,
                extra_env={"UPPER_UI_MODE": ui_mode, "UPPER_HOST_URL": host_url,
                           "UPPER_UI_EVIDENCE_DIR": str(self.evidence / "page")},
                timeout=900.0,
            )
        )
        self.close_stage(stage, required=True)
        # The spec records why it observed nothing; a bare "skipped:13" would
        # hide a version mismatch behind an unactionable number.
        marker = self.evidence / "page" / "upper_control.NOT_RUN.txt"
        command = stage.commands[-1]
        if command.status == NOT_RUN and marker.exists():
            try:
                lines = marker.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            for line in lines:
                if line.startswith("reason="):
                    detail = line[len("reason=") :].strip()
                    if detail:
                        command.reason = f"{command.reason}: {detail[:240]}"
                        stage.reason = f"{stage.reason} | {detail[:240]}"
                    break
        return stage

    def run_legacy(self) -> Stage:
        """Feature-off regression: record the result set, compare to a baseline."""
        stage = self.stage("legacy_regression")
        baseline = self.args.baseline
        recorded: dict[str, Any] = {}
        for name in AEB_LEGACY_TESTS:
            argv = [
                *self.python_for(host=True),
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                f"{PLUGIN_DIR}/tests",
                "-p",
                f"{name}.py",
            ]
            if self.args.host_image:
                argv = self.docker_argv(self.aeb_root, argv[len(self.python_for(host=True)) :])
            result = self.run(f"legacy:aeb:{name}", argv, self.aeb_root)
            stage.commands.append(result)
            recorded[f"aeb:{name}"] = result.as_dict()
        for name in EX_LEGACY_TESTS:
            argv = [
                *self.python_for(host=False),
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-t",
                ".",
                "-p",
                f"{name}.py",
            ]
            result = self.run(f"legacy:ex:{name}", argv, self.ex_root)
            stage.commands.append(result)
            recorded[f"ex:{name}"] = result.as_dict()
        new_failures: list[str] = []
        if baseline and Path(baseline).exists():
            previous = json.loads(Path(baseline).read_text(encoding="utf-8"))
            previous = previous.get("recorded", previous)
            for key, current in recorded.items():
                prior = previous.get(key)
                if prior and prior.get("status") == PASS and current["status"] != PASS:
                    new_failures.append(key)
            self.notes.append(f"baseline={baseline} new_failures={len(new_failures)}")
        (self.evidence / "legacy_regression.json").write_text(
            json.dumps({"recorded": recorded, "new_failures": new_failures}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.close_stage(stage, required=True)
        if new_failures:
            stage.status = FAIL
            stage.reason = "new_failures:" + ",".join(new_failures)
        return stage

    # -- outputs -----------------------------------------------------------

    def verdict(self) -> str:
        if self.args.scope == "all":
            required = {"unit_aeb", "unit_ex", "loopback_aeb", "loopback_ex",
                        "host", "page_spec", "legacy_regression"}
            if not required.issubset({stage.name for stage in self.stages}):
                return "partial"
        # A recorded failure or missing dependency cannot be exempted by a flag.
        return ("wiring_pass" if self.stages and all(stage.status == PASS for stage in self.stages)
                else "partial")

    def results(self) -> dict[str, Any]:
        finished = time.time()
        return {
            "schema": SCHEMA,
            "scope": self.args.scope,
            "verdict": self.verdict(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started_at)),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(finished)),
            "duration_s": round(finished - self.started_at, 3),
            "stages": [s.as_dict() for s in self.stages],
            "notes": self.notes,
            "claim_limit": (
                "PASS covers only the recorded scope. Complete acceptance requires the real "
                "Host page, and every mandatory stage must pass. Missing dependencies and "
                "fake-bridge-only page tests cannot prove real Host wiring. Nothing here proves "
                "execution completion, feedback, ROS motion or a live remote model."
            ),
        }

    def handoff(self) -> dict[str, Any]:
        return {
            "task_id": "B11",
            "code_revision": {
                "aeb": git_state(self.aeb_root),
                "ex": git_state(self.ex_root),
                "host": git_state(self.host_root) if self.host_root else None,
            },
            "changed_files": [
                f"{PLUGIN_DIR}/main.py",
                f"{PLUGIN_DIR}/task_models.py",
                f"{PLUGIN_DIR}/upper_pipeline.py",
                f"{PLUGIN_DIR}/upper_models.py",
                f"{PLUGIN_DIR}/upper_api.py",
                f"{PLUGIN_DIR}/tests/test_upper_api.py",
                f"{PLUGIN_DIR}/tests/test_upper_models.py",
                f"{PLUGIN_DIR}/tests/test_upper_verify_runner.py",
                f"{PLUGIN_DIR}/tests/ui/upper_control.spec.mjs",
                "astrbot_ex/core/step_input.py",
                "tests/test_step_input.py",
                f"{PLUGIN_DIR}/tests/test_upper_composition.py",
                f"{PLUGIN_DIR}/tests/test_upper_e2e.py",
                f"{PLUGIN_DIR}/scripts/verify_upper_control.py",
                f"{PLUGIN_DIR}/docs/upper-control-deploy.md",
                "tests/test_upper_control_e2e.py",
            ],
            "interfaces": [
                "verify_upper_control.py --scope unit|loopback|host|page|all",
                "--ui-mode real_host --host-url <isolated Dashboard URL> (mandatory for all/host)",
                "test_upper_e2e.py (AEB, real pipeline + real ROUTER/DEALER peer)",
                "test_upper_control_e2e.py (EX, ingress + backend request capture)",
                "UpperSettings.control_route_ref (ASTRBOTEX_UPPER_CONTROL_ROUTE_REF)",
            ],
            "cross_task_fixes": [
                {
                    "owner": "B04",
                    "file": "upper_store.py",
                    "defect": "json_value/canonical only accepted dict/list/scalars, so the "
                              "SettingsSnapshot/CatalogSnapshot/PromptSnapshot dataclasses B05 "
                              "passes were rejected as invalid_type and no plan could be saved.",
                    "fix": "_snapshot_value() converts upper snapshots before canonical serialization; "
                           "legacy task_models.py is identical to the source baseline.",
                },
                {
                    "owner": "B05",
                    "file": "upper_pipeline.py",
                    "defect": "save_result read settings_hash/prompt_hash off B02's PromptSnapshot, "
                              "which exposes settings_version/content_hash; the frozen batch then "
                              "carried an empty prompt_version and every injection was invalid_batch.",
                    "fix": "Read the field names B02 actually publishes.",
                },
                {
                    "owner": "B03",
                    "file": "tests/test_upper_models.py",
                    "defect": "Two tests indexed the tool set as dicts; the Host expects "
                              "FunctionTool objects, so the suite failed on the merged branch.",
                    "fix": "Read tool.name instead.",
                },
                {
                    "owner": "B05",
                    "file": "tests/test_upper_pipeline.py",
                    "defect": "The fake prompt snapshot invented settings_hash/prompt_hash, hiding "
                              "the field-name mismatch above.",
                    "fix": "Fake now uses the published field names.",
                },
            ],
            "repair_20261103": [
                "Management APIs stay available with both feature flags off.",
                "The API reads actual SettingsSnapshot.settings and CatalogSnapshot.entries.",
                "Catalog creates stable local IDs and preserves IDs/revisions across edits.",
                "Provider IDs, timeout and origin are frozen from each input snapshot.",
                "Voice public output retains the text peer; control uses its configured peer.",
                "EX ingress reuses the shared strict B00 parser.",
                "The real Dashboard iframe/bridge is tested separately from fake browser UI.",
                "Missing/failed mandatory stages exit non-zero; Windows argv stays intact.",
            ],
            "dependencies": {
                "aeb_core": sorted(set(AEB_REQUIRED_CORE.values())),
                "aeb_page": sorted(set(AEB_REQUIRED_PAGE.values())),
                "ex": sorted(set(EX_REQUIRED.values())),
                "host": "local AstrBot image or importable checkout",
            },
            "entry_points": {
                "cli": "python -B scripts/verify_upper_control.py --help",
                "aeb_tests": f"{PLUGIN_DIR}/tests/test_upper_e2e.py",
                "ex_tests": "tests/test_upper_control_e2e.py",
            },
            "tests": [s.name for s in self.stages],
            "evidence": str(self.evidence),
            "scope": "unit_fake|real_zmq_loopback|host_python_integration|backend_request_capture",
            "page_mode": self.args.ui_mode,
            "page_status": next((s.status for s in self.stages if s.name == "page_spec"), NOT_RUN),
            "known_issues": [
                "A missing or failed mandatory stage makes the verdict partial and exits non-zero.",
                "Host Python integration does not replace a real Dashboard page check.",
                "Forced process recovery and production-adapter output checks are mandatory Host tests.",
                "Host scope needs an importable Host checkout or a local image; no image is pulled.",
                "Upper control requires control_enabled: without the control channel a turn could "
                "classify an input and have nowhere to deliver, so takeover stays off.",
                "EX tests/test_action_runtime_integration.py::test_controller_start_stop_race "
                "is timing sensitive and fails intermittently under full-suite load; it passes "
                "in isolation and is unrelated to upper control.",
            ],
            "consumers_to_retest": ["B04", "B05", "B07", "B08", "B09"],
            "rollback": "Close upper_enabled/control_enabled; keep new tables, logs and pending rows.",
        }

    def report(self, results: dict[str, Any]) -> str:
        lines = [
            "# B11 upper control acceptance report",
            "",
            f"- scope: `{self.args.scope}`",
            f"- verdict: **{results['verdict']}**",
            f"- started: {results['started_at']}",
            f"- finished: {results['finished_at']}",
            f"- duration: {results['duration_s']}s",
            "",
            "## Stage matrix",
            "",
            "| Stage | Status | Gating | Reason | Commands |",
            "|---|---|---|---|---|",
        ]
        for stage in self.stages:
            reason = stage.reason or "-"
            gating = "yes" if stage.gating else "no"
            lines.append(
                f"| {stage.name} | {stage.status} | {gating} | {reason} | {len(stage.commands)} |"
            )
        lines += ["", "## Commands", "", "| # | Command | CWD | Exit | Tests | Skipped | Status | Log |", "|---|---|---|---|---|---|---|---|"]
        for index, command in enumerate(self.commands, start=1):
            lines.append(
                f"| {index} | `{command.name}` | `{command.cwd}` | {command.exit_code} | "
                f"{command.tests} | {command.skipped} | {command.status} | `{command.log}` |"
            )
        lines += [
            "",
            "## Scope and limits",
            "",
            "- `unit_fake`: module/substitute level, no real Host or socket.",
            "- `real_zmq_loopback`: real ROUTER/DEALER on temporary ports.",
            "- `backend_request_capture`: the actual backend request body, captured locally.",
            "- `host`: Python composition/API tests using Host libraries; these do not start a Dashboard.",
            "- `page_spec`: mandatory browser check; all/host scopes require a real Dashboard iframe and bridge.",
            "- A stage is PASS only when every command is PASS and no dependency was skipped.",
            "- No stage here proves execution completion, uplink feedback, ROS motion or a live model.",
            "",
        ]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {note}" for note in self.notes] + [""]
        return "\n".join(lines)

    def finish(self) -> int:
        results = self.results()
        (self.evidence / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.evidence / "handoff.json").write_text(
            json.dumps(self.handoff(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.evidence / "report.md").write_text(self.report(results), encoding="utf-8")
        if not self.args.keep_temp:
            shutil.rmtree(self.temp_root, ignore_errors=True)
        return 0 if results["verdict"] == "wiring_pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-repository upper-control acceptance runner (B11).",
    )
    parser.add_argument("--aeb-root", required=True, help="A.E.B checkout that contains the plugin")
    parser.add_argument("--ex-root", required=True, help="AstrBotEX checkout")
    parser.add_argument("--host-root", default=None, help="AstrBot Host checkout (read-only dependency)")
    parser.add_argument("--evidence-dir", required=True, help="New directory for this run's evidence")
    parser.add_argument(
        "--scope",
        default="all",
        choices=["unit", "loopback", "host", "page", "all"],
        help="Which stages to execute",
    )
    parser.add_argument("--python", default=None, help="Interpreter for EX-side tests")
    parser.add_argument("--host-python", default=None, help="Interpreter that can import astrbot")
    parser.add_argument("--host-image", default=None, help="Local Host image; never pulled automatically")
    parser.add_argument("--node", default="node", help="Node binary for the B08 page spec")
    parser.add_argument("--ui-mode", choices=["real_host", "fake_bridge"], default="real_host",
                        help="all/host require real_host; fake_bridge is only for --scope page")
    parser.add_argument("--host-url", default=None, help="Isolated running Host Dashboard URL")
    parser.add_argument("--baseline", default=None, help="Previous legacy_regression.json to compare")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the isolated temp data root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = Runner(args)
    if runner.evidence.exists() and any(runner.evidence.iterdir()):
        print(f"evidence directory already has content: {runner.evidence}", file=sys.stderr)
        print("use a new --evidence-dir; failed runs are never overwritten", file=sys.stderr)
        return 2
    runner.prepare()
    (runner.evidence / "environment.json").write_text(
        json.dumps(runner.environment(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scope = args.scope
    if scope in ("unit", "all"):
        runner.run_unit()
    if scope in ("loopback", "all"):
        runner.run_loopback()
    if scope in ("host", "all"):
        runner.run_host()
    if scope in ("host", "page", "all"):
        runner.run_page()
    if scope == "all":
        runner.run_legacy()
    return runner.finish()


if __name__ == "__main__":
    raise SystemExit(main())
