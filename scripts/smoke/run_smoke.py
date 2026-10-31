"""Run the repository's reproducibility smoke path.

The default mode runs the full unit-test suite, classifies any failures against
``known_failures.json`` (empty on ``main``: the suite passes), then trains a tiny
DQN policy for 400 agent steps and evaluates the resulting checkpoint on the 15
concrete seeds pinned by Breakout Contract v2.

Every step is a real CLI invocation in a subprocess, so the smoke path checks
the same entry points a reader uses, not private helper functions.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
KNOWN_FAILURES_PATH = Path(__file__).with_name("known_failures.json")
CONTRACT_PATH = Path("configs/eval/breakout_contract_v2.json")
TRAIN_STEPS = 400
TRAIN_RUN_ID = "smoke-train"
EVALUATION_ID = "smoke-evaluation"
EXPECTED_EVALUATION_EPISODES = 15


class SmokeError(RuntimeError):
    """Raised when a smoke stage produces an unexpected result."""


def _command_line(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def _repository_state() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in (
        "torch",
        "ale-py",
        "gymnasium",
        "numpy",
        "onnx",
        "onnxruntime",
        "onnxruntime-gpu",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _load_known_failures() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(KNOWN_FAILURES_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SmokeError(f"{KNOWN_FAILURES_PATH}: expected a JSON object")
    modules = payload.get("modules", {})
    tests = payload.get("tests", {})
    if not isinstance(modules, dict) or not isinstance(tests, dict):
        raise SmokeError(f"{KNOWN_FAILURES_PATH}: modules and tests must be objects")
    return modules, tests


def _failure_key(test: unittest.TestCase) -> tuple[str, str]:
    test_id = test.id()
    if test_id.startswith("unittest.loader._FailedTest."):
        return test_id.rsplit(".", 1)[-1], "module"
    return test_id, "test"


def _known_reason(
    key: str,
    kind: str,
    modules: dict[str, Any],
    tests: dict[str, Any],
) -> str | None:
    entry = modules.get(key) if kind == "module" else tests.get(key)
    if entry is None:
        return None
    if isinstance(entry, str):
        return entry
    return str(entry.get("reason", "known failure"))


def run_unit_tests(*, strict: bool) -> dict[str, Any]:
    """Run every discoverable unit test and classify known failures."""

    modules, tests = _load_known_failures()
    loader = unittest.TestLoader()
    suite = loader.discover(str(TESTS_DIR), pattern="test_*.py")
    print(f"Discovered {suite.countTestCases()} tests under {TESTS_DIR}")
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    elapsed = time.perf_counter() - started

    known: list[dict[str, str]] = []
    unexpected: list[tuple[unittest.TestCase, str]] = []
    for test, traceback in [*result.failures, *result.errors]:
        key, kind = _failure_key(test)
        reason = _known_reason(key, kind, modules, tests)
        if reason is None:
            unexpected.append((test, traceback))
        else:
            known.append({"test": key, "kind": kind, "reason": reason})

    failed_count = len(result.failures) + len(result.errors)
    skipped_count = len(result.skipped)
    passed_count = result.testsRun - failed_count - skipped_count
    if known:
        print(f"\nKnown pre-existing failures on main: {len(known)}")
        for entry in sorted(known, key=lambda item: item["test"]):
            print(f"  - {entry['test']}: {entry['reason']}")
        print("  These are listed in scripts/smoke/known_failures.json.")
        if not strict:
            print("  Use --strict-units to make the smoke fail on them as well.")
    if unexpected:
        print(f"\nUnexpected failures: {len(unexpected)}")
        for test, _traceback in unexpected:
            print(f"  - {test.id()}")

    passed = not unexpected and (not strict or not known)
    return {
        "status": "passed" if passed else "failed",
        "tests_run": result.testsRun,
        "passed": passed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "known_failures": known,
        "unexpected_failures": [test.id() for test, _traceback in unexpected],
        "strict": strict,
        "elapsed_seconds": round(elapsed, 3),
    }


def _run_logged(
    command: Sequence[str],
    *,
    log_path: Path,
    timeout_seconds: int = 900,
) -> float:
    """Run one CLI, echo its output, and keep the full log on disk."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {_command_line(command)}")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            process.kill()
            raise SmokeError(
                f"command exceeded {timeout_seconds}s; see {log_path}"
            ) from error
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise SmokeError(
            f"command exited with status {return_code}; see {log_path}"
        )
    return elapsed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SmokeError(f"{path}: expected a JSON object")
    return payload


def _write_evaluation_config(path: Path, contract: dict[str, Any]) -> None:
    seeds = contract.get("concrete_episode_seeds")
    _require(
        isinstance(seeds, list) and len(seeds) == EXPECTED_EVALUATION_EPISODES,
        "Breakout Contract v2 must pin exactly 15 concrete episode seeds",
    )
    payload = {
        "schema_version": 1,
        "name": "smoke-evaluation",
        "environment_id": contract["environment_id"],
        "seeds": seeds,
        # The contract runtime binding requires one episode per concrete seed.
        # configs/eval/breakout_eval.json uses 3 seed groups x 5 episodes and is
        # rejected by the current evaluator; see docs/repro/known-test-failures.md.
        "episodes_per_seed": 1,
        "epsilon": contract["evaluation_epsilon"],
        "checkpoint_rule": "checkpoint written by the 400-step smoke training run",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_e2e(output_root: Path) -> dict[str, Any]:
    """Train for 400 agent steps, then evaluate the checkpoint once."""

    contract = _load_json(REPO_ROOT / CONTRACT_PATH)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / f"smoke-{timestamp}"
    logs_dir = run_root / "logs"
    train_root = run_root / "train"
    evaluation_dir = run_root / "evaluation"
    evaluation_config = run_root / "eval-config.json"
    run_root.mkdir(parents=True, exist_ok=True)
    _write_evaluation_config(evaluation_config, contract)

    train_command = [
        sys.executable,
        "-m",
        "scripts.training.train_vectorized_dqn",
        "--preset",
        "smoke",
        "--total-steps",
        str(TRAIN_STEPS),
        "--num-envs",
        "1",
        "--device",
        "cpu",
        "--run-dir",
        str(train_root),
        "--run-id",
        TRAIN_RUN_ID,
        "--contract",
        CONTRACT_PATH.as_posix(),
    ]
    train_seconds = _run_logged(
        train_command,
        log_path=logs_dir / "train.log",
    )

    train_run_dir = train_root / TRAIN_RUN_ID
    train_summary_path = train_run_dir / "summary.json"
    _require(train_summary_path.is_file(), f"missing training summary: {train_summary_path}")
    train_summary = _load_json(train_summary_path)
    checkpoint = Path(str(train_summary.get("last_checkpoint", "")))
    _require(checkpoint.is_file(), f"training did not write a checkpoint: {checkpoint}")
    _require(train_summary.get("status") == "completed", "training status is not completed")
    _require(
        train_summary.get("total_agent_steps") == TRAIN_STEPS,
        "training did not record 400 agent steps",
    )
    _require(train_summary.get("total_steps") == TRAIN_STEPS, "training total_steps mismatch")
    _require(train_summary.get("num_envs") == 1, "training did not use one environment")
    _require(train_summary.get("algorithm") == "dqn", "training algorithm mismatch")
    _require(
        train_summary.get("contract_id") == contract["contract_id"],
        "training checkpoint did not record Contract v2",
    )
    frames = train_summary.get("total_emulator_frames")
    _require(
        isinstance(frames, int) and frames >= TRAIN_STEPS,
        f"training emulator frame count is invalid: {frames!r}",
    )

    evaluation_command = [
        sys.executable,
        "-m",
        "scripts.evaluation.evaluate_vectorized_dqn",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(evaluation_config),
        "--contract",
        CONTRACT_PATH.as_posix(),
        "--device",
        "cpu",
        "--output-dir",
        str(evaluation_dir),
        "--evaluation-id",
        EVALUATION_ID,
    ]
    evaluation_seconds = _run_logged(
        evaluation_command,
        log_path=logs_dir / "evaluate.log",
    )

    results_path = evaluation_dir / "results.json"
    _require(results_path.is_file(), f"missing evaluation results: {results_path}")
    results = _load_json(results_path)
    summary = results.get("summary", {})
    _require(
        results.get("evaluation_status") == "completed",
        "evaluation status is not completed",
    )
    _require(
        results.get("total_episodes") == EXPECTED_EVALUATION_EPISODES,
        "evaluation did not run 15 episodes",
    )
    _require(
        summary.get("complete_episodes") == EXPECTED_EVALUATION_EPISODES,
        "evaluation did not complete every episode",
    )
    _require(
        isinstance(results.get("total_agent_steps"), int)
        and results["total_agent_steps"] > 0,
        "evaluation agent-step count is invalid",
    )
    _require(
        isinstance(results.get("total_emulator_frames"), int)
        and results["total_emulator_frames"] >
        results["total_agent_steps"],
        "evaluation emulator-frame count is invalid",
    )
    _require(
        math.isfinite(float(summary.get("mean_return", math.nan))),
        "evaluation mean return is not finite",
    )
    checkpoint_record = results.get("checkpoint", {})
    _require(
        Path(str(checkpoint_record.get("path", ""))).resolve()
        == checkpoint.resolve(),
        "evaluation results do not reference the smoke checkpoint",
    )
    provenance = results.get("contract_provenance", {})
    _require(
        provenance.get("runtime_binding_validated") is True,
        "evaluation did not validate the Contract v2 runtime binding",
    )

    return {
        "status": "passed",
        "run_root": str(run_root),
        "checkpoint": str(checkpoint),
        "train_seconds": round(train_seconds, 3),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "training": {
            "total_agent_steps": train_summary.get("total_agent_steps"),
            "total_emulator_frames": frames,
            "episodes": train_summary.get("episodes"),
            "optimizer_updates": train_summary.get("optimizer_updates"),
        },
        "evaluation": {
            "total_episodes": results.get("total_episodes"),
            "total_agent_steps": results.get("total_agent_steps"),
            "total_emulator_frames": results.get("total_emulator_frames"),
            "mean_return": summary.get("mean_return"),
            "median_return": summary.get("median_return"),
            "contract_id": provenance.get("contract_id"),
            "runtime_binding_validated": provenance.get("runtime_binding_validated"),
        },
    }


def _preflight() -> None:
    _require((REPO_ROOT / CONTRACT_PATH).is_file(), f"missing {CONTRACT_PATH}")
    _require(TESTS_DIR.is_dir(), f"missing tests directory: {TESTS_DIR}")
    try:
        import ale_py
    except ImportError as error:
        raise SmokeError("ale-py is not installed; see docs/repro/setup.md") from error
    roms = Path(ale_py.__file__).resolve().parent / "roms"
    _require(
        (roms / "breakout.bin").is_file(),
        f"ALE Breakout ROM is missing from {roms}; ale-py 0.12.0 should bundle it",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the clean-checkout unit tests and the 400-step training + evaluation smoke.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--units-only",
        action="store_true",
        help="run the unit tests and stop",
    )
    mode.add_argument(
        "--e2e-only",
        action="store_true",
        help="skip the unit tests and run training + evaluation",
    )
    parser.add_argument(
        "--strict-units",
        action="store_true",
        help="fail when a known pre-existing unit-test failure is still present",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="parent directory for smoke artifacts (default: a new temporary directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        _preflight()
        repository = _repository_state()
        print("Breakout RL reproducibility smoke")
        print(f"  repository: {REPO_ROOT}")
        print(f"  commit:     {repository['commit']}")
        print(f"  describe:   {repository['describe']}")
        print(f"  python:     {sys.version.split()[0]} ({platform.platform()})")

        unit_tests: dict[str, Any] | None = None
        e2e: dict[str, Any] | None = None
        if not arguments.e2e_only:
            print("\n[1/2] Unit tests")
            unit_tests = run_unit_tests(strict=arguments.strict_units)
            print(
                f"\nUnit tests: {unit_tests['status'].upper()} "
                f"({unit_tests['passed']} passed, {unit_tests['skipped']} skipped, "
                f"{unit_tests['failed']} failed in {unit_tests['elapsed_seconds']:.1f}s)"
            )
            if unit_tests["status"] != "passed":
                raise SmokeError("unit tests failed")

        if not arguments.units_only:
            print("\n[2/2] End-to-end smoke: 400 training steps + one 15-episode evaluation")
            output_root = (
                arguments.output_dir.resolve()
                if arguments.output_dir is not None
                else Path(tempfile.mkdtemp(prefix="breakout-smoke-"))
            )
            e2e = run_e2e(output_root)
            print(
                f"\nTraining:   PASS in {e2e['train_seconds']:.1f}s "
                f"({e2e['training']['total_agent_steps']} agent steps, "
                f"{e2e['training']['total_emulator_frames']} emulator frames)"
            )
            print(
                f"Evaluation: PASS in {e2e['evaluation_seconds']:.1f}s "
                f"({e2e['evaluation']['total_episodes']} episodes, "
                f"mean return {e2e['evaluation']['mean_return']:.3f}, "
                f"{e2e['evaluation']['total_emulator_frames']} emulator frames)"
            )
            print(f"Artifacts:  {e2e['run_root']}")

        elapsed = time.perf_counter() - started
        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "repository": repository,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "commands": {
                "unit_tests": "python -m scripts.smoke.run_smoke --units-only",
                "train": (
                    "python -m scripts.training.train_vectorized_dqn --preset smoke "
                    f"--total-steps {TRAIN_STEPS} --num-envs 1 --device cpu"
                ),
                "evaluate": "python -m scripts.evaluation.evaluate_vectorized_dqn --device cpu",
            },
            "unit_tests": unit_tests,
            "e2e": e2e,
            "wall_clock_seconds": round(elapsed, 3),
        }
        if e2e is not None:
            manifest_path = Path(e2e["run_root"]) / "smoke-manifest.json"
        else:
            manifest_path = (
                arguments.output_dir.resolve()
                if arguments.output_dir is not None
                else Path(tempfile.mkdtemp(prefix="breakout-smoke-"))
            )
            manifest_path.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_path / "smoke-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Manifest:   {manifest_path}")
        print(f"\nSmoke: PASS in {elapsed:.1f}s")
        return 0
    except SmokeError as error:
        print(f"\nSmoke: FAIL — {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
