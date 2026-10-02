"""Supervise the single frozen Issue #54 command and attach its full logs."""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from breakout_rl.issue54_q_replay_consistency import (
    BASE_COMMIT, CONTRACT_SHA256, FORMAL_COMMAND, INDEX_SHA256, ISSUE50_MANIFEST_SHA256,
    ISSUE50_SOURCE_DIGEST, METADATA_SHA256, MODEL_SHA256, REPORT_SHA256, RESULTS_SHA256,
    ROOT, SPEC_SHA256, STACK_SHA256, SUPERVISOR_STARTUP_ALLOWANCE_SECONDS,
    TRAJECTORY_SHA256, _atomic_json,
    attach_log_capture_hashes, replay_source_digest, sha256_file,
)


OUTPUT_DIR = ROOT / "research/issue-54-q-replay-consistency-artifacts"
STDOUT_PATH = Path("/tmp/issue54-formal.stdout")
STDERR_PATH = Path("/tmp/issue54-formal.stderr")


def run_captured_command(command: Sequence[str], cwd: Path, stdout_path: Path, stderr_path: Path,
                         timeout_seconds: float) -> tuple[int, bool, float]:
    """Run once in a killable process group, preserving partial output on timeout."""
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(list(command), cwd=cwd, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                return_code = process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return_code = process.wait()
            if return_code == 0:
                return_code = 124
    return return_code, timed_out, time.perf_counter() - started


def _preserve_inconclusive(return_code: int, validation: dict) -> None:
    """Replace any incomplete/nonzero run classification with INCONCLUSIVE."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparisons = OUTPUT_DIR / "q_comparisons.json"
    if not comparisons.exists():
        _atomic_json(comparisons, [])
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    expected_hashes = {
        "issue50/decision_stack_index.jsonl": INDEX_SHA256,
        "issue50/trajectory.jsonl": TRAJECTORY_SHA256,
        "issue50/results.json": RESULTS_SHA256,
        "issue50/report.md": REPORT_SHA256,
        "issue50/manifest.json": ISSUE50_MANIFEST_SHA256,
        "web/public/models/final_model/model.onnx": MODEL_SHA256,
        "web/public/models/final_model/model.onnx.metadata.json": METADATA_SHA256,
        "configs/inference/inference_spec.json": SPEC_SHA256,
        "configs/eval/breakout_contract_v2.json": CONTRACT_SHA256,
        **{f"issue50/decision_stacks/{name}": digest for name, digest in STACK_SHA256.items()},
    }
    primary_metric = "max_i max_a |Q_replayed[i,a] - Q_logged[i,a]|"
    failure_analysis = (
        f"Frozen command ended with exit code {return_code} or incomplete outputs. "
        "No valid Q discrepancy or error delta is claimed; the replay outcome is unavailable "
        "because the formal command did not complete with valid artifacts."
    )
    result = {
        "issue": 54, "diagnostic": "INCONCLUSIVE", "integrity_passed": False,
        "hypothesis": "The exact Issue #50 uint8 observation stacks, normalized once with frozen Contract v2 preprocessing, reproduce logged ONNX Q vectors on CPU ONNX Runtime 1.22.1 within 1e-6.",
        "primary_metric": primary_metric, "max_abs_error": None, "error_delta": None,
        "failure_analysis": failure_analysis,
        "logged_q_baseline": {"source": "Issue #50 trajectory.jsonl", "expected_observations": 300,
                              "q_values_per_observation": 4},
        "expected_input_hashes": expected_hashes,
        "has_verified_clear": "NO", "phase1": "INCONCLUSIVE", "round_status": "CONTINUE_RESEARCH",
        "phase2_transition": False, "controller_or_champion_implication": False,
        "source_commit": source_commit, "experiment_branch": branch,
        "research_base_commit": BASE_COMMIT, "issue50_source_digest": ISSUE50_SOURCE_DIGEST,
        "replay_source_digest": replay_source_digest(), "formal_command": FORMAL_COMMAND,
        "runtime_expectation": "ONNX Runtime 1.22.1, CPUExecutionProvider, default SessionOptions, offline inference",
        "runtime": {"status": "unavailable", "reason": "formal command did not yield completed runtime metadata",
                    "observed_version": None,
                    "observed_provider": None, "expected_version": "1.22.1",
                    "expected_provider": "CPUExecutionProvider", "inference_count": "unknown"},
        "pre_run_validation": validation, "exit_code": return_code,
        "issue52_context": "Issue #52 fresh-range probe failed before environment creation/collection (N=0); no fresh-input evidence.",
        "scope_limit": "Only the 300 selected Issue #50 life-window stacks are in scope; HVC inference was not performed.",
        "remaining_uncertainty": "The frozen command did not produce a complete replay result; no claim can be made about its Q error. This experiment covers no new observations or controller behavior.",
        "recommended_next_decision": "Use only a completed integrity-passing replay to decide whether replay implementation fidelity merits further investigation; do not infer controller/champion or Phase 2 status.",
        "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
        "MODEL_ROUTING_MARKER": "MODEL_ROUTING_VERIFICATION: UNAVAILABLE",
    }
    _atomic_json(OUTPUT_DIR / "results.json", result)
    (OUTPUT_DIR / "report.md").write_text(
        "# Issue #54 Offline Q Replay Consistency\n\n**INCONCLUSIVE** — the frozen command ended without a complete valid replay; no Q discrepancy or delta is claimed.\n\n"
        "## Hypothesis and baseline\n\nThe exact 300 Issue #50 uint8 stacks, normalized once with frozen Contract v2 preprocessing, should reproduce the logged four-value ONNX Q vectors on CPU ONNX Runtime 1.22.1 within 1e-6. The existing baseline is Issue #50 trajectory.jsonl; the primary metric is "
        + primary_metric + ". The metric is unavailable because the command did not complete valid artifacts.\n\n"
        "## Failure analysis and reproducibility\n\n" + failure_analysis +
        f" Exit code: {return_code}. Branch: {branch}; source: {source_commit}; research base: {BASE_COMMIT}; Issue #50 source digest: {ISSUE50_SOURCE_DIGEST}. Exact command: {FORMAL_COMMAND}. Pinned expected input hashes, source digest, command, and final validation record are in results.json and manifest.json; stdout/stderr are attached after capture.\n\n"
        "Runtime evidence is unavailable: no observed ORT version/provider or completed inference count can be claimed. Expected runtime was ONNX Runtime 1.22.1 with CPUExecutionProvider and default SessionOptions.\n\n"
        f"## Pre-run validation\n\nCombined measured validation: {validation.get('combined_validation_wall_seconds', 'unavailable')}s. Exact commands, counts/status, timings, and captured stream hashes are in pre_run_validation.json.\n\n"
        "## Limits and next decision\n\nOnly the 300 selected Issue #50 life-window stacks are in scope. No HVC inference or fresh observation collection was performed; Issue #52 ended before environment creation/collection (N=0). No conclusion about a controller or its performance follows. A complete integrity-passing replay is needed before deciding whether implementation fidelity needs further investigation.\n\n"
        "Has Verified Clear: **NO**. Phase 1: **INCONCLUSIVE**. Round status: CONTINUE_RESEARCH. No champion or Phase 2 implication.\n\n"
        "MODEL_ROUTING_VERIFICATION: UNAVAILABLE\n")
    _atomic_json(OUTPUT_DIR / "manifest.json", {
        "issue": 54, "diagnostic": "INCONCLUSIVE", "source_commit": source_commit,
        "experiment_branch": branch, "research_base_commit": BASE_COMMIT,
        "issue50_source_digest": ISSUE50_SOURCE_DIGEST, "replay_source_digest": replay_source_digest(),
        "expected_input_hashes": expected_hashes, "pre_run_validation": validation,
        "formal_command": FORMAL_COMMAND,
        "runtime": {"status": "unavailable", "reason": "formal command did not yield completed runtime metadata",
                    "observed_version": None,
                    "observed_provider": None, "expected_version": "1.22.1",
                    "expected_provider": "CPUExecutionProvider", "inference_count": "unknown"},
        "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
        "MODEL_ROUTING_MARKER": "MODEL_ROUTING_VERIFICATION: UNAVAILABLE",
        "output_hashes": {name: sha256_file(OUTPUT_DIR / name)
                          for name in ("q_comparisons.json", "results.json", "report.md")},
    })


def _guarded_run(validation_record: Path) -> int:
    validation = json.loads(validation_record.read_text())
    started = time.perf_counter()
    remaining = 20.0 - float(validation["combined_validation_wall_seconds"])
    cli_timeout = remaining - SUPERVISOR_STARTUP_ALLOWANCE_SECONDS - 1.5
    if cli_timeout <= 0.0:
        raise TimeoutError("pre-run validation leaves no replay budget")
    return_code, timed_out, command_wall = run_captured_command(
        shlex.split(FORMAL_COMMAND), ROOT, STDOUT_PATH, STDERR_PATH, cli_timeout)
    # A timeout, nonzero status, or incomplete output invalidates any partial classification.
    failure_start = time.perf_counter()
    if timed_out or return_code != 0 or not all((OUTPUT_DIR / name).is_file() for name in
                                                ("q_comparisons.json", "results.json", "report.md", "manifest.json")):
        _preserve_inconclusive(return_code, validation)
    failure_bundle_wall = time.perf_counter() - failure_start
    captures = attach_log_capture_hashes(
        OUTPUT_DIR, STDOUT_PATH, STDERR_PATH, validation_record, command_wall,
        failure_bundle_wall, cli_timeout, timed_out,
        SUPERVISOR_STARTUP_ALLOWANCE_SECONDS)
    validation = json.loads(validation_record.read_text())
    wrapper_wall = time.perf_counter() - started
    print(json.dumps({
        "formal_command": FORMAL_COMMAND, "exit_code": return_code,
        "pre_run_validation_seconds": validation["combined_validation_wall_seconds"],
        "formal_command_wall_seconds": command_wall, "failure_bundle_wall_seconds": failure_bundle_wall,
        "capture_attachment_seconds": captures.get("post_run_capture_attachment_wall_seconds"),
        "combined_validation_plus_guarded_run_wall_seconds": validation["combined_validation_wall_seconds"] + wrapper_wall,
        "supervisor_startup_allowance_seconds": SUPERVISOR_STARTUP_ALLOWANCE_SECONDS,
        "combined_elapsed_scope": "Measured pre-run validation plus guarded run through capture/hash attachment; excludes pre-launch supervisor startup and final metadata refresh. The outer timeout is reduced by the explicit startup allowance and enforces the inclusive cap.",
        "capture_hashes": captures,
    }, sort_keys=True))
    return return_code


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    validation_record = Path(args[args.index("--validation-record") + 1])
    validation = json.loads(validation_record.read_text())
    if "--guarded" in args:
        return _guarded_run(validation_record)
    remaining = 20.0 - float(validation["combined_validation_wall_seconds"])
    outer_timeout = remaining - SUPERVISOR_STARTUP_ALLOWANCE_SECONDS
    if outer_timeout <= 0:
        raise TimeoutError("pre-run validation leaves no supervisor startup budget")
    command = ["timeout", "--signal=KILL", f"{outer_timeout:.6f}s", "env",
               "PYTHONPATH=/tmp/issue41-onnxruntime", "python", "-m",
               "scripts.analysis.run_issue54_once", "--validation-record", str(validation_record), "--guarded"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
