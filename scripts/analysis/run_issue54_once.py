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
    BASE_COMMIT, FORMAL_COMMAND, ISSUE50_SOURCE_DIGEST, ROOT, _atomic_json,
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


def _preserve_inconclusive(return_code: int) -> None:
    """Replace any incomplete/nonzero run classification with INCONCLUSIVE."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparisons = OUTPUT_DIR / "q_comparisons.json"
    if not comparisons.exists():
        _atomic_json(comparisons, [])
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    result = {
        "issue": 54, "diagnostic": "INCONCLUSIVE", "integrity_passed": False,
        "failure_analysis": f"Frozen command ended with exit code {return_code}; no valid Q discrepancy is claimed.",
        "has_verified_clear": "NO", "phase1": "INCONCLUSIVE", "round_status": "CONTINUE_RESEARCH",
        "phase2_transition": False, "controller_or_champion_implication": False,
        "source_commit": source_commit, "experiment_branch": branch,
        "research_base_commit": BASE_COMMIT, "issue50_source_digest": ISSUE50_SOURCE_DIGEST,
        "replay_source_digest": replay_source_digest(), "exit_code": return_code,
        "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
        "MODEL_ROUTING_MARKER": "MODEL_ROUTING_VERIFICATION: UNAVAILABLE",
    }
    _atomic_json(OUTPUT_DIR / "results.json", result)
    (OUTPUT_DIR / "report.md").write_text(
        "# Issue #54 Offline Q Replay Consistency\n\n**INCONCLUSIVE** — no valid Q discrepancy is claimed.\n\n"
        "Has Verified Clear: **NO**. Phase 1: **INCONCLUSIVE**. Round status: `CONTINUE_RESEARCH`.\n\n"
        "`MODEL_ROUTING_VERIFICATION: UNAVAILABLE`\n")
    _atomic_json(OUTPUT_DIR / "manifest.json", {
        "issue": 54, "diagnostic": "INCONCLUSIVE", "source_commit": source_commit,
        "experiment_branch": branch, "research_base_commit": BASE_COMMIT,
        "issue50_source_digest": ISSUE50_SOURCE_DIGEST, "replay_source_digest": replay_source_digest(),
        "MODEL_ROUTING_VERIFICATION": "UNAVAILABLE",
        "MODEL_ROUTING_MARKER": "MODEL_ROUTING_VERIFICATION: UNAVAILABLE",
        "output_hashes": {name: sha256_file(OUTPUT_DIR / name)
                          for name in ("q_comparisons.json", "results.json", "report.md")},
    })


def _guarded_run(validation_record: Path) -> int:
    validation = json.loads(validation_record.read_text())
    started = time.perf_counter()
    remaining = 20.0 - float(validation["combined_validation_wall_seconds"])
    cli_timeout = remaining - 1.5
    if cli_timeout <= 0.0:
        raise TimeoutError("pre-run validation leaves no replay budget")
    return_code, timed_out, command_wall = run_captured_command(
        shlex.split(FORMAL_COMMAND), ROOT, STDOUT_PATH, STDERR_PATH, cli_timeout)
    # A timeout, nonzero status, or incomplete output invalidates any partial classification.
    failure_start = time.perf_counter()
    if timed_out or return_code != 0 or not all((OUTPUT_DIR / name).is_file() for name in
                                                ("q_comparisons.json", "results.json", "report.md", "manifest.json")):
        _preserve_inconclusive(return_code)
    failure_bundle_wall = time.perf_counter() - failure_start
    captures = attach_log_capture_hashes(
        OUTPUT_DIR, STDOUT_PATH, STDERR_PATH, validation_record, command_wall,
        failure_bundle_wall, cli_timeout, timed_out)
    validation = json.loads(validation_record.read_text())
    wrapper_wall = time.perf_counter() - started
    print(json.dumps({
        "formal_command": FORMAL_COMMAND, "exit_code": return_code,
        "pre_run_validation_seconds": validation["combined_validation_wall_seconds"],
        "formal_command_wall_seconds": command_wall, "failure_bundle_wall_seconds": failure_bundle_wall,
        "capture_attachment_seconds": captures.get("post_run_capture_attachment_wall_seconds"),
        "combined_validation_plus_guarded_run_wall_seconds": validation["combined_validation_wall_seconds"] + wrapper_wall,
        "combined_elapsed_scope": "Measured pre-run validation plus guarded run through capture/hash attachment; excludes supervisor startup and final metadata refresh. The outer timeout enforces the inclusive cap.",
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
    command = ["timeout", "--signal=KILL", f"{remaining:.6f}s", "env",
               "PYTHONPATH=/tmp/issue41-onnxruntime", "python", "-m",
               "scripts.analysis.run_issue54_once", "--validation-record", str(validation_record), "--guarded"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
