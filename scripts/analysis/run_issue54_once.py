"""Supervise the single frozen Issue #54 command and attach its full logs."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from breakout_rl.issue54_q_replay_consistency import (
    BASE_COMMIT, FORMAL_COMMAND, ISSUE50_SOURCE_DIGEST, ROOT, _atomic_json,
    attach_log_capture_hashes, replay_source_digest, sha256_file,
)


OUTPUT_DIR = ROOT / "research/issue-54-q-replay-consistency-artifacts"
STDOUT_PATH = Path("/tmp/issue54-formal.stdout")
STDERR_PATH = Path("/tmp/issue54-formal.stderr")


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
    started = time.perf_counter()
    with STDOUT_PATH.open("wb") as stdout, STDERR_PATH.open("wb") as stderr:
        completed = subprocess.run(shlex.split(FORMAL_COMMAND), cwd=ROOT, stdout=stdout,
                                   stderr=stderr, check=False)
    # A nonzero status invalidates any partial diagnostic and retains partial comparisons only.
    if completed.returncode != 0 or not all((OUTPUT_DIR / name).is_file() for name in
                                            ("q_comparisons.json", "results.json", "report.md", "manifest.json")):
        _preserve_inconclusive(completed.returncode)
    elapsed_before_capture = time.perf_counter() - started
    captures = attach_log_capture_hashes(
        OUTPUT_DIR, STDOUT_PATH, STDERR_PATH, validation_record, elapsed_before_capture)
    validation = json.loads(validation_record.read_text())
    wrapper_wall = time.perf_counter() - started
    print(json.dumps({
        "formal_command": FORMAL_COMMAND, "exit_code": completed.returncode,
        "pre_run_validation_seconds": validation["combined_validation_wall_seconds"],
        "formal_and_failure_finalization_seconds": elapsed_before_capture,
        "capture_attachment_seconds": captures.get("post_run_capture_attachment_wall_seconds"),
        "combined_elapsed_seconds": validation["combined_validation_wall_seconds"] + wrapper_wall,
        "capture_hashes": captures,
    }, sort_keys=True))
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    validation_record = Path(args[args.index("--validation-record") + 1])
    validation = json.loads(validation_record.read_text())
    if "--guarded" in args:
        return _guarded_run(validation_record)
    remaining = 20.0 - float(validation["combined_validation_wall_seconds"])
    outer_seconds = max(0.1, remaining - 0.25)
    command = ["timeout", "--signal=KILL", f"{outer_seconds:.6f}s", "env",
               "PYTHONPATH=/tmp/issue41-onnxruntime", "python", "-m",
               "scripts.analysis.run_issue54_once", "--validation-record", str(validation_record), "--guarded"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
