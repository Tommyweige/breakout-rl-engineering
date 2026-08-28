"""Combine Day 16 candidate evaluation artifacts without recomputing results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    if not isinstance(payload.get("summary"), Mapping):
        raise ValueError(f"{path} must contain a summary object")
    return payload


def _git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _candidate(
    *,
    environment_count: int,
    results_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    result = _load(results_path)
    return {
        "environment_count": environment_count,
        "results_path": results_path.as_posix(),
        "results_sha256": _sha256(results_path),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_id": result.get("model_id"),
        "summary": dict(result["summary"]),
    }


def _random_baseline(results_path: Path) -> dict[str, Any]:
    result = _load(results_path)
    return {
        "results_path": results_path.as_posix(),
        "results_sha256": _sha256(results_path),
        "policy_type": result.get("policy_type"),
        "summary": dict(result["summary"]),
    }


def _artifact_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Day 16 evaluation artifacts")
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-environment-count", type=int, default=4)
    parser.add_argument(
        "--screening-training-report",
        type=Path,
        default=Path("assets/day16/vectorized-training.json"),
    )
    parser.add_argument("--validation-reference-results", type=Path, default=None)
    parser.add_argument("--validation-reference-checkpoint", type=Path, default=None)
    parser.add_argument("--validation-candidate-results", type=Path, default=None)
    parser.add_argument("--validation-candidate-checkpoint", type=Path, default=None)
    parser.add_argument("--validation-candidate-environment-count", type=int, default=4)
    parser.add_argument(
        "--validation-training-report",
        type=Path,
        default=Path("assets/day16/vectorized-training-100k.json"),
    )
    parser.add_argument("--random-results", type=Path, default=None)
    parser.add_argument("--fire-diagnostic", type=Path, default=None)
    parser.add_argument("--q-value-diagnostic", type=Path, default=None)
    parser.add_argument("--overestimation-report", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day16/evaluation-summary.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_values = (
        args.validation_reference_results,
        args.validation_reference_checkpoint,
        args.validation_candidate_results,
        args.validation_candidate_checkpoint,
    )
    if any(value is not None for value in validation_values) and not all(
        value is not None for value in validation_values
    ):
        raise ValueError(
            "all four validation reference/candidate paths are required together"
        )

    screening_candidates = [
        _candidate(
            environment_count=1,
            results_path=args.reference_results,
            checkpoint_path=args.reference_checkpoint,
        ),
        _candidate(
            environment_count=args.candidate_environment_count,
            results_path=args.candidate_results,
            checkpoint_path=args.candidate_checkpoint,
        ),
    ]
    report: dict[str, Any] = {
        "schema_version": 2,
        "git_commit_sha": _git_commit_sha(),
        "purpose": "Day 16 Contract v2 learning-regression guardrail",
        "protocol": {
            "evaluation_config": "configs/eval/breakout_eval.json",
            "evaluation_contract": "configs/eval/breakout_contract_v2.json",
            "episodes": 15,
            "epsilon": 0.0,
            "raw_reward": True,
            "selection_rule": (
                "the selected vector candidate is chosen from the near-top throughput "
                "settings and strict action-selection parity; quality is not selected "
                "from the 10K screening"
            ),
            "action_distribution_semantics": "executed/wrapper-resolved action",
            "provenance_fields": [
                "requested_action_distribution",
                "executed_action_distribution",
                "auto_fire_count",
                "auto_fire_reason_counts",
            ],
        },
        "screening_10k": {
            "training_report": args.screening_training_report.as_posix(),
            "total_transitions": 10_000,
            "candidates": screening_candidates,
        },
        # Keep the original top-level field for small downstream readers that
        # consume the pre-finalization summary shape.
        "candidates": screening_candidates,
    }
    if all(value is not None for value in validation_values):
        report["long_validation_100k"] = {
            "training_report": args.validation_training_report.as_posix(),
            "total_transitions": 100_000,
            "candidates": [
                _candidate(
                    environment_count=1,
                    results_path=args.validation_reference_results,
                    checkpoint_path=args.validation_reference_checkpoint,
                ),
                _candidate(
                    environment_count=args.validation_candidate_environment_count,
                    results_path=args.validation_candidate_results,
                    checkpoint_path=args.validation_candidate_checkpoint,
                ),
            ],
        }
    if args.random_results is not None:
        report["random_baseline"] = _random_baseline(args.random_results)
    diagnostic_paths = {
        "fire_sticky": args.fire_diagnostic,
        "q_value": args.q_value_diagnostic,
        "overestimation_simulation": args.overestimation_report,
    }
    selected_diagnostics = {
        name: _artifact_reference(path)
        for name, path in diagnostic_paths.items()
        if path is not None
    }
    if selected_diagnostics:
        report["diagnostics"] = selected_diagnostics
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
