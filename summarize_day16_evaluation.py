"""Combine Day 16 candidate evaluation artifacts without recomputing results."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Day 16 evaluation artifacts")
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day16/evaluation-summary.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = {
        "schema_version": 1,
        "purpose": "Day 16 Contract v2 learning-regression guardrail",
        "protocol": {
            "evaluation_config": "configs/eval/breakout_eval.json",
            "evaluation_contract": "configs/eval/breakout_contract_v2.json",
            "episodes": 15,
            "epsilon": 0.0,
            "raw_reward": True,
            "selection_rule": "envs=8 is a throughput candidate only; quality is not selected from this 10K screening",
        },
        "candidates": [
            _candidate(
                environment_count=1,
                results_path=args.reference_results,
                checkpoint_path=args.reference_checkpoint,
            ),
            _candidate(
                environment_count=8,
                results_path=args.candidate_results,
                checkpoint_path=args.candidate_checkpoint,
            ),
        ],
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
