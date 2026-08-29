"""Generate the canonical Day 16 backend manifest from real run metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.evaluation_contract import (
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.evaluation_artifacts import read_evaluation_results
from breakout_rl.training.backend_manifest import (
    DAY16_BACKEND_MANIFEST_SCHEMA_VERSION,
    DAY16_CANONICAL_BACKEND_ID,
    DAY16_SELECTED_BACKEND_ROLE,
    validate_day16_backend_manifest,
)
from breakout_rl.training.config import DQNConfig


def _load_object(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, repository_root: Path) -> dict[str, str]:
    try:
        relative = path.resolve().relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"artifact must be inside repository: {path}") from error
    return {"path": relative.as_posix(), "sha256": _sha256(path)}


def _training_record(
    report: Mapping[str, Any],
    *,
    environment_count: int,
) -> Mapping[str, Any]:
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("training report must contain a results array")
    matches = [
        value
        for value in results
        if isinstance(value, Mapping)
        and value.get("environment_count") == environment_count
    ]
    if len(matches) != 1:
        raise ValueError(
            f"training report must contain exactly one N={environment_count} result"
        )
    return matches[0]


def build_manifest(
    *,
    run_config_path: Path,
    contract_path: Path,
    training_report_path: Path,
    evaluation_results_path: Path,
    checkpoint_path: Path,
    evaluation_summary_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    run_config = _load_object(run_config_path)
    config = DQNConfig.from_dict(run_config)
    if config.num_envs != 2:
        raise ValueError("selected run config must have num_envs=2")
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    training_report = _load_object(training_report_path)
    record = _training_record(training_report, environment_count=config.num_envs)
    training_summary = record.get("summary")
    if not isinstance(training_summary, Mapping):
        raise ValueError("selected training report result must contain summary")
    evaluation_results = read_evaluation_results(evaluation_results_path)
    evaluation_summary = evaluation_results.get("summary")
    if not isinstance(evaluation_summary, Mapping):
        raise ValueError("selected evaluation artifact must contain summary")
    final_summary = _load_object(evaluation_summary_path)
    if final_summary.get("schema_version") != 2:
        raise ValueError("Day 16 evaluation summary must use schema v2")
    long_validation = final_summary.get("long_validation_100k")
    if not isinstance(long_validation, Mapping):
        raise ValueError("Day 16 evaluation summary is missing long_validation_100k")
    selected_matches = [
        value
        for value in long_validation.get("candidates", [])
        if isinstance(value, Mapping)
        and value.get("environment_count") == config.num_envs
        and value.get("role") == "selected_systems_backend"
    ]
    if len(selected_matches) != 1:
        raise ValueError(
            "Day 16 evaluation summary must identify exactly one selected N=2 candidate"
        )

    payload: dict[str, Any] = {
        "schema_version": DAY16_BACKEND_MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "backend_id": DAY16_CANONICAL_BACKEND_ID,
        "source_day": 16,
        "environment_contract": {
            "path": contract_path.resolve().relative_to(repository_root.resolve()).as_posix(),
            "contract_id": contract.contract_id,
        },
        "trainer": {
            "type": str(training_summary.get("trainer", "vectorized_dqn")),
            "num_envs": config.num_envs,
            "strict_action_selection_parity": config.strict_action_selection_parity,
            "replay_backend": config.replay_backend,
            "device": config.device,
            "precision": config.precision,
            "cpu_threads": config.cpu_threads,
            "config": config.to_dict(),
        },
        "selection": {
            "role": DAY16_SELECTED_BACKEND_ROLE,
            "training_run_id": str(training_summary.get("run_id", "envs-2")),
            "training_transitions": int(training_summary["total_transitions"]),
            "training_transitions_per_second": float(
                training_summary["environment_transitions_per_second"]
            ),
            "evaluation_mean_return": float(evaluation_summary["mean_return"]),
            "evaluation_protocol": "Contract v2, epsilon=0, raw reward, 15 fixed episodes",
        },
        "evidence": {
            "training_report": _artifact(
                training_report_path,
                repository_root=repository_root,
            ),
            "evaluation_results": _artifact(
                evaluation_results_path,
                repository_root=repository_root,
            ),
            "checkpoint": _artifact(
                checkpoint_path,
                repository_root=repository_root,
            ),
            "evaluation_summary": _artifact(
                evaluation_summary_path,
                repository_root=repository_root,
            ),
        },
    }
    validate_day16_backend_manifest(
        payload,
        repository_root=repository_root,
        verify_evidence_files=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("runs/day16-finalization-100k-v2-n2/envs-2/config.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=Path("assets/day16/vectorized-training-100k-n2.json"),
    )
    parser.add_argument(
        "--evaluation-results",
        type=Path,
        default=Path("evaluations/day16-final-100k-envs2-contract-v2/results.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/day16-finalization-100k-v2-n2/envs-2/checkpoints/step-00100000.pt"
        ),
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=Path("assets/day16/evaluation-summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/training/day16-canonical-backend.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path.cwd()
    payload = build_manifest(
        run_config_path=args.run_config,
        contract_path=args.contract,
        training_report_path=args.training_report,
        evaluation_results_path=args.evaluation_results,
        checkpoint_path=args.checkpoint,
        evaluation_summary_path=args.evaluation_summary,
        repository_root=repository_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    print(f"Backend manifest written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
