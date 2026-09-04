"""Repair completed Day 21 provenance without training or model selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.day21_final_training import load_day21_config, sha256_file, write_json


EXPECTED_MODEL_SHA256 = "6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a"
EXPECTED_PRIMARY_TRIGGER = (
    "2.5M evaluation showed substantial improvement, so 5M continuation remained justified."
)
EXPECTED_EVALUATION_ORDER = "selection → final freeze → final holdout"
EXPECTED_TRAINING_SEED = 2022
EXPECTED_TRANSITIONS = 2_500_000


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _resolve_reference(
    manifest_path: Path,
    value: Any,
    *,
    repository_root: Path,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest reference must be a non-empty path")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    from_manifest = (manifest_path.parent / candidate).resolve()
    if from_manifest.exists():
        return from_manifest
    return (repository_root / candidate).resolve()


def _stage_record(manifest: Mapping[str, Any], seed: int, stage: str) -> Mapping[str, Any]:
    for entry in manifest.get("runs", []):
        if not isinstance(entry, Mapping) or int(entry.get("training_seed", -1)) != seed:
            continue
        stages = entry.get("stages")
        record = stages.get(stage) if isinstance(stages, Mapping) else None
        if isinstance(record, Mapping):
            return record
    raise ValueError(f"manifest is missing {stage} for training seed {seed}")


def _mean_return(record: Mapping[str, Any], *, label: str) -> float:
    evaluation = record.get("evaluation")
    summary = evaluation.get("summary") if isinstance(evaluation, Mapping) else None
    value = summary.get("mean_return") if isinstance(summary, Mapping) else None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} has no finite mean_return") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} has no finite mean_return")
    return parsed


def _stage_c_provenance(config: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    decisions = manifest.get("selection_decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("completed Day 21 manifest is missing selection_decisions")
    stage_b_decision = decisions.get("stage_b_2_5m")
    if not isinstance(stage_b_decision, Mapping):
        raise ValueError("completed Day 21 manifest is missing Stage B decision")
    selected_b_seeds = [int(seed) for seed in stage_b_decision.get("selected_training_seeds", [])]
    if selected_b_seeds != [EXPECTED_TRAINING_SEED]:
        raise ValueError(
            "cleanup expects the frozen Stage B selection to remain training seed 2022"
        )
    stage_a = _stage_record(manifest, EXPECTED_TRAINING_SEED, "stage_a_1m")
    stage_b = _stage_record(manifest, EXPECTED_TRAINING_SEED, "stage_b_2_5m")
    if stage_a.get("status") != "completed" or stage_b.get("status") != "completed":
        raise ValueError("cleanup requires completed 1M and 2.5M evidence")
    one_million_mean = _mean_return(stage_a, label="seed 2022 @ 1M")
    two_point_five_million_mean = _mean_return(stage_b, label="seed 2022 @ 2.5M")
    configured = config.raw["execution"]["stage_c_policy"]["trigger_evidence"]
    if not math.isclose(
        one_million_mean,
        float(configured["stage_a_1m_mean_return"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or not math.isclose(
        two_point_five_million_mean,
        float(configured["stage_b_2_5m_mean_return"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("completed evidence does not match configured Stage C trigger evidence")
    return {
        "primary_trigger": EXPECTED_PRIMARY_TRIGGER,
        "trigger_evidence": {
            "training_seed": EXPECTED_TRAINING_SEED,
            "stage_a_1m_mean_return": one_million_mean,
            "stage_b_2_5m_mean_return": two_point_five_million_mean,
            "mean_return_improvement": two_point_five_million_mean - one_million_mean,
        },
        "user_requested_5m": True,
        "request_is_supplemental_provenance": True,
    }


def clean_day21_provenance(
    manifest_path: Path,
    *,
    config_path: Path,
) -> dict[str, Any]:
    config = load_day21_config(config_path)
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("cleanup requires a completed Day 21 manifest")

    final_model = manifest.get("canonical_final_model")
    if not isinstance(final_model, dict):
        raise ValueError("completed Day 21 manifest is missing canonical_final_model")
    if (
        final_model.get("algorithm") != "double_dqn"
        or final_model.get("architecture") != "dueling"
        or final_model.get("training_seed") != EXPECTED_TRAINING_SEED
        or final_model.get("training_transitions") != EXPECTED_TRANSITIONS
        or final_model.get("model_sha256") != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("canonical Final Model identity is not the frozen Day 21 model")
    model_path = _resolve_reference(
        manifest_path,
        final_model.get("model_path"),
        repository_root=config.repository_root,
    )
    if sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("canonical Final Model bytes do not match the frozen SHA256")

    decisions = manifest.get("selection_decisions")
    if not isinstance(decisions, dict):
        raise ValueError("completed Day 21 manifest is missing mutable selection_decisions")
    final_checkpoint = decisions.get("final_checkpoint")
    selected = final_checkpoint.get("selected") if isinstance(final_checkpoint, Mapping) else None
    if not isinstance(selected, Mapping) or (
        selected.get("training_seed") != EXPECTED_TRAINING_SEED
        or selected.get("stage") != "stage_b_2_5m"
        or selected.get("target_transitions") != EXPECTED_TRANSITIONS
    ):
        raise ValueError("cleanup refuses to change a non-canonical final checkpoint")
    if not isinstance(final_checkpoint, dict):
        raise ValueError("final_checkpoint must be an object")
    final_checkpoint["holdout_was_locked"] = True
    final_checkpoint["evaluation_order"] = EXPECTED_EVALUATION_ORDER

    provenance = _stage_c_provenance(config, manifest)
    stage_c_decision = decisions.get("stage_c_5m")
    if not isinstance(stage_c_decision, dict):
        raise ValueError("completed Day 21 manifest is missing Stage C decision")
    stage_c_decision.update(provenance)
    stage_c_decision["reason"] = EXPECTED_PRIMARY_TRIGGER
    for obsolete_key in ("explicit_user_target_override", "override_reason"):
        stage_c_decision.pop(obsolete_key, None)

    holdout = manifest.get("final_holdout")
    if not isinstance(holdout, dict) or holdout.get("status") != "completed":
        raise ValueError("cleanup requires a completed final holdout")
    if holdout.get("opened_after_final_freeze") is not True:
        raise ValueError("cleanup refuses a holdout without post-freeze provenance")
    contract_health = holdout.get("contract_health")
    if not isinstance(contract_health, Mapping) or contract_health.get("healthy") is not True:
        raise ValueError("cleanup requires a healthy Contract v2 final holdout")
    holdout["evaluation_order"] = EXPECTED_EVALUATION_ORDER
    holdout["model_sha256"] = EXPECTED_MODEL_SHA256

    metadata_path = _resolve_reference(
        manifest_path,
        final_model.get("metadata_path"),
        repository_root=config.repository_root,
    )
    metadata = _read_json(metadata_path)
    if metadata.get("model_sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("canonical metadata does not match the frozen model SHA256")
    metadata["stage_c_provenance"] = provenance
    metadata["evaluation_order"] = EXPECTED_EVALUATION_ORDER
    final_holdout_metadata = metadata.get("final_holdout")
    if not isinstance(final_holdout_metadata, dict):
        final_holdout_metadata = {}
    final_holdout_metadata.update(
        {
            "status": "completed",
            "summary": holdout["summary"],
            "results": holdout["results"],
            "episodes": holdout["episodes"],
            "concrete_episode_seeds": holdout["concrete_episode_seeds"],
            "contract_health": contract_health,
            "opened_after_final_freeze": True,
            "evaluation_order": EXPECTED_EVALUATION_ORDER,
            "model_sha256": EXPECTED_MODEL_SHA256,
        }
    )
    metadata["final_holdout"] = final_holdout_metadata
    write_json(metadata_path, metadata)

    final_model["stage_c_provenance"] = provenance
    final_model["evaluation_order"] = EXPECTED_EVALUATION_ORDER
    final_model["final_holdout"] = final_holdout_metadata
    final_model["metadata_sha256"] = sha256_file(metadata_path)
    manifest["protocol"] = config.protocol()
    sources = manifest.get("source_of_truth")
    if not isinstance(sources, dict) or not isinstance(sources.get("config"), dict):
        raise ValueError("completed Day 21 manifest is missing source_of_truth.config")
    sources["config"]["sha256"] = sha256_file(config.source_path)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )
    write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path.as_posix(),
        "metadata": metadata_path.as_posix(),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "stage_c_primary_trigger": EXPECTED_PRIMARY_TRIGGER,
        "evaluation_order": EXPECTED_EVALUATION_ORDER,
        "training_rerun": False,
        "model_selection_changed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean completed Day 21 provenance without retraining or reselecting."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day21-final-long-training/manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final-training/manifest.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = clean_day21_provenance(args.manifest.resolve(), config_path=args.config.resolve())
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"Day 21 provenance cleanup failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
