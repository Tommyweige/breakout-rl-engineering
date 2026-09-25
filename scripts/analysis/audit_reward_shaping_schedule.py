"""Audit schedule parity between the Issue #9 250k and 1M runs.

The audit deliberately separates two questions that are easy to conflate:

* whether the per-transition learning/exploration schedule is unchanged; and
* whether the two experiments are a byte-for-byte or full-runtime replay.

The latter is false when a run has a different budget, checkpoint cadence,
worker population, or cannot be resumed with replay and environment state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.evaluation_contract import load_evaluation_contract
from breakout_rl.exploration import LinearEpsilonSchedule
from breakout_rl.training.config import DQNConfig


SCHEDULE_FIELDS = (
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay_steps",
    "learning_rate",
    "gamma",
    "batch_size",
    "replay_capacity",
    "learning_starts",
    "train_frequency",
    "target_update_interval",
)
UPDATE_FIELDS = (
    "gamma",
    "batch_size",
    "replay_capacity",
    "learning_starts",
    "train_frequency",
    "target_update_interval",
)
NON_EXPERIMENT_FIELDS = {"life_loss_penalty", "contract_id", "contract_path"}

DEFAULT_STAGE2_CONFIG = Path("configs/issue9_reward_shaping_250k_baseline.json")
DEFAULT_STAGE3_CONFIG = Path("configs/issue9_reward_shaping_1m_baseline.json")
DEFAULT_STAGE2_MANIFEST = Path(
    "experiments/issue-9-reward-shaping/stage2-250k/training-sweep.json"
)
DEFAULT_STAGE3_MANIFEST = Path(
    "experiments/issue-9-reward-shaping/stage3-1m/training-sweep.json"
)
DEFAULT_RESUME_VALIDATION = Path(
    "experiments/issue-9-reward-shaping/stage3-1m/resume-validation.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _config(path: Path) -> tuple[dict[str, Any], DQNConfig]:
    payload = _read_json(path)
    raw_config = payload.get("training_config", payload)
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"{path}: training_config must be a JSON object")
    return payload, DQNConfig.from_dict(raw_config)


def _contract_path(config_path: Path, payload: Mapping[str, Any]) -> Path:
    raw_config = payload.get("training_config", payload)
    if isinstance(raw_config, Mapping):
        raw = raw_config.get("contract_path")
        if isinstance(raw, str) and raw.strip():
            return Path(raw)
    for key in ("contract", "environment_contract", "contract_path"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return Path(raw)
        if isinstance(raw, Mapping) and isinstance(raw.get("path"), str):
            return Path(raw["path"])
    del config_path
    return Path("configs/eval/breakout_contract_v2.json")


def _contract_fingerprint(config_path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    path = _contract_path(config_path, payload)
    contract = load_evaluation_contract(path).to_dict()
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "contract_id": str(contract["contract_id"]),
    }


def _config_values(config: DQNConfig) -> dict[str, Any]:
    return config.to_dict()


def _differences(
    left: DQNConfig,
    right: DQNConfig,
    *,
    excluded: set[str] | frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    left_values = _config_values(left)
    right_values = _config_values(right)
    return {
        field: {"stage2": left_values.get(field), "stage3": right_values.get(field)}
        for field in sorted(set(left_values) | set(right_values))
        if field not in excluded and left_values.get(field) != right_values.get(field)
    }


def _schedule_snapshot(config: DQNConfig) -> dict[str, Any]:
    schedule = LinearEpsilonSchedule(
        config.epsilon_start,
        config.epsilon_end,
        config.epsilon_decay_steps,
    )
    sample_steps = (0, 5_000, 10_000, 250_000, 500_000, 750_000, 1_000_000)
    return {
        "fields": {field: getattr(config, field) for field in SCHEDULE_FIELDS},
        "epsilon_at_steps": {
            str(step): schedule.value(step) for step in sample_steps
        },
        "learning_rate_schedule": {
            "type": "constant",
            "value": config.learning_rate,
            "scheduler": None,
        },
    }


def _source_schedule_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for relative_path in (
        "breakout_rl/training/dqn_trainer.py",
        "breakout_rl/training/vectorized.py",
    ):
        path = Path(relative_path)
        source = path.read_text(encoding="utf-8")
        schedule_match = re.search(
            r"self\.schedule\s*=\s*LinearEpsilonSchedule\((?P<body>.*?)\)\n",
            source,
            flags=re.DOTALL,
        )
        optimizer_match = re.search(
            r"self\.optimizer\s*=.*?torch\.optim\.Adam\((?P<body>.*?)\)\n",
            source,
            flags=re.DOTALL,
        )
        schedule_body = schedule_match.group("body") if schedule_match else ""
        optimizer_body = optimizer_match.group("body") if optimizer_match else ""
        checks[relative_path] = {
            "schedule_constructor_found": schedule_match is not None,
            "schedule_uses_explicit_decay_steps": "config.epsilon_decay_steps"
            in schedule_body,
            "schedule_uses_total_steps": "config.total_steps" in schedule_body,
            "optimizer_constructor_found": optimizer_match is not None,
            "optimizer_uses_config_learning_rate": "lr=config.learning_rate"
            in optimizer_body,
            "learning_rate_scheduler_symbols_present": any(
                token in source for token in ("lr_scheduler", "StepLR", "CosineAnnealing")
            ),
        }
    return checks


def _manifest_execution(
    manifest: Mapping[str, Any],
    *,
    observed_parallel: bool,
) -> dict[str, Any]:
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("training manifest must contain a non-empty variants list")
    statuses = [variant.get("status") for variant in variants if isinstance(variant, Mapping)]
    if len(statuses) != len(variants):
        raise ValueError("training manifest variants must be JSON objects")
    manifest_parallel = manifest.get("parallel_execution")
    if isinstance(manifest_parallel, bool):
        parallel = manifest_parallel
        source = "training-sweep.json"
        provenance_verified = True
    else:
        parallel = observed_parallel
        source = "unpersisted legacy invocation hint; not evidence"
        provenance_verified = False
    return {
        "parallel": parallel,
        "parallel_source": source,
        "parallel_provenance_verified": provenance_verified,
        "variant_count": len(variants),
        "worker_count": len(variants) if parallel else 1,
        "statuses": statuses,
        "all_completed": all(status == "completed" for status in statuses),
        "labels": [variant.get("label") for variant in variants],
        "config_paths": [variant.get("config_path") for variant in variants],
    }


def _resume_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": path.as_posix(),
            "present": False,
            "exact_continuation": False,
            "blockers": ["resume-validation artifact is missing"],
        }
    payload = _read_json(path)
    checkpoints = payload.get("checkpoints", [])
    validation_errors: list[str] = []
    if payload.get("schema_version") != 1:
        validation_errors.append("resume-validation schema_version must be 1")
    if payload.get("artifact_type") != "issue9_reward_shaping_resume_validation":
        validation_errors.append("resume-validation artifact_type is invalid")
    if not isinstance(checkpoints, list) or not checkpoints:
        validation_errors.append("resume-validation must contain checkpoints")
        checkpoints = []
    blockers: list[str] = []
    raw_blockers = payload.get("blockers")
    if isinstance(raw_blockers, list):
        blockers.extend(str(item) for item in raw_blockers)
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            validation_errors.append("resume-validation contains a non-object checkpoint")
            continue
        checkpoint_blockers = checkpoint.get("blockers", [])
        if isinstance(checkpoint_blockers, list):
            blockers.extend(str(item) for item in checkpoint_blockers)
        elif checkpoint_blockers is not None:
            validation_errors.append("checkpoint blockers must be a list")
        required_true_fields = (
            "replay_saved",
            "has_replay_state",
            "has_rng_state",
            "has_environment_state",
            "has_model_state",
            "has_step_state",
            "exact_continuation_available",
        )
        for field in required_true_fields:
            if checkpoint.get(field) is not True:
                validation_errors.append(f"checkpoint {field} is not true")
        if checkpoint.get("resume_contract_version") != 1:
            validation_errors.append("checkpoint resume_contract_version must be 1")
    if payload.get("exact_continuation_available") is not True:
        validation_errors.append("resume-validation does not declare exact continuation")
    blockers.extend(validation_errors)
    exact = not blockers
    return {
        "path": path.as_posix(),
        "present": True,
        "exact_continuation": exact,
        "decision": payload.get("decision"),
        "checkpoints": len(checkpoints),
        "blockers": sorted(set(blockers)),
        "validation_errors": sorted(set(validation_errors)),
    }


def build_parity_audit(
    *,
    stage2_config_path: Path = DEFAULT_STAGE2_CONFIG,
    stage3_config_path: Path = DEFAULT_STAGE3_CONFIG,
    stage2_manifest_path: Path = DEFAULT_STAGE2_MANIFEST,
    stage3_manifest_path: Path = DEFAULT_STAGE3_MANIFEST,
    resume_validation_path: Path = DEFAULT_RESUME_VALIDATION,
    stage2_parallel: bool = True,
    stage3_parallel: bool = True,
) -> dict[str, Any]:
    stage2_payload, stage2_config = _config(stage2_config_path)
    stage3_payload, stage3_config = _config(stage3_config_path)
    stage2_manifest = _read_json(stage2_manifest_path)
    stage3_manifest = _read_json(stage3_manifest_path)
    stage2_execution = _manifest_execution(
        stage2_manifest,
        observed_parallel=stage2_parallel,
    )
    stage3_execution = _manifest_execution(
        stage3_manifest,
        observed_parallel=stage3_parallel,
    )

    config_differences = _differences(
        stage2_config,
        stage3_config,
        excluded=NON_EXPERIMENT_FIELDS,
    )
    schedule_differences = {
        field: config_differences[field]
        for field in SCHEDULE_FIELDS
        if field in config_differences
    }
    update_differences = {
        field: config_differences[field]
        for field in UPDATE_FIELDS
        if field in config_differences
    }
    stage2_contract = _contract_fingerprint(stage2_config_path, stage2_payload)
    stage3_contract = _contract_fingerprint(stage3_config_path, stage3_payload)
    source_checks = _source_schedule_checks()
    total_steps_not_used_for_schedule = all(
        check["schedule_constructor_found"]
        and check["schedule_uses_explicit_decay_steps"]
        and not check["schedule_uses_total_steps"]
        for check in source_checks.values()
    )
    constant_learning_rate = all(
        check["optimizer_constructor_found"]
        and check["optimizer_uses_config_learning_rate"]
        and not check["learning_rate_scheduler_symbols_present"]
        for check in source_checks.values()
    )
    environment_equivalent = (
        stage2_contract == stage3_contract
        and stage2_config.reward_clip == stage3_config.reward_clip
        and stage2_config.num_envs == stage3_config.num_envs
        and stage2_config.strict_action_selection_parity
        == stage3_config.strict_action_selection_parity
    )
    epsilon_schedule_equivalent = (
        not any(
            field in schedule_differences
            for field in ("epsilon_start", "epsilon_end", "epsilon_decay_steps")
        )
        and total_steps_not_used_for_schedule
    )
    update_schedule_equivalent = not update_differences
    learning_rate_schedule_equivalent = (
        constant_learning_rate
        and stage2_config.learning_rate == stage3_config.learning_rate
    )
    learning_schedule_equivalent = (
        epsilon_schedule_equivalent
        and learning_rate_schedule_equivalent
        and update_schedule_equivalent
    )
    unexpected_condition_differences = {
        field: values
        for field, values in config_differences.items()
        if field not in {"total_steps", "checkpoint_interval"}
    }
    controlled_conditions_equivalent = (
        not unexpected_condition_differences and environment_equivalent
    )
    schedule_equivalent = learning_schedule_equivalent and controlled_conditions_equivalent

    stage3_variants: list[
        tuple[dict[str, Any], DQNConfig, dict[str, str]]
    ] = []
    for variant in stage3_manifest["variants"]:
        config_path = Path(str(variant["config_path"]))
        payload, config = _config(config_path)
        stage3_variants.append(
            (dict(variant), config, _contract_fingerprint(config_path, payload))
        )
    baseline_entry = next(
        (
            (variant, config, contract)
            for variant, config, contract in stage3_variants
            if config.life_loss_penalty == 0.0
        ),
        None,
    )
    strong_penalty_entry = next(
        (
            (variant, config, contract)
            for variant, config, contract in stage3_variants
            if config.life_loss_penalty == -1.0
        ),
        None,
    )
    if baseline_entry is None or strong_penalty_entry is None:
        raise ValueError("Stage 3 manifest must include penalty 0.0 and -1.0 variants")
    _baseline_variant, baseline, baseline_contract = baseline_entry
    _strong_variant, strong_penalty, strong_penalty_contract = strong_penalty_entry
    internal_differences = _differences(
        baseline,
        strong_penalty,
        excluded=NON_EXPERIMENT_FIELDS,
    )
    internal_manifest_consistency = all(
        isinstance(variant.get("config"), Mapping)
        and variant["config"] == config.to_dict()
        and variant.get("status") == "completed"
        and variant.get("training_seed_replication", config.seed) == config.seed
        for variant, config, _contract in stage3_variants
    ) and (
        stage3_manifest.get("training_seed") == stage3_config.seed
        and stage3_manifest.get("training_transitions") == stage3_config.total_steps
        and stage3_manifest.get("checkpoint_steps")
        == [
            stage3_config.checkpoint_interval * index
            for index in range(
                1,
                stage3_config.total_steps // stage3_config.checkpoint_interval + 1,
            )
        ]
    )
    internal_contract_equivalent = baseline_contract == strong_penalty_contract
    internal_comparison_valid = (
        not internal_differences
        and internal_manifest_consistency
        and internal_contract_equivalent
        and stage3_execution["all_completed"]
        and stage3_execution["parallel"]
        and stage3_execution["parallel_provenance_verified"]
    )
    resume = _resume_snapshot(resume_validation_path)

    differences: list[dict[str, Any]] = []
    for field, values in config_differences.items():
        differences.append(
            {
                "category": "config",
                "field": field,
                "stage2": values["stage2"],
                "stage3": values["stage3"],
                "impact": (
                    "different run budget/checkpoint cadence; it does not alter the "
                    "explicit per-step epsilon or optimizer schedule"
                    if field in {"total_steps", "checkpoint_interval"}
                    else "unexpected training-config drift"
                ),
            }
        )
    if stage2_execution["worker_count"] != stage3_execution["worker_count"]:
        differences.append(
            {
                "category": "execution",
                "field": "parallel_worker_population",
                "stage2": stage2_execution["worker_count"],
                "stage3": stage3_execution["worker_count"],
                "impact": (
                    "separately scheduled CUDA workers can have different numerical "
                    "nondeterminism; do not treat the two 250k checkpoints as an exact "
                    "replication"
                ),
            }
        )
    if resume.get("exact_continuation") is not True:
        differences.append(
            {
                "category": "checkpoint",
                "field": "stage3_training_origin",
                "stage2": "available 250k checkpoint",
                "stage3": "from_scratch",
                "impact": "legacy Stage 2 checkpoint lacks replay and environment/ALE state",
            }
        )

    return {
        "schema_version": 1,
        "artifact_type": "issue9_reward_shaping_stage3b_parity_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stage2": {
            "config_path": stage2_config_path.as_posix(),
            "manifest_path": stage2_manifest_path.as_posix(),
            "training_seed": stage2_config.seed,
            "total_steps": stage2_config.total_steps,
            "checkpoint_steps": [
                stage2_config.checkpoint_interval * index
                for index in range(
                    1,
                    stage2_config.total_steps // stage2_config.checkpoint_interval + 1,
                )
            ],
            "execution": stage2_execution,
            "contract": stage2_contract,
        },
        "stage3": {
            "config_path": stage3_config_path.as_posix(),
            "manifest_path": stage3_manifest_path.as_posix(),
            "training_seed": stage3_config.seed,
            "total_steps": stage3_config.total_steps,
            "checkpoint_steps": [
                stage3_config.checkpoint_interval * index
                for index in range(
                    1,
                    stage3_config.total_steps // stage3_config.checkpoint_interval + 1,
                )
            ],
            "execution": stage3_execution,
            "contract": stage3_contract,
        },
        "normalized_training_config_differences": config_differences,
        "unexpected_control_condition_differences": unexpected_condition_differences,
        "schedule_audit": {
            "stage2_vs_stage3_250k_schedule_equivalent": schedule_equivalent,
            "schedule_scope": (
                "epsilon, optimizer learning rate, update cadence, and controlled "
                "environment conditions"
            ),
            "learning_schedule_equivalent": learning_schedule_equivalent,
            "epsilon_schedule_equivalent": epsilon_schedule_equivalent,
            "epsilon_schedule": {
                "stage2": _schedule_snapshot(stage2_config),
                "stage3": _schedule_snapshot(stage3_config),
            },
            "learning_rate_schedule_equivalent": learning_rate_schedule_equivalent,
            "update_schedule_equivalent": update_schedule_equivalent,
            "total_steps_dependent_annealing": not total_steps_not_used_for_schedule,
            "environment_contract_equivalent": environment_equivalent,
            "controlled_conditions_equivalent": controlled_conditions_equivalent,
            "seed_propagation_equivalent": stage2_config.seed == stage3_config.seed,
            "optimizer": "torch.optim.Adam",
            "source_checks": source_checks,
        },
        "full_runtime_reproduction_equivalent": False,
        "checkpoint_semantics": {
            "same_checkpoint_interval": stage2_config.checkpoint_interval
            == stage3_config.checkpoint_interval,
            "exact_resume_available": resume.get("exact_continuation") is True,
            "resume_validation": resume,
        },
        "differences": differences,
        "does_this_invalidate_stage3_internal_baseline_vs_minus1": not internal_comparison_valid,
        "stage3_internal_comparison": {
            "controlled_comparison_valid": internal_comparison_valid,
            "baseline_vs_minus1_config_differences": internal_differences,
            "baseline_contract": baseline_contract,
            "minus1_contract": strong_penalty_contract,
            "contract_equivalent": internal_contract_equivalent,
            "manifest_consistency_valid": internal_manifest_consistency,
            "reason": (
                "All Stage 3 variants were launched in the same parallel sweep with "
                "the same seed, budget, checkpoint cadence, environment contract, "
                "optimizer, replay, and epsilon schedule; only life_loss_penalty differs."
            ),
        },
        "recommended_action": (
            "Proceed with Stage 3B seed 2023 and 2024 baseline/-1 replication. "
            "Do not treat Stage 2@250k versus Stage 3@250k as an exact reproduction, "
            "and do not start 2.5M before replication is summarized."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-config", type=Path, default=DEFAULT_STAGE2_CONFIG)
    parser.add_argument("--stage3-config", type=Path, default=DEFAULT_STAGE3_CONFIG)
    parser.add_argument("--stage2-manifest", type=Path, default=DEFAULT_STAGE2_MANIFEST)
    parser.add_argument("--stage3-manifest", type=Path, default=DEFAULT_STAGE3_MANIFEST)
    parser.add_argument(
        "--resume-validation",
        type=Path,
        default=DEFAULT_RESUME_VALIDATION,
    )
    parser.add_argument(
        "--stage2-execution-mode",
        choices=("parallel", "sequential"),
        default="parallel",
    )
    parser.add_argument(
        "--stage3-execution-mode",
        choices=("parallel", "sequential"),
        default="parallel",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/issue-9-reward-shaping/stage3b-multiseed/parity-audit.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = build_parity_audit(
        stage2_config_path=args.stage2_config,
        stage3_config_path=args.stage3_config,
        stage2_manifest_path=args.stage2_manifest,
        stage3_manifest_path=args.stage3_manifest,
        resume_validation_path=args.resume_validation,
        stage2_parallel=args.stage2_execution_mode == "parallel",
        stage3_parallel=args.stage3_execution_mode == "parallel",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
