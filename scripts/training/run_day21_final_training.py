"""Run Day 21 final long training with explicit milestone gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from breakout_env import make_breakout_env, make_breakout_vector_env
from breakout_rl.day21_final_training import (
    DAY21_MAX_STAGE_B_CANDIDATES,
    DAY21_MAX_STAGE_C_CANDIDATES,
    DAY21_STAGE_ORDER,
    Day21FinalTrainingConfig,
    assess_evaluation_contract_health,
    assess_training_health,
    build_day21_manifest,
    build_day21_report,
    compact_metrics,
    load_day21_config,
    manifest_reference,
    relative_path,
    render_day21_markdown,
    select_extension_candidates,
    select_final_checkpoint,
    sha256_file,
    utc_timestamp,
    validate_day21_manifest,
    write_json,
)
from breakout_rl.evaluation import (
    evaluate_policy,
    load_dqn_checkpoint,
    write_evaluation_artifacts,
)
from breakout_rl.evaluation_contract import breakout_environment_kwargs
from breakout_rl.training.dqn_trainer import resolve_device
from breakout_rl.training.vectorized import VectorizedDQNTrainer
from scripts.visualization.record_checkpoint_gameplay import record_checkpoint


_CHECKPOINT_RE = re.compile(r"^step-(\d{8})$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Day 21 final long training through the 5M transition target."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final-training/manifest.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day21-final-long-training/manifest.json"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--evaluations-root", type=Path, default=Path("evaluations"))
    parser.add_argument("--evidence-root", type=Path, default=Path("assets/day21"))
    parser.add_argument(
        "--stage",
        choices=("plan", "all"),
        default="all",
        help="plan only or execute 1M -> 2.5M -> 5M gates",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing manifest/checkpoint after an interruption",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = dict(manifest)
    payload["updated_at_utc"] = utc_timestamp()
    write_json(path, payload)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in (run_dir / "checkpoints").glob("step-*.pt")
        if _CHECKPOINT_RE.fullmatch(path.stem)
    ]
    return max(checkpoints, key=_checkpoint_step) if checkpoints else None


def _checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"not a standard Day 21 checkpoint: {path}")
    return int(match.group(1))


def _checkpoint_reference(path: Path, *, manifest_path: Path) -> dict[str, Any]:
    return {
        "path": manifest_reference(path, start=manifest_path.parent),
        "sha256": sha256_file(path),
        "step": _checkpoint_step(path),
    }


def _resolve_manifest_reference(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest reference must be a non-empty path")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def _prepare_run_directory(
    manifest_path: Path,
    entry: dict[str, Any],
) -> tuple[Path, Path | None]:
    """Reuse a checkpoint or preserve an incomplete no-checkpoint attempt."""

    run_dir = _resolve_manifest_reference(manifest_path, entry["run_dir"])
    latest = _latest_checkpoint(run_dir)
    if latest is not None:
        return run_dir, latest
    if not run_dir.exists():
        return run_dir, None

    attempts = entry.setdefault("attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("Day 21 entry attempts must be an array")
    attempts.append(
        {
            "run_dir": manifest_reference(run_dir, start=manifest_path.parent),
            "status": entry.get("status"),
            "error": entry.get("error"),
            "reason": "preserved incomplete attempt with no standard checkpoint",
        }
    )
    attempt_number = len(attempts)
    retry_dir = run_dir.with_name(f"{run_dir.name}-retry-{attempt_number:02d}")
    while retry_dir.exists():
        attempt_number += 1
        retry_dir = run_dir.with_name(f"{run_dir.name}-retry-{attempt_number:02d}")
    entry["run_dir"] = manifest_reference(retry_dir, start=manifest_path.parent)
    stages = entry.get("stages")
    if isinstance(stages, dict):
        for record in stages.values():
            if not isinstance(record, dict):
                continue
            training = record.get("training")
            if isinstance(training, dict):
                training["metrics_path"] = manifest_reference(
                    retry_dir / "metrics.csv",
                    start=manifest_path.parent,
                )
    return retry_dir, None


def _git_command(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_provenance(repository_root: Path) -> dict[str, Any]:
    status = _git_command(repository_root, "status", "--porcelain")
    diff = _git_command(repository_root, "diff", "--binary")
    return {
        "git_commit_sha": _git_command(repository_root, "rev-parse", "HEAD"),
        "git_branch": _git_command(repository_root, "branch", "--show-current"),
        "git_dirty": bool(status),
        "git_status_snapshot": status,
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "git_diff_scope": "tracked files changed relative to HEAD; untracked files are listed separately in status",
    }


def _check_cuda_headroom(config: Day21FinalTrainingConfig) -> torch.device:
    device = resolve_device(config.requested_device)
    if device.type != "cuda":
        raise RuntimeError("Day 21 formal training resolved to a non-CUDA device")
    index = 0 if device.index is None else int(device.index)
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError("unable to measure CUDA free memory for Day 21 headroom gate") from error
    if int(free_bytes) < config.cuda_headroom_bytes:
        raise RuntimeError(
            "CUDA free memory is below the required Day 21 headroom: "
            f"free={int(free_bytes)} bytes, required={config.cuda_headroom_bytes} bytes"
        )
    return device


def _capture_torch_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Capture one trainer's torch RNG stream for interleaved sessions."""

    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    return torch.get_rng_state().clone(), cuda_states


def _restore_torch_rng_state(
    state: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> None:
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def _with_trainer_rng(
    states: dict[int, tuple[torch.Tensor, list[torch.Tensor] | None]],
    seed: int,
    callback: Any,
) -> Any:
    """Run one trainer with its own torch stream, restoring the caller stream."""

    caller_state = _capture_torch_rng_state()
    _restore_torch_rng_state(states[seed])
    try:
        return callback()
    finally:
        states[seed] = _capture_torch_rng_state()
        _restore_torch_rng_state(caller_state)


def _entry_for_seed(manifest: Mapping[str, Any], seed: int) -> dict[str, Any]:
    for entry in manifest.get("runs", []):
        if isinstance(entry, dict) and int(entry.get("training_seed", -1)) == int(seed):
            return entry
    raise ValueError(f"Day 21 manifest has no entry for training seed {seed}")


def _refresh_source_provenance(
    config: Day21FinalTrainingConfig,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    """Record code changes explicitly before continuing an interrupted run."""

    current = _source_provenance(config.repository_root)
    previous = manifest.get("source_provenance")
    if not isinstance(previous, Mapping):
        manifest["source_provenance"] = current
        return
    if previous.get("git_commit_sha") == current.get("git_commit_sha"):
        return
    has_checkpoint = any(
        _latest_checkpoint(
            _resolve_manifest_reference(manifest_path, entry["run_dir"])
        ) is not None
        for entry in manifest.get("runs", [])
        if isinstance(entry, Mapping) and isinstance(entry.get("run_dir"), str)
    )
    if has_checkpoint:
        raise RuntimeError(
            "Day 21 source commit changed while checkpoints exist; create a new "
            "manifest instead of resuming with mixed training code"
        )
    history = manifest.setdefault("source_provenance_history", [])
    if isinstance(history, list):
        history.append(dict(previous))
    manifest["source_provenance"] = current


def _stage_record(entry: Mapping[str, Any], stage: str) -> dict[str, Any]:
    stages = entry.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(stages.get(stage), dict):
        raise ValueError(f"Day 21 entry is missing stage {stage}")
    return stages[stage]


def _selection_record(
    entry: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    record = dict(_stage_record(entry, stage))
    record["training_seed"] = int(entry["training_seed"])
    record["run_id"] = entry.get("run_id")
    record["stage"] = stage
    return record


def _stage_mean_return(record: Mapping[str, Any], *, label: str) -> float:
    evaluation = record.get("evaluation")
    summary = evaluation.get("summary") if isinstance(evaluation, Mapping) else None
    value = summary.get("mean_return") if isinstance(summary, Mapping) else None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Day 21 {label} is missing a finite mean return") from error
    if not math.isfinite(parsed):
        raise RuntimeError(f"Day 21 {label} is missing a finite mean return")
    return parsed


def _stage_c_trigger_provenance(
    config: Day21FinalTrainingConfig,
    *,
    stage_a_records: Sequence[Mapping[str, Any]],
    stage_b_records: Sequence[Mapping[str, Any]],
    selected_b_seeds: Sequence[int],
) -> dict[str, Any]:
    policy = config.raw.get("execution", {}).get("stage_c_policy", {})
    if not isinstance(policy, Mapping):
        raise RuntimeError("Day 21 Stage C policy is missing")
    configured_evidence = policy.get("trigger_evidence")
    if not isinstance(configured_evidence, Mapping):
        raise RuntimeError("Day 21 Stage C trigger evidence is missing")
    if not selected_b_seeds:
        raise RuntimeError("Day 21 Stage C has no selected training seed")
    trigger_seed = int(configured_evidence.get("training_seed", -1))
    if trigger_seed not in {int(seed) for seed in selected_b_seeds}:
        raise RuntimeError(
            "Day 21 Stage C trigger evidence must describe the selected 2.5M seed"
        )
    stage_a = next(
        (
            record
            for record in stage_a_records
            if int(record.get("training_seed", -1)) == trigger_seed
        ),
        None,
    )
    stage_b = next(
        (
            record
            for record in stage_b_records
            if int(record.get("training_seed", -1)) == trigger_seed
        ),
        None,
    )
    if stage_a is None or stage_b is None:
        raise RuntimeError("Day 21 Stage C trigger evidence has no matching stage records")
    one_million_mean = _stage_mean_return(stage_a, label="1M trigger evidence")
    two_point_five_million_mean = _stage_mean_return(
        stage_b,
        label="2.5M trigger evidence",
    )
    configured_one_million = float(configured_evidence["stage_a_1m_mean_return"])
    configured_two_point_five_million = float(
        configured_evidence["stage_b_2_5m_mean_return"]
    )
    if not math.isclose(
        one_million_mean,
        configured_one_million,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Day 21 1M trigger evidence does not match the completed record")
    if not math.isclose(
        two_point_five_million_mean,
        configured_two_point_five_million,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Day 21 2.5M trigger evidence does not match the completed record")
    return {
        "primary_trigger": str(policy["primary_trigger"]),
        "trigger_evidence": {
            "training_seed": trigger_seed,
            "stage_a_1m_mean_return": one_million_mean,
            "stage_b_2_5m_mean_return": two_point_five_million_mean,
            "mean_return_improvement": two_point_five_million_mean - one_million_mean,
        },
        "user_requested_5m": bool(policy["user_requested_5m"]),
        "request_is_supplemental_provenance": bool(
            policy["request_is_supplemental_provenance"]
        ),
    }


def _completed_stage(record: Mapping[str, Any]) -> bool:
    evaluation = record.get("evaluation")
    return bool(
        record.get("status") == "completed"
        and record.get("eligible") is True
        and isinstance(record.get("checkpoint"), Mapping)
        and isinstance(record.get("health"), Mapping)
        and record["health"].get("healthy") is True
        and isinstance(evaluation, Mapping)
        and evaluation.get("status") == "completed"
        and isinstance(evaluation.get("summary"), Mapping)
    )


def _compact_training_record(
    config: Day21FinalTrainingConfig,
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    stage: str,
    summary: Mapping[str, Any],
    resumed_from: Path | None,
) -> dict[str, Any]:
    record = _stage_record(entry, stage)
    run_dir = _resolve_manifest_reference(manifest_path, entry["run_dir"])
    checkpoint = _latest_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"Day 21 stage completed without a checkpoint: {run_dir}")
    metrics_path = run_dir / "metrics.csv"
    evidence_root = config.repository_root / "assets/day21/evidence-runs" / f"seed{entry['training_seed']}"
    target = int(record["target_transitions"])
    compact_path = evidence_root / f"metrics-{stage}.csv"
    compact_info = compact_metrics(
        metrics_path,
        compact_path,
        max_global_step=target,
    )
    summary_path = evidence_root / f"summary-{stage}.json"
    compact_summary = dict(summary)
    write_json(summary_path, compact_summary)
    health = assess_training_health(
        summary,
        metrics_path,
        expected_transitions=int(record["target_transitions"]),
        contract_id=config.contract.contract_id,
        health_rule=config.health_rule,
    )
    if resumed_from is None:
        resume_exact = True
        resume_semantics = "continuous_in_process"
    else:
        resume_provenance = summary.get("resume_provenance")
        replay_saved = (
            isinstance(resume_provenance, Mapping)
            and resume_provenance.get("replay_saved") is True
        )
        resume_exact = replay_saved
        resume_semantics = (
            "exact_replay_restore" if replay_saved else "fresh_replay_with_learning_starts_rewarm"
        )
    record["checkpoint"] = _checkpoint_reference(
        checkpoint,
        manifest_path=manifest_path,
    )
    record["training"] = {
        "metrics_path": manifest_reference(metrics_path, start=manifest_path.parent),
        "compact_metrics_path": manifest_reference(
            compact_path,
            start=manifest_path.parent,
        ),
        "summary_path": manifest_reference(summary_path, start=manifest_path.parent),
        "summary": compact_summary,
        "runtime": summary.get("runtime"),
        "resume_exact": resume_exact,
        "resume_semantics": resume_semantics,
        "resume_reason": (
            "continuous_in_process"
            if resumed_from is None
            else (
                "checkpoint_resume_replay_restored"
                if resume_exact
                else "checkpoint_resume_replay_not_restored"
            )
        ),
        "resumed_from": (
            None
            if resumed_from is None
            else _checkpoint_reference(resumed_from, manifest_path=manifest_path)
        ),
        "compact_metrics": {
            "sampling_interval": compact_info["sampling_interval"],
            "max_global_step": compact_info["max_global_step"],
            "source_rows": compact_info["source_rows"],
            "kept_rows": compact_info["kept_rows"],
            "sha256": compact_info["sha256"],
        },
    }
    record["health"] = health
    record["status"] = "completed" if summary.get("status") == "completed" else "failed"
    record["eligible"] = bool(health["healthy"])
    if not health["healthy"]:
        record["status"] = "failed_health_gate"
    return record


def _refresh_completed_stage_metrics(
    config: Day21FinalTrainingConfig,
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    stage: str,
) -> bool:
    """Repair or verify the immutable compact snapshot for one completed stage."""

    record = _stage_record(entry, stage)
    if record.get("status") != "completed":
        return False
    training = record.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"completed Day 21 stage {stage} is missing training metadata")
    run_dir = _resolve_manifest_reference(manifest_path, entry["run_dir"])
    source_path = run_dir / "metrics.csv"
    target = int(record["target_transitions"])
    destination = (
        config.repository_root
        / "assets/day21/evidence-runs"
        / f"seed{entry['training_seed']}"
        / f"metrics-{stage}.csv"
    )
    expected_reference = manifest_reference(destination, start=manifest_path.parent)
    compact_record = training.get("compact_metrics")
    if (
        destination.is_file()
        and training.get("compact_metrics_path") == expected_reference
        and isinstance(compact_record, Mapping)
        and compact_record.get("max_global_step") == target
        and compact_record.get("sha256") == sha256_file(destination)
    ):
        return False
    info = compact_metrics(
        source_path,
        destination,
        max_global_step=target,
    )
    training["compact_metrics_path"] = expected_reference
    training["compact_metrics"] = {
        "sampling_interval": info["sampling_interval"],
        "max_global_step": info["max_global_step"],
        "source_rows": info["source_rows"],
        "kept_rows": info["kept_rows"],
        "sha256": info["sha256"],
    }
    return True


def _repair_completed_manifest(
    config: Day21FinalTrainingConfig,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> bool:
    """Normalize completed manifests without retraining or reopening holdout."""

    changed = False
    decisions = manifest.get("selection_decisions")
    stage_a_decision = (
        decisions.get("stage_a_1m")
        if isinstance(decisions, Mapping)
        else None
    )
    selected_a = (
        {int(seed) for seed in stage_a_decision.get("selected_training_seeds", [])}
        if isinstance(stage_a_decision, Mapping)
        else set()
    )
    if selected_a:
        for entry in manifest.get("runs", []):
            if not isinstance(entry, dict):
                continue
            seed = int(entry.get("training_seed", -1))
            if seed not in selected_a:
                before_b = _stage_record(entry, "stage_b_2_5m").get("status")
                before_c = _stage_record(entry, "stage_c_5m").get("status")
                _mark_not_selected(
                    entry,
                    "stage_b_2_5m",
                    reason="Stage A aggregate selection did not place this seed in the top two",
                )
                _mark_not_selected(
                    entry,
                    "stage_c_5m",
                    reason="Stage A aggregate selection did not place this seed in the top two",
                )
                changed = changed or before_b != _stage_record(entry, "stage_b_2_5m").get("status")
                changed = changed or before_c != _stage_record(entry, "stage_c_5m").get("status")
    for entry in manifest.get("runs", []):
        if not isinstance(entry, dict):
            continue
        for stage_index, stage in enumerate(DAY21_STAGE_ORDER):
            changed = (
                _refresh_completed_stage_metrics(
                    config,
                    manifest_path=manifest_path,
                    entry=entry,
                    stage=stage,
                )
                or changed
            )
            record = _stage_record(entry, stage)
            training = record.get("training")
            if (
                _completed_stage(record)
                and isinstance(training, dict)
                and training.get("resume_exact") is False
                and not training.get("resume_reason")
            ):
                training["resume_reason"] = "checkpoint_resume_replay_not_restored"
                changed = True
            if stage_index == 0:
                continue
            previous = _stage_record(entry, DAY21_STAGE_ORDER[stage_index - 1])
            resumed_from = training.get("resumed_from") if isinstance(training, Mapping) else None
            if (
                _completed_stage(record)
                and _completed_stage(previous)
                and isinstance(training, dict)
                and isinstance(resumed_from, Mapping)
            ):
                try:
                    resumed_step = int(resumed_from.get("step", -1))
                    previous_target = int(previous["target_transitions"])
                except (TypeError, ValueError, KeyError):
                    resumed_step = previous_target = -1
                if 0 <= resumed_step < previous_target:
                    training["resumed_from"] = None
                    training["resume_exact"] = True
                    training["resume_semantics"] = "continuous_in_process_after_previous_stage"
                    training["resume_reason"] = "continuous_in_process_after_previous_stage"
                    changed = True

    holdout = manifest.get("final_holdout")
    if isinstance(holdout, dict) and holdout.get("status") == "completed":
        results_reference = holdout.get("results")
        if isinstance(results_reference, str):
            results_path = _resolve_manifest_reference(manifest_path, results_reference)
            if results_path.is_file():
                result_payload = json.loads(results_path.read_text(encoding="utf-8"))
                contract_health = assess_evaluation_contract_health(
                    result_payload,
                    contract_id=config.contract.contract_id,
                    expected_seeds=config.holdout_group_seeds,
                    episodes_per_seed=config.holdout_episodes_per_group,
                    expected_concrete_seeds=config.holdout_concrete_seeds,
                    requested_device=config.requested_device,
                )
                if holdout.get("contract_health") != contract_health:
                    holdout["contract_health"] = contract_health
                    changed = True
                if not contract_health["healthy"]:
                    holdout["status"] = "failed_contract_gate"
                    raise RuntimeError(
                        "Day 21 final holdout Contract v2 gate failed during repair: "
                        + ", ".join(contract_health["failures"])
                    )
                final_model = manifest.get("canonical_final_model")
                if isinstance(final_model, dict):
                    metadata_reference = final_model.get("metadata_path")
                    if isinstance(metadata_reference, str):
                        metadata_path = _resolve_manifest_reference(
                            manifest_path,
                            metadata_reference,
                        )
                        if metadata_path.is_file():
                            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                            final_holdout_metadata = metadata.get("final_holdout")
                            if not isinstance(final_holdout_metadata, dict):
                                final_holdout_metadata = {}
                            if final_holdout_metadata.get("contract_health") != contract_health:
                                final_holdout_metadata["contract_health"] = contract_health
                                metadata["final_holdout"] = final_holdout_metadata
                                write_json(metadata_path, metadata)
                                final_model["metadata_sha256"] = sha256_file(metadata_path)
                                changed = True
    if isinstance(decisions, dict):
        stage_a_records = [
            _selection_record(entry, "stage_a_1m")
            for entry in manifest.get("runs", [])
            if isinstance(entry, Mapping)
        ]
        stage_b_records = [
            _selection_record(entry, "stage_b_2_5m")
            for entry in manifest.get("runs", [])
            if isinstance(entry, Mapping)
            and _stage_record(entry, "stage_b_2_5m").get("status") == "completed"
        ]
        stage_b_decision = decisions.get("stage_b_2_5m")
        selected_b_seeds = (
            [int(seed) for seed in stage_b_decision.get("selected_training_seeds", [])]
            if isinstance(stage_b_decision, Mapping)
            else []
        )
        if selected_b_seeds and isinstance(decisions.get("stage_c_5m"), dict):
            trigger = _stage_c_trigger_provenance(
                config,
                stage_a_records=stage_a_records,
                stage_b_records=stage_b_records,
                selected_b_seeds=selected_b_seeds,
            )
            stage_c_decision = decisions["stage_c_5m"]
            for key, value in trigger.items():
                if stage_c_decision.get(key) != value:
                    stage_c_decision[key] = value
                    changed = True
            if stage_c_decision.get("reason") != trigger["primary_trigger"]:
                stage_c_decision["reason"] = trigger["primary_trigger"]
                changed = True
            for obsolete_key in ("explicit_user_target_override", "override_reason"):
                if obsolete_key in stage_c_decision:
                    stage_c_decision.pop(obsolete_key)
                    changed = True
    return changed


def _run_selection_evaluation(
    config: Day21FinalTrainingConfig,
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    stage: str,
    checkpoint: Path,
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    record = _stage_record(entry, stage)
    evaluation = record["evaluation"]
    if not isinstance(evaluation, dict):
        raise ValueError("Day 21 evaluation record must be an object")
    env_factory = lambda: make_breakout_env(**breakout_environment_kwargs(config.contract))
    loaded = load_dqn_checkpoint(
        checkpoint,
        device=config.requested_device,
        env_factory=env_factory,
    )
    try:
        source_contract = loaded.checkpoint_metadata.get("environment_contract")
        if isinstance(source_contract, Mapping):
            if (
                source_contract.get("contract_id") != config.contract.contract_id
                or source_contract.get("contract_sha256") != sha256_file(config.contract_path)
            ):
                raise RuntimeError("Day 21 checkpoint Contract v2 provenance does not match")
        result = evaluate_policy(
            loaded.model,
            episodes=config.selection_episodes_per_seed,
            seeds=config.selection_seeds,
            device=config.requested_device,
            epsilon=0.0,
            model_id=loaded.model_id,
            training_metadata={
                **dict(loaded.training_metadata),
                "source_day21_experiment_id": config.experiment_id,
                "source_day21_run_id": entry["run_id"],
                "training_seed": int(entry["training_seed"]),
                "training_stage": stage,
                "training_budget": int(record["target_transitions"]),
                "trainer_summary": dict(training_summary),
                "evaluation_phase": "selection",
                "environment_contract": config.contract_provenance,
            },
            checkpoint_metadata={
                **dict(loaded.checkpoint_metadata),
                "source_day21_experiment_id": config.experiment_id,
                "source_day21_run_id": entry["run_id"],
                "training_stage": stage,
                "evaluation_contract": config.contract_provenance,
            },
            evaluation_id=(
                f"{config.experiment_id}-seed{entry['training_seed']}-{stage}"
            ),
            env_factory=env_factory,
            metadata={
                "evaluation_phase": "selection",
                "evaluation_config_path": relative_path(
                    config.evaluation_config_path,
                    start=config.repository_root,
                ),
                "evaluation_contract_path": relative_path(
                    config.contract_path,
                    start=config.repository_root,
                ),
                "evaluation_contract": config.contract.to_dict(),
                "evaluation_contract_provenance": config.contract_provenance,
                "source_day21_manifest": relative_path(
                    manifest_path,
                    start=config.repository_root,
                ),
                "selection_concrete_episode_seeds": list(config.selection_concrete_seeds),
                "raw_reward": True,
            },
        )
    finally:
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result_payload = result.to_dict()
    contract_health = assess_evaluation_contract_health(
        result_payload,
        contract_id=config.contract.contract_id,
        expected_seeds=config.selection_seeds,
        episodes_per_seed=config.selection_episodes_per_seed,
        expected_concrete_seeds=config.selection_concrete_seeds,
        requested_device=config.requested_device,
    )
    output_dir = _resolve_manifest_reference(manifest_path, evaluation["directory"])
    results_path, episodes_path = write_evaluation_artifacts(result, output_dir)
    evaluation.update(
        {
            "results": manifest_reference(results_path, start=manifest_path.parent),
            "episodes": manifest_reference(episodes_path, start=manifest_path.parent),
            "results_sha256": sha256_file(results_path),
            "episodes_sha256": sha256_file(episodes_path),
            "summary": result_payload["summary"],
            "contract_health": contract_health,
            "status": "completed" if contract_health["healthy"] else "failed_contract_gate",
            "phase": "selection",
        }
    )
    if not contract_health["healthy"]:
        record["status"] = "failed_contract_gate"
        record["eligible"] = False
        raise RuntimeError(
            f"Day 21 evaluation Contract v2 gate failed for seed {entry['training_seed']} at {stage}: "
            f"{contract_health['failures']}"
        )
    return result_payload


def _record_stage_gameplay(
    config: Day21FinalTrainingConfig,
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    stage: str,
    checkpoint: Path,
) -> dict[str, Any]:
    """Capture one qualitative gameplay artifact under the same Contract v2."""

    seed = int(entry["training_seed"])
    output_dir = config.repository_root / "assets/day21/gameplay" / f"seed{seed}"
    output_path = output_dir / f"{stage}.gif"
    metadata_path = output_dir / f"{stage}.json"
    if output_path.is_file() and metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = record_checkpoint(
        checkpoint,
        output=output_path,
        metadata_path=metadata_path,
        device=config.requested_device,
        evaluation_seed=config.selection_seeds[0],
        episodes=1,
        max_steps=1_000,
        record_every=8,
        fps=10,
        max_width=240,
        contract_path=config.contract_path,
    )
    metadata["day21"] = {
        "experiment_id": config.experiment_id,
        "run_id": entry["run_id"],
        "training_seed": seed,
        "stage": stage,
        "selection_evaluation_seed": config.selection_seeds[0],
        "contract_id": config.contract.contract_id,
        "contract_sha256": sha256_file(config.contract_path),
        "manifest": relative_path(manifest_path, start=config.repository_root),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def _record_and_evaluate_stage(
    config: Day21FinalTrainingConfig,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    entry: dict[str, Any],
    stage: str,
    trainer: VectorizedDQNTrainer,
    resumed_from: Path | None,
    session_rng_states: dict[int, tuple[torch.Tensor, list[torch.Tensor] | None]],
) -> None:
    record = _stage_record(entry, stage)
    if _completed_stage(record):
        _refresh_completed_stage_metrics(
            config,
            manifest_path=manifest_path,
            entry=entry,
            stage=stage,
        )
        checkpoint_reference = record.get("checkpoint")
        if isinstance(checkpoint_reference, Mapping):
            checkpoint = _resolve_manifest_reference(manifest_path, checkpoint_reference["path"])
            if not isinstance(record.get("gameplay"), Mapping):
                record["gameplay"] = _record_stage_gameplay(
                    config,
                    manifest_path=manifest_path,
                    entry=entry,
                    stage=stage,
                    checkpoint=checkpoint,
                )
        return
    target = int(record["target_transitions"])
    if trainer.global_step > target:
        raise RuntimeError(
            f"trainer for seed {entry['training_seed']} is already at {trainer.global_step}, "
            f"past pending milestone {target}"
        )
    summary = _with_trainer_rng(
        session_rng_states,
        int(entry["training_seed"]),
        lambda: trainer.train_until(target, close=False),
    )
    _compact_training_record(
        config,
        manifest_path=manifest_path,
        entry=entry,
        stage=stage,
        summary=summary,
        resumed_from=resumed_from,
    )
    record = _stage_record(entry, stage)
    if not bool(record.get("eligible")):
        _write_manifest(manifest_path, manifest)
        raise RuntimeError(
            f"Day 21 health gate failed for seed {entry['training_seed']} at {stage}: "
            f"{record.get('health', {}).get('failures', [])}"
        )
    checkpoint_reference = record.get("checkpoint")
    if not isinstance(checkpoint_reference, Mapping):
        raise ValueError("Day 21 completed stage is missing checkpoint reference")
    checkpoint = _resolve_manifest_reference(manifest_path, checkpoint_reference["path"])
    _run_selection_evaluation(
        config,
        manifest_path=manifest_path,
        entry=entry,
        stage=stage,
        checkpoint=checkpoint,
        training_summary=summary,
    )
    record["gameplay"] = _record_stage_gameplay(
        config,
        manifest_path=manifest_path,
        entry=entry,
        stage=stage,
        checkpoint=checkpoint,
    )
    _write_manifest(manifest_path, manifest)


def _decision_payload(
    *,
    stage: str,
    records: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for record in records:
        evaluation = record.get("evaluation")
        summary = evaluation.get("summary", {}) if isinstance(evaluation, Mapping) else {}
        values.append(
            {
                "training_seed": record.get("training_seed"),
                "mean_return": summary.get("mean_return") if isinstance(summary, Mapping) else None,
                "median_return": summary.get("median_return") if isinstance(summary, Mapping) else None,
                "std_return": summary.get("std_return") if isinstance(summary, Mapping) else None,
                "count": summary.get("count") if isinstance(summary, Mapping) else None,
                "healthy": record.get("health", {}).get("healthy")
                if isinstance(record.get("health"), Mapping)
                else False,
            }
        )
    return {
        "status": "complete",
        "stage": stage,
        "candidate_training_seeds": [int(item["training_seed"]) for item in records],
        "selected_training_seeds": [int(item["training_seed"]) for item in selected],
        "aggregate_values": values,
        "selection_metric": rule.get("primary_metric"),
        "rule": dict(rule),
        "reason": reason,
    }


def _mark_not_selected(entry: dict[str, Any], stage: str, *, reason: str) -> None:
    record = _stage_record(entry, stage)
    if record.get("status") == "pending":
        record["status"] = "not_selected"
        record["selection_reason"] = reason
        evaluation = record.get("evaluation")
        if isinstance(evaluation, dict):
            evaluation["status"] = "not_run"


def _freeze_canonical_model(
    config: Day21FinalTrainingConfig,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_reference = candidate.get("checkpoint")
    if not isinstance(checkpoint_reference, Mapping):
        raise ValueError("final candidate is missing a checkpoint reference")
    checkpoint_path = _resolve_manifest_reference(manifest_path, checkpoint_reference["path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("online_network"), Mapping):
        raise ValueError("final checkpoint does not contain an online_network state dict")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("final checkpoint is missing model_config")
    stage = str(candidate["stage"])
    seed = int(candidate["training_seed"])
    training = candidate.get("training")
    training_summary = training.get("summary", {}) if isinstance(training, Mapping) else {}
    runtime = training.get("runtime", {}) if isinstance(training, Mapping) else {}
    if not isinstance(runtime, Mapping):
        runtime = {}
    source_provenance = manifest.get("source_provenance", {})
    if not isinstance(source_provenance, Mapping):
        source_provenance = {}
    model_path = config.repository_root / "assets/day21/models/final_model/model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_payload = {
        "format_version": 3,
        "trainer": "canonical_final_model",
        "run_id": f"{config.experiment_id}-canonical",
        "algorithm": config.algorithm,
        "architecture": config.architecture,
        "device": runtime.get("resolved_device", "cuda:0"),
        "requested_device": config.requested_device,
        "contract_id": config.contract.contract_id,
        "contract_path": relative_path(config.contract_path, start=config.repository_root),
        "global_step": int(candidate["target_transitions"]),
        "training_steps": int(candidate["target_transitions"]),
        "model_config": dict(model_config),
        "online_network": dict(payload["online_network"]),
        "config": config.training_config(seed).to_dict(),
        "environment_contract": config.contract_provenance,
        "metadata": {
            "canonical_final_model": True,
            "source_day21_run_id": candidate.get("run_id"),
            "source_checkpoint": checkpoint_reference,
            "selection_evaluation": candidate.get("evaluation", {}).get("summary")
            if isinstance(candidate.get("evaluation"), Mapping)
            else None,
        },
    }
    temporary = model_path.with_suffix(".tmp")
    torch.save(canonical_payload, temporary)
    os.replace(temporary, model_path)
    model_sha256 = sha256_file(model_path)
    training_config_json = json.dumps(
        config.training_config(seed).to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    metadata = {
        "schema_version": 1,
        "artifact_type": "canonical_final_model",
        "model_path": relative_path(model_path, start=config.repository_root),
        "model_sha256": model_sha256,
        "algorithm": config.algorithm,
        "architecture": config.architecture,
        "hidden_dim": model_config.get("hidden_dim"),
        "observation_shape": model_config.get("input_shape"),
        "num_actions": model_config.get("num_actions"),
        "parameter_count": model_config.get("parameter_count"),
        "training_seed": seed,
        "training_transitions": int(candidate["target_transitions"]),
        "source_run_id": candidate.get("run_id"),
        "source_stage": stage,
        "source_checkpoint": {
            **dict(checkpoint_reference),
            "absolute_path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "training_config_hash": hashlib.sha256(
            training_config_json.encode("utf-8")
        ).hexdigest(),
        "training_config": config.training_config(seed).to_dict(),
        "contract_id": config.contract.contract_id,
        "contract_path": relative_path(config.contract_path, start=config.repository_root),
        "contract_sha256": sha256_file(config.contract_path),
        "systems_backend": config.backend_id,
        "runtime": dict(runtime),
        "selection_evaluation": candidate.get("evaluation", {}).get("summary")
        if isinstance(candidate.get("evaluation"), Mapping)
        else None,
        "final_holdout": None,
        "source_provenance": dict(source_provenance),
        "frozen_at_utc": utc_timestamp(),
    }
    metadata_path = model_path.parent / "metadata.json"
    write_json(metadata_path, metadata)
    final_model_record = {
        **metadata,
        "metadata_path": relative_path(metadata_path, start=config.repository_root),
        "metadata_sha256": sha256_file(metadata_path),
    }
    manifest["canonical_final_model"] = final_model_record
    manifest["selection_decisions"]["final_checkpoint"] = {
        "status": "frozen",
        "selected": {
            "training_seed": seed,
            "stage": stage,
            "target_transitions": int(candidate["target_transitions"]),
            "mean_return": candidate.get("evaluation", {}).get("summary", {}).get("mean_return")
            if isinstance(candidate.get("evaluation"), Mapping)
            and isinstance(candidate["evaluation"].get("summary"), Mapping)
            else None,
        },
        "rule": dict(config.selection_rule),
        "holdout_was_locked": True,
    }
    return final_model_record


def _run_final_holdout(
    config: Day21FinalTrainingConfig,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    holdout = manifest.get("final_holdout")
    if not isinstance(holdout, dict):
        raise ValueError("Day 21 manifest is missing final_holdout")
    if holdout.get("status") == "completed":
        return dict(holdout)
    final_model = manifest.get("canonical_final_model")
    if not isinstance(final_model, Mapping):
        raise ValueError("final holdout cannot run before canonical model freeze")
    model_path = config.repository_root / str(final_model["model_path"])
    env_factory = lambda: make_breakout_env(**breakout_environment_kwargs(config.contract))
    loaded = load_dqn_checkpoint(
        model_path,
        device=config.requested_device,
        env_factory=env_factory,
    )
    try:
        result = evaluate_policy(
            loaded.model,
            episodes=config.holdout_episodes_per_group,
            seeds=config.holdout_group_seeds,
            device=config.requested_device,
            epsilon=0.0,
            model_id=loaded.model_id,
            training_metadata={
                **dict(loaded.training_metadata),
                "source_day21_experiment_id": config.experiment_id,
                "evaluation_phase": "final_holdout",
                "holdout_opened_after_final_freeze": True,
                "holdout_concrete_episode_seeds": list(config.holdout_concrete_seeds),
                "environment_contract": config.contract_provenance,
            },
            checkpoint_metadata={
                **dict(loaded.checkpoint_metadata),
                "source_day21_experiment_id": config.experiment_id,
                "evaluation_phase": "final_holdout",
                "evaluation_contract": config.contract_provenance,
            },
            evaluation_id=f"{config.experiment_id}-final-holdout",
            env_factory=env_factory,
            metadata={
                "evaluation_phase": "final_holdout",
                "opened_after_final_freeze": True,
                "evaluation_order": "selection → final freeze → final holdout",
                "holdout_group_seeds": list(config.holdout_group_seeds),
                "holdout_concrete_episode_seeds": list(config.holdout_concrete_seeds),
                "evaluation_contract": config.contract.to_dict(),
                "evaluation_contract_provenance": config.contract_provenance,
                "raw_reward": True,
            },
        )
    finally:
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result_payload = result.to_dict()
    contract_health = assess_evaluation_contract_health(
        result_payload,
        contract_id=config.contract.contract_id,
        expected_seeds=config.holdout_group_seeds,
        episodes_per_seed=config.holdout_episodes_per_group,
        expected_concrete_seeds=config.holdout_concrete_seeds,
        requested_device=config.requested_device,
    )
    output_dir = config.repository_root / "evaluations/day21-final-long-training/final-holdout"
    results_path, episodes_path = write_evaluation_artifacts(result, output_dir)
    holdout.update(
        {
            "status": "completed" if contract_health["healthy"] else "failed_contract_gate",
            "results": manifest_reference(results_path, start=manifest_path.parent),
            "episodes": manifest_reference(episodes_path, start=manifest_path.parent),
            "results_sha256": sha256_file(results_path),
            "episodes_sha256": sha256_file(episodes_path),
            "summary": result_payload["summary"],
            "contract_health": contract_health,
            "opened_after_final_freeze": True,
            "evaluation_order": "selection → final freeze → final holdout",
            "model_sha256": final_model.get("model_sha256"),
        }
    )
    if not contract_health["healthy"]:
        raise RuntimeError(
            "Day 21 final holdout Contract v2 gate failed: "
            + ", ".join(contract_health["failures"])
        )
    metadata_path = config.repository_root / str(final_model["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["final_holdout"] = {
        "status": "completed",
        "summary": holdout["summary"],
        "results": holdout["results"],
        "episodes": holdout["episodes"],
        "concrete_episode_seeds": list(config.holdout_concrete_seeds),
        "evaluation_order": "selection → final freeze → final holdout",
    }
    write_json(metadata_path, metadata)
    final_model["metadata_sha256"] = sha256_file(metadata_path)
    final_model["final_holdout"] = metadata["final_holdout"]
    return holdout


def _refresh_reports(config: Day21FinalTrainingConfig, manifest_path: Path) -> None:
    report = build_day21_report(manifest_path, config=config)
    json_path = config.repository_root / "assets/day21/final-training-report.json"
    write_json(json_path, report)
    markdown_path = config.repository_root / "reports/day21-final-long-training.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_day21_markdown(report), encoding="utf-8")


def run_final_training(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    config = load_day21_config(args.config)
    manifest_path = args.manifest.resolve()
    if manifest_path.is_file():
        if not args.resume and args.stage != "plan":
            raise FileExistsError(
                f"Day 21 manifest already exists: {manifest_path}; pass --resume"
            )
        manifest = validate_day21_manifest(manifest_path, config=config)
    else:
        manifest = build_day21_manifest(
            config,
            manifest_path=manifest_path,
            runs_root=args.runs_root,
            evaluations_root=args.evaluations_root,
            evidence_root=args.evidence_root,
            command=sys.argv,
        )
        manifest["source_provenance"] = _source_provenance(config.repository_root)
        manifest["requested_stage"] = args.stage
        _write_manifest(manifest_path, manifest)
    if args.stage == "plan" or args.dry_run:
        return 0, manifest
    _refresh_source_provenance(config, manifest, manifest_path)
    if _repair_completed_manifest(
        config,
        manifest_path=manifest_path,
        manifest=manifest,
    ):
        _write_manifest(manifest_path, manifest)
    if (
        manifest.get("status") == "completed"
        and isinstance(manifest.get("canonical_final_model"), Mapping)
        and isinstance(manifest.get("final_holdout"), Mapping)
        and manifest["final_holdout"].get("status") == "completed"
    ):
        _refresh_reports(config, manifest_path)
        return 0, manifest

    _check_cuda_headroom(config)
    manifest["status"] = "running"
    manifest.pop("error", None)
    manifest["requested_stage"] = args.stage
    manifest["source_provenance"] = manifest.get(
        "source_provenance",
        _source_provenance(config.repository_root),
    )
    _write_manifest(manifest_path, manifest)
    sessions: dict[int, tuple[VectorizedDQNTrainer, Any, Path | None]] = {}
    session_rng_states: dict[int, tuple[torch.Tensor, list[torch.Tensor] | None]] = {}
    failure: BaseException | None = None
    try:
        for seed in config.training_seeds:
            entry = _entry_for_seed(manifest, seed)
            if args.resume:
                run_dir, latest = _prepare_run_directory(manifest_path, entry)
            else:
                run_dir = _resolve_manifest_reference(manifest_path, entry["run_dir"])
                latest = None
            env = make_breakout_vector_env(
                config.backend_config.num_envs,
                **breakout_environment_kwargs(config.contract),
            )
            try:
                trainer = VectorizedDQNTrainer(
                    env,
                    config.training_config(seed),
                    run_dir=run_dir,
                    resume_from=latest,
                    metadata={
                        "day21_final_training": {
                            "experiment_id": config.experiment_id,
                            "run_id": entry["run_id"],
                            "training_seed": seed,
                            "fresh_training_seed": True,
                            "target_max_transitions": config.stage_targets["stage_c_5m"],
                            "continuous_run": True,
                            "selection_evaluation_seeds": list(config.selection_seeds),
                            "final_holdout_group_seeds": list(config.holdout_group_seeds),
                        },
                        "source_provenance": manifest.get("source_provenance"),
                    },
                    environment_contract=config.contract_provenance,
                )
            except BaseException:
                env.close()
                raise
            sessions[seed] = (trainer, env, latest)
            session_rng_states[seed] = _capture_torch_rng_state()

        stage_resume_sources: dict[int, Path | None] = {
            seed: sessions[seed][2] for seed in config.training_seeds
        }

        # Stage A: all fresh seeds must reach 1M and pass the same selection gate.
        for seed in config.training_seeds:
            trainer, _env, _initial_resume = sessions[seed]
            entry = _entry_for_seed(manifest, seed)
            was_completed = _completed_stage(_stage_record(entry, "stage_a_1m"))
            _record_and_evaluate_stage(
                config,
                manifest_path=manifest_path,
                manifest=manifest,
                entry=entry,
                stage="stage_a_1m",
                trainer=trainer,
                resumed_from=stage_resume_sources[seed],
                session_rng_states=session_rng_states,
            )
            if not was_completed:
                stage_resume_sources[seed] = None
        stage_a_records = [
            _selection_record(_entry_for_seed(manifest, seed), "stage_a_1m")
            for seed in config.training_seeds
        ]
        decisions = manifest["selection_decisions"]
        if decisions.get("stage_a_1m") is None:
            selected_a = select_extension_candidates(
                stage_a_records,
                limit=DAY21_MAX_STAGE_B_CANDIDATES,
            )
            if not selected_a:
                raise RuntimeError("Day 21 Stage A did not produce a healthy candidate")
            decisions["stage_a_1m"] = _decision_payload(
                stage="stage_a_1m",
                records=stage_a_records,
                selected=selected_a,
                rule=config.selection_rule,
                reason="select at most two fresh seeds by aggregate selection evaluation mean",
            )
            _write_manifest(manifest_path, manifest)
        selected_a_seeds = [int(seed) for seed in decisions["stage_a_1m"]["selected_training_seeds"]]
        for seed in config.training_seeds:
            if seed not in selected_a_seeds:
                _mark_not_selected(
                    _entry_for_seed(manifest, seed),
                    "stage_b_2_5m",
                    reason="Stage A aggregate selection did not place this seed in the top two",
                )
                _mark_not_selected(
                    _entry_for_seed(manifest, seed),
                    "stage_c_5m",
                    reason="Stage A aggregate selection did not place this seed in the top two",
                )

        # Stage B: keep the selected trainers alive and extend them to 2.5M.
        for seed in selected_a_seeds:
            trainer, _env, _initial_resume = sessions[seed]
            entry = _entry_for_seed(manifest, seed)
            was_completed = _completed_stage(_stage_record(entry, "stage_b_2_5m"))
            _record_and_evaluate_stage(
                config,
                manifest_path=manifest_path,
                manifest=manifest,
                entry=entry,
                stage="stage_b_2_5m",
                trainer=trainer,
                resumed_from=stage_resume_sources[seed],
                session_rng_states=session_rng_states,
            )
            if not was_completed:
                stage_resume_sources[seed] = None
        stage_b_records = [
            _selection_record(_entry_for_seed(manifest, seed), "stage_b_2_5m")
            for seed in selected_a_seeds
            if _stage_record(_entry_for_seed(manifest, seed), "stage_b_2_5m").get("status") == "completed"
        ]
        if decisions.get("stage_b_2_5m") is None:
            selected_b = select_extension_candidates(
                stage_b_records,
                limit=DAY21_MAX_STAGE_C_CANDIDATES,
            )
            if not selected_b:
                raise RuntimeError("Day 21 Stage B did not produce a healthy candidate")
            decisions["stage_b_2_5m"] = _decision_payload(
                stage="stage_b_2_5m",
                records=stage_b_records,
                selected=selected_b,
                rule=config.selection_rule,
                reason="select one candidate for the requested 5M continuation by aggregate evaluation",
            )
            _write_manifest(manifest_path, manifest)
        selected_b_seeds = [int(seed) for seed in decisions["stage_b_2_5m"]["selected_training_seeds"]]
        for seed in selected_a_seeds:
            if seed not in selected_b_seeds:
                _mark_not_selected(
                    _entry_for_seed(manifest, seed),
                    "stage_c_5m",
                    reason="Stage B aggregate selection did not place this seed in the 5M candidate set",
                )

        # Stage C remains a quality-triggered continuation; the explicit request
        # is retained only as supplemental run-horizon provenance.
        for seed in selected_b_seeds:
            trainer, _env, _initial_resume = sessions[seed]
            entry = _entry_for_seed(manifest, seed)
            was_completed = _completed_stage(_stage_record(entry, "stage_c_5m"))
            _record_and_evaluate_stage(
                config,
                manifest_path=manifest_path,
                manifest=manifest,
                entry=entry,
                stage="stage_c_5m",
                trainer=trainer,
                resumed_from=stage_resume_sources[seed],
                session_rng_states=session_rng_states,
            )
            if not was_completed:
                stage_resume_sources[seed] = None
        stage_c_records = [
            _selection_record(_entry_for_seed(manifest, seed), "stage_c_5m")
            for seed in selected_b_seeds
            if _stage_record(_entry_for_seed(manifest, seed), "stage_c_5m").get("status") == "completed"
        ]
        if decisions.get("stage_c_5m") is None:
            decisions["stage_c_5m"] = _decision_payload(
                stage="stage_c_5m",
                records=stage_c_records,
                selected=stage_c_records,
                rule=config.selection_rule,
                reason="2.5M evaluation showed substantial improvement, so 5M continuation remained justified.",
            )
        stage_c_trigger = _stage_c_trigger_provenance(
            config,
            stage_a_records=stage_a_records,
            stage_b_records=stage_b_records,
            selected_b_seeds=selected_b_seeds,
        )
        decisions["stage_c_5m"].update(stage_c_trigger)
        decisions["stage_c_5m"]["reason"] = stage_c_trigger["primary_trigger"]
        decisions["stage_c_5m"].pop("explicit_user_target_override", None)
        decisions["stage_c_5m"].pop("override_reason", None)
        _write_manifest(manifest_path, manifest)

        all_candidates: list[dict[str, Any]] = []
        for seed in config.training_seeds:
            entry = _entry_for_seed(manifest, seed)
            for stage in DAY21_STAGE_ORDER:
                record = _stage_record(entry, stage)
                if _completed_stage(record):
                    all_candidates.append(_selection_record(entry, stage))
        final_candidate = select_final_checkpoint(
            all_candidates,
            near_equal_absolute_gap=float(config.selection_rule.get("near_equal_absolute_gap", 1.0)),
        )
        if manifest.get("canonical_final_model") is None:
            _freeze_canonical_model(
                config,
                manifest_path=manifest_path,
                manifest=manifest,
                candidate=final_candidate,
            )
            _write_manifest(manifest_path, manifest)
        _run_final_holdout(
            config,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        manifest.pop("error", None)
        manifest["status"] = "completed"
        for entry in manifest["runs"]:
            if isinstance(entry, dict):
                entry["status"] = "completed"
        _write_manifest(manifest_path, manifest)
        _refresh_reports(config, manifest_path)
        return 0, manifest
    except KeyboardInterrupt as error:
        failure = error
        manifest["status"] = "interrupted"
    except BaseException as error:
        failure = error
        manifest["status"] = "blocked" if isinstance(error, RuntimeError) and "CUDA" in str(error) else "failed"
    finally:
        for trainer, env, _resumed_from in sessions.values():
            try:
                trainer.metrics.close()
            finally:
                env.close()
        if failure is not None:
            manifest["error"] = f"{type(failure).__name__}: {failure}"
            _write_manifest(manifest_path, manifest)
            try:
                _refresh_reports(config, manifest_path)
            except Exception:
                pass
    if failure is not None:
        print(f"Day 21 final training failed: {failure}", file=sys.stderr)
        return 1, manifest
    return 0, manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exit_code, manifest = run_final_training(args)
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 21 final training could not start: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "manifest": args.manifest.as_posix(),
                "winner": manifest.get("winner"),
                "selection_decisions": manifest.get("selection_decisions"),
                "canonical_final_model": manifest.get("canonical_final_model"),
                "final_holdout": manifest.get("final_holdout"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
