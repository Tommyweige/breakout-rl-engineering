"""Run the staged Day 20 DQN-family comparison sequentially on CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from breakout_env import make_breakout_env, make_breakout_vector_env
from breakout_rl.day20_comparison import (
    DAY20_FAMILY_IDS,
    DAY20_FORMAL_STAGE,
    DAY20_MILESTONES,
    DAY20_REQUIRED_TRAINING_METRICS,
    DAY20_STAGE_ORDER,
    Day20ExperimentConfig,
    apply_day18_reuse,
    audit_day18_evidence_reuse,
    build_day20_manifest,
    build_day20_report,
    load_day20_config,
    relative_path,
    resolve_manifest_reference,
    sha256_file,
    utc_timestamp,
    validate_day20_manifest,
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
from scripts.analysis.analyze_q_values import analyze_checkpoint


EXTENSION_STAGE = "extension_1m"
EXTENSION_TARGET = 1_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the staged Day 20 DQN-family comparison on one CUDA device."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/comparisons/dqn-family/manifest.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day20-dqn-family/manifest.json"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--evaluations-root", type=Path, default=Path("evaluations"))
    parser.add_argument(
        "--stage",
        choices=("screening", "pilot", "main", "all", "extension"),
        default="pilot",
        help="target stage; main/all run all three seeds, extension runs selected top two",
    )
    parser.add_argument(
        "--family",
        dest="families",
        action="append",
        choices=DAY20_FAMILY_IDS,
        help="limit training to one or more family ids; repeat the option",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="debug-only; resulting entries cannot be formal-quality eligible",
    )
    parser.add_argument(
        "--skip-q-probe",
        action="store_true",
        help="debug-only; resulting entries cannot be formal-quality eligible",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="ignore compatible Day 18 evidence and force fresh family runs",
    )
    return parser


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    mutable = dict(manifest)
    mutable["updated_at_utc"] = utc_timestamp()
    write_json(path, mutable)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "checkpoints").glob("step-*.pt"))
    return candidates[-1] if candidates else None


def _checkpoint_step(path: Path | None) -> int:
    if path is None:
        return 0
    stem = path.stem
    digits = "".join(character for character in stem if character.isdigit())
    return int(digits) if digits else 0


def _checkpoint_reference(
    checkpoint: Path,
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "path": relative_path(checkpoint, start=manifest_path.parent),
        "sha256": sha256_file(checkpoint),
        "step": _checkpoint_step(checkpoint),
    }


def _failure_status(error: BaseException) -> str:
    message = str(error).lower()
    if any(
        marker in message
        for marker in (
            "cuda was requested",
            "cuda is not available",
            "cuda device index",
            "refusing to fall back",
            "headroom",
        )
    ):
        return "blocked"
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    return "failed"


def _invalidate_reused_entries(
    manifest: dict[str, Any],
    *,
    reason: str,
) -> int:
    """Remove stale Day 18 values so an incompatible audit cannot be mixed."""

    invalidated = 0
    for entry in manifest.get("runs", []):
        if not isinstance(entry, dict) or entry.get("family_id") not in {"dqn", "double_dqn"}:
            continue
        source = entry.get("source")
        is_reused = entry.get("status") == "reused" or (
            isinstance(source, Mapping) and source.get("kind") == "day18_evidence_reuse"
        )
        if not is_reused:
            continue
        run_dir = entry.get("run_dir")
        entry["status"] = "pending"
        entry["eligible"] = False
        entry["summary"] = None
        entry["runtime"] = None
        entry["checkpoint"] = None
        entry["resume_from"] = None
        entry["source"] = None
        entry["error"] = reason
        entry["training"] = {
            "metrics_path": (
                f"{run_dir}/metrics.csv" if isinstance(run_dir, str) else None
            ),
            "summary": None,
            "runtime": None,
        }
        evaluation = entry.get("evaluation")
        directory = evaluation.get("directory") if isinstance(evaluation, Mapping) else None
        entry["evaluation"] = {
            "directory": directory,
            "results": None,
            "episodes": None,
            "summary": None,
            "status": "pending",
        }
        entry["q_probe"] = None
        invalidated += 1
    return invalidated


def _invalidate_extension_entries(
    manifest: dict[str, Any],
    *,
    reason: str,
) -> int:
    """Discard extension results when their 500K provenance is no longer valid."""

    invalidated = 0
    for entry in manifest.get("runs", []):
        if not isinstance(entry, dict) or entry.get("stage") != EXTENSION_STAGE:
            continue
        entry["status"] = "pending"
        entry["eligible"] = False
        entry["summary"] = None
        entry["runtime"] = None
        entry["checkpoint"] = None
        entry["resume_from"] = None
        entry["source"] = None
        entry["error"] = reason
        run_dir = entry.get("run_dir")
        entry["training"] = {
            "metrics_path": (
                f"{run_dir}/metrics.csv" if isinstance(run_dir, str) else None
            ),
            "summary": None,
            "runtime": None,
        }
        evaluation = entry.get("evaluation")
        directory = evaluation.get("directory") if isinstance(evaluation, Mapping) else None
        entry["evaluation"] = {
            "directory": directory,
            "results": None,
            "episodes": None,
            "summary": None,
            "status": "pending",
        }
        entry["q_probe"] = None
        invalidated += 1
    return invalidated


def _refresh_artifact_hashes(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    """Backfill hashes for completed artifacts from older runner revisions."""

    for entry in manifest.get("runs", []):
        if not isinstance(entry, dict):
            continue
        training = entry.get("training")
        if isinstance(training, dict):
            metrics_path = training.get("metrics_path")
            if isinstance(metrics_path, str):
                candidate = resolve_manifest_reference(manifest_path, metrics_path)
                if candidate.is_file():
                    training["metrics_sha256"] = sha256_file(candidate)
        evaluation = entry.get("evaluation")
        if isinstance(evaluation, dict):
            for field in ("results", "episodes"):
                value = evaluation.get(field)
                if isinstance(value, str):
                    candidate = resolve_manifest_reference(manifest_path, value)
                    if candidate.is_file():
                        evaluation[f"{field}_sha256"] = sha256_file(candidate)
        probe = entry.get("q_probe")
        if isinstance(probe, dict) and isinstance(probe.get("path"), str):
            candidate = resolve_manifest_reference(manifest_path, probe["path"])
            if candidate.is_file():
                probe["sha256"] = sha256_file(candidate)


def _check_cuda_headroom(config: Day20ExperimentConfig) -> torch.device:
    device = resolve_device(config.requested_device)
    if device.type != "cuda":
        raise RuntimeError("Day 20 formal comparison resolved to a non-CUDA device")
    index = 0 if device.index is None else int(device.index)
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError("unable to measure CUDA free memory for headroom gate") from error
    if int(free_bytes) < config.cuda_headroom_bytes:
        raise RuntimeError(
            "CUDA free memory is below the required Day 20 headroom: "
            f"free={int(free_bytes)} bytes, required={config.cuda_headroom_bytes} bytes"
        )
    return device


def _selected_families(
    config: Day20ExperimentConfig,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if not requested:
        return config.family_ids
    selected = tuple(dict.fromkeys(requested))
    if any(value not in config.family_ids for value in selected):
        raise ValueError("unknown Day 20 family selection")
    return selected


def _requested_selection(
    config: Day20ExperimentConfig,
    stage: str,
    families: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if stage == "screening":
        return config.training_seeds, ("screening",)
    if stage == "pilot":
        return (config.training_seeds[0],), ("screening", "pilot")
    if stage in {"main", "all"}:
        return config.training_seeds, DAY20_STAGE_ORDER
    if stage == "extension":
        return config.training_seeds, (EXTENSION_STAGE,)
    raise ValueError(f"unknown Day 20 requested stage: {stage}")


def _entry_for_key(
    manifest: Mapping[str, Any],
    key: tuple[str, int, str],
) -> dict[str, Any] | None:
    for entry in manifest.get("runs", []):
        if isinstance(entry, dict):
            candidate = (
                str(entry.get("family_id")),
                int(entry.get("training_seed", -1)),
                str(entry.get("stage")),
            )
            if candidate == key:
                return entry
    return None


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(entry["family_id"]),
        int(entry["training_seed"]),
        str(entry["stage"]),
    )


def _previous_entry(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    family_id = str(entry["family_id"])
    seed = int(entry["training_seed"])
    stage = str(entry["stage"])
    if stage == EXTENSION_STAGE:
        previous_stage = DAY20_FORMAL_STAGE
    else:
        try:
            index = DAY20_STAGE_ORDER.index(stage)
        except ValueError:
            return None
        if index == 0:
            return None
        previous_stage = DAY20_STAGE_ORDER[index - 1]
    return _entry_for_key(manifest, (family_id, seed, previous_stage))


def _completed_entry(entry: Mapping[str, Any], *, target: int) -> bool:
    summary = entry.get("summary")
    if not isinstance(summary, Mapping):
        training = entry.get("training")
        summary = training.get("summary", {}) if isinstance(training, Mapping) else {}
    return bool(
        entry.get("status") in {"completed", "reused"}
        and summary.get("status") == "completed"
        and int(summary.get("total_transitions", -1)) == target
        and isinstance(entry.get("evaluation"), Mapping)
        and entry["evaluation"].get("status") in {"completed", "reused"}
        and isinstance(entry.get("q_probe"), Mapping)
        and entry["q_probe"].get("status") in {"completed", "reused"}
    )


def _prepare_run_directory(
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    resume: bool,
) -> tuple[Path, Path | None]:
    run_dir = resolve_manifest_reference(manifest_path, entry["run_dir"])
    if not run_dir.exists():
        return run_dir, None
    checkpoint = _latest_checkpoint(run_dir)
    if resume and checkpoint is not None:
        return run_dir, checkpoint
    if not any(run_dir.iterdir()):
        return run_dir, None
    if not resume:
        raise FileExistsError(
            f"stage run directory already contains artifacts: {run_dir}; pass --resume"
        )
    attempts = entry.setdefault("attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("manifest run attempts must be an array")
    attempts.append(
        {
            "run_dir": relative_path(run_dir, start=manifest_path.parent),
            "status": entry.get("status"),
            "error": entry.get("error"),
            "reason": "partial run has no checkpoint; preserved before a fresh retry",
        }
    )
    attempt_number = len(attempts) + 1
    retry_dir = run_dir.with_name(f"{run_dir.name}-retry-{attempt_number:02d}")
    while retry_dir.exists():
        attempt_number += 1
        retry_dir = run_dir.with_name(f"{run_dir.name}-retry-{attempt_number:02d}")
    entry["run_dir"] = relative_path(retry_dir, start=manifest_path.parent)
    return retry_dir, None


def _compact_metrics(
    source: Path,
    destination: Path,
) -> Path:
    """Keep completed-episode and diagnostic rows as a reviewable artifact."""

    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        interesting = (
            "raw_episode_return",
            "loss",
            "q_mean",
            "q_max",
            "target_mean",
            "td_error_mean_abs",
            "gradient_norm",
        )
        for row in reader:
            if any(row.get(field) not in {None, "", "None"} for field in interesting):
                rows.append(row)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _metrics_have_required_fields(path: Path) -> bool:
    with path.open("r", newline="", encoding="utf-8") as stream:
        fields = set(csv.DictReader(stream).fieldnames or ())
    return all(field in fields for field in DAY20_REQUIRED_TRAINING_METRICS)


def _compact_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    keep = {
        "status",
        "trainer",
        "run_id",
        "seed",
        "algorithm",
        "architecture",
        "num_envs",
        "total_steps",
        "total_transitions",
        "training_steps",
        "physical_environment_steps",
        "vector_iterations",
        "episodes",
        "optimizer_updates",
        "target_sync_count",
        "last_target_sync_step",
        "replay_backend",
        "replay_transfer",
        "replay_size",
        "replay_occupancy",
        "replay_rewarm_steps_remaining",
        "parameter_count",
        "model_config",
        "steps_per_second",
        "environment_transitions_per_second",
        "last_loss",
        "last_q_mean",
        "last_q_max",
        "last_target_mean",
        "last_td_error_mean_abs",
        "last_gradient_norm",
        "action_distribution",
        "random_decision_ratio",
        "last_checkpoint",
        "runtime",
        "stage_start_step",
        "stage_start_counters",
        "stage_counters",
        "stage_rates",
        "stage_training_steps",
        "stage_physical_environment_steps",
        "resume_provenance",
        "metadata",
        "environment_contract",
    }
    return {key: value for key, value in summary.items() if key in keep}


def _run_evaluation(
    config: Day20ExperimentConfig,
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    checkpoint: Path,
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    env_factory = lambda: make_breakout_env(**breakout_environment_kwargs(config.contract))
    loaded = load_dqn_checkpoint(
        checkpoint,
        device=config.requested_device,
        env_factory=env_factory,
    )
    source_checkpoint_contract = loaded.checkpoint_metadata.get("environment_contract")
    if isinstance(source_checkpoint_contract, Mapping) and (
        source_checkpoint_contract.get("contract_id") != config.contract.contract_id
        or source_checkpoint_contract.get("contract_sha256") != sha256_file(config.contract_path)
    ):
        raise RuntimeError("checkpoint Contract v2 provenance does not match the Day 20 contract")
    training_metadata = {
        **dict(loaded.training_metadata),
        "source_day20_experiment_id": config.experiment_id,
        "source_day20_run_id": entry.get("run_id"),
        "family_id": entry["family_id"],
        "training_seed": int(entry["training_seed"]),
        "training_budget": int(entry["target_transitions"]),
        "training_stage": entry["stage"],
        "trainer_runtime": dict(training_summary.get("runtime", {})),
        "environment_contract": config.contract_provenance,
        "resume_provenance": training_summary.get("resume_provenance"),
        "manifest": relative_path(manifest_path, start=config.repository_root),
    }
    checkpoint_metadata = {
        **dict(loaded.checkpoint_metadata),
        "source_day20_experiment_id": config.experiment_id,
        "source_day20_run_id": entry.get("run_id"),
        "family_id": entry["family_id"],
        "training_stage": entry["stage"],
        "source_checkpoint_environment_contract": source_checkpoint_contract,
        "evaluation_contract": config.contract_provenance,
    }
    result = evaluate_policy(
        loaded.model,
        episodes=config.episodes_per_seed,
        seeds=config.evaluation_seeds,
        device=config.requested_device,
        epsilon=0.0,
        model_id=loaded.model_id,
        training_metadata=training_metadata,
        checkpoint_metadata=checkpoint_metadata,
        evaluation_id=(
            f"{config.experiment_id}-{entry['family_id']}-seed{entry['training_seed']}-"
            f"step{entry['target_transitions']}"
        ),
        env_factory=env_factory,
        metadata={
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
            "source_day20_manifest": relative_path(
                manifest_path,
                start=config.repository_root,
            ),
            "purpose": "Day 20 Contract v2 DQN-family comparison evaluation",
            "raw_reward": True,
        },
    )
    evaluation = entry.setdefault("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("Day 20 evaluation entry must be an object")
    evaluation_dir = resolve_manifest_reference(manifest_path, evaluation["directory"])
    results_path, episodes_path = write_evaluation_artifacts(result, evaluation_dir)
    evaluation["results"] = relative_path(results_path, start=manifest_path.parent)
    evaluation["episodes"] = relative_path(episodes_path, start=manifest_path.parent)
    evaluation["results_sha256"] = sha256_file(results_path)
    evaluation["episodes_sha256"] = sha256_file(episodes_path)
    evaluation["status"] = "completed"
    evaluation["summary"] = result.to_dict()["summary"]
    return result.to_dict()


def _run_q_probe(
    config: Day20ExperimentConfig,
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    checkpoint: Path,
    run_dir: Path,
) -> dict[str, Any]:
    payload = analyze_checkpoint(
        checkpoint,
        config.probe_states_path,
        device=config.requested_device,
    )
    payload["comparison"] = {
        "experiment_id": config.experiment_id,
        "run_id": entry.get("run_id"),
        "family_id": entry["family_id"],
        "algorithm": entry["algorithm"],
        "architecture": entry["architecture"],
        "training_seed": entry["training_seed"],
        "stage": entry["stage"],
        "target_transitions": entry["target_transitions"],
        "environment_contract": config.contract_provenance,
    }
    raw_output = run_dir / "diagnostics" / f"q-probe-step-{int(entry['target_transitions']):08d}.json"
    write_json(raw_output, payload)
    compact_output = (
        config.repository_root
        / "assets/day20/evidence-q"
        / f"{entry['family_id']}-seed{entry['training_seed']}-{entry['stage']}.json"
    )
    write_json(compact_output, payload)
    analysis = payload.get("analysis", {})
    if not isinstance(analysis, Mapping):
        analysis = {}
    entry["q_probe"] = {
        "path": relative_path(compact_output, start=manifest_path.parent),
        "sha256": sha256_file(compact_output),
        "summary": {
            field: analysis.get(field)
            for field in (
                "probe_count",
                "action_count",
                "q_mean",
                "q_std",
                "q_min",
                "q_max",
                "max_q_mean",
                "max_q_std",
                "selected_action_distribution",
            )
            if field in analysis
        },
        "status": "completed",
    }
    return payload


def _run_one_entry(
    config: Day20ExperimentConfig,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    entry: dict[str, Any],
    resume: bool,
    skip_evaluation: bool,
    skip_q_probe: bool,
) -> None:
    stage = str(entry["stage"])
    target = EXTENSION_TARGET if stage == EXTENSION_STAGE else int(entry["target_transitions"])
    if stage == EXTENSION_STAGE:
        stage_config = config.backend_config.with_overrides(
            algorithm=str(entry["algorithm"]),
            architecture=str(entry["architecture"]),
            seed=int(entry["training_seed"]),
            total_steps=target,
            checkpoint_interval=min(100_000, target),
            contract_id=config.contract.contract_id,
            contract_path=relative_path(config.contract_path, start=config.repository_root),
        )
    else:
        stage_config = config.training_config(
            family_id=str(entry["family_id"]),
            seed=int(entry["training_seed"]),
            stage=stage,
        )
        stage_config = stage_config.with_overrides(
            checkpoint_interval=min(100_000, target),
        )
    entry["training_config"] = stage_config.to_dict()
    if stage_config.requested_device != "cuda":
        raise ValueError("Day 20 stage config must request exactly cuda")
    _check_cuda_headroom(config)

    previous = _previous_entry(manifest, entry)
    resume_checkpoint: Path | None = None
    if previous is not None:
        if previous.get("status") not in {"completed", "reused"}:
            raise RuntimeError("cannot start a later Day 20 stage before its previous stage completes")
        previous_checkpoint = previous.get("checkpoint")
        if isinstance(previous_checkpoint, Mapping) and isinstance(previous_checkpoint.get("path"), str):
            candidate = resolve_manifest_reference(manifest_path, previous_checkpoint["path"])
            if candidate.is_file():
                resume_checkpoint = candidate
        if resume_checkpoint is None:
            previous_dir = resolve_manifest_reference(manifest_path, previous["run_dir"])
            resume_checkpoint = _latest_checkpoint(previous_dir)
        if resume_checkpoint is None:
            raise FileNotFoundError(
                "previous Day 20 stage completed without a usable checkpoint; "
                "refusing to restart a continuation from step 0"
            )

    run_dir, stage_checkpoint = _prepare_run_directory(
        manifest_path=manifest_path,
        entry=entry,
        resume=resume,
    )
    if stage_checkpoint is not None:
        resume_checkpoint = stage_checkpoint
    run_dir.mkdir(parents=True, exist_ok=True)
    entry["status"] = "running"
    entry["eligible"] = False
    entry["error"] = None
    entry["resume_from"] = (
        _checkpoint_reference(resume_checkpoint, manifest_path=manifest_path)
        if resume_checkpoint is not None
        else None
    )
    _write_manifest(manifest_path, manifest)

    env = make_breakout_vector_env(
        stage_config.num_envs,
        **breakout_environment_kwargs(config.contract),
    )
    summary: dict[str, Any]
    try:
        trainer = VectorizedDQNTrainer(
            env,
            stage_config,
            run_dir=run_dir,
            resume_from=resume_checkpoint,
            metadata={
                "day20_comparison": {
                    "experiment_id": config.experiment_id,
                    "run_id": entry.get("run_id"),
                    "family_id": entry["family_id"],
                    "pair_id": entry.get("pair_id"),
                    "algorithm": entry["algorithm"],
                    "architecture": entry["architecture"],
                    "training_seed": entry["training_seed"],
                    "stage": stage,
                    "target_transitions": target,
                    "resume_from": entry.get("resume_from"),
                }
            },
            environment_contract=config.contract_provenance,
        )
        summary = trainer.train()
    finally:
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if summary.get("status") != "completed" or int(summary.get("total_transitions", -1)) != target:
        raise RuntimeError(
            "stage did not complete its exact transition budget: "
            f"status={summary.get('status')!r}, transitions={summary.get('total_transitions')!r}"
        )
    checkpoint = Path(str(summary.get("last_checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"completed stage checkpoint is missing: {checkpoint}")

    entry["status"] = "completed"
    entry["summary"] = _compact_summary(summary)
    entry["runtime"] = dict(summary.get("runtime", {}))
    entry["checkpoint"] = _checkpoint_reference(checkpoint, manifest_path=manifest_path)
    compact_metrics_path = (
        config.repository_root
        / "assets/day20/evidence-runs"
        / f"{entry['family_id']}-seed{entry['training_seed']}-{stage}"
        / "metrics.csv"
    )
    if not _metrics_have_required_fields(run_dir / "metrics.csv"):
        raise RuntimeError(
            "Day 20 training metrics are missing one or more required fields: "
            + ", ".join(DAY20_REQUIRED_TRAINING_METRICS)
        )
    _compact_metrics(run_dir / "metrics.csv", compact_metrics_path)
    training = entry.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("Day 20 training entry must be an object")
    training["metrics_path"] = relative_path(compact_metrics_path, start=manifest_path.parent)
    training["metrics_sha256"] = sha256_file(compact_metrics_path)
    training["summary"] = entry["summary"]
    training["runtime"] = entry["runtime"]
    _write_manifest(manifest_path, manifest)

    if skip_evaluation:
        entry["evaluation"] = {
            **dict(entry.get("evaluation", {})),
            "status": "skipped",
            "summary": None,
        }
    else:
        _run_evaluation(
            config,
            manifest_path=manifest_path,
            entry=entry,
            checkpoint=checkpoint,
            training_summary=summary,
        )
    if skip_q_probe:
        entry["q_probe"] = {"status": "skipped", "summary": None}
    else:
        _run_q_probe(
            config,
            manifest_path=manifest_path,
            entry=entry,
            checkpoint=checkpoint,
            run_dir=run_dir,
        )
    entry["eligible"] = bool(
        not skip_evaluation
        and not skip_q_probe
        and _completed_entry(entry, target=target)
    )
    _write_manifest(manifest_path, manifest)


def _ensure_extension_entries(
    manifest: dict[str, Any],
    *,
    config: Day20ExperimentConfig,
    family_ids: Sequence[str],
    runs_root: Path,
    evaluations_root: Path,
    manifest_path: Path,
) -> None:
    existing = {
        _entry_key(entry)
        for entry in manifest.get("runs", [])
        if isinstance(entry, Mapping)
    }
    for family_id in family_ids:
        family = config.family(family_id)
        for seed in config.training_seeds:
            key = (family_id, seed, EXTENSION_STAGE)
            if key in existing:
                continue
            run_dir = runs_root.resolve() / config.experiment_id / family_id / f"seed{seed}" / "stage-1000k"
            evaluation_dir = evaluations_root.resolve() / config.experiment_id / family_id / f"seed{seed}" / "step-01000000"
            manifest["runs"].append(
                {
                    "run_id": f"{config.experiment_id}-{family_id}-seed{seed}-extension-1m",
                    "pair_id": f"training-seed-{seed}",
                    "family_id": family_id,
                    "family_label": family.label,
                    "algorithm": family.algorithm,
                    "architecture": family.architecture,
                    "training_seed": seed,
                    "stage": EXTENSION_STAGE,
                    "target_transitions": EXTENSION_TARGET,
                    "status": "pending",
                    "eligible": False,
                    "run_dir": relative_path(run_dir, start=manifest_path.parent),
                    "resume_from": None,
                    "checkpoint": None,
                    "training": {
                        "metrics_path": relative_path(run_dir / "metrics.csv", start=manifest_path.parent),
                        "summary": None,
                        "runtime": None,
                    },
                    "evaluation": {
                        "directory": relative_path(evaluation_dir, start=manifest_path.parent),
                        "results": None,
                        "episodes": None,
                        "summary": None,
                        "status": "pending",
                    },
                    "q_probe": None,
                    "training_config": config.backend_config.with_overrides(
                        algorithm=family.algorithm,
                        architecture=family.architecture,
                        seed=seed,
                        total_steps=EXTENSION_TARGET,
                        checkpoint_interval=EXTENSION_TARGET,
                        contract_id=config.contract.contract_id,
                        contract_path=relative_path(config.contract_path, start=config.repository_root),
                    ).to_dict(),
                    "summary": None,
                    "runtime": None,
                    "source": None,
                    "error": None,
                }
            )


def _manifest_for_run(
    config: Day20ExperimentConfig,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest_path = args.manifest.resolve()
    if manifest_path.is_file():
        if not args.resume:
            raise FileExistsError(
                f"manifest already exists: {manifest_path}; pass --resume to continue"
            )
        manifest = validate_day20_manifest(manifest_path, config=config)
    else:
        manifest = build_day20_manifest(
            config,
            manifest_path=manifest_path,
            runs_root=args.runs_root,
            evaluations_root=args.evaluations_root,
            command=[str(value) for value in sys.argv],
        )
    reuse = manifest.setdefault("evidence_reuse", {})
    existing_decision = reuse.get("decision") if isinstance(reuse, Mapping) else None
    if args.no_reuse or not config.reuse_enabled:
        audit = {
            "schema_version": 1,
            "status": "disabled",
            "reuse_allowed": False,
            "source_manifest": None,
            "checks": [],
            "incompatibilities": ["Day 18 reuse was disabled by configuration or CLI"],
            "reusable_entries": [],
        }
    elif existing_decision in {"disabled", "incompatible_fresh_required"}:
        stored_audit = reuse.get("audit") if isinstance(reuse, Mapping) else None
        audit = (
            dict(stored_audit)
            if isinstance(stored_audit, Mapping)
            else {
                "schema_version": 1,
                "status": "incompatible",
                "reuse_allowed": False,
                "source_manifest": None,
                "checks": [],
                "incompatibilities": [
                    "existing manifest is locked to fresh evidence; create a new manifest to change reuse mode"
                ],
                "reusable_entries": [],
            }
        )
    else:
        audit = audit_day18_evidence_reuse(config)
    audit_path = config.repository_root / "assets/day20/evidence-reuse-audit.json"
    write_json(audit_path, audit)
    if isinstance(reuse, dict):
        if existing_decision in {"disabled", "incompatible_fresh_required"} and not args.no_reuse:
            reuse["decision"] = existing_decision
        else:
            reuse["decision"] = (
                "compatible_reused"
                if audit.get("reuse_allowed") is True
                else "incompatible_fresh_required"
            )
        reuse["audit"] = audit
    if audit.get("reuse_allowed") is True and config.reuse_manifest_path is not None:
        apply_day18_reuse(
            manifest_path,
            manifest,
            config=config,
            audit=audit,
            source_manifest=config.reuse_manifest_path,
        )
    else:
        _invalidate_extension_entries(
            manifest,
            reason="Day 18 reuse was not accepted; prior 1M extension evidence is invalid",
        )
        _invalidate_reused_entries(
            manifest,
            reason="Day 18 evidence reuse was not accepted; fresh training is required",
        )
    _refresh_artifact_hashes(manifest_path, manifest)
    _write_manifest(manifest_path, manifest)
    return manifest_path, manifest, audit


def run_comparison(args: argparse.Namespace) -> tuple[int, Path, dict[str, Any]]:
    config = load_day20_config(
        args.config,
        require_probe_states=not args.skip_q_probe,
    )
    selected_families = _selected_families(config, args.families)
    manifest_path, manifest, audit = _manifest_for_run(config, args)

    if args.stage == "extension":
        report = build_day20_report(manifest_path, config=config)
        selection = report.get("selection", {})
        extension = selection.get("extension", {}) if isinstance(selection, Mapping) else {}
        if not isinstance(extension, Mapping) or not extension.get("triggered"):
            raise RuntimeError("Day 20 1M extension was not triggered by the 500K aggregate rule")
        top = extension.get("top_candidates", [])
        if not isinstance(top, list) or not top:
            raise RuntimeError("Day 20 extension has no aggregate-selected top candidates")
        selected_families = tuple(
            family_id for family_id in top if family_id in config.family_ids
        )[:2]
        _ensure_extension_entries(
            manifest,
            config=config,
            family_ids=selected_families,
            runs_root=args.runs_root,
            evaluations_root=args.evaluations_root,
            manifest_path=manifest_path,
        )
        _write_manifest(manifest_path, manifest)

    selected_seeds, selected_stages = _requested_selection(
        config,
        args.stage,
        selected_families,
    )
    # Repair manifests written by an older runner revision that used the
    # eligibility flag as an input to the completion predicate itself.
    for existing_entry in manifest.get("runs", []):
        if not isinstance(existing_entry, dict):
            continue
        existing_target = (
            EXTENSION_TARGET
            if existing_entry.get("stage") == EXTENSION_STAGE
            else int(existing_entry.get("target_transitions", -1))
        )
        if existing_entry.get("eligible") is not True and _completed_entry(
            existing_entry,
            target=existing_target,
        ):
            existing_entry["eligible"] = True
    manifest["requested_stage"] = args.stage
    manifest["requested_families"] = list(selected_families)
    manifest["requested_training_seeds"] = list(selected_seeds)
    manifest["requested_stages"] = list(selected_stages)
    if args.dry_run:
        manifest["status"] = "planned"
        _write_manifest(manifest_path, manifest)
        return 0, manifest_path, manifest

    manifest["status"] = "running"
    _write_manifest(manifest_path, manifest)
    selected_entries = [
        entry
        for entry in manifest.get("runs", [])
        if isinstance(entry, dict)
        and entry.get("family_id") in set(selected_families)
        and int(entry.get("training_seed", -1)) in set(selected_seeds)
        and entry.get("stage") in set(selected_stages)
    ]
    failures = 0
    interrupted = False
    for entry in selected_entries:
        target = EXTENSION_TARGET if entry.get("stage") == EXTENSION_STAGE else int(entry["target_transitions"])
        if _completed_entry(entry, target=target):
            continue
        try:
            _run_one_entry(
                config,
                manifest_path=manifest_path,
                manifest=manifest,
                entry=entry,
                resume=args.resume,
                skip_evaluation=args.skip_evaluation,
                skip_q_probe=args.skip_q_probe,
            )
        except KeyboardInterrupt as error:
            interrupted = True
            failures += 1
            entry["status"] = "interrupted"
            entry["error"] = str(error) or "keyboard interrupt"
        except Exception as error:
            failures += 1
            entry["status"] = _failure_status(error)
            entry["error"] = str(error)
        _write_manifest(manifest_path, manifest)
        if interrupted:
            break

    statuses = [str(entry.get("status")) for entry in selected_entries]
    all_entries = [entry for entry in manifest.get("runs", []) if isinstance(entry, dict)]
    required_entries = [
        entry
        for entry in all_entries
        if entry.get("stage") in DAY20_STAGE_ORDER
        or entry.get("stage") == EXTENSION_STAGE
    ]
    if interrupted or "interrupted" in statuses:
        manifest["status"] = "interrupted"
    elif any(entry.get("status") == "blocked" for entry in all_entries):
        manifest["status"] = "blocked"
    elif failures or any(entry.get("status") == "failed" for entry in all_entries):
        manifest["status"] = "failed"
    elif all(
        _completed_entry(
            entry,
            target=EXTENSION_TARGET
            if entry.get("stage") == EXTENSION_STAGE
            else int(entry["target_transitions"]),
        )
        for entry in required_entries
    ):
        manifest["status"] = "completed"
    else:
        manifest["status"] = "partial"
    _refresh_artifact_hashes(manifest_path, manifest)
    _write_manifest(manifest_path, manifest)
    return (0 if manifest["status"] == "completed" else 1), manifest_path, manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exit_code, manifest_path, manifest = run_comparison(args)
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 20 family comparison could not start: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "manifest": manifest_path.as_posix(),
                "evidence_reuse": manifest.get("evidence_reuse", {}).get("decision")
                if isinstance(manifest.get("evidence_reuse"), Mapping)
                else None,
                "runs": [
                    {
                        "family_id": entry.get("family_id"),
                        "training_seed": entry.get("training_seed"),
                        "stage": entry.get("stage"),
                        "status": entry.get("status"),
                        "target_transitions": entry.get("target_transitions"),
                    }
                    for entry in manifest.get("runs", [])
                    if isinstance(entry, Mapping)
                    and entry.get("family_id") in set(manifest.get("requested_families", []))
                    and int(entry.get("training_seed", -1)) in set(
                        manifest.get("requested_training_seeds", [])
                    )
                    and entry.get("stage") in set(manifest.get("requested_stages", []))
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
