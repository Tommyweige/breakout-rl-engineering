"""Run the staged, paired Day 18 DQN versus Double DQN comparison."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from breakout_env import make_breakout_vector_env
from breakout_rl.day18_comparison import (
    DAY18_ALGORITHMS,
    DAY18_FORMAL_STAGE,
    Day18ExperimentConfig,
    build_day18_manifest,
    compact_training_summary,
    load_evaluation_entries,
    load_day18_config,
    load_q_probe_entries,
    load_training_entries,
    read_day18_manifest,
    relative_path,
    resolve_manifest_reference,
    sha256_file,
    utc_timestamp,
    write_json,
)
from breakout_rl.evaluation import evaluate_policy, load_dqn_checkpoint, write_evaluation_artifacts
from breakout_rl.evaluation_contract import breakout_environment_kwargs
from breakout_rl.training.dqn_trainer import resolve_device
from breakout_rl.training.vectorized import VectorizedDQNTrainer
from scripts.analysis.analyze_q_values import analyze_checkpoint


STAGE_ORDER = ("screening", "pilot", "main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the staged, paired Day 18 DQN versus Double DQN comparison."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/day18-dqn-vs-double.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day18-dqn-vs-double/manifest.json"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--evaluations-root",
        type=Path,
        default=Path("evaluations"),
    )
    parser.add_argument(
        "--stage",
        choices=("screening", "pilot", "main", "all"),
        default="pilot",
        help="target stage; pilot runs seed 11, main/all runs all three seeds",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing manifest and incomplete stage artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write a planned manifest without starting training",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="debug-only escape hatch; the manifest will not be formal-quality eligible",
    )
    parser.add_argument(
        "--skip-q-probe",
        action="store_true",
        help="debug-only escape hatch; the manifest will not contain all probe diagnostics",
    )
    return parser


def _requested_selection(
    config: Day18ExperimentConfig,
    requested_stage: str,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if requested_stage == "screening":
        return config.training_seeds, config.stages_through("screening")
    if requested_stage == "pilot":
        return (config.training_seeds[0],), config.stages_through("pilot")
    if requested_stage in {"main", "all"}:
        return config.training_seeds, config.stages_through(DAY18_FORMAL_STAGE)
    raise ValueError(f"unknown requested stage: {requested_stage}")


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(entry["algorithm"]),
        int(entry["training_seed"]),
        str(entry["stage"]),
    )


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    mutable = dict(manifest)
    mutable["updated_at_utc"] = utc_timestamp()
    write_json(path, mutable)


def _manifest_for_run(
    config: Day18ExperimentConfig,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], bool]:
    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"manifest already exists: {manifest_path}; pass --resume to continue"
            )
        manifest = read_day18_manifest(manifest_path)
        if manifest.get("experiment_id") != config.experiment_id:
            raise ValueError("existing manifest experiment_id does not match Day 18 config")
        source_config = manifest.get("source_of_truth", {}).get("comparison_config", {})
        if not isinstance(source_config, Mapping) or source_config.get("sha256") != sha256_file(config.source_path):
            raise ValueError("existing manifest was created from a different Day 18 config")
        source_of_truth = manifest.get("source_of_truth", {})
        if not isinstance(source_of_truth, Mapping):
            raise ValueError("existing manifest is missing source-of-truth references")
        for name, path in (
            ("backend_manifest", config.backend_manifest_path),
            ("contract", config.contract_path),
            ("evaluation_config", config.evaluation_config_path),
        ):
            reference = source_of_truth.get(name)
            expected_hash = reference.get("sha256") if isinstance(reference, Mapping) else None
            if expected_hash != sha256_file(path):
                raise ValueError(f"existing manifest {name} is out of date")
        return manifest_path, manifest, True

    manifest = build_day18_manifest(
        config,
        manifest_path=manifest_path,
        runs_root=args.runs_root,
        evaluations_root=args.evaluations_root,
        command=[str(value) for value in sys.argv],
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, manifest)
    return manifest_path, manifest, False


def _checkpoint_reference(
    checkpoint: Path,
    *,
    manifest_path: Path,
    step: int,
) -> dict[str, Any]:
    return {
        "path": relative_path(checkpoint, start=manifest_path.parent),
        "sha256": sha256_file(checkpoint),
        "step": int(step),
    }


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "checkpoints").glob("step-*.pt"))
    return candidates[-1] if candidates else None


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


def _write_failure(
    run_dir: Path,
    *,
    status: str,
    error: BaseException,
    config: Mapping[str, Any],
) -> None:
    write_json(
        run_dir / "failure.json",
        {
            "status": status,
            "error_type": type(error).__name__,
            "error": str(error),
            "requested_device": config.get("device"),
            "algorithm": config.get("algorithm"),
            "seed": config.get("seed"),
            "target_transitions": config.get("total_steps"),
            "created_at_utc": utc_timestamp(),
        },
    )


def _check_cuda_headroom(config: Day18ExperimentConfig) -> torch.device:
    device = resolve_device(config.backend_config.requested_device)
    if device.type != "cuda":
        raise RuntimeError("Day 18 formal comparison resolved to a non-CUDA device")
    index = 0 if device.index is None else int(device.index)
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError("unable to measure CUDA free memory for headroom gate") from error
    if int(free_bytes) < config.cuda_headroom_bytes:
        raise RuntimeError(
            "CUDA free memory is below the required Day 18 headroom: "
            f"free={int(free_bytes)} bytes, required={config.cuda_headroom_bytes} bytes"
        )
    return device


def _find_previous_entry(
    entries: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    current_index = STAGE_ORDER.index(str(current["stage"]))
    if current_index == 0:
        return None
    previous_stage = STAGE_ORDER[current_index - 1]
    for entry in entries:
        if (
            entry.get("algorithm") == current.get("algorithm")
            and int(entry.get("training_seed", -1)) == int(current.get("training_seed", -2))
            and entry.get("stage") == previous_stage
        ):
            return entry
    raise ValueError("manifest is missing a previous stage entry")


def _pilot_gate_passed(
    manifest_path: Path,
    config: Day18ExperimentConfig,
) -> bool:
    training = load_training_entries(manifest_path, include_metrics=False)
    pilot_training = {
        (str(entry["algorithm"]), int(entry["training_seed"])): entry
        for entry in training
        if entry.get("stage") == "pilot"
        and int(entry["training_seed"]) == config.training_seeds[0]
    }
    expected = {
        (algorithm, config.training_seeds[0]) for algorithm in DAY18_ALGORITHMS
    }
    if set(pilot_training) != expected or not all(
        entry.get("eligible") for entry in pilot_training.values()
    ):
        return False
    for entry in pilot_training.values():
        summary = entry.get("summary", {})
        runtime = entry.get("runtime", {})
        if not isinstance(summary, Mapping) or not isinstance(runtime, Mapping):
            return False
        required_values = (
            summary.get("last_loss"),
            summary.get("last_q_mean"),
            summary.get("last_q_max"),
            summary.get("last_target_mean"),
            summary.get("last_td_error_mean_abs"),
            runtime.get("steps_per_second"),
        )
        if any(
            value is None
            or not math.isfinite(float(value))
            or (value == 0 and field == "steps_per_second")
            for field, value in zip(
                (
                    "last_loss",
                    "last_q_mean",
                    "last_q_max",
                    "last_target_mean",
                    "last_td_error_mean_abs",
                    "steps_per_second",
                ),
                required_values,
                strict=True,
            )
        ):
            return False
    evaluations = load_evaluation_entries(manifest_path, training)
    pilot_evaluations = {
        (str(entry["algorithm"]), int(entry["training_seed"]))
        for entry in evaluations
        if entry.get("stage") == "pilot"
        and int(entry["training_seed"]) == config.training_seeds[0]
        and entry.get("eligible")
    }
    probes = load_q_probe_entries(manifest_path, training)
    pilot_probes = {
        (str(entry["algorithm"]), int(entry["training_seed"]))
        for entry in probes
        if entry.get("stage") == "pilot"
        and int(entry["training_seed"]) == config.training_seeds[0]
        and entry.get("eligible")
    }
    return pilot_evaluations == expected and pilot_probes == expected


def _completed_checkpoint(
    manifest_path: Path,
    entry: Mapping[str, Any],
) -> Path | None:
    raw = entry.get("checkpoint")
    if isinstance(raw, Mapping) and isinstance(raw.get("path"), str):
        path = resolve_manifest_reference(manifest_path, raw["path"])
        if path.is_file():
            return path
    run_dir = resolve_manifest_reference(manifest_path, entry.get("run_dir"))
    return _latest_checkpoint(run_dir)


def _prepare_run_directory(
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    resume: bool,
) -> tuple[Path, Path | None]:
    """Choose a resumable directory without appending a fresh run to partial CSV."""

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


def _run_evaluation(
    config: Day18ExperimentConfig,
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    checkpoint: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    from breakout_env import make_breakout_env

    env_factory = lambda: make_breakout_env(**breakout_environment_kwargs(config.contract))
    loaded = load_dqn_checkpoint(
        checkpoint,
        device=config.backend_config.requested_device,
        env_factory=env_factory,
    )
    training_metadata = {
        **dict(loaded.training_metadata),
        "source_day18_experiment_id": config.experiment_id,
        "source_day18_run_id": entry.get("run_id"),
        "training_seed": int(entry["training_seed"]),
        "training_budget": int(entry["target_transitions"]),
        "training_stage": entry["stage"],
        "trainer_runtime": dict(summary.get("runtime", {})),
        "environment_contract": config.contract_provenance,
        "resume_provenance": summary.get("resume_provenance"),
        "manifest": relative_path(manifest_path, start=config.repository_root),
    }
    checkpoint_metadata = {
        **dict(loaded.checkpoint_metadata),
        "source_day18_experiment_id": config.experiment_id,
        "source_day18_run_id": entry.get("run_id"),
        "training_stage": entry["stage"],
        "environment_contract": config.contract_provenance,
    }
    evaluation_id = (
        f"{config.experiment_id}-{entry['algorithm']}-seed{entry['training_seed']}-"
        f"step{entry['target_transitions']}"
    )
    result = evaluate_policy(
        loaded.model,
        episodes=config.episodes_per_seed,
        seeds=config.evaluation_seeds,
        device=config.backend_config.requested_device,
        epsilon=0.0,
        model_id=loaded.model_id,
        training_metadata=training_metadata,
        checkpoint_metadata=checkpoint_metadata,
        evaluation_id=evaluation_id,
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
            "source_day18_manifest": relative_path(
                manifest_path,
                start=config.repository_root,
            ),
            "purpose": "Day 18 Contract v2 paired evaluation",
            "raw_reward": True,
        },
    )
    evaluation = entry.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
        entry["evaluation"] = evaluation
    evaluation_dir = resolve_manifest_reference(
        manifest_path,
        evaluation.get("directory"),
    )
    results_path, episodes_path = write_evaluation_artifacts(result, evaluation_dir)
    evaluation["results"] = relative_path(results_path, start=manifest_path.parent)
    evaluation["episodes"] = relative_path(episodes_path, start=manifest_path.parent)
    evaluation["status"] = "completed"
    evaluation["summary"] = result.to_dict()["summary"]
    return result.to_dict()


def _run_q_probe(
    config: Day18ExperimentConfig,
    *,
    manifest_path: Path,
    entry: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    run_dir = resolve_manifest_reference(manifest_path, entry["run_dir"])
    output = run_dir / "diagnostics" / f"q-probe-step-{int(entry['target_transitions']):08d}.json"
    payload = analyze_checkpoint(
        checkpoint,
        config.probe_states_path,
        device=config.backend_config.requested_device,
    )
    payload["comparison"] = {
        "experiment_id": config.experiment_id,
        "run_id": entry.get("run_id"),
        "algorithm": entry["algorithm"],
        "training_seed": entry["training_seed"],
        "stage": entry["stage"],
        "target_transitions": entry["target_transitions"],
        "environment_contract": config.contract_provenance,
    }
    write_json(output, payload)
    entry["q_probe"] = relative_path(output, start=manifest_path.parent)
    return payload


def _run_one_entry(
    config: Day18ExperimentConfig,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    entry: dict[str, Any],
    resume: bool,
    skip_evaluation: bool,
    skip_q_probe: bool,
) -> None:
    target = int(entry["target_transitions"])
    stage_config = config.training_config(
        algorithm=str(entry["algorithm"]),
        seed=int(entry["training_seed"]),
        stage=str(entry["stage"]),
    )
    if stage_config.requested_device != "cuda":
        raise ValueError("Day 18 stage config must request exactly cuda")
    _check_cuda_headroom(config)

    previous = _find_previous_entry(manifest["runs"], entry)
    resume_checkpoint: Path | None = None
    if previous is not None:
        if previous.get("status") != "completed":
            raise RuntimeError(
                "cannot start a later Day 18 stage before its previous stage completes"
            )
        resume_checkpoint = _completed_checkpoint(manifest_path, previous)
        if resume_checkpoint is None:
            raise FileNotFoundError("previous stage completed without a checkpoint")
    run_dir, stage_checkpoint = _prepare_run_directory(
        manifest_path=manifest_path,
        entry=entry,
        resume=resume,
    )
    if stage_checkpoint is not None:
        resume_checkpoint = stage_checkpoint
    run_dir.mkdir(parents=True, exist_ok=True)

    entry["status"] = "running"
    entry["error"] = None
    entry["resume_from"] = (
        _checkpoint_reference(
            resume_checkpoint,
            manifest_path=manifest_path,
            step=int(_checkpoint_step_from_name(resume_checkpoint)),
        )
        if resume_checkpoint is not None
        else None
    )
    _write_manifest(manifest_path, manifest)

    environment_metadata = {
        "day18_comparison": {
            "experiment_id": config.experiment_id,
            "run_id": entry.get("run_id"),
            "pair_id": entry.get("pair_id"),
            "algorithm": entry["algorithm"],
            "training_seed": entry["training_seed"],
            "stage": entry["stage"],
            "target_transitions": target,
            "resume_from": entry.get("resume_from"),
        }
    }
    trainer = None
    env = make_breakout_vector_env(
        stage_config.num_envs,
        **breakout_environment_kwargs(config.contract),
    )
    try:
        trainer = VectorizedDQNTrainer(
            env,
            stage_config,
            run_dir=run_dir,
            resume_from=resume_checkpoint,
            metadata=environment_metadata,
            environment_contract=config.contract_provenance,
        )
        summary = trainer.train()
    finally:
        env.close()
    if summary.get("status") != "completed" or int(summary.get("total_transitions", -1)) != target:
        raise RuntimeError(
            "stage did not complete its exact transition budget: "
            f"status={summary.get('status')!r}, transitions={summary.get('total_transitions')!r}"
        )
    checkpoint = Path(str(summary.get("last_checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"completed stage checkpoint is missing: {checkpoint}")
    entry["status"] = "completed"
    entry["summary"] = compact_training_summary(summary)
    entry["checkpoint"] = _checkpoint_reference(
        checkpoint,
        manifest_path=manifest_path,
        step=target,
    )
    _write_manifest(manifest_path, manifest)

    if not skip_evaluation:
        try:
            _run_evaluation(
                config,
                manifest_path=manifest_path,
                entry=entry,
                checkpoint=checkpoint,
                summary=summary,
            )
        except Exception as error:
            evaluation = entry.setdefault("evaluation", {})
            if isinstance(evaluation, dict):
                evaluation["status"] = "failed"
                evaluation["error"] = str(error)
            entry["error"] = f"evaluation failed: {error}"
    else:
        evaluation = entry.setdefault("evaluation", {})
        if isinstance(evaluation, dict):
            evaluation["status"] = "skipped"
            evaluation["error"] = "evaluation was explicitly skipped"
    _write_manifest(manifest_path, manifest)

    if not skip_q_probe:
        try:
            _run_q_probe(
                config,
                manifest_path=manifest_path,
                entry=entry,
                checkpoint=checkpoint,
            )
        except Exception as error:
            entry["error"] = (
                f"{entry.get('error')}; Q probe failed: {error}"
                if entry.get("error")
                else f"Q probe failed: {error}"
            )
    else:
        entry["q_probe"] = None
    _write_manifest(manifest_path, manifest)


def _checkpoint_step_from_name(path: Path | None) -> int:
    if path is None:
        return 0
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else 0


def run_comparison(args: argparse.Namespace) -> tuple[int, Path, dict[str, Any]]:
    config = load_day18_config(args.config, require_probe_states=not args.skip_q_probe)
    if args.skip_evaluation or args.skip_q_probe:
        # These options are useful for local debugging but explicitly prevent
        # a false impression that the resulting manifest is formal evidence.
        pass
    manifest_path, manifest, existing = _manifest_for_run(config, args)
    selected_seeds, selected_stages = _requested_selection(config, args.stage)
    if args.stage in {"main", "all"} and not _pilot_gate_passed(manifest_path, config):
        raise RuntimeError(
            "Day 18 main stage requires the complete seed-11 250K paired pilot "
            "training/evaluation/Q-probe gate"
        )
    manifest["requested_stage"] = args.stage
    manifest["requested_training_seeds"] = list(selected_seeds)
    manifest["requested_stages"] = list(selected_stages)
    if args.dry_run:
        manifest["status"] = "planned"
        _write_manifest(manifest_path, manifest)
        return 0, manifest_path, manifest

    manifest["status"] = "running"
    _write_manifest(manifest_path, manifest)
    selected_indices = [
        index
        for index, entry in enumerate(manifest["runs"])
        if int(entry["training_seed"]) in selected_seeds
        and str(entry["stage"]) in selected_stages
    ]
    failures = 0
    interrupted = False
    for index in selected_indices:
        entry = manifest["runs"][index]
        if entry.get("status") == "completed":
            checkpoint = _completed_checkpoint(manifest_path, entry)
            if checkpoint is not None and entry.get("summary", {}).get("total_transitions") == entry.get("target_transitions"):
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
            status = "interrupted"
            run_dir = resolve_manifest_reference(manifest_path, entry["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_failure(run_dir, status=status, error=error, config=entry["training_config"])
            entry["status"] = status
            entry["error"] = str(error) or "keyboard interrupt"
        except Exception as error:
            failures += 1
            status = _failure_status(error)
            run_dir = resolve_manifest_reference(manifest_path, entry["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_failure(run_dir, status=status, error=error, config=entry["training_config"])
            entry["status"] = status
            entry["error"] = str(error)
        _write_manifest(manifest_path, manifest)
        if interrupted:
            break

    selected_statuses = [
        str(manifest["runs"][index].get("status"))
        for index in selected_indices
    ]
    if interrupted or "interrupted" in selected_statuses:
        manifest["status"] = "interrupted"
    elif any(status == "blocked" for status in selected_statuses):
        manifest["status"] = "blocked"
    elif failures or any(status == "failed" for status in selected_statuses):
        manifest["status"] = "failed"
    elif all(status == "completed" for status in selected_statuses):
        manifest["status"] = "completed"
    else:
        manifest["status"] = "partial"
    _write_manifest(manifest_path, manifest)
    return (0 if manifest["status"] == "completed" else 1), manifest_path, manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exit_code, manifest_path, manifest = run_comparison(args)
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 18 comparison could not start: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "manifest": manifest_path.as_posix(),
                "runs": [
                    {
                        "algorithm": entry.get("algorithm"),
                        "training_seed": entry.get("training_seed"),
                        "stage": entry.get("stage"),
                        "status": entry.get("status"),
                        "target_transitions": entry.get("target_transitions"),
                    }
                    for entry in manifest.get("runs", [])
                    if int(entry.get("training_seed", -1)) in set(
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
