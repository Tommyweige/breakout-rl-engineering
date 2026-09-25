"""Run the frozen three-seed PER training and evaluation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from breakout_env import make_breakout_vector_env
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.prioritized_replay import beta_for_transition
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.vectorized import (
    VectorizedDQNTrainer,
    VectorizedTrainingStepSnapshot,
)
from scripts.analysis.audit_per_baseline import audit_baseline
from scripts.evaluation.evaluate_vectorized_dqn import run_evaluation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def _git_json(commit: str, path: str) -> dict[str, Any]:
    value = json.loads(_git_text("show", f"{commit}:{path}"))
    if not isinstance(value, dict):
        raise ValueError(f"{commit}:{path} must contain a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _completed_matrix_is_present(
    manifest: Mapping[str, Any],
    *,
    output_root: Path,
    training_seeds: Sequence[int],
    milestones: Sequence[int],
) -> bool:
    runs = manifest.get("runs")
    if not isinstance(runs, Mapping):
        return False
    for seed in training_seeds:
        run_record = runs.get(str(seed))
        if not isinstance(run_record, Mapping):
            return False
        training = run_record.get("training")
        evaluations = run_record.get("evaluations")
        if (
            not isinstance(training, Mapping)
            or training.get("status") != "completed"
            or not isinstance(evaluations, Mapping)
        ):
            return False
        summary_path = output_root / "runs" / f"per-seed{seed}" / "summary.json"
        if not summary_path.is_file():
            return False
        for transitions in milestones:
            evaluation = evaluations.get(str(transitions))
            if (
                not isinstance(evaluation, Mapping)
                or evaluation.get("status") != "completed"
            ):
                return False
            results_path = output_root / "evaluations" / f"seed-{seed}" / (
                f"step-{transitions:08d}"
            ) / "results.json"
            episodes_path = results_path.with_name("episodes.csv")
            if not results_path.is_file() or not episodes_path.is_file():
                return False
    return True


def _run_postprocessing(
    *,
    config_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    repository_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.analysis.compare_per_experiment",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=repository_root,
    )
    comparison_path = output_root / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if not isinstance(comparison, dict):
        raise ValueError(f"{comparison_path} must contain a JSON object")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.visualization.visualize_per_experiment",
            "--comparison",
            str(comparison_path),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=repository_root,
    )
    figures = [
        output_root / "visualizations" / "evaluation-by-transitions.png",
        output_root / "visualizations" / "evaluation-by-wall-clock.png",
        output_root / "visualizations" / "priority-diagnostics-seed11.png",
    ]
    evidence_files = [
        *figures,
        output_root / "training-stability.csv",
        output_root / "report.md",
    ]
    missing_outputs = [path for path in evidence_files if not path.is_file()]
    if missing_outputs:
        raise FileNotFoundError(
            "PER postprocessing did not create required evidence files: "
            + ", ".join(str(path) for path in missing_outputs)
        )
    return comparison, figures


def _emit(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **values,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return config


def _validate_completed_run(
    run_dir: Path,
    *,
    config: DQNConfig,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    if not summary_path.is_file() or not config_path.is_file():
        raise FileExistsError(
            f"{run_dir} exists without completed summary/config; inspect or move this partial run"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = json.loads(json.dumps(config.to_dict()))
    config_matches = all(
        observed_config.get(name) == value
        for name, value in expected_config.items()
    )
    if (
        summary.get("status") != "completed"
        or summary.get("seed") != config.seed
        or summary.get("total_steps") != config.total_steps
        or summary.get("replay_sampling") != "prioritized"
        or not config_matches
    ):
        raise ValueError(f"{summary_path} does not match the frozen PER run")
    return summary


def _evaluate_checkpoint(
    *,
    checkpoint: Path,
    evaluation_config: Path,
    contract_path: Path,
    output_dir: Path,
    device: str,
    training_seed: int,
    transitions: int,
) -> tuple[Path, Path, dict[str, Any]]:
    results_path = output_dir / "results.json"
    episodes_path = output_dir / "episodes.csv"
    if results_path.is_file() and episodes_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        training = payload.get("training", {})
        runtime = training.get("trainer_runtime", {}) if isinstance(training, Mapping) else {}
        checkpoint_metadata = payload.get("checkpoint", {})
        evaluation_config_payload = _load_config(evaluation_config)
        if (
            payload.get("evaluation_seeds") == evaluation_config_payload.get("seeds")
            and payload.get("episodes_per_seed")
            == evaluation_config_payload.get("episodes_per_seed")
            and payload.get("evaluation_epsilon") == evaluation_config_payload.get("epsilon")
            and isinstance(training, Mapping)
            and training.get("training_seed") == training_seed
            and training.get("training_steps") == transitions
            and isinstance(runtime, Mapping)
            and isinstance(runtime.get("wall_clock_seconds"), (int, float))
            and isinstance(checkpoint_metadata, Mapping)
            and checkpoint_metadata.get("training_steps") == transitions
            and checkpoint_metadata.get("format_version") == 2
        ):
            return results_path, episodes_path, payload

    arguments = argparse.Namespace(
        checkpoint=checkpoint,
        config=evaluation_config,
        contract=contract_path,
        device=device,
        output_dir=output_dir,
        evaluation_id=f"issue12-per-seed{training_seed}-step{transitions}",
    )
    return run_evaluation(arguments)


def run_experiment(
    *,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_baseline(config_path)
    audit_path = output_root / "baseline-compatibility.json"
    _write_json(audit_path, audit)
    if audit.get("status") != "compatible" or audit.get("reuse_allowed") is not True:
        raise ValueError("baseline audit failed; refusing to start PER training")

    training_values = dict(config["training_config"])
    training_seeds = [int(seed) for seed in config["training_seeds"]]
    milestones = [int(step) for step in config["milestones"]]
    evaluation = config["evaluation"]
    evaluation_config = Path(evaluation["config"]).resolve()
    contract_path = Path(evaluation["contract"]).resolve()
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)

    baseline_manifest = _git_json(
        str(config["baseline"]["source_commit"]),
        str(config["baseline"]["manifest_path"]),
    )
    minimum_free_cuda_bytes = int(
        baseline_manifest["protocol"].get("cuda_headroom_bytes", 1_073_741_824)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("PER experiment requires CUDA; refusing to fall back to CPU")
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device("cuda"))
    if free_bytes < minimum_free_cuda_bytes:
        raise RuntimeError(
            "insufficient CUDA headroom before formal run: "
            f"free={free_bytes}, required={minimum_free_cuda_bytes}"
        )

    config_sha = _sha256(config_path)
    current_commit = _git_text("rev-parse", "HEAD")
    dirty_state = _git_text("status", "--porcelain", "--untracked-files=normal")
    if dirty_state:
        raise RuntimeError(
            "formal PER training requires a clean, committed worktree so run provenance is stable"
        )
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matrix_complete = _completed_matrix_is_present(
            manifest,
            output_root=output_root,
            training_seeds=training_seeds,
            milestones=milestones,
        )
        if (
            manifest.get("experiment_config_sha256") != config_sha
            or manifest.get("baseline_audit_sha256") != _sha256(audit_path)
            or (
                manifest.get("implementation_base_commit") != current_commit
                and not matrix_complete
            )
        ):
            raise ValueError(
                f"{manifest_path} belongs to a different config, baseline audit, "
                "or incomplete training commit"
            )
        if manifest.get("implementation_base_commit") != current_commit:
            manifest["postprocessing_code_commit"] = current_commit
    else:
        manifest = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_config_path": config_path.as_posix(),
            "experiment_config_sha256": config_sha,
            "baseline_audit_path": audit_path.as_posix(),
            "baseline_audit_sha256": _sha256(audit_path),
            "baseline_compatibility_status": audit["status"],
            "baseline_reuse_decision": audit["decision"],
            "baseline_source_commit": config["baseline"]["source_commit"],
            "implementation_base_commit": current_commit,
            "required_device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_free_bytes_at_start": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
            "training_seeds": training_seeds,
            "milestones": milestones,
            "training_config": training_values,
            "evaluation": {
                **dict(evaluation),
                "contract_sha256": audit["protocol"]["contract_sha256"],
                "config_sha256": audit["protocol"]["evaluation_config_sha256"],
            },
            "comparison_rules": dict(config["comparison"]),
            "runs": {},
            "visualization_commands": [
                "python -m scripts.visualization.visualize_per_experiment"
            ],
        }
        _write_json(manifest_path, manifest)

    run_root = output_root / "runs"
    for seed in training_seeds:
        run_id = f"per-seed{seed}"
        run_dir = run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_for_seed = DQNConfig.from_dict(training_values).with_overrides(
            seed=seed,
            contract_id=contract.contract_id,
            contract_path=contract_path.as_posix(),
        )
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            summary = _validate_completed_run(
                run_dir,
                config=config_for_seed,
            )
            _emit(
                "training_reused",
                training_seed=seed,
                total_transitions=summary["total_steps"],
                run_dir=run_dir.as_posix(),
            )
        else:
            if any(run_dir.iterdir()):
                raise FileExistsError(
                    f"{run_dir} has partial contents without a completed summary"
                )
            environment = make_breakout_vector_env(
                config_for_seed.num_envs,
                **breakout_environment_kwargs(contract),
            )
            trainer: VectorizedDQNTrainer | None = None
            progress_interval_seconds = 30.0
            last_reported_at = time.perf_counter()
            run_started = last_reported_at

            def on_step(snapshot: VectorizedTrainingStepSnapshot) -> None:
                nonlocal last_reported_at
                now = time.perf_counter()
                if (
                    now - last_reported_at >= progress_interval_seconds
                    or snapshot.global_step in milestones
                ):
                    last_reported_at = now
                    elapsed = max(now - run_started, 1e-9)
                    _emit(
                        "training_progress",
                        training_seed=seed,
                        transitions=snapshot.global_step,
                        target_transitions=config_for_seed.total_steps,
                        elapsed_seconds=elapsed,
                        transitions_per_second=snapshot.global_step / elapsed,
                        optimizer_updates=snapshot.optimizer_updates,
                        replay_size=snapshot.replay_size,
                        beta=beta_for_transition(
                            snapshot.global_step,
                            beta_start=config_for_seed.per_beta_start,
                            beta_end=config_for_seed.per_beta_end,
                            anneal_transitions=config_for_seed.per_beta_anneal_transitions,
                        ),
                    )

            try:
                trainer = VectorizedDQNTrainer(
                    environment,
                    config_for_seed,
                    run_dir=run_dir,
                    environment_contract=contract.to_dict(),
                    metadata={
                        "experiment_id": config["experiment_id"],
                        "issue": config["issue"],
                        "baseline_source_commit": config["baseline"]["source_commit"],
                        "experiment_config_sha256": config_sha,
                    },
                    on_step=on_step,
                )
                summary = trainer.train()
            finally:
                environment.close()
            if (
                summary.get("status") != "completed"
                or summary.get("total_steps") != config_for_seed.total_steps
            ):
                raise RuntimeError(
                    f"PER training seed {seed} did not complete its transition budget"
                )
            _emit(
                "training_complete",
                training_seed=seed,
                total_transitions=summary["total_steps"],
                wall_clock_seconds=summary["runtime"]["wall_clock_seconds"],
                transitions_per_second=summary["steps_per_second"],
                run_dir=run_dir.as_posix(),
            )

        run_manifest = manifest["runs"].setdefault(
            str(seed),
            {
                "training": {
                    "status": "completed",
                    "summary_path": (run_dir / "summary.json").relative_to(output_root).as_posix(),
                    "metrics_path": (run_dir / "metrics.csv").relative_to(output_root).as_posix(),
                    "config_path": (run_dir / "config.json").relative_to(output_root).as_posix(),
                    "summary_sha256": _sha256(run_dir / "summary.json"),
                },
                "evaluations": {},
            },
        )
        for transitions in milestones:
            checkpoint = run_dir / "checkpoints" / f"step-{transitions:08d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"missing required PER milestone checkpoint: {checkpoint}"
                )
            evaluation_dir = (
                output_root
                / "evaluations"
                / f"seed-{seed}"
                / f"step-{transitions:08d}"
            )
            results_path, episodes_path, result = _evaluate_checkpoint(
                checkpoint=checkpoint,
                evaluation_config=evaluation_config,
                contract_path=contract_path,
                output_dir=evaluation_dir,
                device="cuda",
                training_seed=seed,
                transitions=transitions,
            )
            if (
                result.get("training", {}).get("training_seed") != seed
                or result.get("training", {}).get("training_steps") != transitions
                or result.get("checkpoint", {}).get("training_steps") != transitions
                or result.get("total_episodes")
                != int(evaluation["episodes_per_seed"]) * len(evaluation["seeds"])
            ):
                raise ValueError(
                    f"{results_path}: PER evaluation provenance does not match its checkpoint"
                )
            run_manifest["evaluations"][str(transitions)] = {
                "status": "completed",
                "results_path": results_path.relative_to(output_root).as_posix(),
                "episodes_path": episodes_path.relative_to(output_root).as_posix(),
                "results_sha256": _sha256(results_path),
                "checkpoint_sha256": result.get("checkpoint", {}).get("sha256"),
                "mean_raw_return": result.get("summary", {}).get("mean_return"),
            }
            _write_json(manifest_path, manifest)
            _emit(
                "evaluation_complete",
                training_seed=seed,
                transitions=transitions,
                mean_raw_return=result.get("summary", {}).get("mean_return"),
                results=results_path.as_posix(),
            )

        keep = {f"step-{transitions:08d}.pt" for transitions in milestones}
        checkpoint_dir = run_dir / "checkpoints"
        for checkpoint in checkpoint_dir.glob("step-*.pt"):
            if checkpoint.name not in keep and not checkpoint.name.endswith("-diagnostic.pt"):
                checkpoint.unlink()

    manifest["status"] = "postprocessing"
    _write_json(manifest_path, manifest)
    comparison, figures = _run_postprocessing(
        config_path=config_path,
        output_root=output_root,
    )
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["comparison_path"] = (output_root / "comparison.json").relative_to(output_root).as_posix()
    manifest["comparison_sha256"] = _sha256(output_root / "comparison.json")
    manifest["conclusion"] = comparison["conclusion"]
    manifest["training_stability_path"] = (
        output_root / "training-stability.csv"
    ).relative_to(output_root).as_posix()
    manifest["training_stability_sha256"] = _sha256(
        output_root / "training-stability.csv"
    )
    manifest["report_path"] = (output_root / "report.md").relative_to(output_root).as_posix()
    manifest["report_sha256"] = _sha256(output_root / "report.md")
    manifest["runtime_overhead_path"] = (
        output_root / "runtime-overhead.json"
    ).relative_to(output_root).as_posix()
    manifest["visualizations"] = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in figures
    ]
    _write_json(manifest_path, manifest)
    _emit(
        "experiment_complete",
        experiment_id=config["experiment_id"],
        status=manifest["status"],
        comparison_milestones=len(comparison["milestones"]),
        output_root=output_root.as_posix(),
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen three-seed PER-vs-Uniform experiment sequentially."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/per_experiment.json"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/per"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_experiment(config_path=args.config, output_root=args.output_root)
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"PER experiment failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
