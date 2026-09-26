"""Run one replay mode of the frozen three-seed comparison experiment."""

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


def _training_source_fingerprint(commit: str) -> dict[str, Any]:
    components = {
        path: _git_text("rev-parse", f"{commit}:{path}")
        for path in (
            "breakout_rl",
            "breakout_env.py",
            "configs/eval",
        )
    }
    digest = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {"sha256": digest, "components": components}


def _method_key(replay_sampling: str) -> str:
    return "per" if replay_sampling == "prioritized" else "uniform"


def _run_id(replay_sampling: str, seed: int) -> str:
    return f"{_method_key(replay_sampling)}-seed{seed}"


def _runs_manifest_key(replay_sampling: str) -> str:
    return "runs" if replay_sampling == "prioritized" else "uniform_runs"


def _evaluation_dir(
    output_root: Path,
    *,
    replay_sampling: str,
    seed: int,
    transitions: int,
) -> Path:
    method_prefix = Path() if replay_sampling == "prioritized" else Path("uniform")
    return (
        output_root
        / "evaluations"
        / method_prefix
        / f"seed-{seed}"
        / f"step-{transitions:08d}"
    )


def _method_provenance_from_summaries(
    output_root: Path,
    *,
    replay_sampling: str,
    training_seeds: Sequence[int],
) -> dict[str, Any] | None:
    by_seed: dict[str, dict[str, Any]] = {}
    source_commit: str | None = None
    source_fingerprint: dict[str, Any] | None = None
    for seed in training_seeds:
        summary_path = (
            output_root
            / "runs"
            / _run_id(replay_sampling, seed)
            / "summary.json"
        )
        if not summary_path.is_file():
            return None
        summary = _load_config(summary_path)
        runtime = summary.get("runtime", {})
        commit = runtime.get("git_commit_sha") if isinstance(runtime, Mapping) else None
        if summary.get("replay_sampling") != replay_sampling:
            raise ValueError(f"{summary_path}: replay mode does not match its run path")
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError(f"{summary_path}: missing full training source commit")
        fingerprint = _training_source_fingerprint(commit)
        if source_commit is not None and source_commit != commit:
            raise ValueError(f"{replay_sampling} summaries mix source commits")
        if (
            source_fingerprint is not None
            and source_fingerprint["sha256"] != fingerprint["sha256"]
        ):
            raise ValueError(f"{replay_sampling} summaries mix training source trees")
        source_commit = commit
        source_fingerprint = fingerprint
        by_seed[str(seed)] = {
            "source_commit": commit,
            "training_source_fingerprint_sha256": fingerprint["sha256"],
            "python_version": runtime.get("python_version"),
            "pytorch_version": runtime.get("pytorch_version"),
            "torch_cuda_version": runtime.get("torch_cuda_version"),
            "cuda_device_name": runtime.get("cuda_device_name"),
            "gpu_model": runtime.get("gpu_model"),
            "cpu_thread_count": runtime.get("cpu_thread_count"),
            "precision": runtime.get("precision"),
            "replay_sampling": replay_sampling,
        }
    if source_commit is None or source_fingerprint is None:
        return None
    return {
        "source_commit": source_commit,
        "training_source_fingerprint": source_fingerprint,
        "by_training_seed": by_seed,
    }


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
    replay_sampling: str,
) -> bool:
    runs = manifest.get(_runs_manifest_key(replay_sampling))
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
        summary_path = (
            output_root
            / "runs"
            / _run_id(replay_sampling, seed)
            / "summary.json"
        )
        if not summary_path.is_file():
            return False
        for transitions in milestones:
            evaluation = evaluations.get(str(transitions))
            if (
                not isinstance(evaluation, Mapping)
                or evaluation.get("status") != "completed"
            ):
                return False
            results_path = _evaluation_dir(
                output_root,
                replay_sampling=replay_sampling,
                seed=seed,
                transitions=transitions,
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
            "replay postprocessing did not create required evidence files: "
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
    replay_sampling: str,
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
        or summary.get("replay_sampling") != replay_sampling
        or not config_matches
    ):
        raise ValueError(
            f"{summary_path} does not match the frozen {replay_sampling} run"
        )
    return summary


def _evaluate_checkpoint(
    *,
    checkpoint: Path,
    evaluation_config: Path,
    contract_path: Path,
    output_dir: Path,
    device: str,
    replay_sampling: str,
    training_seed: int,
    transitions: int,
) -> tuple[Path, Path, dict[str, Any]]:
    results_path = output_dir / "results.json"
    episodes_path = output_dir / "episodes.csv"
    if results_path.is_file() and episodes_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        training = payload.get("training", {})
        runtime = training.get("trainer_runtime", {}) if isinstance(training, Mapping) else {}
        observed_training_config = (
            training.get("training_config", {})
            if isinstance(training, Mapping)
            else {}
        )
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
            and isinstance(observed_training_config, Mapping)
            and observed_training_config.get("replay_sampling") == replay_sampling
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
        evaluation_id=(
            f"issue12-{_method_key(replay_sampling)}-seed{training_seed}"
            f"-step{transitions}"
        ),
    )
    return run_evaluation(arguments)


def run_experiment(
    *,
    config_path: Path,
    output_root: Path,
    replay_sampling: str | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_baseline(config_path)
    audit_path = output_root / "baseline-compatibility.json"
    _write_json(audit_path, audit)
    continuation = audit.get("continuation_semantics", {})
    continuation_failures = [
        check
        for check in audit.get("failed_checks", [])
        if isinstance(check, Mapping)
        and any(
            phrase in str(check.get("name", ""))
            for phrase in (
                "staged resume",
                "replay state is saved",
                "replay history",
            )
        )
    ]
    unrelated_audit_failures = [
        check
        for check in audit.get("failed_checks", [])
        if check not in continuation_failures
    ]
    if (
        audit.get("status") != "incompatible"
        or audit.get("reuse_allowed") is not False
        or continuation.get("candidate_lifecycle") != "continuous_single_run"
        or continuation.get("historical_reference_compatible_with_primary") is not False
        or not continuation_failures
        or unrelated_audit_failures
    ):
        raise ValueError(
            "historical Day 20 evidence was not cleanly rejected for the known "
            "replay-resume lifecycle mismatch"
        )

    base_training_values = dict(config["training_config"])
    selected_sampling = replay_sampling or str(
        base_training_values.get("replay_sampling", "prioritized")
    )
    if selected_sampling not in {"uniform", "prioritized"}:
        raise ValueError("replay_sampling must be 'uniform' or 'prioritized'")
    training_values = dict(base_training_values)
    training_values["replay_sampling"] = selected_sampling
    method_key = _method_key(selected_sampling)
    manifest_runs_key = _runs_manifest_key(selected_sampling)
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
            "formal replay training requires a clean, committed worktree so run provenance is stable"
        )
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        per_matrix_complete = _completed_matrix_is_present(
            manifest,
            output_root=output_root,
            training_seeds=training_seeds,
            milestones=milestones,
            replay_sampling="prioritized",
        )
        active_matrix_complete = _completed_matrix_is_present(
            manifest,
            output_root=output_root,
            training_seeds=training_seeds,
            milestones=milestones,
            replay_sampling=selected_sampling,
        )
        existing_evaluation = manifest.get("evaluation", {})
        protocol_matches = (
            manifest.get("training_config") == base_training_values
            and manifest.get("training_seeds") == training_seeds
            and manifest.get("milestones") == milestones
            and isinstance(existing_evaluation, Mapping)
            and all(
                existing_evaluation.get(key) == value
                for key, value in evaluation.items()
            )
        )
        audit_upgrade_is_expected = (
            manifest.get("baseline_compatibility_status") == "compatible"
            and manifest.get("baseline_reuse_decision") == "reuse_day20_uniform"
            and audit.get("status") == "incompatible"
            and not unrelated_audit_failures
            and per_matrix_complete
        )
        if (
            (
                manifest.get("experiment_config_sha256") != config_sha
                and not (protocol_matches and per_matrix_complete)
            )
            or (
                manifest.get("baseline_audit_sha256") != _sha256(audit_path)
                and not audit_upgrade_is_expected
            )
            or (
                manifest.get("method_source_commits", {}).get(method_key)
                not in (None, current_commit)
                and not active_matrix_complete
            )
        ):
            raise ValueError(
                f"{manifest_path} belongs to a different frozen protocol, baseline "
                "audit, or incomplete method source commit"
            )
        manifest.setdefault("runs", {})
        manifest.setdefault("uniform_runs", {})
        manifest.setdefault("method_source_commits", {})
        manifest.setdefault("method_provenance", {})
        manifest.setdefault("method_training_configs", {})
        if audit_upgrade_is_expected or manifest.get("baseline_audit_sha256") != _sha256(audit_path):
            manifest["baseline_compatibility_status"] = audit["status"]
            manifest["baseline_reuse_decision"] = audit["decision"]
            manifest["baseline_audit_path"] = audit_path.as_posix()
            manifest["baseline_audit_sha256"] = _sha256(audit_path)
        if manifest.get("experiment_config_sha256") != config_sha:
            manifest.setdefault("prior_experiment_config_sha256", []).append(
                manifest["experiment_config_sha256"]
            )
            manifest["experiment_config_sha256"] = config_sha
            manifest["experiment_config_path"] = config_path.as_posix()
        if selected_sampling == "uniform" and method_key not in manifest["method_source_commits"]:
            manifest["method_source_commits"][method_key] = current_commit
        if manifest["method_source_commits"].get(method_key) != current_commit and active_matrix_complete:
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
            "primary_training_lifecycle": config["primary_training_lifecycle"],
            "historical_baseline_role": "secondary_reference_only",
            "required_device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_free_bytes_at_start": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
            "training_seeds": training_seeds,
            "milestones": milestones,
            "training_config": base_training_values,
            "evaluation": {
                **dict(evaluation),
                "contract_sha256": audit["protocol"]["contract_sha256"],
                "config_sha256": audit["protocol"]["evaluation_config_sha256"],
            },
            "comparison_rules": dict(config["comparison"]),
            "runs": {},
            "uniform_runs": {},
            "method_source_commits": (
                {method_key: current_commit}
                if selected_sampling == "uniform"
                else {}
            ),
            "method_provenance": {},
            "method_training_configs": {},
            "visualization_commands": [
                "python -m scripts.visualization.visualize_per_experiment"
            ],
        }
    manifest["primary_training_lifecycle"] = config["primary_training_lifecycle"]
    manifest["historical_baseline_role"] = "secondary_reference_only"
    manifest["baseline_compatibility_status"] = audit["status"]
    manifest["baseline_reuse_decision"] = audit["decision"]
    manifest["baseline_audit_path"] = audit_path.as_posix()
    manifest["baseline_audit_sha256"] = _sha256(audit_path)
    for method_sampling in ("prioritized", "uniform"):
        existing_provenance = _method_provenance_from_summaries(
            output_root,
            replay_sampling=method_sampling,
            training_seeds=training_seeds,
        )
        method_name = _method_key(method_sampling)
        if existing_provenance is not None:
            saved_provenance = manifest["method_provenance"].get(method_name)
            if (
                isinstance(saved_provenance, Mapping)
                and (
                    saved_provenance.get("source_commit")
                    != existing_provenance["source_commit"]
                    or saved_provenance.get("training_source_fingerprint", {}).get(
                        "sha256"
                    )
                    != existing_provenance["training_source_fingerprint"]["sha256"]
                )
            ):
                raise ValueError(f"{method_name} source provenance changed")
            manifest["method_provenance"][method_name] = existing_provenance
            manifest["method_source_commits"][method_name] = existing_provenance[
                "source_commit"
            ]
    manifest["method_training_configs"][method_key] = training_values
    manifest["status"] = f"training_{method_key}"
    manifest["active_method"] = selected_sampling
    _write_json(manifest_path, manifest)

    run_root = output_root / "runs"
    for seed in training_seeds:
        run_id = _run_id(selected_sampling, seed)
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
                replay_sampling=selected_sampling,
            )
            _emit(
                "training_reused",
                training_seed=seed,
                total_transitions=summary["total_steps"],
                replay_sampling=selected_sampling,
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
                        beta=(
                            beta_for_transition(
                                snapshot.global_step,
                                beta_start=config_for_seed.per_beta_start,
                                beta_end=config_for_seed.per_beta_end,
                                anneal_transitions=config_for_seed.per_beta_anneal_transitions,
                            )
                            if selected_sampling == "prioritized"
                            else None
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
                        "replay_sampling": selected_sampling,
                        "historical_reference_commit": config["baseline"]["source_commit"],
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
                    f"{selected_sampling} training seed {seed} did not complete "
                    "its transition budget"
                )
            _emit(
                "training_complete",
                training_seed=seed,
                total_transitions=summary["total_steps"],
                replay_sampling=selected_sampling,
                wall_clock_seconds=summary["runtime"]["wall_clock_seconds"],
                transitions_per_second=summary["steps_per_second"],
                run_dir=run_dir.as_posix(),
            )

        run_runtime = summary.get("runtime", {})
        method_source_commit = run_runtime.get("git_commit_sha")
        if not isinstance(method_source_commit, str) or len(method_source_commit) != 40:
            raise ValueError(f"{summary_path}: missing full training source commit")
        known_method_source = manifest["method_source_commits"].get(method_key)
        if known_method_source is not None and known_method_source != method_source_commit:
            raise ValueError(
                f"{method_key} runs mix source commits: "
                f"{known_method_source} and {method_source_commit}"
            )
        manifest["method_source_commits"][method_key] = method_source_commit
        source_fingerprint = _training_source_fingerprint(method_source_commit)
        method_provenance = manifest["method_provenance"].setdefault(
            method_key,
            {
                "source_commit": method_source_commit,
                "training_source_fingerprint": source_fingerprint,
                "by_training_seed": {},
            },
        )
        if (
            method_provenance.get("source_commit") != method_source_commit
            or method_provenance.get("training_source_fingerprint", {}).get(
                "sha256"
            )
            != source_fingerprint["sha256"]
        ):
            raise ValueError(f"{method_key} training source fingerprint drifted")
        method_provenance["by_training_seed"][str(seed)] = {
            "source_commit": method_source_commit,
            "training_source_fingerprint_sha256": source_fingerprint["sha256"],
            "python_version": run_runtime.get("python_version"),
            "pytorch_version": run_runtime.get("pytorch_version"),
            "torch_cuda_version": run_runtime.get("torch_cuda_version"),
            "cuda_device_name": run_runtime.get("cuda_device_name"),
            "gpu_model": run_runtime.get("gpu_model"),
            "cpu_thread_count": run_runtime.get("cpu_thread_count"),
            "precision": run_runtime.get("precision"),
            "replay_sampling": summary.get("replay_sampling"),
        }
        if (
            summary.get("resume_provenance") is not None
            or run_runtime.get("stage_start_step") != 0
            or run_runtime.get("replay_rewarm_steps_remaining") != 0
        ):
            raise ValueError(
                f"{summary_path}: primary {method_key} run is not a fresh, "
                "continuous training run"
            )

        run_manifest = manifest[manifest_runs_key].setdefault(
            str(seed),
            {
                "training": {
                    "status": "completed",
                    "replay_sampling": selected_sampling,
                    "summary_path": (run_dir / "summary.json").relative_to(output_root).as_posix(),
                    "metrics_path": (run_dir / "metrics.csv").relative_to(output_root).as_posix(),
                    "config_path": (run_dir / "config.json").relative_to(output_root).as_posix(),
                    "summary_sha256": _sha256(run_dir / "summary.json"),
                },
                "evaluations": {},
            },
        )
        run_manifest["training"]["status"] = "completed"
        run_manifest["training"]["replay_sampling"] = selected_sampling
        run_manifest["training"]["summary_sha256"] = _sha256(run_dir / "summary.json")
        for transitions in milestones:
            checkpoint = run_dir / "checkpoints" / f"step-{transitions:08d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"missing required {method_key} milestone checkpoint: {checkpoint}"
                )
            evaluation_dir = _evaluation_dir(
                output_root,
                replay_sampling=selected_sampling,
                seed=seed,
                transitions=transitions,
            )
            results_path, episodes_path, result = _evaluate_checkpoint(
                checkpoint=checkpoint,
                evaluation_config=evaluation_config,
                contract_path=contract_path,
                output_dir=evaluation_dir,
                device="cuda",
                replay_sampling=selected_sampling,
                training_seed=seed,
                transitions=transitions,
            )
            if (
                result.get("training", {}).get("training_seed") != seed
                or result.get("training", {}).get("training_steps") != transitions
                or result.get("checkpoint", {}).get("training_steps") != transitions
                or result.get("training", {}).get("training_config", {}).get(
                    "replay_sampling"
                )
                != selected_sampling
                or result.get("checkpoint", {}).get("sha256") != _sha256(checkpoint)
                or result.get("training", {}).get("trainer_runtime", {}).get(
                    "stage_start_step"
                )
                != 0
                or result.get("training", {}).get("trainer_runtime", {}).get(
                    "stage_training_steps"
                )
                != transitions
                or result.get("training", {}).get("resume_provenance") is not None
                or result.get("total_episodes")
                != int(evaluation["episodes_per_seed"]) * len(evaluation["seeds"])
            ):
                raise ValueError(
                    f"{results_path}: {method_key} evaluation provenance does not "
                    "match its continuous-run checkpoint"
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
                replay_sampling=selected_sampling,
                mean_raw_return=result.get("summary", {}).get("mean_return"),
                results=results_path.as_posix(),
            )

        keep = {f"step-{transitions:08d}.pt" for transitions in milestones}
        checkpoint_dir = run_dir / "checkpoints"
        for checkpoint in checkpoint_dir.glob("step-*.pt"):
            if checkpoint.name not in keep and not checkpoint.name.endswith("-diagnostic.pt"):
                checkpoint.unlink()

    per_matrix_complete = _completed_matrix_is_present(
        manifest,
        output_root=output_root,
        training_seeds=training_seeds,
        milestones=milestones,
        replay_sampling="prioritized",
    )
    uniform_matrix_complete = _completed_matrix_is_present(
        manifest,
        output_root=output_root,
        training_seeds=training_seeds,
        milestones=milestones,
        replay_sampling="uniform",
    )
    if not (per_matrix_complete and uniform_matrix_complete):
        manifest["status"] = "awaiting_continuous_uniform_control"
        manifest["primary_comparison_status"] = "incomplete"
        _write_json(manifest_path, manifest)
        _emit(
            "method_matrix_complete",
            method=selected_sampling,
            per_complete=per_matrix_complete,
            uniform_complete=uniform_matrix_complete,
            output_root=output_root.as_posix(),
        )
        return manifest

    manifest["status"] = "postprocessing"
    manifest["primary_comparison_status"] = "continuous_uniform_vs_continuous_per"
    manifest["postprocessing_code_commit"] = current_commit
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
        description="Run the three-seed replay comparison sequentially."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/per_experiment.json"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/per"))
    parser.add_argument(
        "--method",
        choices=("uniform", "prioritized"),
        default=None,
        help="replay mode to train; defaults to the mode frozen in the config",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_experiment(
            config_path=args.config,
            output_root=args.output_root,
            replay_sampling=args.method,
        )
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
