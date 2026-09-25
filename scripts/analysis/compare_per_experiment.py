"""Build matched-seed Uniform Replay vs PER comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _git_json(commit: str, path: str) -> dict[str, Any]:
    value = json.loads(
        subprocess.check_output(
            ["git", "show", f"{commit}:{path}"],
            text=True,
            encoding="utf-8",
        )
    )
    if not isinstance(value, dict):
        raise ValueError(f"{commit}:{path} must contain a JSON object")
    return value


def _git_csv(commit: str, path: str) -> list[dict[str, str]]:
    contents = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        text=True,
        encoding="utf-8",
    )
    return list(csv.DictReader(contents.splitlines()))


def _local_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _episode_returns(payload: Mapping[str, Any], *, label: str) -> dict[tuple[int, int], float]:
    episodes = payload.get("per_episode")
    if not isinstance(episodes, list):
        raise ValueError(f"{label}: per_episode must be a list")
    values: dict[tuple[int, int], float] = {}
    for item in episodes:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}: episode records must be objects")
        eval_seed = item.get("evaluation_seed", item.get("seed"))
        episode_index = item.get("episode_index")
        episode_return = item.get("episode_return", item.get("return"))
        if not isinstance(eval_seed, int) or not isinstance(episode_index, int):
            raise ValueError(f"{label}: episode pair keys must be integers")
        if not isinstance(episode_return, (int, float)) or not math.isfinite(
            float(episode_return)
        ):
            raise ValueError(f"{label}: raw episode returns must be finite numbers")
        key = (eval_seed, episode_index)
        if key in values:
            raise ValueError(f"{label}: duplicate paired episode key {key}")
        values[key] = float(episode_return)
    return values


def _cumulative_timing_from_metrics(
    rows: Sequence[Mapping[str, str]],
    transitions: int,
) -> dict[str, float]:
    for row in reversed(rows):
        try:
            step = int(float(row.get("global_step", "")))
        except (TypeError, ValueError):
            continue
        if step != transitions:
            continue
        raw_rate = (
            row.get("steps_per_second")
            or row.get("environment_transitions_per_second")
            or row.get("sps")
        )
        try:
            rate = float(raw_rate or "")
        except (TypeError, ValueError):
            continue
        if math.isfinite(rate) and rate > 0.0:
            return {
                "elapsed_seconds": float(transitions / rate),
                "environment_transitions_per_second": rate,
            }
    raise ValueError(
        f"PER metrics do not contain a valid throughput record at {transitions} transitions"
    )


def _training_runtime(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError(f"{label}: missing training metadata")
    runtime = training.get("trainer_runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"{label}: missing checkpoint training runtime")
    wall_clock = runtime.get("wall_clock_seconds")
    transitions_per_second = runtime.get("steps_per_second")
    if (
        not isinstance(wall_clock, (int, float))
        or not math.isfinite(float(wall_clock))
        or float(wall_clock) <= 0.0
    ):
        raise ValueError(f"{label}: invalid checkpoint training wall-clock duration")
    if (
        not isinstance(transitions_per_second, (int, float))
        or not math.isfinite(float(transitions_per_second))
        or float(transitions_per_second) <= 0.0
    ):
        raise ValueError(f"{label}: invalid checkpoint transition throughput")
    return dict(runtime)


def _stage_runtime_value(
    runtime: Mapping[str, Any],
    stage_name: str,
    field: str,
) -> float | None:
    stages = runtime.get("stage_timings")
    if not isinstance(stages, Mapping):
        return None
    stage = stages.get(stage_name)
    if not isinstance(stage, Mapping):
        return None
    value = stage.get(field)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _metric_value(rows: Sequence[Mapping[str, str]], field: str) -> float | None:
    for row in reversed(rows):
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None


def _metric_value_at_transition(
    rows: Sequence[Mapping[str, str]],
    field: str,
    transitions: int,
) -> float | None:
    for row in reversed(rows):
        try:
            step = int(float(row.get("global_step", "")))
        except (TypeError, ValueError):
            continue
        if step != transitions:
            continue
        value = row.get(field)
        try:
            parsed = float(value or "")
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    parsed = [float(value) for value in values]
    if not parsed:
        raise ValueError("cannot summarize an empty value sequence")
    return {
        "n": len(parsed),
        "mean": float(statistics.mean(parsed)),
        "median": float(statistics.median(parsed)),
        "sample_std": float(statistics.stdev(parsed)) if len(parsed) > 1 else 0.0,
        "min": float(min(parsed)),
        "max": float(max(parsed)),
        "p10": float(np.percentile(parsed, 10)),
        "p90": float(np.percentile(parsed, 90)),
    }


def compare_experiment(
    config_path: Path,
    *,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _read_json(config_path)
    output_root = output_root.resolve()
    audit_path = output_root / "baseline-compatibility.json"
    audit = _read_json(audit_path)
    if audit.get("status") != "compatible" or audit.get("reuse_allowed") is not True:
        raise ValueError("the Day 20 baseline audit does not permit reuse")

    baseline = config["baseline"]
    source_commit = str(baseline["source_commit"])
    seeds = [int(seed) for seed in config["training_seeds"]]
    milestones = [int(step) for step in config["milestones"]]
    evaluation = config["evaluation"]
    expected_eval_seeds = [int(seed) for seed in evaluation["seeds"]]
    contract_payload = _read_json(Path(str(evaluation["contract"])))
    expected_contract_id = contract_payload.get("contract_id")
    expected_episode_count = len(expected_eval_seeds) * int(
        evaluation["episodes_per_seed"]
    )
    detailed_rows: list[dict[str, Any]] = []
    milestone_reports: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    uniform_cumulative_seconds = {seed: 0.0 for seed in seeds}
    per_cumulative_seconds = {seed: 0.0 for seed in seeds}
    per_update_count_cursor = {seed: 0.0 for seed in seeds}
    previous_transition_milestone = 0

    for transitions in milestones:
        per_seed: list[dict[str, Any]] = []
        uniform_seed_means: list[float] = []
        per_seed_means: list[float] = []
        paired_seed_effects: list[float] = []
        uniform_seed_medians: list[float] = []
        per_seed_medians: list[float] = []
        uniform_seed_spreads: list[float] = []
        per_seed_spreads: list[float] = []
        for training_seed in seeds:
            baseline_path = str(baseline["evaluation_template"]).format(
                seed=training_seed,
                step=transitions,
            )
            uniform_eval = _git_json(source_commit, baseline_path)
            per_eval_path = (
                output_root
                / "evaluations"
                / f"seed-{training_seed}"
                / f"step-{transitions:08d}"
                / "results.json"
            )
            per_eval = _read_json(per_eval_path)
            per_run_dir = output_root / "runs" / f"per-seed{training_seed}"
            per_summary = _read_json(per_run_dir / "summary.json")
            if per_summary.get("status") != "completed":
                raise ValueError(f"{per_run_dir}: PER training did not complete")
            if per_summary.get("total_steps") != int(config["training_config"]["total_steps"]):
                raise ValueError(f"{per_run_dir}: PER training stopped before the frozen budget")
            if uniform_eval.get("evaluation_seeds") != expected_eval_seeds:
                raise ValueError(f"{baseline_path}: Uniform evaluation seeds drifted")
            if per_eval.get("evaluation_seeds") != expected_eval_seeds:
                raise ValueError(f"{per_eval_path}: PER evaluation seeds drifted")
            if uniform_eval.get("episodes_per_seed") != evaluation["episodes_per_seed"]:
                raise ValueError(f"{baseline_path}: Uniform episodes per seed drifted")
            if per_eval.get("episodes_per_seed") != evaluation["episodes_per_seed"]:
                raise ValueError(f"{per_eval_path}: PER episodes per seed drifted")
            if uniform_eval.get("evaluation_epsilon") != evaluation["epsilon"]:
                raise ValueError(f"{baseline_path}: Uniform evaluation epsilon drifted")
            if per_eval.get("evaluation_epsilon") != evaluation["epsilon"]:
                raise ValueError(f"{per_eval_path}: PER evaluation epsilon drifted")
            if uniform_eval.get("total_episodes") != expected_episode_count:
                raise ValueError(f"{baseline_path}: Uniform episode count drifted")
            if per_eval.get("total_episodes") != expected_episode_count:
                raise ValueError(f"{per_eval_path}: PER episode count drifted")
            candidate_training = per_eval.get("training")
            if not isinstance(candidate_training, Mapping):
                raise ValueError(f"{per_eval_path}: missing PER training metadata")
            candidate_config = candidate_training.get("training_config")
            if not isinstance(candidate_config, Mapping):
                raise ValueError(f"{per_eval_path}: missing PER training config")
            for field, expected in config["training_config"].items():
                expected_value = training_seed if field == "seed" else expected
                if candidate_config.get(field) != expected_value:
                    raise ValueError(
                        f"{per_eval_path}: PER training config field {field} "
                        f"expected {expected_value!r}, got {candidate_config.get(field)!r}"
                    )
            if candidate_training.get("contract_id") != expected_contract_id:
                raise ValueError(f"{per_eval_path}: PER training Contract v2 id drifted")
            if candidate_eval_checkpoint_step := per_eval.get("checkpoint", {}).get(
                "training_steps"
            ):
                if candidate_eval_checkpoint_step != transitions:
                    raise ValueError(f"{per_eval_path}: checkpoint transition count drifted")
            elif per_eval.get("checkpoint", {}).get("step") != transitions:
                raise ValueError(f"{per_eval_path}: checkpoint transition count is missing")
            if per_eval.get("schema_version") != 2:
                raise ValueError(f"{per_eval_path}: PER evaluation schema changed")
            if per_eval.get("checkpoint", {}).get("format_version") != 2:
                raise ValueError(f"{per_eval_path}: PER checkpoint schema changed")

            uniform_returns = _episode_returns(uniform_eval, label=baseline_path)
            per_returns = _episode_returns(per_eval, label=str(per_eval_path))
            if set(uniform_returns) != set(per_returns):
                raise ValueError(
                    f"seed {training_seed} at {transitions}: paired evaluation episode keys differ"
                )
            if set(key[0] for key in uniform_returns) != set(expected_eval_seeds):
                raise ValueError(f"{baseline_path}: evaluation episode seed grouping drifted")
            paired_differences = [
                per_returns[key] - uniform_returns[key]
                for key in sorted(uniform_returns)
            ]
            uniform_values = [uniform_returns[key] for key in sorted(uniform_returns)]
            per_values = [per_returns[key] for key in sorted(per_returns)]
            uniform_stats = _describe(uniform_values)
            per_stats = _describe(per_values)
            paired_stats = _describe(paired_differences)
            uniform_seed_means.append(float(uniform_stats["mean"]))
            per_seed_means.append(float(per_stats["mean"]))
            paired_seed_effects.append(float(paired_stats["mean"]))
            uniform_seed_medians.append(float(uniform_stats["median"]))
            per_seed_medians.append(float(per_stats["median"]))
            uniform_seed_spreads.append(float(uniform_stats["sample_std"]))
            per_seed_spreads.append(float(per_stats["sample_std"]))

            baseline_metrics_path = str(baseline["training_metrics_template"]).format(
                seed=training_seed
            )
            baseline_metrics = _git_csv(source_commit, baseline_metrics_path)
            per_metrics = _local_csv(per_run_dir / "metrics.csv")
            uniform_runtime = _training_runtime(
                uniform_eval,
                label=baseline_path,
            )
            per_runtime = _training_runtime(
                per_eval,
                label=str(per_eval_path),
            )
            milestone_delta = transitions - previous_transition_milestone
            uniform_stage_seconds = float(uniform_runtime["wall_clock_seconds"])
            uniform_cumulative_seconds[training_seed] += uniform_stage_seconds
            uniform_timing = {
                "stage_elapsed_seconds": uniform_stage_seconds,
                "elapsed_seconds": uniform_cumulative_seconds[training_seed],
                "environment_transitions_per_second": (
                    milestone_delta / uniform_stage_seconds
                ),
                "source": "checkpoint trainer_runtime; cumulative stages summed",
            }
            per_checkpoint_timing = _cumulative_timing_from_metrics(
                per_metrics,
                transitions,
            )
            per_cumulative_seconds[training_seed] = per_checkpoint_timing[
                "elapsed_seconds"
            ]
            prior_per_cumulative = 0.0
            if previous_transition_milestone > 0:
                previous_checkpoint_timing = _cumulative_timing_from_metrics(
                    per_metrics,
                    previous_transition_milestone,
                )
                prior_per_cumulative = previous_checkpoint_timing["elapsed_seconds"]
            per_stage_seconds = (
                per_cumulative_seconds[training_seed] - prior_per_cumulative
            )
            if per_stage_seconds <= 0.0:
                raise ValueError(f"{per_eval_path}: non-increasing PER checkpoint wall time")
            per_timing = {
                "stage_elapsed_seconds": per_stage_seconds,
                "elapsed_seconds": per_cumulative_seconds[training_seed],
                "environment_transitions_per_second": milestone_delta / per_stage_seconds,
                "source": "metrics.csv cumulative throughput at exact milestone; one continuous PER run",
            }
            uniform_updates_per_second = uniform_runtime.get(
                "optimizer_updates_per_second"
            )
            if not isinstance(uniform_updates_per_second, (int, float)):
                uniform_updates_per_second = _metric_value(
                    baseline_metrics,
                    "optimizer_updates_per_second",
                )
            candidate_updates = _metric_value_at_transition(
                per_metrics,
                "optimizer_updates",
                transitions,
            )
            if candidate_updates is None:
                raise ValueError(
                    f"{per_run_dir}: metrics do not record optimizer_updates at {transitions}"
                )
            per_updates_per_second = (
                (candidate_updates - per_update_count_cursor[training_seed])
                / per_stage_seconds
            )
            per_update_count_cursor[training_seed] = candidate_updates
            uniform_stage_timings = uniform_runtime.get("stage_timings", {})
            per_stage_timings = per_runtime.get("stage_timings", {})
            if not isinstance(uniform_stage_timings, Mapping):
                uniform_stage_timings = {}
            if not isinstance(per_stage_timings, Mapping):
                per_stage_timings = {}
            row = {
                "transitions": transitions,
                "training_seed": training_seed,
                "uniform_mean_return": uniform_stats["mean"],
                "uniform_median_return": uniform_stats["median"],
                "uniform_std_return": uniform_stats["sample_std"],
                "uniform_p10_return": uniform_stats["p10"],
                "uniform_p90_return": uniform_stats["p90"],
                "per_mean_return": per_stats["mean"],
                "per_median_return": per_stats["median"],
                "per_std_return": per_stats["sample_std"],
                "per_p10_return": per_stats["p10"],
                "per_p90_return": per_stats["p90"],
                "paired_mean_return_difference": paired_stats["mean"],
                "paired_median_return_difference": paired_stats["median"],
                "paired_episode_count": paired_stats["n"],
                "uniform_elapsed_seconds": uniform_timing["elapsed_seconds"],
                "per_elapsed_seconds": per_timing["elapsed_seconds"],
                "per_checkpoint_runtime_seconds": per_runtime["wall_clock_seconds"],
                "per_transitions_per_second": per_timing[
                    "environment_transitions_per_second"
                ],
                "uniform_transitions_per_second": uniform_timing[
                    "environment_transitions_per_second"
                ],
                "uniform_optimizer_updates_per_second": uniform_updates_per_second,
                "per_optimizer_updates_per_second": per_updates_per_second,
                "uniform_replay_bytes": uniform_runtime.get("replay_bytes"),
                "per_replay_bytes": per_runtime.get(
                    "replay_bytes",
                    per_summary.get("replay_bytes"),
                ),
                "uniform_replay_sample_gpu_seconds": _stage_runtime_value(
                    uniform_runtime,
                    "gpu_replay_gather_cast",
                    "gpu_seconds",
                ),
                "per_sampling_gpu_seconds": _stage_runtime_value(
                    per_runtime,
                    "per_replay_sample",
                    "gpu_seconds",
                ),
                "per_priority_update_gpu_seconds": _stage_runtime_value(
                    per_runtime,
                    "per_priority_update",
                    "gpu_seconds",
                ),
                "per_sampling_wall_seconds": _stage_runtime_value(
                    per_runtime,
                    "per_replay_sample",
                    "wall_seconds",
                ),
                "per_priority_update_wall_seconds": _stage_runtime_value(
                    per_runtime,
                    "per_priority_update",
                    "wall_seconds",
                ),
                "uniform_cuda_peak_allocated_bytes": uniform_runtime.get(
                    "cuda_peak_allocated_bytes"
                ),
                "per_cuda_peak_allocated_bytes": per_runtime.get(
                    "cuda_peak_allocated_bytes"
                ),
                "uniform_checkpoint_sha256": uniform_eval.get("checkpoint", {}).get(
                    "sha256"
                ),
                "per_checkpoint_sha256": per_eval.get("checkpoint", {}).get("sha256"),
                "uniform_eval_artifact": baseline_path,
                "per_eval_artifact": per_eval_path.relative_to(output_root).as_posix(),
            }
            detailed_rows.append(row)
            per_seed.append(row)
            runtime_rows.append(
                {
                    "transitions": transitions,
                    "training_seed": training_seed,
                    "uniform": uniform_timing,
                    "per": per_timing,
                    "elapsed_time_ratio_per_over_uniform": (
                        per_timing["elapsed_seconds"]
                        / uniform_timing["elapsed_seconds"]
                    ),
                    "transition_throughput_ratio_per_over_uniform": (
                        per_timing["environment_transitions_per_second"]
                        / uniform_timing["environment_transitions_per_second"]
                    ),
                    "optimizer_updates_per_second": {
                        "uniform": uniform_updates_per_second,
                        "per": per_updates_per_second,
                    },
                    "replay_sampling": {
                        "uniform_gpu_seconds": row["uniform_replay_sample_gpu_seconds"],
                        "per_gpu_seconds": row["per_sampling_gpu_seconds"],
                        "per_host_dispatch_seconds": row["per_sampling_wall_seconds"],
                    },
                    "per_priority_update": {
                        "gpu_seconds": row["per_priority_update_gpu_seconds"],
                        "host_dispatch_seconds": row["per_priority_update_wall_seconds"],
                    },
                    "replay_bytes": {
                        "uniform": row["uniform_replay_bytes"],
                        "per": row["per_replay_bytes"],
                    },
                    "cuda_peak_allocated_bytes": {
                        "uniform": row["uniform_cuda_peak_allocated_bytes"],
                        "per": row["per_cuda_peak_allocated_bytes"],
                    },
                }
            )

        previous_transition_milestone = transitions
        milestone_reports.append(
            {
                "transitions": transitions,
                "training_seed_count": len(seeds),
                "independent_unit": "training_seed",
                "paired_eval_keys": ["evaluation_seed", "episode_index"],
                "uniform_training_seed_mean_returns": uniform_seed_means,
                "per_training_seed_mean_returns": per_seed_means,
                "paired_training_seed_mean_differences": paired_seed_effects,
                "uniform_across_training_seeds": _describe(uniform_seed_means),
                "per_across_training_seeds": _describe(per_seed_means),
                "paired_difference_across_training_seeds": _describe(paired_seed_effects),
                "uniform_training_seed_medians": uniform_seed_medians,
                "per_training_seed_medians": per_seed_medians,
                "uniform_training_seed_episode_spreads": uniform_seed_spreads,
                "per_training_seed_episode_spreads": per_seed_spreads,
                "paired_evaluation_episodes_are_not_training_replicates": True,
                "per_seed": per_seed,
            }
        )

    comparison = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "baseline_audit": "baseline-compatibility.json",
        "baseline_source_commit": source_commit,
        "candidate_algorithm": {
            "algorithm": config["training_config"]["algorithm"],
            "architecture": config["training_config"]["architecture"],
            "replay_sampling": "prioritized",
            "alpha": config["training_config"]["per_alpha"],
            "beta_start": config["training_config"]["per_beta_start"],
            "beta_end": config["training_config"]["per_beta_end"],
            "beta_anneal_transitions": config["training_config"][
                "per_beta_anneal_transitions"
            ],
        },
        "evaluation_protocol": {
            "contract": evaluation["contract"],
            "seeds": expected_eval_seeds,
            "episodes_per_seed": evaluation["episodes_per_seed"],
            "epsilon": evaluation["epsilon"],
            "raw_reward": evaluation["raw_reward"],
        },
        "interpretation_limits": [
            "Training seeds, not evaluation episodes, are the independent cross-run units.",
            "Evaluation episodes are paired by evaluation_seed and episode_index within each training seed.",
            "Cross-training-seed summaries use n=3 and are descriptive; no single episode or best seed is a success criterion.",
        ],
        "milestones": milestone_reports,
        "engineering_cost": {
            "comparison_unit": "same training seed and transition milestone",
            "source_note": (
                "Day 20 stage runtimes are accumulated across screening, pilot, and main checkpoints. "
                "PER cumulative wall time comes from metrics.csv throughput at exact transition milestones; "
                "the checkpoint runtime is retained as a cross-check. The current profiler records replay sampling and priority-update stages."
            ),
            "by_seed_and_milestone": runtime_rows,
        },
    }
    return comparison, detailed_rows


def write_comparison(
    config_path: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    comparison, rows = compare_experiment(config_path, output_root=output_root)
    comparison_path = output_root / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = output_root / "comparison.csv"
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    runtime_path = output_root / "runtime-overhead.json"
    runtime_path.write_text(
        json.dumps(comparison["engineering_cost"], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the frozen Day 20 Uniform baseline with completed PER runs."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/per_experiment.json"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/per"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        comparison = write_comparison(args.config, output_root=args.output_root)
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"PER comparison failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "experiment_id": comparison["experiment_id"],
                "milestones": [
                    {
                        "transitions": milestone["transitions"],
                        "uniform_mean": milestone["uniform_across_training_seeds"]["mean"],
                        "per_mean": milestone["per_across_training_seeds"]["mean"],
                        "paired_seed_delta": milestone[
                            "paired_difference_across_training_seeds"
                        ]["mean"],
                    }
                    for milestone in comparison["milestones"]
                ],
                "output": args.output_root.as_posix(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
