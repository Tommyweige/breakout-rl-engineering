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


_STABILITY_METRICS = (
    "loss",
    "q_mean",
    "q_max",
    "td_error_mean_abs",
    "td_error_max_abs",
    "gradient_norm",
)
_ACTION_COUNT_FIELDS = {
    "noop": "noop_count",
    "fire": "fire_count",
    "right": "right_count",
    "left": "left_count",
}
_POLICY_DECISION_FIELDS = {
    "random": "random_decision_count",
    "greedy": "greedy_decision_count",
}


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


def _last_row_at_or_before_transition(
    rows: Sequence[Mapping[str, str]],
    transitions: int,
) -> tuple[int, Mapping[str, str]] | None:
    selected: tuple[int, Mapping[str, str]] | None = None
    for row in rows:
        try:
            step = int(float(row.get("global_step", "")))
        except (TypeError, ValueError):
            continue
        if step <= transitions and (selected is None or step >= selected[0]):
            selected = (step, row)
    return selected


def _summarize_training_diagnostics(
    rows: Sequence[Mapping[str, str]],
    *,
    start_transition: int,
    end_transition: int,
) -> dict[str, Any]:
    window: list[Mapping[str, str]] = []
    observed_steps: list[int] = []
    for row in rows:
        try:
            step = int(float(row.get("global_step", "")))
        except (TypeError, ValueError):
            continue
        if start_transition <= step <= end_transition:
            window.append(row)
            observed_steps.append(step)
    if not window:
        raise ValueError(
            "training metrics do not cover diagnostic window "
            f"{start_transition}..{end_transition}"
        )

    metrics: dict[str, dict[str, float | int] | None] = {}
    non_finite_records: set[int] = set()
    non_finite_values = 0
    invalid_values = 0
    for field in _STABILITY_METRICS:
        finite_values: list[float] = []
        for row_index, row in enumerate(window):
            raw_value = row.get(field)
            if raw_value in (None, ""):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                invalid_values += 1
                non_finite_records.add(row_index)
                continue
            if not math.isfinite(value):
                non_finite_values += 1
                non_finite_records.add(row_index)
                continue
            finite_values.append(value)
        metrics[field] = _describe(finite_values) if finite_values else None

    return {
        "requested_transition_window": {
            "start": start_transition,
            "end": end_transition,
        },
        "observed_transition_window": {
            "start": min(observed_steps),
            "end": max(observed_steps),
        },
        "logged_record_count": len(window),
        "metrics": metrics,
        "non_finite_record_count": len(non_finite_records),
        "non_finite_value_count": non_finite_values,
        "invalid_value_count": invalid_values,
    }


def _cumulative_distribution_at_transition(
    rows: Sequence[Mapping[str, str]],
    *,
    transitions: int,
    fields: Mapping[str, str],
) -> dict[str, Any]:
    selected = _last_row_at_or_before_transition(rows, transitions)
    if selected is None:
        raise ValueError(f"no action counters recorded by {transitions} transitions")
    selected_step, selected_row = selected

    counts: dict[str, int] = {}
    for label, field in fields.items():
        try:
            value = float(selected_row.get(field, ""))
        except (TypeError, ValueError) as error:
            raise ValueError(f"missing cumulative action counter {field}") from error
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(f"invalid cumulative action counter {field}: {value}")
        counts[label] = int(value)
    total = sum(counts.values())
    if total != selected_step:
        raise ValueError(
            "cumulative action counters do not match the observed transition step: "
            f"counts={total}, step={selected_step}"
        )

    return {
        "observed_transition": selected_step,
        "counts": counts,
        "fractions": {label: count / total for label, count in counts.items()},
    }


def _policy_decision_distribution(
    rows: Sequence[Mapping[str, str]],
    *,
    transitions: int,
) -> dict[str, int | float]:
    selected = _last_row_at_or_before_transition(rows, transitions)
    if selected is None:
        raise ValueError(f"no policy decision counters recorded by {transitions}")
    selected_step, selected_row = selected
    counts: dict[str, int] = {}
    for label, field in _POLICY_DECISION_FIELDS.items():
        try:
            value = float(selected_row.get(field, ""))
        except (TypeError, ValueError) as error:
            raise ValueError(f"missing policy decision counter {field}") from error
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(f"invalid policy decision counter {field}: {value}")
        counts[label] = int(value)
    total = sum(counts.values())
    if total != selected_step:
        raise ValueError(
            "policy decision counters do not match the observed transition step: "
            f"counts={total}, step={selected_step}"
        )
    return {
        **counts,
        "random_fraction": counts["random"] / total,
        "greedy_fraction": counts["greedy"] / total,
    }


def _aggregate_training_stability(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    start_transition: int,
    end_transition: int,
) -> dict[str, Any]:
    metric_means: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in _STABILITY_METRICS:
        metric_means[field] = {}
        for method in ("uniform", "per"):
            means = [
                float(seed_result[method]["diagnostics"]["metrics"][field]["mean"])
                for seed_result in per_seed
                if seed_result[method]["diagnostics"]["metrics"][field] is not None
            ]
            if len(means) != len(per_seed):
                raise ValueError(
                    f"not every {method} seed has finite {field} diagnostics"
                )
            metric_means[field][method] = _describe(means)

    action_fraction_means: dict[str, dict[str, dict[str, float | int]]] = {}
    for action in _ACTION_COUNT_FIELDS:
        action_fraction_means[action] = {}
        for method in ("uniform", "per"):
            fractions = [
                float(seed_result[method]["actions"]["fractions"][action])
                for seed_result in per_seed
            ]
            action_fraction_means[action][method] = _describe(fractions)

    failure_events_by_seed = [
        {
            "training_seed": seed_result["training_seed"],
            **{
                method: int(
                    _has_non_finite_diagnostic_event(seed_result[method])
                )
                for method in ("uniform", "per")
            },
        }
        for seed_result in per_seed
    ]

    return {
        "comparison_window": {
            "start_transition": start_transition,
            "end_transition": end_transition,
        },
        "independent_unit": "training_seed",
        "aggregation": (
            "summarize logged diagnostics within each run, then summarize the "
            "three per-seed means; log records are not independent replicates"
        ),
        "diagnostic_metrics": list(_STABILITY_METRICS),
        "by_training_seed": list(per_seed),
        "across_training_seed_means": metric_means,
        "action_fraction_across_training_seeds": action_fraction_means,
        "non_finite_diagnostic_records": {
            method: sum(
                int(seed_result[method]["diagnostics"]["non_finite_record_count"])
                for seed_result in per_seed
            )
            for method in ("uniform", "per")
        },
        "non_finite_diagnostic_values": {
            method: sum(
                int(seed_result[method]["diagnostics"]["non_finite_value_count"])
                for seed_result in per_seed
            )
            for method in ("uniform", "per")
        },
        "invalid_diagnostic_values": {
            method: sum(
                int(seed_result[method]["diagnostics"]["invalid_value_count"])
                for seed_result in per_seed
            )
            for method in ("uniform", "per")
        },
        "numerical_failure_events": {
            "unit": "training run, at most one event per method and seed",
            "rule": (
                "an event is a non-finite or invalid training diagnostic in the "
                "comparison window"
            ),
            "by_training_seed": failure_events_by_seed,
            "by_method": {
                method: sum(int(row[method]) for row in failure_events_by_seed)
                for method in ("uniform", "per")
            },
        },
        "run_completion": {
            "by_training_seed": [
                {
                    "training_seed": seed_result["training_seed"],
                    "uniform": seed_result["uniform"]["training_status"],
                    "per": seed_result["per"]["training_status"],
                }
                for seed_result in per_seed
            ],
            "completed_run_count_by_method": {
                method: sum(
                    seed_result[method]["training_status"] == "completed"
                    for seed_result in per_seed
                )
                for method in ("uniform", "per")
            },
        },
        "behavioral_collapse_assessment": {
            "status": "not_classified",
            "reason": (
                "Issue #12 does not freeze a reward-based collapse threshold. "
                "The report compares paired evaluation returns and training "
                "diagnostics without inventing a behavioral-collapse label."
            ),
        },
    }


def _has_non_finite_diagnostic_event(method_result: Mapping[str, Any]) -> bool:
    diagnostics = method_result["diagnostics"]
    return (
        int(diagnostics["non_finite_record_count"]) > 0
        or int(diagnostics["non_finite_value_count"]) > 0
        or int(diagnostics["invalid_value_count"]) > 0
    )


def _build_experiment_conclusion(
    milestones: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    quality: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    for milestone in milestones:
        transitions = int(milestone["transitions"])
        uniform_mean = float(milestone["uniform_across_training_seeds"]["mean"])
        per_mean = float(milestone["per_across_training_seeds"]["mean"])
        quality.append(
            {
                "transitions": transitions,
                "uniform_mean_return": uniform_mean,
                "per_mean_return": per_mean,
                "per_minus_uniform_mean_return": per_mean - uniform_mean,
            }
        )
        rows = [row for row in runtime_rows if row["transitions"] == transitions]
        uniform_seconds = statistics.mean(
            float(row["uniform"]["elapsed_seconds"]) for row in rows
        )
        per_seconds = statistics.mean(
            float(row["per"]["elapsed_seconds"]) for row in rows
        )
        runtime.append(
            {
                "transitions": transitions,
                "uniform_mean_cumulative_seconds": uniform_seconds,
                "per_mean_cumulative_seconds": per_seconds,
                "per_over_uniform_cumulative_time_ratio": per_seconds / uniform_seconds,
            }
        )

    deltas = [row["per_minus_uniform_mean_return"] for row in quality]
    if all(delta > 0 for delta in deltas):
        category = "descriptive_per_quality_advantage"
        conclusion = (
            "PER has a positive cross-seed mean return difference at every "
            "measured transition milestone. With only three training seeds, "
            "this remains descriptive evidence rather than a superiority claim."
        )
    elif all(delta < 0 for delta in deltas):
        category = "descriptive_per_quality_disadvantage"
        conclusion = (
            "PER has a negative cross-seed mean return difference at every "
            "measured transition milestone. With only three training seeds, "
            "this remains descriptive evidence rather than a general claim."
        )
    else:
        category = "mixed_no_established_advantage"
        conclusion = (
            "The measured cross-seed return differences change direction across "
            "milestones, so this experiment does not establish a consistent PER "
            "quality advantage. The three-seed results are descriptive."
        )
    final_runtime = runtime[-1] if runtime else None
    if final_runtime is not None:
        ratio = float(final_runtime["per_over_uniform_cumulative_time_ratio"])
        conclusion += (
            f" At {final_runtime['transitions']:,} transitions, mean cumulative "
            f"training time was {ratio:.2f}x the Uniform time."
        )
    return {
        "outcome_category": category,
        "statement": conclusion,
        "independent_unit": "training_seed",
        "training_seed_count": len(
            milestones[0]["paired_training_seed_mean_differences"]
        )
        if milestones
        else 0,
        "quality_by_milestone": quality,
        "cumulative_runtime_by_milestone": runtime,
        "claim_limit": (
            "Cross-training-seed summaries use n=3 and are descriptive; paired "
            "evaluation episodes are not independent training replicates."
        ),
    }


def _format_report_number(value: float, *, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _format_signed_report_number(value: float, *, digits: int = 2) -> str:
    return f"{value:+,.{digits}f}"


def _render_markdown_report(comparison: Mapping[str, Any]) -> str:
    conclusion = comparison["conclusion"]
    stability = comparison["training_stability"]
    failure_events = stability["numerical_failure_events"]["by_method"]
    completed_runs = stability["run_completion"]["completed_run_count_by_method"]
    training_seed_count = len(stability["by_training_seed"])
    lines = [
        "# PER vs Uniform Replay experiment report",
        "",
        f"**Conclusion:** {conclusion['statement']}",
        "",
        "## Evaluation quality and cumulative training time",
        "",
        "| Transitions | Uniform mean return | PER mean return | PER − Uniform | Uniform seconds | PER seconds | PER / Uniform time |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    runtime_by_step = {
        int(row["transitions"]): row
        for row in conclusion["cumulative_runtime_by_milestone"]
    }
    for quality in conclusion["quality_by_milestone"]:
        runtime = runtime_by_step[int(quality["transitions"])]
        lines.append(
            "| {steps:,} | {uniform} | {per} | {delta} | {uniform_time} | "
            "{per_time} | {ratio:.2f}x |".format(
                steps=int(quality["transitions"]),
                uniform=_format_report_number(float(quality["uniform_mean_return"])),
                per=_format_report_number(float(quality["per_mean_return"])),
                delta=_format_signed_report_number(
                    float(quality["per_minus_uniform_mean_return"])
                ),
                uniform_time=_format_report_number(
                    float(runtime["uniform_mean_cumulative_seconds"]), digits=1
                ),
                per_time=_format_report_number(
                    float(runtime["per_mean_cumulative_seconds"]), digits=1
                ),
                ratio=float(runtime["per_over_uniform_cumulative_time_ratio"]),
            )
        )

    lines.extend(
        [
            "",
            "## Training stability diagnostics",
            "",
            "The table reports the mean of each run's logged diagnostic values, then summarizes those per-seed means (`n=3`). The metric rows are descriptive observations, not independent replicates.",
            "",
            "| Diagnostic | Uniform mean across seed means | PER mean across seed means |",
            "| --- | ---: | ---: |",
        ]
    )
    metric_labels = {
        "loss": "Loss",
        "q_mean": "Q mean",
        "q_max": "Q max",
        "td_error_mean_abs": "Mean absolute TD error",
        "td_error_max_abs": "Maximum absolute TD error",
        "gradient_norm": "Gradient norm",
    }
    for field, label in metric_labels.items():
        methods = stability["across_training_seed_means"][field]
        lines.append(
            f"| {label} | "
            f"{_format_report_number(float(methods['uniform']['mean']), digits=4)} | "
            f"{_format_report_number(float(methods['per']['mean']), digits=4)} |"
        )
    lines.extend(
        [
            "",
            "### Action and policy decision distributions",
            "",
            "Action fractions are cumulative through 500K transitions. Random and greedy fractions describe policy decisions.",
            "",
            "| Measure | Uniform | PER |",
            "| --- | ---: | ---: |",
        ]
    )
    action_methods = stability["action_fraction_across_training_seeds"]
    for action in _ACTION_COUNT_FIELDS:
        uniform_fraction = float(action_methods[action]["uniform"]["mean"])
        per_fraction = float(action_methods[action]["per"]["mean"])
        lines.append(
            f"| {action.title()} action | {uniform_fraction:.1%} | {per_fraction:.1%} |"
        )
    decisions = stability["by_training_seed"]
    uniform_random = statistics.mean(
        float(row["uniform"]["policy_decisions"]["random_fraction"])
        for row in decisions
    )
    per_random = statistics.mean(
        float(row["per"]["policy_decisions"]["random_fraction"])
        for row in decisions
    )
    lines.append(f"| Random decisions | {uniform_random:.1%} | {per_random:.1%} |")
    lines.extend(
        [
            "",
            "### Non-finite diagnostics, run completion, and collapse",
            "",
            f"Diagnostic event rule (one event per run): at least one non-finite or invalid diagnostic in the final 250K window. Events: Uniform {failure_events['uniform']}, PER {failure_events['per']}. Non-finite scalar values: Uniform {stability['non_finite_diagnostic_values']['uniform']}, PER {stability['non_finite_diagnostic_values']['per']}; invalid scalar entries: Uniform {stability['invalid_diagnostic_values']['uniform']}, PER {stability['invalid_diagnostic_values']['per']}.",
            f"Runs completing the frozen 500K budget: Uniform {completed_runs['uniform']}/{training_seed_count}, PER {completed_runs['per']}/{training_seed_count}. Incomplete or invalid runs fail the comparator before it emits a final experiment report.",
            "",
            "Behavioral score collapse is not classified because Issue #12 defines no reward-based collapse threshold.",
            stability["behavioral_collapse_assessment"]["reason"],
            "",
            "## Interpretation limits",
            "",
            f"- {conclusion['claim_limit']}",
            "- Day 20 Uniform training metrics in the pinned source contain detailed scalar diagnostics from the final 250K stage; the stability comparison uses that same transition window for PER.",
            "- The baseline and PER runs record diagnostics at different frequencies; compare per-run summaries descriptively.",
            "- Priority size is a replay diagnostic, not a model-quality measure.",
            "- This result does not promote or replace the canonical model.",
            "",
        ]
    )
    return "\n".join(lines)


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
    training_stability_seed_results: list[dict[str, Any]] = []
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
            if transitions == milestones[-1]:
                stability_start = milestones[-2] if len(milestones) > 1 else 0
                uniform_diagnostics = _summarize_training_diagnostics(
                    baseline_metrics,
                    start_transition=stability_start,
                    end_transition=transitions,
                )
                per_diagnostics = _summarize_training_diagnostics(
                    per_metrics,
                    start_transition=stability_start,
                    end_transition=transitions,
                )
                uniform_actions = _cumulative_distribution_at_transition(
                    baseline_metrics,
                    transitions=transitions,
                    fields=_ACTION_COUNT_FIELDS,
                )
                per_actions = _cumulative_distribution_at_transition(
                    per_metrics,
                    transitions=transitions,
                    fields=_ACTION_COUNT_FIELDS,
                )
                uniform_policy_decisions = _policy_decision_distribution(
                    baseline_metrics,
                    transitions=transitions,
                )
                per_policy_decisions = _policy_decision_distribution(
                    per_metrics,
                    transitions=transitions,
                )
                training_stability_seed_results.append(
                    {
                        "training_seed": training_seed,
                        "uniform": {
                            "training_status": "completed",
                            "diagnostics": uniform_diagnostics,
                            "actions": uniform_actions,
                            "policy_decisions": uniform_policy_decisions,
                        },
                        "per": {
                            "training_status": per_summary["status"],
                            "diagnostics": per_diagnostics,
                            "actions": per_actions,
                            "policy_decisions": per_policy_decisions,
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

    stability_start = milestones[-2] if len(milestones) > 1 else 0
    training_stability = _aggregate_training_stability(
        training_stability_seed_results,
        start_transition=stability_start,
        end_transition=milestones[-1],
    )
    conclusion = _build_experiment_conclusion(milestone_reports, runtime_rows)
    comparison = {
        "schema_version": 2,
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
        "training_stability": training_stability,
        "conclusion": conclusion,
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
    stability_csv_path = output_root / "training-stability.csv"
    stability_rows: list[dict[str, Any]] = []
    for seed_result in comparison["training_stability"]["by_training_seed"]:
        for method in ("uniform", "per"):
            method_result = seed_result[method]
            diagnostics = method_result["diagnostics"]
            stability_row: dict[str, Any] = {
                "training_seed": seed_result["training_seed"],
                "method": method,
                "training_status": method_result["training_status"],
                "non_finite_diagnostic_event": int(
                    _has_non_finite_diagnostic_event(method_result)
                ),
                "requested_window_start": diagnostics[
                    "requested_transition_window"
                ]["start"],
                "requested_window_end": diagnostics["requested_transition_window"][
                    "end"
                ],
                "observed_window_start": diagnostics["observed_transition_window"][
                    "start"
                ],
                "observed_window_end": diagnostics["observed_transition_window"][
                    "end"
                ],
                "logged_record_count": diagnostics["logged_record_count"],
                "non_finite_record_count": diagnostics["non_finite_record_count"],
                "non_finite_value_count": diagnostics["non_finite_value_count"],
                "invalid_value_count": diagnostics["invalid_value_count"],
                "random_decision_count": method_result["policy_decisions"][
                    "random"
                ],
                "greedy_decision_count": method_result["policy_decisions"][
                    "greedy"
                ],
                "random_decision_fraction": method_result["policy_decisions"][
                    "random_fraction"
                ],
            }
            for field in _STABILITY_METRICS:
                summary = diagnostics["metrics"][field]
                for statistic in ("n", "mean", "median", "sample_std", "p10", "p90"):
                    stability_row[f"{field}_{statistic}"] = (
                        summary[statistic] if summary is not None else None
                    )
            for action in _ACTION_COUNT_FIELDS:
                stability_row[f"{action}_count"] = method_result["actions"][
                    "counts"
                ][action]
                stability_row[f"{action}_fraction"] = method_result["actions"][
                    "fractions"
                ][action]
            stability_rows.append(stability_row)
    stability_fields = list(stability_rows[0]) if stability_rows else []
    with stability_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=stability_fields)
        writer.writeheader()
        writer.writerows(stability_rows)

    report_path = output_root / "report.md"
    report_path.write_text(
        _render_markdown_report(comparison),
        encoding="utf-8",
    )
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
