"""Build matched-seed Uniform Replay vs PER comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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


def _local_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _evaluation_dir(
    output_root: Path,
    *,
    replay_sampling: str,
    seed: int,
    transitions: int,
) -> Path:
    if replay_sampling == "prioritized":
        method_path = Path()
    elif replay_sampling == "uniform":
        method_path = Path("uniform")
    else:
        raise ValueError(f"unsupported replay sampling mode: {replay_sampling}")
    return (
        output_root
        / "evaluations"
        / method_path
        / f"seed-{seed}"
        / f"step-{transitions:08d}"
    )


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
        f"training metrics do not contain a valid throughput record at {transitions} transitions"
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


def _validate_continuous_run(
    summary: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    transitions: int,
    total_steps: int,
) -> dict[str, Any]:
    summary_runtime = summary.get("runtime")
    if not isinstance(summary_runtime, Mapping):
        raise ValueError(f"{method} seed {seed}: missing run runtime metadata")
    checkpoint_runtime = _training_runtime(
        evaluation,
        label=f"{method} seed {seed} step {transitions}",
    )
    if (
        summary.get("status") != "completed"
        or summary.get("total_steps") != total_steps
        or summary_runtime.get("stage_start_step") != 0
        or summary_runtime.get("stage_training_steps") != total_steps
        or summary_runtime.get("resume_provenance") is not None
        or summary_runtime.get("replay_rewarm_steps_remaining") != 0
        or checkpoint_runtime.get("stage_start_step") != 0
        or checkpoint_runtime.get("stage_training_steps") != transitions
    ):
        raise ValueError(
            f"{method} seed {seed} step {transitions}: staged resume or replay "
            "re-warm is incompatible with the continuous primary protocol"
        )
    training = evaluation.get("training")
    if not isinstance(training, Mapping) or training.get("resume_provenance") is not None:
        raise ValueError(
            f"{method} seed {seed} step {transitions}: checkpoint records a resumed run"
        )
    return checkpoint_runtime


def _validate_training_configs_differ_only_by_sampling(
    per_config: Mapping[str, Any],
    uniform_config: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any],
    seed: int,
    label: str,
) -> None:
    for field, expected in expected_config.items():
        expected_value = seed if field == "seed" else expected
        per_expected = "prioritized" if field == "replay_sampling" else expected_value
        uniform_expected = "uniform" if field == "replay_sampling" else expected_value
        if per_config.get(field) != per_expected:
            raise ValueError(
                f"{label}: PER training config field {field} expected "
                f"{per_expected!r}, got {per_config.get(field)!r}"
            )
        if uniform_config.get(field) != uniform_expected:
            raise ValueError(
                f"{label}: Uniform training config field {field} expected "
                f"{uniform_expected!r}, got {uniform_config.get(field)!r}"
            )
        if field != "replay_sampling" and per_config.get(field) != uniform_config.get(field):
            raise ValueError(
                f"{label}: methods differ on non-replay training field {field}"
            )


def _validate_runtime_parity(
    uniform_runtime: Mapping[str, Any],
    per_runtime: Mapping[str, Any],
    *,
    seed: int,
) -> None:
    fields = (
        "python_version",
        "pytorch_version",
        "torch_cuda_version",
        "gymnasium_version",
        "ale_version",
        "numpy_version",
        "cpu_thread_count",
        "precision",
        "resolved_device",
        "cuda_device_name",
        "gpu_model",
    )
    for field in fields:
        if uniform_runtime.get(field) != per_runtime.get(field):
            raise ValueError(
                f"seed {seed}: Uniform and PER runtime differ on {field}: "
                f"{uniform_runtime.get(field)!r} != {per_runtime.get(field)!r}"
            )


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
    primary = comparison["primary_comparison"]
    historical = comparison["historical_reference"]
    provenance = comparison["method_provenance"]
    failure_events = stability["numerical_failure_events"]["by_method"]
    completed_runs = stability["run_completion"]["completed_run_count_by_method"]
    training_seed_count = len(stability["by_training_seed"])
    lines = [
        "# Continuous PER vs Uniform Replay control report",
        "",
        f"**Conclusion:** {conclusion['statement']}",
        "",
        (
            f"Primary comparison: {primary['control']} vs {primary['candidate']}; "
            f"`{primary['only_intentional_training_config_difference']}` is the "
            "only intentional training-config difference."
        ),
        (
            "Historical Day 20 Uniform evidence is excluded from the primary A/B: "
            f"the audit classified it as `{historical['audit_status']}` because resumed "
            "milestones reset replay and re-warm. Those artifacts remain unchanged."
        ),
        (
            "Recorded source commits: Uniform `"
            f"{provenance['uniform']['source_commit']}`; PER "
            f"`{provenance['per']['source_commit']}`. Their training implementation "
            "fingerprints match; the preserved PER run predates later audit, runner, "
            "and reporting commits."
        ),
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

    engineering = comparison["engineering_cost"]["by_seed_and_milestone"]
    lines.extend(
        [
            "",
            "## Transition and replay overhead",
            "",
            "Stage throughput and optimizer rates use elapsed wall time between adjacent checkpoints. Replay profiling values are cumulative from each continuous run through that checkpoint.",
            "",
            "| Transitions | Uniform transitions/s | PER transitions/s | Uniform updates/s | PER updates/s | Uniform replay GPU s | PER sampling GPU s | PER priority-update GPU s |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for transitions in sorted({int(row["transitions"]) for row in engineering}):
        rows = [row for row in engineering if int(row["transitions"]) == transitions]

        def average(path: Sequence[str]) -> str:
            values: list[float] = []
            for row in rows:
                value: Any = row
                for key in path:
                    value = value.get(key) if isinstance(value, Mapping) else None
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            return (
                _format_report_number(statistics.mean(values), digits=2)
                if values
                else "n/a"
            )

        lines.append(
            "| {steps:,} | {uniform_rate} | {per_rate} | {uniform_updates} | "
            "{per_updates} | {uniform_sample} | {per_sample} | {priority_update} |".format(
                steps=transitions,
                uniform_rate=average(("uniform", "environment_transitions_per_second")),
                per_rate=average(("per", "environment_transitions_per_second")),
                uniform_updates=average(("optimizer_updates_per_second", "uniform")),
                per_updates=average(("optimizer_updates_per_second", "per")),
                uniform_sample=average(("replay_sampling", "uniform_gpu_seconds")),
                per_sample=average(("replay_sampling", "per_gpu_seconds")),
                priority_update=average(("per_priority_update", "gpu_seconds")),
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
            "- Stability diagnostics use the final 250K transitions from both current-code methods and summarize each run before comparing the three training seeds.",
            "- Exact Git commits differ because the completed PER runs were preserved; the recorded fingerprint covers the model, replay, trainer, environment, and evaluation source trees and matches across methods.",
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
    continuation = audit.get("continuation_semantics", {})
    if (
        audit.get("status") != "incompatible"
        or audit.get("reuse_allowed") is not False
        or continuation.get("candidate_lifecycle") != "continuous_single_run"
        or continuation.get("historical_reference_compatible_with_primary") is not False
    ):
        raise ValueError(
            "primary comparison requires the staged Day 20 reference to be "
            "explicitly excluded by the continuation audit"
        )
    manifest = _read_json(output_root / "manifest.json")
    if manifest.get("primary_comparison_status") != "continuous_uniform_vs_continuous_per":
        raise ValueError("both continuous current-code method matrices must be complete")

    historical_reference = config["baseline"]
    historical_source_commit = str(historical_reference["source_commit"])
    seeds = [int(seed) for seed in config["training_seeds"]]
    milestones = [int(step) for step in config["milestones"]]
    total_steps = int(config["training_config"]["total_steps"])
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
    run_context: dict[int, dict[str, Any]] = {}
    method_provenance = manifest.get("method_provenance", {})
    if not isinstance(method_provenance, Mapping):
        raise ValueError("experiment manifest is missing per-method source provenance")
    per_source = method_provenance.get("per")
    uniform_source = method_provenance.get("uniform")
    if not isinstance(per_source, Mapping) or not isinstance(uniform_source, Mapping):
        raise ValueError("both current method source records are required")
    per_fingerprint = per_source.get("training_source_fingerprint", {})
    uniform_fingerprint = uniform_source.get("training_source_fingerprint", {})
    if (
        not isinstance(per_fingerprint, Mapping)
        or not isinstance(uniform_fingerprint, Mapping)
        or per_fingerprint.get("sha256") != uniform_fingerprint.get("sha256")
    ):
        raise ValueError("Uniform and PER training implementation fingerprints differ")

    stability_start = milestones[-2] if len(milestones) > 1 else 0
    for training_seed in seeds:
        per_run_dir = output_root / "runs" / f"per-seed{training_seed}"
        uniform_run_dir = output_root / "runs" / f"uniform-seed{training_seed}"
        per_summary = _read_json(per_run_dir / "summary.json")
        uniform_summary = _read_json(uniform_run_dir / "summary.json")
        if (
            per_summary.get("status") != "completed"
            or uniform_summary.get("status") != "completed"
            or per_summary.get("total_steps") != total_steps
            or uniform_summary.get("total_steps") != total_steps
            or per_summary.get("replay_sampling") != "prioritized"
            or uniform_summary.get("replay_sampling") != "uniform"
        ):
            raise ValueError(f"seed {training_seed}: primary training run is incomplete")

        per_config = _read_json(per_run_dir / "config.json")
        uniform_config = _read_json(uniform_run_dir / "config.json")
        _validate_training_configs_differ_only_by_sampling(
            per_config,
            uniform_config,
            expected_config=config["training_config"],
            seed=training_seed,
            label=f"seed {training_seed}",
        )
        if per_summary.get("model_config") != uniform_summary.get("model_config"):
            raise ValueError(f"seed {training_seed}: model configurations differ")

        per_runtime = per_summary.get("runtime", {})
        uniform_runtime = uniform_summary.get("runtime", {})
        if not isinstance(per_runtime, Mapping) or not isinstance(
            uniform_runtime, Mapping
        ):
            raise ValueError(f"seed {training_seed}: training runtime metadata is missing")
        _validate_runtime_parity(uniform_runtime, per_runtime, seed=training_seed)

        for method, summary, source_record in (
            ("per", per_summary, per_source),
            ("uniform", uniform_summary, uniform_source),
        ):
            source_by_seed = source_record.get("by_training_seed", {})
            source_for_seed = (
                source_by_seed.get(str(training_seed))
                if isinstance(source_by_seed, Mapping)
                else None
            )
            summary_runtime = summary.get("runtime", {})
            if (
                not isinstance(source_for_seed, Mapping)
                or not isinstance(summary_runtime, Mapping)
                or source_for_seed.get("source_commit")
                != summary_runtime.get("git_commit_sha")
                or source_for_seed.get("training_source_fingerprint_sha256")
                != per_fingerprint.get("sha256")
            ):
                raise ValueError(
                    f"{method} seed {training_seed}: source commit/fingerprint "
                    "does not match its run summary"
                )

        per_metrics = _local_csv(per_run_dir / "metrics.csv")
        uniform_metrics = _local_csv(uniform_run_dir / "metrics.csv")
        per_checkpoint_metrics: dict[int, dict[str, Any]] = {}
        uniform_checkpoint_metrics: dict[int, dict[str, Any]] = {}
        for transitions in milestones:
            per_updates = _metric_value_at_transition(
                per_metrics,
                "optimizer_updates",
                transitions,
            )
            uniform_updates = _metric_value_at_transition(
                uniform_metrics,
                "optimizer_updates",
                transitions,
            )
            if per_updates is None or uniform_updates is None:
                raise ValueError(
                    f"seed {training_seed}: optimizer update count missing at "
                    f"{transitions} transitions"
                )
            per_checkpoint_metrics[transitions] = {
                "optimizer_updates": per_updates,
                "throughput": _cumulative_timing_from_metrics(
                    per_metrics,
                    transitions,
                ),
            }
            uniform_checkpoint_metrics[transitions] = {
                "optimizer_updates": uniform_updates,
                "throughput": _cumulative_timing_from_metrics(
                    uniform_metrics,
                    transitions,
                ),
            }

        per_diagnostics = _summarize_training_diagnostics(
            per_metrics,
            start_transition=stability_start,
            end_transition=milestones[-1],
        )
        uniform_diagnostics = _summarize_training_diagnostics(
            uniform_metrics,
            start_transition=stability_start,
            end_transition=milestones[-1],
        )
        per_actions = _cumulative_distribution_at_transition(
            per_metrics,
            transitions=milestones[-1],
            fields=_ACTION_COUNT_FIELDS,
        )
        uniform_actions = _cumulative_distribution_at_transition(
            uniform_metrics,
            transitions=milestones[-1],
            fields=_ACTION_COUNT_FIELDS,
        )
        per_policy_decisions = _policy_decision_distribution(
            per_metrics,
            transitions=milestones[-1],
        )
        uniform_policy_decisions = _policy_decision_distribution(
            uniform_metrics,
            transitions=milestones[-1],
        )
        run_context[training_seed] = {
            "per_run_dir": per_run_dir,
            "uniform_run_dir": uniform_run_dir,
            "per_summary": per_summary,
            "uniform_summary": uniform_summary,
            "per_metrics": per_checkpoint_metrics,
            "uniform_metrics": uniform_checkpoint_metrics,
            "per_diagnostics": per_diagnostics,
            "uniform_diagnostics": uniform_diagnostics,
            "per_actions": per_actions,
            "uniform_actions": uniform_actions,
            "per_policy_decisions": per_policy_decisions,
            "uniform_policy_decisions": uniform_policy_decisions,
        }

    per_update_count_cursor = {seed: 0.0 for seed in seeds}
    uniform_update_count_cursor = {seed: 0.0 for seed in seeds}
    per_cumulative_seconds = {seed: 0.0 for seed in seeds}
    uniform_cumulative_seconds = {seed: 0.0 for seed in seeds}
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
            context = run_context[training_seed]
            uniform_eval_path = (
                _evaluation_dir(
                    output_root,
                    replay_sampling="uniform",
                    seed=training_seed,
                    transitions=transitions,
                )
                / "results.json"
            )
            per_eval_path = (
                _evaluation_dir(
                    output_root,
                    replay_sampling="prioritized",
                    seed=training_seed,
                    transitions=transitions,
                )
                / "results.json"
            )
            uniform_eval = _read_json(uniform_eval_path)
            per_eval = _read_json(per_eval_path)
            per_run_dir = context["per_run_dir"]
            uniform_run_dir = context["uniform_run_dir"]
            per_summary = context["per_summary"]
            uniform_summary = context["uniform_summary"]
            if uniform_eval.get("evaluation_seeds") != expected_eval_seeds:
                raise ValueError(f"{uniform_eval_path}: Uniform evaluation seeds drifted")
            if per_eval.get("evaluation_seeds") != expected_eval_seeds:
                raise ValueError(f"{per_eval_path}: PER evaluation seeds drifted")
            if uniform_eval.get("episodes_per_seed") != evaluation["episodes_per_seed"]:
                raise ValueError(f"{uniform_eval_path}: Uniform episodes per seed drifted")
            if per_eval.get("episodes_per_seed") != evaluation["episodes_per_seed"]:
                raise ValueError(f"{per_eval_path}: PER episodes per seed drifted")
            if uniform_eval.get("evaluation_epsilon") != evaluation["epsilon"]:
                raise ValueError(f"{uniform_eval_path}: Uniform evaluation epsilon drifted")
            if per_eval.get("evaluation_epsilon") != evaluation["epsilon"]:
                raise ValueError(f"{per_eval_path}: PER evaluation epsilon drifted")
            if uniform_eval.get("total_episodes") != expected_episode_count:
                raise ValueError(f"{uniform_eval_path}: Uniform episode count drifted")
            if per_eval.get("total_episodes") != expected_episode_count:
                raise ValueError(f"{per_eval_path}: PER episode count drifted")
            candidate_training = per_eval.get("training")
            uniform_training = uniform_eval.get("training")
            if not isinstance(candidate_training, Mapping) or not isinstance(
                uniform_training, Mapping
            ):
                raise ValueError(
                    f"seed {training_seed} at {transitions}: a method evaluation "
                    "is missing training metadata"
                )
            candidate_config = candidate_training.get("training_config")
            uniform_config = uniform_training.get("training_config")
            if not isinstance(candidate_config, Mapping) or not isinstance(
                uniform_config, Mapping
            ):
                raise ValueError(
                    f"seed {training_seed} at {transitions}: a method evaluation "
                    "is missing training_config"
                )
            _validate_training_configs_differ_only_by_sampling(
                candidate_config,
                uniform_config,
                expected_config=config["training_config"],
                seed=training_seed,
                label=f"seed {training_seed} step {transitions}",
            )
            if (
                candidate_training.get("contract_id") != expected_contract_id
                or uniform_training.get("contract_id") != expected_contract_id
            ):
                raise ValueError(
                    f"seed {training_seed} at {transitions}: Contract v2 id drifted"
                )
            for method, payload, path in (
                ("Uniform", uniform_eval, uniform_eval_path),
                ("PER", per_eval, per_eval_path),
            ):
                checkpoint_metadata = payload.get("checkpoint", {})
                if (
                    not isinstance(checkpoint_metadata, Mapping)
                    or payload.get("schema_version") != 2
                    or checkpoint_metadata.get("format_version") != 2
                    or checkpoint_metadata.get("training_steps") != transitions
                    or not isinstance(checkpoint_metadata.get("sha256"), str)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", checkpoint_metadata.get("sha256", "")
                    )
                ):
                    raise ValueError(
                        f"{path}: {method} evaluation/checkpoint schema, hash, or "
                        "step drifted"
                    )
            uniform_runtime = _validate_continuous_run(
                uniform_summary,
                uniform_eval,
                method="uniform",
                seed=training_seed,
                transitions=transitions,
                total_steps=total_steps,
            )
            per_runtime = _validate_continuous_run(
                per_summary,
                per_eval,
                method="prioritized",
                seed=training_seed,
                transitions=transitions,
                total_steps=total_steps,
            )

            uniform_returns = _episode_returns(
                uniform_eval,
                label=str(uniform_eval_path),
            )
            per_returns = _episode_returns(per_eval, label=str(per_eval_path))
            if set(uniform_returns) != set(per_returns):
                raise ValueError(
                    f"seed {training_seed} at {transitions}: paired evaluation episode keys differ"
                )
            if set(key[0] for key in uniform_returns) != set(expected_eval_seeds):
                raise ValueError(
                    f"{uniform_eval_path}: evaluation episode seed grouping drifted"
                )
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

            milestone_delta = transitions - previous_transition_milestone
            prior_uniform_seconds = uniform_cumulative_seconds[training_seed]
            uniform_elapsed_seconds = float(uniform_runtime["wall_clock_seconds"])
            uniform_stage_seconds = uniform_elapsed_seconds - prior_uniform_seconds
            if uniform_stage_seconds <= 0.0:
                raise ValueError(
                    f"{uniform_eval_path}: non-increasing continuous Uniform wall time"
                )
            uniform_cumulative_seconds[training_seed] = uniform_elapsed_seconds
            uniform_timing = {
                "stage_elapsed_seconds": uniform_stage_seconds,
                "elapsed_seconds": uniform_elapsed_seconds,
                "environment_transitions_per_second": milestone_delta / uniform_stage_seconds,
                "source": "continuous-run checkpoint trainer_runtime",
            }
            prior_per_seconds = per_cumulative_seconds[training_seed]
            per_elapsed_seconds = float(per_runtime["wall_clock_seconds"])
            per_stage_seconds = per_elapsed_seconds - prior_per_seconds
            if per_stage_seconds <= 0.0:
                raise ValueError(
                    f"{per_eval_path}: non-increasing continuous PER wall time"
                )
            per_cumulative_seconds[training_seed] = per_elapsed_seconds
            per_timing = {
                "stage_elapsed_seconds": per_stage_seconds,
                "elapsed_seconds": per_elapsed_seconds,
                "environment_transitions_per_second": milestone_delta / per_stage_seconds,
                "source": "continuous-run checkpoint trainer_runtime",
            }
            uniform_updates = context["uniform_metrics"][transitions][
                "optimizer_updates"
            ]
            per_updates = context["per_metrics"][transitions]["optimizer_updates"]
            uniform_updates_per_second = (
                (uniform_updates - uniform_update_count_cursor[training_seed])
                / uniform_stage_seconds
            )
            per_updates_per_second = (
                (per_updates - per_update_count_cursor[training_seed])
                / per_stage_seconds
            )
            uniform_update_count_cursor[training_seed] = uniform_updates
            per_update_count_cursor[training_seed] = per_updates
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
                "uniform_checkpoint_runtime_seconds": uniform_runtime[
                    "wall_clock_seconds"
                ],
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
                "uniform_replay_sample_wall_seconds": _stage_runtime_value(
                    uniform_runtime,
                    "gpu_replay_gather_cast",
                    "wall_seconds",
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
                "uniform_eval_artifact": uniform_eval_path.relative_to(
                    output_root
                ).as_posix(),
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
                        "uniform_host_dispatch_seconds": row[
                            "uniform_replay_sample_wall_seconds"
                        ],
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
                training_stability_seed_results.append(
                    {
                        "training_seed": training_seed,
                        "uniform": {
                            "training_status": uniform_summary["status"],
                            "diagnostics": context["uniform_diagnostics"],
                            "actions": context["uniform_actions"],
                            "policy_decisions": context[
                                "uniform_policy_decisions"
                            ],
                        },
                        "per": {
                            "training_status": per_summary["status"],
                            "diagnostics": context["per_diagnostics"],
                            "actions": context["per_actions"],
                            "policy_decisions": context["per_policy_decisions"],
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
    runtime_provenance_by_method = {}
    for method in ("uniform", "per"):
        source_record = method_provenance[method]
        runtime_provenance_by_method[method] = {
            "source_commit": source_record["source_commit"],
            "training_source_fingerprint_sha256": source_record[
                "training_source_fingerprint"
            ]["sha256"],
            "by_training_seed": {
                str(seed): {
                    "python_version": run_context[seed][f"{method}_summary"][
                        "runtime"
                    ].get("python_version"),
                    "pytorch_version": run_context[seed][f"{method}_summary"][
                        "runtime"
                    ].get("pytorch_version"),
                    "torch_cuda_version": run_context[seed][f"{method}_summary"][
                        "runtime"
                    ].get("torch_cuda_version"),
                    "cuda_device_name": run_context[seed][f"{method}_summary"][
                        "runtime"
                    ].get("cuda_device_name"),
                    "gpu_model": run_context[seed][f"{method}_summary"][
                        "runtime"
                    ].get("gpu_model"),
                    "cpu_thread_count": run_context[seed][f"{method}_summary"][
                        "runtime"
                    ].get("cpu_thread_count"),
                    "precision": run_context[seed][f"{method}_summary"][
                        "runtime"
                    ].get("precision"),
                }
                for seed in seeds
            },
        }
    comparison = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "baseline_audit": "baseline-compatibility.json",
        "primary_comparison": {
            "control": "continuous Uniform Replay",
            "candidate": "continuous Prioritized Experience Replay",
            "lifecycle": "continuous_single_run",
            "only_intentional_training_config_difference": "replay_sampling",
            "training_source_fingerprint_sha256": per_fingerprint["sha256"],
        },
        "historical_reference": {
            "source_commit": historical_source_commit,
            "audit_status": audit["status"],
            "role": "secondary_reference_only",
            "excluded_from_primary_reason": (
                "Day 20 250K/500K milestones resumed from checkpoints without "
                "saved replay state and re-warmed a fresh replay buffer."
            ),
        },
        "method_provenance": runtime_provenance_by_method,
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
                "Uniform and PER are measured by the same current-code continuous runner. "
                "Cumulative wall time comes from each checkpoint's trainer_runtime; "
                "stage rates and optimizer updates/sec are differences between adjacent "
                "continuous checkpoints. Historical Day 20 timing is excluded."
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
                "diagnostic_event": int(
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
        description="Compare completed continuous Uniform and PER runs."
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
