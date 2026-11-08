"""Day 14 gate provenance: which training run, manifest, and checkpoint step a
formal evaluation is allowed to claim."""

from __future__ import annotations

import csv
import json
import math
import operator
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from breakout_rl.artifacts import portable_artifact_path
from breakout_rl.experiments import load_experiment_config


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(payload)


def _resolve_optional_reference(path: str | Path, *, source: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    for option in (candidate, Path.cwd() / candidate, source.parent / candidate):
        if option.is_file():
            return option.resolve()
    return (Path.cwd() / candidate).resolve()


def _day14_gate_evidence(
    *,
    run_dir: Path | None,
    summary: Mapping[str, Any],
    metrics_path: Path | None,
    variant: Mapping[str, Any],
    expected_step: int,
    gpu_profiling_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Turn Day 14's observed long-run artifacts into an explicit Gate A result."""

    reasons: list[str] = []
    summary_status = summary.get("status") == "completed"
    summary_step = summary.get("total_steps") == expected_step
    summary_run = summary.get("run_id") == variant.get("run_id")
    summary_provenance = summary_status and summary_step and summary_run
    if not summary_provenance:
        reasons.append("summary.json is not a completed, matching final run")

    baseline_mean = gpu_profiling_summary.get("baseline_recent_episode_return")
    baseline_count = gpu_profiling_summary.get("baseline_recent_episode_count")
    try:
        baseline_mean = float(baseline_mean)
        baseline_count = int(baseline_count)
    except (TypeError, ValueError):
        baseline_mean = None
        baseline_count = None

    completed_returns: list[tuple[int, float]] = []
    diagnostic_fields = (
        "loss",
        "q_mean",
        "q_max",
        "q_min",
        "target_mean",
        "target_max",
        "td_error_mean_abs",
        "td_error_max_abs",
        "gradient_norm",
    )
    diagnostic_counts = {
        field: {"observed": 0, "finite": 0} for field in diagnostic_fields
    }
    if metrics_path is not None and metrics_path.is_file():
        try:
            with metrics_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    try:
                        step = int(row["global_step"])
                    except (KeyError, TypeError, ValueError):
                        reasons.append("metrics.csv contains an invalid global_step")
                        continue
                    raw_return = row.get("raw_episode_return", "")
                    if raw_return not in (None, ""):
                        try:
                            parsed_return = float(raw_return)
                        except (TypeError, ValueError):
                            parsed_return = float("nan")
                        if math.isfinite(parsed_return):
                            completed_returns.append((step, parsed_return))
                        else:
                            reasons.append("metrics.csv contains a non-finite episode return")
                    for field in diagnostic_fields:
                        raw_value = row.get(field, "")
                        if raw_value in (None, ""):
                            continue
                        diagnostic_counts[field]["observed"] += 1
                        try:
                            parsed_value = float(raw_value)
                        except (TypeError, ValueError):
                            parsed_value = float("nan")
                        if math.isfinite(parsed_value):
                            diagnostic_counts[field]["finite"] += 1
        except (OSError, csv.Error) as error:
            reasons.append(f"unable to read metrics.csv: {error}")
    else:
        reasons.append("Day 14 metrics.csv is unavailable")

    completed_returns.sort(key=lambda item: item[0])
    final_recent_mean: float | None = None
    return_signal = False
    if baseline_mean is not None and baseline_count is not None and baseline_count > 0:
        if len(completed_returns) >= baseline_count:
            final_values = [value for _, value in completed_returns[-baseline_count:]]
            final_recent_mean = float(fmean(final_values))
            return_signal = final_recent_mean > baseline_mean
    if not return_signal:
        reasons.append(
            "100K recent return does not show an interpretable improvement over the 10K reference"
        )

    summary_diagnostic_fields = (
        "last_loss",
        "last_q_mean",
        "last_q_max",
        "last_q_min",
        "last_target_mean",
        "last_target_max",
        "last_td_error_mean_abs",
        "last_td_error_max_abs",
    )
    summary_diagnostics_finite = all(
        isinstance(summary.get(field), Real)
        and not isinstance(summary.get(field), bool)
        and math.isfinite(float(summary[field]))
        for field in summary_diagnostic_fields
    )
    metrics_diagnostics_finite = all(
        counts["observed"] > 0 and counts["observed"] == counts["finite"]
        for counts in diagnostic_counts.values()
    )
    diagnostics_healthy = summary_diagnostics_finite and metrics_diagnostics_finite
    if not diagnostics_healthy:
        reasons.append("Day 14 diagnostics contain missing or non-finite required values")

    selection_rule = gpu_profiling_summary.get("selection_rule")
    guardrails_passed = gpu_profiling_summary.get("regression_guardrails_passed") is True
    multiple_episode_evidence = int(summary.get("episodes", 0) or 0) > 1
    selection_evidence = bool(selection_rule) and guardrails_passed and multiple_episode_evidence
    if not selection_evidence:
        reasons.append(
            "config selection lacks recorded quality guardrails and multi-episode evidence"
        )

    provenance_complete = all(
        variant.get(field)
        for field in ("run_id", "config_path", "run_dir", "status", "step_budget")
    ) and summary_provenance
    if not provenance_complete:
        reasons.append("Day 14 checkpoint provenance is incomplete")

    passed = return_signal and diagnostics_healthy and selection_evidence and provenance_complete

    return {
        "status": "passed" if passed else "not_satisfied",
        "criteria": {
            "return_signal": return_signal,
            "diagnostics_healthy": diagnostics_healthy,
            "selection_not_single_best_episode": selection_evidence,
            "checkpoint_provenance_complete": provenance_complete,
        },
        "return_signal": {
            "reference_10k_recent_mean": baseline_mean,
            "reference_recent_episode_count": baseline_count,
            "final_100k_recent_mean": final_recent_mean,
            "improvement": (
                None
                if baseline_mean is None or final_recent_mean is None
                else float(final_recent_mean - baseline_mean)
            ),
        },
        "diagnostics": {
            "summary_status": summary.get("status"),
            "summary_values_finite": summary_diagnostics_finite,
            "metrics_values_finite": metrics_diagnostics_finite,
            "field_counts": diagnostic_counts,
            "summary_source": (
                portable_artifact_path(run_dir / "summary.json") if run_dir is not None else None
            ),
            "metrics_source": (
                portable_artifact_path(metrics_path) if metrics_path is not None else None
            ),
        },
        "selection": {
            "rationale": selection_rule,
            "selected_profile_run_id": gpu_profiling_summary.get("selected_run_id"),
            "regression_guardrails_passed": guardrails_passed,
            "final_run_episode_count": summary.get("episodes"),
        },
        "provenance": {
            "run_id": variant.get("run_id"),
            "expected_step": expected_step,
            "summary_step": summary.get("total_steps"),
            "config_reference": variant.get("config_path"),
            "run_dir": portable_artifact_path(run_dir) if run_dir is not None else None,
        },
        "reasons": reasons,
    }


def load_day14_provenance(
    path: str | Path,
    *,
    profiling_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the completed single final variant from the latest manifest."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _read_json_mapping(source)
    if payload.get("status") != "completed":
        raise ValueError(f"{source}: Day 14 manifest is not completed")
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != 1:
        raise ValueError(f"{source}: expected one final Day 14 variant")
    variant = variants[0]
    if not isinstance(variant, Mapping):
        raise ValueError(f"{source}: final variant must be an object")
    raw_config = variant.get("config_values", {})
    if not isinstance(raw_config, Mapping):
        raw_config = {}
    config_values = dict(raw_config)
    expected_step = variant.get("step_budget", config_values.get("total_steps"))
    try:
        expected_step = operator.index(expected_step)
    except TypeError as error:
        raise TypeError("Day 14 step budget must be an integer") from error
    if expected_step < 1:
        raise ValueError("Day 14 step budget must be at least 1")

    raw_config_reference = variant.get("config_path")
    if raw_config_reference is None:
        base_config = payload.get("base_config", {})
        if isinstance(base_config, Mapping):
            raw_config_reference = base_config.get("config_path")
    config_reference = None
    config_path: Path | None = None
    if isinstance(raw_config_reference, str):
        candidate = Path(raw_config_reference)
        config_path = (
            candidate
            if candidate.is_absolute()
            else source.parent / candidate
        ).resolve()
        config_reference = portable_artifact_path(config_path)
        if config_path.is_file():
            effective_config = load_experiment_config(config_path)
            effective_values = dict(effective_config.values)
            effective_values.update(config_values)
            config_values = effective_values
    config_values.setdefault("replay_backend", "cpu")

    run_dir: Path | None = None
    raw_run_dir = variant.get("run_dir")
    if isinstance(raw_run_dir, str) and raw_run_dir.strip():
        candidate = Path(raw_run_dir)
        run_dir = (candidate if candidate.is_absolute() else source.parent / candidate).resolve()
    runtime: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    metrics_path: Path | None = None
    if run_dir is not None and (run_dir / "config.json").is_file():
        run_config = _read_json_mapping(run_dir / "config.json")
        raw_runtime = run_config.get("runtime")
        if isinstance(raw_runtime, Mapping):
            runtime = dict(raw_runtime)
    if run_dir is not None and (run_dir / "summary.json").is_file():
        summary = _read_json_mapping(run_dir / "summary.json")
    if run_dir is not None and (run_dir / "metrics.csv").is_file():
        metrics_path = run_dir / "metrics.csv"

    gpu_profiling_summary: dict[str, Any] = {}
    profiling_source: str | None = None
    if profiling_report_path is not None:
        profiling_path = _resolve_optional_reference(
            profiling_report_path,
            source=source,
        )
        if not profiling_path.is_file():
            raise FileNotFoundError(profiling_path)
        profiling_report = _read_json_mapping(profiling_path)
        runs = profiling_report.get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"{profiling_path}: profiling runs must be an array")
        matching_runs = [
            candidate
            for candidate in runs
            if isinstance(candidate, Mapping)
            and candidate.get("status") == "completed"
            and candidate.get("batch_size") == config_values.get("batch_size")
        ]
        if len(matching_runs) != 1:
            raise ValueError(
                f"{profiling_path}: expected exactly one completed profiling run for "
                f"batch_size={config_values.get('batch_size')}, found {len(matching_runs)}"
            )
        selected_run = matching_runs[0]
        if selected_run.get("completed_steps") != selected_run.get("expected_steps"):
            raise ValueError(
                f"{profiling_path}: selected profiling run did not complete its step budget"
            )
        profiling = selected_run.get("profiling", {})
        if not isinstance(profiling, Mapping):
            profiling = {}
        profiling_source = portable_artifact_path(profiling_path)
        gpu_profiling_summary = {
            "source": profiling_source,
            "selected_batch_size": config_values.get("batch_size"),
            "selection_rule": profiling_report.get("selection_rule", {}),
            "selected_run_id": selected_run.get("run_id"),
            "selected_run_status": selected_run.get("status"),
            "baseline_recent_episode_return": selected_run.get(
                "mean_recent_episode_return"
            ),
            "baseline_recent_episode_count": (
                selected_run.get("recent_return_trend", {}).get("count")
                if isinstance(selected_run.get("recent_return_trend"), Mapping)
                else None
            ),
            "regression_guardrails_passed": (
                selected_run.get("regression_guardrails", {}).get("guardrails_passed")
                if isinstance(selected_run.get("regression_guardrails"), Mapping)
                else None
            ),
            "end_to_end_sps": selected_run.get("end_to_end_sps"),
            "training_samples_per_second": selected_run.get("training_samples_per_second"),
            "profiling": {
                "sample_csv": (
                    portable_artifact_path((profiling_path.parent / profiling["sample_csv"]).resolve())
                    if isinstance(profiling.get("sample_csv"), str)
                    else None
                ),
                "sampling_method": profiling.get("sampling_method"),
                "gpu_utilization_percent": profiling.get("gpu_utilization_percent"),
                "gpu_power_watts": profiling.get("gpu_power_watts"),
                "gpu_memory_used_bytes": profiling.get("gpu_memory_used_bytes"),
                "gpu_memory_total_bytes": profiling.get("gpu_memory_total_bytes"),
            },
        }

    day14_gate = _day14_gate_evidence(
        run_dir=run_dir,
        summary=summary,
        metrics_path=metrics_path,
        variant=variant,
        expected_step=expected_step,
        gpu_profiling_summary=gpu_profiling_summary,
    )

    return {
        "manifest_path": portable_artifact_path(source),
        "source_of_truth": "latest Day 14 final manifest and referenced run artifacts",
        "experiment_id": payload.get("experiment_id", source.parent.name),
        "status": payload.get("status"),
        "run_id": variant.get("run_id"),
        "label": variant.get("label"),
        "expected_checkpoint_step": expected_step,
        "config_values": config_values,
        "config_reference": config_reference,
        "source_day14_profiling_report": profiling_source,
        "run_dir": portable_artifact_path(run_dir) if run_dir is not None else None,
        "runtime": runtime,
        "requested_device": variant.get("requested_device"),
        "resolved_device": variant.get("resolved_device"),
        "replay_backend": config_values.get("replay_backend"),
        "selection_rule": f"final checkpoint at {expected_step} environment steps",
        "selection_rationale": gpu_profiling_summary.get("selection_rule", {}),
        "gpu_profiling_summary": gpu_profiling_summary,
        "day14_gate": day14_gate,
    }


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    training: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Ensure the loaded checkpoint is the manifest's frozen final variant."""

    expected_run_id = provenance.get("run_id")
    if expected_run_id and training.get("source_day14_run_id") != expected_run_id:
        raise ValueError(
            "checkpoint run id does not match the Day 14 final manifest: "
            f"expected {expected_run_id}, got {training.get('source_day14_run_id')}"
        )
    expected_step = provenance.get("expected_checkpoint_step")
    if expected_step is not None and checkpoint.get("step") != expected_step:
        raise ValueError(
            "checkpoint step does not match the Day 14 final selection rule: "
            f"expected {expected_step}, got {checkpoint.get('step')}"
        )

    expected_config = provenance.get("config_values", {})
    if not isinstance(expected_config, Mapping):
        return
    training_fields = {
        "total_steps": "training_budget",
        "seed": "training_seed",
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
        "train_frequency": "train_frequency",
        "replay_backend": "replay_backend",
    }
    for config_field, training_field in training_fields.items():
        expected = expected_config.get(config_field)
        actual = training.get(training_field)
        if expected is not None and actual != expected:
            raise ValueError(
                "checkpoint training config does not match the Day 14 final "
                f"manifest for {config_field}: expected {expected}, got {actual}"
            )


__all__ = ["load_day14_provenance", "validate_checkpoint_provenance"]
