"""Compare end-to-end training throughput before and after hot-path changes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.experiments import load_run_report, read_metrics, write_json_object


SYSTEM_ONLY_FIELDS = {
    "checkpoint_interval",
    "diagnostics_interval",
    "metrics_flush_interval",
    "cpu_threads",
}


def _numeric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _runtime_value(report: Mapping[str, Any], name: str, fallback: Any = None) -> Any:
    runtime = report.get("runtime", {})
    if isinstance(runtime, Mapping) and runtime.get(name) is not None:
        return runtime[name]
    return fallback


def _finite_metric_count(run_dir: Path) -> dict[str, int]:
    rows = read_metrics(run_dir)
    fields = ("loss", "q_mean", "target_mean", "gradient_norm", "epsilon")
    result: dict[str, int] = {}
    for field in fields:
        result[field] = sum(1 for row in rows if _numeric(row.get(field)) is not None)
    return result


def _epsilon_summary(run_dir: Path) -> dict[str, float | None]:
    values = [
        parsed
        for row in read_metrics(run_dir)
        if (parsed := _numeric(row.get("epsilon"))) is not None
    ]
    return {
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "count": len(values),
    }


def _phase(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    wall_clock = _numeric(report.get("wall_clock_seconds"))
    if wall_clock is None or wall_clock <= 0:
        wall_clock = None
    optimizer_updates = int(summary.get("optimizer_updates", 0) or 0)
    update_sps = optimizer_updates / wall_clock if wall_clock else None
    batch_size = int(report.get("config", {}).get("batch_size", 0) or 0)
    configured_threads = _runtime_value(
        report,
        "configured_cpu_threads",
        report.get("config", {}).get("cpu_threads")
        if isinstance(report.get("config"), Mapping)
        else None,
    )
    flush_interval = _runtime_value(
        report,
        "metrics_flush_interval",
        report.get("config", {}).get("metrics_flush_interval")
        if isinstance(report.get("config"), Mapping)
        else 1,
    )
    diagnostics_interval = _runtime_value(
        report,
        "diagnostics_interval",
        report.get("config", {}).get("diagnostics_interval")
        if isinstance(report.get("config"), Mapping)
        else 1,
    )
    peak_reserved = _numeric(report.get("gpu_memory", {}).get("peak_reserved_bytes"))
    total_memory = _numeric(_runtime_value(report, "gpu_memory_total_bytes"))
    return {
        "run_id": report.get("run_id"),
        "status": report.get("status"),
        "completed_steps": report.get("completed_steps"),
        "end_to_end_sps": _numeric(_runtime_value(report, "steps_per_second"))
        or report.get("sps", {}).get("runtime"),
        "optimizer_updates": optimizer_updates,
        "optimizer_updates_per_second": update_sps,
        "training_samples_per_second": update_sps * batch_size
        if update_sps is not None
        else None,
        "batch_size": batch_size,
        "wall_clock_seconds": wall_clock,
        "cpu_logical_count": _runtime_value(report, "cpu_logical_count"),
        "cpu_thread_count": _runtime_value(report, "cpu_thread_count"),
        "configured_cpu_threads": configured_threads,
        "gpu_utilization_percent": _runtime_value(
            report, "gpu_utilization_percent"
        ),
        "gpu_utilization_source": _runtime_value(
            report, "gpu_utilization_source", "unavailable"
        ),
        "peak_allocated_vram_bytes": report.get("gpu_memory", {}).get(
            "peak_allocated_bytes"
        ),
        "peak_reserved_vram_bytes": peak_reserved,
        "gpu_memory_total_bytes": total_memory,
        "vram_headroom_bytes": (
            total_memory - peak_reserved
            if total_memory is not None and peak_reserved is not None
            else None
        ),
        "diagnostics_interval": diagnostics_interval,
        "metrics_flush_interval": flush_interval,
        "metrics_row_cadence": _runtime_value(report, "metrics_row_cadence", 1),
        "finite_metric_counts": _finite_metric_count(Path(report["run_dir"])),
        "epsilon_summary": _epsilon_summary(Path(report["run_dir"])),
        "replay_occupancy": summary.get("replay_occupancy"),
        "target_sync_count": summary.get("target_sync_count"),
        "episodes": report.get("episodes"),
        "mean_recent_episode_return": report.get("mean_recent_episode_return"),
        "recent_return_trend": report.get("recent_return_trend"),
        "replay_capacity": report.get("config", {}).get("replay_capacity"),
    }


def benchmark_runs(before_dir: str | Path, after_dir: str | Path) -> dict[str, Any]:
    before_path = Path(before_dir).resolve()
    after_path = Path(after_dir).resolve()
    before_report = load_run_report(before_path)
    after_report = load_run_report(after_path)
    before_config = before_report.get("config", {})
    after_config = after_report.get("config", {})
    learning_diff = {
        key: {"before": before_config.get(key), "after": after_config.get(key)}
        for key in sorted(set(before_config) | set(after_config))
        if key not in SYSTEM_ONLY_FIELDS and before_config.get(key) != after_config.get(key)
    }
    before_phase = _phase(before_report)
    after_phase = _phase(after_report)
    before_sps = _numeric(before_phase["end_to_end_sps"])
    after_sps = _numeric(after_phase["end_to_end_sps"])
    speedup = after_sps / before_sps if before_sps and after_sps else None
    def _same_float(name: str, tolerance: float = 1e-9) -> bool:
        before_value = before_phase["epsilon_summary"].get(name)
        after_value = after_phase["epsilon_summary"].get(name)
        return (
            before_value is not None
            and after_value is not None
            and math.isclose(before_value, after_value, rel_tol=tolerance, abs_tol=tolerance)
        )

    same_replay = (
        before_phase["replay_capacity"] == after_phase["replay_capacity"]
        and before_phase["replay_occupancy"] == after_phase["replay_occupancy"]
    )
    epsilon_schedule_consistent = (
        before_phase["epsilon_summary"]["count"] > 0
        and after_phase["epsilon_summary"]["count"] > 0
        and all(_same_float(name) for name in ("first", "last", "minimum", "maximum"))
    )
    episode_behavior_present = all(
        phase["episodes"] is not None
        and int(phase["episodes"]) > 0
        and phase["mean_recent_episode_return"] is not None
        for phase in (before_phase, after_phase)
    )
    return {
        "schema_version": 1,
        "before": {"run_dir": str(before_path), **before_phase},
        "after": {"run_dir": str(after_path), **after_phase},
        "optimization": {
            "same_learning_config": not learning_diff,
            "learning_config_diff": learning_diff,
            "end_to_end_sps_speedup": speedup,
            "target_speedup": 1.5,
            "target_met": speedup is not None and speedup >= 1.5,
            "target_is_engineering_goal_not_correctness_gate": True,
        },
        "10k_regression": {
            "both_completed": before_report.get("status") == "completed"
            and after_report.get("status") == "completed",
            "same_step_budget": before_report.get("expected_steps")
            == after_report.get("expected_steps"),
            "same_optimizer_updates": before_phase["optimizer_updates"]
            == after_phase["optimizer_updates"],
            "same_target_sync_count": before_report.get("summary", {}).get(
                "target_sync_count"
            )
            == after_report.get("summary", {}).get("target_sync_count"),
            "finite_metrics_present": all(
                all(count > 0 for count in phase["finite_metric_counts"].values())
                for phase in (before_phase, after_phase)
            ),
            "replay_guardrail": same_replay,
            "epsilon_schedule_consistent": epsilon_schedule_consistent,
            "episode_behavior_present": episode_behavior_present,
            "semantic_guardrails_passed": same_replay
            and epsilon_schedule_consistent
            and episode_behavior_present,
            "bit_exact_curve_required": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare real 10K throughput artifacts before and after optimization."
    )
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = benchmark_runs(args.before, args.after)
    if args.output is not None:
        write_json_object(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
