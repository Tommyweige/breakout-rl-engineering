"""Run and profile the Day 14 batch-size experiment from real CUDA runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import psutil
except ImportError:  # pragma: no cover - optional profiling dependency
    psutil = None  # type: ignore[assignment]

from breakout_env import make_breakout_env
from breakout_rl.experiments import (
    build_manifest,
    compare_manifest,
    load_experiment_configs,
    load_run_report,
    read_metrics,
    relative_path,
    slugify,
    update_manifest,
    write_json_object,
)
from breakout_rl.training.dqn_trainer import DQNTrainer


SAMPLE_FIELDS = (
    "timestamp_utc",
    "elapsed_seconds",
    "gpu_index",
    "gpu_utilization_percent",
    "gpu_power_watts",
    "gpu_memory_used_bytes",
    "gpu_memory_total_bytes",
    "process_cpu_percent",
    "process_rss_bytes",
    "sample_status",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _parse_nvidia_value(value: str, *, unit: str | None = None) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    if unit == "MiB":
        parsed *= 1024 * 1024
    return parsed if math.isfinite(parsed) else None


def _query_gpu(gpu_index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
        return {"sample_status": f"nvidia-smi unavailable: {error}"}
    row = next(csv.reader([completed.stdout.strip()]), [])
    if len(row) < 4:
        return {"sample_status": "nvidia-smi returned an incomplete row"}
    return {
        "gpu_utilization_percent": _parse_nvidia_value(row[0]),
        "gpu_power_watts": _parse_nvidia_value(row[1]),
        "gpu_memory_used_bytes": _parse_nvidia_value(row[2], unit="MiB"),
        "gpu_memory_total_bytes": _parse_nvidia_value(row[3], unit="MiB"),
        "sample_status": "ok",
    }


class RuntimeSampler:
    """Sample GPU and process metrics at a fixed cadence during one run."""

    def __init__(self, output_path: Path, *, interval_seconds: float, gpu_index: int) -> None:
        if interval_seconds <= 0:
            raise ValueError("sampling interval must be greater than zero")
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._last_cpu_seconds = 0.0
        self._last_cpu_wall = 0.0
        self._logical_cpu_count = os.cpu_count() or 1
        self._process = psutil.Process(os.getpid()) if psutil is not None else None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._started_at = time.perf_counter()
        self._last_cpu_wall = self._started_at
        self._last_cpu_seconds = time.process_time()
        if self._process is not None:
            self._process.cpu_percent(None)
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="day14-runtime-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 2.0))

    def _sample_loop(self) -> None:
        with self.output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            next_sample_at = self._started_at
            while not self._stop.is_set():
                now = time.perf_counter()
                elapsed = now - self._started_at
                cpu_seconds = time.process_time()
                wall_delta = max(now - self._last_cpu_wall, 1e-9)
                process_cpu_percent = max(
                    0.0,
                    (cpu_seconds - self._last_cpu_seconds)
                    / wall_delta
                    / self._logical_cpu_count
                    * 100.0,
                )
                self._last_cpu_wall = now
                self._last_cpu_seconds = cpu_seconds
                row: dict[str, Any] = {
                    "timestamp_utc": _utc_now(),
                    "elapsed_seconds": elapsed,
                    "gpu_index": self.gpu_index,
                    "gpu_utilization_percent": None,
                    "gpu_power_watts": None,
                    "gpu_memory_used_bytes": None,
                    "gpu_memory_total_bytes": None,
                    "process_cpu_percent": process_cpu_percent,
                    "process_rss_bytes": None,
                    "sample_status": "unavailable",
                }
                row.update(_query_gpu(self.gpu_index))
                if self._process is not None:
                    try:
                        row["process_cpu_percent"] = self._process.cpu_percent(None)
                        row["process_rss_bytes"] = self._process.memory_info().rss
                    except (psutil.Error, OSError):
                        row["sample_status"] = "process sample unavailable"
                writer.writerow(row)
                stream.flush()
                next_sample_at += self.interval_seconds
                self._stop.wait(max(0.0, next_sample_at - time.perf_counter()))


def _sample_summary(samples_path: Path, *, interval_seconds: float) -> dict[str, Any]:
    with samples_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    def values(field: str) -> list[float]:
        return [
            parsed
            for row in rows
            if (parsed := _finite(row.get(field))) is not None
        ]

    gpu_util = values("gpu_utilization_percent")
    gpu_power = values("gpu_power_watts")
    gpu_memory = values("gpu_memory_used_bytes")
    gpu_total = values("gpu_memory_total_bytes")
    process_cpu = values("process_cpu_percent")
    return {
        "sampling_method": "in-process fixed-interval sampler; nvidia-smi query per sample",
        "sampling_interval_seconds": interval_seconds,
        "process_cpu_percent_method": (
            "time.process_time normalized by wall time and logical CPU count"
        ),
        "cpu_logical_count": os.cpu_count() or 1,
        "sample_count": len(rows),
        "gpu_sample_count": len(gpu_util),
        "gpu_utilization_percent": {
            "mean": statistics.mean(gpu_util) if gpu_util else None,
            "p50": _percentile(gpu_util, 0.50),
            "p95": _percentile(gpu_util, 0.95),
            "max": max(gpu_util) if gpu_util else None,
        },
        "gpu_power_watts": {
            "mean": statistics.mean(gpu_power) if gpu_power else None,
            "p50": _percentile(gpu_power, 0.50),
            "p95": _percentile(gpu_power, 0.95),
            "max": max(gpu_power) if gpu_power else None,
        },
        "gpu_memory_used_bytes": {
            "mean": statistics.mean(gpu_memory) if gpu_memory else None,
            "peak": max(gpu_memory) if gpu_memory else None,
        },
        "gpu_memory_total_bytes": max(gpu_total) if gpu_total else None,
        "process_cpu_percent": {
            "mean": statistics.mean(process_cpu) if process_cpu else None,
            "p50": _percentile(process_cpu, 0.50),
            "p95": _percentile(process_cpu, 0.95),
            "max": max(process_cpu) if process_cpu else None,
        },
        "available": bool(gpu_util),
    }


def _finite_metric_counts(run_dir: Path) -> dict[str, int]:
    rows = read_metrics(run_dir)
    fields = ("loss", "q_mean", "target_mean", "gradient_norm", "epsilon")
    return {
        field: sum(1 for row in rows if _finite(row.get(field)) is not None)
        for field in fields
    }


def _parse_reuse_runs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            raise ValueError("--reuse-run must use LABEL=RUN_DIR")
        if label.strip() in result:
            raise ValueError(f"duplicate --reuse-run label: {label.strip()}")
        path = Path(raw_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        result[label.strip()] = path
    return result


def _set_entry(
    manifest: dict[str, Any],
    manifest_path: Path,
    index: int,
    *,
    run_dir: Path,
    status: str,
    summary: Mapping[str, Any] | None = None,
    profiling: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> None:
    entry = manifest["variants"][index]
    entry["run_dir"] = relative_path(run_dir, start=manifest_path.parent)
    entry["run_id"] = run_dir.name
    entry["status"] = status
    if summary is not None:
        runtime = summary.get("runtime", {})
        runtime = runtime if isinstance(runtime, Mapping) else {}
        entry["resolved_device"] = runtime.get("resolved_device")
        entry["summary"] = {
            "status": summary.get("status"),
            "total_steps": summary.get("total_steps"),
            "episodes": summary.get("episodes"),
            "optimizer_updates": summary.get("optimizer_updates"),
            "steps_per_second": summary.get("steps_per_second"),
            "wall_clock_seconds": runtime.get("wall_clock_seconds"),
        }
    if profiling is not None:
        entry["profiling"] = dict(profiling)
    if error is not None:
        entry["error"] = error


def _failure_status(requested_device: str, error: BaseException) -> str:
    message = str(error).lower()
    if requested_device.startswith("cuda") and any(
        marker in message
        for marker in (
            "cuda was requested",
            "cuda is not available",
            "cuda device index",
            "refusing to fall back",
        )
    ):
        return "blocked"
    return "failed"


def _write_failure(run_dir: Path, *, status: str, error: BaseException, requested_device: str) -> None:
    write_json_object(
        run_dir / "failure.json",
        {
            "status": status,
            "error": str(error),
            "error_type": type(error).__name__,
            "requested_device": requested_device,
        },
    )


def _batch_report(
    manifest_path: Path,
    *,
    comparison: Mapping[str, Any],
    profiling_by_label: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run in comparison.get("runs", []):
        label = str(run.get("label"))
        config = run.get("config", {})
        config = config if isinstance(config, Mapping) else {}
        summary = run.get("summary", {})
        summary = summary if isinstance(summary, Mapping) else {}
        run_path = Path(str(run["run_dir"])) if run.get("run_dir") else None
        if run_path is not None and not run_path.is_absolute():
            run_path = (manifest_path.parent / run_path).resolve()
        wall = _finite(run.get("wall_clock_seconds"))
        updates = int(summary.get("optimizer_updates", 0) or 0)
        batch_size = int(config.get("batch_size", 0) or 0)
        update_sps = updates / wall if wall and wall > 0 else None
        samples_sps = update_sps * batch_size if update_sps is not None else None
        finite_counts = (
            _finite_metric_counts(run_path)
            if run_path is not None
            else {}
        )
        metric_rows = (
            read_metrics(run_path)
            if run_path is not None
            else []
        )
        epsilon_values = [
            parsed
            for row in metric_rows
            if (parsed := _finite(row.get("epsilon"))) is not None
        ]
        replay_occupancy = summary.get("replay_occupancy")
        replay_guardrail = isinstance(replay_occupancy, Mapping) and (
            int(replay_occupancy.get("size", 0) or 0) > 0
            and int(replay_occupancy.get("capacity", 0) or 0) > 0
            and _finite(replay_occupancy.get("ratio")) is not None
        )
        epsilon_guardrail = bool(
            epsilon_values
            and 0.0 <= min(epsilon_values) <= max(epsilon_values) <= 1.0
        )
        episode_guardrail = bool(
            int(run.get("episodes", 0) or 0) > 0
            and run.get("mean_recent_episode_return") is not None
        )
        profile = dict(profiling_by_label.get(label, {}))
        runs.append(
            {
                "label": label,
                "run_id": run.get("run_id"),
                # Keep the durable report portable. The path is relative to
                # the manifest directory, matching each manifest variant.
                "run_dir": (
                    relative_path(run_path, start=manifest_path.parent)
                    if run_path is not None
                    else None
                ),
                "stage": run.get("stage"),
                "status": run.get("status"),
                "batch_size": batch_size,
                "cpu_threads": config.get("cpu_threads"),
                "learning_rate": config.get("learning_rate"),
                "expected_steps": run.get("expected_steps"),
                "completed_steps": run.get("completed_steps"),
                "episodes": run.get("episodes"),
                "mean_recent_episode_return": run.get("mean_recent_episode_return"),
                "median_recent_episode_return": run.get("median_recent_episode_return"),
                "recent_return_trend": run.get("recent_return_trend"),
                "best_rolling_return": run.get("best_rolling_return"),
                "td_error_summary": run.get("td_error_summary"),
                "end_to_end_sps": run.get("sps", {}).get("runtime"),
                "optimizer_updates": updates,
                "optimizer_updates_per_second": update_sps,
                "training_samples_per_second": samples_sps,
                "wall_clock_seconds": wall,
                "replay_backend": run.get("replay_backend", "cpu"),
                "replay_transfer": run.get("replay_transfer", "direct"),
                "replay_memory": run.get("replay_memory"),
                "stage_timings": summary.get("runtime", {}).get("stage_timings")
                if isinstance(summary.get("runtime", {}), Mapping)
                else None,
                "gpu_memory": run.get("gpu_memory"),
                "finite_metric_counts": finite_counts,
                "regression_guardrails": {
                    "replay_occupancy": replay_occupancy,
                    "replay_guardrail": replay_guardrail,
                    "target_sync_count": summary.get("target_sync_count"),
                    "action_distribution": summary.get("action_distribution"),
                    "epsilon": {
                        "finite_count": len(epsilon_values),
                        "first": epsilon_values[0] if epsilon_values else None,
                        "last": epsilon_values[-1] if epsilon_values else None,
                        "minimum": min(epsilon_values) if epsilon_values else None,
                        "maximum": max(epsilon_values) if epsilon_values else None,
                    },
                    "episode_behavior": {
                        "episodes": run.get("episodes"),
                        "mean_recent_episode_return": run.get(
                            "mean_recent_episode_return"
                        ),
                        "recent_return_trend": run.get("recent_return_trend"),
                    },
                    "epsilon_guardrail": epsilon_guardrail,
                    "episode_guardrail": episode_guardrail,
                    "guardrails_passed": replay_guardrail
                    and epsilon_guardrail
                    and episode_guardrail,
                },
                "milestone_snapshots": run.get("milestone_snapshots"),
                "profiling": profile,
            }
        )

    baseline = next((run for run in runs if run["batch_size"] == 32), None)
    baseline_sps = _finite(baseline.get("end_to_end_sps")) if baseline else None
    for run in runs:
        sps = _finite(run.get("end_to_end_sps"))
        speedup = sps / baseline_sps if sps and baseline_sps else None
        run["efficiency_gain_vs_batch32"] = (
            speedup - 1.0 if speedup is not None else None
        )
        run["candidate_for_100k_validation"] = bool(
            run["batch_size"] != 32
            and run["status"] == "completed"
            and run["completed_steps"] == run.get("expected_steps", run["completed_steps"])
            and speedup is not None
            and speedup > 1.0
            and all(count > 0 for count in run["finite_metric_counts"].values())
            and run["regression_guardrails"]["guardrails_passed"]
        )

    return {
        "schema_version": 1,
        "experiment_id": comparison.get("experiment_id", manifest_path.parent.name),
        "manifest": relative_path(manifest_path, start=Path.cwd()),
        "manifest_status": comparison.get("manifest_status"),
        "stage": comparison.get("experiment_stage", "unknown"),
        "comparison_conditions": comparison.get("comparison_conditions", {}),
        "sampling": {
            "source": "profiling samples are stored beside each run",
            "no_manual_values": True,
        },
        "runs": runs,
        "selection_rule": {
            "short_stage": "candidate requires completed 10K, finite metrics, and end-to-end SPS strictly above batch 32",
            "long_stage": "validate every short-stage candidate with 100K learning metrics before freezing",
            "quality_over_speed": "do not select a batch size from GPU utilization alone when return or numerical guardrails regress",
        },
    }


def run_experiment(args: argparse.Namespace) -> tuple[int, Path, dict[str, Any]]:
    configs = load_experiment_configs(args.configs)
    reuse = _parse_reuse_runs(args.reuse_run)
    labels = {config.label for config in configs}
    unknown_reuse = set(reuse) - labels
    if unknown_reuse:
        raise ValueError("--reuse-run label(s) not present in configs: " + ", ".join(sorted(unknown_reuse)))
    run_keys = [(slugify(config.label), config.config.seed) for config in configs]
    if len(set(run_keys)) != len(run_keys):
        raise ValueError("config labels/seed pairs must be unique")
    stages = {config.stage for config in configs}
    if len(stages) != 1:
        raise ValueError("one batch-size experiment cannot mix stages")
    if args.require_cuda:
        if any(
            not (
                config.config.requested_device == "cuda"
                or config.config.requested_device.startswith("cuda:")
            )
            for config in configs
        ):
            raise ValueError("--require-cuda requires every config to request CUDA")
        requested = {config.config.requested_device for config in configs}
        if len(requested) != 1:
            raise ValueError("--require-cuda requires one requested CUDA device")

    experiment_id = slugify(args.experiment_id)
    experiments_root = args.experiments_root.resolve()
    runs_root = args.runs_root.resolve()
    samples_root = args.samples_root.resolve()
    experiment_dir = experiments_root / experiment_id
    run_parent = runs_root / experiment_id
    if experiment_dir.exists() or run_parent.exists():
        raise FileExistsError(f"experiment output already exists for {experiment_id}")
    experiment_dir.mkdir(parents=True, exist_ok=False)
    run_parent.mkdir(parents=True, exist_ok=False)
    manifest_path = experiment_dir / "manifest.json"
    manifest = build_manifest(
        experiment_id=experiment_id,
        configs=configs,
        manifest_path=manifest_path,
        command=[str(value) for value in sys.argv],
    )
    update_manifest(manifest_path, manifest)

    profiling_by_label: dict[str, dict[str, Any]] = {}
    for index, config in enumerate(configs):
        run_dir = reuse.get(config.label)
        reused = run_dir is not None
        if run_dir is None:
            run_dir = run_parent / f"{slugify(config.label)}-seed{config.config.seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
        sample_dir = samples_root / experiment_id / slugify(config.label)
        sample_path = sample_dir / "runtime-samples.csv"
        sample_summary_path = sample_dir / "runtime-samples-summary.json"
        profiler = None if reused else RuntimeSampler(
            sample_path,
            interval_seconds=args.sample_interval,
            gpu_index=args.gpu_index,
        )
        summary: Mapping[str, Any] | None = None
        error: BaseException | None = None
        status = "completed"
        try:
            if reused:
                reused_report = load_run_report(run_dir)
                status = str(reused_report.get("status", "incomplete"))
                summary = reused_report.get("summary", {})
            else:
                assert profiler is not None
                profiler.start()
                env = make_breakout_env()
                try:
                    trainer = DQNTrainer(env, config.config, run_dir=run_dir)
                    summary = trainer.train()
                finally:
                    env.close()
        except Exception as caught:
            error = caught
            status = _failure_status(config.config.requested_device, caught)
            _write_failure(
                run_dir,
                status=status,
                error=caught,
                requested_device=config.config.requested_device,
            )
        finally:
            if profiler is not None:
                profiler.stop()
        if sample_path.is_file():
            sample_summary = _sample_summary(
                sample_path,
                interval_seconds=args.sample_interval,
            )
            write_json_object(sample_summary_path, sample_summary)
            profiling = {
                "sample_csv": relative_path(sample_path, start=manifest_path.parent),
                "sample_summary": relative_path(sample_summary_path, start=manifest_path.parent),
                **sample_summary,
            }
        else:
            profiling = {
                "reused_run": reused,
                "sample_csv": None,
                "sample_summary": None,
                "available": False,
            }
        profiling_by_label[config.label] = profiling
        _set_entry(
            manifest,
            manifest_path,
            index,
            run_dir=run_dir,
            status=status,
            summary=summary,
            profiling=profiling,
            error=str(error) if error is not None else None,
        )
        update_manifest(manifest_path, manifest)

    statuses = [str(entry.get("status")) for entry in manifest["variants"]]
    if all(status == "completed" for status in statuses):
        manifest["status"] = "completed"
    elif any(status == "blocked" for status in statuses):
        manifest["status"] = "blocked"
    else:
        manifest["status"] = "failed"
    update_manifest(manifest_path, manifest)

    comparison = compare_manifest(manifest_path)
    report = _batch_report(
        manifest_path,
        comparison=comparison,
        profiling_by_label=profiling_by_label,
    )
    report_path = experiment_dir / "batch-size-comparison.json"
    write_json_object(report_path, report)
    exit_code = 0 if manifest["status"] == "completed" else 1
    return exit_code, manifest_path, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-interval CUDA/CPU profiling for Day 14 batch-size configs."
    )
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--samples-root", type=Path, default=Path("assets/day14/batch-size-profiling"))
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--reuse-run",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="reuse an existing run for a validation baseline instead of retraining",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exit_code, manifest_path, report = run_experiment(args)
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as error:
        print(f"Unable to run batch-size experiment: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report.get("manifest_status"),
                "manifest": str(manifest_path),
                "runs": [
                    {
                        "label": run["label"],
                        "batch_size": run["batch_size"],
                        "status": run["status"],
                        "sps": run["end_to_end_sps"],
                        "candidate_for_100k_validation": run["candidate_for_100k_validation"],
                    }
                    for run in report["runs"]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
