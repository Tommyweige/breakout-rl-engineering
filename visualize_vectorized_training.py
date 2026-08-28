"""Plot Day 16 figures directly from the vectorized benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt


def _records(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = report.get("results")
    if not isinstance(values, list) or not values:
        raise ValueError("benchmark report must contain a non-empty results list")
    records: list[Mapping[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("every benchmark result must be an object")
        summary = value.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("every benchmark result must contain a summary object")
        records.append(value)
    return sorted(records, key=lambda item: int(item["environment_count"]))


def _summary(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("summary")
    if not isinstance(value, Mapping):
        raise ValueError("benchmark result is missing summary")
    return value


def _runtime(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _summary(record).get("runtime")
    return value if isinstance(value, Mapping) else {}


def _stage(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    timings = _runtime(record).get("stage_timings")
    if not isinstance(timings, Mapping):
        return {}
    value = timings.get(name)
    return value if isinstance(value, Mapping) else {}


def _positive_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _bar_axis(ax: plt.Axes, counts: list[int]) -> None:
    ax.set_xticks(range(len(counts)), [str(count) for count in counts])
    ax.set_xlabel("Number of environments")
    ax.grid(axis="y", alpha=0.25)


def plot_throughput(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    counts = [int(record["environment_count"]) for record in records]
    transition_sps = [
        float(_summary(record)["environment_transitions_per_second"])
        for record in records
    ]
    wall_seconds = [
        float(_runtime(record)["wall_clock_seconds"])
        for record in records
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(len(counts)), transition_sps, color="#3366cc")
    axes[0].set_ylabel("Accepted transitions / second")
    axes[0].set_title("End-to-end throughput")
    _bar_axis(axes[0], counts)
    axes[1].bar(range(len(counts)), wall_seconds, color="#dc3912")
    axes[1].set_ylabel("Wall-clock seconds")
    axes[1].set_title("Same transition budget")
    _bar_axis(axes[1], counts)
    fig.suptitle("Vectorized DQN systems screening")
    fig.tight_layout()
    _save(fig, output, "vectorized-throughput.png")


def plot_batched_inference(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    counts = [int(record["environment_count"]) for record in records]
    throughputs: list[float] = []
    batch_latencies_ms: list[float] = []
    for record in records:
        timing = _stage(record, "batched_action_inference")
        calls = _positive_number(timing.get("calls"))
        wall_seconds = _positive_number(timing.get("wall_seconds"))
        transitions = _positive_number(
            _summary(record).get("action_inference_transitions")
        )
        if calls is None or wall_seconds is None or transitions is None:
            raise ValueError("batched inference timing is missing from benchmark report")
        throughputs.append(transitions / wall_seconds)
        batch_latencies_ms.append(wall_seconds / calls * 1000.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(counts, throughputs, marker="o", color="#109618")
    axes[0].set_ylabel("Inference transitions / second")
    axes[0].set_title("Batched inference throughput")
    axes[0].set_xticks(counts)
    axes[0].grid(alpha=0.25)
    axes[1].plot(counts, batch_latencies_ms, marker="o", color="#990099")
    axes[1].set_ylabel("Milliseconds / batched forward")
    axes[1].set_title("One call handles N observations")
    axes[1].set_xticks(counts)
    axes[1].grid(alpha=0.25)
    for ax in axes:
        ax.set_xlabel("Number of environments")
    fig.tight_layout()
    _save(fig, output, "batched-inference.png")


def plot_replay_insertion(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    counts = [int(record["environment_count"]) for record in records]
    throughputs: list[float] = []
    latencies_ms: list[float] = []
    for record in records:
        timing = _stage(record, "batched_replay_insert")
        calls = _positive_number(timing.get("calls"))
        wall_seconds = _positive_number(timing.get("wall_seconds"))
        transitions = _positive_number(
            _summary(record).get("replay_insertion_transitions")
        )
        if calls is None or wall_seconds is None or transitions is None:
            raise ValueError("batched replay timing is missing from benchmark report")
        throughputs.append(transitions / wall_seconds)
        latencies_ms.append(wall_seconds / calls * 1000.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(counts, throughputs, marker="o", color="#ff9900")
    axes[0].set_ylabel("Inserted transitions / second")
    axes[0].set_title("Batched replay insertion")
    axes[1].plot(counts, latencies_ms, marker="o", color="#0099c6")
    axes[1].set_ylabel("Milliseconds / insertion call")
    axes[1].set_title("Cost per call")
    for ax in axes:
        ax.set_xlabel("Number of environments")
        ax.set_xticks(counts)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output, "replay-insertion.png")


def plot_replay_insertion_microbenchmark(
    report: Mapping[str, Any],
    output: Path,
) -> None:
    values = report.get("results")
    if not isinstance(values, list) or not values:
        raise ValueError("replay insertion report must contain results")
    records = [value for value in values if isinstance(value, Mapping)]
    if len(records) != len(values):
        raise ValueError("replay insertion results must be objects")
    records.sort(key=lambda item: int(item["batch_size"]))
    batch_sizes = [int(record["batch_size"]) for record in records]
    throughputs = [float(record["transitions_per_second"]) for record in records]
    latencies = [float(record["latency_ms_per_call"]) for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(batch_sizes, throughputs, marker="o", color="#ff9900")
    axes[0].set_ylabel("Inserted transitions / second")
    axes[0].set_title("Batched replay insertion microbenchmark")
    axes[1].plot(batch_sizes, latencies, marker="o", color="#0099c6")
    axes[1].set_ylabel("Milliseconds / add_batch call")
    axes[1].set_title("Amortized call cost")
    for ax in axes:
        ax.set_xlabel("Insertion batch size")
        ax.set_xticks(batch_sizes)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output, "replay-insertion.png")


def plot_system_utilization(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    counts = [int(record["environment_count"]) for record in records]
    gpu_values: list[float] = []
    cpu_values: list[float] = []
    gpu_counts: list[int] = []
    cpu_counts: list[int] = []
    for index, record in enumerate(records):
        profile = record.get("runtime_profile")
        if not isinstance(profile, Mapping):
            continue
        gpu = profile.get("gpu_utilization_percent")
        cpu = profile.get("process_cpu_percent")
        if isinstance(gpu, Mapping) and gpu.get("mean") is not None:
            gpu_values.append(float(gpu["mean"]))
            gpu_counts.append(counts[index])
        if isinstance(cpu, Mapping) and cpu.get("mean") is not None:
            cpu_values.append(float(cpu["mean"]))
            cpu_counts.append(counts[index])

    fig, ax = plt.subplots(figsize=(7, 4))
    if gpu_values:
        ax.plot(gpu_counts, gpu_values, marker="o", label="GPU utilization (%)")
    if cpu_values:
        ax.plot(cpu_counts, cpu_values, marker="o", label="Process CPU (%)")
    if not gpu_values and not cpu_values:
        raise ValueError("runtime profile contains no CPU or GPU utilization samples")
    ax.set_xlabel("Number of environments")
    ax.set_ylabel("Mean sampled utilization (%)")
    ax.set_title("Runtime utilization from fixed-interval samples")
    ax.set_xticks(counts)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, output, "system-utilization.png")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Day 16 benchmark evidence")
    parser.add_argument("report", type=Path)
    parser.add_argument("--insertion-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/day16"))
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("benchmark report must contain a JSON object")
    records = _records(report)
    plot_throughput(records, args.output_dir)
    plot_batched_inference(records, args.output_dir)
    if args.insertion_report is None:
        plot_replay_insertion(records, args.output_dir)
    else:
        insertion_report = json.loads(
            args.insertion_report.read_text(encoding="utf-8")
        )
        if not isinstance(insertion_report, Mapping):
            raise ValueError("insertion report must contain a JSON object")
        plot_replay_insertion_microbenchmark(insertion_report, args.output_dir)
    plot_system_utilization(records, args.output_dir)
    print(f"Wrote Day 16 figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
