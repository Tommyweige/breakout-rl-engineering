"""Render batch-size efficiency evidence from the profiling report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def render_batch_size_efficiency(report_path: str | Path, output_path: str | Path) -> Path:
    source = Path(report_path).resolve()
    output = Path(output_path)
    report = json.loads(source.read_text(encoding="utf-8"))
    runs = sorted(
        report["runs"],
        key=lambda run: _number(run.get("batch_size")),
    )
    labels = [str(run["batch_size"]) for run in runs]
    x = np.arange(len(runs))
    sps = [_number(run.get("end_to_end_sps")) for run in runs]
    samples = [_number(run.get("training_samples_per_second")) for run in runs]
    gpu = [
        _number(run.get("profiling", {}).get("gpu_utilization_percent", {}).get("mean"))
        for run in runs
    ]
    gpu_p95 = [
        _number(run.get("profiling", {}).get("gpu_utilization_percent", {}).get("p95"))
        for run in runs
    ]
    power = [
        _number(run.get("profiling", {}).get("gpu_power_watts", {}).get("mean"))
        for run in runs
    ]
    vram = [
        _number(run.get("profiling", {}).get("gpu_memory_used_bytes", {}).get("peak"))
        / (1024 * 1024)
        for run in runs
    ]
    cpu = [
        _number(run.get("profiling", {}).get("process_cpu_percent", {}).get("mean"))
        for run in runs
    ]
    recent = [_number(run.get("mean_recent_episode_return")) for run in runs]
    best = [_number(run.get("best_rolling_return")) for run in runs]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure.suptitle("Day 14 batch-size GPU efficiency profiling", fontsize=16)
    width = 0.36

    axes[0, 0].bar(x, sps, width, label="environment SPS", color="tab:blue")
    axes[0, 0].set_title("Throughput: environment work vs optimizer work")
    axes[0, 0].set_ylabel("environment steps per second")
    sample_axis = axes[0, 0].twinx()
    sample_axis.plot(
        x,
        samples,
        "o--",
        color="tab:orange",
        label="training samples/s",
    )
    sample_axis.set_ylabel("training samples per second")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    handles_2, labels_2 = sample_axis.get_legend_handles_labels()
    axes[0, 0].legend(handles + handles_2, labels_ + labels_2)

    axes[0, 1].bar(x - width / 2, gpu, width, label="GPU utilization mean")
    axes[0, 1].bar(x + width / 2, gpu_p95, width, label="GPU utilization p95")
    axes[0, 1].plot(x, cpu, "ko--", label="process CPU %")
    axes[0, 1].set_title("Fixed-interval utilization summary")
    axes[0, 1].set_ylabel("percent")
    axes[0, 1].legend()

    axes[1, 0].bar(x, vram, width, label="peak VRAM (MiB)", color="tab:orange")
    axes[1, 0].set_title("Power and memory cost")
    axes[1, 0].set_ylabel("peak VRAM (MiB)")
    power_axis = axes[1, 0].twinx()
    power_axis.plot(
        x,
        power,
        "o--",
        color="tab:blue",
        label="GPU power mean (W)",
    )
    power_axis.set_ylabel("GPU power mean (W)")
    handles, labels_ = axes[1, 0].get_legend_handles_labels()
    handles_2, labels_2 = power_axis.get_legend_handles_labels()
    axes[1, 0].legend(handles + handles_2, labels_ + labels_2)

    axes[1, 1].bar(x - width / 2, recent, width, label="recent return mean")
    axes[1, 1].bar(x + width / 2, best, width, label="best rolling20 mean")
    axes[1, 1].set_title("Short-run learning guardrails")
    axes[1, 1].set_ylabel("episode return")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.set_xlabel("batch size")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    metadata = {
        "source_report": str(source),
        "run_ids": [run.get("run_id") for run in runs],
        "x_field": "batch_size",
        "metrics": [
            "end_to_end_sps",
            "training_samples_per_second",
            "gpu_utilization_percent",
            "gpu_power_watts",
            "peak_vram",
            "process_cpu_percent",
            "recent_return_mean",
            "best_rolling20_mean",
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = render_batch_size_efficiency(args.report, args.output)
    print(f"Wrote batch-size efficiency figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
