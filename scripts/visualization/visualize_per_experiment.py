"""Render source-backed PER quality and replay-diagnostic figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _read_comparison(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _quality_by_transitions(comparison: dict[str, Any], output: Path) -> None:
    milestones = comparison["milestones"]
    transitions = np.asarray(
        [float(item["transitions"]) for item in milestones],
        dtype=np.float64,
    )
    uniform = np.asarray(
        [float(item["uniform_across_training_seeds"]["mean"]) for item in milestones]
    )
    per = np.asarray(
        [float(item["per_across_training_seeds"]["mean"]) for item in milestones]
    )
    uniform_sd = np.asarray(
        [float(item["uniform_across_training_seeds"]["sample_std"]) for item in milestones]
    )
    per_sd = np.asarray(
        [float(item["per_across_training_seeds"]["sample_std"]) for item in milestones]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.errorbar(
        transitions,
        uniform,
        yerr=uniform_sd,
        marker="o",
        linewidth=2.2,
        capsize=4,
        color="#496A9B",
        label="Uniform Replay",
    )
    ax.errorbar(
        transitions,
        per,
        yerr=per_sd,
        marker="o",
        linewidth=2.2,
        capsize=4,
        color="#D17645",
        label="Prioritized Replay",
    )
    ax.set_xscale("log")
    ax.set_xticks(transitions, labels=["100K", "250K", "500K"])
    ax.set_xlabel("Accepted environment transitions")
    ax.set_ylabel("Mean raw return (mean across 3 training seeds)")
    ax.set_title("Breakout evaluation quality vs training transitions")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _quality_by_wall_clock(comparison: dict[str, Any], output: Path) -> None:
    milestones = comparison["milestones"]
    uniform_x: list[float] = []
    per_x: list[float] = []
    uniform_y: list[float] = []
    per_y: list[float] = []
    uniform_sd: list[float] = []
    per_sd: list[float] = []
    for milestone in milestones:
        rows = milestone["per_seed"]
        uniform_x.append(float(np.mean([row["uniform_elapsed_seconds"] for row in rows])))
        per_x.append(float(np.mean([row["per_elapsed_seconds"] for row in rows])))
        uniform_y.append(float(milestone["uniform_across_training_seeds"]["mean"]))
        per_y.append(float(milestone["per_across_training_seeds"]["mean"]))
        uniform_sd.append(float(milestone["uniform_across_training_seeds"]["sample_std"]))
        per_sd.append(float(milestone["per_across_training_seeds"]["sample_std"]))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.errorbar(
        uniform_x,
        uniform_y,
        yerr=uniform_sd,
        marker="o",
        linewidth=2.2,
        capsize=4,
        color="#496A9B",
        label="Uniform Replay",
    )
    ax.errorbar(
        per_x,
        per_y,
        yerr=per_sd,
        marker="o",
        linewidth=2.2,
        capsize=4,
        color="#D17645",
        label="Prioritized Replay",
    )
    ax.set_xlabel("Cumulative training wall-clock seconds")
    ax.set_ylabel("Mean raw return (mean across 3 training seeds)")
    ax.set_title("Breakout evaluation quality vs training time")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _priority_diagnostics(run_metrics: Path, output: Path) -> None:
    rows: list[dict[str, str]] = []
    with run_metrics.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("optimizer_updated") != "True" or not row.get("priority_mean"):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"{run_metrics}: no PER diagnostics were recorded")

    filtered: list[dict[str, str]] = []
    last_key: tuple[str, str, str, str] | None = None
    for row in rows:
        key = (
            row.get("priority_mean", ""),
            row.get("priority_p99", ""),
            row.get("importance_weight_mean", ""),
            row.get("sampling_effective_sample_size", ""),
        )
        if key != last_key:
            filtered.append(row)
            last_key = key
    steps = np.asarray([float(row["global_step"]) for row in filtered])
    priority_mean = np.asarray([float(row["priority_mean"]) for row in filtered])
    priority_p99 = np.asarray([float(row["priority_p99"]) for row in filtered])
    weight_mean = np.asarray([float(row["importance_weight_mean"]) for row in filtered])
    effective_size = np.asarray(
        [float(row["sampling_effective_sample_size"]) for row in filtered]
    )
    replay_capacity = float(rows[-1]["replay_capacity"])

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot(steps, priority_mean, color="#496A9B", linewidth=1.8, label="Priority mean")
    ax.plot(steps, priority_p99, color="#D17645", linewidth=1.8, label="Priority p99")
    ax.set_xlabel("Accepted environment transitions")
    ax.set_ylabel("Stored absolute TD-error priority")
    ax.set_title("PER priority and correction diagnostics (training seed 11)")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        steps,
        weight_mean,
        color="#43856A",
        linewidth=1.6,
        linestyle="--",
        label="Mean IS weight",
    )
    ax2.plot(
        steps,
        effective_size / replay_capacity,
        color="#8B6BA8",
        linewidth=1.6,
        linestyle=":",
        label="Effective sample size / active replay",
    )
    ax2.set_ylabel("Normalized importance weight / effective replay fraction")
    lines, labels = ax.get_legend_handles_labels()
    second_lines, second_labels = ax2.get_legend_handles_labels()
    ax.legend(lines + second_lines, labels + second_labels, frameon=False, loc="best")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def generate_visualizations(
    *,
    comparison_path: Path,
    output_root: Path,
) -> list[Path]:
    comparison = _read_comparison(comparison_path)
    output_dir = output_root / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "evaluation-by-transitions.png",
        output_dir / "evaluation-by-wall-clock.png",
        output_dir / "priority-diagnostics-seed11.png",
    ]
    _quality_by_transitions(comparison, outputs[0])
    _quality_by_wall_clock(comparison, outputs[1])
    _priority_diagnostics(
        output_root / "runs" / "per-seed11" / "metrics.csv",
        outputs[2],
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create real-data figures for the PER experiment."
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("experiments/per/comparison.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/per"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = generate_visualizations(
            comparison_path=args.comparison,
            output_root=args.output_root,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"PER figure generation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps([path.as_posix() for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
