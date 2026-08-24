"""Summarize a DQN run and render evidence-bearing diagnostic curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
from pathlib import Path
import queue
import sys
from typing import Any, Iterable, Mapping, Sequence

from breakout_rl.training.diagnostics import aggregate_training_metrics


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _paired_series(
    rows: Sequence[Mapping[str, Any]],
    x_field: str,
    y_field: str,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        x_value = _number(row.get(x_field))
        y_value = _number(row.get(y_field))
        if x_value is None or y_value is None:
            continue
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        x_values.append(x_value)
        y_values.append(y_value)
    return x_values, y_values


def _plot_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[tuple[str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    plotted = False
    for field, label in fields:
        x_values, y_values = _paired_series(rows, "global_step", field)
        if not y_values:
            continue
        axis.plot(x_values, y_values, linewidth=1.2, label=label)
        plotted = True

    axis.set_title(title)
    axis.set_xlabel("Environment step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if plotted and len(fields) > 1:
        axis.legend()
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No finite samples in metrics.csv",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    figure.tight_layout()
    figure.savefig(output_path, format="png")
    plt.close(figure)


def _write_plots_local(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Render the required Day 13 plots from metrics rows."""

    materialized = [dict(row) for row in rows]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plot_specs = {
        "return_curve": (
            (("raw_episode_return", "Raw episode return"),),
            "Episode return over training",
            "Raw episode return",
            "return-curve.png",
        ),
        "loss_curve": (
            (("loss", "Huber loss"),),
            "Training loss over optimizer updates",
            "Loss",
            "loss-curve.png",
        ),
        "q_values": (
            (
                ("q_mean", "Selected Q mean"),
                ("q_max", "Selected Q max"),
                ("q_min", "Selected Q min"),
                ("target_mean", "Target mean"),
                ("target_max", "Target max"),
            ),
            "Q-values and Bellman targets",
            "Value",
            "q-values.png",
        ),
        "gradient_norm": (
            (("gradient_norm", "Gradient norm"),),
            "Gradient norm before clipping",
            "L2 norm",
            "gradient-norm.png",
        ),
        "epsilon_curve": (
            (("epsilon", "Epsilon"),),
            "Exploration probability over training",
            "Epsilon",
            "epsilon-curve.png",
        ),
    }
    paths: dict[str, str] = {}
    for key, (fields, title, ylabel, filename) in plot_specs.items():
        output_path = destination / filename
        _plot_metric(
            materialized,
            fields=fields,
            title=title,
            ylabel=ylabel,
            output_path=output_path,
        )
        paths[key] = str(output_path)
    return paths


def _plot_worker(
    rows: list[dict[str, Any]],
    output_dir: str,
    result_queue: Any,
) -> None:
    try:
        result_queue.put({"paths": _write_plots_local(rows, output_dir)})
    except Exception as error:  # pragma: no cover - exercised in child process
        result_queue.put({"error": f"{type(error).__name__}: {error}"})


def write_plots(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Render plots, isolating Matplotlib from an already-loaded PyTorch DLL."""

    materialized = [dict(row) for row in rows]
    destination = str(Path(output_dir))
    if "torch" not in sys.modules:
        return _write_plots_local(materialized, destination)

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_plot_worker,
        args=(materialized, destination, result_queue),
    )
    process.start()
    process.join()
    try:
        result = result_queue.get(timeout=5)
    except queue.Empty:
        result = None
    result_queue.close()
    if process.exitcode != 0 or not isinstance(result, dict) or "error" in result:
        error = result.get("error") if isinstance(result, dict) else "unknown plotting error"
        raise RuntimeError(f"plot worker failed: {error}")
    return result["paths"]


def analyze_run(
    run_dir: str | Path,
    *,
    plots_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read a run's three durable artifacts, summarize them, and plot them."""

    path = Path(run_dir)
    config = _read_json(path / "config.json")
    rows = _read_metrics(path / "metrics.csv")
    summary = _read_json(path / "summary.json")
    report = aggregate_training_metrics(rows)
    report.update(
        {
            "run_id": str(config.get("run_id", path.name)),
            "metadata": config.get("runtime", {}),
            "run_summary": summary,
            "plots": write_plots(rows, plots_dir or path / "plots"),
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize and plot a DQN training run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="where to write PNGs (default: <run_dir>/plots)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_run(args.run_dir, plots_dir=args.plots_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
