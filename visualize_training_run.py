"""Plot real Day 12 CSV metrics from one training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize metrics.csv produced by train_dqn.py."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/day12"))
    return parser


def _read_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"metrics file is empty: {metrics_path}")
    return rows


def _series(
    rows: Iterable[dict[str, str]],
    value_field: str,
) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for row in rows:
        raw_value = row.get(value_field, "")
        if raw_value in (None, ""):
            continue
        steps.append(int(row["global_step"]))
        values.append(float(raw_value))
    return steps, values


def _metadata(run_dir: Path) -> tuple[str, int | str]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return run_dir.name, "unknown"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return str(config.get("run_id", run_dir.name)), config.get("seed", "unknown")


def _save_line_plot(
    *,
    steps: list[int],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    label: str,
    run_id: str,
    seed: int | str,
) -> None:
    if not values:
        raise ValueError(f"metrics.csv contains no values for {ylabel}")
    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    ax.plot(steps, values, color="#1769aa", linewidth=1.4, label=label)
    ax.set_title(f"{title}\nrun={run_id}, seed={seed}")
    ax.set_xlabel("Environment step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_overview(
    *,
    rows: list[dict[str, str]],
    output_path: Path,
    run_id: str,
    seed: int | str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
    plots = (
        ("current_raw_episode_return", "Raw episode return", "#1769aa"),
        ("loss", "Huber loss", "#b24c2f"),
        ("q_mean", "Selected Q mean", "#2f7d32"),
        ("epsilon", "Epsilon", "#7b4ca0"),
    )
    for axis, (field, label, color) in zip(axes.flat, plots, strict=True):
        steps, values = _series(rows, field)
        if not values:
            axis.text(0.5, 0.5, f"No {field} values", ha="center", va="center")
            axis.set_axis_off()
            continue
        axis.plot(steps, values, color=color, linewidth=1.1)
        axis.set_title(label)
        axis.set_xlabel("Environment step")
        axis.grid(True, alpha=0.2)
    figure.suptitle(f"Day 12 training metrics\nrun={run_id}, seed={seed}")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def visualize_run(
    run_id: str,
    *,
    runs_dir: Path = Path("runs"),
    output_dir: Path = Path("assets/day12"),
) -> dict[str, str]:
    """Read one run's metrics and save evidence-bearing PNGs."""

    run_dir = runs_dir / run_id
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    rows = _read_rows(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_run_id, seed = _metadata(run_dir)

    completed_steps, completed_returns = _series(rows, "raw_episode_return")
    if completed_returns:
        return_steps, return_values = completed_steps, completed_returns
        return_label = "completed raw episode return"
    else:
        return_steps, return_values = _series(rows, "current_raw_episode_return")
        return_label = "current raw episode return"

    loss_steps, loss_values = _series(rows, "loss")
    if not loss_values:
        raise ValueError("metrics.csv contains no optimizer loss values")

    return_path = output_dir / "training-return.png"
    loss_path = output_dir / "training-loss.png"
    overview_path = output_dir / "training-overview.png"
    _save_line_plot(
        steps=return_steps,
        values=return_values,
        title="Raw return during DQN training",
        ylabel="Raw Atari reward",
        output_path=return_path,
        label=return_label,
        run_id=metadata_run_id,
        seed=seed,
    )
    _save_line_plot(
        steps=loss_steps,
        values=loss_values,
        title="DQN Huber loss",
        ylabel="Loss",
        output_path=loss_path,
        label="optimizer update loss",
        run_id=metadata_run_id,
        seed=seed,
    )
    _save_overview(
        rows=rows,
        output_path=overview_path,
        run_id=metadata_run_id,
        seed=seed,
    )
    return {
        "run_dir": str(run_dir),
        "metrics": str(metrics_path),
        "training_return": str(return_path),
        "training_loss": str(loss_path),
        "training_overview": str(overview_path),
        "rows": str(len(rows)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = visualize_run(
        args.run_id,
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
