"""Plot the target values produced by the Day 17 crafted fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize actual Vanilla and Double DQN target computations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/day17/dqn-vs-double-targets.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day17/dqn-vs-double-targets.png"),
    )
    return parser


def plot_target_comparison(payload: dict[str, Any], output: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key in ("vanilla", "double_dqn"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"target comparison is missing {key}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    methods = ("Vanilla DQN", "Double DQN")
    vanilla = payload["vanilla"]
    double = payload["double_dqn"]
    evaluated_values = [
        float(vanilla["evaluated_next_value"]),
        float(double["evaluated_next_value"]),
    ]
    final_targets = [
        float(vanilla["final_target"]),
        float(double["final_target"]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), constrained_layout=True)
    bars = axes[0].bar(methods, evaluated_values, color=("#6b7280", "#2563eb"))
    axes[0].set_title("next-state evaluated value")
    axes[0].set_ylabel("Q-value")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, evaluated_values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    bars = axes[1].bar(methods, final_targets, color=("#6b7280", "#2563eb"))
    axes[1].set_title("final Bellman target")
    axes[1].set_ylabel("reward + gamma × value")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, final_targets, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    figure.suptitle(
        "Online selects, target evaluates\n"
        f"selected actions: {vanilla['selected_action']} vs {double['selected_action']}"
    )
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"target comparison not found: {args.input}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("target comparison must be a JSON object")
    plot_target_comparison(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
