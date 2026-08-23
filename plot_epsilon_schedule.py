"""Render a reproducible epsilon-greedy schedule figure."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from breakout_rl.exploration import LinearEpsilonSchedule

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "assets/day10/epsilon-schedule.png"


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot epsilon and its random/greedy branch probabilities."
    )
    parser.add_argument("--start", type=float, default=0.9)
    parser.add_argument("--end", type=float, default=0.05)
    parser.add_argument("--decay-steps", type=_non_negative_int, default=1000)
    parser.add_argument(
        "--max-step",
        type=_non_negative_int,
        default=1200,
        help="last environment step shown in the figure (default: 1200)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def schedule_points(
    schedule: LinearEpsilonSchedule,
    *,
    max_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the real schedule at every plotted environment step."""

    if max_step < 0:
        raise ValueError("max_step must be greater than or equal to 0")
    steps = np.arange(max_step + 1, dtype=np.int64)
    epsilon = np.asarray(
        [schedule.value(int(step)) for step in steps],
        dtype=np.float64,
    )
    return steps, epsilon


def create_figure(
    schedule: LinearEpsilonSchedule,
    *,
    max_step: int,
    output: Path,
    command: str,
) -> Path:
    """Save the schedule and branch probabilities produced by the schedule."""

    steps, epsilon = schedule_points(schedule, max_step=max_step)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        constrained_layout=True,
    )
    figure.suptitle("Epsilon-greedy schedule: exploration → exploitation", fontsize=16)

    schedule_axis, branch_axis = axes
    schedule_axis.plot(
        steps,
        epsilon,
        color="#1f77b4",
        linewidth=2.5,
        label="epsilon (random-action probability)",
    )
    schedule_axis.axvline(
        schedule.decay_steps,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"decay end = {schedule.decay_steps:,}",
    )
    schedule_axis.set_ylabel("epsilon")
    schedule_axis.set_ylim(-0.02, 1.02)
    schedule_axis.set_title(
        f"start={schedule.start:g} → end={schedule.end:g}",
        loc="left",
        fontsize=11,
    )
    schedule_axis.grid(axis="y", alpha=0.25)
    schedule_axis.legend(loc="best", frameon=False)

    branch_axis.plot(
        steps,
        epsilon,
        color="#d62728",
        linewidth=2.0,
        label="random branch = epsilon",
    )
    branch_axis.plot(
        steps,
        1.0 - epsilon,
        color="#2ca02c",
        linewidth=2.0,
        label="greedy branch = 1 - epsilon",
    )
    branch_axis.set_xlabel("environment step")
    branch_axis.set_ylabel("branch probability")
    branch_axis.set_ylim(-0.02, 1.02)
    branch_axis.grid(axis="y", alpha=0.25)
    branch_axis.legend(loc="center right", frameon=False)

    figure.savefig(output, dpi=160, format="png")
    plt.close(figure)

    metadata = {
        "command": command,
        "source": "breakout_rl.exploration.LinearEpsilonSchedule",
        "start": float(schedule.start),
        "end": float(schedule.end),
        "decay_steps": int(schedule.decay_steps),
        "max_step": int(max_step),
        "point_count": int(len(steps)),
        "random_branch": "epsilon",
        "greedy_branch": "1 - epsilon",
        "sampled_points": [
            {
                "step": int(step),
                "epsilon": float(schedule.value(int(step))),
            }
            for step in sorted(
                {0, schedule.decay_steps // 2, schedule.decay_steps, max_step}
            )
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        schedule = LinearEpsilonSchedule(
            start=args.start,
            end=args.end,
            decay_steps=args.decay_steps,
        )
        if args.max_step < schedule.decay_steps:
            raise ValueError("max_step must be at least decay_steps")
        output = create_figure(
            schedule,
            max_step=args.max_step,
            output=args.output,
            command=(
                "python plot_epsilon_schedule.py "
                f"--start {args.start:g} --end {args.end:g} "
                f"--decay-steps {args.decay_steps} --max-step {args.max_step}"
            ),
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    print(f"Saved figure: {output}")
    print(f"Saved metadata: {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
