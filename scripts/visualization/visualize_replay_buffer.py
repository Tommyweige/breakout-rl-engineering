"""Visualize real replay writes, sampling, and estimated memory usage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from breakout_env import make_breakout_env
from breakout_rl.replay import (
    DEFAULT_OBSERVATION_SHAPE,
    ReplayBuffer,
    estimate_replay_memory_bytes,
)


DEFAULT_OUTPUT = Path("assets/day09/replay-buffer.png")
DEFAULT_MEMORY_CAPACITIES = (10_000, 100_000, 1_000_000)


def positive_int(value: str) -> int:
    """Parse a positive integer for the visualization CLI."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_capacities(value: str) -> tuple[int, ...]:
    """Parse comma-separated capacities used by the memory estimator."""

    try:
        capacities = tuple(positive_int(part.strip()) for part in value.split(","))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError(
            "must be comma-separated positive integers"
        ) from error
    if not capacities:
        raise argparse.ArgumentTypeError("must contain at least one capacity")
    return capacities


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse visualization options."""

    parser = argparse.ArgumentParser(
        description=(
            "Collect real Breakout transitions and visualize replay "
            "wraparound and memory estimates."
        )
    )
    parser.add_argument(
        "--capacity",
        type=positive_int,
        default=5,
        help="small ring-buffer capacity shown in the write map (default: 5)",
    )
    parser.add_argument(
        "--writes",
        type=positive_int,
        default=8,
        help="number of real transitions to add to the visualization buffer (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=3,
        help="number of sampled transitions shown below the write map (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for real Breakout actions and replay sampling (default: 42)",
    )
    parser.add_argument(
        "--memory-capacities",
        type=parse_capacities,
        default=DEFAULT_MEMORY_CAPACITIES,
        help="comma-separated capacities for the estimator (default: 10000,100000,1000000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path (default: assets/day09/replay-buffer.png)",
    )
    args = parser.parse_args(argv)
    if args.batch_size > args.capacity:
        parser.error("--batch-size cannot exceed --capacity")
    return args


def collect_real_replay(
    *,
    capacity: int,
    writes: int,
    seed: int,
) -> tuple[ReplayBuffer, list[dict[str, int | float | bool]]]:
    """Collect real preprocessed Breakout transitions into a replay buffer."""

    buffer = ReplayBuffer(capacity=capacity)
    history: list[dict[str, int | float | bool]] = []
    env = make_breakout_env(render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=seed)
        env.action_space.seed(seed)

        for write_number in range(1, writes + 1):
            state = np.asarray(observation)
            action = int(env.action_space.sample())
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = np.asarray(next_observation)
            slot = buffer.add(
                state,
                action,
                reward,
                next_state,
                terminated,
                truncated,
            )
            history.append(
                {
                    "write_number": write_number,
                    "slot": slot,
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

            if terminated or truncated:
                observation, _ = env.reset()
            else:
                observation = next_observation
    finally:
        env.close()

    return buffer, history


def _draw_write_map(
    ax: Axes,
    *,
    buffer: ReplayBuffer,
    history: list[dict[str, int | float | bool]],
) -> None:
    """Draw the physical slots touched by each real write."""

    matrix = np.full((buffer.capacity, len(history)), np.nan, dtype=np.float32)
    for event in history:
        slot = int(event["slot"])
        write_number = int(event["write_number"])
        matrix[slot, write_number - 1] = write_number

    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=1,
        vmax=max(1, len(history)),
    )
    image.cmap.set_bad("#f1f1f1")

    for row in range(buffer.capacity):
        for column in range(len(history)):
            value = matrix[row, column]
            if not np.isnan(value):
                ax.text(
                    column,
                    row,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xticks(range(len(history)))
    ax.set_xticklabels([str(event["write_number"]) for event in history])
    ax.set_yticks(range(buffer.capacity))
    ax.set_yticklabels([str(slot) for slot in range(buffer.capacity)])
    ax.set_xlabel("write order")
    ax.set_ylabel("physical slot")
    ax.set_title(
        "Real writes into a fixed-capacity ring buffer\n"
        f"active slots, oldest → newest: {buffer.chronological_indices().tolist()}"
    )
    ax.grid(False)


def _draw_memory_estimates(
    ax: Axes,
    *,
    observation_shape: tuple[int, ...],
    capacities: tuple[int, ...],
) -> list[dict[str, int | float]]:
    """Draw memory estimates computed by the replay estimator."""

    estimates = [
        {
            "capacity": capacity,
            "bytes": estimate_replay_memory_bytes(capacity, observation_shape),
        }
        for capacity in capacities
    ]
    gib_values = np.asarray(
        [estimate["bytes"] for estimate in estimates],
        dtype=np.float64,
    ) / 1024**3
    bars = ax.bar(
        [f"{int(estimate['capacity']):,}" for estimate in estimates],
        gib_values,
        color="#4c78a8",
    )
    for bar, gib in zip(bars, gib_values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{gib:.3f} GiB",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("capacity (transitions)")
    ax.set_ylabel("allocated bytes (GiB)")
    ax.set_title(
        "Baseline replay memory estimate\n"
        f"shape={observation_shape}, storage=uint8 states + next_states"
    )
    ax.grid(axis="y", alpha=0.25)
    return [
        {
            "capacity": int(estimate["capacity"]),
            "bytes": int(estimate["bytes"]),
            "gib": float(gib),
        }
        for estimate, gib in zip(estimates, gib_values, strict=True)
    ]


def _draw_sampled_states(
    figure: Figure,
    grid_spec: SubplotSpec,
    *,
    batch_states: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Draw the newest frame from several sampled stacked observations."""

    subgrid = grid_spec.subgridspec(2, len(indices), height_ratios=(0.24, 1.0))
    title_axis = figure.add_subplot(subgrid[0, :])
    title_axis.set_title(
        "Real uniformly sampled states\n"
        f"batch={len(indices)}, each state shape={tuple(batch_states.shape[1:])}"
    )
    title_axis.axis("off")
    for column, (state, slot) in enumerate(zip(batch_states, indices, strict=True)):
        ax = figure.add_subplot(subgrid[1, column])
        ax.imshow(state[-1], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"slot {int(slot)}\nnewest frame")
        ax.set_xticks([])
        ax.set_yticks([])


def build_visualization(
    *,
    capacity: int,
    writes: int,
    batch_size: int,
    seed: int,
    memory_capacities: tuple[int, ...],
    output: Path,
) -> dict[str, object]:
    """Generate the PNG and machine-readable metadata from real run data."""

    if batch_size > capacity:
        raise ValueError("batch_size cannot exceed capacity")

    buffer, history = collect_real_replay(
        capacity=capacity,
        writes=writes,
        seed=seed,
    )
    batch, sample_indices = buffer.sample_with_indices(
        batch_size,
        np.random.default_rng(seed),
    )

    figure = plt.figure(figsize=(14, 10), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, height_ratios=(1.1, 1.0))
    write_axis = figure.add_subplot(layout[0, :])
    memory_axis = figure.add_subplot(layout[1, 0])

    _draw_write_map(write_axis, buffer=buffer, history=history)
    memory_estimates = _draw_memory_estimates(
        memory_axis,
        observation_shape=buffer.observation_shape,
        capacities=memory_capacities,
    )
    _draw_sampled_states(
        figure,
        layout[1, 1],
        batch_states=batch.states,
        indices=sample_indices,
    )
    figure.suptitle(
        "Day 9 Experience Replay: storage, wraparound, and sampling",
        fontsize=16,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    metadata: dict[str, object] = {
        "generated_by": "visualize_replay_buffer.py",
        "source": "make_breakout_env(render_mode='rgb_array')",
        "seed": seed,
        "capacity": capacity,
        "writes": writes,
        "batch_size": batch_size,
        "observation_shape": list(buffer.observation_shape),
        "observation_dtype": str(buffer.states.dtype),
        "buffer_size": len(buffer),
        "write_index": buffer.write_index,
        "oldest_to_newest_slots": buffer.chronological_indices().tolist(),
        "allocated_bytes": buffer.allocated_bytes,
        "write_history": history,
        "sampled_slots": sample_indices.tolist(),
        "sampled_state_shape": list(batch.states.shape),
        "sampled_state_dtype": str(batch.states.dtype),
        "memory_estimates": memory_estimates,
        "reproduction_command": (
            "conda run --name breakout-rl-engineering python "
            "visualize_replay_buffer.py --seed 42"
        ),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Day 9 visualization command."""

    args = parse_args(argv)
    metadata = build_visualization(
        capacity=args.capacity,
        writes=args.writes,
        batch_size=args.batch_size,
        seed=args.seed,
        memory_capacities=args.memory_capacities,
        output=args.output,
    )
    print(f"Saved figure: {args.output}")
    print(f"Saved metadata: {args.output.with_suffix('.json')}")
    print(f"Oldest → newest slots: {metadata['oldest_to_newest_slots']}")
    print(f"Sampled slots: {metadata['sampled_slots']}")


if __name__ == "__main__":
    main()
