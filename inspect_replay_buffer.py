"""Collect real Breakout transitions and inspect replay storage contracts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.replay import ReplayBuffer
from breakout_rl.replay_tensors import replay_batch_to_tensors


def positive_int(value: str) -> int:
    """Parse a positive integer for the inspection CLI."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse replay inspection options."""

    parser = argparse.ArgumentParser(
        description="Collect Breakout transitions and inspect ReplayBuffer contracts."
    )
    parser.add_argument(
        "--capacity",
        type=positive_int,
        default=10_000,
        help="number of transitions preallocated in the ring buffer (default: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=32,
        help="number of transitions to sample (default: 32)",
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=64,
        help="number of real Breakout transitions to collect (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used for environment actions and replay sampling (default: 42)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device for the model-boundary tensor check (default: cpu)",
    )
    args = parser.parse_args(argv)
    if args.batch_size > args.capacity:
        parser.error("--batch-size cannot exceed --capacity")
    return args


def format_bytes(byte_count: int) -> str:
    """Format a byte count with binary units."""

    if byte_count < 1024:
        return f"{byte_count} B"
    mebibytes = byte_count / 1024**2
    if mebibytes < 1024:
        return f"{mebibytes:.3f} MiB"
    return f"{mebibytes / 1024:.3f} GiB"


def collect_real_transitions(
    buffer: ReplayBuffer,
    *,
    steps: int,
    seed: int,
) -> None:
    """Collect transitions from the project's actual preprocessed environment."""

    env = make_breakout_env(render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=seed)
        env.action_space.seed(seed)

        for _ in range(steps):
            state = np.asarray(observation)
            action = int(env.action_space.sample())
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = np.asarray(next_observation)
            buffer.add(
                state,
                action,
                reward,
                next_state,
                terminated,
                truncated,
            )

            if terminated or truncated:
                observation, _ = env.reset()
            else:
                observation = next_observation
    finally:
        env.close()


def inspect_replay_buffer(
    *,
    capacity: int,
    batch_size: int,
    steps: int,
    seed: int,
    device: str,
) -> None:
    """Run the storage, sampling, and model-boundary inspection."""

    if batch_size > capacity:
        raise ValueError("batch_size cannot exceed capacity")

    buffer = ReplayBuffer(capacity=capacity)
    collect_real_transitions(buffer, steps=max(steps, batch_size), seed=seed)

    batch, indices = buffer.sample_with_indices(
        batch_size,
        np.random.default_rng(seed),
    )
    tensor_batch = replay_batch_to_tensors(batch, device=device)

    print("ReplayBuffer")
    print(f"  capacity          : {buffer.capacity}")
    print(f"  current size      : {len(buffer)}")
    print(f"  write index       : {buffer.write_index}")
    print(f"  observation shape : {buffer.observation_shape}")
    print(f"  observation dtype : {buffer.states.dtype}")
    print(f"  allocated memory  : {format_bytes(buffer.allocated_bytes)}")
    print(f"  sampled slots     : {indices.tolist()}")
    print()

    print("Sampled NumPy batch")
    for field in (
        "states",
        "actions",
        "rewards",
        "next_states",
        "terminated",
        "truncated",
    ):
        value = getattr(batch, field)
        print(f"  {field:12}: shape={value.shape}, dtype={value.dtype}")
    print()

    print("Model-boundary tensor batch")
    for field in (
        "states",
        "actions",
        "rewards",
        "next_states",
        "terminated",
        "truncated",
    ):
        value = getattr(tensor_batch, field)
        print(
            f"  {field:12}: shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, device={value.device}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the replay inspection CLI."""

    args = parse_args(argv)
    inspect_replay_buffer(
        capacity=args.capacity,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
