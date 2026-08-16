"""Inspect the raw, preprocessed, and stacked Breakout observations."""

from __future__ import annotations

import argparse

from breakout_env import (
    make_breakout_env,
    make_breakout_preprocessed_env,
    make_breakout_raw_env,
)


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for the command-line interface."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse the options needed by the preprocessing inspection."""

    parser = argparse.ArgumentParser(
        description="Print the Breakout preprocessing observation contracts."
    )
    parser.add_argument(
        "--steps",
        type=non_negative_int,
        default=8,
        help="number of processed environment steps to inspect (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used for reset and random action sampling (default: 42)",
    )
    return parser.parse_args()


def print_observation_contract(label: str, observation: object) -> None:
    """Print a compact observation shape and dtype summary."""

    print(label)
    print(f"  shape : {observation.shape}")  # type: ignore[attr-defined]
    print(f"  dtype : {observation.dtype}")  # type: ignore[attr-defined]


def inspect_preprocessing(steps: int, seed: int) -> None:
    """Show the three observation contracts and a short real interaction."""

    raw_env = make_breakout_raw_env(render_mode="rgb_array")
    preprocessed_env = make_breakout_preprocessed_env(render_mode="rgb_array")
    stacked_env = make_breakout_env(render_mode="rgb_array")

    try:
        raw_observation, _ = raw_env.reset(seed=seed)
        preprocessed_observation, _ = preprocessed_env.reset(seed=seed)
        stacked_observation, _ = stacked_env.reset(seed=seed)

        print_observation_contract("Raw Atari observation", raw_observation)
        print()
        print_observation_contract(
            "After AtariPreprocessing", preprocessed_observation
        )
        print()
        print_observation_contract(
            "After FrameStackObservation(stack_size=4)", stacked_observation
        )
        print(f"  min   : {int(stacked_observation.min())}")
        print(f"  max   : {int(stacked_observation.max())}")
        print()

        action_meanings = stacked_env.unwrapped.get_action_meanings()
        stacked_env.action_space.seed(seed)
        episode_return = 0.0
        completed_episodes = 0

        for step_index in range(steps):
            action = int(stacked_env.action_space.sample())
            (
                stacked_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = stacked_env.step(action)

            episode_return += float(reward)
            print(f"Step {step_index}")
            print(
                "  action       : "
                f"{action} ({action_meanings[action]})"
            )
            print(
                "  stacked_obs  : "
                f"shape={stacked_observation.shape}, "
                f"dtype={stacked_observation.dtype}"
            )
            print(f"  reward       : {reward}")
            print(f"  terminated   : {terminated}")
            print(f"  truncated    : {truncated}")

            if terminated or truncated:
                completed_episodes += 1
                episode_return = 0.0
                stacked_observation, _ = stacked_env.reset()

        print()
        print(f"Processed steps: {steps}")
        print(f"Episode return: {episode_return}")
        print(f"Episodes ended: {completed_episodes}")
    finally:
        raw_env.close()
        preprocessed_env.close()
        stacked_env.close()


def main() -> None:
    """Run the preprocessing inspection CLI."""

    args = parse_args()
    inspect_preprocessing(steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
