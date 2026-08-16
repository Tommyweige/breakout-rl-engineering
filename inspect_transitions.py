"""Inspect the data produced by one step of Atari Breakout."""

from __future__ import annotations

import argparse

import ale_py
import gymnasium as gym


gym.register_envs(ale_py)

ENVIRONMENT_ID = "ALE/Breakout-v5"


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for the command-line interface."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse the small set of options needed by the transition demo."""

    parser = argparse.ArgumentParser(
        description="Print compact state/action/reward transition summaries."
    )
    parser.add_argument(
        "--steps",
        type=non_negative_int,
        default=20,
        help="number of environment steps to inspect (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used for the environment and random action sampling (default: 42)",
    )
    return parser.parse_args()


def observation_summary(observation: object) -> str:
    """Return shape and dtype without printing the full RGB frame."""

    return f"shape={observation.shape}, dtype={observation.dtype}"  # type: ignore[attr-defined]


def inspect_transitions(steps: int, seed: int) -> None:
    """Collect and print a short sequence of real Breakout transitions."""

    env = gym.make(ENVIRONMENT_ID, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        action_meanings = env.unwrapped.get_action_meanings()

        episode_return = 0.0
        completed_episode_returns: list[float] = []

        for step_index in range(steps):
            state = observation
            action = int(env.action_space.sample())
            (
                next_state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            episode_return += float(reward)
            action_meaning = action_meanings[action]

            print(f"Step {step_index}")
            print(f"  state       : {observation_summary(state)}")
            print(f"  action      : {action} ({action_meaning})")
            print(f"  reward      : {reward}")
            print(f"  next_state  : {observation_summary(next_state)}")
            print(f"  terminated  : {terminated}")
            print(f"  truncated   : {truncated}")

            observation = next_state

            if terminated or truncated:
                completed_episode_returns.append(episode_return)
                episode_return = 0.0
                observation, _ = env.reset()

        print()
        print(f"Transitions collected: {steps}")
        print(f"Episode return: {episode_return}")
        if completed_episode_returns:
            print(
                "Last completed episode return: "
                f"{completed_episode_returns[-1]}"
            )
        print(f"Episodes ended: {len(completed_episode_returns)}")
    finally:
        env.close()


def main() -> None:
    """Run the transition inspection CLI."""

    args = parse_args()
    inspect_transitions(steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
