"""Collect a reproducible random-policy reference on the project environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.training.diagnostics import action_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a random-policy Breakout baseline."
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def collect_random_baseline(*, episodes: int, seed: int) -> dict[str, Any]:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    env = make_breakout_env()
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    episode_lengths: list[int] = []
    actions: list[int] = []
    try:
        observation, _ = env.reset(seed=seed)
        del observation
        for episode_index in range(episodes):
            if episode_index > 0:
                observation, _ = env.reset()
                del observation
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = int(rng.integers(0, env.action_space.n))
                actions.append(action)
                _, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
            returns.append(episode_return)
            episode_lengths.append(episode_length)
    finally:
        env.close()

    return {
        "policy": "random",
        "seed": seed,
        "episodes": episodes,
        "returns": returns,
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "episode_lengths": episode_lengths,
        "action_counts": action_counts(actions),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = collect_random_baseline(episodes=args.episodes, seed=args.seed)
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"Random baseline failed to start: {error}")
        return 2

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Baseline artifact written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
