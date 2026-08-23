"""Inspect reproducible epsilon-greedy decisions on fixed Q-values."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch

from breakout_rl.exploration import select_epsilon_greedy_action

DEFAULT_Q_VALUES = (1.0, 2.0, 5.0, 3.0)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 inclusive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect reproducible epsilon-greedy action sampling."
    )
    parser.add_argument(
        "--epsilon",
        type=_probability,
        default=0.1,
        help="probability of taking a random action (default: 0.1)",
    )
    parser.add_argument(
        "--samples",
        type=_non_negative_int,
        default=100,
        help="number of action selections to sample (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for the explicit NumPy generator (default: 42)",
    )
    parser.add_argument(
        "--q-values",
        type=float,
        nargs="+",
        default=DEFAULT_Q_VALUES,
        metavar="Q",
        help="one fixed Q-value per action (default: 1 2 5 3)",
    )
    return parser.parse_args(argv)


def inspect_exploration(
    *,
    q_values: Sequence[float],
    epsilon: float,
    samples: int,
    seed: int,
) -> tuple[int, int, Counter[int]]:
    """Return random/greedy counts and action counts from one seeded run."""

    if samples < 0:
        raise ValueError("samples must be greater than or equal to 0")

    values = torch.tensor(tuple(q_values), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    random_decisions = 0
    greedy_decisions = 0
    action_counts: Counter[int] = Counter()

    for _ in range(samples):
        action, source = select_epsilon_greedy_action(
            values,
            epsilon,
            action_space_n=values.numel(),
            rng=rng,
        )
        action_counts[action] += 1
        if source == "random":
            random_decisions += 1
        else:
            greedy_decisions += 1

    return random_decisions, greedy_decisions, action_counts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.q_values:
        raise SystemExit("error: --q-values must contain at least one value")

    try:
        random_decisions, greedy_decisions, action_counts = inspect_exploration(
            q_values=args.q_values,
            epsilon=args.epsilon,
            samples=args.samples,
            seed=args.seed,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    q_values = [float(value) for value in args.q_values]
    greedy_action_index = int(np.argmax(q_values))
    print(f"epsilon             : {args.epsilon:g}")
    print(f"samples             : {args.samples}")
    print(f"seed                : {args.seed}")
    print(f"q-values            : {q_values}")
    print(f"greedy action index : {greedy_action_index}")
    print(f"random decisions    : {random_decisions}")
    print(f"greedy decisions    : {greedy_decisions}")
    print("action counts")
    for action_index in range(len(q_values)):
        print(f"  action {action_index}: {action_counts[action_index]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
