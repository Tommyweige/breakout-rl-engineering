"""Small, deterministic examples for discounted returns and Bellman targets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _validate_gamma(gamma: float) -> None:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1 inclusive")


def discounted_return(rewards: list[float], gamma: float) -> float:
    """Return the discounted sum of rewards from the current time step."""

    _validate_gamma(gamma)

    total = 0.0
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
    return total


def bellman_target(
    reward: float,
    next_value: float,
    gamma: float,
    terminated: bool,
) -> float:
    """Compute a one-step Bellman target without bootstrapping terminals."""

    _validate_gamma(gamma)
    if terminated:
        return float(reward)
    return float(reward) + gamma * float(next_value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demonstrate discounted returns and one-step Bellman targets."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor in the inclusive range [0, 1] (default: 0.99)",
    )
    return parser


def _format_rewards(rewards: Sequence[float]) -> str:
    return "[" + ", ".join(f"{reward:g}" for reward in rewards) + "]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_gamma(args.gamma)
    except ValueError as error:
        parser.error(str(error))

    safe_rewards = [1.0]
    wait_rewards = [0.0, 0.0, 3.0]
    safe_return = discounted_return(safe_rewards, args.gamma)
    wait_return = discounted_return(wait_rewards, args.gamma)

    print("Toy MDP: choose between SAFE and WAIT")
    print(f"gamma = {args.gamma:g}")
    print()
    print("SAFE: reward 1 -> TERMINAL")
    print(f"  rewards = {_format_rewards(safe_rewards)}")
    print(f"  discounted return = {safe_return:.6f}")
    print()
    print("WAIT: reward 0 -> GOOD_STATE -> reward 0 -> FINISH -> reward 3")
    print(f"  rewards = {_format_rewards(wait_rewards)}")
    print(f"  discounted return = {wait_return:.6f}")
    print()
    print("One-step Bellman targets")
    terminal_target = bellman_target(
        reward=1.0,
        next_value=999.0,
        gamma=args.gamma,
        terminated=True,
    )
    continuing_target = bellman_target(
        reward=0.0,
        next_value=3.0,
        gamma=args.gamma,
        terminated=False,
    )
    print(f"  terminal:    reward 1 + no bootstrap = {terminal_target:.6f}")
    print(
        "  non-terminal: reward 0 + "
        f"{args.gamma:g} * next_value 3 = {continuing_target:.6f}"
    )
    print()
    print("A zero immediate reward can still lead to a valuable future.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
