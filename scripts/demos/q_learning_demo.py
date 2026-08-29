"""A small, reproducible tabular Q-Learning example for Day 6."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias


ACTIONS = ("LEFT", "RIGHT")
STATES = (0, 1)
TERMINAL_STATE = 2
QTable: TypeAlias = dict[int, dict[str, float]]
TRACE_FIELDS = (
    "episode",
    "step",
    "state",
    "action",
    "reward",
    "next_state",
    "terminated",
    "current_q",
    "next_q_max",
    "target",
    "td_error",
    "updated_q",
)


@dataclass(frozen=True)
class TrainingStep:
    """The observable values produced by one Q-Learning update."""

    episode: int
    step: int
    state: int
    action: str
    reward: float
    next_state: int
    terminated: bool
    current_q: float
    next_q_max: float
    target: float
    td_error: float
    updated_q: float


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")


def q_learning_update(
    current_q: float,
    reward: float,
    next_q_max: float,
    alpha: float,
    gamma: float,
    terminated: bool,
) -> tuple[float, float, float]:
    """Return ``updated_q``, ``td_error`` and ``target`` for one transition."""

    _validate_unit_interval("alpha", alpha)
    _validate_unit_interval("gamma", gamma)

    target = float(reward)
    if not terminated:
        target += gamma * float(next_q_max)

    td_error = target - float(current_q)
    updated_q = float(current_q) + alpha * td_error
    return updated_q, td_error, target


def epsilon_greedy_action(
    q_values: Sequence[float],
    epsilon: float,
    rng: random.Random,
) -> int:
    """Choose an action index by epsilon-greedy exploration."""

    _validate_unit_interval("epsilon", epsilon)
    if not q_values:
        raise ValueError("q_values must contain at least one value")

    if rng.random() < epsilon:
        return rng.randrange(len(q_values))

    return max(range(len(q_values)), key=q_values.__getitem__)


def toy_environment_step(
    state: int,
    action: str,
) -> tuple[int, float, bool]:
    """Advance the deterministic two-state toy environment by one action.

    From state 0, RIGHT reaches state 1 without an immediate reward. From
    state 1, RIGHT reaches the terminal state with reward 1. Every LEFT action
    ends the episode with reward 0.
    """

    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}")

    if state == 0 and action == "RIGHT":
        return 1, 0.0, False

    reward = 1.0 if state == 1 and action == "RIGHT" else 0.0
    return TERMINAL_STATE, reward, True


def _new_q_table() -> QTable:
    return {state: {action: 0.0 for action in ACTIONS} for state in STATES}


def _q_values_for_state(q_table: QTable, state: int) -> list[float]:
    return [q_table[state][action] for action in ACTIONS]


def _run_training(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int,
    trace_limit: int | None,
) -> tuple[QTable, list[TrainingStep]]:
    if episodes < 0:
        raise ValueError("episodes must be zero or greater")
    if trace_limit is not None and trace_limit < 0:
        raise ValueError("trace_limit must be zero or greater")

    # Validate all hyperparameters before consuming any random numbers.
    _validate_unit_interval("alpha", alpha)
    _validate_unit_interval("gamma", gamma)
    _validate_unit_interval("epsilon", epsilon)

    rng = random.Random(seed)
    q_table = _new_q_table()
    trace: list[TrainingStep] = []

    for episode in range(1, episodes + 1):
        state = 0
        step = 0

        while state != TERMINAL_STATE:
            action_index = epsilon_greedy_action(
                _q_values_for_state(q_table, state), epsilon, rng
            )
            action = ACTIONS[action_index]
            next_state, reward, terminated = toy_environment_step(state, action)

            next_q_max = (
                max(q_table[next_state].values())
                if not terminated
                else 0.0
            )
            current_q = q_table[state][action]
            updated_q, td_error, target = q_learning_update(
                current_q=current_q,
                reward=reward,
                next_q_max=next_q_max,
                alpha=alpha,
                gamma=gamma,
                terminated=terminated,
            )
            q_table[state][action] = updated_q

            if trace_limit is None or len(trace) < trace_limit:
                trace.append(
                    TrainingStep(
                        episode=episode,
                        step=step,
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        terminated=terminated,
                        current_q=current_q,
                        next_q_max=next_q_max,
                        target=target,
                        td_error=td_error,
                        updated_q=updated_q,
                    )
                )

            state = next_state
            step += 1

    return q_table, trace


def _trace_rows(trace: Sequence[TrainingStep]) -> list[dict[str, object]]:
    return [asdict(step) for step in trace]


def write_trace_csv(path: str | Path, trace: Sequence[TrainingStep]) -> None:
    """Write one machine-readable row for every actual Q-learning update."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _trace_rows(trace)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_trace_json(
    path: str | Path,
    trace: Sequence[TrainingStep],
    *,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int,
) -> None:
    """Write trace rows together with the parameters used to produce them."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "episodes": episodes,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "seed": seed,
            "states": list(STATES),
            "actions": list(ACTIONS),
            "terminal_state": TERMINAL_STATE,
        },
        "updates": _trace_rows(trace),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train_q_learning(
    episodes: int = 20,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    seed: int = 42,
) -> QTable:
    """Train a Q-table on the toy environment and return the learned values."""

    q_table, _ = _run_training(
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        seed=seed,
        trace_limit=0,
    )
    return q_table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demonstrate tabular Q-Learning on a deterministic toy MDP."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="number of training episodes (default: 20)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="learning rate in the inclusive range [0, 1] (default: 0.1)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor in the inclusive range [0, 1] (default: 0.99)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="exploration probability in the inclusive range [0, 1] (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--trace-csv",
        type=Path,
        help="write the complete update trace to a CSV file",
    )
    parser.add_argument(
        "--trace-json",
        type=Path,
        help="write the update trace and run metadata to a JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="skip human-readable output after exporting a trace",
    )
    return parser


def _format_state(state: int) -> str:
    return "TERMINAL" if state == TERMINAL_STATE else str(state)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        export_trace = args.trace_csv is not None or args.trace_json is not None
        q_table, trace = _run_training(
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=args.seed,
            trace_limit=None if export_trace else 8,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.trace_csv is not None:
        write_trace_csv(args.trace_csv, trace)
    if args.trace_json is not None:
        write_trace_json(
            args.trace_json,
            trace,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=args.seed,
        )

    if args.quiet:
        return 0

    print("Toy MDP: tabular Q-Learning")
    print("state 0 --RIGHT / 0--> state 1 --RIGHT / 1--> TERMINAL")
    print("any LEFT action ends the episode with reward 0")
    print(
        f"episodes = {args.episodes}, alpha = {args.alpha:g}, "
        f"gamma = {args.gamma:g}, epsilon = {args.epsilon:g}, seed = {args.seed}"
    )
    print()
    print("First Q-Learning updates")
    for item in trace:
        print(
            f"  episode {item.episode}, step {item.step}: "
            f"state={item.state} action={item.action} "
            f"reward={item.reward:g} "
            f"next_state={_format_state(item.next_state)} "
            f"target={item.target:.6f} "
            f"td_error={item.td_error:.6f} "
            f"updated_q={item.updated_q:.6f}"
        )
    if not trace:
        print("  (no episodes or updates requested)")

    print()
    print("Final Q-table")
    for state in STATES:
        values = ", ".join(
            f"{action}={q_table[state][action]:.6f}" for action in ACTIONS
        )
        print(f"  state {state}: {values}")

    print()
    print("Greedy policy from the learned table")
    for state in STATES:
        best_action = ACTIONS[
            epsilon_greedy_action(
                _q_values_for_state(q_table, state),
                epsilon=0.0,
                rng=random.Random(0),
            )
        ]
        print(f"  state {state} -> {best_action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
