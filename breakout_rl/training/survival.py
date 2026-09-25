"""Survival metrics shared by training and raw-score evaluation."""

from __future__ import annotations

import math
import operator
from numbers import Real
from typing import Any, Sequence


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(parsed)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite real number")
    return parsed


def compute_episode_survival_metrics(
    *,
    raw_score: float,
    episode_length: int,
    life_loss_steps: Sequence[int],
) -> dict[str, Any]:
    """Compute interpretable survival metrics for one completed episode.

    ``life_loss_steps`` is one-indexed and records the environment transition
    on which ``info["fire_reset_life_loss"]`` became true.  The number of
    life-loss events is deliberately not treated as the main survival metric:
    a completed Atari episode can consume the same fixed number of lives for
    every policy.  ``score_per_life`` uses one life per loss event, with one
    life as the minimum denominator for episodes with no loss.
    """

    parsed_score = _finite_float(raw_score, name="raw_score")
    parsed_length = _positive_int(episode_length, name="episode_length")
    if isinstance(life_loss_steps, (str, bytes)):
        raise TypeError("life_loss_steps must be a sequence of integers")

    parsed_steps: list[int] = []
    for index, raw_step in enumerate(life_loss_steps):
        step = _positive_int(raw_step, name=f"life_loss_steps[{index}]")
        if step > parsed_length:
            raise ValueError("life-loss step must not exceed episode_length")
        if parsed_steps and step <= parsed_steps[-1]:
            raise ValueError("life_loss_steps must be strictly increasing")
        parsed_steps.append(step)

    # The first loss is reported separately as ``time_to_first_life_loss``;
    # this series contains only intervals *between* successive losses.
    intervals = [
        step - previous for previous, step in zip(parsed_steps[:-1], parsed_steps[1:])
    ]
    return {
        "life_loss_count": len(parsed_steps),
        "score_per_life": float(parsed_score / max(1, len(parsed_steps))),
        "frames_between_life_losses": (
            float(sum(intervals) / len(intervals)) if intervals else None
        ),
        "time_to_first_life_loss": parsed_steps[0] if parsed_steps else None,
        "life_losses_per_1000_steps": float(
            len(parsed_steps) / parsed_length * 1000.0
        ),
        "life_loss_intervals": intervals,
    }


def life_losses_per_1000_steps(life_loss_count: int, steps: int) -> float:
    """Compute a cumulative life-loss rate with a zero-step guard."""

    try:
        parsed_count = operator.index(life_loss_count)
    except TypeError as error:
        raise TypeError("life_loss_count must be a non-negative integer") from error
    try:
        parsed_steps = operator.index(steps)
    except TypeError as error:
        raise TypeError("steps must be a non-negative integer") from error
    if parsed_count < 0:
        raise ValueError("life_loss_count must be non-negative")
    if parsed_steps < 0:
        raise ValueError("steps must be non-negative")
    if parsed_steps == 0:
        return 0.0
    return float(parsed_count / parsed_steps * 1000.0)


__all__ = [
    "compute_episode_survival_metrics",
    "life_losses_per_1000_steps",
]
