"""Deterministic action-control summaries for training transition logs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from math import floor, isfinite
from typing import Any


def _row_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0", ""}:
            return False
    raise ValueError(f"{field} must be a boolean value")


def _row_action(row: Mapping[str, Any]) -> str:
    value = row.get("action_name")
    if value is None or value == "":
        value = row.get("action")
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"0", "1", "2", "3"}:
            return {"0": "NOOP", "1": "FIRE", "2": "RIGHT", "3": "LEFT"}[normalized]
        if normalized:
            return normalized
    if isinstance(value, int) and not isinstance(value, bool):
        return {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}.get(
            value,
            f"ACTION_{value}",
        )
    raise ValueError("training metrics row must contain an action_name or action")


def summarize_training_control_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    high_score_threshold: float = 500.0,
    late_episode_fraction: float = 0.75,
) -> dict[str, Any]:
    """Summarize action changes and score/episode-phase action distributions.

    Action-switch rates compare adjacent transitions from the same environment
    and episode. A direct LEFT-to-RIGHT or RIGHT-to-LEFT pair is one reversal;
    a NOOP or FIRE between the two paddle directions breaks that pair. The
    late-episode view uses the final configured fraction of completed episodes.
    The high-score view includes transitions whose cumulative raw episode score
    has reached ``high_score_threshold``.
    """

    if isinstance(high_score_threshold, bool) or not isinstance(
        high_score_threshold,
        (int, float),
    ):
        raise TypeError("high_score_threshold must be a finite number")
    if not isfinite(float(high_score_threshold)) or high_score_threshold < 0:
        raise ValueError("high_score_threshold must be a finite non-negative number")
    if isinstance(late_episode_fraction, bool) or not isinstance(
        late_episode_fraction,
        (int, float),
    ):
        raise TypeError("late_episode_fraction must be between 0 and 1")
    if not isfinite(float(late_episode_fraction)) or not 0 < late_episode_fraction < 1:
        raise ValueError("late_episode_fraction must be between 0 and 1")

    action_counts: Counter[str] = Counter()
    high_score_counts: Counter[str] = Counter()
    late_episode_counts: Counter[str] = Counter()
    previous_action: dict[int, str] = {}
    active_episode_actions: dict[int, list[str]] = {}
    transition_count = 0
    adjacent_action_pairs = 0
    action_switch_count = 0
    left_right_reversal_count = 0
    completed_episodes = 0
    high_score_sample_count = 0

    for row in rows:
        try:
            environment_index = int(row["environment_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "training metrics row must contain an integer environment_index"
            ) from error
        action = _row_action(row)
        try:
            raw_score = float(row["current_raw_episode_return"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "training metrics row must contain current_raw_episode_return"
            ) from error
        if not isfinite(raw_score):
            raise ValueError("current_raw_episode_return must be finite")

        done = _row_bool(row.get("terminated", False), field="terminated") or _row_bool(
            row.get("truncated", False),
            field="truncated",
        )
        if done:
            completed_score = row.get("raw_episode_return")
            if completed_score not in (None, ""):
                try:
                    raw_score = max(raw_score, float(completed_score))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "raw_episode_return must be numeric when present"
                    ) from error
                if not isfinite(raw_score):
                    raise ValueError("raw_episode_return must be finite")

        transition_count += 1
        action_counts[action] += 1
        active_episode_actions.setdefault(environment_index, []).append(action)
        if raw_score >= high_score_threshold:
            high_score_counts[action] += 1
            high_score_sample_count += 1

        previous = previous_action.get(environment_index)
        if previous is not None:
            adjacent_action_pairs += 1
            if action != previous:
                action_switch_count += 1
            if (previous, action) in {("LEFT", "RIGHT"), ("RIGHT", "LEFT")}:
                left_right_reversal_count += 1
        previous_action[environment_index] = action

        if done:
            completed_episodes += 1
            episode_actions = active_episode_actions.pop(environment_index)
            start = floor(len(episode_actions) * late_episode_fraction)
            late_episode_counts.update(episode_actions[start:])
            previous_action.pop(environment_index, None)

    late_episode_sample_count = sum(late_episode_counts.values())
    steering_action_count = action_counts["LEFT"] + action_counts["RIGHT"]
    return {
        "transition_count": transition_count,
        "action_distribution": dict(sorted(action_counts.items())),
        "noop_count": action_counts["NOOP"],
        "noop_rate": (
            action_counts["NOOP"] / transition_count if transition_count else 0.0
        ),
        "paddle_control_action_count": steering_action_count,
        "paddle_control_cadence": (
            steering_action_count / transition_count if transition_count else 0.0
        ),
        "adjacent_action_pairs": adjacent_action_pairs,
        "action_switch_count": action_switch_count,
        "action_switch_rate": (
            action_switch_count / adjacent_action_pairs
            if adjacent_action_pairs
            else 0.0
        ),
        "left_right_reversal_count": left_right_reversal_count,
        "left_right_reversal_rate": (
            left_right_reversal_count / adjacent_action_pairs
            if adjacent_action_pairs
            else 0.0
        ),
        "completed_episodes": completed_episodes,
        "late_episode": {
            "selection_rule": (
                f"last {1.0 - late_episode_fraction:.0%} of completed episodes "
                "by agent transition count"
            ),
            "episode_count": completed_episodes,
            "sample_count": late_episode_sample_count,
            "action_distribution": dict(sorted(late_episode_counts.items())),
        },
        "high_score": {
            "selection_rule": (
                "cumulative raw episode score is at least the configured threshold"
            ),
            "threshold_raw_score": float(high_score_threshold),
            "sample_count": high_score_sample_count,
            "action_distribution": dict(sorted(high_score_counts.items())),
        },
    }


__all__ = ["summarize_training_control_metrics"]
