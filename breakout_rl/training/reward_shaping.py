"""Training-reward shaping helpers.

The environment reward is deliberately kept separate from the reward used by
the Bellman update.  In particular, life-loss shaping is applied *after* any
game-reward clipping so a future experiment can change the penalty magnitude
without changing the Atari score or collapsing the penalty through a second
sign operation.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


LIFE_LOSS_INFO_KEY = "fire_reset_life_loss"


def validate_life_loss_penalty(value: Any) -> float:
    """Validate and normalize one configured life-loss penalty.

    Reward shaping is allowed to be neutral or negative.  A positive value is
    rejected because it would reward the event the experiment is intended to
    penalize.  Keeping this validation here makes direct callers and config
    construction share the same fail-closed behavior.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("life_loss_penalty must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("life_loss_penalty must be finite")
    if parsed > 0.0:
        raise ValueError("life_loss_penalty must be less than or equal to zero")
    return parsed


def shape_training_reward(
    raw_reward: float,
    *,
    reward_clip: bool,
    life_loss: bool,
    life_loss_penalty: float,
) -> float:
    """Build the reward stored in replay from one raw environment transition.

    ``raw_reward`` is never mutated.  With clipping enabled, only the Atari
    reward is sign-clipped; the configured life-loss penalty is then added as a
    separate term.  Thus ``raw=1, life_loss=True, penalty=-1`` produces ``0``
    while ``raw=0, life_loss=True, penalty=-1`` produces ``-1``.
    """

    if isinstance(raw_reward, bool) or not isinstance(raw_reward, Real):
        raise TypeError("raw_reward must be a real number")
    parsed_raw_reward = float(raw_reward)
    if not math.isfinite(parsed_raw_reward):
        raise ValueError("raw_reward must be finite")
    if not isinstance(reward_clip, bool):
        raise TypeError("reward_clip must be a boolean")
    if not isinstance(life_loss, bool):
        raise TypeError("life_loss must be a boolean")
    parsed_penalty = validate_life_loss_penalty(life_loss_penalty)

    clipped_game_reward = (
        float(math.copysign(1.0, parsed_raw_reward))
        if parsed_raw_reward != 0.0 and reward_clip
        else 0.0
        if reward_clip
        else parsed_raw_reward
    )
    training_reward = clipped_game_reward + (
        parsed_penalty if life_loss else 0.0
    )
    if not math.isfinite(training_reward):
        raise ValueError("training reward must be finite")
    return float(training_reward)


def reward_design_metadata(*, life_loss_penalty: float) -> dict[str, Any]:
    """Return explicit provenance for training artifacts."""

    penalty = validate_life_loss_penalty(life_loss_penalty)
    return {
        "raw_reward_source": "environment.step reward; never shaped",
        "training_reward_definition": (
            "sign(raw_reward) when reward_clip=true, plus life_loss_penalty "
            "when info['fire_reset_life_loss']=true"
        ),
        "life_loss_signal": f"info[{LIFE_LOSS_INFO_KEY!r}]",
        "life_loss_penalty": penalty,
        "evaluation_score_definition": "sum raw environment rewards without clipping",
    }


__all__ = [
    "LIFE_LOSS_INFO_KEY",
    "reward_design_metadata",
    "shape_training_reward",
    "validate_life_loss_penalty",
]
