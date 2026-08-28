"""Exploration policies and epsilon schedules for DQN action selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch

ActionSource = Literal["random", "greedy"]

__all__ = [
    "ActionSource",
    "LinearEpsilonSchedule",
    "select_epsilon_greedy_action",
    "select_epsilon_greedy_actions",
]


def _validate_probability(name: str, value: float) -> None:
    """Validate a probability-like value without accepting NaN or infinity."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 1 inclusive")


@dataclass(frozen=True)
class LinearEpsilonSchedule:
    """Linearly move epsilon from ``start`` to ``end`` over fixed steps.

    Epsilon is the probability of taking a random action. Values at or before
    step zero use ``start``; values at or after ``decay_steps`` use ``end``.
    """

    start: float
    end: float
    decay_steps: int

    def __post_init__(self) -> None:
        _validate_probability("start", self.start)
        _validate_probability("end", self.end)
        if (
            isinstance(self.decay_steps, bool)
            or not isinstance(self.decay_steps, Integral)
            or self.decay_steps <= 0
        ):
            raise ValueError("decay_steps must be a positive integer")

    def value(self, step: int) -> float:
        """Return the scheduled epsilon for one environment step."""

        if isinstance(step, bool) or not isinstance(step, Integral):
            raise TypeError("step must be an integer")

        if step <= 0:
            return float(self.start)
        if step >= self.decay_steps:
            return float(self.end)

        progress = float(step) / float(self.decay_steps)
        return float(self.start) + (float(self.end) - float(self.start)) * progress


def select_epsilon_greedy_action(
    q_values: torch.Tensor,
    epsilon: float,
    *,
    action_space_n: int,
    rng: np.random.Generator,
) -> tuple[int, ActionSource]:
    """Select an action and report whether it was random or greedy.

    The greedy branch uses deterministic ``torch.argmax`` behavior: when
    multiple actions share the maximum Q-value, the first index wins. The
    caller should compute ``q_values`` under ``torch.no_grad()`` when it comes
    from a model; the helper also keeps the argmax operation out of autograd.
    """

    import torch

    _validate_probability("epsilon", epsilon)
    if (
        isinstance(action_space_n, bool)
        or not isinstance(action_space_n, Integral)
        or action_space_n < 1
    ):
        raise ValueError("action_space_n must be a positive integer")
    if not isinstance(q_values, torch.Tensor):
        raise TypeError("q_values must be a torch.Tensor")
    if q_values.ndim != 1:
        raise ValueError("q_values must be a one-dimensional tensor")
    if q_values.numel() != action_space_n:
        raise ValueError("q_values length must equal action_space_n")
    if not torch.isfinite(q_values).all().item():
        raise ValueError("q_values must contain only finite values")

    if rng.random() < float(epsilon):
        return int(rng.integers(0, action_space_n)), "random"

    with torch.no_grad():
        greedy_action = int(torch.argmax(q_values).item())
    return greedy_action, "greedy"


def select_epsilon_greedy_actions(
    q_values: torch.Tensor,
    epsilon: float | Sequence[float] | np.ndarray,
    *,
    action_space_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one independently explored action for every Q-value row.

    ``epsilon`` may be one scalar shared by the batch or one value per row.
    The model is evaluated once by the caller; this helper only applies the
    per-environment random-versus-greedy decision.
    """

    import torch

    if not isinstance(q_values, torch.Tensor):
        raise TypeError("q_values must be a torch.Tensor")
    if q_values.ndim != 2:
        raise ValueError("q_values must have shape (B, action_count)")
    batch_size = int(q_values.shape[0])
    if int(q_values.shape[1]) != action_space_n:
        raise ValueError("q_values width must equal action_space_n")
    if batch_size < 1:
        raise ValueError("q_values must contain at least one environment")
    if not torch.isfinite(q_values).all().item():
        raise ValueError("q_values must contain only finite values")
    if (
        isinstance(action_space_n, bool)
        or not isinstance(action_space_n, Integral)
        or action_space_n < 1
    ):
        raise ValueError("action_space_n must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    epsilon_array = np.asarray(epsilon, dtype=np.float64)
    if epsilon_array.ndim == 0:
        epsilon_array = np.full(batch_size, float(epsilon_array), dtype=np.float64)
    elif epsilon_array.ndim == 1 and int(epsilon_array.shape[0]) == batch_size:
        epsilon_array = np.ascontiguousarray(epsilon_array, dtype=np.float64)
    else:
        raise ValueError("epsilon must be a scalar or have shape (B,)")
    for value in epsilon_array:
        _validate_probability("epsilon", float(value))

    with torch.no_grad():
        greedy_actions = torch.argmax(q_values, dim=1).detach().cpu().numpy()
    random_mask = np.asarray(rng.random(batch_size) < epsilon_array, dtype=bool)
    actions = np.asarray(greedy_actions, dtype=np.int64)
    if random_mask.any():
        actions[random_mask] = rng.integers(
            0,
            int(action_space_n),
            size=int(random_mask.sum()),
            dtype=np.int64,
        )
    sources = np.where(random_mask, "random", "greedy").astype("<U7")
    return actions, sources
