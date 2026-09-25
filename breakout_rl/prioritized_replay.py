"""Math and result types for proportional prioritized replay."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import torch

from breakout_rl.replay_tensors import ReplayTensorBatch


def _finite_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a finite number in [0, 1]") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return parsed


def beta_for_transition(
    transitions: int,
    *,
    beta_start: float = 0.4,
    beta_end: float = 1.0,
    anneal_transitions: int = 500_000,
) -> float:
    """Linearly anneal beta by accepted environment transitions."""

    if isinstance(transitions, bool):
        raise TypeError("transitions must be a non-negative integer")
    try:
        step = operator.index(transitions)
    except TypeError as error:
        raise TypeError("transitions must be a non-negative integer") from error
    if step < 0:
        raise ValueError("transitions must be non-negative")
    if isinstance(anneal_transitions, bool):
        raise TypeError("anneal_transitions must be a positive integer")
    try:
        horizon = operator.index(anneal_transitions)
    except TypeError as error:
        raise TypeError("anneal_transitions must be a positive integer") from error
    if horizon < 1:
        raise ValueError("anneal_transitions must be greater than zero")
    start = _finite_probability(beta_start, name="beta_start")
    end = _finite_probability(beta_end, name="beta_end")
    fraction = min(step / horizon, 1.0)
    return start + (end - start) * fraction


def assert_tensor_condition(condition: torch.Tensor, *, message: str) -> None:
    if condition.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def importance_sampling_weights(
    sample_probabilities: torch.Tensor,
    *,
    population_size: int,
    beta: float,
) -> torch.Tensor:
    """Compute batch-max-normalized PER weights on the probabilities' device."""

    if not isinstance(sample_probabilities, torch.Tensor):
        raise TypeError("sample_probabilities must be a torch.Tensor")
    if sample_probabilities.ndim != 1 or sample_probabilities.numel() < 1:
        raise ValueError("sample_probabilities must be a non-empty vector")
    if not sample_probabilities.is_floating_point():
        raise TypeError("sample_probabilities must be floating point")
    if isinstance(population_size, bool):
        raise TypeError("population_size must be a positive integer")
    try:
        population = operator.index(population_size)
    except TypeError as error:
        raise TypeError("population_size must be a positive integer") from error
    if population < 1:
        raise ValueError("population_size must be greater than zero")
    parsed_beta = _finite_probability(beta, name="beta")

    valid = torch.all(torch.isfinite(sample_probabilities) & (sample_probabilities > 0))
    assert_tensor_condition(valid, message="sample_probabilities must be finite and positive")
    weights = torch.pow(sample_probabilities * float(population), -parsed_beta)
    return weights / weights.max()


@dataclass(frozen=True)
class PrioritizedReplaySample:
    """A prioritized batch and the sampling data needed by its trainer."""

    batch: ReplayTensorBatch
    indices: torch.Tensor
    probabilities: torch.Tensor
    importance_weights: torch.Tensor
    effective_sample_size: torch.Tensor


__all__ = [
    "PrioritizedReplaySample",
    "assert_tensor_condition",
    "beta_for_transition",
    "importance_sampling_weights",
]
