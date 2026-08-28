"""Target-network and vanilla DQN Bellman-target utilities."""

from __future__ import annotations

import math
from numbers import Real
import operator
from typing import Final

import torch
from torch import nn


_MIN_GAMMA: Final[float] = 0.0
_MAX_GAMMA: Final[float] = 1.0


def _validate_gamma(gamma: float) -> float:
    """Validate and normalize a discount factor."""

    if isinstance(gamma, bool) or not isinstance(gamma, Real):
        raise ValueError("gamma must be a real number in the range [0, 1]")

    parsed = float(gamma)
    if not math.isfinite(parsed) or not _MIN_GAMMA <= parsed <= _MAX_GAMMA:
        raise ValueError("gamma must be a real number in the range [0, 1]")
    return parsed


def _validate_integer(value: int, *, name: str, minimum: int) -> int:
    """Validate an integer configuration value."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(parsed)


def hard_update(target_network: nn.Module, online_network: nn.Module) -> None:
    """Copy online parameters into a separate, inference-only target network.

    ``load_state_dict`` copies parameter and buffer values; it does not make
    the two modules aliases. Target parameters are also frozen so an optimizer
    cannot accidentally update them between synchronization points.
    """

    if not isinstance(target_network, nn.Module):
        raise TypeError("target_network must be a torch.nn.Module")
    if not isinstance(online_network, nn.Module):
        raise TypeError("online_network must be a torch.nn.Module")
    if target_network is online_network:
        raise ValueError("target_network and online_network must be distinct objects")

    with torch.no_grad():
        target_network.load_state_dict(online_network.state_dict())

    for parameter in target_network.parameters():
        parameter.requires_grad_(False)


def _validate_batch_vector(tensor: torch.Tensor, *, name: str) -> int:
    """Validate a one-dimensional batch field and return its batch size."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape (B,)")
    return int(tensor.shape[0])


def _terminated_mask(terminated: torch.Tensor) -> torch.Tensor:
    """Return a boolean termination mask while accepting 0/1 numeric flags."""

    if terminated.dtype == torch.bool:
        return terminated
    if not (terminated.is_floating_point() or terminated.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }):
        raise TypeError("terminated must be a boolean or numeric 0/1 tensor")
    if terminated.numel() and not torch.isfinite(terminated).all().item():
        raise ValueError("terminated must contain only 0/1 values")
    if terminated.numel() and not torch.all((terminated == 0) | (terminated == 1)).item():
        raise ValueError("terminated must contain only 0/1 values")
    return terminated.to(dtype=torch.bool)


def _validate_target_batch(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
) -> tuple[float, int, torch.Tensor]:
    """Validate shared Bellman-target inputs and return normalized values."""

    parsed_gamma = _validate_gamma(gamma)
    batch_size = _validate_batch_vector(rewards, name="rewards")
    terminated_batch_size = _validate_batch_vector(terminated, name="terminated")

    if not isinstance(next_states, torch.Tensor):
        raise TypeError("next_states must be a torch.Tensor")
    if next_states.ndim < 1:
        raise ValueError("next_states must have a batch dimension")
    if int(next_states.shape[0]) != batch_size:
        raise ValueError("rewards, next_states, and terminated must share batch size")
    if terminated_batch_size != batch_size:
        raise ValueError("rewards, next_states, and terminated must share batch size")
    if rewards.device != next_states.device or terminated.device != next_states.device:
        raise ValueError("rewards, next_states, and terminated must share a device")

    return parsed_gamma, batch_size, _terminated_mask(terminated)


def _validate_q_values(
    q_values: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
    action_count: int | None = None,
) -> None:
    """Validate a network output used by a Bellman target."""

    if not isinstance(q_values, torch.Tensor):
        raise TypeError(f"{name} must return a torch.Tensor")
    if q_values.ndim != 2 or int(q_values.shape[0]) != batch_size:
        raise ValueError(f"{name} output must have shape (B, action_count)")
    if int(q_values.shape[1]) < 1:
        raise ValueError(f"{name} must return at least one action value")
    if action_count is not None and int(q_values.shape[1]) != action_count:
        raise ValueError("online_network and target_network must return the same action count")
    if not q_values.is_floating_point():
        raise TypeError(f"{name} output must be a floating-point tensor")
    if q_values.device != device:
        raise ValueError(f"{name} output must share the input device")


def compute_dqn_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    terminated: torch.Tensor,
    target_network: nn.Module,
    gamma: float,
) -> torch.Tensor:
    """Compute vanilla DQN Bellman targets with a frozen target-network pass.

    The helper uses ``target_network`` only to estimate the best next-state
    value. ``terminated`` controls bootstrapping; ``truncated`` is deliberately
    not inferred here because an externally truncated transition is not
    automatically a true terminal state.
    """

    parsed_gamma, batch_size, done_mask = _validate_target_batch(
        rewards,
        next_states,
        terminated,
        gamma,
    )
    if not isinstance(target_network, nn.Module):
        raise TypeError("target_network must be a torch.nn.Module")

    with torch.no_grad():
        next_q_values = target_network(next_states)

    _validate_q_values(
        next_q_values,
        batch_size=batch_size,
        device=rewards.device,
        name="target_network",
    )

    next_q_max = next_q_values.max(dim=1).values
    reward_values = rewards.to(dtype=next_q_values.dtype)
    bootstrap_mask = (~done_mask).to(dtype=next_q_values.dtype)
    return reward_values + parsed_gamma * bootstrap_mask * next_q_max


def compute_double_dqn_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    terminated: torch.Tensor,
    online_network: nn.Module,
    target_network: nn.Module,
    gamma: float,
) -> torch.Tensor:
    """Compute Double DQN targets with separate selection and evaluation.

    The online network chooses the best next action, while the target network
    evaluates only that action. Both next-state passes are detached because
    they define the Bellman target rather than the current-state prediction.
    ``terminated`` controls bootstrapping; truncated transitions remain
    eligible unless the caller has explicitly represented them as terminated.
    """

    parsed_gamma, batch_size, done_mask = _validate_target_batch(
        rewards,
        next_states,
        terminated,
        gamma,
    )
    if not isinstance(online_network, nn.Module):
        raise TypeError("online_network must be a torch.nn.Module")
    if not isinstance(target_network, nn.Module):
        raise TypeError("target_network must be a torch.nn.Module")
    if online_network is target_network:
        raise ValueError("online_network and target_network must be distinct objects")

    with torch.no_grad():
        online_next_q = online_network(next_states)
        _validate_q_values(
            online_next_q,
            batch_size=batch_size,
            device=rewards.device,
            name="online_network",
        )
        next_actions = online_next_q.argmax(dim=1)

        target_next_q = target_network(next_states)
        _validate_q_values(
            target_next_q,
            batch_size=batch_size,
            device=rewards.device,
            name="target_network",
            action_count=int(online_next_q.shape[1]),
        )
        next_values = target_next_q.gather(
            1,
            next_actions[:, None],
        ).squeeze(1)

    reward_values = rewards.to(dtype=next_values.dtype)
    bootstrap_mask = (~done_mask).to(dtype=next_values.dtype)
    return reward_values + parsed_gamma * bootstrap_mask * next_values


def should_update_target(step: int, interval: int) -> bool:
    """Return whether a hard target sync is due at a non-negative step."""

    parsed_step = _validate_integer(step, name="step", minimum=0)
    parsed_interval = _validate_integer(interval, name="interval", minimum=1)
    return parsed_step % parsed_interval == 0


__all__ = [
    "compute_double_dqn_targets",
    "compute_dqn_targets",
    "hard_update",
    "should_update_target",
]
