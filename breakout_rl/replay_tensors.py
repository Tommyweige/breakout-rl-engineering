"""Model-boundary conversion for sampled replay batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from breakout_rl.replay import TransitionBatch
from breakout_rl.tensors import observation_to_tensor


@dataclass(frozen=True)
class ReplayTensorBatch:
    """The tensor contract consumed by a future DQN training loop."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


def _resolve_device(device: torch.device | str) -> torch.device:
    """Resolve a device and reject an explicitly unavailable CUDA device."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but it is not available in this environment."
        )
    return resolved


def replay_batch_to_tensors(
    batch: TransitionBatch,
    device: torch.device | str,
) -> ReplayTensorBatch:
    """Normalize observations and convert the other fields at the model boundary."""

    if not isinstance(batch, TransitionBatch):
        raise TypeError("batch must be a TransitionBatch")

    resolved_device = _resolve_device(device)
    return ReplayTensorBatch(
        states=observation_to_tensor(batch.states, device=resolved_device),
        actions=torch.as_tensor(
            np.ascontiguousarray(batch.actions),
            dtype=torch.long,
            device=resolved_device,
        ),
        rewards=torch.as_tensor(
            np.ascontiguousarray(batch.rewards),
            dtype=torch.float32,
            device=resolved_device,
        ),
        next_states=observation_to_tensor(
            batch.next_states,
            device=resolved_device,
        ),
        terminated=torch.as_tensor(
            np.ascontiguousarray(batch.terminated),
            dtype=torch.bool,
            device=resolved_device,
        ),
        truncated=torch.as_tensor(
            np.ascontiguousarray(batch.truncated),
            dtype=torch.bool,
            device=resolved_device,
        ),
    )


# A descriptive alias for callers that prefer the transition terminology.
TransitionTensorBatch = ReplayTensorBatch


__all__ = [
    "ReplayTensorBatch",
    "TransitionTensorBatch",
    "replay_batch_to_tensors",
]
