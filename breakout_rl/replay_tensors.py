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


class PreallocatedReplayBatchTransfer:
    """Reuse pinned host and device buffers for repeated replay transfers.

    The direct conversion path allocates new CUDA tensors for every sampled
    batch. Large batches make those allocations and pageable host-to-device
    copies dominate the optimizer update. This transfer owns one reusable
    batch-shaped buffer and keeps the public :class:`ReplayTensorBatch`
    contract unchanged.

    A stream synchronization before reusing the buffers is intentional: the
    trainer uses the returned device batch immediately, so overwriting it
    before the previous update has finished would create a data race. A later
    double-buffered prefetcher can relax this synchronization without changing
    the batch contract.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        observation_shape: tuple[int, ...],
        device: torch.device | str,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not observation_shape or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
            for dimension in observation_shape
        ):
            raise ValueError("observation_shape must contain positive integers")

        self.batch_size = batch_size
        self.observation_shape = tuple(observation_shape)
        self.device = _resolve_device(device)
        self._cuda_buffers = self.device.type == "cuda"
        pin_memory = self._cuda_buffers

        self._host_states = torch.empty(
            (batch_size, *self.observation_shape),
            dtype=torch.uint8,
            pin_memory=pin_memory,
        )
        self._host_next_states = torch.empty(
            (batch_size, *self.observation_shape),
            dtype=torch.uint8,
            pin_memory=pin_memory,
        )
        self._host_actions = torch.empty(
            (batch_size,), dtype=torch.long, pin_memory=pin_memory
        )
        self._host_rewards = torch.empty(
            (batch_size,), dtype=torch.float32, pin_memory=pin_memory
        )
        self._host_terminated = torch.empty(
            (batch_size,), dtype=torch.bool, pin_memory=pin_memory
        )
        self._host_truncated = torch.empty(
            (batch_size,), dtype=torch.bool, pin_memory=pin_memory
        )

        if self._cuda_buffers:
            self._device_batch = ReplayTensorBatch(
                states=torch.empty(
                    (batch_size, *self.observation_shape),
                    dtype=torch.float32,
                    device=self.device,
                ),
                actions=torch.empty(
                    (batch_size,), dtype=torch.long, device=self.device
                ),
                rewards=torch.empty(
                    (batch_size,), dtype=torch.float32, device=self.device
                ),
                next_states=torch.empty(
                    (batch_size, *self.observation_shape),
                    dtype=torch.float32,
                    device=self.device,
                ),
                terminated=torch.empty(
                    (batch_size,), dtype=torch.bool, device=self.device
                ),
                truncated=torch.empty(
                    (batch_size,), dtype=torch.bool, device=self.device
                ),
            )
        else:
            self._device_batch = None

    def _copy_to_host(self, batch: TransitionBatch) -> None:
        if not isinstance(batch, TransitionBatch):
            raise TypeError("batch must be a TransitionBatch")
        expected_shape = (self.batch_size, *self.observation_shape)
        if tuple(batch.states.shape) != expected_shape or tuple(batch.next_states.shape) != expected_shape:
            raise ValueError(
                "sampled observations do not match the configured batch shape"
            )

        self._host_states.copy_(torch.from_numpy(np.ascontiguousarray(batch.states)))
        self._host_next_states.copy_(
            torch.from_numpy(np.ascontiguousarray(batch.next_states))
        )
        self._host_actions.copy_(torch.from_numpy(np.ascontiguousarray(batch.actions)))
        self._host_rewards.copy_(torch.from_numpy(np.ascontiguousarray(batch.rewards)))
        self._host_terminated.copy_(
            torch.from_numpy(np.ascontiguousarray(batch.terminated))
        )
        self._host_truncated.copy_(
            torch.from_numpy(np.ascontiguousarray(batch.truncated))
        )

    def transfer(self, batch: TransitionBatch) -> ReplayTensorBatch:
        """Copy one sampled batch into the reusable model-input buffers."""

        self._copy_to_host(batch)
        if not self._cuda_buffers:
            return replay_batch_to_tensors(batch, device=self.device)

        torch.cuda.current_stream(self.device).synchronize()
        assert self._device_batch is not None
        self._device_batch.states.copy_(
            self._host_states, non_blocking=True
        ).div_(255.0)
        self._device_batch.next_states.copy_(
            self._host_next_states, non_blocking=True
        ).div_(255.0)
        self._device_batch.actions.copy_(self._host_actions, non_blocking=True)
        self._device_batch.rewards.copy_(self._host_rewards, non_blocking=True)
        self._device_batch.terminated.copy_(
            self._host_terminated, non_blocking=True
        )
        self._device_batch.truncated.copy_(self._host_truncated, non_blocking=True)
        return self._device_batch


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
    "PreallocatedReplayBatchTransfer",
    "ReplayTensorBatch",
    "TransitionTensorBatch",
    "replay_batch_to_tensors",
]
