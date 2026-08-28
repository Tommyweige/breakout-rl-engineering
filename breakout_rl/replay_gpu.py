"""GPU-resident replay storage with the same transition contract as NumPy replay."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from math import prod
from typing import Sequence

import numpy as np
import torch

from breakout_rl.replay import (
    DEFAULT_OBSERVATION_SHAPE,
    _validate_transition_batch,
)
from breakout_rl.replay_tensors import ReplayTensorBatch


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return int(parsed)


def _observation_shape(observation_shape: Sequence[int]) -> tuple[int, ...]:
    try:
        parsed = tuple(observation_shape)
    except TypeError as error:
        raise TypeError("observation_shape must be an iterable of integers") from error
    if not parsed:
        raise ValueError("observation_shape must not be empty")
    normalized: list[int] = []
    for dimension in parsed:
        if isinstance(dimension, bool):
            raise TypeError("observation_shape dimensions must be integers")
        try:
            value = operator.index(dimension)
        except TypeError as error:
            raise TypeError(
                "observation_shape dimensions must be integers"
            ) from error
        if value <= 0:
            raise ValueError("observation_shape dimensions must be greater than zero")
        normalized.append(int(value))
    return tuple(normalized)


def _validate_observation(
    observation: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if not isinstance(observation, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if observation.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} must have dtype uint8")
    if tuple(observation.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; received {tuple(observation.shape)}"
        )
    return np.ascontiguousarray(observation)


def _scalar_action(value: int | np.integer) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("action must be an integer scalar")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError("action must be an integer scalar") from error


def _scalar_reward(value: float | int | np.floating | np.integer) -> np.float32:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("reward must be a numeric scalar")
    parsed = np.asarray(value)
    if parsed.ndim != 0:
        raise TypeError("reward must be a numeric scalar")
    try:
        return np.float32(parsed.item())
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("reward must be a numeric scalar") from error


def _scalar_flag(value: bool | np.bool_ | int | np.integer, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise TypeError(f"{name} must be a boolean scalar")


def _resolve_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPUReplayBuffer requested CUDA, but CUDA is not available."
        )
    return resolved


@dataclass
class _BatchBuffers:
    raw_states: torch.Tensor
    raw_next_states: torch.Tensor
    batch: ReplayTensorBatch


class GPUReplayBuffer:
    """Store uint8 transitions on one device and gather normalized batches there.

    The active physical slots intentionally mirror :class:`ReplayBuffer`: while
    the ring is not full, slots ``[0, size)`` are active; once full, every slot
    is active and ``write_index`` identifies the next slot to overwrite.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...] = DEFAULT_OBSERVATION_SHAPE,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        self.capacity = _positive_int(capacity, name="capacity")
        self.observation_shape = _observation_shape(observation_shape)
        self.device = _resolve_device(device)

        self.states = torch.empty(
            (self.capacity, *self.observation_shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(
            self.capacity,
            dtype=torch.long,
            device=self.device,
        )
        self.rewards = torch.empty(
            self.capacity,
            dtype=torch.float32,
            device=self.device,
        )
        self.terminated = torch.empty(
            self.capacity,
            dtype=torch.bool,
            device=self.device,
        )
        self.truncated = torch.empty(
            self.capacity,
            dtype=torch.bool,
            device=self.device,
        )

        self.write_index = 0
        self.size = 0
        self._batch_buffers: dict[int, _BatchBuffers] = {}

    @property
    def bytes_per_transition(self) -> int:
        observation_bytes = prod(self.observation_shape)
        return int((2 * observation_bytes) + 8 + 4 + 1 + 1)

    @property
    def allocated_bytes(self) -> int:
        return int(
            self.states.numel() * self.states.element_size()
            + self.next_states.numel() * self.next_states.element_size()
            + self.actions.numel() * self.actions.element_size()
            + self.rewards.numel() * self.rewards.element_size()
            + self.terminated.numel() * self.terminated.element_size()
            + self.truncated.numel() * self.truncated.element_size()
        )

    @property
    def memory_bytes(self) -> int:
        return self.allocated_bytes

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int | np.integer,
        reward: float | int | np.floating | np.integer,
        next_state: np.ndarray,
        terminated: bool | np.bool_ | int | np.integer,
        truncated: bool | np.bool_ | int | np.integer,
    ) -> int:
        """Copy one transition into the next physical ring slot."""

        state_array = _validate_observation(
            state,
            expected_shape=self.observation_shape,
            name="state",
        )
        next_state_array = _validate_observation(
            next_state,
            expected_shape=self.observation_shape,
            name="next_state",
        )
        slot = self.write_index
        self.states[slot].copy_(torch.from_numpy(state_array))
        self.next_states[slot].copy_(torch.from_numpy(next_state_array))
        self.actions[slot] = _scalar_action(action)
        self.rewards[slot] = float(_scalar_reward(reward))
        self.terminated[slot] = _scalar_flag(terminated, name="terminated")
        self.truncated[slot] = _scalar_flag(truncated, name="truncated")

        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return slot

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> np.ndarray:
        """Copy a batch of transitions into consecutive device-side slots."""

        batch = _validate_transition_batch(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            expected_shape=self.observation_shape,
        )
        batch_size = int(batch.states.shape[0])
        initial_write_index = self.write_index
        slots = (
            initial_write_index + np.arange(batch_size, dtype=np.int64)
        ) % self.capacity
        source_start = max(0, batch_size - self.capacity)
        write_start = int(slots[source_start])
        write_count = batch_size - source_start

        def copy_ring(destination: torch.Tensor, source: np.ndarray) -> None:
            first_count = min(write_count, self.capacity - write_start)
            destination[write_start : write_start + first_count].copy_(
                torch.from_numpy(source[source_start : source_start + first_count])
            )
            remaining = write_count - first_count
            if remaining:
                destination[:remaining].copy_(
                    torch.from_numpy(
                        source[
                            source_start + first_count : source_start + first_count + remaining
                        ]
                    )
                )

        copy_ring(self.states, batch.states)
        copy_ring(self.actions, batch.actions)
        copy_ring(self.rewards, batch.rewards)
        copy_ring(self.next_states, batch.next_states)
        copy_ring(self.terminated, batch.terminated)
        copy_ring(self.truncated, batch.truncated)

        self.write_index = (initial_write_index + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)
        return slots

    def _validate_batch_size(self, batch_size: int) -> int:
        parsed = _positive_int(batch_size, name="batch_size")
        if parsed > self.size:
            raise ValueError(
                f"batch_size ({parsed}) cannot exceed current buffer size ({self.size})"
            )
        return parsed

    def sample_indices(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample active physical slots uniformly without replacement."""

        parsed = self._validate_batch_size(batch_size)
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None")
        kwargs = {"generator": generator} if generator is not None else {}
        return torch.randperm(
            self.size,
            device=self.device,
            **kwargs,
        )[:parsed]

    def _validate_indices(
        self,
        indices: torch.Tensor | np.ndarray | Sequence[int],
    ) -> torch.Tensor:
        if isinstance(indices, torch.Tensor):
            physical = indices.to(device=self.device, dtype=torch.long)
        else:
            physical = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=self.device,
            )
        if physical.ndim != 1 or physical.numel() < 1:
            raise ValueError("indices must be a non-empty one-dimensional sequence")
        if self.size < 1:
            raise ValueError("cannot gather from an empty replay buffer")
        if bool(torch.any(physical < 0).item()) or bool(
            torch.any(physical >= self.size).item()
        ):
            raise ValueError("indices must refer to active replay slots")
        return physical

    def _buffers_for(self, batch_size: int) -> _BatchBuffers:
        buffers = self._batch_buffers.get(batch_size)
        if buffers is not None:
            return buffers

        raw_states = torch.empty(
            (batch_size, *self.observation_shape),
            dtype=torch.uint8,
            device=self.device,
        )
        raw_next_states = torch.empty_like(raw_states)
        buffers = _BatchBuffers(
            raw_states=raw_states,
            raw_next_states=raw_next_states,
            batch=ReplayTensorBatch(
                states=torch.empty(
                    (batch_size, *self.observation_shape),
                    dtype=torch.float32,
                    device=self.device,
                ),
                actions=torch.empty(
                    batch_size,
                    dtype=torch.long,
                    device=self.device,
                ),
                rewards=torch.empty(
                    batch_size,
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_states=torch.empty(
                    (batch_size, *self.observation_shape),
                    dtype=torch.float32,
                    device=self.device,
                ),
                terminated=torch.empty(
                    batch_size,
                    dtype=torch.bool,
                    device=self.device,
                ),
                truncated=torch.empty(
                    batch_size,
                    dtype=torch.bool,
                    device=self.device,
                ),
            ),
        )
        self._batch_buffers[batch_size] = buffers
        return buffers

    def _gather_validated(self, physical: torch.Tensor) -> ReplayTensorBatch:
        buffers = self._buffers_for(int(physical.shape[0]))
        torch.index_select(self.states, 0, physical, out=buffers.raw_states)
        torch.index_select(
            self.next_states,
            0,
            physical,
            out=buffers.raw_next_states,
        )
        torch.index_select(self.actions, 0, physical, out=buffers.batch.actions)
        torch.index_select(self.rewards, 0, physical, out=buffers.batch.rewards)
        torch.index_select(
            self.terminated,
            0,
            physical,
            out=buffers.batch.terminated,
        )
        torch.index_select(
            self.truncated,
            0,
            physical,
            out=buffers.batch.truncated,
        )
        buffers.batch.states.copy_(buffers.raw_states).div_(255.0)
        buffers.batch.next_states.copy_(buffers.raw_next_states).div_(255.0)
        return buffers.batch

    def gather(
        self,
        indices: torch.Tensor | np.ndarray | Sequence[int],
    ) -> ReplayTensorBatch:
        """Gather uint8 transitions and normalize them into reusable GPU tensors."""

        return self._gather_validated(self._validate_indices(indices))

    def sample_with_indices(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[ReplayTensorBatch, torch.Tensor]:
        indices = self.sample_indices(batch_size, generator=generator)
        # randperm over the active range already proves these indices are
        # valid; avoid a device-to-host validation sync on every trainer update.
        return self._gather_validated(indices), indices

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> ReplayTensorBatch:
        """Sample a normalized replay batch without replacement."""

        batch, _ = self.sample_with_indices(batch_size, generator=generator)
        return batch


__all__ = ["GPUReplayBuffer"]
