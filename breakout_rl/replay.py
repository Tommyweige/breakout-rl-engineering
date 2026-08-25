"""Fixed-capacity replay storage for preprocessed Breakout transitions."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from math import prod
from typing import Final

import numpy as np


DEFAULT_OBSERVATION_SHAPE: Final[tuple[int, int, int]] = (4, 84, 84)
"""The stacked observation shape emitted by the Breakout environment."""

STATE_DTYPE: Final[np.dtype] = np.dtype(np.uint8)
ACTION_DTYPE: Final[np.dtype] = np.dtype(np.int64)
REWARD_DTYPE: Final[np.dtype] = np.dtype(np.float32)
FLAG_DTYPE: Final[np.dtype] = np.dtype(np.bool_)


@dataclass(frozen=True)
class TransitionBatch:
    """A named mini-batch sampled from a :class:`ReplayBuffer`.

    The arrays are independent copies of the replay storage. This keeps a
    sampled batch safe to normalize or otherwise transform at the model
    boundary without mutating the buffer itself.
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


def _positive_int(value: int, *, name: str) -> int:
    """Validate an integer configuration value that must be positive."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")

    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error

    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return int(parsed)


def _observation_shape(
    observation_shape: tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    """Validate and normalize an observation shape."""

    try:
        parsed_shape = tuple(observation_shape)
    except TypeError as error:
        raise TypeError("observation_shape must be an iterable of integers") from error

    if not parsed_shape:
        raise ValueError("observation_shape must contain at least one dimension")

    normalized: list[int] = []
    for dimension in parsed_shape:
        if isinstance(dimension, (bool, np.bool_)):
            raise TypeError("observation_shape dimensions must be integers")
        try:
            parsed_dimension = operator.index(dimension)
        except TypeError as error:
            raise TypeError(
                "observation_shape dimensions must be integers"
            ) from error
        if parsed_dimension <= 0:
            raise ValueError("observation_shape dimensions must be greater than zero")
        normalized.append(int(parsed_dimension))

    return tuple(normalized)


def _validate_observation(
    observation: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    """Validate the storage-side observation contract."""

    if not isinstance(observation, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if observation.dtype != STATE_DTYPE:
        raise TypeError(
            f"{name} must have dtype uint8; normalize only at the model boundary"
        )
    if tuple(observation.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; "
            f"received {tuple(observation.shape)}"
        )


def _scalar_bool(value: bool | np.bool_ | int | np.integer, *, name: str) -> bool:
    """Convert a scalar episode flag while rejecting array-shaped values."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise TypeError(f"{name} must be a boolean scalar")


def _scalar_reward(value: float | int | np.floating | np.integer) -> np.float32:
    """Convert one scalar reward to the replay storage dtype."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("reward must be a numeric scalar")
    try:
        parsed = np.asarray(value)
        if parsed.ndim != 0:
            raise ValueError("reward must be a numeric scalar")
        return np.float32(parsed.item())
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("reward must be a numeric scalar") from error


def _scalar_action(value: int | np.integer) -> np.int64:
    """Convert one discrete action to the replay storage dtype."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("action must be an integer scalar")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError("action must be an integer scalar") from error

    if not np.iinfo(ACTION_DTYPE).min <= parsed <= np.iinfo(ACTION_DTYPE).max:
        raise ValueError("action is outside the int64 range")
    return np.int64(parsed)


def estimate_replay_memory_bytes(
    capacity: int,
    observation_shape: tuple[int, ...] = DEFAULT_OBSERVATION_SHAPE,
) -> int:
    """Estimate bytes allocated by the baseline replay arrays.

    The estimate intentionally matches the arrays allocated by
    :class:`ReplayBuffer`: two uint8 observation arrays, int64 actions,
    float32 rewards, and separate boolean termination flags.
    """

    parsed_capacity = _positive_int(capacity, name="capacity")
    parsed_shape = _observation_shape(observation_shape)
    observation_bytes = prod(parsed_shape) * STATE_DTYPE.itemsize

    return int(
        (2 * parsed_capacity * observation_bytes)
        + (parsed_capacity * ACTION_DTYPE.itemsize)
        + (parsed_capacity * REWARD_DTYPE.itemsize)
        + (2 * parsed_capacity * FLAG_DTYPE.itemsize)
    )


class ReplayBuffer:
    """Store transitions in a fixed-capacity NumPy ring buffer."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...] = DEFAULT_OBSERVATION_SHAPE,
    ) -> None:
        self.capacity = _positive_int(capacity, name="capacity")
        self.observation_shape = _observation_shape(observation_shape)

        self.states = np.empty(
            (self.capacity, *self.observation_shape), dtype=STATE_DTYPE
        )
        self.next_states = np.empty(
            (self.capacity, *self.observation_shape), dtype=STATE_DTYPE
        )
        self.actions = np.empty(self.capacity, dtype=ACTION_DTYPE)
        self.rewards = np.empty(self.capacity, dtype=REWARD_DTYPE)
        self.terminated = np.empty(self.capacity, dtype=FLAG_DTYPE)
        self.truncated = np.empty(self.capacity, dtype=FLAG_DTYPE)

        self.write_index = 0
        self.size = 0

    @property
    def allocated_bytes(self) -> int:
        """Return the bytes owned by all preallocated replay arrays."""

        return int(
            self.states.nbytes
            + self.next_states.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.terminated.nbytes
            + self.truncated.nbytes
        )

    @property
    def memory_bytes(self) -> int:
        """Alias for :attr:`allocated_bytes` used by inspection code."""

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
        """Copy one transition into the next ring-buffer slot.

        Returns the physical slot used for the transition. The return value
        is useful for inspection and visualization; callers that only need
        storage can ignore it.
        """

        _validate_observation(
            state,
            expected_shape=self.observation_shape,
            name="state",
        )
        _validate_observation(
            next_state,
            expected_shape=self.observation_shape,
            name="next_state",
        )

        slot = self.write_index
        self.states[slot] = state
        self.next_states[slot] = next_state
        self.actions[slot] = _scalar_action(action)
        self.rewards[slot] = _scalar_reward(reward)
        self.terminated[slot] = _scalar_bool(terminated, name="terminated")
        self.truncated[slot] = _scalar_bool(truncated, name="truncated")

        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return slot

    def _validate_sample_request(self, batch_size: int) -> int:
        """Validate a sampling request against the current buffer size."""

        parsed_batch_size = _positive_int(batch_size, name="batch_size")
        if parsed_batch_size > self.size:
            raise ValueError(
                f"batch_size ({parsed_batch_size}) cannot exceed current "
                f"buffer size ({self.size})"
            )
        return parsed_batch_size

    @staticmethod
    def _validate_rng(rng: np.random.Generator) -> None:
        """Require the explicit Generator used for reproducible sampling."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

    def sample_indices(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample physical storage slots uniformly without replacement."""

        parsed_batch_size = self._validate_sample_request(batch_size)
        self._validate_rng(rng)
        return rng.choice(self.size, size=parsed_batch_size, replace=False)

    def _batch_from_indices(self, indices: np.ndarray) -> TransitionBatch:
        """Select a contiguous, independent batch from the ring arrays.

        NumPy advanced indexing already allocates independent arrays. Avoiding
        a second explicit ``.copy()`` keeps the replay-to-model hot path from
        copying every selected field twice while preserving the public copy
        contract.
        """

        return TransitionBatch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
        )

    def sample_with_indices(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[TransitionBatch, np.ndarray]:
        """Sample a batch and return its physical slots for inspection."""

        indices = self.sample_indices(batch_size, rng)
        return self._batch_from_indices(indices), indices

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> TransitionBatch:
        """Sample a uniform mini-batch without replacement."""

        batch, _ = self.sample_with_indices(batch_size, rng)
        return batch

    def chronological_indices(self) -> np.ndarray:
        """Return active physical slots from oldest to newest."""

        if self.size == 0:
            return np.empty(0, dtype=np.int64)
        if self.size < self.capacity:
            return np.arange(self.size, dtype=np.int64)

        return (
            np.arange(self.capacity, dtype=np.int64) + self.write_index
        ) % self.capacity

    @property
    def oldest_index(self) -> int | None:
        """Return the oldest active slot, if the buffer is non-empty."""

        indices = self.chronological_indices()
        return int(indices[0]) if len(indices) else None

    @property
    def newest_index(self) -> int | None:
        """Return the newest active slot, if the buffer is non-empty."""

        indices = self.chronological_indices()
        return int(indices[-1]) if len(indices) else None


__all__ = [
    "ACTION_DTYPE",
    "DEFAULT_OBSERVATION_SHAPE",
    "FLAG_DTYPE",
    "REWARD_DTYPE",
    "ReplayBuffer",
    "STATE_DTYPE",
    "TransitionBatch",
    "estimate_replay_memory_bytes",
]
