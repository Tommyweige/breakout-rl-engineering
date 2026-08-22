"""Tests for fixed-capacity replay storage and model-boundary conversion."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from breakout_rl.replay import ReplayBuffer, estimate_replay_memory_bytes
from breakout_rl.replay_tensors import replay_batch_to_tensors


class ReplayBufferTests(unittest.TestCase):
    SMALL_SHAPE = (2, 3)

    @staticmethod
    def transition(index: int, shape: tuple[int, ...] = SMALL_SHAPE) -> tuple[object, ...]:
        state = np.full(shape, index % 256, dtype=np.uint8)
        next_state = np.full(shape, (index + 1) % 256, dtype=np.uint8)
        return (
            state,
            index,
            index + 0.5,
            next_state,
            index % 2 == 0,
            index % 3 == 0,
        )

    def add_indices(self, buffer: ReplayBuffer, count: int) -> None:
        for index in range(count):
            buffer.add(*self.transition(index))

    def test_empty_buffer_has_zero_length_and_rejects_sampling(self) -> None:
        buffer = ReplayBuffer(capacity=4, observation_shape=self.SMALL_SHAPE)

        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.write_index, 0)
        with self.assertRaisesRegex(ValueError, "current buffer size"):
            buffer.sample(1, np.random.default_rng(42))

    def test_add_keeps_length_dtypes_and_copies_input_observations(self) -> None:
        buffer = ReplayBuffer(capacity=4, observation_shape=self.SMALL_SHAPE)
        state, action, reward, next_state, terminated, truncated = self.transition(7)

        buffer.add(state, action, reward, next_state, terminated, truncated)
        state[...] = 0
        next_state[...] = 0

        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.states.dtype, np.uint8)
        self.assertEqual(buffer.next_states.dtype, np.uint8)
        self.assertEqual(buffer.actions.dtype, np.int64)
        self.assertEqual(buffer.rewards.dtype, np.float32)
        self.assertEqual(buffer.terminated.dtype, np.bool_)
        self.assertEqual(buffer.truncated.dtype, np.bool_)
        self.assertEqual(int(buffer.states[0, 0, 0]), 7)
        self.assertEqual(int(buffer.next_states[0, 0, 0]), 8)

    def test_capacity_overflow_wraps_and_overwrites_oldest_entries(self) -> None:
        buffer = ReplayBuffer(capacity=4, observation_shape=self.SMALL_SHAPE)
        self.add_indices(buffer, 6)

        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.write_index, 2)
        np.testing.assert_array_equal(
            buffer.chronological_indices(),
            np.array([2, 3, 0, 1]),
        )
        np.testing.assert_array_equal(
            buffer.states[buffer.chronological_indices(), 0, 0],
            np.array([2, 3, 4, 5], dtype=np.uint8),
        )
        self.assertEqual(buffer.oldest_index, 2)
        self.assertEqual(buffer.newest_index, 1)

    def test_sample_returns_named_batch_with_expected_shapes_and_dtypes(self) -> None:
        buffer = ReplayBuffer(capacity=8, observation_shape=self.SMALL_SHAPE)
        self.add_indices(buffer, 5)

        batch, indices = buffer.sample_with_indices(3, np.random.default_rng(42))

        self.assertEqual(batch.states.shape, (3, *self.SMALL_SHAPE))
        self.assertEqual(batch.next_states.shape, (3, *self.SMALL_SHAPE))
        self.assertEqual(batch.actions.shape, (3,))
        self.assertEqual(batch.rewards.shape, (3,))
        self.assertEqual(batch.terminated.shape, (3,))
        self.assertEqual(batch.truncated.shape, (3,))
        self.assertEqual(batch.states.dtype, np.uint8)
        self.assertEqual(batch.actions.dtype, np.int64)
        self.assertEqual(batch.rewards.dtype, np.float32)
        self.assertEqual(batch.terminated.dtype, np.bool_)
        self.assertEqual(batch.truncated.dtype, np.bool_)
        self.assertEqual(len(np.unique(indices)), 3)

    def test_sampling_is_reproducible_with_the_same_generator_seed(self) -> None:
        buffer = ReplayBuffer(capacity=8, observation_shape=self.SMALL_SHAPE)
        self.add_indices(buffer, 8)

        first_batch, first_indices = buffer.sample_with_indices(
            4,
            np.random.default_rng(42),
        )
        second_batch, second_indices = buffer.sample_with_indices(
            4,
            np.random.default_rng(42),
        )

        np.testing.assert_array_equal(first_indices, second_indices)
        np.testing.assert_array_equal(first_batch.states, second_batch.states)
        np.testing.assert_array_equal(first_batch.actions, second_batch.actions)

    def test_terminated_and_truncated_are_stored_separately(self) -> None:
        buffer = ReplayBuffer(capacity=2, observation_shape=self.SMALL_SHAPE)
        state, action, reward, next_state, _, _ = self.transition(1)
        buffer.add(state, action, reward, next_state, True, False)
        buffer.add(state, action, reward, next_state, False, True)

        np.testing.assert_array_equal(
            buffer.terminated[:2],
            np.array([True, False], dtype=np.bool_),
        )
        np.testing.assert_array_equal(
            buffer.truncated[:2],
            np.array([False, True], dtype=np.bool_),
        )

    def test_sampled_batch_is_a_copy_of_storage(self) -> None:
        buffer = ReplayBuffer(capacity=2, observation_shape=self.SMALL_SHAPE)
        self.add_indices(buffer, 1)

        batch = buffer.sample(1, np.random.default_rng(42))
        batch.states[...] = 0
        batch.actions[...] = 99

        self.assertEqual(int(buffer.states[0, 0, 0]), 0)
        self.assertEqual(int(buffer.actions[0]), 0)

    def test_batch_larger_than_buffer_and_zero_batch_are_rejected(self) -> None:
        buffer = ReplayBuffer(capacity=4, observation_shape=self.SMALL_SHAPE)
        self.add_indices(buffer, 2)
        rng = np.random.default_rng(42)

        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            buffer.sample(3, rng)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            buffer.sample(0, rng)

    def test_memory_estimator_matches_actual_numpy_allocations(self) -> None:
        buffer = ReplayBuffer(capacity=7, observation_shape=self.SMALL_SHAPE)
        actual = (
            buffer.states.nbytes
            + buffer.next_states.nbytes
            + buffer.actions.nbytes
            + buffer.rewards.nbytes
            + buffer.terminated.nbytes
            + buffer.truncated.nbytes
        )

        self.assertEqual(buffer.allocated_bytes, actual)
        self.assertEqual(
            estimate_replay_memory_bytes(7, self.SMALL_SHAPE),
            actual,
        )

    def test_replay_batch_conversion_normalizes_only_at_model_boundary(self) -> None:
        buffer = ReplayBuffer(capacity=2)
        zero = np.zeros((4, 84, 84), dtype=np.uint8)
        full = np.full((4, 84, 84), 255, dtype=np.uint8)
        buffer.add(zero, 2, 1.5, full, True, False)
        buffer.add(full, 3, -1.0, zero, False, True)

        batch = buffer.sample(2, np.random.default_rng(42))
        tensors = replay_batch_to_tensors(batch, device="cpu")

        self.assertEqual(tensors.states.shape, (2, 4, 84, 84))
        self.assertEqual(tensors.states.dtype, torch.float32)
        self.assertGreaterEqual(float(tensors.states.min()), 0.0)
        self.assertLessEqual(float(tensors.states.max()), 1.0)
        self.assertEqual(tensors.actions.dtype, torch.long)
        self.assertEqual(tensors.rewards.dtype, torch.float32)
        self.assertEqual(tensors.terminated.dtype, torch.bool)
        self.assertEqual(tensors.truncated.dtype, torch.bool)
        self.assertTrue(
            torch.all(
                (tensors.states == 0.0) | (tensors.states == 1.0)
            )
        )


if __name__ == "__main__":
    unittest.main()
