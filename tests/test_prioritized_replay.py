"""Behavioral tests for proportional prioritized GPU replay."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from breakout_rl.prioritized_replay import (
    beta_for_transition,
    importance_sampling_weights,
)
from breakout_rl.replay_gpu import GPUReplayBuffer


class PrioritizedReplayMathTests(unittest.TestCase):
    def test_beta_starts_anneals_and_clamps_at_horizon(self) -> None:
        self.assertEqual(beta_for_transition(0), 0.4)
        self.assertAlmostEqual(beta_for_transition(250_000), 0.7)
        self.assertEqual(beta_for_transition(500_000), 1.0)
        self.assertEqual(beta_for_transition(900_000), 1.0)

    def test_importance_weights_match_hand_computable_values(self) -> None:
        probabilities = torch.tensor([0.25, 0.5], dtype=torch.float32)

        weights = importance_sampling_weights(
            probabilities,
            population_size=4,
            beta=1.0,
        )

        torch.testing.assert_close(weights, torch.tensor([1.0, 0.5]))

    def test_importance_weights_normalize_largest_weight_to_one(self) -> None:
        probabilities = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float32)

        weights = importance_sampling_weights(
            probabilities,
            population_size=10,
            beta=0.4,
        )

        self.assertEqual(float(weights.max()), 1.0)
        self.assertTrue(torch.all(weights > 0.0).item())
        self.assertTrue(torch.all(weights <= 1.0).item())


class PrioritizedGPUReplayBufferTests(unittest.TestCase):
    SHAPE = (2, 3, 3)

    @staticmethod
    def transition(index: int) -> tuple[np.ndarray, int, float, np.ndarray, bool, bool]:
        state = np.full(PrioritizedGPUReplayBufferTests.SHAPE, index, dtype=np.uint8)
        next_state = np.full(
            PrioritizedGPUReplayBufferTests.SHAPE,
            index + 1,
            dtype=np.uint8,
        )
        return state, index % 4, float(index), next_state, False, False

    def make_buffer(self, capacity: int = 8) -> GPUReplayBuffer:
        return GPUReplayBuffer(
            capacity,
            observation_shape=self.SHAPE,
            device="cpu",
            prioritized=True,
        )

    def fill(self, replay: GPUReplayBuffer, size: int) -> None:
        for index in range(size):
            replay.add(*self.transition(index))

    def test_first_transition_gets_a_finite_positive_priority(self) -> None:
        replay = self.make_buffer(capacity=3)

        replay.add(*self.transition(0))

        self.assertEqual(float(replay.priorities[0]), 1.0)
        self.assertTrue(torch.isfinite(replay.priorities[0]).item())
        self.assertGreater(float(replay.priorities[0]), 0.0)

    def test_new_entries_use_current_max_priority_and_overwrites_replace_slot(self) -> None:
        replay = self.make_buffer(capacity=3)
        self.fill(replay, 3)
        replay.update_priorities(torch.tensor([0, 1, 2]), torch.tensor([1.0, 9.0, 4.0]))

        replay.add(*self.transition(3))

        self.assertEqual(replay.write_index, 1)
        torch.testing.assert_close(
            replay.priorities,
            torch.tensor(
                [
                    9.0 + replay.priority_epsilon,
                    9.0 + replay.priority_epsilon,
                    4.0 + replay.priority_epsilon,
                ]
            ),
        )

    def test_cached_active_max_tracks_a_priority_reduction(self) -> None:
        replay = self.make_buffer(capacity=3)
        self.fill(replay, 3)
        replay.update_priorities(
            torch.tensor([0, 1, 2]),
            torch.tensor([20.0, 8.0, 4.0]),
        )
        replay.update_priorities(torch.tensor([0]), torch.tensor([0.0]))

        self.assertAlmostEqual(
            float(replay.max_priority),
            8.0 + replay.priority_epsilon,
        )
        replay.add(*self.transition(3))
        self.assertAlmostEqual(
            float(replay.priorities[0]),
            8.0 + replay.priority_epsilon,
        )

    def test_add_batch_initializes_every_inserted_slot_at_active_max(self) -> None:
        replay = self.make_buffer(capacity=4)
        self.fill(replay, 2)
        replay.update_priorities(torch.tensor([0, 1]), torch.tensor([2.0, 7.0]))

        transitions = [self.transition(index) for index in (2, 3)]
        replay.add_batch(
            np.stack([item[0] for item in transitions]),
            np.asarray([item[1] for item in transitions]),
            np.asarray([item[2] for item in transitions], dtype=np.float32),
            np.stack([item[3] for item in transitions]),
            np.asarray([item[4] for item in transitions], dtype=np.bool_),
            np.asarray([item[5] for item in transitions], dtype=np.bool_),
        )

        torch.testing.assert_close(
            replay.priorities[2:4],
            torch.full((2,), 7.0 + replay.priority_epsilon),
        )

    def test_partial_buffer_never_samples_inactive_slots(self) -> None:
        replay = self.make_buffer(capacity=8)
        self.fill(replay, 2)
        replay.priorities[2:] = 10_000.0

        sample = replay.sample_prioritized(
            100,
            alpha=0.6,
            beta=0.4,
            generator=torch.Generator(device="cpu").manual_seed(21),
        )

        self.assertEqual(set(sample.indices.tolist()), {0, 1})
        self.assertTrue(torch.all(sample.probabilities == 0.5).item())

    def test_higher_priority_has_higher_sampling_frequency(self) -> None:
        replay = self.make_buffer(capacity=4)
        self.fill(replay, 4)
        replay.update_priorities(
            torch.arange(4),
            torch.tensor([1.0, 1.0, 1.0, 20.0]),
        )
        generator = torch.Generator(device="cpu").manual_seed(29)
        counts = torch.zeros(4, dtype=torch.long)

        for _ in range(3_000):
            sample = replay.sample_prioritized(
                1,
                alpha=0.6,
                beta=0.4,
                generator=generator,
            )
            counts[sample.indices[0]] += 1

        self.assertGreater(int(counts[3]), int(counts[:3].max()) * 2)

    def test_alpha_zero_samples_all_active_slots_equally(self) -> None:
        replay = self.make_buffer(capacity=4)
        self.fill(replay, 4)
        replay.update_priorities(torch.arange(4), torch.tensor([1.0, 2.0, 7.0, 20.0]))
        generator = torch.Generator(device="cpu").manual_seed(31)
        counts = torch.zeros(4, dtype=torch.long)

        for _ in range(4_000):
            sample = replay.sample_prioritized(
                1,
                alpha=0.0,
                beta=0.4,
                generator=generator,
            )
            counts[sample.indices[0]] += 1

        self.assertTrue(torch.all(counts > 800).item(), counts)
        self.assertTrue(torch.all(counts < 1_200).item(), counts)

    def test_sampling_is_seedable_and_returns_aligned_weights(self) -> None:
        first = self.make_buffer(capacity=4)
        second = self.make_buffer(capacity=4)
        self.fill(first, 4)
        self.fill(second, 4)
        errors = torch.tensor([1.0, 2.0, 4.0, 8.0])
        first.update_priorities(torch.arange(4), errors)
        second.update_priorities(torch.arange(4), errors)

        sample_a = first.sample_prioritized(
            32,
            alpha=0.6,
            beta=0.7,
            generator=torch.Generator(device="cpu").manual_seed(45),
        )
        sample_b = second.sample_prioritized(
            32,
            alpha=0.6,
            beta=0.7,
            generator=torch.Generator(device="cpu").manual_seed(45),
        )

        torch.testing.assert_close(sample_a.indices, sample_b.indices)
        torch.testing.assert_close(
            sample_a.batch.states[:, 0, 0, 0],
            sample_a.indices.to(torch.float32) / 255.0,
        )
        torch.testing.assert_close(sample_a.probabilities, sample_b.probabilities)
        torch.testing.assert_close(sample_a.importance_weights, sample_b.importance_weights)
        torch.testing.assert_close(
            sample_a.importance_weights,
            importance_sampling_weights(
                sample_a.probabilities,
                population_size=4,
                beta=0.7,
            ),
        )

    def test_duplicate_priority_updates_keep_the_largest_td_error(self) -> None:
        replay = self.make_buffer(capacity=3)
        self.fill(replay, 3)

        replay.update_priorities(
            torch.tensor([0, 0, 1]),
            torch.tensor([2.0, 7.0, 4.0]),
        )

        self.assertAlmostEqual(float(replay.priorities[0]), 7.0 + replay.priority_epsilon)
        self.assertAlmostEqual(float(replay.priorities[1]), 4.0 + replay.priority_epsilon)

    def test_invalid_priorities_fail_and_epsilon_keeps_zero_td_errors_positive(self) -> None:
        replay = self.make_buffer(capacity=3)
        self.fill(replay, 2)

        replay.update_priorities(torch.tensor([0]), torch.tensor([0.0]))

        self.assertAlmostEqual(float(replay.priorities[0]), replay.priority_epsilon)
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            replay.update_priorities(torch.tensor([1]), torch.tensor([float("nan")]))
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            replay.update_priorities(torch.tensor([1]), torch.tensor([-1.0]))
        with self.assertRaises(ValueError):
            replay.update_priorities(torch.tensor([7]), torch.tensor([1.0]))

    def test_priority_storage_is_optional_for_uniform_mode(self) -> None:
        replay = GPUReplayBuffer(
            4,
            observation_shape=self.SHAPE,
            device="cpu",
        )

        self.assertIsNone(replay.priorities)
        self.assertIsNone(replay.max_priority)
        self.assertEqual(replay.allocated_bytes, replay.capacity * replay.bytes_per_transition)
        prioritized = self.make_buffer(capacity=4)
        self.assertEqual(
            prioritized.allocated_bytes,
            prioritized.capacity * prioritized.bytes_per_transition
            + prioritized.max_priority.element_size(),
        )
        with self.assertRaisesRegex(ValueError, "prioritized replay is disabled"):
            replay.sample_prioritized(1, alpha=0.6, beta=0.4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device-residency checks")
    def test_prioritized_sampling_and_weights_stay_on_cuda(self) -> None:
        replay = GPUReplayBuffer(
            4,
            observation_shape=self.SHAPE,
            device="cuda",
            prioritized=True,
        )
        self.fill(replay, 4)
        sample = replay.sample_prioritized(
            8,
            alpha=0.6,
            beta=0.4,
            generator=torch.Generator(device="cuda").manual_seed(51),
        )

        self.assertEqual(replay.priorities.device.type, "cuda")
        self.assertEqual(sample.batch.states.device.type, "cuda")
        self.assertEqual(sample.indices.device.type, "cuda")
        self.assertEqual(sample.probabilities.device.type, "cuda")
        self.assertEqual(sample.importance_weights.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
