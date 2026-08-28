"""Behavioral tests for the GPU-resident replay backend."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from breakout_rl.replay import ReplayBuffer
from breakout_rl.replay_gpu import GPUReplayBuffer


class GPUReplayBufferTests(unittest.TestCase):
    SHAPE = (2, 3, 3)

    @staticmethod
    def transition(index: int) -> tuple[np.ndarray, int, float, np.ndarray, bool, bool]:
        state = np.full(GPUReplayBufferTests.SHAPE, index, dtype=np.uint8)
        next_state = np.full(
            GPUReplayBufferTests.SHAPE,
            index + 1,
            dtype=np.uint8,
        )
        return state, index % 4, float(index) + 0.5, next_state, index == 2, index == 3

    def make_buffer(self, capacity: int = 8) -> GPUReplayBuffer:
        return GPUReplayBuffer(
            capacity,
            observation_shape=self.SHAPE,
            device="cpu",
        )

    def test_partial_replay_only_samples_active_slots(self) -> None:
        replay = self.make_buffer()
        self.assertEqual(replay.bytes_per_transition, 2 * 18 + 8 + 4 + 1 + 1)
        self.assertEqual(replay.allocated_bytes, 8 * replay.bytes_per_transition)
        replay.add(*self.transition(0))
        replay.add(*self.transition(1))

        generator = torch.Generator(device="cpu").manual_seed(42)
        batch, indices = replay.sample_with_indices(2, generator=generator)

        self.assertEqual(replay.size, 2)
        self.assertEqual(replay.write_index, 2)
        self.assertEqual(set(indices.tolist()), {0, 1})
        self.assertTrue(torch.all(batch.states >= 0.0))
        self.assertTrue(torch.all(batch.states <= 1.0))

        with self.assertRaises(ValueError):
            replay.sample(3, generator=generator)

    def test_ring_overwrite_preserves_active_physical_slots(self) -> None:
        replay = self.make_buffer(capacity=3)
        for index in range(4):
            replay.add(*self.transition(index))

        self.assertEqual(replay.size, 3)
        self.assertEqual(replay.write_index, 1)
        batch = replay.gather(torch.tensor([0, 1, 2], dtype=torch.long))

        np.testing.assert_allclose(
            batch.states[:, 0, 0, 0].numpy(),
            np.array([3.0, 1.0, 2.0], dtype=np.float32) / 255.0,
        )

    def test_add_batch_preserves_order_and_wraps_the_ring(self) -> None:
        replay = self.make_buffer(capacity=4)
        transitions = [self.transition(index) for index in range(3)]
        replay.add_batch(
            np.stack([transition[0] for transition in transitions]),
            np.asarray([transition[1] for transition in transitions]),
            np.asarray([transition[2] for transition in transitions]),
            np.stack([transition[3] for transition in transitions]),
            np.asarray([transition[4] for transition in transitions]),
            np.asarray([transition[5] for transition in transitions]),
        )
        replay.add_batch(
            np.stack([self.transition(index)[0] for index in range(3, 6)]),
            np.asarray([3, 0, 1]),
            np.asarray([3.5, 4.5, 5.5]),
            np.stack([self.transition(index)[3] for index in range(3, 6)]),
            np.asarray([False, False, False]),
            np.asarray([True, False, False]),
        )

        self.assertEqual(replay.size, 4)
        self.assertEqual(replay.write_index, 2)
        np.testing.assert_array_equal(
            replay.states[:, 0, 0, 0].cpu().numpy(),
            np.array([4, 5, 2, 3], dtype=np.uint8),
        )
        batch = replay.gather(np.array([2, 3, 0, 1], dtype=np.int64))
        np.testing.assert_array_equal(
            batch.states[:, 0, 0, 0].cpu().numpy(),
            np.array([2, 3, 4, 5], dtype=np.float32) / 255.0,
        )

    def test_sampling_is_without_replacement_and_seedable(self) -> None:
        first = self.make_buffer(capacity=6)
        second = self.make_buffer(capacity=6)
        for index in range(6):
            transition = self.transition(index)
            first.add(*transition)
            second.add(*transition)

        first_generator = torch.Generator(device="cpu").manual_seed(7)
        second_generator = torch.Generator(device="cpu").manual_seed(7)
        _, first_indices = first.sample_with_indices(6, generator=first_generator)
        _, second_indices = second.sample_with_indices(6, generator=second_generator)

        self.assertEqual(len(set(first_indices.tolist())), 6)
        torch.testing.assert_close(first_indices, second_indices)

    def test_sampling_is_uniform_over_active_slots(self) -> None:
        replay = self.make_buffer(capacity=6)
        for index in range(6):
            replay.add(*self.transition(index))

        generator = torch.Generator(device="cpu").manual_seed(19)
        counts = np.zeros(6, dtype=np.int64)
        for _ in range(6_000):
            index = int(replay.sample_indices(1, generator=generator)[0].item())
            counts[index] += 1

        self.assertTrue(np.all(counts > 700), counts)
        self.assertTrue(np.all(counts < 1_300), counts)

    def test_fixed_index_batch_matches_numpy_replay_contract(self) -> None:
        cpu = ReplayBuffer(capacity=6, observation_shape=self.SHAPE)
        gpu = self.make_buffer(capacity=6)
        for index in range(6):
            transition = self.transition(index)
            cpu.add(*transition)
            gpu.add(*transition)

        indices = np.array([2, 3, 5], dtype=np.int64)
        gpu_batch = gpu.gather(torch.from_numpy(indices))

        # The fixed physical-index comparison is the semantic seam: both
        # backends must expose the same transition values for these slots.
        np.testing.assert_allclose(
            gpu_batch.states.numpy(),
            cpu.states[indices].astype(np.float32) / 255.0,
        )
        np.testing.assert_allclose(
            gpu_batch.next_states.numpy(),
            cpu.next_states[indices].astype(np.float32) / 255.0,
        )
        np.testing.assert_array_equal(gpu_batch.actions.numpy(), cpu.actions[indices])
        np.testing.assert_allclose(gpu_batch.rewards.numpy(), cpu.rewards[indices])
        np.testing.assert_array_equal(
            gpu_batch.terminated.numpy(),
            cpu.terminated[indices],
        )
        np.testing.assert_array_equal(
            gpu_batch.truncated.numpy(),
            cpu.truncated[indices],
        )
        self.assertEqual(gpu_batch.states.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for storage integration")
    def test_cuda_storage_preserves_full_transition_tensor_contract(self) -> None:
        replay = GPUReplayBuffer(
            capacity=4,
            observation_shape=self.SHAPE,
            device="cuda",
        )
        for index in range(2):
            replay.add(*self.transition(index))

        batch = replay.sample(
            2,
            generator=torch.Generator(device="cuda").manual_seed(11),
        )
        self.assertEqual(batch.states.device.type, "cuda")
        self.assertEqual(batch.next_states.device.type, "cuda")
        self.assertEqual(batch.states.shape, (2, *self.SHAPE))
        self.assertEqual(batch.next_states.shape, (2, *self.SHAPE))
        self.assertEqual(batch.states.dtype, torch.float32)
        self.assertEqual(batch.next_states.dtype, torch.float32)
        self.assertEqual(batch.actions.dtype, torch.int64)
        self.assertEqual(batch.rewards.dtype, torch.float32)
        self.assertEqual(batch.terminated.dtype, torch.bool)
        self.assertEqual(batch.truncated.dtype, torch.bool)
        self.assertTrue(torch.all(batch.states >= 0.0).item())
        self.assertTrue(torch.all(batch.states <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
