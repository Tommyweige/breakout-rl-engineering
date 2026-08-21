"""Tests for the observation-to-model tensor contract."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from breakout_rl.tensors import observation_to_tensor


class ObservationToTensorTests(unittest.TestCase):
    def setUp(self) -> None:
        values = np.arange(4 * 84 * 84, dtype=np.uint32) % 256
        self.observation = values.astype(np.uint8).reshape(4, 84, 84)

    def test_single_observation_becomes_normalized_batched_float_tensor(self) -> None:
        tensor = observation_to_tensor(self.observation, device="cpu")

        self.assertEqual(tensor.shape, (1, 4, 84, 84))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 1]), 1.0 / 255.0)

    def test_batch_observation_keeps_one_batch_dimension(self) -> None:
        batch = np.stack([self.observation, self.observation], axis=0)

        tensor = observation_to_tensor(batch, device="cpu")

        self.assertEqual(tensor.shape, (2, 4, 84, 84))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_add_batch_dim_false_keeps_single_observation_rank(self) -> None:
        tensor = observation_to_tensor(
            self.observation,
            device=torch.device("cpu"),
            add_batch_dim=False,
        )

        self.assertEqual(tensor.shape, (4, 84, 84))

    def test_non_uint8_observation_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            observation_to_tensor(self.observation.astype(np.float32), device="cpu")

    def test_wrong_observation_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            observation_to_tensor(self.observation.transpose(1, 2, 0), device="cpu")

        with self.assertRaises(ValueError):
            observation_to_tensor(self.observation[None, ...][..., :83], device="cpu")

    def test_unavailable_cuda_is_rejected_instead_of_silently_falling_back(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is available in this environment")

        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            observation_to_tensor(self.observation, device="cuda")


if __name__ == "__main__":
    unittest.main()
