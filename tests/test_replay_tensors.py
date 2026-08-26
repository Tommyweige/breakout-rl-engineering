"""Tests for replay batch transfer strategies."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from breakout_rl.replay import TransitionBatch
from breakout_rl.replay_tensors import PreallocatedReplayBatchTransfer


class PreallocatedReplayBatchTransferTests(unittest.TestCase):
    def test_cpu_transfer_preserves_transition_fields_and_normalizes_pixels(self) -> None:
        states = np.zeros((2, 4, 84, 84), dtype=np.uint8)
        next_states = np.full((2, 4, 84, 84), 255, dtype=np.uint8)
        batch = TransitionBatch(
            states=states,
            actions=np.array([1, 3], dtype=np.int64),
            rewards=np.array([1.0, -1.0], dtype=np.float32),
            next_states=next_states,
            terminated=np.array([False, True], dtype=np.bool_),
            truncated=np.array([False, False], dtype=np.bool_),
        )

        transfer = PreallocatedReplayBatchTransfer(
            batch_size=2,
            observation_shape=(4, 84, 84),
            device="cpu",
        )
        result = transfer.transfer(batch)

        self.assertEqual(result.states.device.type, "cpu")
        self.assertEqual(result.states.dtype, torch.float32)
        self.assertEqual(result.next_states.dtype, torch.float32)
        self.assertTrue(torch.equal(result.actions, torch.tensor([1, 3])))
        self.assertTrue(torch.equal(result.terminated, torch.tensor([False, True])))
        self.assertTrue(torch.equal(result.states, torch.zeros_like(result.states)))
        self.assertTrue(torch.equal(result.next_states, torch.ones_like(result.next_states)))

    def test_transfer_rejects_an_unexpected_batch_shape(self) -> None:
        transfer = PreallocatedReplayBatchTransfer(
            batch_size=2,
            observation_shape=(4, 84, 84),
            device="cpu",
        )
        invalid = TransitionBatch(
            states=np.zeros((1, 4, 84, 84), dtype=np.uint8),
            actions=np.array([0], dtype=np.int64),
            rewards=np.array([0.0], dtype=np.float32),
            next_states=np.zeros((1, 4, 84, 84), dtype=np.uint8),
            terminated=np.array([False], dtype=np.bool_),
            truncated=np.array([False], dtype=np.bool_),
        )

        with self.assertRaises(ValueError):
            transfer.transfer(invalid)


if __name__ == "__main__":
    unittest.main()
