"""Tests for the shared checkpoint file module."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from breakout_rl.checkpoint import (
    capture_rng_state,
    check_resume_identity,
    read_checkpoint,
    restore_q_network,
    restore_rng_state,
    write_checkpoint,
)
from breakout_rl.models.factory import build_q_network


class CheckpointModuleTests(unittest.TestCase):
    def test_write_then_read_restores_the_same_network(self) -> None:
        source = build_q_network("dueling", num_actions=4, input_shape=(4, 84, 84), hidden_dim=32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-00000001.pt"
            write_checkpoint(
                {"architecture": "dueling", "online_network": source.state_dict()},
                path,
            )
            self.assertFalse(path.with_suffix(".tmp").exists())
            payload = read_checkpoint(path)
        restored = restore_q_network(
            payload["online_network"],
            architecture="dueling",
            num_actions=4,
            input_shape=[4, 84, 84],
            hidden_dim=32,
            device="cpu",
        )
        self.assertFalse(restored.training)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_read_rejects_non_mapping_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save([1, 2, 3], path)
            with self.assertRaisesRegex(ValueError, "checkpoint must contain a mapping"):
                read_checkpoint(path)

    def test_rng_round_trip_replays_the_same_draws(self) -> None:
        action_rng = np.random.default_rng(7)
        state = capture_rng_state(action_rng)
        first = (random.random(), np.random.rand(), torch.rand(1).item(), action_rng.random())
        restore_rng_state(state, action_rng)
        second = (random.random(), np.random.rand(), torch.rand(1).item(), action_rng.random())
        self.assertEqual(first, second)

    def test_resume_identity_normalizes_algorithm_and_checks_architecture(self) -> None:
        payload = {"replay_saved": False, "algorithm": " Double_DQN ", "architecture": "standard"}
        check_resume_identity(
            payload, algorithm="double_dqn", architecture="standard", allow_replay_rewarm=True
        )
        with self.assertRaisesRegex(ValueError, "exact continuation is unavailable"):
            check_resume_identity(
                payload, algorithm="double_dqn", architecture="standard", allow_replay_rewarm=False
            )
        with self.assertRaisesRegex(ValueError, "architecture does not match"):
            check_resume_identity(
                payload, algorithm="double_dqn", architecture="dueling", allow_replay_rewarm=True
            )


if __name__ == "__main__":
    unittest.main()
