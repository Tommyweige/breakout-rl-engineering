"""Tests for the public single-batch DQN optimizer-update seam."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from breakout_rl.replay_tensors import ReplayTensorBatch
from breakout_rl.training.dqn_trainer import (
    DQNTrainingStepResult,
    dqn_training_step,
)


class TinyQNetwork(nn.Module):
    """Small deterministic Q network whose outputs are easy to calculate."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.linear(states)


def make_batch() -> ReplayTensorBatch:
    return ReplayTensorBatch(
        states=torch.tensor([[1.0, 2.0], [2.0, -1.0]]),
        actions=torch.tensor([2, 1], dtype=torch.long),
        rewards=torch.tensor([1.0, -1.0]),
        next_states=torch.tensor([[1.0, 2.0], [2.0, -1.0]]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
    )


class DQNTrainingStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.online = TinyQNetwork()
        self.target = TinyQNetwork()
        weights = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [-1.0, 0.5],
            ]
        )
        with torch.no_grad():
            self.online.linear.weight.copy_(weights)
            self.target.linear.weight.copy_(weights)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def test_selected_action_q_values_and_terminal_targets_are_correct(self) -> None:
        optimizer = torch.optim.SGD(self.online.parameters(), lr=0.01)

        result = dqn_training_step(
            self.online,
            self.target,
            optimizer,
            make_batch(),
            gamma=0.5,
            gradient_clip_norm=None,
        )

        self.assertIsInstance(result, DQNTrainingStepResult)
        torch.testing.assert_close(
            result.selected_q_values,
            torch.tensor([0.0, -2.0]),
        )
        torch.testing.assert_close(result.targets, torch.tensor([3.0, -1.0]))
        self.assertAlmostEqual(result.loss, 1.5)
        self.assertTrue(torch.isfinite(torch.tensor(result.gradient_norm)))

    def test_optimizer_changes_only_online_network_parameters(self) -> None:
        optimizer = torch.optim.SGD(self.online.parameters(), lr=0.01)
        online_before = [parameter.detach().clone() for parameter in self.online.parameters()]
        target_before = [parameter.detach().clone() for parameter in self.target.parameters()]

        dqn_training_step(
            self.online,
            self.target,
            optimizer,
            make_batch(),
            gamma=0.5,
            gradient_clip_norm=0.1,
        )

        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(online_before, self.online.parameters(), strict=True)
            )
        )
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(target_before, self.target.parameters(), strict=True)
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in self.target.parameters()))

    def test_non_finite_q_values_are_rejected_before_optimizer_step(self) -> None:
        with torch.no_grad():
            self.online.linear.weight[0, 0] = float("nan")
        optimizer = torch.optim.SGD(self.online.parameters(), lr=0.01)

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            dqn_training_step(
                self.online,
                self.target,
                optimizer,
                make_batch(),
                gamma=0.5,
                gradient_clip_norm=None,
            )


if __name__ == "__main__":
    unittest.main()
