"""Behavioral tests for Day 11 target-network utilities."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from breakout_rl.targets import (
    compute_double_dqn_targets,
    compute_dqn_targets,
    hard_update,
    should_update_target,
)


class TinyQNetwork(nn.Module):
    """Small deterministic Q-value model for target calculations."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.linear(states)


class FixedQNetwork(nn.Module):
    """Return a fixed action table while keeping a real module boundary."""

    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values), requires_grad=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(states.shape[0], -1)


class TargetNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.online = TinyQNetwork()
        self.target = TinyQNetwork()
        with torch.no_grad():
            self.online.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                        [-1.0, 0.5],
                    ]
                )
            )
            self.target.linear.weight.fill_(99.0)

    def test_hard_update_copies_values_without_sharing_parameters(self) -> None:
        hard_update(self.target, self.online)

        self.assertIsNot(self.target, self.online)
        self.assertTrue(
            all(
                torch.equal(target_value, online_value)
                for target_value, online_value in zip(
                    self.target.state_dict().values(),
                    self.online.state_dict().values(),
                    strict=True,
                )
            )
        )
        self.assertNotEqual(
            self.target.linear.weight.data_ptr(),
            self.online.linear.weight.data_ptr(),
        )
        self.assertFalse(self.target.linear.weight.requires_grad)

    def test_online_changes_do_not_change_target_until_next_sync(self) -> None:
        hard_update(self.target, self.online)
        original_target = self.target.linear.weight.detach().clone()

        with torch.no_grad():
            self.online.linear.weight.add_(1.0)

        self.assertTrue(torch.equal(self.target.linear.weight, original_target))
        self.assertFalse(torch.equal(self.target.linear.weight, self.online.linear.weight))

        hard_update(self.target, self.online)
        self.assertTrue(torch.equal(self.target.linear.weight, self.online.linear.weight))

    def test_hard_update_rejects_using_one_object_for_both_roles(self) -> None:
        with self.assertRaises(ValueError):
            hard_update(self.online, self.online)

    def test_vanilla_target_bootstraps_only_non_terminal_rows(self) -> None:
        hard_update(self.target, self.online)
        next_states = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
        rewards = torch.tensor([1.0, -1.0])
        terminated = torch.tensor([False, True])

        targets = compute_dqn_targets(
            rewards,
            next_states,
            terminated,
            self.target,
            gamma=0.5,
        )

        # max Q(next_state) is 4.0 for row 0 and 2.0 for row 1.
        torch.testing.assert_close(targets, torch.tensor([3.0, -1.0]))
        self.assertEqual(tuple(targets.shape), (2,))

    def test_truncated_like_non_terminal_row_still_bootstraps(self) -> None:
        hard_update(self.target, self.online)
        targets = compute_dqn_targets(
            torch.tensor([0.0]),
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([False]),
            self.target,
            gamma=0.5,
        )

        torch.testing.assert_close(targets, torch.tensor([2.0]))

    def test_double_dqn_selects_with_online_and_evaluates_with_target(self) -> None:
        online = FixedQNetwork([1.0, 5.0, 2.0, 0.0])
        target = FixedQNetwork([4.0, 3.0, 2.0, 1.0])

        targets = compute_double_dqn_targets(
            torch.tensor([1.0]),
            torch.zeros(1, 2),
            torch.tensor([False]),
            online,
            target,
            gamma=0.5,
        )

        # Online selects action 1 (5.0); target evaluates action 1 (3.0),
        # instead of taking target's independent maximum (4.0).
        torch.testing.assert_close(targets, torch.tensor([2.5]))

    def test_double_dqn_does_not_bootstrap_terminated_rows_or_create_gradients(self) -> None:
        online = FixedQNetwork([1.0, 5.0, 2.0, 0.0])
        target = FixedQNetwork([4.0, 3.0, 2.0, 1.0])
        next_states = torch.zeros(2, 2, requires_grad=True)

        targets = compute_double_dqn_targets(
            torch.tensor([1.0, -2.0]),
            next_states,
            torch.tensor([False, True]),
            online,
            target,
            gamma=0.5,
        )

        torch.testing.assert_close(targets, torch.tensor([2.5, -2.0]))
        self.assertFalse(targets.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in online.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))

    def test_target_inference_does_not_create_gradients(self) -> None:
        hard_update(self.target, self.online)
        targets = compute_dqn_targets(
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [2.0, -1.0]], requires_grad=True),
            torch.tensor([False, False]),
            self.target,
            gamma=0.9,
        )

        self.assertFalse(targets.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.target.parameters()))

    def test_numeric_termination_flags_must_be_zero_or_one(self) -> None:
        hard_update(self.target, self.online)
        with self.assertRaises(ValueError):
            compute_dqn_targets(
                torch.tensor([0.0]),
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([2.0]),
                self.target,
                gamma=0.5,
            )

    def test_invalid_gamma_is_rejected(self) -> None:
        for gamma in (-0.1, 1.1, float("nan"), True):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    compute_dqn_targets(
                        torch.tensor([0.0]),
                        torch.zeros(1, 2),
                        torch.tensor([False]),
                        self.target,
                        gamma=gamma,
                    )

    def test_update_schedule_uses_positive_interval_and_includes_zero(self) -> None:
        self.assertTrue(should_update_target(0, 4))
        self.assertFalse(should_update_target(3, 4))
        self.assertTrue(should_update_target(8, 4))

        for step, interval in ((-1, 4), (0, 0), (0, -1), (0, True)):
            with self.subTest(step=step, interval=interval):
                with self.assertRaises(ValueError):
                    should_update_target(step, interval)


if __name__ == "__main__":
    unittest.main()
