"""Behavioral tests for batched per-environment exploration decisions."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from breakout_rl.exploration import select_epsilon_greedy_actions


class BatchedEpsilonGreedyActionTests(unittest.TestCase):
    def test_batch_greedy_actions_match_each_q_value_row(self) -> None:
        actions, sources = select_epsilon_greedy_actions(
            torch.tensor([[1.0, 4.0], [9.0, 2.0]]),
            epsilon=0.0,
            action_space_n=2,
            rng=np.random.default_rng(42),
        )

        np.testing.assert_array_equal(actions, np.array([1, 0]))
        np.testing.assert_array_equal(sources, np.array(["greedy", "greedy"]))

    def test_epsilon_is_decided_independently_for_each_environment(self) -> None:
        q_values = torch.tensor([[1.0, 4.0], [9.0, 2.0]])
        first_actions, first_sources = select_epsilon_greedy_actions(
            q_values,
            epsilon=np.array([1.0, 0.0]),
            action_space_n=2,
            rng=np.random.default_rng(7),
        )
        second_actions, second_sources = select_epsilon_greedy_actions(
            q_values,
            epsilon=np.array([1.0, 0.0]),
            action_space_n=2,
            rng=np.random.default_rng(7),
        )

        np.testing.assert_array_equal(first_actions, second_actions)
        np.testing.assert_array_equal(first_sources, second_sources)
        self.assertEqual(first_sources.tolist(), ["random", "greedy"])
        self.assertEqual(int(first_actions[1]), 0)

    def test_batch_shape_and_epsilon_length_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            select_epsilon_greedy_actions(
                torch.ones(0, 2),
                epsilon=0.0,
                action_space_n=2,
                rng=np.random.default_rng(1),
            )
        with self.assertRaises(ValueError):
            select_epsilon_greedy_actions(
                torch.ones(2, 2),
                epsilon=[0.0],
                action_space_n=2,
                rng=np.random.default_rng(1),
            )


if __name__ == "__main__":
    unittest.main()
