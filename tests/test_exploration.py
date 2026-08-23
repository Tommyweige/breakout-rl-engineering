"""Behavioral tests for Day 10 exploration and epsilon scheduling."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from breakout_rl.exploration import (
    LinearEpsilonSchedule,
    select_epsilon_greedy_action,
)


class LinearEpsilonScheduleTests(unittest.TestCase):
    def test_steps_at_or_before_zero_use_the_start_value(self) -> None:
        schedule = LinearEpsilonSchedule(start=0.9, end=0.1, decay_steps=10)

        self.assertEqual(schedule.value(-1), 0.9)
        self.assertEqual(schedule.value(0), 0.9)

    def test_midpoint_is_linearly_interpolated(self) -> None:
        schedule = LinearEpsilonSchedule(start=0.9, end=0.1, decay_steps=10)

        self.assertAlmostEqual(schedule.value(5), 0.5)

    def test_decay_end_and_later_steps_use_the_end_value(self) -> None:
        schedule = LinearEpsilonSchedule(start=0.9, end=0.1, decay_steps=10)

        self.assertEqual(schedule.value(10), 0.1)
        self.assertEqual(schedule.value(100), 0.1)

    def test_schedule_does_not_overshoot_between_endpoints(self) -> None:
        schedule = LinearEpsilonSchedule(start=0.9, end=0.1, decay_steps=10)

        values = [schedule.value(step) for step in range(-3, 20)]

        self.assertTrue(all(0.1 <= value <= 0.9 for value in values))
        self.assertEqual(values, sorted(values, reverse=True))

    def test_probability_and_decay_configuration_are_validated(self) -> None:
        for start, end in ((-0.1, 0.1), (0.1, 1.1), (float("nan"), 0.1)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    LinearEpsilonSchedule(start=start, end=end, decay_steps=10)

        for decay_steps in (0, -1, True):
            with self.subTest(decay_steps=decay_steps):
                with self.assertRaises(ValueError):
                    LinearEpsilonSchedule(start=0.9, end=0.1, decay_steps=decay_steps)


class EpsilonGreedyActionTests(unittest.TestCase):
    Q_VALUES = torch.tensor([1.0, 2.0, 5.0, 3.0])

    def test_zero_epsilon_always_uses_the_greedy_branch(self) -> None:
        rng = np.random.default_rng(42)

        decisions = [
            select_epsilon_greedy_action(
                self.Q_VALUES,
                epsilon=0.0,
                action_space_n=4,
                rng=rng,
            )
            for _ in range(20)
        ]

        self.assertEqual(decisions, [(2, "greedy")] * 20)

    def test_full_epsilon_always_uses_the_random_branch(self) -> None:
        rng = np.random.default_rng(42)

        decisions = [
            select_epsilon_greedy_action(
                self.Q_VALUES,
                epsilon=1.0,
                action_space_n=4,
                rng=rng,
            )
            for _ in range(20)
        ]

        self.assertTrue(all(source == "random" for _, source in decisions))
        self.assertGreater(len({action for action, _ in decisions}), 1)

    def test_same_rng_seed_reproduces_the_action_sequence(self) -> None:
        def sample(seed: int) -> list[tuple[int, str]]:
            rng = np.random.default_rng(seed)
            return [
                select_epsilon_greedy_action(
                    self.Q_VALUES,
                    epsilon=0.2,
                    action_space_n=4,
                    rng=rng,
                )
                for _ in range(50)
            ]

        self.assertEqual(sample(42), sample(42))

    def test_greedy_ties_choose_the_first_maximum(self) -> None:
        action, source = select_epsilon_greedy_action(
            torch.tensor([5.0, 5.0, 1.0]),
            epsilon=0.0,
            action_space_n=3,
            rng=np.random.default_rng(42),
        )

        self.assertEqual((action, source), (0, "greedy"))

    def test_invalid_epsilon_is_rejected(self) -> None:
        for epsilon in (-0.1, 1.1, float("nan")):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    select_epsilon_greedy_action(
                        self.Q_VALUES,
                        epsilon=epsilon,
                        action_space_n=4,
                        rng=np.random.default_rng(42),
                    )

    def test_q_values_and_action_space_must_match(self) -> None:
        with self.assertRaises(ValueError):
            select_epsilon_greedy_action(
                self.Q_VALUES,
                epsilon=0.0,
                action_space_n=3,
                rng=np.random.default_rng(42),
            )

        with self.assertRaises(ValueError):
            select_epsilon_greedy_action(
                torch.tensor([[1.0, 2.0]]),
                epsilon=0.0,
                action_space_n=2,
                rng=np.random.default_rng(42),
            )


if __name__ == "__main__":
    unittest.main()
