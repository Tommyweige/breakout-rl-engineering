"""Unit tests for the deterministic Bellman and return examples."""

from __future__ import annotations

import unittest

from scripts.demos.bellman_demo import bellman_target, discounted_return


class DiscountedReturnTests(unittest.TestCase):
    def test_gamma_zero_keeps_only_the_first_reward(self) -> None:
        self.assertAlmostEqual(discounted_return([2.0, 5.0, 7.0], gamma=0.0), 2.0)

    def test_gamma_one_returns_the_undiscounted_sum(self) -> None:
        self.assertAlmostEqual(discounted_return([2.0, 5.0, 7.0], gamma=1.0), 14.0)

    def test_general_gamma_applies_a_larger_discount_to_later_rewards(self) -> None:
        rewards = [0.0, 0.0, 3.0]

        self.assertAlmostEqual(discounted_return(rewards, gamma=0.9), 2.43)

    def test_empty_reward_sequence_has_zero_return(self) -> None:
        self.assertEqual(discounted_return([], gamma=0.9), 0.0)

    def test_invalid_gamma_is_rejected(self) -> None:
        for gamma in (-0.01, 1.01):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    discounted_return([1.0], gamma=gamma)


class BellmanTargetTests(unittest.TestCase):
    def test_terminal_target_does_not_bootstrap(self) -> None:
        target = bellman_target(
            reward=1.0,
            next_value=999.0,
            gamma=0.99,
            terminated=True,
        )

        self.assertEqual(target, 1.0)

    def test_non_terminal_target_includes_discounted_next_value(self) -> None:
        target = bellman_target(
            reward=0.0,
            next_value=3.0,
            gamma=0.9,
            terminated=False,
        )

        self.assertAlmostEqual(target, 2.7)

    def test_bellman_target_rejects_invalid_gamma(self) -> None:
        with self.assertRaises(ValueError):
            bellman_target(
                reward=1.0,
                next_value=3.0,
                gamma=2.0,
                terminated=False,
            )


if __name__ == "__main__":
    unittest.main()
