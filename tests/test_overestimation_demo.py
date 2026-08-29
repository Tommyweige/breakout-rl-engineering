"""Behavioral tests for the reproducible overestimation toy experiment."""

from __future__ import annotations

import unittest

import numpy as np

from breakout_rl.analysis.overestimation import (
    generate_noisy_estimates,
    run_noise_sweep,
    simulate_overestimation,
)


class OverestimationDemoTests(unittest.TestCase):
    def test_fixed_seed_reproduces_estimator_shapes_and_summary(self) -> None:
        first = simulate_overestimation(actions=4, trials=1000, noise_std=1.0, seed=42)
        second = simulate_overestimation(actions=4, trials=1000, noise_std=1.0, seed=42)

        self.assertEqual(first, second)
        estimates_a, estimates_b = generate_noisy_estimates(
            actions=4,
            trials=7,
            noise_std=0.5,
            seed=42,
        )
        self.assertEqual(estimates_a.shape, (7, 4))
        self.assertEqual(estimates_b.shape, (7, 4))
        self.assertTrue(np.isfinite(estimates_a).all())

    def test_decoupled_path_uses_a_for_selection_and_b_for_evaluation(self) -> None:
        result = simulate_overestimation(actions=2, trials=1, noise_std=0.0, seed=42)

        self.assertEqual(result["selected_action_counts"], {"0": 1, "1": 0})
        self.assertEqual(result["true_value"], 0.0)
        self.assertEqual(result["single_estimate_mean"], 0.0)
        self.assertEqual(result["vanilla_max_mean"], 0.0)
        self.assertEqual(result["decoupled_mean"], 0.0)

    def test_noise_sweep_contains_real_bias_columns(self) -> None:
        rows = run_noise_sweep(
            actions=4,
            trials=1000,
            noise_stds=(0.1, 0.5, 1.0),
            seed=42,
        )

        self.assertEqual([row["noise_std"] for row in rows], [0.1, 0.5, 1.0])
        for row in rows:
            self.assertIn("vanilla_bias", row)
            self.assertIn("decoupled_bias", row)
            self.assertEqual(row["true_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
