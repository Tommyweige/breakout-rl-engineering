"""Tests for the reproducible Q-value overestimation toy evidence."""

from __future__ import annotations

import unittest

from overestimation_demo import run_simulation


class OverestimationEvidenceTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_separates_the_estimators(self) -> None:
        kwargs = {
            "seed": 42,
            "trials": 5_000,
            "true_action_values": [1.0, 1.0, 1.0, 1.0],
            "noise_stds": [0.0, 0.5],
            "chunk_size": 257,
        }
        first = run_simulation(**kwargs)
        second = run_simulation(**kwargs)

        self.assertEqual(first["results"], second["results"])
        zero_noise = first["results"][0]
        self.assertAlmostEqual(zero_noise["vanilla_max_mean"], 1.0)
        self.assertAlmostEqual(zero_noise["decoupled_estimator_mean"], 1.0)
        noisy = first["results"][1]
        self.assertGreater(noisy["vanilla_max_bias"], 0.0)

if __name__ == "__main__":
    unittest.main()
