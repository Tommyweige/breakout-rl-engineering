"""Tests for the public Q-value parity metric seam."""

from __future__ import annotations

import unittest

import numpy as np

from breakout_rl.onnx_parity import compare_q_values


class ONNXParityMetricTests(unittest.TestCase):
    def test_metrics_report_errors_actions_margins_and_disagreement_ids(self) -> None:
        reference = np.array(
            [
                [1.0, 2.0, 3.0, 0.0],
                [4.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        candidate = np.array(
            [
                [1.0, 2.0, 2.9, 0.0],
                [4.0, 4.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        result = compare_q_values(
            reference,
            candidate,
            sample_ids=[10, 20],
        )

        self.assertAlmostEqual(result["max_absolute_error"], 3.1, places=6)
        self.assertAlmostEqual(result["mean_absolute_error"], 0.4, places=6)
        self.assertEqual(result["action_agreement_rate"], 0.5)
        self.assertEqual(result["disagreement_sample_ids"], [20])
        self.assertEqual(result["reference_actions"], [2, 0])
        self.assertEqual(result["candidate_actions"], [2, 1])
        self.assertEqual(result["reference_top_2_q_margin"]["min"], 1.0)
        self.assertEqual(result["reference_top_2_q_margin"]["max"], 3.0)

    def test_metrics_reject_shape_or_non_finite_mismatches(self) -> None:
        with self.assertRaises(ValueError):
            compare_q_values(np.zeros((1, 4)), np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            compare_q_values(
                np.array([[0.0, np.nan, 0.0, 0.0]], dtype=np.float32),
                np.zeros((1, 4), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
