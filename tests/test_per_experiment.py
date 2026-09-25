"""Tests for PER experiment provenance and paired return analysis."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.analysis.audit_per_baseline import audit_baseline
from scripts.analysis.compare_per_experiment import (
    _describe,
    _episode_returns,
    _metric_value_at_transition,
)


class PERExperimentAnalysisTests(unittest.TestCase):
    def test_day20_uniform_baseline_audit_is_complete_and_compatible(self) -> None:
        audit = audit_baseline(Path("configs/per_experiment.json"))

        self.assertEqual(audit["status"], "compatible")
        self.assertTrue(audit["reuse_allowed"])
        self.assertEqual(audit["decision"], "reuse_day20_uniform")
        self.assertEqual(audit["failed_checks"], [])
        self.assertEqual(len(audit["checkpoints"]), 9)
        self.assertEqual(audit["protocol"]["training_seeds"], [11, 22, 33])
        self.assertEqual(audit["protocol"]["milestones"], [100_000, 250_000, 500_000])

    def test_episode_returns_pair_by_evaluation_seed_and_episode_index(self) -> None:
        payload = {
            "per_episode": [
                {
                    "evaluation_seed": 101,
                    "episode_index": 1,
                    "episode_return": 7.0,
                },
                {
                    "evaluation_seed": 101,
                    "episode_index": 2,
                    "episode_return": 9.0,
                },
                {
                    "evaluation_seed": 202,
                    "episode_index": 1,
                    "episode_return": 11.0,
                },
            ]
        }

        paired = _episode_returns(payload, label="fixture")

        self.assertEqual(
            paired,
            {
                (101, 1): 7.0,
                (101, 2): 9.0,
                (202, 1): 11.0,
            },
        )

    def test_duplicate_episode_keys_fail_closed(self) -> None:
        payload = {
            "per_episode": [
                {"evaluation_seed": 101, "episode_index": 1, "episode_return": 7.0},
                {"evaluation_seed": 101, "episode_index": 1, "episode_return": 8.0},
            ]
        }

        with self.assertRaisesRegex(ValueError, "duplicate paired episode key"):
            _episode_returns(payload, label="duplicate-fixture")

    def test_milestone_metrics_do_not_leak_values_from_later_steps(self) -> None:
        rows = [
            {"global_step": "100", "optimizer_updates": "20"},
            {"global_step": "250", "optimizer_updates": "40"},
            {"global_step": "500", "optimizer_updates": "90"},
        ]

        self.assertEqual(
            _metric_value_at_transition(rows, "optimizer_updates", 250),
            40.0,
        )
        self.assertIsNone(
            _metric_value_at_transition(rows, "optimizer_updates", 200)
        )

    def test_descriptive_summary_reports_spread_without_pooling(self) -> None:
        summary = _describe([1.0, 3.0, 8.0])

        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["mean"], 4.0)
        self.assertEqual(summary["median"], 3.0)
        self.assertAlmostEqual(summary["sample_std"], 3.605551275463989)


if __name__ == "__main__":
    unittest.main()
