"""Tests for PER experiment provenance and paired return analysis."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.analysis.audit_per_baseline import audit_baseline
from scripts.analysis.compare_per_experiment import (
    _build_experiment_conclusion,
    _cumulative_distribution_at_transition,
    _describe,
    _episode_returns,
    _has_non_finite_diagnostic_event,
    _last_row_at_or_before_transition,
    _metric_value_at_transition,
    _policy_decision_distribution,
    _summarize_training_diagnostics,
    _validate_continuous_run,
    _validate_training_configs_differ_only_by_sampling,
)


class PERExperimentAnalysisTests(unittest.TestCase):
    def test_day20_staged_replay_reset_is_rejected_as_primary_control(self) -> None:
        audit = audit_baseline(Path("configs/per_experiment.json"))

        self.assertEqual(audit["status"], "incompatible")
        self.assertFalse(audit["reuse_allowed"])
        self.assertEqual(audit["decision"], "rerun_uniform_baseline")
        self.assertEqual(len(audit["failed_checks"]), 18)
        self.assertEqual(len(audit["checkpoints"]), 9)
        self.assertEqual(audit["protocol"]["training_seeds"], [11, 22, 33])
        self.assertEqual(audit["protocol"]["milestones"], [100_000, 250_000, 500_000])
        self.assertFalse(
            audit["continuation_semantics"][
                "historical_reference_compatible_with_primary"
            ]
        )
        for check_suffix in (
            "primary run has no staged resume",
            "resumed replay state is saved",
            "resume preserves replay history",
        ):
            matching = [
                check
                for check in audit["checks"]
                if check["name"].endswith(check_suffix)
            ]
            self.assertEqual(
                sum(not check["passed"] for check in matching),
                6,
            )
            if check_suffix == "resumed replay state is saved":
                self.assertTrue(all(check["observed"] is False for check in matching))
            if check_suffix == "resume preserves replay history":
                self.assertTrue(
                    all(
                        check["observed"]
                        == "fresh_replay_with_learning_starts_rewarm"
                        for check in matching
                    )
                )
            if check_suffix == "primary run has no staged resume":
                self.assertEqual(len(matching), 9)

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

    def test_stability_summary_uses_window_and_counts_non_finite_records(self) -> None:
        rows = [
            {
                "global_step": "10",
                "loss": "100",
                "q_mean": "100",
            },
            {
                "global_step": "20",
                "loss": "2",
                "q_mean": "3",
                "q_max": "4",
                "td_error_mean_abs": "1",
                "td_error_max_abs": "2",
                "gradient_norm": "0.5",
            },
            {
                "global_step": "30",
                "loss": "nan",
                "q_mean": "5",
                "q_max": "6",
                "td_error_mean_abs": "2",
                "td_error_max_abs": "3",
                "gradient_norm": "0.75",
            },
        ]

        summary = _summarize_training_diagnostics(
            rows,
            start_transition=20,
            end_transition=30,
        )

        self.assertEqual(summary["logged_record_count"], 2)
        self.assertEqual(summary["observed_transition_window"], {"start": 20, "end": 30})
        self.assertEqual(summary["metrics"]["loss"]["mean"], 2.0)
        self.assertEqual(summary["non_finite_record_count"], 1)
        self.assertEqual(summary["non_finite_value_count"], 1)

    def test_diagnostic_event_marks_nonfinite_but_not_run_completion(self) -> None:
        completed = {
            "training_status": "completed",
            "diagnostics": {
                "non_finite_record_count": 0,
                "non_finite_value_count": 0,
                "invalid_value_count": 0,
            },
        }
        non_finite = {
            "training_status": "completed",
            "diagnostics": {
                "non_finite_record_count": 1,
                "non_finite_value_count": 2,
                "invalid_value_count": 0,
            },
        }
        incomplete = {
            "training_status": "incomplete",
            "diagnostics": {
                "non_finite_record_count": 0,
                "non_finite_value_count": 0,
                "invalid_value_count": 0,
            },
        }

        self.assertFalse(_has_non_finite_diagnostic_event(completed))
        self.assertTrue(_has_non_finite_diagnostic_event(non_finite))
        self.assertFalse(_has_non_finite_diagnostic_event(incomplete))

    def test_comparison_rejects_staged_replay_rewarm_checkpoint(self) -> None:
        summary = {
            "status": "completed",
            "total_steps": 500_000,
            "runtime": {
                "wall_clock_seconds": 20.0,
                "steps_per_second": 25_000.0,
                "stage_start_step": 100_000,
                "stage_training_steps": 400_000,
                "resume_provenance": {
                    "replay_saved": False,
                    "replay_rewarm_steps_remaining": 1_000,
                },
                "replay_rewarm_steps_remaining": 0,
            },
        }
        evaluation = {
            "training": {
                "resume_provenance": {
                    "replay_saved": False,
                    "replay_resume_semantics": (
                        "fresh_replay_with_learning_starts_rewarm"
                    ),
                },
                "trainer_runtime": {
                    "wall_clock_seconds": 10.0,
                    "steps_per_second": 25_000.0,
                    "stage_start_step": 100_000,
                    "stage_training_steps": 250_000,
                },
            }
        }

        with self.assertRaisesRegex(ValueError, "staged resume or replay re-warm"):
            _validate_continuous_run(
                summary,
                evaluation,
                method="uniform",
                seed=11,
                transitions=250_000,
                total_steps=500_000,
            )

    def test_method_configs_may_differ_only_by_replay_sampling(self) -> None:
        per_config = {
            "seed": 11,
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "replay_sampling": "prioritized",
        }
        uniform_config = {**per_config, "replay_sampling": "uniform"}
        expected = {**per_config}

        _validate_training_configs_differ_only_by_sampling(
            per_config,
            uniform_config,
            expected_config=expected,
            seed=11,
            label="fixture",
        )

        mismatched_uniform = {**uniform_config, "gamma": 0.98}
        with self.assertRaisesRegex(ValueError, "training config field gamma"):
            _validate_training_configs_differ_only_by_sampling(
                per_config,
                mismatched_uniform,
                expected_config=expected,
                seed=11,
                label="fixture",
            )

    def test_action_and_policy_distributions_use_cumulative_checkpoint_counters(self) -> None:
        rows = [
            {
                "global_step": "30",
                "noop_count": "3",
                "fire_count": "4",
                "right_count": "12",
                "left_count": "11",
                "random_decision_count": "6",
                "greedy_decision_count": "24",
            }
        ]

        actions = _cumulative_distribution_at_transition(
            rows,
            transitions=30,
            fields={
                "noop": "noop_count",
                "fire": "fire_count",
                "right": "right_count",
                "left": "left_count",
            },
        )
        policy = _policy_decision_distribution(rows, transitions=30)
        selected = _last_row_at_or_before_transition(rows, transitions=30)

        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 30)
        self.assertEqual(actions["counts"]["fire"], 4)
        self.assertAlmostEqual(actions["fractions"]["left"], 11 / 30)
        self.assertAlmostEqual(policy["random_fraction"], 0.2)
        self.assertAlmostEqual(policy["greedy_fraction"], 0.8)

    def test_experiment_conclusion_reports_mixed_quality_and_cumulative_time(self) -> None:
        milestones = [
            {
                "transitions": 100,
                "uniform_across_training_seeds": {"mean": 8.0},
                "per_across_training_seeds": {"mean": 7.0},
                "paired_training_seed_mean_differences": [-1.0, 0.0, -2.0],
            },
            {
                "transitions": 200,
                "uniform_across_training_seeds": {"mean": 10.0},
                "per_across_training_seeds": {"mean": 11.0},
                "paired_training_seed_mean_differences": [1.0, 2.0, 0.0],
            },
        ]
        runtime_rows = [
            {
                "transitions": step,
                "uniform": {"elapsed_seconds": 100.0},
                "per": {"elapsed_seconds": 150.0},
            }
            for step in (100, 200)
        ]

        conclusion = _build_experiment_conclusion(milestones, runtime_rows)

        self.assertEqual(conclusion["outcome_category"], "mixed_no_established_advantage")
        self.assertEqual(conclusion["training_seed_count"], 3)
        self.assertIn("does not establish a consistent PER quality advantage", conclusion["statement"])
        self.assertIn("1.50x", conclusion["statement"])

    def test_descriptive_summary_reports_spread_without_pooling(self) -> None:
        summary = _describe([1.0, 3.0, 8.0])

        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["mean"], 4.0)
        self.assertEqual(summary["median"], 3.0)
        self.assertAlmostEqual(summary["sample_std"], 3.605551275463989)


if __name__ == "__main__":
    unittest.main()
