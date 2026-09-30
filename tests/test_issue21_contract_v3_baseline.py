from __future__ import annotations

import json
import unittest
from pathlib import Path

from breakout_rl.contract_v3_screening import build_screening_assessment
from breakout_rl.evaluation_contract import load_evaluation_contract
from breakout_rl.training.config import DQNConfig
from scripts.analysis.control_diagnostics import summarize_training_control_metrics


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _milestone_record(
    agent_steps: int,
    *,
    mean_score: float,
    median_score: float,
    max_score: float,
    clear_count: int = 0,
) -> dict[str, object]:
    return {
        "agent_steps": agent_steps,
        "summary": {
            "mean_return": mean_score,
            "median_return": median_score,
            "max_return": max_score,
            "clear_count": clear_count,
            "clear_rate": clear_count / 15,
            "mean_life_loss_count": 4.0,
            "life_losses_per_1000_steps": 0.1,
            "mean_score_per_life": 0.5,
            "mean_frames_between_life_losses": 150.0,
            "mean_time_to_first_life_loss": 200.0,
        },
    }


class Issue21ContractV3BaselineTests(unittest.TestCase):
    def test_no_clear_without_score_progression_is_inconclusive(self) -> None:
        assessment = build_screening_assessment(
            [
                _milestone_record(
                    400_000, mean_score=2.0, median_score=2.0, max_score=4.0
                ),
                _milestone_record(
                    1_000_000, mean_score=2.0, median_score=2.0, max_score=4.0
                ),
                _milestone_record(
                    2_000_000, mean_score=1.5, median_score=1.0, max_score=3.0
                ),
            ]
        )

        self.assertEqual(assessment["status"], "inconclusive")
        self.assertIsNone(assessment["promising"])
        self.assertEqual(assessment["decision"], "hold_for_more_evidence")
        self.assertEqual(
            assessment["criteria"]["first_verified_clear"]["status"], "not_met"
        )
        self.assertEqual(
            assessment["criteria"]["consistent_raw_score_progression"]["status"],
            "not_observed",
        )
        self.assertFalse(assessment["second_training_seed_started"])
        self.assertFalse(assessment["next_algorithm_started"])

    def test_any_verified_clear_makes_the_screen_promising(self) -> None:
        assessment = build_screening_assessment(
            [
                _milestone_record(
                    400_000, mean_score=2.0, median_score=2.0, max_score=4.0
                ),
                _milestone_record(
                    1_000_000,
                    mean_score=3.0,
                    median_score=3.0,
                    max_score=5.0,
                    clear_count=1,
                ),
                _milestone_record(
                    2_000_000, mean_score=2.0, median_score=2.0, max_score=4.0
                ),
            ]
        )

        self.assertEqual(assessment["status"], "promising")
        self.assertTrue(assessment["promising"])
        self.assertEqual(assessment["decision"], "promote_for_follow_up")
        self.assertEqual(assessment["clear_checkpoints"], [1_000_000])

    def test_consistent_raw_score_progression_makes_the_screen_promising(self) -> None:
        assessment = build_screening_assessment(
            [
                _milestone_record(
                    400_000, mean_score=2.13, median_score=2.0, max_score=4.0
                ),
                _milestone_record(
                    1_000_000, mean_score=4.6, median_score=5.0, max_score=7.0
                ),
                _milestone_record(
                    2_000_000, mean_score=6.93, median_score=7.0, max_score=12.0
                ),
            ]
        )

        self.assertEqual(assessment["status"], "promising")
        self.assertTrue(assessment["promising"])
        self.assertEqual(assessment["decision"], "promote_for_follow_up")
        self.assertEqual(assessment["clear_checkpoints"], [])
        self.assertEqual(
            assessment["criteria"]["consistent_raw_score_progression"]["status"],
            "observed",
        )
        self.assertEqual(
            assessment["criteria"]["materially_higher_max_score"]["increase_points"],
            8.0,
        )

    def test_screening_config_declares_the_requested_continuous_baseline(self) -> None:
        config_path = (
            REPOSITORY_ROOT
            / "configs"
            / "training"
            / "issue21_contract_v3_baseline.json"
        )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        training = DQNConfig.from_dict(payload["training_config"])
        contract = load_evaluation_contract(REPOSITORY_ROOT / payload["contract"])

        self.assertEqual(payload["issue"], 21)
        self.assertEqual(payload["training_lifecycle"], "continuous_single_run")
        self.assertEqual(training.seed, 11)
        self.assertEqual(training.total_steps, 2_000_000)
        self.assertEqual(training.checkpoint_steps, (400_000, 1_000_000, 2_000_000))
        self.assertEqual(training.algorithm, "double_dqn")
        self.assertEqual(training.architecture, "dueling")
        self.assertEqual(training.replay_sampling, "uniform")
        self.assertEqual(training.life_loss_penalty, 0.0)
        self.assertEqual(contract.contract_id, "breakout-evaluation-v3-frame-skip-1")
        self.assertEqual(contract.frame_skip, 1)

    def test_control_summary_counts_switches_reversals_and_late_score_actions(
        self,
    ) -> None:
        rows = [
            {
                "environment_index": 0,
                "action_name": "LEFT",
                "current_raw_episode_return": 100,
                "terminated": False,
                "truncated": False,
            },
            {
                "environment_index": 1,
                "action_name": "NOOP",
                "current_raw_episode_return": 0,
                "terminated": False,
                "truncated": False,
            },
            {
                "environment_index": 0,
                "action_name": "RIGHT",
                "current_raw_episode_return": 400,
                "terminated": False,
                "truncated": False,
            },
            {
                "environment_index": 1,
                "action_name": "NOOP",
                "current_raw_episode_return": 0,
                "terminated": False,
                "truncated": False,
            },
            {
                "environment_index": 0,
                "action_name": "RIGHT",
                "current_raw_episode_return": 500,
                "terminated": False,
                "truncated": False,
            },
            {
                "environment_index": 1,
                "action_name": "LEFT",
                "current_raw_episode_return": 400,
                "terminated": False,
                "truncated": False,
            },
            {
                "environment_index": 0,
                "action_name": "NOOP",
                "current_raw_episode_return": 600,
                "terminated": True,
                "truncated": False,
            },
            {
                "environment_index": 1,
                "action_name": "RIGHT",
                "current_raw_episode_return": 700,
                "terminated": True,
                "truncated": False,
            },
        ]

        summary = summarize_training_control_metrics(rows)

        self.assertEqual(summary["transition_count"], 8)
        self.assertEqual(summary["noop_count"], 3)
        self.assertEqual(summary["action_switch_count"], 4)
        self.assertEqual(summary["adjacent_action_pairs"], 6)
        self.assertEqual(summary["left_right_reversal_count"], 2)
        self.assertEqual(summary["completed_episodes"], 2)
        self.assertEqual(
            summary["late_episode"]["action_distribution"], {"NOOP": 1, "RIGHT": 1}
        )
        self.assertEqual(summary["high_score"]["sample_count"], 3)
        self.assertEqual(
            summary["high_score"]["action_distribution"], {"NOOP": 1, "RIGHT": 2}
        )


if __name__ == "__main__":
    unittest.main()
