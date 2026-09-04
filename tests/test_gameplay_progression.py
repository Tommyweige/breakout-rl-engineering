"""Contract tests for the Day 21 gameplay progression recorder."""

from __future__ import annotations

import unittest

from breakout_rl.evaluation_contract import load_evaluation_contract
from scripts.visualization.record_gameplay_progression import (
    EXPECTED_ACTION_MAPPING,
    EXPECTED_ARCHITECTURE,
    EXPECTED_FINAL_MODEL_SHA256,
    EXPECTED_LEARNED_SEED,
    GameplayProgressionConfig,
    load_gameplay_progression_config,
)


class GameplayProgressionTests(unittest.TestCase):
    def test_manifest_freezes_one_fair_learned_trajectory(self) -> None:
        config = load_gameplay_progression_config()

        self.assertIsInstance(config, GameplayProgressionConfig)
        self.assertEqual(config.showcase_evaluation_seed, 101)
        self.assertEqual(config.evaluation_epsilon, 0.0)
        self.assertEqual(config.max_steps_per_episode, 27000)
        self.assertEqual(config.codec_preference, ("avc1",))
        self.assertEqual(
            [stage.training_seed for stage in config.stages[1:]],
            [EXPECTED_LEARNED_SEED] * 6,
        )
        self.assertEqual(
            [stage.actual_checkpoint_step for stage in config.stages[1:]],
            [100_000, 200_000, 500_000, 1_000_000, 2_500_000, 5_000_000],
        )
        self.assertEqual(config.stages[5].checkpoint_sha256, EXPECTED_FINAL_MODEL_SHA256)
        self.assertEqual(config.stages[5].architecture, EXPECTED_ARCHITECTURE)

    def test_manifest_records_nearest_checkpoint_for_250k_target(self) -> None:
        config = load_gameplay_progression_config()
        stage = config.stages[2]

        self.assertEqual(stage.requested_transitions, 250_000)
        self.assertEqual(stage.actual_checkpoint_step, 200_000)
        self.assertIsNotNone(stage.substitution_reason)
        self.assertIn("250K", stage.substitution_reason or "")
        inventory = config.raw["checkpoint_inventory"]
        self.assertEqual(
            [candidate["step"] for candidate in inventory["target_250k_candidates"]],
            [200_000, 300_000],
        )
        self.assertEqual(inventory["selected_step"], 200_000)

    def test_manifest_preserves_real_time_video_timing(self) -> None:
        config = load_gameplay_progression_config()
        contract = load_evaluation_contract(config.contract_path)

        self.assertEqual(config.video_fps, 30)
        self.assertEqual(config.frame_repeat, 2)
        self.assertEqual(config.capture_every_agent_step, 1)
        self.assertAlmostEqual(
            config.frame_size_seconds_per_agent_step,
            1 / 15,
        )
        self.assertEqual(
            config.frame_repeat / config.video_fps,
            contract.frame_skip / config.native_atari_fps,
        )

    def test_manifest_has_random_baseline_and_unique_ordered_outputs(self) -> None:
        config = load_gameplay_progression_config()

        self.assertEqual(config.stages[0].policy, "random")
        self.assertIsNone(config.stages[0].checkpoint)
        self.assertEqual(
            [stage.ordinal for stage in config.stages],
            list(range(7)),
        )
        self.assertEqual(len({stage.output for stage in config.stages}), 7)

    def test_action_mapping_matches_atari_contract(self) -> None:
        self.assertEqual(
            EXPECTED_ACTION_MAPPING,
            {"0": "NOOP", "1": "FIRE", "2": "RIGHT", "3": "LEFT"},
        )


if __name__ == "__main__":
    unittest.main()
