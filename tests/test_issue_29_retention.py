"""Pure boundary tests for the frozen Issue 29 paired survival metric."""

from __future__ import annotations

import unittest

from scripts.analysis.compare_vision_dropout_retention import (
    classify_result,
    controller_options_for_arm,
    paired_survival_gain_seed_count,
    validate_arms,
)


def _episodes(deltas: tuple[int, int, int]) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    for seed, delta in zip((101, 202, 303), deltas):
        episodes.extend(
            [
                {"max_missing_frames": 4, "episode_seed": seed, "game_over_ale_frame": 4000, "stop_reason": "game_over"},
                {"max_missing_frames": 12, "episode_seed": seed, "game_over_ale_frame": 4000 + delta, "stop_reason": "game_over"},
            ]
        )
    return episodes


class Issue29RetentionTests(unittest.TestCase):
    def test_only_frozen_arm_values_are_accepted(self) -> None:
        self.assertEqual(validate_arms(4, 12), (4, 12))
        for arms in ((4, 11), (3, 12), (12, 4)):
            with self.subTest(arms=arms), self.assertRaises(ValueError):
                validate_arms(*arms)
        with self.assertRaises(TypeError):
            validate_arms(True, 12)
        with self.assertRaises(TypeError):
            validate_arms(4.0, 12)

    def test_arm_wiring_changes_only_max_missing_frames(self) -> None:
        baseline = {"deadband": 3.0, "hysteresis": 1.0, "max_missing_frames": 4, "max_ball_speed_pixels_per_frame": 8.0}
        candidate = controller_options_for_arm(baseline, 12)
        self.assertEqual(candidate, {**baseline, "max_missing_frames": 12})
        self.assertEqual(baseline["max_missing_frames"], 4)
        with self.assertRaises(ValueError):
            controller_options_for_arm({**baseline, "max_missing_frames": 5}, 12)

    def test_paired_gain_threshold_is_inclusive_and_requires_game_over(self) -> None:
        episodes = _episodes((500, 499, 2000))
        self.assertEqual(paired_survival_gain_seed_count(episodes), 2)
        episodes[-1]["stop_reason"] = "step_cap"
        self.assertEqual(paired_survival_gain_seed_count(episodes), 1)

    def test_promotion_rejection_and_one_pair_inconclusive_boundaries(self) -> None:
        promoted, count, deltas = classify_result(_episodes((500, 501, -1000)), provenance_valid=True)
        self.assertEqual((promoted, count), ("PROMOTED", 2))
        self.assertEqual(deltas, {"101": 500, "202": 501, "303": -1000})
        rejected, count, _ = classify_result(_episodes((499, 0, -1000)), provenance_valid=True)
        self.assertEqual((rejected, count), ("REJECTED", 0))
        inconclusive, count, _ = classify_result(_episodes((500, 0, -1000)), provenance_valid=True)
        self.assertEqual((inconclusive, count), ("INCONCLUSIVE", 1))

    def test_inconclusive_on_large_regression_invalid_provenance_or_missing_episode(self) -> None:
        episodes = _episodes((500, 500, -1001))
        self.assertEqual(classify_result(episodes, provenance_valid=True)[0], "INCONCLUSIVE")
        self.assertEqual(classify_result(_episodes((500, 500, 500)), provenance_valid=False)[0], "INCONCLUSIVE")
        self.assertEqual(classify_result(_episodes((500, 500, 500))[:-1], provenance_valid=True)[0], "INCONCLUSIVE")

    def test_duplicate_arm_seed_is_rejected(self) -> None:
        episodes = _episodes((0, 0, 0))
        episodes.append(episodes[0])
        with self.assertRaises(ValueError):
            paired_survival_gain_seed_count(episodes)


if __name__ == "__main__":
    unittest.main()
