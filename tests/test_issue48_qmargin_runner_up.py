from __future__ import annotations

import unittest

import numpy as np

from breakout_rl.evaluation_artifacts import VERIFIED_CLEAR_PROVENANCE_FIELDS
from breakout_rl.completion import BREAKOUT_COMPLETION_SOURCE
from breakout_rl.issue48_qmargin_runner_up import (
    ACTION_MEANINGS, CALIBRATION_SHA256, COLLECTION_WALL_LIMIT, CONTRACT_SHA256,
    FRAME_LIMIT, MODEL_SHA256, SCHEDULE, SEEDS, STEP_LIMIT, THRESHOLD,
    TOTAL_FRAME_LIMIT, aggregate_cap_reached, classify_round, rank_actions, select_action,
    select_candidate_action, select_greedy_action, stop_after_episode,
    verify_provenance,
)


class QMarginSelectorTests(unittest.TestCase):
    def test_candidate_uses_runner_up_strictly_below_threshold(self):
        q = np.array([[0.5, 0.5 - np.float32(THRESHOLD / 2), -1.0, -2.0]], dtype=np.float32)
        action, order, margin = select_action(q, candidate=True)
        self.assertEqual(order[:2], (0, 1))
        self.assertLess(margin, THRESHOLD)
        self.assertEqual(action, 1)

    def test_equality_and_above_threshold_keep_greedy_action(self):
        equal = np.array([[np.float32(THRESHOLD), 0.0, -1.0, -2.0]], dtype=np.float32)
        above = np.array([[np.float32(THRESHOLD + 1e-4), 0.0, -1.0, -2.0]], dtype=np.float32)
        self.assertEqual(select_action(equal, candidate=True)[0], select_greedy_action(equal))
        self.assertEqual(select_action(above, candidate=True)[0], select_greedy_action(above))

    def test_stable_ties_use_ascending_action_index(self):
        q = np.array([[2.0, 2.0, 1.0, 2.0]], dtype=np.float32)
        order, margin = rank_actions(q)
        self.assertEqual(order, (0, 1, 3, 2))
        self.assertEqual(margin, 0.0)
        self.assertEqual(select_greedy_action(q), 0)
        self.assertEqual(select_candidate_action(q), 1)

    def test_selection_is_deterministic_and_baseline_remains_raw_argmax(self):
        q = np.array([[0.1, 0.2, 0.2005, 0.0]], dtype=np.float32)
        self.assertEqual(select_action(q, candidate=True), select_action(q, candidate=True))
        self.assertEqual(select_action(q, candidate=False)[0], int(np.argmax(q[0])))
        self.assertEqual(select_greedy_action(q), 2)
        self.assertEqual(ACTION_MEANINGS, ("NOOP", "FIRE", "RIGHT", "LEFT"))


class FrozenScheduleAndProvenanceTests(unittest.TestCase):
    def test_classification_is_separate_from_goal_reached_round_status(self):
        self.assertEqual(classify_round(verified_clear_arm="candidate", complete_schedule=False),
            {"classification": "PROMOTED", "round_status": "GOAL_REACHED",
             "candidate_hypothesis_status": "PROMOTED"})
        self.assertEqual(classify_round(verified_clear_arm="baseline", complete_schedule=False),
            {"classification": "INCONCLUSIVE", "round_status": "GOAL_REACHED",
             "candidate_hypothesis_status": "NOT_ADJUDICATED"})
        self.assertEqual(classify_round(verified_clear_arm=None, complete_schedule=True),
            {"classification": "INCONCLUSIVE", "round_status": "CONTINUE_RESEARCH",
             "candidate_hypothesis_status": "INCONCLUSIVE"})
        self.assertEqual(classify_round(verified_clear_arm=None, complete_schedule=False)["classification"],
                         "INCONCLUSIVE")

    def test_schedule_order_and_budgets_are_frozen(self):
        self.assertEqual(SEEDS, (104, 205, 306))
        self.assertEqual(SCHEDULE, ((104, "baseline"), (104, "candidate"),
                                    (205, "baseline"), (205, "candidate"),
                                    (306, "baseline"), (306, "candidate")))
        self.assertEqual((FRAME_LIMIT, STEP_LIMIT, TOTAL_FRAME_LIMIT, COLLECTION_WALL_LIMIT),
                         (108_000, 27_000, 648_000, 560.0))
        self.assertEqual(THRESHOLD, 0.0027604103088378906)
        self.assertEqual(CALIBRATION_SHA256, "5d89f8bc30fb1eba2acb21dc8355160addf0c17938ab054f310703401d5043dc")

    def test_caps_and_verified_clear_stop_behavior(self):
        self.assertFalse(aggregate_cap_reached(559.99, 647_996))
        self.assertTrue(aggregate_cap_reached(560.0, 1))
        self.assertTrue(aggregate_cap_reached(0.0, 648_000))
        self.assertFalse(stop_after_episode(verified_clear=False, elapsed_seconds=1, native_frames=4))
        self.assertTrue(stop_after_episode(verified_clear=True, elapsed_seconds=1, native_frames=4))
        self.assertTrue(stop_after_episode(verified_clear=False, elapsed_seconds=1, native_frames=648_000))

    def test_provenance_gate_requires_complete_canonical_identity(self):
        fields = {field: "present" for field in VERIFIED_CLEAR_PROVENANCE_FIELDS}
        fields.update({"checkpoint_id": MODEL_SHA256, "training_seed": 2022,
            "training_transition_count": 2_500_000, "contract_id": "day15-breakout-evaluation-v2-fire-reset",
            "contract_sha256": CONTRACT_SHA256, "source_working_tree_dirty": False,
            "completion_detector_id": "ale-breakout-two-wall-score-864-v2",
            "contract_validation_status": "canonical_contract_v2", "source_commit": "a" * 40,
            "completion_source_sha256": "b" * 64,
            "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
            "evaluation_seed": 104, "episode_seed": 104, "episode_index": 1,
            "clear_score": 864.0, "raw_score": 864.0})
        args = dict(commit="a" * 40, digest="b" * 64, seed=104, episode_index=1)
        self.assertEqual(verify_provenance(fields, **args), (True, []))
        incomplete = dict(fields, clear_emulator_frame=None)
        valid, missing = verify_provenance(incomplete, **args)
        self.assertFalse(valid)
        self.assertIn("clear_emulator_frame", missing)


if __name__ == "__main__":
    unittest.main()
