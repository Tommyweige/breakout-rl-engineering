from __future__ import annotations

import unittest

import numpy as np

from breakout_rl.issue43_rolling_q_survival import (
    ARM_ORDER,
    FRAME_LIMIT,
    CONTRACT_SHA256,
    MODEL_SHA256,
    SEEDS,
    SMOOTHING_WINDOW,
    clear_provenance_check,
    paired_clear_statuses,
    smooth_q_history,
)
from breakout_rl.completion import BREAKOUT_COMPLETION_DETECTOR_ID, BREAKOUT_COMPLETION_SOURCE
from breakout_rl.evaluation_artifacts import VERIFIED_CLEAR_PROVENANCE_FIELDS
from breakout_rl.evaluation_contract import BREAKOUT_CONTRACT_V2_ID


class RollingQTests(unittest.TestCase):
    def test_first_decision_uses_current_vector(self):
        history: list[np.ndarray] = []
        current = np.asarray([[1, 2, 3, 4]], dtype=np.float32)
        result = smooth_q_history(history, current)
        np.testing.assert_array_equal(result, current)
        self.assertEqual(result.dtype, np.float32)

    def test_window_averages_current_and_three_previous_vectors(self):
        history: list[np.ndarray] = []
        outputs = [smooth_q_history(history, np.full((1, 4), value, dtype=np.float32))
                   for value in (0, 4, 8, 12, 16)]
        np.testing.assert_array_equal(outputs[1], np.full((1, 4), 2, dtype=np.float32))
        np.testing.assert_array_equal(outputs[3], np.full((1, 4), 6, dtype=np.float32))
        np.testing.assert_array_equal(outputs[4], np.full((1, 4), 10, dtype=np.float32))
        self.assertEqual(len(history), SMOOTHING_WINDOW)

    def test_frozen_episode_and_arm_order(self):
        self.assertEqual(SEEDS, (102, 203, 304))
        self.assertEqual(ARM_ORDER, ("raw-greedy", "rolling-q4"))
        self.assertEqual(FRAME_LIMIT, 108_000)

    def test_paired_report_preserves_both_arm_clear_statuses(self):
        completed = {"raw_greedy_clear_status": "NO_CLEAR",
                     "rolling_q4_clear_status": "CANONICAL_CLEAR_UNVERIFIED"}
        self.assertEqual(paired_clear_statuses(completed),
            ("NO_CLEAR", "CANONICAL_CLEAR_UNVERIFIED"))
        incomplete = {"raw_greedy": {"clear_status": "VERIFIED_CLEAR"},
                      "rolling_q4": None,
                      "raw_greedy_clear_status": "VERIFIED_CLEAR",
                      "rolling_q4_clear_status": "NOT_RUN"}
        self.assertEqual(paired_clear_statuses(incomplete), ("VERIFIED_CLEAR", "NOT_RUN"))

    def test_rejects_invalid_q_contract(self):
        history: list[np.ndarray] = []
        with self.assertRaises(ValueError):
            smooth_q_history(history, np.zeros((1, 4), dtype=np.float64))
        with self.assertRaises(ValueError):
            smooth_q_history(history, np.asarray([[0, 1, np.nan, 3]], dtype=np.float32))

    def test_verified_clear_gate_binds_full_provenance(self):
        commit, digest, seed, index = "a" * 40, "b" * 64, 102, 1
        provenance = {field: "present" for field in VERIFIED_CLEAR_PROVENANCE_FIELDS}
        provenance.update({
            "checkpoint_id": MODEL_SHA256, "training_seed": 2022,
            "training_transition_count": 2_500_000,
            "evaluation_seed": seed, "episode_seed": seed, "episode_index": index,
            "contract_id": BREAKOUT_CONTRACT_V2_ID, "contract_sha256": CONTRACT_SHA256,
            "source_commit": commit, "source_working_tree_dirty": False,
            "completion_source_sha256": digest, "raw_score": 864.0, "clear_score": 864.0,
            "clear_agent_step": 1, "clear_emulator_frame": 4, "lives_remaining_at_clear": 3,
            "completion_detection_source": BREAKOUT_COMPLETION_SOURCE,
            "completion_detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
            "contract_validation_status": "canonical_contract_v2",
        })
        self.assertEqual(clear_provenance_check(provenance, commit=commit, digest=digest,
            seed=seed, episode_index=index), (True, []))
        for key, wrong in (
            ("source_commit", "c" * 40),
            ("completion_source_sha256", "d" * 64),
            ("episode_seed", 203),
            ("evaluation_seed", 203),
            ("episode_index", 2),
            ("completion_detector_id", "wrong-detector"),
            ("completion_detection_source", "video_inspection"),
            ("clear_emulator_frame", None),
        ):
            altered = dict(provenance)
            altered[key] = wrong
            valid, missing = clear_provenance_check(altered, commit=commit, digest=digest,
                seed=seed, episode_index=index)
            self.assertFalse(valid, key)
            self.assertIn(key, missing)
        invalid_seed_index = dict(provenance, episode_index=3)
        self.assertFalse(clear_provenance_check(invalid_seed_index, commit=commit,
            digest=digest, seed=seed, episode_index=3)[0])


if __name__ == "__main__":
    unittest.main()
