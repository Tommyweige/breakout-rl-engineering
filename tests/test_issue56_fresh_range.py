from __future__ import annotations

import unittest

import numpy as np

from breakout_rl.issue56_fresh_range import (
    BALL_ROI, COMPLETION_SOURCE_FILES, PADDLE_ROI, ROOT, quantile_summary,
    classify_clear_state, classify_diagnostic, observation_summaries, roi_slice,
    should_stop_after_clear, source_digest, threshold_reached, setup_test_budget,
    verify_stack_index,
)
from breakout_rl.issue41_onnx_dqn_clear import (
    COMPLETION_SOURCE_FILES as ISSUE41_COMPLETION_SOURCE_FILES,
    completion_source_digest as issue41_completion_source_digest,
)


class GrayscaleRangePureTests(unittest.TestCase):
    def test_frozen_roi_half_open_boundaries(self):
        plane = np.zeros((84, 84), dtype=np.uint8)
        plane[54, 6] = 200
        self.assertEqual(int(roi_slice(plane, BALL_ROI)[0, 0]), 200)
        plane[73, 6] = 201
        self.assertEqual(int(roi_slice(plane, PADDLE_ROI)[0, 0]), 201)
        plane[83, 6] = 255
        self.assertNotEqual(int(roi_slice(plane, PADDLE_ROI)[-1, 0]), 255)
        plane[72, 78] = 255
        self.assertNotEqual(int(roi_slice(plane, BALL_ROI)[-1, -1]), 255)

    def test_reachability_threshold_counts_and_fraction(self):
        planes = [np.zeros((84, 84), dtype=np.uint8) for _ in range(4)]
        planes[1][54, 6] = 200
        planes[2][73, 77] = 255
        self.assertEqual([threshold_reached(p) for p in planes], [False, True, True, False])
        self.assertEqual(sum(threshold_reached(p) for p in planes) / len(planes), 0.5)
        self.assertFalse(threshold_reached(np.full((84, 84), 199, dtype=np.uint8)))

    def test_observation_summaries_use_channel_three_only(self):
        observation = np.zeros((4, 84, 84), dtype=np.uint8)
        observation[0:3] = 255
        observation[3] = 17
        summary = observation_summaries(observation)
        self.assertEqual(summary["channel_3_full_plane"]["max"], 17)
        self.assertEqual(summary["channel_3_ball_roi"]["max"], 17)
        self.assertEqual(summary["channel_3_paddle_roi"]["max"], 17)

    def test_quantiles_are_linear_and_fixed(self):
        summary = quantile_summary(np.arange(100, dtype=np.uint8))
        self.assertEqual((summary["min"], summary["max"]), (0, 99))
        self.assertEqual((summary["p50"], summary["p90"], summary["p95"], summary["p99"]), (49.5, 89.10000000000001, 94.05, 98.01))
        self.assertEqual(summary["quantile_method"], "linear")

    def test_stack_index_binds_exact_stream_offsets_hashes(self):
        stack = np.arange(4 * 84 * 84, dtype=np.uint8).tobytes()
        import hashlib
        rows = [{"byte_offset": 0, "byte_length": len(stack), "stack_sha256": hashlib.sha256(stack).hexdigest()}]
        self.assertTrue(verify_stack_index(rows, stack))
        rows[0]["byte_offset"] = 1
        self.assertFalse(verify_stack_index(rows, stack))

    def test_zero_frame_preflight_contract_is_explicit(self):
        # Runtime preflight constructs the canonical env and inspects its support only;
        # it never resets or steps ALE. This pure assertion guards the declared contract.
        from breakout_rl.issue56_fresh_range import SEED, FRAME_CAP, DECISION_CAP
        self.assertEqual((SEED, FRAME_CAP, DECISION_CAP), (509, 350, 87))

    def test_diagnostic_classification_requires_complete_valid_sample(self):
        self.assertEqual(classify_diagnostic(n=0, reach_count=0,
            stop_reason="native_frame_cap", capture_valid=True), "INCONCLUSIVE")
        self.assertEqual(classify_diagnostic(n=12, reach_count=0,
            stop_reason="collection_wall_cap", capture_valid=True), "INCONCLUSIVE")
        self.assertEqual(classify_diagnostic(n=12, reach_count=0,
            stop_reason="terminated", capture_valid=False), "INCONCLUSIVE")
        self.assertEqual(classify_diagnostic(n=87, reach_count=0,
            stop_reason="decision_cap", capture_valid=True), "PROMOTED")
        self.assertEqual(classify_diagnostic(n=87, reach_count=1,
            stop_reason="decision_cap", capture_valid=True), "REJECTED")

    def test_any_canonical_clear_triggers_immediate_stop(self):
        self.assertFalse(should_stop_after_clear(False))
        # Provenance is checked after this signal; failure still cannot extend the run.
        self.assertTrue(should_stop_after_clear(True))

    def test_completion_provenance_digest_matches_canonical_issue41_algorithm(self):
        self.assertEqual(COMPLETION_SOURCE_FILES, ISSUE41_COMPLETION_SOURCE_FILES)
        self.assertEqual(source_digest(ROOT, COMPLETION_SOURCE_FILES), issue41_completion_source_digest(ROOT))

    def test_canonical_clear_with_incomplete_provenance_requires_human_review(self):
        self.assertEqual(classify_clear_state(False, False), {
            "classification": "INCONCLUSIVE", "round_status": "CONTINUE_RESEARCH"})
        self.assertEqual(classify_clear_state(True, True), {
            "classification": "INCONCLUSIVE", "round_status": "GOAL_REACHED"})
        self.assertEqual(classify_clear_state(True, False), {
            "classification": "INCONCLUSIVE", "round_status": "HUMAN_REVIEW_REQUIRED"})
        with self.assertRaises(ValueError):
            classify_clear_state(False, True)

    def test_combined_setup_and_test_budget_is_enforced(self):
        validation = {"overall_pre_run_wall_seconds": 0.4}
        budget = setup_test_budget(1.0, validation)
        self.assertEqual(budget["cap"], 20.0)
        self.assertAlmostEqual(budget["pre_run_consumed"], 0.4)
        self.assertAlmostEqual(budget["total_consumed"], 1.4)
        self.assertAlmostEqual(budget["remaining"], 18.6)


if __name__ == "__main__":
    unittest.main()
