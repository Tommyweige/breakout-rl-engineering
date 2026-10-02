from __future__ import annotations

import unittest
import json

import numpy as np

from breakout_rl.issue58_pixel_paddle_alignment import (
    ACTION_MEANINGS, BALL_ROI, BRIGHT_THRESHOLD, CALIBRATION_SHA256,
    CALIBRATION_INDEX_SHA256, CALIBRATION_RESULTS_SHA256, CALIBRATION_TRAJECTORY_SHA256,
    FRAME_LIMIT, PADDLE_ROI, SCHEDULE, SEEDS, STEP_LIMIT, TOTAL_FRAME_LIMIT, TOTAL_STEP_LIMIT,
    detect_ball_and_paddle, detect_object, select_action, select_greedy_action,
    classify_round, evaluation_status, scan_calibration, clear_stop_outcome,
    continue_after_episode, stack_capture_reasons, preflight, frozen_calibration_record,
)
import breakout_rl.issue58_pixel_paddle_alignment as issue58
from unittest.mock import patch


def obs() -> np.ndarray:
    return np.zeros((4, 84, 84), dtype=np.uint8)


def valid_scene(delta: int = 0) -> np.ndarray:
    image = obs()
    paddle_x = 40
    ball_x = paddle_x - 1 + delta  # paddle center is floor((37 + 42) / 2) == 39
    image[3, 76:78, paddle_x - 3:paddle_x + 3] = 255
    image[3, 60, ball_x:ball_x + 2] = 255
    return image


class PixelDetectorTests(unittest.TestCase):
    def test_frozen_rois_threshold_and_channel(self):
        image = obs()
        self.assertEqual((BALL_ROI, PADDLE_ROI, BRIGHT_THRESHOLD), ((54, 73, 6, 78), (73, 83, 6, 78), 80))
        image[2, 60, 20] = 255  # detector must inspect channel 3 only
        image[3, 53, 20] = 255  # outside ball ROI
        image[3, 60, 20] = 79
        det = detect_ball_and_paddle(image)
        self.assertEqual(det["ball"]["component_count"], 0)
        image[3, 60, 20] = 79
        self.assertEqual(detect_ball_and_paddle(image)["ball"]["component_count"], 0)
        image[3, 60, 20] = 80
        self.assertEqual(detect_ball_and_paddle(image)["ball"]["valid"]["center_x"], 20)

    def test_half_open_roi_endpoints_for_ball_and_paddle(self):
        for y, x in ((54, 6), (72, 77)):
            image = obs(); image[3, y, x] = 255
            self.assertEqual(detect_ball_and_paddle(image)["ball"]["status"], "unique")
        for y, x in ((53, 20), (73, 20), (60, 5), (60, 78)):
            image = obs(); image[3, y, x] = 255
            self.assertEqual(detect_ball_and_paddle(image)["ball"]["status"], "absent")
        for y, x0 in ((73, 6), (82, 73)):
            image = obs(); image[3, y, x0:x0 + 5] = 255
            self.assertEqual(detect_ball_and_paddle(image)["paddle"]["status"], "unique")
        for y, x0 in ((72, 20), (83, 20), (76, 5), (76, 78)):
            image = obs(); image[3, y, x0:x0 + 5] = 255
            self.assertEqual(detect_ball_and_paddle(image)["paddle"]["status"], "absent")

    def test_8_connectivity_and_valid_component_geometry(self):
        image = obs()
        image[3, 60, 20] = 255
        image[3, 61, 21] = 255  # diagonal pixels form one component
        result = detect_ball_and_paddle(image)["ball"]
        self.assertEqual(result["component_count"], 1)
        self.assertEqual(result["valid"]["area"], 2)
        self.assertEqual((result["valid"]["width"], result["valid"]["height"]), (2, 2))
        self.assertEqual(result["valid"]["center_x"], 20)

    def test_component_area_and_inclusive_shape_limits(self):
        image = obs()
        image[3, 60:63, 20:23] = 255
        image[3, 60, 20] = 0  # area 8 with inclusive 3x3 bounds is valid
        ball = detect_ball_and_paddle(image)["ball"]
        self.assertEqual((ball["valid"]["area"], ball["valid"]["width"], ball["valid"]["height"]), (8, 3, 3))
        image[3, 60, 20] = 255  # area 9 exceeds ball max
        self.assertEqual(detect_ball_and_paddle(image)["ball"]["status"], "absent")
        image = obs(); image[3, 60, 20:24] = 255  # ball width 4
        self.assertEqual(detect_ball_and_paddle(image)["ball"]["status"], "absent")
        image = obs(); image[3, 60:64, 20] = 255  # ball height 4
        self.assertEqual(detect_ball_and_paddle(image)["ball"]["status"], "absent")
        image = obs(); image[3, 76, 20:25] = 255  # minimum valid paddle: area 5, width 5, height 1
        self.assertEqual(detect_ball_and_paddle(image)["paddle"]["valid"]["area"], 5)
        image = obs(); image[3, 76:80, 20:36] = 255  # maximum valid paddle: 64 pixels, 16 x 4
        self.assertEqual(detect_ball_and_paddle(image)["paddle"]["valid"]["area"], 64)
        image = obs(); image[3, 76, 20:24] = 255  # width 4 is too small
        self.assertEqual(detect_ball_and_paddle(image)["paddle"]["status"], "absent")
        image = obs(); image[3, 75:80, 20:25] = 255  # height 5 is too large
        self.assertEqual(detect_ball_and_paddle(image)["paddle"]["status"], "absent")

    def test_absent_and_ambiguous_detection(self):
        image = obs()
        self.assertEqual(detect_ball_and_paddle(image)["ball"]["status"], "absent")
        image[3, 60, 20] = 255
        image[3, 60, 40] = 255
        self.assertEqual(detect_ball_and_paddle(image)["ball"]["status"], "ambiguous")
        image[3, 77, 30:36] = 255
        image[3, 77, 50:56] = 255
        self.assertEqual(detect_ball_and_paddle(image)["paddle"]["status"], "ambiguous")

    def test_inclusive_bounds_even_width_paddle_and_integer_center(self):
        image = obs()
        image[3, 76, 30:36] = 255
        paddle = detect_ball_and_paddle(image)["paddle"]["valid"]
        self.assertEqual((paddle["xmin"], paddle["xmax"], paddle["width"]), (30, 35, 6))
        self.assertEqual((paddle["ymin"], paddle["ymax"], paddle["height"]), (76, 76, 1))
        self.assertEqual(paddle["center_x"], 32)
        component = detect_object(np.asarray([[1, 1]], dtype=np.uint8) * 255,
            roi=(0, 1, 0, 2), area=(2, 2), width=(2, 2), height=(1, 1))["valid"]
        self.assertEqual(component["width"], component["xmax"] - component["xmin"] + 1)
        self.assertEqual(component["center_x"], (component["xmin"] + component["xmax"]) // 2)


class ActionRuleTests(unittest.TestCase):
    def test_action_boundaries_and_greedy_fallback(self):
        q = np.array([[0.0, 0.1, 0.8, 0.2]], dtype=np.float32)
        self.assertEqual(select_greedy_action(q), 2)
        for delta, expected in ((2, 2), (-2, 3), (1, 0), (0, 0), (-1, 0)):
            chosen = select_action(q, detect_ball_and_paddle(valid_scene(delta)), candidate=True)
            self.assertEqual(chosen["delta"], delta)
            self.assertEqual(chosen["action"], expected)
            self.assertTrue(chosen["override_used"])
        missing = select_action(q, detect_ball_and_paddle(obs()), candidate=True)
        self.assertEqual(missing["action"], int(np.argmax(q[0])))
        self.assertFalse(missing["override_used"])
        baseline = select_action(q, detect_ball_and_paddle(valid_scene(-20)), candidate=False)
        self.assertEqual(baseline["action"], int(np.argmax(q[0])))
        self.assertFalse(baseline["override_used"])
        ambiguous_ball = obs()
        ambiguous_ball[3, 60, 20] = ambiguous_ball[3, 60, 40] = 255
        ambiguous_ball[3, 77, 30:36] = 255
        ambiguous_paddle = valid_scene(8)
        ambiguous_paddle[3, 77, 60:66] = 255
        for image in (ambiguous_ball, ambiguous_paddle):
            fallback = select_action(q, detect_ball_and_paddle(image), candidate=True)
            self.assertEqual(fallback["action"], int(np.argmax(q[0])))
            self.assertFalse(fallback["override_used"])

    def test_schedule_caps_and_calibration_is_provenance_only(self):
        self.assertEqual(SEEDS, (510, 511, 512))
        self.assertEqual(SCHEDULE, ((510, "baseline"), (510, "candidate"), (511, "baseline"),
            (511, "candidate"), (512, "baseline"), (512, "candidate")))
        self.assertEqual((FRAME_LIMIT, STEP_LIMIT, TOTAL_FRAME_LIMIT, TOTAL_STEP_LIMIT), (108000, 27000, 648000, 162000))
        self.assertEqual(CALIBRATION_SHA256, "757eab8a54327e0663e2d83e5156958b776c1c5e7a09521a3355a9af43746978")
        self.assertEqual(CALIBRATION_INDEX_SHA256, CALIBRATION_SHA256)
        self.assertEqual(CALIBRATION_RESULTS_SHA256, "6954370d9270b154de55c38c26d107b25347a82d3ecf86bc723cdf03b68cf582")
        self.assertEqual(CALIBRATION_TRAJECTORY_SHA256, "48c3ad056962862ba8a80219909caa6f4349a704642cef65b3210f62b47a2398")
        self.assertEqual(ACTION_MEANINGS, ("NOOP", "FIRE", "RIGHT", "LEFT"))

    def test_pinned_calibration_hashes_and_frozen_threshold_selection(self):
        result = scan_calibration()
        self.assertEqual((result["indexed_rows"], result["unique_pixel_observations"]), (300, 150))
        self.assertEqual(result["selected_threshold"], 80)
        self.assertEqual(result["activation_counts_by_threshold"], {
            80: 64, 90: 52, 95: 50, 100: 42, 105: 12, 110: 2,
            115: 0, 120: 0, 130: 0, 148: 0, 200: 0})
        self.assertEqual(result["candidate_activation_at_selected_threshold"],
            "64/300 indexed rows = 32/150 unique observations")
        self.assertFalse(result["activation_is_object_truth"])
        self.assertEqual(json.loads(json.dumps(result)), json.loads(json.dumps(frozen_calibration_record())))

    def test_hvc_classification_and_partial_cap_status(self):
        self.assertEqual(classify_round("candidate"), {
            "classification": "GOAL_REACHED", "round_status": "GOAL_REACHED",
            "candidate_hypothesis_status": "GOAL_REACHED", "hvc_status": "YES"})
        self.assertEqual(classify_round("baseline"), {
            "classification": "GOAL_REACHED", "round_status": "GOAL_REACHED",
            "candidate_hypothesis_status": "NOT_ADJUDICATED", "hvc_status": "YES"})
        self.assertEqual(classify_round(None), {
            "classification": "INCONCLUSIVE", "round_status": "CONTINUE_RESEARCH",
            "candidate_hypothesis_status": "INCONCLUSIVE", "hvc_status": "NO"})
        self.assertEqual(clear_stop_outcome(True, True, "baseline")["stop"], True)
        unverified = clear_stop_outcome(True, False, "baseline")
        self.assertEqual((unverified["stop"], unverified["classification"], unverified["round_status"]),
            (True, "HUMAN_REVIEW_REQUIRED", "HUMAN_REVIEW_REQUIRED"))
        self.assertFalse(clear_stop_outcome(False, False, None)["stop"])
        for provenance_complete in (True, False):
            executed = []
            for episode in SCHEDULE:
                executed.append(episode)
                decision = clear_stop_outcome(True, provenance_complete, episode[1])
                if not continue_after_episode(decision["stop"]):
                    break
            self.assertEqual(executed, [SCHEDULE[0]])
        self.assertIn("canonical_clear", stack_capture_reasons(
            override=False, life_loss=False, canonical_clear=True, life_window_active=False))

    def test_preflight_is_zero_frame_and_does_not_construct_ale(self):
        with patch.object(issue58, "make_breakout_env", side_effect=AssertionError("ALE env constructed")):
            result = preflight()
        self.assertEqual(result["native_frames"], 0)
        self.assertFalse(result["ale_environment_created"])
        self.assertEqual(evaluation_status(6, False), "completed")
        self.assertEqual(evaluation_status(2, False), "incomplete_run")
        self.assertEqual(evaluation_status(2, True), "completed")


if __name__ == "__main__":
    unittest.main()
