"""Focused tests for the frozen Issue #37 reachable paddle-contact diagnostic."""
from __future__ import annotations
import inspect
import json
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from breakout_rl.issue37_reachable_contact_probe import (
    EXPECTED_ORDER, _process_contact_event, analyze_probe_runs, clamp_target_center,
    detect_post_bounce_collision, load_probe_config, select_outgoing_vx,
)
from breakout_rl.vision_controller import (
    BallEstimate, PaddleDetection, PlayfieldBounds, PredictiveBreakoutController,
    VisionObservation, choose_paddle_action, predict_paddle_intercept,
)


def row(*, frame=10, direct=True, vy=2.0, vx=1.0, horizon=2.0, y=185.0, top=189.0,
        intercept=80.0, paddle=75.0):
    return {"ale_emulator_frame": frame, "ball_directly_detected": direct, "ball_vy": vy,
            "ball_vx": vx, "time_to_intercept_frames": horizon, "ball_y": y,
            "paddle_top": top, "paddle_left": 68.0, "paddle_right": 84.0,
            "paddle_center_x": paddle, "predicted_intercept_x": intercept}


class ProbeTargetTests(unittest.TestCase):
    def test_offset_sign_and_legal_center_clamping(self):
        self.assertEqual(clamp_target_center(100, -6, 15.5, 143.5), 106)
        self.assertEqual(clamp_target_center(100, 6, 15.5, 143.5), 94)
        self.assertEqual(clamp_target_center(10, 6, 15.5, 143.5), 15.5)
        self.assertEqual(clamp_target_center(150, -6, 15.5, 143.5), 143.5)

    def test_zero_offset_method_matches_default_rgb_controller(self):
        frame = np.zeros((210, 160, 3), dtype=np.uint8)
        frame[17:193, :8] = 142; frame[17:193, 152:] = 142; frame[17:32, :] = 142
        frame[189:193, 72:88] = (200, 72, 72); frame[120:124, 80:82] = (200, 72, 72)
        baseline = PredictiveBreakoutController().select_action(frame)
        probe = PredictiveBreakoutController().select_action_with_target_offset(frame, target_offset_px=0)
        self.assertEqual((baseline.action, baseline.predicted_intercept_x, baseline.paddle_error),
                         (probe.action, probe.predicted_intercept_x, probe.paddle_error))

    def test_nonzero_offset_moves_target_in_the_declared_sign(self):
        def frame(ball_x, ball_y):
            image = np.zeros((210, 160, 3), dtype=np.uint8)
            image[17:193, :8] = 142; image[17:193, 152:] = 142; image[17:32, :] = 142
            image[189:193, 72:88] = (200, 72, 72)
            image[ball_y:ball_y + 4, ball_x:ball_x + 2] = (200, 72, 72)
            return image
        decisions = []
        for offset in (0, -6, 6):
            controller = PredictiveBreakoutController()
            controller.select_action_with_target_offset(frame(50, 120), target_offset_px=offset)
            decisions.append(controller.select_action_with_target_offset(frame(51, 121), target_offset_px=offset))
        baseline, aim_right, aim_left = decisions
        self.assertIsNotNone(baseline.predicted_intercept_x)
        self.assertAlmostEqual(aim_right.paddle_error - baseline.paddle_error, 6.0)
        self.assertAlmostEqual(aim_left.paddle_error - baseline.paddle_error, -6.0)

    def test_probe_action_api_has_no_evaluator_arguments(self):
        signature = inspect.signature(PredictiveBreakoutController.select_action_with_target_offset)
        self.assertEqual(tuple(signature.parameters), ("self", "frame", "target_offset_px"))

    def test_zero_offset_preserves_unclamped_baseline_near_both_side_bounds(self):
        bounds = PlayfieldBounds(8.0, 151.0, 32.0, 192.0, 1.0)
        paddle = PaddleDetection(72.0, 87.0, 79.5, 189.0, 192.0, 1.0)

        class ObservationFixtureController(PredictiveBreakoutController):
            def __init__(self, observation):
                super().__init__()
                self.fixture_observation = observation

            def observe(self, _frame):
                return self.fixture_observation

        for side, ball_x in (("left", 10.0), ("right", 149.0)):
            with self.subTest(side=side):
                ball = BallEstimate(ball_x, 170.0, 0.0, 2.0, 1.0, 1, 0, True)
                observation = VisionObservation(1, bounds, paddle, ball, 1)
                controller = ObservationFixtureController(observation)
                predicted = predict_paddle_intercept(
                    ball_x=ball_x, ball_y=170.0, ball_vx=0.0, ball_vy=2.0,
                    paddle_plane_y=paddle.top - 2.0, bounds=bounds,
                )
                self.assertIsNotNone(predicted)
                intercept, _horizon = predicted
                legal_min, legal_max = bounds.left + 7.5, bounds.right - 7.5
                self.assertTrue(intercept < legal_min if side == "left" else intercept > legal_max)

                decision = controller.select_action_with_target_offset(object(), target_offset_px=0.0)
                expected_error = intercept - paddle.center_x
                expected_action = choose_paddle_action(
                    error=expected_error, previous_direction="NOOP", deadband=3.0, hysteresis=1.0,
                )
                self.assertEqual(decision.predicted_intercept_x, intercept)
                self.assertEqual(decision.paddle_error, expected_error)
                self.assertEqual(decision.action, expected_action)

    def test_nonzero_offset_preserves_baseline_when_no_descending_intercept_exists(self):
        bounds = PlayfieldBounds(8.0, 151.0, 32.0, 192.0, 1.0)
        paddle = PaddleDetection(72.0, 87.0, 79.5, 189.0, 192.0, 1.0)
        observation = VisionObservation(1, bounds, paddle, BallEstimate(80, 100, 1.0, -2.0, 1.0, 1, 0, True), 1)

        class ObservationFixtureController(PredictiveBreakoutController):
            def observe(self, _frame):
                return observation

        baseline = ObservationFixtureController().select_action(object())
        steered = ObservationFixtureController().select_action_with_target_offset(object(), target_offset_px=6)
        self.assertIsNone(steered.predicted_intercept_x)
        self.assertEqual((steered.action, steered.paddle_error), (baseline.action, baseline.paddle_error))


class ProbeContactTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_probe_config()

    def test_precontact_window_includes_zero_and_three_only(self):
        candidate, event = _process_contact_event(None, row(horizon=0), self.cfg, None)
        self.assertIsNotNone(candidate); self.assertIsNone(event)
        candidate, event = _process_contact_event(None, row(horizon=3), self.cfg, None)
        self.assertIsNotNone(candidate)
        candidate, event = _process_contact_event(None, row(horizon=3.001), self.cfg, None)
        self.assertIsNone(candidate)

    def test_precontact_candidate_must_overlap_paddle_plus_ball_radius(self):
        candidate, _ = _process_contact_event(None, row(intercept=86), self.cfg, None)
        self.assertIsNotNone(candidate)
        candidate, _ = _process_contact_event(None, row(intercept=86.001), self.cfg, None)
        self.assertIsNone(candidate)
        no_edges = row()
        no_edges.pop("paddle_left"); no_edges.pop("paddle_right")
        candidate, _ = _process_contact_event(None, no_edges, self.cfg, None)
        self.assertIsNone(candidate)

    def test_bounce_requires_direct_observation_and_contact_band(self):
        candidate = {"frame": 10, "observed_impact_offset_px": 5.0}
        _, event = _process_contact_event(None, row(frame=11, direct=False, vy=-1, y=189), self.cfg, candidate)
        self.assertIsNone(event)
        _, event = _process_contact_event(None, row(frame=11, vy=-1, y=194.001), self.cfg, candidate)
        self.assertIsNone(event)
        _, event = _process_contact_event(None, row(frame=11, vy=-1, y=194), self.cfg, candidate)
        self.assertEqual(event["observed_impact_offset_px"], 5.0)

    def test_bounce_search_window_includes_eight_native_frames(self):
        candidate = {"frame": 10, "observed_impact_offset_px": 5.0}
        _, event = _process_contact_event(None, row(frame=18, vy=-1, y=189), self.cfg, candidate)
        self.assertIsNotNone(event)
        _, event = _process_contact_event(None, row(frame=19, vy=-1, y=189), self.cfg, candidate)
        self.assertIsNone(event)

    def test_outgoing_velocity_uses_first_three_direct_ascending_samples(self):
        obs = [
            {"frame": 18, "vx": 9, "vy": -1, "direct": False},
            {"frame": 18, "vx": 1, "vy": -1, "direct": True},
            {"frame": 19, "vx": 3, "vy": -1, "direct": True},
            {"frame": 20, "vx": 2, "vy": -1, "direct": True},
            {"frame": 21, "vx": 99, "vy": -1, "direct": True},
        ]
        self.assertEqual(select_outgoing_vx(obs, 18), 2)
        self.assertIsNone(select_outgoing_vx(obs[:3], 18))
        self.assertIsNone(select_outgoing_vx(obs, 18, collision=True))
        self.assertIsNone(select_outgoing_vx(obs, 10))

    def test_collision_detection_checks_full_direct_post_bounce_track(self):
        ascending = {"frame": 20, "x": 80.0, "vx": 1.0, "vy": -2.0, "direct": True}
        descending_after_brick = {"frame": 22, "x": 82.0, "vx": 1.0, "vy": 2.0, "direct": True}
        self.assertTrue(detect_post_bounce_collision(
            ascending, descending_after_brick, left_bound=8.0, right_bound=151.0))
        nondirect = dict(descending_after_brick, direct=False)
        self.assertFalse(detect_post_bounce_collision(
            ascending, nondirect, left_bound=8.0, right_bound=151.0))
        wall = dict(ascending, x=11.0)
        self.assertTrue(detect_post_bounce_collision(
            None, wall, left_bound=8.0, right_bound=151.0))


class ProbeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_probe_config()

    def _runs(self, xs, ys):
        return [{"seed": 404, "requested_offset_px": o, "observed_impact_offset_px": x,
                 "outgoing_vx_px_per_frame": y} for o, x, y in zip((-6, 0, 6), xs, ys)]

    def test_slope_threshold_and_ordered_manipulation_span(self):
        at_threshold = analyze_probe_runs(self._runs([0, 4, 8], [0, .08, .16]), self.cfg)
        self.assertEqual(at_threshold["positive_response_seed_count"], 1)
        self.assertAlmostEqual(at_threshold["seed_triplets"][0]["ols_slope_px_per_frame_per_px"], .02)
        below = analyze_probe_runs(self._runs([0, 4, 8], [0, .079, .158]), self.cfg)
        self.assertEqual(below["positive_response_seed_count"], 0)
        wrong_order = analyze_probe_runs(self._runs([0, 8, 7], [0, .3, .4]), self.cfg)
        self.assertFalse(wrong_order["seed_triplets"][0]["positive_response"])
        self.assertFalse(wrong_order["seed_triplets"][0]["observed_offsets_strictly_ordered"])
        small_span = analyze_probe_runs(self._runs([0, 3, 7.999], [0, .3, .6]), self.cfg)
        self.assertFalse(small_span["seed_triplets"][0]["positive_response"])
        self.assertEqual(small_span["measurement_floor_seed_count"], 0)

    def test_classification_boundaries(self):
        promoted = []
        rejected = []
        one_positive = []
        for seed in (404, 505, 606):
            promoted += [{"seed": seed, "requested_offset_px": o, "observed_impact_offset_px": x,
                          "outgoing_vx_px_per_frame": x * .03} for o, x in zip((-6,0,6),(0,4,8))]
            rejected += [{"seed": seed, "requested_offset_px": o, "observed_impact_offset_px": x,
                         "outgoing_vx_px_per_frame": 0.0} for o, x in zip((-6,0,6),(0,4,8))]
            one_positive += [{"seed": seed, "requested_offset_px": o, "observed_impact_offset_px": x,
                              "outgoing_vx_px_per_frame": x * (.03 if seed == 404 else 0)} for o, x in zip((-6,0,6),(0,4,8))]
        self.assertEqual(analyze_probe_runs(promoted, self.cfg)["classification"], "PROMOTED")
        self.assertEqual(analyze_probe_runs(rejected, self.cfg)["classification"], "REJECTED")
        self.assertEqual(analyze_probe_runs(one_positive, self.cfg)["classification"], "INCONCLUSIVE")
        self.assertEqual(analyze_probe_runs(rejected[:3], self.cfg)["classification"], "INCONCLUSIVE")
        partial_positive = analyze_probe_runs(promoted[:6], self.cfg)
        self.assertTrue(partial_positive["positive_response_seed_count"] >= 2)
        self.assertEqual(partial_positive["classification"], "INCONCLUSIVE")

    def test_config_hash_seeds_arms_and_order_are_frozen(self):
        cfg = load_probe_config()
        self.assertEqual(cfg.seeds, (404, 505, 606))
        self.assertEqual(cfg.offsets_px, (-6, 0, 6))
        self.assertEqual(EXPECTED_ORDER, [(404,-6),(404,0),(404,6),(505,-6),(505,0),(505,6),(606,-6),(606,0),(606,6)])
        path = cfg.config_path
        value = json.loads(path.read_text())
        value["offsets_px"] = [-7, 0, 7]
        temp = path.with_name("issue37-invalid-temp.json")
        try:
            temp.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_probe_config(temp)
        finally:
            temp.unlink(missing_ok=True)

    def test_source_files_are_tracked_and_output_directory_is_ignored(self):
        root = Path(__file__).resolve().parents[1]
        required = ["breakout_rl/issue37_reachable_contact_probe.py",
                    "scripts/evaluation/run_issue37_paddle_contact_probe.py",
                    "tests/test_issue37_reachable_contact_probe.py",
                    "configs/eval/issue37_paddle_contact_probe_v1.json"]
        for filename in required:
            self.assertTrue((root / filename).is_file())
        ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q",
                                  "outputs/issue-37-paddle-contact-probe/results.json"])
        self.assertEqual(ignored.returncode, 0)


if __name__ == "__main__":
    unittest.main()
