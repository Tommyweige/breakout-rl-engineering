"""Focused tests for the frozen Issue #35 paddle-contact diagnostic."""
from __future__ import annotations
import inspect
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from breakout_rl.paddle_contact_probe import (
    EXPECTED_ORDER, _process_contact_event, analyze_probe_runs, clamp_target_center,
    load_probe_config, select_outgoing_vx,
)
from breakout_rl.vision_controller import PredictiveBreakoutController


def row(*, frame=10, direct=True, vy=2.0, vx=1.0, horizon=2.0, y=185.0, top=189.0,
        intercept=80.0, paddle=75.0):
    return {"ale_emulator_frame": frame, "ball_directly_detected": direct, "ball_vy": vy,
            "ball_vx": vx, "time_to_intercept_frames": horizon, "ball_y": y,
            "paddle_top": top, "paddle_left": 68.0, "paddle_right": 84.0,
            "paddle_center_x": paddle, "predicted_intercept_x": intercept}


class ProbeTargetTests(unittest.TestCase):
    def test_offset_sign_and_legal_center_clamping(self):
        self.assertEqual(clamp_target_center(100, -12, 15.5, 143.5), 112)
        self.assertEqual(clamp_target_center(100, 12, 15.5, 143.5), 88)
        self.assertEqual(clamp_target_center(10, 12, 15.5, 143.5), 15.5)
        self.assertEqual(clamp_target_center(150, -12, 15.5, 143.5), 143.5)

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
        for offset in (0, -12, 12):
            controller = PredictiveBreakoutController()
            controller.select_action_with_target_offset(frame(50, 120), target_offset_px=offset)
            decisions.append(controller.select_action_with_target_offset(frame(51, 121), target_offset_px=offset))
        baseline, aim_right, aim_left = decisions
        self.assertIsNotNone(baseline.predicted_intercept_x)
        self.assertAlmostEqual(aim_right.paddle_error - baseline.paddle_error, 12.0)
        self.assertAlmostEqual(aim_left.paddle_error - baseline.paddle_error, -12.0)

    def test_probe_action_api_has_no_evaluator_arguments(self):
        signature = inspect.signature(PredictiveBreakoutController.select_action_with_target_offset)
        self.assertEqual(tuple(signature.parameters), ("self", "frame", "target_offset_px"))


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


class ProbeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_probe_config()

    def _runs(self, xs, ys):
        return [{"seed": 101, "requested_offset_px": o, "observed_impact_offset_px": x,
                 "outgoing_vx_px_per_frame": y} for o, x, y in zip((-12, 0, 12), xs, ys)]

    def test_slope_threshold_and_ordered_manipulation_span(self):
        at_threshold = analyze_probe_runs(self._runs([0, 12, 24], [0, .24, .48]), self.cfg)
        self.assertEqual(at_threshold["positive_response_seed_count"], 1)
        self.assertAlmostEqual(at_threshold["seed_triplets"][0]["ols_slope_px_per_frame_per_px"], .02)
        below = analyze_probe_runs(self._runs([0, 12, 24], [0, .239, .478]), self.cfg)
        self.assertEqual(below["positive_response_seed_count"], 0)
        wrong_order = analyze_probe_runs(self._runs([0, 12, 11], [0, .3, .4]), self.cfg)
        self.assertFalse(wrong_order["seed_triplets"][0]["positive_response"])
        small_span = analyze_probe_runs(self._runs([0, 5, 10], [0, .3, .6]), self.cfg)
        self.assertFalse(small_span["seed_triplets"][0]["positive_response"])

    def test_classification_boundaries(self):
        promoted = []
        rejected = []
        one_positive = []
        for seed in (101, 202, 303):
            promoted += [{"seed": seed, "requested_offset_px": o, "observed_impact_offset_px": x,
                          "outgoing_vx_px_per_frame": x * .03} for o, x in zip((-12,0,12),(0,12,24))]
            rejected += [{"seed": seed, "requested_offset_px": o, "observed_impact_offset_px": x,
                         "outgoing_vx_px_per_frame": 0.0} for o, x in zip((-12,0,12),(0,12,24))]
            one_positive += [{"seed": seed, "requested_offset_px": o, "observed_impact_offset_px": x,
                              "outgoing_vx_px_per_frame": x * (.03 if seed == 101 else 0)} for o, x in zip((-12,0,12),(0,12,24))]
        self.assertEqual(analyze_probe_runs(promoted, self.cfg)["classification"], "PROMOTED")
        self.assertEqual(analyze_probe_runs(rejected, self.cfg)["classification"], "REJECTED")
        self.assertEqual(analyze_probe_runs(one_positive, self.cfg)["classification"], "INCONCLUSIVE")
        self.assertEqual(analyze_probe_runs(rejected[:3], self.cfg)["classification"], "INCONCLUSIVE")
        partial_positive = analyze_probe_runs(promoted[:6], self.cfg)
        self.assertTrue(partial_positive["positive_response_seed_count"] >= 2)
        self.assertEqual(partial_positive["classification"], "INCONCLUSIVE")

    def test_config_hash_seeds_arms_and_order_are_frozen(self):
        cfg = load_probe_config()
        self.assertEqual(cfg.seeds, (101, 202, 303))
        self.assertEqual(cfg.offsets_px, (-12, 0, 12))
        self.assertEqual(EXPECTED_ORDER, [(101,-12),(101,0),(101,12),(202,-12),(202,0),(202,12),(303,-12),(303,0),(303,12)])
        path = cfg.config_path
        value = json.loads(path.read_text())
        value["offsets_px"] = [-13, 0, 13]
        temp = path.with_name("issue35-invalid-temp.json")
        try:
            temp.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_probe_config(temp)
        finally:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
