"""Correctness checks for the deterministic pixel-driven Breakout controller."""

from __future__ import annotations

import inspect
import csv
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from breakout_rl.completion import (
    BREAKOUT_FULL_CLEAR_SCORE,
    CompletionSupport,
    inspect_breakout_completion_support,
    read_ale_episode_frame,
)
from breakout_rl.vision_controller import (
    LEFT,
    NOOP,
    RIGHT,
    PlayfieldBounds,
    PredictiveBreakoutController,
    choose_paddle_action,
    detect_playfield_bounds,
    predict_paddle_intercept,
    reflect_x,
)
from breakout_rl.vision_evaluation import (
    evaluate_predictive_controller,
    load_vision_evaluation_protocol,
    make_vision_breakout_env,
    write_vision_evaluation_artifacts,
)


BALL_RGB = np.array([200, 72, 72], dtype=np.uint8)
RAIL_RGB = np.array([142, 142, 142], dtype=np.uint8)


def screen_fixture(
    *,
    ball: tuple[int, int] | None = None,
    paddle_left: int = 96,
    bricks: bool = True,
) -> np.ndarray:
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    frame[17:193, :8] = RAIL_RGB
    frame[17:193, 152:] = RAIL_RGB
    frame[17:32, :] = RAIL_RGB
    if bricks:
        frame[57:75, 8:152] = BALL_RGB
    frame[189:193, paddle_left : paddle_left + 16] = BALL_RGB
    if ball is not None:
        x, y = ball
        frame[y : y + 4, x : x + 2] = BALL_RGB
    return frame


class PredictiveVisionPerceptionTests(unittest.TestCase):
    def test_ball_connected_component_and_playfield_bounds(self) -> None:
        controller = PredictiveBreakoutController()
        observation = controller.observe(screen_fixture(ball=(78, 120)))
        self.assertEqual(
            (observation.bounds.left, observation.bounds.right, observation.bounds.top, observation.bounds.bottom),
            (8.0, 151.0, 32.0, 192.0),
        )
        self.assertIsNotNone(observation.ball.x)
        self.assertAlmostEqual(observation.ball.x or 0.0, 78.5)
        self.assertAlmostEqual(observation.ball.y or 0.0, 121.5)
        self.assertEqual(observation.ball.confidence, 1.0)
        self.assertTrue(observation.ball.directly_detected)

    def test_paddle_detection_uses_visible_edges(self) -> None:
        observation = PredictiveBreakoutController().observe(
            screen_fixture(ball=(78, 120), paddle_left=44)
        )
        paddle = observation.paddle
        self.assertIsNotNone(paddle)
        assert paddle is not None
        self.assertEqual(paddle.left, 44.0)
        self.assertEqual(paddle.right, 59.0)
        self.assertEqual(paddle.center_x, 51.5)
        self.assertEqual(paddle.top, 189.0)

    def test_paddle_center_is_recovered_when_sprite_is_clipped_at_a_wall(self) -> None:
        observation = PredictiveBreakoutController().observe(
            screen_fixture(ball=(78, 120), paddle_left=144)
        )
        paddle = observation.paddle
        self.assertIsNotNone(paddle)
        assert paddle is not None
        self.assertEqual((paddle.left, paddle.right), (144.0, 151.0))
        self.assertEqual(paddle.center_x, 143.5)

    def test_static_brick_fixture_does_not_create_a_ball(self) -> None:
        controller = PredictiveBreakoutController()
        observation = controller.observe(screen_fixture(ball=None))
        self.assertIsNone(observation.ball.x)
        self.assertIsNone(observation.ball.y)
        self.assertFalse(observation.ball.directly_detected)

    def test_one_frame_lost_ball_is_extrapolated_and_reacquired(self) -> None:
        controller = PredictiveBreakoutController(max_missing_frames=4)
        first = controller.observe(screen_fixture(ball=(80, 120)))
        self.assertTrue(first.ball.directly_detected)
        second = controller.observe(screen_fixture(ball=(79, 121)))
        self.assertEqual((second.ball.vx, second.ball.vy), (-1.0, 1.0))
        lost = controller.observe(screen_fixture(ball=None))
        self.assertFalse(lost.ball.directly_detected)
        self.assertEqual(lost.ball.age_frames, 1)
        self.assertAlmostEqual(lost.ball.x or 0.0, 78.5)
        self.assertAlmostEqual(lost.ball.y or 0.0, 123.5)
        reacquired = controller.observe(screen_fixture(ball=(77, 123)))
        self.assertTrue(reacquired.ball.directly_detected)
        self.assertAlmostEqual(reacquired.ball.vx or 0.0, -1.0)
        self.assertAlmostEqual(reacquired.ball.vy or 0.0, 1.0)

    def test_ball_track_expires_instead_of_hallucinating(self) -> None:
        controller = PredictiveBreakoutController(max_missing_frames=3)
        controller.observe(screen_fixture(ball=(80, 120)))
        controller.observe(screen_fixture(ball=(78, 122)))
        for _ in range(4):
            observation = controller.observe(screen_fixture(ball=None))
        self.assertIsNone(observation.ball.x)
        self.assertIsNone(observation.ball.y)
        self.assertIsNone(observation.ball.vx)
        self.assertEqual(observation.ball.age_frames, 4)

    def test_velocity_sign_change_is_applied_on_the_next_observation(self) -> None:
        controller = PredictiveBreakoutController()
        controller.observe(screen_fixture(ball=(50, 100)))
        moving_right = controller.observe(screen_fixture(ball=(52, 102)))
        moving_left = controller.observe(screen_fixture(ball=(50, 104)))
        self.assertEqual(moving_right.ball.vx, 2.0)
        self.assertEqual(moving_left.ball.vx, -2.0)
        self.assertEqual(moving_left.ball.vy, 2.0)


class PredictiveControlMathTests(unittest.TestCase):
    def test_horizontal_reflection_handles_multiple_wall_bounces(self) -> None:
        self.assertAlmostEqual(reflect_x(170.0, 10.0, 150.0), 130.0)
        self.assertAlmostEqual(reflect_x(-30.0, 10.0, 150.0), 50.0)

    def test_paddle_plane_intercept_includes_side_wall_reflection(self) -> None:
        bounds = PlayfieldBounds(10.0, 150.0, 30.0, 190.0, 1.0)
        prediction = predict_paddle_intercept(
            ball_x=120.0,
            ball_y=120.0,
            ball_vx=3.0,
            ball_vy=2.0,
            paddle_plane_y=140.0,
            bounds=bounds,
            ball_radius=2.0,
        )
        self.assertIsNotNone(prediction)
        intercept, frames = prediction or (0.0, 0.0)
        self.assertEqual(frames, 10.0)
        self.assertAlmostEqual(intercept, 146.0)

    def test_upward_ball_has_no_paddle_intercept(self) -> None:
        self.assertIsNone(
            predict_paddle_intercept(
                ball_x=80.0,
                ball_y=100.0,
                ball_vx=1.0,
                ball_vy=-2.0,
                paddle_plane_y=180.0,
                bounds=PlayfieldBounds(8.0, 151.0, 32.0, 192.0, 1.0),
            )
        )

    def test_deadband_hysteresis_prevents_left_right_chatter(self) -> None:
        self.assertEqual(
            choose_paddle_action(error=3.5, previous_direction=NOOP), NOOP
        )
        self.assertEqual(
            choose_paddle_action(error=4.5, previous_direction=NOOP), RIGHT
        )
        self.assertEqual(
            choose_paddle_action(error=2.5, previous_direction=RIGHT), RIGHT
        )
        self.assertEqual(
            choose_paddle_action(error=1.5, previous_direction=RIGHT), NOOP
        )
        self.assertEqual(
            choose_paddle_action(error=-5.0, previous_direction=RIGHT), LEFT
        )

    def test_closed_loop_uses_currently_observed_paddle_position(self) -> None:
        controller = PredictiveBreakoutController()
        controller.select_action(
            screen_fixture(ball=(60, 120), paddle_left=72)
        )
        decision = controller.select_action(
            screen_fixture(ball=(62, 122), paddle_left=100)
        )
        self.assertIsNotNone(decision.predicted_intercept_x)
        self.assertIsNotNone(decision.observation.paddle)
        assert decision.observation.paddle is not None
        self.assertEqual(decision.observation.paddle.center_x, 107.5)
        self.assertAlmostEqual(
            decision.paddle_error or 0.0,
            (decision.predicted_intercept_x or 0.0)
            - decision.observation.paddle.center_x,
        )
        self.assertEqual(decision.action, RIGHT)

    def test_control_api_accepts_only_a_screen_frame_and_never_requests_fire(self) -> None:
        parameters = tuple(inspect.signature(PredictiveBreakoutController.select_action).parameters)
        self.assertEqual(parameters, ("self", "frame"))
        controller = PredictiveBreakoutController()
        decision = controller.select_action(screen_fixture(ball=(80, 120)))
        self.assertIn(decision.action, (NOOP, LEFT, RIGHT))
        self.assertFalse(hasattr(controller, "ale"))
        self.assertFalse(hasattr(controller, "completion_detector"))


class VisionEvaluationSemanticsTests(unittest.TestCase):
    def evaluate_fixture(self, *, clear: bool = True) -> dict:
        """Exercise reporting with synthetic detector events, not an ALE evaluation."""
        class FixtureEnv:
            spec = SimpleNamespace(id="ALE/Breakout-v5")

            @property
            def unwrapped(self):
                return self

            def get_action_meanings(self):
                return ("NOOP", "FIRE", "RIGHT", "LEFT")

            def reset(self, *, seed):
                self.frame = 0
                self.score = 0
                return screen_fixture(), {}

            def step(self, action):
                self.frame += 1
                self.score = BREAKOUT_FULL_CLEAR_SCORE if clear and self.frame == 2 else 0
                # The wrapper sends FIRE on step 1. No physical/sticky-resolved
                # action is exposed, and step 2 tests the requested-action fallback.
                info = {"fire_reset_executed_action": 1, "fire_reset_auto": True} if self.frame == 1 else {}
                return screen_fixture(), self.score, self.frame == 2, False, info

            def close(self):
                pass

        support = CompletionSupport(
            supported=True, environment_id="ALE/Breakout-v5", game="breakout",
            mode=0, difficulty=0, ale_py_version="fixture", rom_sha256="fixture",
        )
        protocol = replace(load_vision_evaluation_protocol(), seeds=(101,), trace_episodes=1)
        with (
            patch("breakout_rl.vision_evaluation.inspect_breakout_completion_support", return_value=support),
            patch("breakout_rl.vision_evaluation.read_ale_episode_frame", side_effect=lambda env: env.frame),
            patch("breakout_rl.vision_evaluation.read_breakout_score", side_effect=lambda env: env.score),
            patch("breakout_rl.vision_evaluation.read_ale_lives", return_value=2),
            patch("breakout_rl.vision_evaluation._source_provenance", return_value={"fixture": True}),
        ):
            return evaluate_predictive_controller(protocol, env_factory=FixtureEnv, max_steps_per_episode=2)

    def test_detector_events_are_reported_as_canonical_detections(self) -> None:
        for clear in (False, True):
            with self.subTest(clear=clear), TemporaryDirectory() as directory:
                payload = self.evaluate_fixture(clear=clear)
                self.assertIs(payload["episodes"][0]["cleared"], clear)
                self.assertEqual(payload["summary"]["clear_count"], int(clear))
                self.assertEqual(payload["summary"]["clear_rate"], float(clear))
                if clear:
                    self.assertEqual(payload["episodes"][0]["failure_reason"], "canonical_clear_detected")
                self.assertNotIn("verified_clears", payload["summary"])
                write_vision_evaluation_artifacts(payload, directory)
                report = (Path(directory) / "report.md").read_text()
                self.assertIn(f"| Canonical clear detections | {int(clear)} / 1 |", report)
                self.assertNotIn("Verified clears", report)
                self.assertNotIn("verified clear.", report)
                self.assertIn("audited `BreakoutCompletionDetector`", report)
                self.assertIn("separate contract/source/provenance validation", report)

    def test_ale_input_action_preserves_fire_override_and_requested_fallback(self) -> None:
        payload = self.evaluate_fixture()
        rows = payload["step_traces"][0]["rows"]
        self.assertEqual(rows[0]["requested_action"], "NOOP")
        self.assertEqual(rows[0]["ale_input_action"], "FIRE")
        self.assertEqual(rows[1]["ale_input_action"], rows[1]["requested_action"])
        for row in rows:
            self.assertNotIn("executed_action", row)
        for metrics in (payload["summary"], payload["episodes"][0]):
            self.assertEqual(metrics["requested_action_distribution"]["NOOP"], 2)
            self.assertEqual(metrics["ale_input_action_distribution"]["FIRE"], 1)
            self.assertEqual(metrics["ale_input_action_distribution"]["NOOP"], 1)
            self.assertNotIn("executed_action_distribution", metrics)
        self.assertEqual(payload["environment"]["sticky_action_probability"], 0.25)
        self.assertIn("physical action resolved by ALE is not observed", payload["evaluation_protocol"]["action_semantics"]["ale_input_action"])

    def test_csv_diagnostics_and_report_use_ale_input_terminology(self) -> None:
        payload = self.evaluate_fixture()
        with TemporaryDirectory() as directory:
            results, episodes, diagnostics = write_vision_evaluation_artifacts(payload, directory)
            with episodes.open(newline="") as stream:
                episode = next(csv.DictReader(stream))
            self.assertIn("ale_input_action_distribution", episode)
            self.assertNotIn("executed_action_distribution", episode)
            self.assertEqual(json.loads(episode["ale_input_action_distribution"])["FIRE"], 1)
            trace_path = Path(directory) / "controller_trace_seed_101.csv"
            with trace_path.open(newline="") as stream:
                trace = next(csv.DictReader(stream))
            self.assertEqual(trace["requested_action"], "NOOP")
            self.assertEqual(trace["ale_input_action"], "FIRE")
            self.assertNotIn("executed_action", trace)
            for path in (results, diagnostics):
                summary = json.loads(path.read_text())["summary"]
                self.assertIn("ale_input_action_distribution", summary)
                self.assertNotIn("executed_action_distribution", summary)
            report = (Path(directory) / "report.md").read_text()
            self.assertIn("ALE input actions (after FIRE wrapper)", report)
            self.assertNotIn("Executed actions", report)
            self.assertIn("sticky-action probability 0.25", report)
            self.assertIn("physical action resolved by ALE is not observed", report)


class RealALEVisionSmokeTests(unittest.TestCase):
    def test_raw_rgb_control_handles_fire_and_tracks_real_ale_frames(self) -> None:
        protocol = load_vision_evaluation_protocol()
        env = make_vision_breakout_env(protocol.contract)
        try:
            support = inspect_breakout_completion_support(env)
            self.assertTrue(support.supported, support.reason)
            meanings = env.unwrapped.get_action_meanings()
            action_index = {name: index for index, name in enumerate(meanings)}
            observation, _ = env.reset(seed=101)
            self.assertEqual(observation.shape, (210, 160, 3))
            controller = PredictiveBreakoutController(**protocol.controller_options)
            previous_frame = read_ale_episode_frame(env)
            detected_ball_frames = 0
            auto_fire_actions = []
            for _ in range(36):
                decision = controller.select_action(observation)
                self.assertIn(decision.action, (NOOP, LEFT, RIGHT))
                observation, _reward, terminated, truncated, info = env.step(
                    action_index[decision.action]
                )
                current_frame = read_ale_episode_frame(env)
                self.assertEqual(current_frame - previous_frame, 1)
                previous_frame = current_frame
                if info.get("fire_reset_auto"):
                    auto_fire_actions.append(info["fire_reset_executed_action"])
                    self.assertEqual(
                        info["fire_reset_executed_action"], action_index["FIRE"]
                    )
                if decision.observation.ball.directly_detected:
                    detected_ball_frames += 1
                if terminated or truncated:
                    break
            self.assertGreater(detected_ball_frames, 20)
            self.assertGreaterEqual(len(auto_fire_actions), 1)
            self.assertGreater(controller.diagnostics["paddle_detection_success_rate"], 0.95)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
