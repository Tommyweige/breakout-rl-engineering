"""Correctness checks for the deterministic pixel-driven Breakout controller."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from breakout_rl.completion import inspect_breakout_completion_support, read_ale_episode_frame
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
    load_vision_evaluation_protocol,
    make_vision_breakout_env,
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
