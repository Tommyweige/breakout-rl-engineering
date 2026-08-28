"""Contract tests for the preprocessed Breakout environment."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from breakout_env import make_breakout_env


class BreakoutEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_breakout_env()

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_returns_four_uint8_84_by_84_frames(self) -> None:
        observation, _ = self.env.reset(seed=42)

        self.assertEqual(observation.shape, (4, 84, 84))
        self.assertEqual(observation.dtype, np.uint8)

    def test_reset_observation_stays_in_uint8_pixel_range(self) -> None:
        observation, _ = self.env.reset(seed=42)

        self.assertGreaterEqual(observation.min(), 0)
        self.assertLessEqual(observation.max(), 255)

    def test_preprocessing_preserves_breakout_actions(self) -> None:
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 4)
        self.assertEqual(
            self.env.unwrapped.get_action_meanings(),
            ["NOOP", "FIRE", "RIGHT", "LEFT"],
        )

    def test_steps_keep_the_stacked_observation_contract(self) -> None:
        observation, _ = self.env.reset(seed=42)
        self.env.action_space.seed(42)

        for _ in range(4):
            action = int(self.env.action_space.sample())
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )

            self.assertEqual(next_observation.shape, (4, 84, 84))
            self.assertEqual(next_observation.dtype, np.uint8)
            self.assertIsInstance(reward, (float, int, np.floating, np.integer))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)

            observation = next_observation
            if terminated or truncated:
                observation, _ = self.env.reset()

        self.assertEqual(observation.shape, (4, 84, 84))

    def test_fire_reset_wrapper_reports_the_environment_executed_fire(self) -> None:
        env = make_breakout_env(fire_reset=True)
        try:
            env.reset(seed=42)
            _, _, terminated, truncated, info = env.step(2)

            self.assertFalse(terminated or truncated)
            self.assertTrue(info["fire_reset_auto"])
            self.assertEqual(info["fire_reset_reason"], "initial_serve")
            self.assertEqual(info["fire_reset_requested_action"], 2)
            self.assertEqual(info["fire_reset_executed_action"], 1)
        finally:
            env.close()

    def test_reset_does_not_leak_previous_episode_frames(self) -> None:
        fresh_env = make_breakout_env()
        try:
            self.env.reset(seed=42)
            self.env.action_space.seed(42)
            for action in (2, 3, 0, 1):
                _, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break

            reset_observation, _ = self.env.reset(seed=99)
            fresh_observation, _ = fresh_env.reset(seed=99)

            np.testing.assert_array_equal(reset_observation, fresh_observation)
            for frame in reset_observation[1:]:
                np.testing.assert_array_equal(frame, reset_observation[0])
        finally:
            fresh_env.close()


if __name__ == "__main__":
    unittest.main()
