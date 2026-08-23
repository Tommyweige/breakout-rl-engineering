"""Tests for the validated DQN training configuration."""

from __future__ import annotations

import unittest

from breakout_rl.training.config import DQNConfig


class DQNConfigTests(unittest.TestCase):
    def test_development_defaults_are_serializable_and_round_trip(self) -> None:
        config = DQNConfig()

        self.assertGreater(config.total_steps, 0)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_starts, 1_000)
        self.assertTrue(config.reward_clip)

        restored = DQNConfig.from_dict(config.to_dict())

        self.assertEqual(restored, config)

    def test_smoke_preset_keeps_real_training_order_but_is_small(self) -> None:
        config = DQNConfig.smoke(total_steps=1000, device="cpu")

        self.assertEqual(config.total_steps, 1000)
        self.assertLess(config.replay_capacity, DQNConfig().replay_capacity)
        self.assertGreaterEqual(config.learning_starts, config.batch_size)
        self.assertEqual(config.train_frequency, 4)
        self.assertEqual(config.device, "cpu")

    def test_zero_discount_is_a_valid_boundary_value(self) -> None:
        self.assertEqual(DQNConfig(gamma=0.0).gamma, 0.0)

    def test_batch_and_warmup_relationships_are_validated(self) -> None:
        invalid_values = (
            {"batch_size": 0},
            {"replay_capacity": 31},
            {"learning_starts": 31},
        )

        for overrides in invalid_values:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    DQNConfig(**overrides)

    def test_numeric_ranges_and_types_are_validated(self) -> None:
        invalid_values = (
            {"total_steps": 0},
            {"seed": -1},
            {"gamma": 1.1},
            {"learning_rate": 0.0},
            {"epsilon_start": float("nan")},
            {"gradient_clip_norm": 0.0},
            {"reward_clip": 1},
            {"device": ""},
        )

        for overrides in invalid_values:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    DQNConfig(**overrides)


if __name__ == "__main__":
    unittest.main()
