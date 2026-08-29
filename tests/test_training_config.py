"""Tests for the validated DQN training configuration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.training.train_dqn import _config_from_args, build_parser
from scripts.training.train_vectorized_dqn import (
    _config_from_args as _vectorized_config_from_args,
    build_parser as build_vectorized_parser,
)
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.dqn_trainer import resolve_device


class DQNConfigTests(unittest.TestCase):
    def test_development_defaults_are_serializable_and_round_trip(self) -> None:
        config = DQNConfig()

        self.assertGreater(config.total_steps, 0)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_starts, 1_000)
        self.assertTrue(config.reward_clip)
        self.assertFalse(config.profile_stages)

        restored = DQNConfig.from_dict(config.to_dict())

        self.assertEqual(restored, config)
        self.assertEqual(config.replay_backend, "cpu")
        self.assertEqual(config.algorithm, "dqn")

    def test_algorithm_is_validated_and_round_trips_through_metadata(self) -> None:
        config = DQNConfig(algorithm="DOUBLE_DQN")

        self.assertEqual(config.algorithm, "double_dqn")
        self.assertEqual(DQNConfig.from_dict(config.to_dict()).algorithm, "double_dqn")

        with self.assertRaises(ValueError):
            DQNConfig(algorithm="dueling")

    def test_smoke_preset_keeps_real_training_order_but_is_small(self) -> None:
        config = DQNConfig.smoke(total_steps=1000, device="cpu")

        self.assertEqual(config.total_steps, 1000)
        self.assertLess(config.replay_capacity, DQNConfig().replay_capacity)
        self.assertGreaterEqual(config.learning_starts, config.batch_size)
        self.assertEqual(config.train_frequency, 4)
        self.assertEqual(config.device, "cpu")

    def test_debug_preset_keeps_training_logic_and_shortens_observation_window(self) -> None:
        config = DQNConfig.debug(device="cpu")

        self.assertGreaterEqual(config.total_steps, 10_000)
        self.assertEqual(config.learning_starts, 1_000)
        self.assertEqual(config.target_update_interval, 500)
        self.assertEqual(config.checkpoint_interval, 500)
        self.assertEqual(config.device, "cpu")

    def test_debug_cli_preset_defaults_to_explicit_cuda(self) -> None:
        args = build_parser().parse_args(["--preset", "debug"])

        config = _config_from_args(args)

        self.assertEqual(config.device, "cuda")

    def test_cli_can_select_double_dqn(self) -> None:
        args = build_parser().parse_args(["--algorithm", "double_dqn", "--device", "cpu"])

        config = _config_from_args(args)

        self.assertEqual(config.algorithm, "double_dqn")

    def test_day17_vectorized_smoke_preset_uses_canonical_backend(self) -> None:
        args = build_vectorized_parser().parse_args(
            ["--preset", "smoke", "--algorithm", "double_dqn", "--device", "cuda"]
        )

        config = _vectorized_config_from_args(args)

        self.assertEqual(config.algorithm, "double_dqn")
        self.assertEqual(config.num_envs, 2)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_starts, 1000)
        self.assertEqual(config.train_frequency, 4)
        self.assertEqual(config.target_update_interval, 500)
        self.assertEqual(config.replay_backend, "gpu")
        self.assertTrue(config.strict_action_selection_parity)
        self.assertEqual(config.cpu_threads, 2)

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

    def test_device_request_and_precision_are_normalized_for_metadata(self) -> None:
        config = DQNConfig(device="AUTO", precision="fp32")

        self.assertEqual(config.requested_device, "auto")
        self.assertEqual(config.precision, "float32")
        self.assertEqual(config.to_dict()["device"], "auto")

    def test_replay_transfer_mode_is_normalized_and_validated(self) -> None:
        config = DQNConfig(replay_transfer="PREALLOCATED")

        self.assertEqual(config.replay_transfer, "preallocated")
        self.assertEqual(
            DQNConfig.from_dict(config.to_dict()).replay_transfer,
            "preallocated",
        )

        with self.assertRaises(ValueError):
            DQNConfig(replay_transfer="unknown")

    def test_replay_backend_is_normalized_and_validated(self) -> None:
        config = DQNConfig(replay_backend="GPU")

        self.assertEqual(config.replay_backend, "gpu")
        self.assertEqual(
            DQNConfig.from_dict(config.to_dict()).replay_backend,
            "gpu",
        )

        with self.assertRaises(ValueError):
            DQNConfig(replay_backend="unknown")
        with self.assertRaises(ValueError):
            DQNConfig(replay_backend="gpu", replay_transfer="preallocated")

    def test_stage_profiling_flag_is_boolean(self) -> None:
        self.assertTrue(DQNConfig(profile_stages=True).profile_stages)
        with self.assertRaises(TypeError):
            DQNConfig(profile_stages=1)

    def test_num_envs_is_a_positive_integer(self) -> None:
        self.assertEqual(DQNConfig(num_envs=4).num_envs, 4)
        with self.assertRaises((TypeError, ValueError)):
            DQNConfig(num_envs=0)
        with self.assertRaises((TypeError, ValueError)):
            DQNConfig(num_envs=True)

    def test_strict_action_selection_parity_flag_is_boolean(self) -> None:
        self.assertTrue(
            DQNConfig(strict_action_selection_parity=True).strict_action_selection_parity
        )
        with self.assertRaises(TypeError):
            DQNConfig(strict_action_selection_parity=1)

    def test_auto_can_use_cpu_but_explicit_cuda_never_falls_back(self) -> None:
        with patch("breakout_rl.training.dqn_trainer.torch.cuda.is_available", return_value=False):
            self.assertEqual(str(resolve_device("auto")), "cpu")
            with self.assertRaisesRegex(RuntimeError, "refusing to fall back to CPU"):
                resolve_device("cuda")


if __name__ == "__main__":
    unittest.main()
