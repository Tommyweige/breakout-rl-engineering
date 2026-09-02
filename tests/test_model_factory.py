"""Tests for selecting independent DQN algorithms and architectures."""

from __future__ import annotations

import unittest

from breakout_rl.models import DQNNetwork, DuelingDQNNetwork
from breakout_rl.models.factory import (
    build_q_network,
    checkpoint_architecture,
    normalize_architecture,
)
from breakout_rl.training.config import DQNConfig


class ModelFactoryTests(unittest.TestCase):
    def test_factory_builds_standard_and_dueling_networks(self) -> None:
        standard = build_q_network("standard", num_actions=4)
        dueling = build_q_network("dueling", num_actions=4)

        self.assertIsInstance(standard, DQNNetwork)
        self.assertIsInstance(dueling, DuelingDQNNetwork)

    def test_factory_rejects_unknown_architecture(self) -> None:
        with self.assertRaises(ValueError):
            build_q_network("dueling_double_dqn", num_actions=4)
        with self.assertRaises(ValueError):
            normalize_architecture("normal")

    def test_only_known_format_one_dqn_checkpoints_get_legacy_standard_fallback(self) -> None:
        self.assertEqual(
            checkpoint_architecture(
                {
                    "format_version": 1,
                    "trainer": "dqn",
                    "config": {},
                    "online_network": {},
                }
            ),
            "standard",
        )
        with self.assertRaisesRegex(ValueError, "missing architecture metadata"):
            checkpoint_architecture({"online_network": {}})
        with self.assertRaisesRegex(ValueError, "missing architecture metadata"):
            checkpoint_architecture(
                {
                    "format_version": 2,
                    "trainer": "unknown",
                    "config": {},
                    "online_network": {},
                }
            )

    def test_algorithm_and_architecture_are_independent_config_fields(self) -> None:
        for algorithm in ("dqn", "double_dqn"):
            for architecture in ("standard", "dueling"):
                with self.subTest(algorithm=algorithm, architecture=architecture):
                    config = DQNConfig(
                        algorithm=algorithm,
                        architecture=architecture,
                    )
                    self.assertEqual(config.algorithm, algorithm)
                    self.assertEqual(config.architecture, architecture)


if __name__ == "__main__":
    unittest.main()
