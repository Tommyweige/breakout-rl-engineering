"""Integration tests for the real Day 8 Breakout → DQN inspection path."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.analysis.inspect_dqn_network import collect_dqn_inspection


class DQNInspectionTests(unittest.TestCase):
    def test_real_breakout_state_reaches_one_q_value_per_action(self) -> None:
        inspection = collect_dqn_inspection(seed=42, device_name="cpu")

        self.assertEqual(inspection.device.type, "cpu")
        self.assertEqual(inspection.observation.shape, (4, 84, 84))
        self.assertEqual(inspection.observation.dtype, np.uint8)
        self.assertEqual(tuple(inspection.model_input.shape), (1, 4, 84, 84))
        self.assertEqual(tuple(inspection.features.shape), (1, 3136))
        self.assertEqual(tuple(inspection.q_values.shape), (1, 4))
        self.assertEqual(len(inspection.action_meanings), 4)
        self.assertTrue(torch.isfinite(inspection.q_values).all().item())
        self.assertGreaterEqual(inspection.greedy_action_index, 0)
        self.assertLess(inspection.greedy_action_index, 4)
        self.assertGreater(inspection.parameter_count, 0)
        self.assertEqual(inspection.state_dict_roundtrip_max_abs_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
