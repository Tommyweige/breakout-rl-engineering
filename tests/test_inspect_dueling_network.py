"""Integration tests for real Contract v2 Dueling inspection."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.analysis.inspect_dueling_network import (
    collect_dueling_inspection,
    inspection_payload,
)


class DuelingInspectionTests(unittest.TestCase):
    def test_real_breakout_observation_reconstructs_four_q_values(self) -> None:
        inspection = collect_dueling_inspection(seed=42, device_name="cpu")

        self.assertEqual(str(inspection.device), "cpu")
        self.assertEqual(inspection.observation.shape, (4, 84, 84))
        self.assertEqual(inspection.observation.dtype, np.uint8)
        self.assertEqual(tuple(inspection.features.shape), (1, 3136))
        self.assertEqual(tuple(inspection.value.shape), (1, 1))
        self.assertEqual(tuple(inspection.advantage.shape), (1, 4))
        self.assertEqual(tuple(inspection.q_values.shape), (1, 4))
        self.assertLessEqual(inspection.reconstruction_max_abs_error, 1e-5)
        self.assertTrue(torch.isfinite(inspection.q_values).all().item())

        payload = inspection_payload(inspection, seed=42)
        self.assertEqual(payload["observation_source"], "seeded real Breakout observation under Contract v2")
        self.assertEqual(payload["contract"]["contract_id"], "day15-breakout-evaluation-v2-fire-reset")
        self.assertEqual(payload["model_config"]["architecture"], "dueling")
        self.assertEqual(len(payload["q_values"]), 4)
        self.assertIn("not a learned policy", payload["trained_policy_claim"])


if __name__ == "__main__":
    unittest.main()
