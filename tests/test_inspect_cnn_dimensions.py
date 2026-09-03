"""Integration tests for the real Day 7 shape inspection path."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.analysis.inspect_cnn_dimensions import collect_cnn_inspection, resolve_device


class CnnInspectionTests(unittest.TestCase):
    def test_real_breakout_observation_reaches_flattened_features(self) -> None:
        inspection = collect_cnn_inspection(seed=42, device_name="cpu")

        self.assertEqual(inspection.device.type, "cpu")
        self.assertEqual(inspection.observation.shape, (4, 84, 84))
        self.assertEqual(inspection.observation.dtype, np.uint8)
        self.assertEqual(tuple(inspection.model_input.shape), (1, 4, 84, 84))
        self.assertEqual(tuple(inspection.features.shape), (1, 3136))
        self.assertEqual(inspection.shapes["conv3"], (1, 64, 7, 7))
        self.assertEqual(inspection.shapes["flatten"], (1, 3136))

    def test_auto_selects_cpu_when_cuda_is_unavailable(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is available in this environment")

        self.assertEqual(resolve_device("auto").type, "cpu")


if __name__ == "__main__":
    unittest.main()
