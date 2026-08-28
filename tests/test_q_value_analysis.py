"""Behavioral tests for fixed-probe Q-value analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from breakout_rl.analysis.q_values import (
    analyze_q_values,
    load_probe_states,
    save_probe_states,
    summarize_q_values,
)


class FixedProbeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([1.0, 3.0, 2.0, -1.0]))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0).expand(states.shape[0], -1)


class QValueAnalysisTests(unittest.TestCase):
    def test_analyzer_accepts_uint8_probe_batch_and_uses_no_grad(self) -> None:
        probes = np.zeros((3, 4, 84, 84), dtype=np.uint8)
        model = FixedProbeNetwork()

        result = analyze_q_values(model, probes, device="cpu")

        self.assertEqual(result["probe_count"], 3)
        self.assertEqual(result["observation_shape"], [4, 84, 84])
        self.assertEqual(result["q_mean"], 1.25)
        self.assertEqual(result["q_std"], float(np.std([1.0, 3.0, 2.0, -1.0])))
        self.assertEqual(result["q_min"], -1.0)
        self.assertEqual(result["q_max"], 3.0)
        self.assertEqual(result["max_q_mean"], 3.0)
        self.assertEqual(result["selected_action_distribution"], {"FIRE": 3})
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_summary_aggregation_counts_selected_actions(self) -> None:
        summary = summarize_q_values(
            np.array(
                [
                    [1.0, 2.0, 0.0, -1.0],
                    [4.0, 3.0, 2.0, 1.0],
                ],
                dtype=np.float32,
            )
        )

        self.assertEqual(summary["selected_action_distribution"], {"FIRE": 1, "NOOP": 1})
        self.assertEqual(summary["max_q_mean"], 3.0)

    def test_probe_states_round_trip_with_metadata(self) -> None:
        probes = np.zeros((2, 4, 84, 84), dtype=np.uint8)
        metadata = {"contract_id": "test-contract", "records": [{"step": 0}]}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.npz"
            save_probe_states(path, probes, metadata)
            loaded, loaded_metadata = load_probe_states(path)

        np.testing.assert_array_equal(loaded, probes)
        self.assertEqual(loaded_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
