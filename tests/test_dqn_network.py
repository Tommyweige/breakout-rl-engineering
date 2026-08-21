"""Tests for the Day 8 DQN network contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from breakout_rl.models import DQNNetwork


class DQNNetworkTests(unittest.TestCase):
    def test_batch_one_and_eight_produce_one_q_value_per_action(self) -> None:
        model = DQNNetwork(num_actions=4).cpu()

        for batch_size in (1, 8):
            with self.subTest(batch_size=batch_size):
                observations = torch.zeros(batch_size, 4, 84, 84)
                q_values = model(observations)

                self.assertEqual(q_values.shape, (batch_size, 4))
                self.assertEqual(q_values.dtype, torch.float32)

    def test_action_count_is_configurable(self) -> None:
        model = DQNNetwork(num_actions=6).cpu()
        q_values = model(torch.zeros(2, 4, 84, 84))

        self.assertEqual(q_values.shape, (2, 6))

    def test_invalid_action_count_is_rejected(self) -> None:
        for invalid in (0, -1, True, 2.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    DQNNetwork(num_actions=invalid)  # type: ignore[arg-type]

    def test_output_is_raw_q_values_not_softmax_probabilities(self) -> None:
        torch.manual_seed(42)
        model = DQNNetwork(num_actions=4).cpu().eval()
        observations = torch.randn(3, 4, 84, 84)

        with torch.inference_mode():
            q_values = model(observations)

        probability_sums = q_values.sum(dim=1)
        self.assertFalse(
            torch.allclose(
                probability_sums,
                torch.ones_like(probability_sums),
                atol=1e-5,
            )
        )
        self.assertFalse(
            all(
                isinstance(layer, torch.nn.Softmax)
                for layer in model.q_head
            )
        )

    def test_backward_reaches_cnn_and_q_head_parameters(self) -> None:
        model = DQNNetwork(num_actions=4).cpu()
        observations = torch.randn(2, 4, 84, 84)

        model(observations).sum().backward()

        cnn_gradients = [
            parameter.grad
            for parameter in model.feature_extractor.parameters()
        ]
        head_gradients = [parameter.grad for parameter in model.q_head.parameters()]

        self.assertTrue(cnn_gradients)
        self.assertTrue(head_gradients)
        self.assertTrue(all(gradient is not None for gradient in cnn_gradients))
        self.assertTrue(all(gradient is not None for gradient in head_gradients))

    def test_feature_extractor_is_reused_by_forward(self) -> None:
        model = DQNNetwork(num_actions=4).cpu().eval()
        observations = torch.randn(2, 4, 84, 84)

        with torch.inference_mode():
            features = model.forward_features(observations)
            q_values = model(observations)
            expected = model.q_head(features)

        self.assertEqual(features.shape, (2, model.feature_dim))
        self.assertTrue(torch.equal(q_values, expected))

    def test_state_dict_save_load_preserves_outputs(self) -> None:
        torch.manual_seed(42)
        model = DQNNetwork(num_actions=4).cpu().eval()
        observations = torch.randn(2, 4, 84, 84)

        with torch.inference_mode():
            expected = model(observations)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dqn-state-dict.pt"
            torch.save(model.state_dict(), path)

            clone = DQNNetwork(num_actions=4).cpu().eval()
            clone.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )

            with torch.inference_mode():
                actual = clone(observations)

        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
