"""Tests for the Dueling DQN representation contract."""

from __future__ import annotations

import unittest

import torch

from breakout_rl.models import AtariFeatureExtractor, DuelingDQNNetwork


class DuelingDQNNetworkTests(unittest.TestCase):
    def test_components_and_q_values_keep_the_public_batch_contract(self) -> None:
        model = DuelingDQNNetwork(num_actions=4).cpu().eval()

        for batch_size in (1, 8):
            with self.subTest(batch_size=batch_size):
                observations = torch.zeros(batch_size, 4, 84, 84)
                value, advantage, q_values = model.forward_components(observations)

                self.assertEqual(value.shape, (batch_size, 1))
                self.assertEqual(advantage.shape, (batch_size, 4))
                self.assertEqual(q_values.shape, (batch_size, 4))
                self.assertEqual(model(observations).shape, (batch_size, 4))

    def test_mean_centered_advantage_reconstructs_q_values(self) -> None:
        torch.manual_seed(42)
        model = DuelingDQNNetwork(num_actions=4).cpu().eval()
        observations = torch.randn(2, 4, 84, 84)

        with torch.inference_mode():
            value, advantage, q_values = model.forward_components(observations)
            expected = value + advantage - advantage.mean(dim=1, keepdim=True)

        torch.testing.assert_close(q_values, expected)
        torch.testing.assert_close(
            (advantage - advantage.mean(dim=1, keepdim=True)).mean(dim=1),
            torch.zeros(2),
            atol=1e-6,
            rtol=0.0,
        )

    def test_forward_matches_inspection_components(self) -> None:
        model = DuelingDQNNetwork(num_actions=4).cpu().eval()
        observations = torch.randn(2, 4, 84, 84)

        with torch.inference_mode():
            expected = model.forward_components(observations)[2]
            actual = model(observations)

        self.assertTrue(torch.equal(actual, expected))

    def test_reuses_the_shared_atari_feature_extractor(self) -> None:
        model = DuelingDQNNetwork(num_actions=4)

        self.assertIsInstance(model.feature_extractor, AtariFeatureExtractor)
        self.assertEqual(model.feature_dim, model.feature_extractor.feature_dim)

    def test_backward_reaches_cnn_and_both_streams(self) -> None:
        model = DuelingDQNNetwork(num_actions=4).cpu()
        observations = torch.randn(2, 4, 84, 84)

        model(observations).sum().backward()

        for group in (
            model.feature_extractor.parameters(),
            model.value_stream.parameters(),
            model.advantage_stream.parameters(),
        ):
            gradients = list(group)
            self.assertTrue(gradients)
            self.assertTrue(all(parameter.grad is not None for parameter in gradients))

    def test_invalid_dimensions_are_rejected(self) -> None:
        for invalid in (0, -1, True, 2.0):
            with self.subTest(num_actions=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    DuelingDQNNetwork(num_actions=invalid)  # type: ignore[arg-type]
            with self.subTest(hidden_dim=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    DuelingDQNNetwork(num_actions=4, hidden_dim=invalid)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
