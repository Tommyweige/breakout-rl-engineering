"""Tests for the Atari CNN feature extractor public contract."""

from __future__ import annotations

import unittest

import torch

from breakout_rl.models import AtariFeatureExtractor


class AtariFeatureExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = AtariFeatureExtractor().cpu()

    def test_batch_one_and_two_produce_flattened_feature_vectors(self) -> None:
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                observations = torch.zeros(batch_size, 4, 84, 84)

                features = self.model(observations)

                self.assertEqual(features.shape, (batch_size, self.model.feature_dim))
                self.assertEqual(features.ndim, 2)

    def test_feature_dim_matches_the_runtime_flatten_shape(self) -> None:
        observations = torch.zeros(2, 4, 84, 84)

        features, shapes = self.model.forward_features_with_shapes(observations)

        self.assertEqual(shapes["flatten"], tuple(features.shape))
        self.assertEqual(
            self.model.feature_dim,
            self.model.feature_map_shape[0]
            * self.model.feature_map_shape[1]
            * self.model.feature_map_shape[2],
        )

    def test_baseline_convolution_shapes_are_observed_from_forward(self) -> None:
        observations = torch.zeros(1, 4, 84, 84)

        _, shapes = self.model.forward_features_with_shapes(observations)

        self.assertEqual(shapes["conv1"], (1, 32, 20, 20))
        self.assertEqual(shapes["conv2"], (1, 64, 9, 9))
        self.assertEqual(shapes["conv3"], (1, 64, 7, 7))
        self.assertEqual(shapes["flatten"], (1, 64 * 7 * 7))

    def test_backward_reaches_feature_extractor_parameters(self) -> None:
        observations = torch.randn(2, 4, 84, 84, requires_grad=True)

        self.model(observations).sum().backward()

        gradients = [parameter.grad for parameter in self.model.parameters()]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all().item() for gradient in gradients if gradient is not None)
        )

    def test_invalid_model_input_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.zeros(4, 84, 84))

        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 3, 84, 84))


if __name__ == "__main__":
    unittest.main()
