"""The convolutional feature extractor used by the Atari DQN baseline."""

from __future__ import annotations

from math import prod
from typing import Final

import torch
from torch import nn


DEFAULT_INPUT_SHAPE: Final[tuple[int, int, int]] = (4, 84, 84)


class AtariFeatureExtractor(nn.Module):
    """Extract a flat feature vector from stacked preprocessed Atari frames.

    The architecture follows the convolutional trunk from the classic Atari
    DQN baseline. It intentionally stops before the action-value head that will
    be introduced on Day 8.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    ) -> None:
        super().__init__()

        if len(input_shape) != 3 or any(dimension < 1 for dimension in input_shape):
            raise ValueError("input_shape must contain three positive dimensions")

        self.input_shape = tuple(int(dimension) for dimension in input_shape)
        input_channels, _, _ = self.input_shape

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            dummy_feature_map = self.features(dummy_input)

        self.feature_map_shape = tuple(
            int(dimension) for dimension in dummy_feature_map.shape[1:]
        )
        self.feature_dim = prod(self.feature_map_shape)

    def _validate_input(self, observations: torch.Tensor) -> None:
        if observations.ndim != 4:
            raise ValueError(
                "AtariFeatureExtractor expects NCHW input with shape "
                f"(B, {self.input_shape[0]}, {self.input_shape[1]}, "
                f"{self.input_shape[2]}); received {tuple(observations.shape)}"
            )
        if tuple(observations.shape[1:]) != self.input_shape:
            raise ValueError(
                "AtariFeatureExtractor received an unexpected per-sample shape: "
                f"expected {self.input_shape}, got {tuple(observations.shape[1:])}"
            )

    def _forward_with_inspection(
        self,
        observations: torch.Tensor,
        *,
        collect_activations: bool,
    ) -> tuple[
        torch.Tensor,
        dict[str, tuple[int, ...]],
        dict[str, torch.Tensor],
    ]:
        self._validate_input(observations)

        shapes: dict[str, tuple[int, ...]] = {
            "input": tuple(observations.shape),
        }
        activations: dict[str, torch.Tensor] = {}
        current = observations
        convolution_index = 0

        for layer in self.features:
            current = layer(current)
            if isinstance(layer, nn.Conv2d):
                convolution_index += 1
                name = f"conv{convolution_index}"
                shapes[name] = tuple(current.shape)
                if collect_activations:
                    activations[name] = current

        flattened = self.flatten(current)
        shapes["flatten"] = tuple(flattened.shape)
        return flattened, shapes, activations

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one flattened feature vector per NCHW observation."""

        features, _, _ = self._forward_with_inspection(
            observations,
            collect_activations=False,
        )
        return features

    def forward_features_with_shapes(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        """Return features and the tensor shapes observed during forward."""

        features, shapes, _ = self._forward_with_inspection(
            observations,
            collect_activations=False,
        )
        return features, shapes

    def forward_features_with_activations(
        self,
        observations: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, tuple[int, ...]],
        dict[str, torch.Tensor],
    ]:
        """Return features, runtime shapes, and convolution activations."""

        return self._forward_with_inspection(
            observations,
            collect_activations=True,
        )
