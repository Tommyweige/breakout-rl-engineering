"""Deep Q-Network built on top of the reusable Atari CNN feature extractor."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from breakout_rl.models.atari_cnn import AtariFeatureExtractor, DEFAULT_INPUT_SHAPE

DEFAULT_HIDDEN_DIM: Final[int] = 512


class DQNNetwork(nn.Module):
    """Map normalized stacked Atari frames to one raw Q-value per action."""

    architecture = "standard"

    def __init__(
        self,
        num_actions: int,
        *,
        input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        if isinstance(num_actions, bool) or not isinstance(num_actions, int) or num_actions < 1:
            raise ValueError("num_actions must be a positive integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")

        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.feature_extractor = AtariFeatureExtractor(input_shape=input_shape)
        self.q_head = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    @property
    def feature_dim(self) -> int:
        """Flattened CNN feature dimension consumed by the Q-value head."""

        return self.feature_extractor.feature_dim

    def forward_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Expose the shared CNN features without duplicating convolution logic."""

        return self.feature_extractor(observations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return raw, unnormalized Q-values with shape ``(B, num_actions)``."""

        features = self.forward_features(observations)
        return self.q_head(features)
