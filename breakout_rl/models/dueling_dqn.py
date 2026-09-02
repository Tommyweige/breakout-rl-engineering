"""Dueling DQN action-value network for the Breakout agent."""

from __future__ import annotations

import torch
from torch import nn

from breakout_rl.models.atari_cnn import AtariFeatureExtractor, DEFAULT_INPUT_SHAPE
from breakout_rl.models.dqn import DEFAULT_HIDDEN_DIM


class DuelingDQNNetwork(nn.Module):
    """Represent Q-values as a state value plus centered action advantages.

    The CNN trunk is shared with :class:`DQNNetwork`.  The two heads expose
    the internal representation for low-frequency inspection while ``forward``
    keeps the same ``(B, num_actions)`` action-value contract as standard DQN.
    """

    architecture = "dueling"

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
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    @property
    def feature_dim(self) -> int:
        """Flattened CNN feature dimension consumed by both streams."""

        return self.feature_extractor.feature_dim

    def forward_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Expose the shared CNN features for inspection and diagnostics."""

        return self.feature_extractor(observations)

    def forward_components(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``V(s)``, raw ``A(s,a)``, and mean-centered ``Q(s,a)``."""

        features = self.forward_features(observations)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return value, advantage, q_values

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return raw Q-values with the standard ``(B, num_actions)`` shape."""

        return self.forward_components(observations)[2]


__all__ = ["DuelingDQNNetwork"]
