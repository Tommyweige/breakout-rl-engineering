"""Neural-network building blocks used by the Breakout agent."""

from breakout_rl.models.atari_cnn import AtariFeatureExtractor
from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.models.dueling_dqn import DuelingDQNNetwork
from breakout_rl.models.factory import (
    SUPPORTED_ARCHITECTURES,
    build_q_network,
    checkpoint_architecture,
    normalize_architecture,
)

__all__ = [
    "AtariFeatureExtractor",
    "DQNNetwork",
    "DuelingDQNNetwork",
    "SUPPORTED_ARCHITECTURES",
    "build_q_network",
    "checkpoint_architecture",
    "normalize_architecture",
]
