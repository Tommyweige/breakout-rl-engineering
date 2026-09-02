"""Central model construction for independent DQN algorithms/architectures."""

from __future__ import annotations

from typing import Any, Final, Mapping

from torch import nn

from breakout_rl.models.atari_cnn import DEFAULT_INPUT_SHAPE
from breakout_rl.models.dqn import DEFAULT_HIDDEN_DIM, DQNNetwork
from breakout_rl.models.dueling_dqn import DuelingDQNNetwork


SUPPORTED_ARCHITECTURES: Final[tuple[str, ...]] = ("standard", "dueling")


def normalize_architecture(value: str) -> str:
    """Normalize and validate one supported Q-network architecture."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "architecture must be one of " + ", ".join(SUPPORTED_ARCHITECTURES)
        )
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "architecture must be one of " + ", ".join(SUPPORTED_ARCHITECTURES)
        )
    return normalized


def checkpoint_architecture(payload: Mapping[str, Any]) -> str:
    """Resolve architecture metadata without guessing unknown checkpoints.

    Checkpoints written before Day 19 use format version 1 and identify their
    trainer.  Those known legacy DQN formats are explicitly treated as
    standard; an arbitrary payload with missing architecture metadata is
    rejected instead of being silently interpreted.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        model_config = {}
    config = payload.get("config")
    if not isinstance(config, Mapping):
        config = {}
    candidates = [
        ("checkpoint", payload.get("architecture")),
        ("model_config", model_config.get("architecture")),
        ("config", config.get("architecture")),
    ]
    normalized_candidates = [
        (source, normalize_architecture(str(value)))
        for source, value in candidates
        if value is not None
    ]
    if normalized_candidates:
        architectures = {value for _source, value in normalized_candidates}
        if len(architectures) != 1:
            details = ", ".join(
                f"{source}={value!r}" for source, value in normalized_candidates
            )
            raise ValueError(f"checkpoint architecture metadata conflicts: {details}")
        return normalized_candidates[0][1]

    if (
        payload.get("format_version") == 1
        and payload.get("trainer") in {"dqn", "vectorized_dqn"}
        and isinstance(payload.get("online_network"), Mapping)
        and isinstance(payload.get("config"), Mapping)
    ):
        return "standard"
    raise ValueError(
        "checkpoint is missing architecture metadata; only known format_version=1 "
        "DQN checkpoints can use the legacy standard architecture"
    )


def build_q_network(
    architecture: str = "standard",
    *,
    num_actions: int,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
) -> nn.Module:
    """Build a standard or Dueling Q-network from one canonical seam."""

    normalized = normalize_architecture(architecture)
    network_type = DQNNetwork if normalized == "standard" else DuelingDQNNetwork
    return network_type(
        num_actions,
        input_shape=input_shape,
        hidden_dim=hidden_dim,
    )


__all__ = [
    "SUPPORTED_ARCHITECTURES",
    "build_q_network",
    "checkpoint_architecture",
    "normalize_architecture",
]
