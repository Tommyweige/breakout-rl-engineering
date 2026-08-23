"""Reusable data, model, and tensor utilities for the Breakout RL project."""

from typing import Any

__all__ = [
    "ReplayBuffer",
    "ReplayTensorBatch",
    "TransitionBatch",
    "estimate_replay_memory_bytes",
    "observation_to_tensor",
    "replay_batch_to_tensors",
]


def __getattr__(name: str) -> Any:
    """Load optional tensor-heavy helpers only when a caller requests them."""

    if name in {"ReplayBuffer", "TransitionBatch", "estimate_replay_memory_bytes"}:
        from breakout_rl.replay import (
            ReplayBuffer,
            TransitionBatch,
            estimate_replay_memory_bytes,
        )

        return {
            "ReplayBuffer": ReplayBuffer,
            "TransitionBatch": TransitionBatch,
            "estimate_replay_memory_bytes": estimate_replay_memory_bytes,
        }[name]

    if name in {"ReplayTensorBatch", "replay_batch_to_tensors"}:
        from breakout_rl.replay_tensors import (
            ReplayTensorBatch,
            replay_batch_to_tensors,
        )

        return {
            "ReplayTensorBatch": ReplayTensorBatch,
            "replay_batch_to_tensors": replay_batch_to_tensors,
        }[name]

    if name == "observation_to_tensor":
        from breakout_rl.tensors import observation_to_tensor

        return observation_to_tensor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
