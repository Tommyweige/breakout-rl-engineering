"""One owner for DQN-family checkpoint files: read, write, RNG state, network restore.

Payload keys stay owned by each trainer's ``save_checkpoint``; this module owns
how the file is stored and how a recorded online network is rebuilt.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from breakout_rl.models.factory import build_q_network, checkpoint_architecture
from breakout_rl.training.config import normalize_algorithm


def read_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint payload and require it to be a mapping."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch builds do not accept ``weights_only``.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint must contain a mapping")
    return payload


def write_checkpoint(payload: Mapping[str, Any], path: Path) -> None:
    """Write atomically so an interrupted save never leaves a truncated ``.pt``."""

    temporary_path = path.with_suffix(".tmp")
    torch.save(dict(payload), temporary_path)
    os.replace(temporary_path, path)


def capture_rng_state(action_rng: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "action_rng": action_rng.bit_generator.state,
    }


def restore_rng_state(
    state: Mapping[str, Any],
    action_rng: np.random.Generator,
) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy_global" in state:
        np.random.set_state(state["numpy_global"])
    if "torch_cpu" in state:
        cpu_state = state["torch_cpu"]
        if not isinstance(cpu_state, torch.Tensor):
            raise ValueError("checkpoint torch_cpu RNG state must be a tensor")
        torch.set_rng_state(cpu_state.detach().to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        raw_cuda_states = state["torch_cuda"]
        if not isinstance(raw_cuda_states, (list, tuple)):
            raise ValueError("checkpoint torch_cuda RNG state must be a sequence")
        cuda_states = [
            value.detach().to(device="cpu", dtype=torch.uint8)
            if isinstance(value, torch.Tensor)
            else value
            for value in raw_cuda_states
        ]
        if not all(isinstance(value, torch.Tensor) for value in cuda_states):
            raise ValueError("checkpoint torch_cuda RNG state must contain tensors")
        torch.cuda.set_rng_state_all(cuda_states)
    if "action_rng" in state:
        action_rng.bit_generator.state = state["action_rng"]


def check_resume_identity(
    payload: Mapping[str, Any],
    *,
    algorithm: str,
    architecture: str,
    allow_replay_rewarm: bool,
) -> None:
    """Reject a resume whose replay promise, algorithm, or architecture differs."""

    checkpoint_advertises_replay = bool(payload.get("replay_saved", False))
    if checkpoint_advertises_replay and not allow_replay_rewarm:
        raise ValueError(
            "checkpoint advertises replay state, but this trainer does not "
            "restore replay contents or environment/ALE state; exact "
            "continuation is unavailable. Pass allow_replay_rewarm=True only "
            "for an explicit non-equivalent warm-start."
        )
    if not checkpoint_advertises_replay and not allow_replay_rewarm:
        raise ValueError(
            "checkpoint does not contain replay state; exact continuation is "
            "unavailable. Pass allow_replay_rewarm=True only for an explicit "
            "non-equivalent warm-start."
        )

    saved_algorithm = payload.get("algorithm")
    config_payload = payload.get("config")
    if saved_algorithm is None and isinstance(config_payload, Mapping):
        saved_algorithm = config_payload.get("algorithm", "dqn")
    if (
        saved_algorithm is not None
        and normalize_algorithm(str(saved_algorithm)) != algorithm
    ):
        raise ValueError(
            "checkpoint algorithm does not match trainer config: "
            f"checkpoint={saved_algorithm!r}, config={algorithm!r}"
        )

    saved_architecture = checkpoint_architecture(payload)
    if saved_architecture != architecture:
        raise ValueError(
            "checkpoint architecture does not match trainer config: "
            f"checkpoint={saved_architecture!r}, config={architecture!r}"
        )


def restore_q_network(
    state_dict: Mapping[str, Any],
    *,
    architecture: str,
    num_actions: int,
    input_shape: Sequence[int],
    hidden_dim: int,
    device: torch.device | str,
) -> nn.Module:
    """Rebuild a recorded online network in eval mode; strict key matching."""

    model = build_q_network(
        architecture,
        num_actions=num_actions,
        input_shape=tuple(input_shape),  # type: ignore[arg-type]
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


__all__ = [
    "capture_rng_state",
    "check_resume_identity",
    "read_checkpoint",
    "restore_q_network",
    "restore_rng_state",
    "write_checkpoint",
]
