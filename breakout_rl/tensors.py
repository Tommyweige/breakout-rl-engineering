"""Conversion helpers for the project's uint8 observation contract."""

from __future__ import annotations

from typing import Final

import numpy as np
import torch


OBSERVATION_SHAPE: Final[tuple[int, int, int]] = (4, 84, 84)
"""The single-state shape emitted by the preprocessed Breakout environment."""


def _resolve_device(device: torch.device | str) -> torch.device:
    """Resolve a device and reject unavailable explicit CUDA requests."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but it is not available in this environment."
        )
    return resolved


def _validate_observation(observation: np.ndarray) -> None:
    """Validate the shape and dtype before transferring observation data."""

    if not isinstance(observation, np.ndarray):
        raise TypeError("observation must be a numpy.ndarray")
    if observation.dtype != np.uint8:
        raise TypeError(
            "observation must have dtype uint8; normalize only at the model boundary"
        )

    if observation.ndim == 3:
        actual_shape = tuple(observation.shape)
    elif observation.ndim == 4:
        actual_shape = tuple(observation.shape[1:])
    else:
        raise ValueError(
            "observation must have shape (4, 84, 84) or (B, 4, 84, 84)"
        )

    if actual_shape != OBSERVATION_SHAPE:
        raise ValueError(
            "observation must have shape (4, 84, 84) or (B, 4, 84, 84); "
            f"received {tuple(observation.shape)}"
        )


def observation_to_tensor(
    observation: np.ndarray,
    *,
    device: torch.device | str,
    add_batch_dim: bool = True,
) -> torch.Tensor:
    """Convert uint8 Breakout pixels to a normalized model-input tensor.

    A single observation has shape ``(4, 84, 84)``. A batch already has shape
    ``(B, 4, 84, 84)`` and is kept at that rank. Pixel values are converted to
    ``float32`` and divided by 255 only at this boundary, so replay storage can
    remain compact ``uint8`` data.
    """

    _validate_observation(observation)
    resolved_device = _resolve_device(device)

    contiguous_observation = np.ascontiguousarray(observation)
    tensor = torch.from_numpy(contiguous_observation).to(
        device=resolved_device,
        dtype=torch.float32,
    )
    tensor = tensor.div(255.0)

    if observation.ndim == 3 and add_batch_dim:
        tensor = tensor.unsqueeze(0)

    return tensor
