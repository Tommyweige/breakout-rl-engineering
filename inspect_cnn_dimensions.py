"""Inspect real Breakout observations as they pass through the Atari CNN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from breakout_env import make_breakout_env
from breakout_rl.models import AtariFeatureExtractor
from breakout_rl.tensors import observation_to_tensor


@dataclass(frozen=True)
class CnnInspection:
    """Runtime evidence collected from one seeded environment/model forward."""

    device: torch.device
    observation: np.ndarray
    model_input: torch.Tensor
    features: torch.Tensor
    shapes: dict[str, tuple[int, ...]]
    activations: dict[str, torch.Tensor]


def resolve_device(device_name: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` without hiding CUDA errors."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but it is not available in this environment."
            )
        return torch.device("cuda")
    raise ValueError("device_name must be one of: auto, cpu, cuda")


def collect_cnn_inspection(
    *,
    seed: int = 42,
    device_name: str = "auto",
) -> CnnInspection:
    """Collect observation, shapes, and activations from a real forward pass."""

    device = resolve_device(device_name)
    torch.manual_seed(seed)

    environment = make_breakout_env()
    try:
        observation, _ = environment.reset(seed=seed)
        observation = np.asarray(observation).copy()
    finally:
        environment.close()

    model = AtariFeatureExtractor().to(device).eval()
    model_input = observation_to_tensor(observation, device=device)

    with torch.inference_mode():
        features, shapes, activations = model.forward_features_with_activations(
            model_input
        )

    return CnnInspection(
        device=device,
        observation=observation,
        model_input=model_input,
        features=features,
        shapes=shapes,
        activations=activations,
    )


def parse_args() -> argparse.Namespace:
    """Parse options for the dimension inspection CLI."""

    parser = argparse.ArgumentParser(
        description="Inspect real Breakout tensor shapes through the Atari CNN."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device; auto selects CUDA when available (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used for the Breakout reset and model initialization (default: 42)",
    )
    return parser.parse_args()


def _format_shape(shape: tuple[int, ...]) -> str:
    return str(shape)


def print_inspection(inspection: CnnInspection) -> None:
    """Print a compact, runtime-derived dimension report."""

    observation = inspection.observation
    model_input = inspection.model_input
    print(f"Device                  : {inspection.device}")
    print(
        "Environment observation : "
        f"{_format_shape(tuple(observation.shape))} "
        f"{observation.dtype}, range={int(observation.min())}..{int(observation.max())}"
    )
    print(
        "Model input             : "
        f"{_format_shape(tuple(model_input.shape))} "
        f"{model_input.dtype}, range={float(model_input.min()):.4f}.."
        f"{float(model_input.max()):.4f}"
    )

    for name in ("conv1", "conv2", "conv3", "flatten"):
        print(f"{name.title():<24}: {_format_shape(inspection.shapes[name])}")

    print(f"Feature dimension       : {inspection.features.shape[-1]}")


def main() -> None:
    """Run the CNN dimension inspection CLI."""

    args = parse_args()
    try:
        inspection = collect_cnn_inspection(
            seed=args.seed,
            device_name=args.device,
        )
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error

    print_inspection(inspection)


if __name__ == "__main__":
    main()
