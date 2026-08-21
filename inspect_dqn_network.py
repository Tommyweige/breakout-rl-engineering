"""Inspect an untrained DQN on one real preprocessed Breakout state."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from breakout_env import make_breakout_env
from breakout_rl.models import DQNNetwork
from breakout_rl.tensors import observation_to_tensor


@dataclass(frozen=True)
class DQNInspection:
    """Runtime values collected from one seeded environment/model forward."""

    device: torch.device
    observation: np.ndarray
    model_input: torch.Tensor
    features: torch.Tensor
    q_values: torch.Tensor
    action_meanings: tuple[str, ...]
    greedy_action_index: int
    parameter_count: int
    state_dict_roundtrip_max_abs_diff: float


def resolve_device(device_name: str) -> torch.device:
    """Resolve auto/cpu/cuda without silently hiding unavailable CUDA."""

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


def _state_dict_roundtrip_max_abs_diff(
    model: DQNNetwork,
    model_input: torch.Tensor,
) -> float:
    """Save/load a temporary state_dict and confirm the same forward output."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "dqn-state-dict.pt"
        torch.save(model.state_dict(), path)

        clone = DQNNetwork(num_actions=model.num_actions).to(model_input.device).eval()
        state_dict = torch.load(path, map_location=model_input.device, weights_only=True)
        clone.load_state_dict(state_dict)

        with torch.inference_mode():
            expected = model(model_input)
            actual = clone(model_input)

    return float((expected - actual).abs().max().item())


def collect_dqn_inspection(
    *,
    seed: int = 42,
    device_name: str = "auto",
) -> DQNInspection:
    """Run one real Breakout observation through an untrained DQN."""

    device = resolve_device(device_name)
    torch.manual_seed(seed)

    environment = make_breakout_env()
    try:
        observation, _ = environment.reset(seed=seed)
        observation = np.asarray(observation).copy()
        num_actions = int(environment.action_space.n)
        action_meanings = tuple(environment.unwrapped.get_action_meanings())
    finally:
        environment.close()

    if len(action_meanings) != num_actions:
        raise RuntimeError(
            "Environment action meanings do not match action_space.n: "
            f"{len(action_meanings)} meanings vs {num_actions} actions"
        )

    model = DQNNetwork(num_actions=num_actions).to(device).eval()
    model_input = observation_to_tensor(observation, device=device)

    with torch.inference_mode():
        features = model.forward_features(model_input)
        q_values = model(model_input)

    greedy_action_index = int(q_values.argmax(dim=1).item())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    roundtrip_diff = _state_dict_roundtrip_max_abs_diff(model, model_input)

    return DQNInspection(
        device=device,
        observation=observation,
        model_input=model_input,
        features=features,
        q_values=q_values,
        action_meanings=action_meanings,
        greedy_action_index=greedy_action_index,
        parameter_count=parameter_count,
        state_dict_roundtrip_max_abs_diff=roundtrip_diff,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one real Breakout state through an untrained DQN."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="forward device; auto selects CUDA when available (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for the environment reset and model initialization (default: 42)",
    )
    return parser.parse_args()


def print_inspection(inspection: DQNInspection) -> None:
    q_values = inspection.q_values[0].detach().cpu().tolist()
    print(f"Device             : {inspection.device}")
    print(
        "Observation        : "
        f"{tuple(inspection.observation.shape)} {inspection.observation.dtype}"
    )
    print(
        "Model input        : "
        f"{tuple(inspection.model_input.shape)} {inspection.model_input.dtype}"
    )
    print(f"Feature shape      : {tuple(inspection.features.shape)}")
    print(f"Output shape       : {tuple(inspection.q_values.shape)}")
    print(f"Action meanings    : {' '.join(inspection.action_meanings)}")
    print(
        "Q-values           : "
        + "["
        + ", ".join(f"{value:+.6f}" for value in q_values)
        + "]"
    )
    print(
        "Greedy action      : "
        f"{inspection.greedy_action_index} "
        f"({inspection.action_meanings[inspection.greedy_action_index]})"
    )
    print(f"Parameter count    : {inspection.parameter_count:,}")
    print(
        "state_dict diff    : "
        f"{inspection.state_dict_roundtrip_max_abs_diff:.8f}"
    )
    print(
        "Interpretation     : untrained random-weight outputs; "
        "the greedy action is not a learned policy"
    )


def main() -> None:
    args = parse_args()
    try:
        inspection = collect_dqn_inspection(seed=args.seed, device_name=args.device)
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error
    print_inspection(inspection)


if __name__ == "__main__":
    main()
