"""Reusable Contract v2 inspection for Dueling DQN value decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from breakout_env import make_breakout_env
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.models import DuelingDQNNetwork, build_q_network, checkpoint_architecture
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.training.dqn_trainer import resolve_device, seed_everything


DEFAULT_CONTRACT = Path("configs/eval/breakout_contract_v2.json")


@dataclass(frozen=True)
class DuelingInspection:
    """Real observation and model values collected for teaching/diagnostics."""

    device: torch.device
    contract: BreakoutEvaluationContractV2
    observation: np.ndarray
    model_input: torch.Tensor
    features: torch.Tensor
    value: torch.Tensor
    advantage: torch.Tensor
    centered_advantage: torch.Tensor
    q_values: torch.Tensor
    action_meanings: tuple[str, ...]
    greedy_action_index: int
    hidden_dim: int
    parameter_count: int
    reconstruction_max_abs_error: float
    checkpoint: str | None
    checkpoint_metadata: Mapping[str, Any]


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    return payload


def _model_for_inspection(
    *,
    num_actions: int,
    input_shape: tuple[int, int, int],
    device: torch.device,
    checkpoint: Path | None,
) -> tuple[DuelingDQNNetwork, dict[str, Any]]:
    if checkpoint is None:
        model = build_q_network(
            "dueling",
            num_actions=num_actions,
            input_shape=input_shape,
        )
        if not isinstance(model, DuelingDQNNetwork):
            raise AssertionError("dueling factory returned an unexpected network")
        return model.to(device), {}

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = _load_payload(checkpoint)
    state_dict = payload.get("online_network")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint does not contain online_network")
    model_config = payload.get("model_config", {})
    if not isinstance(model_config, Mapping):
        model_config = {}
    architecture = checkpoint_architecture(payload)
    if architecture != "dueling":
        raise ValueError(
            "inspect_dueling_network requires a checkpoint with architecture='dueling'"
        )
    saved_actions = model_config.get("num_actions", num_actions)
    if int(saved_actions) != num_actions:
        raise ValueError("checkpoint num_actions does not match the environment")
    raw_shape = model_config.get("input_shape", input_shape)
    saved_shape = tuple(int(value) for value in raw_shape)
    if saved_shape != input_shape:
        raise ValueError("checkpoint input_shape does not match the environment")
    hidden_value = model_config.get("hidden_dim")
    hidden_dim = 512 if hidden_value is None else int(hidden_value)
    model = build_q_network(
        architecture,
        num_actions=num_actions,
        input_shape=input_shape,
        hidden_dim=hidden_dim,
    )
    if not isinstance(model, DuelingDQNNetwork):
        raise AssertionError("dueling factory returned an unexpected network")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint does not match the Dueling architecture") from error
    return model.to(device), {
        "path": checkpoint.as_posix(),
        "step": payload.get("global_step"),
        "algorithm": payload.get("algorithm"),
        "architecture": architecture,
        "contract_id": payload.get("contract_id"),
        "contract_path": payload.get("contract_path"),
    }


def collect_dueling_inspection(
    *,
    seed: int = 42,
    device_name: str = "auto",
    contract_path: str | Path = DEFAULT_CONTRACT,
    checkpoint: str | Path | None = None,
) -> DuelingInspection:
    """Run one real Contract v2 observation through a Dueling network."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    device = resolve_device(device_name)
    seed_everything(seed)
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)

    environment = make_breakout_env(**breakout_environment_kwargs(contract))
    try:
        observation, _ = environment.reset(seed=seed)
        observation = np.asarray(observation).copy()
        num_actions = int(environment.action_space.n)
        action_meanings = tuple(environment.unwrapped.get_action_meanings())
        input_shape = tuple(int(value) for value in environment.observation_space.shape)
    finally:
        environment.close()

    if observation.dtype != np.uint8 or tuple(observation.shape) != input_shape:
        raise ValueError(
            "Contract v2 observation must be uint8 with shape "
            f"{input_shape}; got {observation.dtype} {tuple(observation.shape)}"
        )
    if len(action_meanings) != num_actions:
        raise ValueError("environment action meanings do not match action_space.n")

    checkpoint_path = None if checkpoint is None else Path(checkpoint)
    model, checkpoint_metadata = _model_for_inspection(
        num_actions=num_actions,
        input_shape=input_shape,  # type: ignore[arg-type]
        device=device,
        checkpoint=checkpoint_path,
    )
    model.eval()
    model_input = observation_to_tensor(observation, device=device)
    with torch.inference_mode():
        features = model.forward_features(model_input)
        value, advantage, q_values = model.forward_components(model_input)
        centered_advantage = advantage - advantage.mean(dim=1, keepdim=True)
        reconstructed = value + centered_advantage
    reconstruction_error = float((q_values - reconstructed).abs().max().item())
    if not torch.isfinite(q_values).all().item():
        raise ValueError("Dueling Q-values contain non-finite values")

    return DuelingInspection(
        device=device,
        contract=contract,
        observation=observation,
        model_input=model_input,
        features=features,
        value=value,
        advantage=advantage,
        centered_advantage=centered_advantage,
        q_values=q_values,
        action_meanings=action_meanings,
        greedy_action_index=int(q_values.argmax(dim=1).item()),
        hidden_dim=int(model.hidden_dim),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        reconstruction_max_abs_error=reconstruction_error,
        checkpoint=None if checkpoint_path is None else checkpoint_path.as_posix(),
        checkpoint_metadata=checkpoint_metadata,
    )


def _row_values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor[0].detach().cpu().tolist()]


def inspection_payload(inspection: DuelingInspection, *, seed: int) -> dict[str, Any]:
    """Serialize inspection values without moving tensors during training."""

    return {
        "schema_version": 1,
        "seed": seed,
        "device": str(inspection.device),
        "observation_source": "seeded real Breakout observation under Contract v2",
        "contract": inspection.contract.to_dict(),
        "checkpoint": inspection.checkpoint,
        "checkpoint_metadata": dict(inspection.checkpoint_metadata),
        "model_config": {
            "algorithm": inspection.checkpoint_metadata.get("algorithm")
            if inspection.checkpoint
            else "untrained",
            "architecture": "dueling",
            "num_actions": len(inspection.action_meanings),
            "input_shape": list(inspection.observation.shape),
            "feature_shape": list(inspection.features.shape),
            "hidden_dim": inspection.hidden_dim,
            "parameter_count": inspection.parameter_count,
        },
        "observation_shape": list(inspection.observation.shape),
        "observation_dtype": str(inspection.observation.dtype),
        "model_input_shape": list(inspection.model_input.shape),
        "feature_shape": list(inspection.features.shape),
        "value_shape": list(inspection.value.shape),
        "advantage_shape": list(inspection.advantage.shape),
        "q_shape": list(inspection.q_values.shape),
        "action_meanings": list(inspection.action_meanings),
        "value": float(inspection.value[0, 0].detach().cpu().item()),
        "raw_advantage": _row_values(inspection.advantage),
        "mean_advantage": float(inspection.advantage.mean().detach().cpu().item()),
        "centered_advantage": _row_values(inspection.centered_advantage),
        "q_values": _row_values(inspection.q_values),
        "reconstruction_max_abs_error": inspection.reconstruction_max_abs_error,
        "argmax_action_index": inspection.greedy_action_index,
        "argmax_action_meaning": inspection.action_meanings[
            inspection.greedy_action_index
        ],
        "trained_policy_claim": (
            "checkpoint output is an infrastructure/representation smoke result; "
            "it is not a model-quality claim"
            if inspection.checkpoint
            else "untrained random-weight outputs; argmax is not a learned policy"
        ),
    }


__all__ = [
    "DEFAULT_CONTRACT",
    "DuelingInspection",
    "collect_dueling_inspection",
    "inspection_payload",
]
