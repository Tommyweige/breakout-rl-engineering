"""Reusable lineage and checkpoint loading helpers for deployment experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from breakout_rl.artifacts import repository_relative_path, sha256_file
from breakout_rl.evaluation_contract import (
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.inference import InferenceSpec
from breakout_rl.models.factory import build_q_network, checkpoint_architecture


def _json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: expected a JSON object")
    return dict(value)


def source_identity(
    source_model: str | Path,
    source_metadata: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Validate and return the Day 21 canonical model identity."""

    metadata = _json_object(source_metadata)
    model_hash = sha256_file(source_model)
    if metadata.get("model_sha256") != model_hash:
        raise ValueError(
            "source model hash does not match Day 21 metadata: "
            f"declared={metadata.get('model_sha256')!r}, observed={model_hash}"
        )
    if metadata.get("artifact_type") != "canonical_final_model":
        raise ValueError("source metadata is not the Day 21 canonical final model")
    source_checkpoint = metadata.get("source_checkpoint")
    if not isinstance(source_checkpoint, Mapping):
        raise ValueError("Day 21 metadata is missing source checkpoint provenance")
    checkpoint_hash = source_checkpoint.get("sha256")
    if (
        not isinstance(checkpoint_hash, str)
        or len(checkpoint_hash) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_hash.lower())
    ):
        raise ValueError("Day 21 metadata has no valid source checkpoint SHA256")
    identity = {
        "model_path": repository_relative_path(source_model, root=root),
        "model_sha256": model_hash,
        "algorithm": metadata.get("algorithm"),
        "architecture": metadata.get("architecture"),
        "hidden_dim": metadata.get("hidden_dim"),
        "observation_shape": metadata.get("observation_shape"),
        "num_actions": metadata.get("num_actions"),
        "parameter_count": metadata.get("parameter_count"),
        "training_seed": metadata.get("training_seed"),
        "training_transitions": metadata.get("training_transitions"),
        "source_run_id": metadata.get("source_run_id"),
        "source_stage": metadata.get("source_stage"),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_checkpoint_step": source_checkpoint.get("step"),
    }
    required = (
        "algorithm",
        "architecture",
        "hidden_dim",
        "observation_shape",
        "num_actions",
        "parameter_count",
        "source_run_id",
        "source_stage",
    )
    missing = [name for name in required if identity.get(name) is None]
    if missing:
        raise ValueError("Day 21 source metadata is missing model identity fields: " + ", ".join(missing))
    return identity


def validate_environment_contract(spec: InferenceSpec, *, root: str | Path) -> Path:
    """Validate the Contract v2 artifact referenced by an inference spec."""

    contract_path = (Path(root) / spec.environment_contract_path).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    observed_hash = sha256_file(contract_path)
    if observed_hash != spec.environment_contract_sha256:
        raise ValueError(
            "inference spec is bound to a different Contract v2 artifact: "
            f"declared={spec.environment_contract_sha256}, observed={observed_hash}"
        )
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    if contract.contract_id != spec.environment_contract_id:
        raise ValueError("inference spec environment contract id does not match the loaded contract")
    return contract_path


def load_deployment_model(
    source_model: str | Path,
    *,
    device: torch.device,
    identity: Mapping[str, Any],
    spec: InferenceSpec,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Load the canonical model and validate its architecture/provenance contract."""

    payload = torch.load(source_model, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Day 21 model checkpoint must contain a mapping")
    architecture = checkpoint_architecture(payload)
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("Day 21 model checkpoint is missing model_config")
    input_shape = tuple(int(value) for value in model_config.get("input_shape", ()))
    hidden_dim = int(model_config.get("hidden_dim", 0))
    num_actions = int(model_config.get("num_actions", 0))
    if input_shape != tuple(spec.source_observation_shape):
        raise ValueError(
            "source model input shape does not match inference spec: "
            f"model={input_shape}, spec={spec.source_observation_shape}"
        )
    if num_actions != len(spec.action_meanings):
        raise ValueError(
            "source model action count does not match inference spec: "
            f"model={num_actions}, spec={len(spec.action_meanings)}"
        )
    if architecture != identity["architecture"] or hidden_dim != int(identity["hidden_dim"]):
        raise ValueError("source model architecture does not match Day 21 metadata")
    if model_config.get("architecture") != architecture:
        raise ValueError("source model_config architecture conflicts with checkpoint metadata")
    if payload.get("algorithm") != identity["algorithm"]:
        raise ValueError("source model algorithm does not match Day 21 metadata")
    if payload.get("global_step") != identity["source_checkpoint_step"]:
        raise ValueError("source model checkpoint step does not match Day 21 metadata")
    if payload.get("contract_id") != spec.environment_contract_id:
        raise ValueError("source model Contract v2 id does not match the inference spec")
    payload_metadata = payload.get("metadata")
    if isinstance(payload_metadata, Mapping):
        payload_source_checkpoint = payload_metadata.get("source_checkpoint")
        if (
            isinstance(payload_source_checkpoint, Mapping)
            and payload_source_checkpoint.get("sha256") != identity["source_checkpoint_sha256"]
        ):
            raise ValueError("source checkpoint provenance conflicts inside the model payload")
    state_dict = payload.get("online_network")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Day 21 model checkpoint is missing online_network")
    model = build_q_network(
        architecture,
        num_actions=num_actions,
        input_shape=input_shape,  # type: ignore[arg-type]
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    observed_parameters = sum(parameter.numel() for parameter in model.parameters())
    if observed_parameters != int(identity["parameter_count"]):
        raise ValueError(
            "source model parameter count does not match Day 21 metadata: "
            f"model={observed_parameters}, metadata={identity['parameter_count']}"
        )
    return model, dict(payload), dict(model_config)


__all__ = ["load_deployment_model", "source_identity", "validate_environment_contract"]
