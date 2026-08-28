"""Load and validate machine-readable training backend manifests."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from breakout_rl.evaluation_contract import (
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.training.config import DQNConfig


DAY16_BACKEND_MANIFEST_SCHEMA_VERSION = 1
DAY16_CANONICAL_BACKEND_ID = "day16-vectorized-dqn-n2"
DAY16_SELECTED_BACKEND_ROLE = "selected_systems_backend"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _resolve_path(repository_root: Path, value: Any, *, name: str) -> Path:
    raw = Path(_non_empty_string(value, name=name))
    return raw if raw.is_absolute() else repository_root / raw


def _artifact_reference(
    value: Any,
    *,
    name: str,
    repository_root: Path,
    verify_file: bool,
) -> None:
    artifact = _mapping(value, name=name)
    path_value = _non_empty_string(artifact.get("path"), name=f"{name}.path")
    digest = _non_empty_string(artifact.get("sha256"), name=f"{name}.sha256")
    if not _SHA256_PATTERN.fullmatch(digest.lower()):
        raise ValueError(f"{name}.sha256 must be a lowercase SHA-256 digest")
    if verify_file:
        path = _resolve_path(repository_root, path_value, name=f"{name}.path")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest.lower():
            raise ValueError(
                f"{name}.sha256 does not match {path}: expected {digest}, got {actual}"
            )


def validate_day16_backend_manifest(
    payload: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    verify_evidence_files: bool = False,
) -> None:
    """Validate the Day 16 selected backend and its provenance references."""

    if not isinstance(payload, Mapping):
        raise ValueError("backend manifest must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != DAY16_BACKEND_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "backend manifest schema_version must be "
            f"{DAY16_BACKEND_MANIFEST_SCHEMA_VERSION}"
        )
    if payload.get("backend_id") != DAY16_CANONICAL_BACKEND_ID:
        raise ValueError(
            f"backend_id must be {DAY16_CANONICAL_BACKEND_ID!r}"
        )
    if payload.get("source_day") != 16:
        raise ValueError("source_day must be 16")

    root = Path(repository_root) if repository_root is not None else Path.cwd()
    contract_info = _mapping(
        payload.get("environment_contract"),
        name="environment_contract",
    )
    contract_path = _resolve_path(
        root,
        contract_info.get("path"),
        name="environment_contract.path",
    )
    contract_id = _non_empty_string(
        contract_info.get("contract_id"),
        name="environment_contract.contract_id",
    )
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    if contract.contract_id != contract_id:
        raise ValueError(
            "environment_contract.contract_id does not match the contract file"
        )

    trainer = _mapping(payload.get("trainer"), name="trainer")
    if trainer.get("type") != "vectorized_dqn":
        raise ValueError("trainer.type must be 'vectorized_dqn'")
    training_config = _mapping(
        trainer.get("config"),
        name="trainer.config",
    )
    config = DQNConfig.from_dict(training_config)
    expected_trainer_values = {
        "num_envs": config.num_envs,
        "strict_action_selection_parity": config.strict_action_selection_parity,
        "replay_backend": config.replay_backend,
        "device": config.device,
        "precision": config.precision,
        "cpu_threads": config.cpu_threads,
    }
    for field, expected in expected_trainer_values.items():
        if trainer.get(field) != expected:
            raise ValueError(
                f"trainer.{field} does not match trainer.config: "
                f"expected {expected!r}, got {trainer.get(field)!r}"
            )
    if config.num_envs != 2:
        raise ValueError("Day 16 canonical backend requires num_envs=2")
    if not config.strict_action_selection_parity:
        raise ValueError(
            "Day 16 canonical backend requires strict_action_selection_parity=true"
        )
    if config.replay_backend != "gpu":
        raise ValueError("Day 16 canonical backend requires replay_backend='gpu'")
    if config.device != "cuda":
        raise ValueError("Day 16 canonical backend requires device='cuda'")
    if config.precision != "float32":
        raise ValueError("Day 16 canonical backend requires precision='float32'")
    if config.total_steps != 100_000:
        raise ValueError("Day 16 canonical backend requires total_steps=100000")

    selection = _mapping(payload.get("selection"), name="selection")
    if selection.get("role") != DAY16_SELECTED_BACKEND_ROLE:
        raise ValueError(
            f"selection.role must be {DAY16_SELECTED_BACKEND_ROLE!r}"
        )

    evidence = _mapping(payload.get("evidence"), name="evidence")
    for name in (
        "training_report",
        "evaluation_results",
        "checkpoint",
        "evaluation_summary",
    ):
        _artifact_reference(
            evidence.get(name),
            name=f"evidence.{name}",
            repository_root=root,
            verify_file=verify_evidence_files,
        )


def load_day16_backend_manifest(
    path: str | Path = "configs/training/day16-canonical-backend.json",
    *,
    repository_root: str | Path | None = None,
    verify_evidence_files: bool = False,
) -> dict[str, Any]:
    """Load and validate a Day 16 backend manifest."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if repository_root is None:
        repository_root = source.parent.parent.parent
    validate_day16_backend_manifest(
        payload,
        repository_root=repository_root,
        verify_evidence_files=verify_evidence_files,
    )
    return dict(payload)


__all__ = [
    "DAY16_BACKEND_MANIFEST_SCHEMA_VERSION",
    "DAY16_CANONICAL_BACKEND_ID",
    "DAY16_SELECTED_BACKEND_ROLE",
    "load_day16_backend_manifest",
    "validate_day16_backend_manifest",
]
