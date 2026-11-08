"""Reusable frozen-policy evaluation for the Breakout DQN."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import platform
import re
import subprocess
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TimeLimit
from torch import nn

from breakout_env import BreakoutFireResetWrapper, ENVIRONMENT_ID, make_breakout_env
from breakout_rl.completion import (
    BREAKOUT_AUDIT_CONFIG_PATH,
    BreakoutCompletionDetector,
    inspect_breakout_completion_support,
    read_ale_episode_frame,
    read_ale_lives,
    read_breakout_score,
)
from breakout_rl.checkpoint import read_checkpoint, restore_q_network
from breakout_rl.models.factory import checkpoint_architecture
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.evaluation_artifacts import (
    EVALUATION_ARTIFACT_SCHEMA_VERSION,
    EpisodeResult,
    EvaluationResult,
    read_evaluation_results,
    summarize_returns,
    write_evaluation_artifacts,
)
from breakout_rl.artifacts import portable_artifact_path
from breakout_rl.day14_provenance import (
    load_day14_provenance,
    validate_checkpoint_provenance,
)
from breakout_rl.evaluation_contract import (
    BREAKOUT_CONTRACT_V2_ID,
    BREAKOUT_CONTRACT_V3_ID,
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    validate_breakout_runtime_contract,
)
from breakout_rl.training.diagnostics import ATARI_ACTION_NAMES
from breakout_rl.training.dqn_trainer import resolve_device
from breakout_rl.training.survival import compute_episode_survival_metrics


EVALUATION_CONFIG_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = EVALUATION_ARTIFACT_SCHEMA_VERSION
EnvironmentFactory = Callable[[], Any]


def _capture_source_provenance(repository_root: Path) -> dict[str, Any]:
    """Record the evaluator's Git revision when it runs inside a checkout."""

    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        revision = None
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        working_tree_dirty: bool | None = bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        working_tree_dirty = None
    source_files = (
        Path("breakout_env.py"),
        Path("breakout_rl/completion.py"),
        Path("breakout_rl/evaluation.py"),
        Path("breakout_rl/evaluation_artifacts.py"),
        Path("breakout_rl/evaluation_contract.py"),
        Path("configs/eval/breakout_completion_audit_v1.json"),
    )
    source_digest = hashlib.sha256()
    try:
        for relative_path in source_files:
            source_digest.update(relative_path.as_posix().encode("utf-8"))
            source_digest.update(b"\0")
            source_digest.update((repository_root / relative_path).read_bytes())
            source_digest.update(b"\0")
        completion_source_sha256: str | None = source_digest.hexdigest()
    except OSError:
        completion_source_sha256 = None
    return {
        "source_commit": revision or None,
        "working_tree_dirty": working_tree_dirty,
        "completion_source_sha256": completion_source_sha256,
        "completion_source_files": [path.as_posix() for path in source_files],
    }


def _contract_provenance(
    metadata: Mapping[str, Any] | None,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    values = metadata or {}
    path_value = values.get("evaluation_contract_path")
    contract_value = values.get("evaluation_contract")
    provenance: dict[str, Any] = {
        "contract_id": None,
        "contract_path": None,
        "contract_sha256": None,
        "hash_semantics": "SHA-256 of the exact contract file bytes",
        "definition_status": "not_supplied",
        "definition_reason": "No explicit evaluation contract was supplied.",
    }
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        return provenance
    path = Path(path_value)
    if not path.is_absolute():
        path = repository_root / path
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
        file_payload = parsed if isinstance(parsed, Mapping) else {}
        contract_sha256: str | None = hashlib.sha256(raw).hexdigest()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        file_payload = {}
        contract_sha256 = None
    try:
        relative_path = path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        relative_path = path.as_posix()
    provenance.update(
        {
            "contract_id": file_payload.get("contract_id"),
            "contract_path": relative_path,
            "contract_sha256": contract_sha256,
            "definition_status": "invalid",
            "definition_reason": "The explicit contract file or metadata is invalid.",
        }
    )
    if not isinstance(contract_value, Mapping):
        provenance["definition_reason"] = (
            "The contract file was supplied without its validated contract object."
        )
        return provenance

    try:
        file_contract = BreakoutEvaluationContractV2.from_mapping(file_payload)
        metadata_contract = BreakoutEvaluationContractV2.from_mapping(contract_value)
        validate_breakout_runtime_contract(
            file_contract,
            allow_contract_v3=True,
        )
        validate_breakout_runtime_contract(
            metadata_contract,
            allow_contract_v3=True,
        )
    except (TypeError, ValueError):
        return provenance
    if file_contract.to_dict() != metadata_contract.to_dict():
        provenance["definition_reason"] = (
            "The supplied contract object does not match the contract file."
        )
        return provenance

    try:
        audit = json.loads(BREAKOUT_AUDIT_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        audit = {}
    audit_environment = (
        audit.get("environment", {}) if isinstance(audit, Mapping) else {}
    )
    if not isinstance(audit_environment, Mapping):
        audit_environment = {}
    expected_contract_id = audit_environment.get("contract_id")
    expected_contract_sha256 = audit_environment.get("contract_sha256")
    expected_contract_path = None
    if file_contract.contract_id == BREAKOUT_CONTRACT_V2_ID:
        validation_status = "canonical_contract_v2"
    elif file_contract.contract_id == BREAKOUT_CONTRACT_V3_ID:
        validation_status = "canonical_contract_v3"
        additional_contracts = audit.get("additional_contracts", [])
        if isinstance(additional_contracts, Sequence) and not isinstance(
            additional_contracts, (str, bytes)
        ):
            matching_contract = next(
                (
                    item
                    for item in additional_contracts
                    if isinstance(item, Mapping)
                    and item.get("contract_id") == file_contract.contract_id
                ),
                None,
            )
        else:
            matching_contract = None
        if matching_contract is None:
            expected_contract_id = None
            expected_contract_sha256 = None
        else:
            expected_contract_id = matching_contract.get("contract_id")
            expected_contract_sha256 = matching_contract.get("contract_sha256")
            expected_contract_path = matching_contract.get("contract_file")
    else:
        validation_status = None
        expected_contract_id = None
        expected_contract_sha256 = None
    if (
        validation_status is None
        or file_contract.contract_id != expected_contract_id
        or contract_sha256 != expected_contract_sha256
        or (
            file_contract.contract_id == BREAKOUT_CONTRACT_V3_ID
            and not isinstance(expected_contract_path, str)
        )
        or (
            expected_contract_path is not None
            and relative_path != expected_contract_path
        )
    ):
        provenance["definition_reason"] = (
            "The explicit contract does not match an audited Breakout contract "
            "identity, path, and hash."
        )
        return provenance

    provenance.update(
        {
            "contract_id": file_contract.contract_id,
            "definition_status": "validated",
            "definition_reason": None,
            "validation_status": validation_status,
        }
    )
    return provenance


def _contract_runtime_binding(
    env: Any,
    metadata: Mapping[str, Any] | None,
    contract_provenance: Mapping[str, Any],
    *,
    evaluation_seeds: Sequence[int],
    episodes_per_seed: int,
    epsilon: float,
) -> tuple[bool, str | None]:
    """Confirm the live wrappers and episode protocol match the pinned contract."""

    contract_label = "Breakout contract"

    if contract_provenance.get("definition_status") != "validated":
        return False, str(
            contract_provenance.get("definition_reason")
            or "The canonical Breakout contract definition was not validated."
        )
    values = metadata or {}
    contract_value = values.get("evaluation_contract")
    if not isinstance(contract_value, Mapping):
        return False, "The validated Breakout contract object is unavailable."
    try:
        contract = BreakoutEvaluationContractV2.from_mapping(contract_value)
        validate_breakout_runtime_contract(contract, allow_contract_v3=True)
    except (TypeError, ValueError) as error:
        return False, f"The Breakout contract object is invalid: {error}"
    contract_label = (
        "Contract v3"
        if contract.contract_id == BREAKOUT_CONTRACT_V3_ID
        else "Contract v2"
    )

    if tuple(evaluation_seeds) != contract.concrete_episode_seeds:
        return False, f"The evaluation seed list does not match {contract_label}."
    if episodes_per_seed != 1:
        return False, f"Canonical {contract_label} evaluation requires one episode per seed."
    if epsilon != contract.evaluation_epsilon:
        return False, f"The evaluation epsilon does not match {contract_label}."

    if not isinstance(env, BreakoutFireResetWrapper):
        return False, f"The runtime is missing the {contract_label} FIRE-reset wrapper."
    fire_wrapper = env
    stack_wrapper = getattr(fire_wrapper, "env", None)
    if not isinstance(stack_wrapper, FrameStackObservation):
        return False, f"The runtime is missing the {contract_label} frame-stack wrapper."
    preprocessing = getattr(stack_wrapper, "env", None)
    if not isinstance(preprocessing, AtariPreprocessing):
        return False, f"The runtime is missing the {contract_label} Atari preprocessing wrapper."

    expected_kwargs = breakout_environment_kwargs(
        contract,
        allow_contract_v3=True,
    )
    if stack_wrapper.stack_size != contract.frame_stack:
        return False, f"The runtime frame stack differs from {contract_label}."
    try:
        runtime_action_count = _action_count(env)
        runtime_action_names = _action_names(env, runtime_action_count)
        runtime_observation_shape = _observation_shape(env)
    except (TypeError, ValueError) as error:
        return False, f"The runtime observation/action contract is invalid: {error}"
    observation_dtype = getattr(
        getattr(env, "observation_space", None),
        "dtype",
        None,
    )
    try:
        observation_is_uint8 = (
            observation_dtype is not None
            and np.dtype(observation_dtype) == np.dtype(np.uint8)
        )
    except (TypeError, ValueError):
        observation_is_uint8 = False
    if (
        runtime_action_count != 4
        or runtime_action_names != ("NOOP", "FIRE", "RIGHT", "LEFT")
        or runtime_observation_shape != (contract.frame_stack, 84, 84)
        or not observation_is_uint8
    ):
        return False, f"The runtime observation/action contract differs from {contract_label}."
    if (
        preprocessing.frame_skip != contract.frame_skip
        or preprocessing.terminal_on_life_loss != contract.terminal_on_life_loss
        or tuple(preprocessing.screen_size) != (84, 84)
        or not preprocessing.grayscale_obs
        or preprocessing.scale_obs
    ):
        return False, f"The runtime preprocessing semantics differ from {contract_label}."
    wrapper = getattr(preprocessing, "env", None)
    visited_wrappers: set[int] = set()
    while wrapper is not None and id(wrapper) not in visited_wrappers:
        if isinstance(wrapper, TimeLimit):
            return False, "The runtime contains an external TimeLimit wrapper."
        visited_wrappers.add(id(wrapper))
        wrapper = getattr(wrapper, "env", None)
    if (
        fire_wrapper.max_fire_attempts != expected_kwargs["fire_reset_max_attempts"]
        or fire_wrapper.confirmation_steps != expected_kwargs["fire_confirmation_steps"]
        or fire_wrapper.min_observation_change_fraction
        != expected_kwargs["fire_confirmation_change_fraction"]
        or fire_wrapper.confirmation_operator
        != expected_kwargs["fire_confirmation_operator"]
        or fire_wrapper.confirmation_signals
        != tuple(expected_kwargs["fire_confirmation_signals"])
    ):
        return False, f"The runtime FIRE-reset settings differ from {contract_label}."

    base = getattr(env, "unwrapped", env)
    spec = getattr(env, "spec", None)
    spec_kwargs = getattr(spec, "kwargs", {})
    if not isinstance(spec_kwargs, Mapping):
        spec_kwargs = {}
    if (
        getattr(spec, "id", None) != contract.environment_id
        or getattr(base, "_game", None) != "breakout"
        or getattr(base, "_game_mode", None) != 0
        or getattr(base, "_game_difficulty", None) != 0
        or getattr(base, "_frameskip", None) != 1
        or spec_kwargs.get("mode") != 0
        or spec_kwargs.get("difficulty") != 0
        or spec_kwargs.get("max_num_frames_per_episode")
        != contract.time_limit_semantics["max_num_frames_per_episode"]
    ):
        return False, f"The live ALE game identity or frame limit differs from {contract_label}."
    ale = getattr(base, "ale", None)
    get_float = getattr(ale, "getFloat", None)
    if not callable(get_float):
        return False, "The live ALE runtime does not expose sticky-action probability."
    try:
        sticky_action_probability = float(
            get_float("repeat_action_probability")
        )
    except (RuntimeError, TypeError, ValueError):
        return False, "The live ALE sticky-action probability is unavailable."
    if not math.isclose(
        sticky_action_probability,
        contract.sticky_action_probability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False, f"The live ALE sticky-action probability differs from {contract_label}."
    return True, None


def _declares_canonical_breakout_contract(
    metadata: Mapping[str, Any] | None,
) -> bool:
    values = metadata or {}
    contract_value = values.get("evaluation_contract")
    if isinstance(contract_value, Mapping) and contract_value.get("contract_id") in {
        BREAKOUT_CONTRACT_V2_ID,
        BREAKOUT_CONTRACT_V3_ID,
    }:
        return True
    path_value = values.get("evaluation_contract_path")
    if isinstance(path_value, (str, Path)) and str(path_value).strip():
        return Path(path_value).name.casefold() in {
            "breakout_contract_v2.json",
            "breakout_contract_v3.json",
        }
    return False


def _integer(value: Any, *, name: str, minimum: int) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < minimum:
        message = "must not be negative" if minimum == 0 else f"must be at least {minimum}"
        raise ValueError(f"{name} {message}")
    return int(parsed)


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number between 0 and 1")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return parsed


def _seed_values(values: Any) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("seeds must be a non-empty sequence of integers")
    if not values:
        raise ValueError("seeds must contain at least one value")
    parsed = tuple(_integer(value, name="seed", minimum=0) for value in values)
    if len(set(parsed)) != len(parsed):
        raise ValueError("seeds must be unique so each evaluation group is traceable")
    return parsed


@dataclass(frozen=True)
class EvaluationConfig:
    """The fixed protocol shared by every Day 15 policy run."""

    seeds: tuple[int, ...]
    episodes_per_seed: int = 5
    epsilon: float = 0.0
    environment_id: str = ENVIRONMENT_ID
    source_day14_manifest: str | None = None
    source_day14_profiling_report: str | None = None
    checkpoint_rule: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", _seed_values(self.seeds))
        object.__setattr__(
            self,
            "episodes_per_seed",
            _integer(self.episodes_per_seed, name="episodes_per_seed", minimum=1),
        )
        object.__setattr__(self, "epsilon", _probability(self.epsilon, name="epsilon"))
        if not isinstance(self.environment_id, str) or not self.environment_id.strip():
            raise ValueError("environment_id must be a non-empty string")
        for name, value in (
            ("source_day14_manifest", self.source_day14_manifest),
            ("source_day14_profiling_report", self.source_day14_profiling_report),
            ("checkpoint_rule", self.checkpoint_rule),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")

    @property
    def total_episodes(self) -> int:
        return len(self.seeds) * self.episodes_per_seed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_CONFIG_SCHEMA_VERSION,
            "seeds": list(self.seeds),
            "episodes_per_seed": self.episodes_per_seed,
            "epsilon": self.epsilon,
            "environment_id": self.environment_id,
            "source_day14_manifest": self.source_day14_manifest,
            "source_day14_profiling_report": self.source_day14_profiling_report,
            "checkpoint_rule": self.checkpoint_rule,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EvaluationConfig":
        if not isinstance(values, Mapping):
            raise TypeError("evaluation config must be a JSON object")
        environment_id = values.get("environment_id", values.get("environment", ENVIRONMENT_ID))
        manifest = values.get("source_day14_manifest", values.get("day14_manifest"))
        profiling_report = values.get("source_day14_profiling_report")
        return cls(
            seeds=tuple(values.get("seeds", ())),
            episodes_per_seed=values.get("episodes_per_seed", 5),
            epsilon=values.get("epsilon", 0.0),
            environment_id=environment_id,
            source_day14_manifest=manifest,
            source_day14_profiling_report=profiling_report,
            checkpoint_rule=values.get("checkpoint_rule"),
        )


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: evaluation config must be a JSON object")
    return EvaluationConfig.from_mapping(payload)


class EvaluationPolicy(Protocol):
    policy_type: str

    def select_action(
        self,
        observation: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> int:
        ...


class RandomPolicy:
    """Uniformly sample a legal action from the episode-local RNG."""

    policy_type = "random"

    def __init__(self, action_count: int) -> None:
        self.action_count = _integer(action_count, name="action_count", minimum=1)

    def select_action(
        self,
        observation: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> int:
        del observation
        return int(rng.integers(0, self.action_count))


class DQNPolicy:
    """Select greedy or fixed-epsilon actions from a frozen network."""

    policy_type = "dqn"

    def __init__(
        self,
        model: nn.Module,
        *,
        action_count: int,
        device: torch.device,
        epsilon: float,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.action_count = _integer(action_count, name="action_count", minimum=1)
        self.device = device
        self.epsilon = _probability(epsilon, name="epsilon")
        self.model.eval()
        parameter_devices = {str(parameter.device) for parameter in self.model.parameters()}
        if parameter_devices and parameter_devices != {str(device)}:
            raise ValueError(
                "model parameters must be on the resolved evaluation device; "
                f"found {sorted(parameter_devices)}, expected {device}"
            )

    def select_action(
        self,
        observation: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> int:
        state = observation_to_tensor(observation, device=self.device)
        with torch.no_grad():
            q_values = self.model(state)
        if not isinstance(q_values, torch.Tensor):
            raise ValueError("DQN model must return a torch.Tensor")
        if tuple(q_values.shape) != (1, self.action_count):
            raise ValueError(
                f"DQN model must return shape (1, {self.action_count}); "
                f"received {tuple(q_values.shape)}"
            )
        if not torch.isfinite(q_values).all().item():
            raise ValueError("DQN model returned non-finite Q-values")
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.action_count))
        return int(torch.argmax(q_values[0]).item())


def _requested_device_name(device: torch.device | str) -> str:
    if isinstance(device, torch.device):
        return str(device)
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string or torch.device")
    return device.strip().lower()


def _action_count(env: Any) -> int:
    raw_count = getattr(getattr(env, "action_space", None), "n", None)
    try:
        return _integer(raw_count, name="env.action_space.n", minimum=1)
    except (TypeError, ValueError) as error:
        raise ValueError("env.action_space.n must be a positive integer") from error


def _observation_shape(env: Any) -> tuple[int, ...]:
    raw_shape = getattr(getattr(env, "observation_space", None), "shape", None)
    if raw_shape is None:
        raise ValueError("env.observation_space.shape is required")
    try:
        shape = tuple(_integer(value, name="observation dimension", minimum=1) for value in raw_shape)
    except TypeError as error:
        raise ValueError("env.observation_space.shape must be a sequence") from error
    if not shape:
        raise ValueError("env.observation_space.shape must not be empty")
    return shape


def _environment_id(env: Any) -> str:
    spec = getattr(env, "spec", None)
    return str(getattr(spec, "id", None) or ENVIRONMENT_ID)


def _action_names(env: Any, action_count: int) -> tuple[str, ...]:
    unwrapped = getattr(env, "unwrapped", env)
    get_meanings = getattr(unwrapped, "get_action_meanings", None)
    if callable(get_meanings):
        meanings = tuple(str(value) for value in get_meanings())
        if len(meanings) == action_count and all(meanings):
            return meanings
    return tuple(
        str(ATARI_ACTION_NAMES.get(index, f"ACTION_{index}"))
        for index in range(action_count)
    )


def _seed_action_space(env: Any, seed: int) -> None:
    seed_method = getattr(getattr(env, "action_space", None), "seed", None)
    if callable(seed_method):
        seed_method(seed)


def _resolved_action_from_info(
    info: Mapping[str, Any] | Any,
    *,
    requested_action: int,
    action_count: int,
) -> tuple[int, bool, str | None]:
    """Read wrapper provenance without pretending to see hidden ALE randomness."""

    if not isinstance(info, Mapping):
        return requested_action, False, None
    raw_requested = info.get("fire_reset_requested_action", requested_action)
    try:
        resolved_requested = operator.index(raw_requested)
    except TypeError as error:
        raise ValueError(
            "fire_reset_requested_action must be an integer"
        ) from error
    if int(resolved_requested) != requested_action:
        raise ValueError(
            "environment fire_reset_requested_action does not match policy action"
        )
    auto_fire = bool(info.get("fire_reset_auto", False))
    if auto_fire and "fire_reset_executed_action" not in info:
        raise ValueError(
            "environment-side FIRE must report fire_reset_executed_action"
        )
    raw_executed = info.get("fire_reset_executed_action", requested_action)
    try:
        executed_action = operator.index(raw_executed)
    except TypeError as error:
        raise ValueError("fire_reset_executed_action must be an integer") from error
    if not 0 <= int(executed_action) < action_count:
        raise ValueError(
            "environment executed an illegal action "
            f"{executed_action}; expected 0 <= action < {action_count}"
        )
    raw_reason = info.get("fire_reset_reason")
    if raw_reason is not None and not isinstance(raw_reason, str):
        raise ValueError("fire_reset_reason must be a string or None")
    return int(executed_action), auto_fire, raw_reason


def _time_limit_signal(
    env: Any,
    truncated: bool,
    info: Mapping[str, Any] | None,
) -> tuple[bool, str | None]:
    """Identify ALE/TimeLimit truncation without inferring it from score."""

    if not truncated:
        return False, None
    ale = getattr(getattr(env, "unwrapped", env), "ale", None)
    game_truncated = getattr(ale, "game_truncated", None)
    if callable(game_truncated):
        try:
            if bool(game_truncated()):
                return True, "ale.game_truncated"
        except RuntimeError:
            pass
    if isinstance(info, Mapping):
        if bool(info.get("TimeLimit.truncated", False)):
            return True, "info.TimeLimit.truncated"
        if bool(info.get("time_limit", False)):
            return True, "info.time_limit"
    return False, None


def _runtime_metadata(
    *,
    requested_device: str,
    resolved_device: torch.device,
    evaluation_steps: int,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "evaluation_steps": evaluation_steps,
        "wall_clock_seconds": float(wall_clock_seconds),
        "steps_per_second": float(evaluation_steps / max(wall_clock_seconds, 1e-9)),
    }
    if resolved_device.type == "cuda":
        index = 0 if resolved_device.index is None else int(resolved_device.index)
        name = torch.cuda.get_device_name(index)
        metadata.update(
            {
                "cuda_device_index": index,
                "gpu_name": name,
                "cuda_device_name": name,
                "gpu_model": name,
            }
        )
    else:
        metadata.update(
            {
                "cuda_device_index": None,
                "gpu_name": None,
                "cuda_device_name": None,
                "gpu_model": None,
            }
        )
    return metadata


def evaluate_policy(
    model: nn.Module | None,
    *,
    episodes: int,
    seeds: Sequence[int],
    device: torch.device | str,
    epsilon: float = 0.0,
    env_factory: EnvironmentFactory = make_breakout_env,
    max_steps_per_episode: int | None = None,
    model_id: str | None = None,
    training_metadata: Mapping[str, Any] | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    evaluation_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    """Evaluate Random or DQN using the same environment and episode loop.

    ``episodes`` is the number of episodes per seed group. If a caller passes
    ``max_steps_per_episode`` it is a safety guard only: reaching it without
    an environment ``terminated`` or ``truncated`` signal raises an error and
    never emits a partial episode as a formal result.
    """

    episodes_per_seed = _integer(episodes, name="episodes", minimum=1)
    evaluation_seeds = _seed_values(seeds)
    epsilon = _probability(epsilon, name="epsilon")
    if max_steps_per_episode is not None:
        max_steps_per_episode = _integer(
            max_steps_per_episode,
            name="max_steps_per_episode",
            minimum=1,
        )
    requested_device = _requested_device_name(device)
    resolved_device = resolve_device(requested_device)
    if not callable(env_factory):
        raise TypeError("env_factory must be callable")
    if model is not None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module or None")
        model.to(resolved_device)
        model.eval()

    env = env_factory()
    repository_root = Path(__file__).resolve().parents[1]
    completion_support = inspect_breakout_completion_support(env)
    completion_detector = BreakoutCompletionDetector(completion_support)
    source_provenance = _capture_source_provenance(repository_root)
    contract_provenance = _contract_provenance(
        metadata,
        repository_root=repository_root,
    )
    runtime_contract_validated, runtime_contract_reason = (
        _contract_runtime_binding(
            env,
            metadata,
            contract_provenance,
            evaluation_seeds=evaluation_seeds,
            episodes_per_seed=episodes_per_seed,
            epsilon=epsilon,
        )
    )
    contract_provenance.update(
        {
            "runtime_binding_validated": runtime_contract_validated,
            "validation_status": (
                contract_provenance.get("validation_status")
                if runtime_contract_validated
                else "unverified"
            ),
            "validation_reason": runtime_contract_reason,
        }
    )
    result_metadata = dict(metadata or {})
    result_metadata["completion_detector"] = completion_support.to_dict()
    result_metadata["source_provenance"] = source_provenance
    result_metadata["contract_provenance"] = contract_provenance
    canonical_contract_declared = _declares_canonical_breakout_contract(metadata)
    if (
        (
            contract_provenance.get("definition_status") == "validated"
            or canonical_contract_declared
        )
        and not runtime_contract_validated
    ):
        env.close()
        raise RuntimeError(
            "Breakout contract runtime validation failed; refusing to evaluate: "
            f"{runtime_contract_reason or 'runtime semantics are unverified'}"
        )
    started_at = time.perf_counter()
    try:
        action_count = _action_count(env)
        observation_shape = _observation_shape(env)
        action_names = _action_names(env, action_count)
        environment_id = _environment_id(env)
        if model is None:
            policy: EvaluationPolicy = RandomPolicy(action_count)
            resolved_model_id = model_id or "random-policy"
        else:
            policy = DQNPolicy(
                model,
                action_count=action_count,
                device=resolved_device,
                epsilon=epsilon,
            )
            resolved_model_id = model_id or "dqn-policy"

        episode_results: list[EpisodeResult] = []
        inference_context = torch.no_grad() if model is not None else nullcontext()
        with inference_context:
            for seed_index, evaluation_seed in enumerate(evaluation_seeds):
                for episode_index in range(1, episodes_per_seed + 1):
                    episode_seed = evaluation_seed + episode_index - 1
                    observation, _ = env.reset(seed=episode_seed)
                    completion_detector.reset()
                    emulator_frame_origin = read_ale_episode_frame(env)
                    _seed_action_space(env, episode_seed)
                    rng = np.random.default_rng(episode_seed)
                    episode_return = 0.0
                    requested_action_values: list[int] = []
                    executed_action_values: list[int] = []
                    auto_fire_count = 0
                    auto_fire_reason_counts: Counter[str] = Counter()
                    life_loss_count = 0
                    life_loss_steps: list[int] = []
                    terminated = False
                    truncated = False
                    emulator_frame_now: int | None = None
                    while True:
                        if (
                            max_steps_per_episode is not None
                            and len(requested_action_values) >= max_steps_per_episode
                        ):
                            raise RuntimeError(
                                "evaluation episode did not finish within "
                                f"{max_steps_per_episode} steps; refusing to emit a partial result"
                            )
                        action = int(policy.select_action(observation, rng=rng))
                        if not 0 <= action < action_count:
                            raise ValueError(
                                f"policy returned illegal action {action}; "
                                f"expected 0 <= action < {action_count}"
                            )
                        requested_action_values.append(action)
                        (
                            observation,
                            reward,
                            terminated_raw,
                            truncated_raw,
                            info,
                        ) = env.step(action)
                        emulator_frame_now = read_ale_episode_frame(env)
                        executed_action, auto_fire, fire_reason = _resolved_action_from_info(
                            info,
                            requested_action=action,
                            action_count=action_count,
                        )
                        executed_action_values.append(executed_action)
                        if auto_fire:
                            auto_fire_count += 1
                            if fire_reason is not None:
                                auto_fire_reason_counts[fire_reason] += 1
                        if isinstance(info, Mapping) and bool(
                            info.get("fire_reset_life_loss", False)
                        ):
                            life_loss_count += 1
                            life_loss_steps.append(len(executed_action_values))
                        reward_value = float(reward)
                        if not math.isfinite(reward_value):
                            raise ValueError("environment reward must be finite")
                        episode_return += reward_value
                        terminated = bool(terminated_raw)
                        truncated = bool(truncated_raw)
                        time_limit, time_limit_source = _time_limit_signal(
                            env,
                            truncated,
                            info if isinstance(info, Mapping) else None,
                        )
                        if completion_support.supported:
                            elapsed_emulator_frame = (
                                emulator_frame_now - emulator_frame_origin
                                if emulator_frame_now is not None
                                and emulator_frame_origin is not None
                                and emulator_frame_now >= emulator_frame_origin
                                else None
                            )
                            completion_detector.observe(
                                cumulative_score=episode_return,
                                ram_score=read_breakout_score(env),
                                agent_step=len(executed_action_values),
                                emulator_frame=elapsed_emulator_frame,
                                lives_remaining=read_ale_lives(env),
                            )
                        if terminated or truncated:
                            break

                    requested_counts = {name: 0 for name in action_names}
                    executed_counts = {name: 0 for name in action_names}
                    for action in requested_action_values:
                        requested_counts[action_names[action]] += 1
                    for action in executed_action_values:
                        executed_counts[action_names[action]] += 1
                    survival_metrics = compute_episode_survival_metrics(
                        raw_score=episode_return,
                        episode_length=len(executed_action_values),
                        life_loss_steps=life_loss_steps,
                    )
                    episode_results.append(
                        EpisodeResult(
                            evaluation_seed=evaluation_seed,
                            episode_seed=episode_seed,
                            seed_index=seed_index,
                            episode_index=episode_index,
                            episode_return=float(episode_return),
                            episode_length=len(executed_action_values),
                            terminated=terminated,
                            truncated=truncated,
                            action_distribution=executed_counts,
                            time_limit=time_limit,
                            time_limit_source=time_limit_source,
                            requested_action_distribution=requested_counts,
                            executed_action_distribution=executed_counts,
                            auto_fire_count=auto_fire_count,
                            auto_fire_reason_counts=dict(
                                sorted(auto_fire_reason_counts.items())
                            ),
                            life_loss_count=life_loss_count,
                            score_per_life=float(survival_metrics["score_per_life"]),
                            frames_between_life_losses=(
                                None
                                if survival_metrics["frames_between_life_losses"] is None
                                else float(survival_metrics["frames_between_life_losses"])
                            ),
                            time_to_first_life_loss=(
                                None
                                if survival_metrics["time_to_first_life_loss"] is None
                                else int(survival_metrics["time_to_first_life_loss"])
                            ),
                            life_losses_per_1000_steps=float(
                                survival_metrics["life_losses_per_1000_steps"]
                            ),
                            total_emulator_frames=(
                                emulator_frame_now - emulator_frame_origin
                                if emulator_frame_now is not None
                                and emulator_frame_origin is not None
                                and emulator_frame_now >= emulator_frame_origin
                                else None
                            ),
                            completion_state=completion_detector.state,
                        )
                    )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    wall_clock_seconds = max(time.perf_counter() - started_at, 1e-9)
    runtime = _runtime_metadata(
        requested_device=requested_device,
        resolved_device=resolved_device,
        evaluation_steps=sum(episode.episode_length for episode in episode_results),
        wall_clock_seconds=wall_clock_seconds,
    )
    return EvaluationResult(
        policy_type=policy.policy_type,
        model_id=resolved_model_id,
        environment_id=environment_id,
        observation_shape=observation_shape,
        action_count=action_count,
        action_names=action_names,
        evaluation_seeds=evaluation_seeds,
        episodes_per_seed=episodes_per_seed,
        evaluation_epsilon=epsilon,
        requested_device=requested_device,
        resolved_device=str(resolved_device),
        runtime=runtime,
        episodes=tuple(episode_results),
        training=training_metadata,
        checkpoint=checkpoint_metadata,
        evaluation_id=evaluation_id,
        metadata=result_metadata,
        completion_detector=completion_support.to_dict(),
        source_provenance=source_provenance,
        contract_provenance=contract_provenance,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_step(payload: Mapping[str, Any], path: Path) -> int:
    raw_step = payload.get("global_step")
    if isinstance(raw_step, Integral) and not isinstance(raw_step, bool) and raw_step >= 0:
        return int(raw_step)
    match = re.search(r"step-(\d+)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError("checkpoint does not contain a recoverable global_step")


def _load_torch_payload(path: Path) -> dict[str, Any]:
    return read_checkpoint(path)


@dataclass(frozen=True)
class LoadedDQNCheckpoint:
    model: nn.Module
    model_id: str
    training_metadata: Mapping[str, Any]
    checkpoint_metadata: Mapping[str, Any]


def load_dqn_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
    env_factory: EnvironmentFactory = make_breakout_env,
    source_day14_manifest: str | Path | None = None,
) -> LoadedDQNCheckpoint:
    """Load the online network and retain checkpoint/training provenance."""

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    requested_device = _requested_device_name(device)
    resolved_device = resolve_device(requested_device)
    payload = _load_torch_payload(checkpoint_path)
    state_dict = payload.get("online_network")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint does not contain an online_network state_dict")
    saved_config = payload.get("config", {})
    if not isinstance(saved_config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    saved_config = dict(saved_config)

    env = env_factory()
    try:
        action_count = _action_count(env)
        observation_shape = _observation_shape(env)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    model_config = payload.get("model_config", {})
    if not isinstance(model_config, Mapping):
        model_config = {}

    saved_action_count = model_config.get("num_actions")
    if saved_action_count is not None:
        saved_action_count = _integer(
            saved_action_count,
            name="model num_actions",
            minimum=1,
        )
        if saved_action_count != action_count:
            raise ValueError(
                "checkpoint model action count does not match the evaluation environment"
            )

    raw_input_shape = model_config.get("input_shape", observation_shape)
    if isinstance(raw_input_shape, (str, bytes)) or not isinstance(raw_input_shape, Sequence):
        raise ValueError("checkpoint model input_shape must be a sequence")
    try:
        saved_input_shape = tuple(
            _integer(value, name="model input dimension", minimum=1)
            for value in raw_input_shape
        )
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint model input_shape is invalid") from error
    if saved_input_shape != observation_shape:
        raise ValueError(
            "checkpoint model input shape does not match the evaluation environment"
        )

    hidden_dim = _integer(
        512 if model_config.get("hidden_dim") is None else model_config.get("hidden_dim"),
        name="model hidden_dim",
        minimum=1,
    )
    architecture = checkpoint_architecture(payload)
    try:
        model = restore_q_network(
            state_dict,
            architecture=architecture,
            num_actions=action_count,
            input_shape=saved_input_shape,
            hidden_dim=hidden_dim,
            device=resolved_device,
        )
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValueError(
            "checkpoint architecture/action count does not match the evaluation environment"
        ) from error

    step = _checkpoint_step(payload, checkpoint_path)
    source_run_id = str(payload.get("run_id") or checkpoint_path.parent.parent.name)
    manifest_value = (
        portable_artifact_path(Path(source_day14_manifest))
        if source_day14_manifest is not None
        else None
    )
    runtime_payload = payload.get("runtime", {})
    if not isinstance(runtime_payload, Mapping):
        runtime_payload = {}
    training_metadata: dict[str, Any] = {
        "source_day14_run_id": source_run_id,
        "algorithm": payload.get("algorithm", saved_config.get("algorithm", "dqn")),
        "architecture": architecture,
        "num_envs": payload.get("num_envs", saved_config.get("num_envs", 1)),
        "training_seed": saved_config.get("seed"),
        "training_budget": saved_config.get("total_steps"),
        "total_agent_steps": payload.get(
            "total_agent_steps",
            payload.get("training_steps", payload.get("global_step")),
        ),
        "total_emulator_frames": payload.get("total_emulator_frames"),
        "learning_rate": saved_config.get("learning_rate"),
        "batch_size": saved_config.get("batch_size"),
        "train_frequency": saved_config.get("train_frequency"),
        "replay_backend": payload.get(
            "replay_backend",
            saved_config.get("replay_backend", "cpu"),
        ),
        "training_device": saved_config.get("device"),
        "device": payload.get("device", runtime_payload.get("device")),
        "training_precision": saved_config.get("precision"),
        "training_config": dict(saved_config),
        "config_reference": payload.get("contract_path", saved_config.get("contract_path")),
        "contract_id": payload.get("contract_id", saved_config.get("contract_id")),
        "contract_path": payload.get("contract_path", saved_config.get("contract_path")),
        "source_day14_manifest": manifest_value,
        "trainer_runtime": dict(runtime_payload),
    }
    checkpoint_metadata: dict[str, Any] = {
        "path": portable_artifact_path(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "step": step,
        "source_day14_run_id": source_run_id,
        "source_day14_manifest": manifest_value,
        "format_version": payload.get("format_version"),
        "algorithm": payload.get("algorithm", saved_config.get("algorithm", "dqn")),
        "architecture": architecture,
        "contract_id": payload.get("contract_id", saved_config.get("contract_id")),
        "contract_path": payload.get("contract_path", saved_config.get("contract_path")),
        "num_envs": payload.get("num_envs", saved_config.get("num_envs", 1)),
        "replay_backend": payload.get(
            "replay_backend",
            saved_config.get("replay_backend", "cpu"),
        ),
        "training_steps": payload.get("training_steps", payload.get("global_step")),
        "total_agent_steps": payload.get(
            "total_agent_steps",
            payload.get("training_steps", payload.get("global_step")),
        ),
        "total_emulator_frames": payload.get("total_emulator_frames"),
        "model_config": {
            "num_actions": action_count,
            "input_shape": list(saved_input_shape),
            "hidden_dim": hidden_dim,
            "architecture": architecture,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "requested_device": requested_device,
        "device": payload.get("device", str(resolved_device)),
        "resolved_device": str(resolved_device),
        "trainer_runtime": dict(runtime_payload),
    }
    return LoadedDQNCheckpoint(
        model=model,
        model_id=f"{source_run_id}@step-{step:08d}",
        training_metadata=training_metadata,
        checkpoint_metadata=checkpoint_metadata,
    )


__all__ = [
    "DQNPolicy",
    "EnvironmentFactory",
    "EpisodeResult",
    "EvaluationConfig",
    "EvaluationPolicy",
    "EvaluationResult",
    "LoadedDQNCheckpoint",
    "RandomPolicy",
    "evaluate_policy",
    "load_day14_provenance",
    "load_dqn_checkpoint",
    "load_evaluation_config",
    "read_evaluation_results",
    "summarize_returns",
    "validate_checkpoint_provenance",
    "write_evaluation_artifacts",
]
