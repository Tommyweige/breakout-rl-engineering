"""Reusable frozen-policy evaluation for the Breakout DQN."""

from __future__ import annotations

import csv
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
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TimeLimit
from torch import nn

from breakout_env import BreakoutFireResetWrapper, ENVIRONMENT_ID, make_breakout_env
from breakout_rl.completion import (
    BREAKOUT_AUDIT_CONFIG_PATH,
    BreakoutCompletionDetector,
    CompletionState,
    inspect_breakout_completion_support,
    read_ale_episode_frame,
    read_ale_lives,
    read_breakout_score,
)
from breakout_rl.models.factory import build_q_network, checkpoint_architecture
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.experiments import load_experiment_config
from breakout_rl.evaluation_artifacts import (
    ACTION_DISTRIBUTION_SEMANTICS,
    EVALUATION_ARTIFACT_SCHEMA_VERSION,
    VERIFIED_CLEAR_PROVENANCE_FIELDS,
    read_evaluation_results,
    summarize_returns,
    summary_from_episode_rows,
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


def _first_present(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


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


@dataclass(frozen=True)
class EpisodeResult:
    """Raw result for one environment episode."""

    evaluation_seed: int
    episode_seed: int
    seed_index: int
    episode_index: int
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    action_distribution: Mapping[str, int]
    time_limit: bool = False
    time_limit_source: str | None = None
    requested_action_distribution: Mapping[str, int] | None = None
    executed_action_distribution: Mapping[str, int] | None = None
    auto_fire_count: int = 0
    auto_fire_reason_counts: Mapping[str, int] | None = None
    life_loss_count: int = 0
    score_per_life: float = 0.0
    frames_between_life_losses: float | None = None
    time_to_first_life_loss: int | None = None
    life_losses_per_1000_steps: float = 0.0
    total_emulator_frames: int | None = None
    completion_state: CompletionState = CompletionState(cleared=None)

    @property
    def complete(self) -> bool:
        return self.terminated or self.truncated

    @property
    def stop_reason(self) -> str:
        if self.time_limit:
            return "time_limit"
        if self.terminated:
            return "terminated"
        if self.truncated:
            return "truncated"
        return "incomplete"

    @property
    def cleared(self) -> bool | None:
        return self.completion_state.cleared

    @property
    def clear_agent_step(self) -> int | None:
        return self.completion_state.clear_agent_step

    @property
    def clear_emulator_frame(self) -> int | None:
        return self.completion_state.clear_emulator_frame

    @property
    def clear_score(self) -> float | None:
        return self.completion_state.clear_score

    @property
    def lives_remaining_at_clear(self) -> int | None:
        return self.completion_state.lives_remaining_at_clear

    @property
    def completion_detection_source(self) -> str | None:
        return self.completion_state.completion_detection_source

    @property
    def completion_detection_reason(self) -> str | None:
        return self.completion_state.unavailable_reason

    @property
    def completion_outcome(self) -> str:
        if self.cleared is True:
            return "cleared"
        if self.cleared is None:
            return "clear_status_unavailable"
        if self.time_limit:
            return "time_limit"
        if self.truncated:
            return "truncated"
        if self.terminated:
            return (
                "game_over"
                if self.completion_detection_source is not None
                else "terminated"
            )
        return "incomplete"

    def to_dict(self) -> dict[str, Any]:
        executed_distribution = dict(
            self.executed_action_distribution or self.action_distribution
        )
        requested_distribution = dict(
            self.requested_action_distribution or executed_distribution
        )
        return {
            "evaluation_seed": self.evaluation_seed,
            "seed": self.episode_seed,
            "episode_seed": self.episode_seed,
            "seed_index": self.seed_index,
            "episode_index": self.episode_index,
            "episode_return": self.episode_return,
            "return": self.episode_return,
            "episode_length": self.episode_length,
            "total_agent_steps": self.episode_length,
            "total_emulator_frames": self.total_emulator_frames,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "time_limit": self.time_limit,
            "time_limit_source": self.time_limit_source,
            "complete": self.complete,
            "stop_reason": self.stop_reason,
            "completion_outcome": self.completion_outcome,
            "cleared": self.completion_state.cleared,
            "clear_agent_step": self.completion_state.clear_agent_step,
            "clear_emulator_frame": self.completion_state.clear_emulator_frame,
            "clear_score": self.completion_state.clear_score,
            "lives_remaining_at_clear": (
                self.completion_state.lives_remaining_at_clear
            ),
            "completion_detection_source": (
                self.completion_state.completion_detection_source
            ),
            "completion_detection_reason": (
                self.completion_state.unavailable_reason
            ),
            # Keep the historical field, but define it explicitly as the
            # action sent to the wrapped environment.
            "action_distribution": executed_distribution,
            "action_distribution_semantics": ACTION_DISTRIBUTION_SEMANTICS,
            "requested_action_distribution": requested_distribution,
            "executed_action_distribution": executed_distribution,
            "auto_fire_count": int(self.auto_fire_count),
            "auto_fire_reason_counts": dict(self.auto_fire_reason_counts or {}),
            "life_loss_count": int(self.life_loss_count),
            "score_per_life": float(self.score_per_life),
            "frames_between_life_losses": self.frames_between_life_losses,
            "time_to_first_life_loss": self.time_to_first_life_loss,
            "life_losses_per_1000_steps": float(self.life_losses_per_1000_steps),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """One policy's machine-readable result under a fixed protocol."""

    policy_type: str
    model_id: str | None
    environment_id: str
    observation_shape: tuple[int, ...]
    action_count: int
    action_names: tuple[str, ...]
    evaluation_seeds: tuple[int, ...]
    episodes_per_seed: int
    evaluation_epsilon: float
    requested_device: str
    resolved_device: str
    runtime: Mapping[str, Any]
    episodes: tuple[EpisodeResult, ...]
    training: Mapping[str, Any] | None = None
    checkpoint: Mapping[str, Any] | None = None
    evaluation_id: str | None = None
    metadata: Mapping[str, Any] | None = None
    completion_detector: Mapping[str, Any] | None = None
    source_provenance: Mapping[str, Any] | None = None
    contract_provenance: Mapping[str, Any] | None = None

    @property
    def total_steps(self) -> int:
        return sum(episode.episode_length for episode in self.episodes)

    @property
    def total_emulator_frames(self) -> int | None:
        values = [episode.total_emulator_frames for episode in self.episodes]
        if not values or any(value is None for value in values):
            return None
        return sum(int(value) for value in values)

    @property
    def action_distribution(self) -> dict[str, int]:
        counts = {name: 0 for name in self.action_names}
        for episode in self.episodes:
            for name, count in episode.action_distribution.items():
                counts[name] = counts.get(name, 0) + int(count)
        return counts

    @property
    def requested_action_distribution(self) -> dict[str, int]:
        counts = {name: 0 for name in self.action_names}
        for episode in self.episodes:
            values = episode.requested_action_distribution or episode.action_distribution
            for name, count in values.items():
                counts[name] = counts.get(name, 0) + int(count)
        return counts

    @property
    def executed_action_distribution(self) -> dict[str, int]:
        return self.action_distribution

    @property
    def auto_fire_count(self) -> int:
        return sum(int(episode.auto_fire_count) for episode in self.episodes)

    @property
    def auto_fire_reason_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for episode in self.episodes:
            counts.update(
                {
                    str(name): int(count)
                    for name, count in (episode.auto_fire_reason_counts or {}).items()
                }
            )
        return dict(sorted(counts.items()))

    @property
    def life_loss_count(self) -> int:
        """Return the number of wrapper-reported life losses."""

        return sum(int(episode.life_loss_count) for episode in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        returns = [episode.episode_return for episode in self.episodes]
        lengths = [episode.episode_length for episode in self.episodes]
        per_episode: list[dict[str, Any]] = []
        verified_clears: list[dict[str, Any]] = []
        for episode in self.episodes:
            row = episode.to_dict()
            if episode.cleared is True:
                training = dict(self.training or {})
                checkpoint = dict(self.checkpoint or {})
                contract = dict(self.contract_provenance or {})
                source = dict(self.source_provenance or {})
                detector = dict(self.completion_detector or {})
                training_config = training.get("training_config")
                if not isinstance(training_config, Mapping):
                    training_config = {}
                training_seed = _first_present(training, "training_seed", "seed")
                if training_seed is None:
                    training_seed = training_config.get("seed")
                training_transition_count = _first_present(
                    training,
                    "training_steps",
                    "training_transitions",
                    "global_step",
                    "training_budget",
                )
                if training_transition_count is None:
                    training_transition_count = training_config.get("total_steps")
                clear_provenance = {
                    "model_id": self.model_id,
                    "checkpoint_id": _first_present(
                        checkpoint, "sha256", "path", "step"
                    ),
                    "checkpoint": checkpoint,
                    "training_seed": training_seed,
                    "training_transition_count": training_transition_count,
                    "training": training,
                    "evaluation_seed": episode.evaluation_seed,
                    "episode_seed": episode.episode_seed,
                    "episode_index": episode.episode_index,
                    "contract_id": contract.get("contract_id"),
                    "contract_sha256": contract.get("contract_sha256"),
                    "contract_hash_semantics": contract.get("hash_semantics"),
                    "source_commit": source.get("source_commit"),
                    "source_working_tree_dirty": source.get(
                        "working_tree_dirty"
                    ),
                    "completion_source_sha256": source.get(
                        "completion_source_sha256"
                    ),
                    "raw_score": episode.episode_return,
                    "clear_score": episode.clear_score,
                    "clear_agent_step": episode.clear_agent_step,
                    "clear_emulator_frame": episode.clear_emulator_frame,
                    "lives_remaining_at_clear": episode.lives_remaining_at_clear,
                    "completion_detection_source": (
                        episode.completion_detection_source
                    ),
                    "completion_detector_id": detector.get("detector_id"),
                    "contract_validation_status": contract.get(
                        "validation_status"
                    ),
                }
                missing_provenance_fields = [
                    field
                    for field in VERIFIED_CLEAR_PROVENANCE_FIELDS
                    if clear_provenance.get(field) is None
                ]
                if contract.get("validation_status") not in {
                    "canonical_contract_v2",
                    "canonical_contract_v3",
                }:
                    missing_provenance_fields.append(
                        "validated canonical Breakout contract runtime binding"
                    )
                clear_provenance["provenance_status"] = (
                    "complete" if not missing_provenance_fields else "incomplete"
                )
                clear_provenance["missing_provenance_fields"] = (
                    missing_provenance_fields
                )
                row["completion_provenance"] = clear_provenance
                if clear_provenance["provenance_status"] == "complete":
                    verified_clears.append(clear_provenance)
            per_episode.append(row)
        summary = summary_from_episode_rows(
            per_episode
        )
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "evaluation_id": self.evaluation_id,
            "policy_type": self.policy_type,
            "model_id": self.model_id,
            "environment_id": self.environment_id,
            "environment": {
                "id": self.environment_id,
                "observation_shape": list(self.observation_shape),
                "action_count": self.action_count,
                "action_names": list(self.action_names),
            },
            "observation_shape": list(self.observation_shape),
            "action_count": self.action_count,
            "action_names": list(self.action_names),
            "evaluation_seeds": list(self.evaluation_seeds),
            "episodes_per_seed": self.episodes_per_seed,
            "total_episodes": len(self.episodes),
            "total_agent_steps": self.total_steps,
            "total_emulator_frames": self.total_emulator_frames,
            "emulator_frame_count_source": (
                "ALEInterface.getEpisodeFrameNumber"
                if self.total_emulator_frames is not None
                else None
            ),
            "emulator_frame_count_scope": (
                "native ALE frame delta from post-reset baseline through episode termination"
            ),
            "evaluation_epsilon": self.evaluation_epsilon,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "runtime": dict(self.runtime),
            "training": dict(self.training or {}),
            "checkpoint": dict(self.checkpoint or {}),
            "per_episode": per_episode,
            "per_episode_returns": returns,
            "per_episode_lengths": lengths,
            "action_distribution": self.executed_action_distribution,
            "action_distribution_semantics": ACTION_DISTRIBUTION_SEMANTICS,
            "requested_action_distribution": self.requested_action_distribution,
            "executed_action_distribution": self.executed_action_distribution,
            "auto_fire_count": self.auto_fire_count,
            "auto_fire_reason_counts": self.auto_fire_reason_counts,
            "life_loss_count": self.life_loss_count,
            "mean_score_per_life": float(
                fmean(episode.score_per_life for episode in self.episodes)
            ),
            "mean_frames_between_life_losses": (
                float(
                    fmean(
                        episode.frames_between_life_losses
                        for episode in self.episodes
                        if episode.frames_between_life_losses is not None
                    )
                )
                if any(
                    episode.frames_between_life_losses is not None
                    for episode in self.episodes
                )
                else None
            ),
            "mean_time_to_first_life_loss": (
                float(
                    fmean(
                        episode.time_to_first_life_loss
                        for episode in self.episodes
                        if episode.time_to_first_life_loss is not None
                    )
                )
                if any(
                    episode.time_to_first_life_loss is not None
                    for episode in self.episodes
                )
                else None
            ),
            "life_losses_per_1000_steps": float(
                self.life_loss_count / max(1, self.total_steps) * 1000.0
            ),
            "completion_detector": dict(self.completion_detector or {}),
            "source_provenance": dict(self.source_provenance or {}),
            "contract_provenance": dict(self.contract_provenance or {}),
            "verified_clears": verified_clears,
            "summary": summary,
            "metadata": dict(self.metadata or {}),
        }


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _csv_column_name(action_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", action_name.lower()).strip("_")
    return f"action_{normalized or 'unknown'}"


def write_evaluation_artifacts(
    result: EvaluationResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write result JSON and the raw per-episode CSV."""

    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "results.json"
    episodes_path = destination / "episodes.csv"
    payload = result.to_dict()
    payload["artifacts"] = {
        "results_json": results_path.name,
        "episodes_csv": episodes_path.name,
    }
    results_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    action_columns = [_csv_column_name(name) for name in result.action_names]
    requested_action_columns = [
        f"requested_{column}" for column in action_columns
    ]
    executed_action_columns = [
        f"executed_{column}" for column in action_columns
    ]
    fieldnames = [
        "schema_version",
        "policy_type",
        "action_distribution_semantics",
        "evaluation_seed",
        "seed_index",
        "episode_index",
        "episode_seed",
        "episode_return",
        "episode_length",
        "total_agent_steps",
        "total_emulator_frames",
        "terminated",
        "truncated",
        "time_limit",
        "time_limit_source",
        "complete",
        "stop_reason",
        "completion_outcome",
        "cleared",
        "clear_agent_step",
        "clear_emulator_frame",
        "clear_score",
        "lives_remaining_at_clear",
        "completion_detection_source",
        "completion_detection_reason",
        "completion_provenance_json",
        *action_columns,
        "action_distribution_json",
        *requested_action_columns,
        *executed_action_columns,
        "requested_action_distribution_json",
        "executed_action_distribution_json",
        "auto_fire_count",
        "auto_fire_reason_counts_json",
        "life_loss_count",
        "score_per_life",
        "frames_between_life_losses",
        "time_to_first_life_loss",
        "life_losses_per_1000_steps",
    ]
    with episodes_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for episode, episode_payload in zip(
            result.episodes,
            payload["per_episode"],
            strict=True,
        ):
            executed_distribution = dict(
                episode.executed_action_distribution or episode.action_distribution
            )
            requested_distribution = dict(
                episode.requested_action_distribution or executed_distribution
            )
            row: dict[str, Any] = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "policy_type": result.policy_type,
                "action_distribution_semantics": ACTION_DISTRIBUTION_SEMANTICS,
                "evaluation_seed": episode.evaluation_seed,
                "seed_index": episode.seed_index,
                "episode_index": episode.episode_index,
                "episode_seed": episode.episode_seed,
                "episode_return": episode.episode_return,
                "episode_length": episode.episode_length,
                "total_agent_steps": episode.episode_length,
                "total_emulator_frames": episode.total_emulator_frames,
                "terminated": episode.terminated,
                "truncated": episode.truncated,
                "time_limit": episode.time_limit,
                "time_limit_source": episode.time_limit_source,
                "complete": episode.complete,
                "stop_reason": episode.stop_reason,
                "completion_outcome": episode.completion_outcome,
                "cleared": json.dumps(episode.cleared),
                "clear_agent_step": episode.clear_agent_step,
                "clear_emulator_frame": episode.clear_emulator_frame,
                "clear_score": episode.clear_score,
                "lives_remaining_at_clear": episode.lives_remaining_at_clear,
                "completion_detection_source": episode.completion_detection_source,
                "completion_detection_reason": episode.completion_detection_reason,
                "completion_provenance_json": json.dumps(
                    episode_payload.get("completion_provenance"),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                ),
                "action_distribution_json": json.dumps(
                    executed_distribution,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "requested_action_distribution_json": json.dumps(
                    requested_distribution,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "executed_action_distribution_json": json.dumps(
                    executed_distribution,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "auto_fire_count": int(episode.auto_fire_count),
                "auto_fire_reason_counts_json": json.dumps(
                    dict(episode.auto_fire_reason_counts or {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "life_loss_count": int(episode.life_loss_count),
                "score_per_life": float(episode.score_per_life),
                "frames_between_life_losses": episode.frames_between_life_losses,
                "time_to_first_life_loss": episode.time_to_first_life_loss,
                "life_losses_per_1000_steps": float(
                    episode.life_losses_per_1000_steps
                ),
            }
            for action_name, column, requested_column, executed_column in zip(
                result.action_names,
                action_columns,
                requested_action_columns,
                executed_action_columns,
            ):
                row[column] = int(executed_distribution.get(action_name, 0))
                row[requested_column] = int(
                    requested_distribution.get(action_name, 0)
                )
                row[executed_column] = int(
                    executed_distribution.get(action_name, 0)
                )
            writer.writerow(row)
    return results_path, episodes_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(path: Path) -> str:
    """Keep provenance portable when a checkpoint came from another worktree."""

    parts = path.resolve().parts
    markers = {"assets", "configs", "evaluations", "experiments", "reports"}
    for index, part in enumerate(parts):
        if part.lower() in markers:
            return Path(*parts[index:]).as_posix()
    return path.as_posix()


def _checkpoint_step(payload: Mapping[str, Any], path: Path) -> int:
    raw_step = payload.get("global_step")
    if isinstance(raw_step, Integral) and not isinstance(raw_step, bool) and raw_step >= 0:
        return int(raw_step)
    match = re.search(r"step-(\d+)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError("checkpoint does not contain a recoverable global_step")


def _load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    return payload


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
        model = build_q_network(
            architecture,
            num_actions=action_count,
            input_shape=saved_input_shape,  # type: ignore[arg-type]
            hidden_dim=hidden_dim,
        ).to(resolved_device)
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValueError(
            "checkpoint architecture/action count does not match the evaluation environment"
        ) from error
    model.eval()

    step = _checkpoint_step(payload, checkpoint_path)
    source_run_id = str(payload.get("run_id") or checkpoint_path.parent.parent.name)
    manifest_value = (
        _repository_path(Path(source_day14_manifest))
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
        "path": _repository_path(checkpoint_path),
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


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(payload)


def _resolve_optional_reference(path: str | Path, *, source: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    for option in (candidate, Path.cwd() / candidate, source.parent / candidate):
        if option.is_file():
            return option.resolve()
    return (Path.cwd() / candidate).resolve()


def _day14_gate_evidence(
    *,
    run_dir: Path | None,
    summary: Mapping[str, Any],
    metrics_path: Path | None,
    variant: Mapping[str, Any],
    expected_step: int,
    gpu_profiling_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Turn Day 14's observed long-run artifacts into an explicit Gate A result."""

    reasons: list[str] = []
    summary_status = summary.get("status") == "completed"
    summary_step = summary.get("total_steps") == expected_step
    summary_run = summary.get("run_id") == variant.get("run_id")
    summary_provenance = summary_status and summary_step and summary_run
    if not summary_provenance:
        reasons.append("summary.json is not a completed, matching final run")

    baseline_mean = gpu_profiling_summary.get("baseline_recent_episode_return")
    baseline_count = gpu_profiling_summary.get("baseline_recent_episode_count")
    try:
        baseline_mean = float(baseline_mean)
        baseline_count = int(baseline_count)
    except (TypeError, ValueError):
        baseline_mean = None
        baseline_count = None

    completed_returns: list[tuple[int, float]] = []
    diagnostic_fields = (
        "loss",
        "q_mean",
        "q_max",
        "q_min",
        "target_mean",
        "target_max",
        "td_error_mean_abs",
        "td_error_max_abs",
        "gradient_norm",
    )
    diagnostic_counts = {
        field: {"observed": 0, "finite": 0} for field in diagnostic_fields
    }
    if metrics_path is not None and metrics_path.is_file():
        try:
            with metrics_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    try:
                        step = int(row["global_step"])
                    except (KeyError, TypeError, ValueError):
                        reasons.append("metrics.csv contains an invalid global_step")
                        continue
                    raw_return = row.get("raw_episode_return", "")
                    if raw_return not in (None, ""):
                        try:
                            parsed_return = float(raw_return)
                        except (TypeError, ValueError):
                            parsed_return = float("nan")
                        if math.isfinite(parsed_return):
                            completed_returns.append((step, parsed_return))
                        else:
                            reasons.append("metrics.csv contains a non-finite episode return")
                    for field in diagnostic_fields:
                        raw_value = row.get(field, "")
                        if raw_value in (None, ""):
                            continue
                        diagnostic_counts[field]["observed"] += 1
                        try:
                            parsed_value = float(raw_value)
                        except (TypeError, ValueError):
                            parsed_value = float("nan")
                        if math.isfinite(parsed_value):
                            diagnostic_counts[field]["finite"] += 1
        except (OSError, csv.Error) as error:
            reasons.append(f"unable to read metrics.csv: {error}")
    else:
        reasons.append("Day 14 metrics.csv is unavailable")

    completed_returns.sort(key=lambda item: item[0])
    final_recent_mean: float | None = None
    return_signal = False
    if baseline_mean is not None and baseline_count is not None and baseline_count > 0:
        if len(completed_returns) >= baseline_count:
            final_values = [value for _, value in completed_returns[-baseline_count:]]
            final_recent_mean = float(fmean(final_values))
            return_signal = final_recent_mean > baseline_mean
    if not return_signal:
        reasons.append(
            "100K recent return does not show an interpretable improvement over the 10K reference"
        )

    summary_diagnostic_fields = (
        "last_loss",
        "last_q_mean",
        "last_q_max",
        "last_q_min",
        "last_target_mean",
        "last_target_max",
        "last_td_error_mean_abs",
        "last_td_error_max_abs",
    )
    summary_diagnostics_finite = all(
        isinstance(summary.get(field), Real)
        and not isinstance(summary.get(field), bool)
        and math.isfinite(float(summary[field]))
        for field in summary_diagnostic_fields
    )
    metrics_diagnostics_finite = all(
        counts["observed"] > 0 and counts["observed"] == counts["finite"]
        for counts in diagnostic_counts.values()
    )
    diagnostics_healthy = summary_diagnostics_finite and metrics_diagnostics_finite
    if not diagnostics_healthy:
        reasons.append("Day 14 diagnostics contain missing or non-finite required values")

    selection_rule = gpu_profiling_summary.get("selection_rule")
    guardrails_passed = gpu_profiling_summary.get("regression_guardrails_passed") is True
    multiple_episode_evidence = int(summary.get("episodes", 0) or 0) > 1
    selection_evidence = bool(selection_rule) and guardrails_passed and multiple_episode_evidence
    if not selection_evidence:
        reasons.append(
            "config selection lacks recorded quality guardrails and multi-episode evidence"
        )

    provenance_complete = all(
        variant.get(field)
        for field in ("run_id", "config_path", "run_dir", "status", "step_budget")
    ) and summary_provenance
    if not provenance_complete:
        reasons.append("Day 14 checkpoint provenance is incomplete")

    passed = return_signal and diagnostics_healthy and selection_evidence and provenance_complete

    return {
        "status": "passed" if passed else "not_satisfied",
        "criteria": {
            "return_signal": return_signal,
            "diagnostics_healthy": diagnostics_healthy,
            "selection_not_single_best_episode": selection_evidence,
            "checkpoint_provenance_complete": provenance_complete,
        },
        "return_signal": {
            "reference_10k_recent_mean": baseline_mean,
            "reference_recent_episode_count": baseline_count,
            "final_100k_recent_mean": final_recent_mean,
            "improvement": (
                None
                if baseline_mean is None or final_recent_mean is None
                else float(final_recent_mean - baseline_mean)
            ),
        },
        "diagnostics": {
            "summary_status": summary.get("status"),
            "summary_values_finite": summary_diagnostics_finite,
            "metrics_values_finite": metrics_diagnostics_finite,
            "field_counts": diagnostic_counts,
            "summary_source": (
                _repository_path(run_dir / "summary.json") if run_dir is not None else None
            ),
            "metrics_source": (
                _repository_path(metrics_path) if metrics_path is not None else None
            ),
        },
        "selection": {
            "rationale": selection_rule,
            "selected_profile_run_id": gpu_profiling_summary.get("selected_run_id"),
            "regression_guardrails_passed": guardrails_passed,
            "final_run_episode_count": summary.get("episodes"),
        },
        "provenance": {
            "run_id": variant.get("run_id"),
            "expected_step": expected_step,
            "summary_step": summary.get("total_steps"),
            "config_reference": variant.get("config_path"),
            "run_dir": _repository_path(run_dir) if run_dir is not None else None,
        },
        "reasons": reasons,
    }


def load_day14_provenance(
    path: str | Path,
    *,
    profiling_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the completed single final variant from the latest manifest."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _read_json_mapping(source)
    if payload.get("status") != "completed":
        raise ValueError(f"{source}: Day 14 manifest is not completed")
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != 1:
        raise ValueError(f"{source}: expected one final Day 14 variant")
    variant = variants[0]
    if not isinstance(variant, Mapping):
        raise ValueError(f"{source}: final variant must be an object")
    raw_config = variant.get("config_values", {})
    if not isinstance(raw_config, Mapping):
        raw_config = {}
    config_values = dict(raw_config)
    expected_step = variant.get("step_budget", config_values.get("total_steps"))
    expected_step = _integer(expected_step, name="Day 14 step budget", minimum=1)

    raw_config_reference = variant.get("config_path")
    if raw_config_reference is None:
        base_config = payload.get("base_config", {})
        if isinstance(base_config, Mapping):
            raw_config_reference = base_config.get("config_path")
    config_reference = None
    config_path: Path | None = None
    if isinstance(raw_config_reference, str):
        candidate = Path(raw_config_reference)
        config_path = (
            candidate
            if candidate.is_absolute()
            else source.parent / candidate
        ).resolve()
        config_reference = _repository_path(config_path)
        if config_path.is_file():
            effective_config = load_experiment_config(config_path)
            effective_values = dict(effective_config.values)
            effective_values.update(config_values)
            config_values = effective_values
    config_values.setdefault("replay_backend", "cpu")

    run_dir: Path | None = None
    raw_run_dir = variant.get("run_dir")
    if isinstance(raw_run_dir, str) and raw_run_dir.strip():
        candidate = Path(raw_run_dir)
        run_dir = (candidate if candidate.is_absolute() else source.parent / candidate).resolve()
    runtime: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    metrics_path: Path | None = None
    if run_dir is not None and (run_dir / "config.json").is_file():
        run_config = _read_json_mapping(run_dir / "config.json")
        raw_runtime = run_config.get("runtime")
        if isinstance(raw_runtime, Mapping):
            runtime = dict(raw_runtime)
    if run_dir is not None and (run_dir / "summary.json").is_file():
        summary = _read_json_mapping(run_dir / "summary.json")
    if run_dir is not None and (run_dir / "metrics.csv").is_file():
        metrics_path = run_dir / "metrics.csv"

    gpu_profiling_summary: dict[str, Any] = {}
    profiling_source: str | None = None
    if profiling_report_path is not None:
        profiling_path = _resolve_optional_reference(
            profiling_report_path,
            source=source,
        )
        if not profiling_path.is_file():
            raise FileNotFoundError(profiling_path)
        profiling_report = _read_json_mapping(profiling_path)
        runs = profiling_report.get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"{profiling_path}: profiling runs must be an array")
        matching_runs = [
            candidate
            for candidate in runs
            if isinstance(candidate, Mapping)
            and candidate.get("status") == "completed"
            and candidate.get("batch_size") == config_values.get("batch_size")
        ]
        if len(matching_runs) != 1:
            raise ValueError(
                f"{profiling_path}: expected exactly one completed profiling run for "
                f"batch_size={config_values.get('batch_size')}, found {len(matching_runs)}"
            )
        selected_run = matching_runs[0]
        if selected_run.get("completed_steps") != selected_run.get("expected_steps"):
            raise ValueError(
                f"{profiling_path}: selected profiling run did not complete its step budget"
            )
        profiling = selected_run.get("profiling", {})
        if not isinstance(profiling, Mapping):
            profiling = {}
        profiling_source = _repository_path(profiling_path)
        gpu_profiling_summary = {
            "source": profiling_source,
            "selected_batch_size": config_values.get("batch_size"),
            "selection_rule": profiling_report.get("selection_rule", {}),
            "selected_run_id": selected_run.get("run_id"),
            "selected_run_status": selected_run.get("status"),
            "baseline_recent_episode_return": selected_run.get(
                "mean_recent_episode_return"
            ),
            "baseline_recent_episode_count": (
                selected_run.get("recent_return_trend", {}).get("count")
                if isinstance(selected_run.get("recent_return_trend"), Mapping)
                else None
            ),
            "regression_guardrails_passed": (
                selected_run.get("regression_guardrails", {}).get("guardrails_passed")
                if isinstance(selected_run.get("regression_guardrails"), Mapping)
                else None
            ),
            "end_to_end_sps": selected_run.get("end_to_end_sps"),
            "training_samples_per_second": selected_run.get("training_samples_per_second"),
            "profiling": {
                "sample_csv": (
                    _repository_path((profiling_path.parent / profiling["sample_csv"]).resolve())
                    if isinstance(profiling.get("sample_csv"), str)
                    else None
                ),
                "sampling_method": profiling.get("sampling_method"),
                "gpu_utilization_percent": profiling.get("gpu_utilization_percent"),
                "gpu_power_watts": profiling.get("gpu_power_watts"),
                "gpu_memory_used_bytes": profiling.get("gpu_memory_used_bytes"),
                "gpu_memory_total_bytes": profiling.get("gpu_memory_total_bytes"),
            },
        }

    day14_gate = _day14_gate_evidence(
        run_dir=run_dir,
        summary=summary,
        metrics_path=metrics_path,
        variant=variant,
        expected_step=expected_step,
        gpu_profiling_summary=gpu_profiling_summary,
    )

    return {
        "manifest_path": _repository_path(source),
        "source_of_truth": "latest Day 14 final manifest and referenced run artifacts",
        "experiment_id": payload.get("experiment_id", source.parent.name),
        "status": payload.get("status"),
        "run_id": variant.get("run_id"),
        "label": variant.get("label"),
        "expected_checkpoint_step": expected_step,
        "config_values": config_values,
        "config_reference": config_reference,
        "source_day14_profiling_report": profiling_source,
        "run_dir": _repository_path(run_dir) if run_dir is not None else None,
        "runtime": runtime,
        "requested_device": variant.get("requested_device"),
        "resolved_device": variant.get("resolved_device"),
        "replay_backend": config_values.get("replay_backend"),
        "selection_rule": f"final checkpoint at {expected_step} environment steps",
        "selection_rationale": gpu_profiling_summary.get("selection_rule", {}),
        "gpu_profiling_summary": gpu_profiling_summary,
        "day14_gate": day14_gate,
    }


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    training: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Ensure the loaded checkpoint is the manifest's frozen final variant."""

    expected_run_id = provenance.get("run_id")
    if expected_run_id and training.get("source_day14_run_id") != expected_run_id:
        raise ValueError(
            "checkpoint run id does not match the Day 14 final manifest: "
            f"expected {expected_run_id}, got {training.get('source_day14_run_id')}"
        )
    expected_step = provenance.get("expected_checkpoint_step")
    if expected_step is not None and checkpoint.get("step") != expected_step:
        raise ValueError(
            "checkpoint step does not match the Day 14 final selection rule: "
            f"expected {expected_step}, got {checkpoint.get('step')}"
        )

    expected_config = provenance.get("config_values", {})
    if not isinstance(expected_config, Mapping):
        return
    training_fields = {
        "total_steps": "training_budget",
        "seed": "training_seed",
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
        "train_frequency": "train_frequency",
        "replay_backend": "replay_backend",
    }
    for config_field, training_field in training_fields.items():
        expected = expected_config.get(config_field)
        actual = training.get(training_field)
        if expected is not None and actual != expected:
            raise ValueError(
                "checkpoint training config does not match the Day 14 final "
                f"manifest for {config_field}: expected {expected}, got {actual}"
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
