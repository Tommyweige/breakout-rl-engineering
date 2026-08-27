"""Reusable frozen-policy evaluation for the Breakout DQN."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import operator
import platform
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import nn

from breakout_env import ENVIRONMENT_ID, make_breakout_env
from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.experiments import load_experiment_config
from breakout_rl.evaluation_artifacts import (
    read_evaluation_results,
    summarize_returns,
    summary_from_episode_rows,
)
from breakout_rl.training.diagnostics import ATARI_ACTION_NAMES
from breakout_rl.training.dqn_trainer import resolve_device


EVALUATION_SCHEMA_VERSION = 1
EnvironmentFactory = Callable[[], Any]


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
            "schema_version": EVALUATION_SCHEMA_VERSION,
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

    @property
    def complete(self) -> bool:
        return self.terminated or self.truncated

    @property
    def stop_reason(self) -> str:
        if self.terminated:
            return "terminated"
        if self.truncated:
            return "truncated"
        return "incomplete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_seed": self.evaluation_seed,
            "seed": self.episode_seed,
            "episode_seed": self.episode_seed,
            "seed_index": self.seed_index,
            "episode_index": self.episode_index,
            "episode_return": self.episode_return,
            "return": self.episode_return,
            "episode_length": self.episode_length,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "complete": self.complete,
            "stop_reason": self.stop_reason,
            "action_distribution": dict(self.action_distribution),
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

    @property
    def total_steps(self) -> int:
        return sum(episode.episode_length for episode in self.episodes)

    @property
    def action_distribution(self) -> dict[str, int]:
        counts = {name: 0 for name in self.action_names}
        for episode in self.episodes:
            for name, count in episode.action_distribution.items():
                counts[name] = counts.get(name, 0) + int(count)
        return counts

    def to_dict(self) -> dict[str, Any]:
        returns = [episode.episode_return for episode in self.episodes]
        lengths = [episode.episode_length for episode in self.episodes]
        summary = summary_from_episode_rows(
            [episode.to_dict() for episode in self.episodes]
        )
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
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
            "evaluation_epsilon": self.evaluation_epsilon,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "runtime": dict(self.runtime),
            "training": dict(self.training or {}),
            "checkpoint": dict(self.checkpoint or {}),
            "per_episode": [episode.to_dict() for episode in self.episodes],
            "per_episode_returns": returns,
            "per_episode_lengths": lengths,
            "action_distribution": self.action_distribution,
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
                    _seed_action_space(env, episode_seed)
                    rng = np.random.default_rng(episode_seed)
                    episode_return = 0.0
                    action_values: list[int] = []
                    terminated = False
                    truncated = False
                    while True:
                        if (
                            max_steps_per_episode is not None
                            and len(action_values) >= max_steps_per_episode
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
                        action_values.append(action)
                        observation, reward, terminated_raw, truncated_raw, _ = env.step(action)
                        reward_value = float(reward)
                        if not math.isfinite(reward_value):
                            raise ValueError("environment reward must be finite")
                        episode_return += reward_value
                        terminated = bool(terminated_raw)
                        truncated = bool(truncated_raw)
                        if terminated or truncated:
                            break

                    counts = {name: 0 for name in action_names}
                    for action in action_values:
                        counts[action_names[action]] += 1
                    episode_results.append(
                        EpisodeResult(
                            evaluation_seed=evaluation_seed,
                            episode_seed=episode_seed,
                            seed_index=seed_index,
                            episode_index=episode_index,
                            episode_return=float(episode_return),
                            episode_length=len(action_values),
                            terminated=terminated,
                            truncated=truncated,
                            action_distribution=counts,
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
        metadata=metadata,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
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
    fieldnames = [
        "policy_type",
        "evaluation_seed",
        "seed_index",
        "episode_index",
        "episode_seed",
        "episode_return",
        "episode_length",
        "terminated",
        "truncated",
        "complete",
        "stop_reason",
        *action_columns,
        "action_distribution_json",
    ]
    with episodes_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for episode in result.episodes:
            row: dict[str, Any] = {
                "policy_type": result.policy_type,
                "evaluation_seed": episode.evaluation_seed,
                "seed_index": episode.seed_index,
                "episode_index": episode.episode_index,
                "episode_seed": episode.episode_seed,
                "episode_return": episode.episode_return,
                "episode_length": episode.episode_length,
                "terminated": episode.terminated,
                "truncated": episode.truncated,
                "complete": episode.complete,
                "stop_reason": episode.stop_reason,
                "action_distribution_json": json.dumps(
                    dict(episode.action_distribution),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for action_name, column in zip(result.action_names, action_columns):
                row[column] = int(episode.action_distribution.get(action_name, 0))
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
    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint config must be a mapping")

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
    hidden_dim = _integer(model_config.get("hidden_dim", 512), name="model hidden_dim", minimum=1)
    try:
        model = DQNNetwork(
            action_count,
            input_shape=observation_shape,  # type: ignore[arg-type]
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
    training_metadata: dict[str, Any] = {
        "source_day14_run_id": source_run_id,
        "training_seed": saved_config.get("seed"),
        "training_budget": saved_config.get("total_steps"),
        "learning_rate": saved_config.get("learning_rate"),
        "batch_size": saved_config.get("batch_size"),
        "train_frequency": saved_config.get("train_frequency"),
        "replay_backend": saved_config.get("replay_backend", "cpu"),
        "training_device": saved_config.get("device"),
        "training_precision": saved_config.get("precision"),
        "training_config": dict(saved_config),
        "config_reference": None,
        "source_day14_manifest": manifest_value,
    }
    checkpoint_metadata: dict[str, Any] = {
        "path": _repository_path(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "step": step,
        "source_day14_run_id": source_run_id,
        "source_day14_manifest": manifest_value,
        "format_version": payload.get("format_version"),
        "model_config": {
            "num_actions": action_count,
            "input_shape": list(observation_shape),
            "hidden_dim": hidden_dim,
        },
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
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
