"""Reusable, frozen-policy evaluation for the Breakout DQN."""

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
from statistics import fmean, median, pstdev
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import nn

from breakout_env import ENVIRONMENT_ID, make_breakout_env
from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.training.diagnostics import ATARI_ACTION_NAMES
from breakout_rl.training.dqn_trainer import resolve_device


DEFAULT_MAX_STEPS_PER_EPISODE = 10_000
EVALUATION_SCHEMA_VERSION = 1
EnvironmentFactory = Callable[[], Any]


def _positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be greater than zero")
    return int(parsed)


def _non_negative_int(value: Any, *, name: str) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"{name} must not be negative")
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
    parsed = tuple(_non_negative_int(value, name="seed") for value in values)
    if len(set(parsed)) != len(parsed):
        raise ValueError("seeds must be unique so each evaluation seed is traceable")
    return parsed


@dataclass(frozen=True)
class EvaluationConfig:
    """Fixed settings shared by the Day 15 policy evaluations.

    ``episodes_per_seed`` is the number of episodes in each seed group. To
    avoid five identical environment resets inside a group, episode ``i``
    uses ``seed + i`` as its concrete Gymnasium reset seed. The base seed is
    retained on every result row as ``evaluation_seed``.
    """

    seeds: tuple[int, ...]
    episodes_per_seed: int = 5
    epsilon: float = 0.0
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE
    environment_id: str = ENVIRONMENT_ID
    source_day14_manifest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", _seed_values(self.seeds))
        object.__setattr__(
            self,
            "episodes_per_seed",
            _positive_int(self.episodes_per_seed, name="episodes_per_seed"),
        )
        object.__setattr__(self, "epsilon", _probability(self.epsilon, name="epsilon"))
        object.__setattr__(
            self,
            "max_steps_per_episode",
            _positive_int(
                self.max_steps_per_episode,
                name="max_steps_per_episode",
            ),
        )
        if not isinstance(self.environment_id, str) or not self.environment_id.strip():
            raise ValueError("environment_id must be a non-empty string")
        if self.source_day14_manifest is not None and not isinstance(
            self.source_day14_manifest,
            str,
        ):
            raise TypeError("source_day14_manifest must be a string or None")

    @property
    def total_episodes(self) -> int:
        return len(self.seeds) * self.episodes_per_seed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "seeds": list(self.seeds),
            "episodes_per_seed": self.episodes_per_seed,
            "epsilon": self.epsilon,
            "max_steps_per_episode": self.max_steps_per_episode,
            "environment_id": self.environment_id,
            "source_day14_manifest": self.source_day14_manifest,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EvaluationConfig":
        if not isinstance(values, Mapping):
            raise TypeError("evaluation config must be a JSON object")
        raw_environment = values.get("environment_id", values.get("environment", ENVIRONMENT_ID))
        raw_manifest = values.get("source_day14_manifest", values.get("day14_manifest"))
        return cls(
            seeds=tuple(values.get("seeds", ())),
            episodes_per_seed=values.get("episodes_per_seed", 5),
            epsilon=values.get("epsilon", 0.0),
            max_steps_per_episode=values.get(
                "max_steps_per_episode",
                DEFAULT_MAX_STEPS_PER_EPISODE,
            ),
            environment_id=raw_environment,
            source_day14_manifest=raw_manifest,
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
    """The complete raw observation of one evaluated episode."""

    evaluation_seed: int
    episode_seed: int
    seed_index: int
    episode_index: int
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    stop_reason: str
    action_distribution: Mapping[str, int]

    @property
    def evaluator_truncated(self) -> bool:
        """Whether the local safety cap ended the episode before the env did."""

        return self.stop_reason == "max_steps" and not self.terminated and not self.truncated

    @property
    def complete(self) -> bool:
        """Whether the environment, rather than the evaluator, ended the episode."""

        return self.terminated or self.truncated

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
            "stop_reason": self.stop_reason,
            "evaluator_truncated": self.evaluator_truncated,
            "complete": self.complete,
            "action_distribution": dict(self.action_distribution),
        }


def summarize_returns(values: Sequence[float]) -> dict[str, float | int]:
    """Aggregate the full evaluated population without hiding its spread."""

    if not values:
        raise ValueError("at least one episode return is required")
    finite_values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("episode returns must be finite")
    return {
        "count": len(finite_values),
        "mean_return": float(fmean(finite_values)),
        "median_return": float(median(finite_values)),
        "std_return": float(pstdev(finite_values)),
        "min_return": float(min(finite_values)),
        "max_return": float(max(finite_values)),
    }


@dataclass(frozen=True)
class EvaluationResult:
    """JSON/CSV-ready result for one policy under one frozen protocol."""

    policy_type: str
    model_id: str | None
    environment_id: str
    observation_shape: tuple[int, ...]
    action_count: int
    action_names: tuple[str, ...]
    evaluation_seeds: tuple[int, ...]
    episodes_per_seed: int
    evaluation_epsilon: float
    max_steps_per_episode: int
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
        summary = summarize_returns(returns)
        summary["mean_episode_length"] = float(fmean(lengths))
        summary["complete_episodes"] = sum(episode.complete for episode in self.episodes)
        summary["evaluator_truncated_episodes"] = sum(
            episode.evaluator_truncated for episode in self.episodes
        )
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00",
                "Z",
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
            "max_steps_per_episode": self.max_steps_per_episode,
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
    """Public policy seam shared by the episode harness."""

    policy_type: str

    def select_action(
        self,
        observation: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> int:
        ...


class RandomPolicy:
    """Choose a legal action uniformly using the episode-local RNG."""

    policy_type = "random"

    def __init__(self, action_count: int) -> None:
        self.action_count = _positive_int(action_count, name="action_count")

    def select_action(
        self,
        observation: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> int:
        del observation
        return int(rng.integers(0, self.action_count))


class DQNPolicy:
    """Run a frozen DQN in eval mode with an optional fixed epsilon."""

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
        self.action_count = _positive_int(action_count, name="action_count")
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
                "DQN model must return shape "
                f"(1, {self.action_count}); received {tuple(q_values.shape)}"
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


def _action_count(env: Any) -> int:
    raw_count = getattr(getattr(env, "action_space", None), "n", None)
    try:
        return _positive_int(raw_count, name="env.action_space.n")
    except (TypeError, ValueError) as error:
        raise ValueError("env.action_space.n must be a positive integer") from error


def _observation_shape(env: Any) -> tuple[int, ...]:
    raw_shape = getattr(getattr(env, "observation_space", None), "shape", None)
    if raw_shape is None:
        raise ValueError("env.observation_space.shape is required")
    try:
        shape = tuple(_positive_int(value, name="observation dimension") for value in raw_shape)
    except TypeError as error:
        raise ValueError("env.observation_space.shape must be a sequence") from error
    if not shape:
        raise ValueError("env.observation_space.shape must not be empty")
    return shape


def _environment_id(env: Any) -> str:
    spec = getattr(env, "spec", None)
    value = getattr(spec, "id", None)
    return str(value) if value else ENVIRONMENT_ID


def _episode_seed(base_seed: int, episode_index: int) -> int:
    """Derive distinct, reproducible reset seeds within a seed group."""

    return int(base_seed + episode_index - 1)


def _seed_action_space(env: Any, seed: int) -> None:
    action_space = getattr(env, "action_space", None)
    seed_method = getattr(action_space, "seed", None)
    if callable(seed_method):
        seed_method(seed)


def _runtime_metadata(
    *,
    requested_device: str,
    resolved_device: torch.device,
    evaluation_steps: int,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    resolved_name = str(resolved_device)
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "requested_device": requested_device,
        "resolved_device": resolved_name,
        "cuda_available": bool(torch.cuda.is_available()),
        "evaluation_steps": evaluation_steps,
        "wall_clock_seconds": float(wall_clock_seconds),
        "steps_per_second": float(evaluation_steps / max(wall_clock_seconds, 1e-9)),
    }
    if resolved_device.type == "cuda":
        index = 0 if resolved_device.index is None else int(resolved_device.index)
        metadata.update(
            {
                "cuda_device_index": index,
                "gpu_name": torch.cuda.get_device_name(index),
                "cuda_device_name": torch.cuda.get_device_name(index),
                "gpu_model": torch.cuda.get_device_name(index),
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
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE,
    model_id: str | None = None,
    training_metadata: Mapping[str, Any] | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    evaluation_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    """Evaluate a frozen model or Random policy on the same episode harness.

    ``episodes`` means episodes per value in ``seeds``. Each episode uses a
    deterministic derived reset seed and preserves the environment's raw
    reward, ``terminated`` flag, and ``truncated`` flag.
    """

    episodes_per_seed = _positive_int(episodes, name="episodes")
    evaluation_seeds = _seed_values(seeds)
    epsilon = _probability(epsilon, name="epsilon")
    max_steps_per_episode = _positive_int(
        max_steps_per_episode,
        name="max_steps_per_episode",
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
        policy: EvaluationPolicy
        if model is None:
            policy = RandomPolicy(action_count)
            policy_model_id = model_id or "random-policy"
        else:
            policy = DQNPolicy(
                model,
                action_count=action_count,
                device=resolved_device,
                epsilon=epsilon,
            )
            policy_model_id = model_id or "dqn-policy"

        episode_results: list[EpisodeResult] = []
        inference_context = torch.no_grad() if model is not None else nullcontext()
        with inference_context:
            for seed_index, evaluation_seed in enumerate(evaluation_seeds):
                for episode_index in range(1, episodes_per_seed + 1):
                    episode_seed = _episode_seed(evaluation_seed, episode_index)
                    observation, _ = env.reset(seed=episode_seed)
                    _seed_action_space(env, episode_seed)
                    rng = np.random.default_rng(episode_seed)
                    episode_return = 0.0
                    action_values: list[int] = []
                    terminated = False
                    truncated = False
                    stop_reason = "max_steps"

                    for _ in range(max_steps_per_episode):
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
                            stop_reason = "terminated" if terminated else "truncated"
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
                            stop_reason=stop_reason,
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
        model_id=policy_model_id,
        environment_id=environment_id,
        observation_shape=observation_shape,
        action_count=action_count,
        action_names=action_names,
        evaluation_seeds=evaluation_seeds,
        episodes_per_seed=episodes_per_seed,
        evaluation_epsilon=epsilon,
        max_steps_per_episode=max_steps_per_episode,
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
    """Write the complete JSON result and a flat per-episode CSV."""

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
        "stop_reason",
        "evaluator_truncated",
        "complete",
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
                "stop_reason": episode.stop_reason,
                "evaluator_truncated": episode.evaluator_truncated,
                "complete": episode.complete,
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


def read_evaluation_results(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: evaluation results must be a JSON object")
    episodes = payload.get("per_episode")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{source}: per_episode must be a non-empty array")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative path even when a worktree produced it."""

    parts = path.resolve().parts
    for index, part in enumerate(parts):
        if part.lower() == "assets":
            return Path(*parts[index:]).as_posix()
    return path.as_posix()


def _repository_path(path: Path) -> str:
    """Return a stable path for a known repository artifact directory."""

    resolved_parts = path.resolve().parts
    markers = {"assets", "configs", "evaluations", "experiments", "reports"}
    for index, part in enumerate(resolved_parts):
        if part.lower() in markers:
            return Path(*resolved_parts[index:]).as_posix()
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
    """A model plus provenance extracted from a Day 14 checkpoint."""

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
    """Load only the online network and preserve all evaluation provenance."""

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
    hidden_dim = model_config.get("hidden_dim", 512)
    hidden_dim = _positive_int(hidden_dim, name="model hidden_dim")
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
    source_run_id = str(
        payload.get("run_id")
        or checkpoint_path.parent.parent.name
    )
    manifest_value = (
        _portable_path(Path(source_day14_manifest))
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
        "config_reference": "configs/final/day14-vanilla-dqn.json",
        "source_day14_manifest": manifest_value,
    }
    checkpoint_metadata: dict[str, Any] = {
        "path": _portable_path(checkpoint_path),
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
    model_id = f"{source_run_id}@step-{step:08d}"
    return LoadedDQNCheckpoint(
        model=model,
        model_id=model_id,
        training_metadata=training_metadata,
        checkpoint_metadata=checkpoint_metadata,
    )


def load_day14_provenance(path: str | Path) -> dict[str, Any]:
    """Read the single final Day 14 variant used by formal evaluation."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: manifest must be a JSON object")
    if payload.get("status") != "completed":
        raise ValueError(f"{source}: Day 14 manifest is not completed")
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != 1:
        raise ValueError(f"{source}: expected one final Day 14 variant")
    variant = variants[0]
    if not isinstance(variant, Mapping):
        raise ValueError(f"{source}: final variant must be an object")
    config_values = variant.get("config_values", {})
    if not isinstance(config_values, Mapping):
        config_values = {}
    expected_step = variant.get("step_budget", config_values.get("total_steps"))
    expected_step = _positive_int(expected_step, name="Day 14 step budget")
    base_config = payload.get("base_config", {})
    if not isinstance(base_config, Mapping):
        base_config = {}
    config_reference = variant.get("config_path") or base_config.get("config_path")
    if isinstance(config_reference, str):
        config_reference = _repository_path(
            (source.parent / config_reference).resolve()
            if not Path(config_reference).is_absolute()
            else Path(config_reference)
        )
    run_dir_raw = variant.get("run_dir")
    run_dir: Path | None = None
    if isinstance(run_dir_raw, str) and run_dir_raw.strip():
        candidate = Path(run_dir_raw)
        run_dir = (
            candidate if candidate.is_absolute() else source.parent / candidate
        ).resolve()
    runtime: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    if run_dir is not None:
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        if config_path.is_file():
            try:
                run_config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"{config_path}: invalid JSON") from error
            if isinstance(run_config, Mapping) and isinstance(
                run_config.get("runtime"),
                Mapping,
            ):
                runtime = dict(run_config["runtime"])
        if summary_path.is_file():
            try:
                run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"{summary_path}: invalid JSON") from error
            if isinstance(run_summary, Mapping):
                summary = dict(run_summary)
    gpu_profiling_summary: dict[str, Any] = {}
    batch_comparison_path = source.parent.parent / (
        "day14-batch-size-profiling-final/batch-size-comparison.json"
    )
    if batch_comparison_path.is_file():
        try:
            batch_comparison = json.loads(
                batch_comparison_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"{batch_comparison_path}: invalid JSON") from error
        if isinstance(batch_comparison, Mapping):
            selected_batch_size = config_values.get("batch_size")
            selected_run: Mapping[str, Any] | None = None
            runs = batch_comparison.get("runs", [])
            if isinstance(runs, list):
                for candidate in runs:
                    if isinstance(candidate, Mapping) and candidate.get(
                        "batch_size"
                    ) == selected_batch_size:
                        selected_run = candidate
                        break
            profiling = selected_run.get("profiling", {}) if selected_run else {}
            if not isinstance(profiling, Mapping):
                profiling = {}
            gpu_profiling_summary = {
                "source": _repository_path(batch_comparison_path),
                "selected_batch_size": selected_batch_size,
                "selection_rule": batch_comparison.get("selection_rule", {}),
                "selected_run_id": selected_run.get("run_id") if selected_run else None,
                "end_to_end_sps": selected_run.get("end_to_end_sps")
                if selected_run
                else None,
                "training_samples_per_second": selected_run.get(
                    "training_samples_per_second"
                )
                if selected_run
                else None,
                "recent_return_mean": selected_run.get("mean_recent_episode_return")
                if selected_run
                else None,
                "profiling": {
                    "sample_csv": (
                        _repository_path(
                            (batch_comparison_path.parent / profiling["sample_csv"]).resolve()
                        )
                        if isinstance(profiling.get("sample_csv"), str)
                        else None
                    ),
                    "sampling_method": profiling.get("sampling_method"),
                    "gpu_utilization_percent": profiling.get(
                        "gpu_utilization_percent"
                    ),
                    "gpu_power_watts": profiling.get("gpu_power_watts"),
                    "gpu_memory_used_bytes": profiling.get("gpu_memory_used_bytes"),
                    "gpu_memory_total_bytes": profiling.get("gpu_memory_total_bytes"),
                },
            }
    return {
        "manifest_path": _portable_path(source),
        "experiment_id": payload.get("experiment_id", source.parent.name),
        "status": payload.get("status"),
        "run_id": variant.get("run_id"),
        "label": variant.get("label"),
        "expected_checkpoint_step": expected_step,
        "config_values": dict(config_values),
        "config_reference": config_reference,
        "selection_rationale": (
            gpu_profiling_summary.get("selection_rule", {})
            if gpu_profiling_summary
            else None
        ),
        "gpu_profiling_summary": gpu_profiling_summary,
        "run_dir": _repository_path(run_dir) if run_dir is not None else None,
        "runtime": runtime,
        "summary": summary,
        "requested_device": variant.get("requested_device"),
        "resolved_device": variant.get("resolved_device"),
        "stage": variant.get("stage", payload.get("stage")),
        "selection_rule": f"final checkpoint at {expected_step} environment steps",
    }


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    training: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Reject a checkpoint that is not the manifest's frozen final variant."""

    expected_run_id = provenance.get("run_id")
    actual_run_id = training.get("source_day14_run_id")
    if expected_run_id and actual_run_id != expected_run_id:
        raise ValueError(
            "checkpoint run id does not match the Day 14 final manifest: "
            f"expected {expected_run_id}, got {actual_run_id}"
        )
    expected_step = provenance.get("expected_checkpoint_step")
    actual_step = checkpoint.get("step")
    if expected_step is not None and actual_step != expected_step:
        raise ValueError(
            "checkpoint step does not match the Day 14 final selection rule: "
            f"expected {expected_step}, got {actual_step}"
        )
    expected_values = provenance.get("config_values", {})
    if isinstance(expected_values, Mapping):
        for field in ("total_steps", "seed", "learning_rate", "batch_size"):
            expected = expected_values.get(field)
            actual = training.get(
                {"total_steps": "training_budget", "seed": "training_seed"}.get(
                    field,
                    field,
                )
            )
            if expected is not None and actual != expected:
                raise ValueError(
                    "checkpoint training config does not match the Day 14 final "
                    f"manifest for {field}: expected {expected}, got {actual}"
                )


__all__ = [
    "DEFAULT_MAX_STEPS_PER_EPISODE",
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
