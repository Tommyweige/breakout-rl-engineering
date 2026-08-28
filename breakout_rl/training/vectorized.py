"""Transition-counted vectorized DQN training.

The vectorized trainer keeps the existing DQN update contract, but advances
the environment, exploration schedule, replay insertion, and optimizer
schedule in units of actual environment transitions rather than vector
iterations.
"""

from __future__ import annotations

import copy
import operator
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from breakout_rl.exploration import (
    LinearEpsilonSchedule,
    select_epsilon_greedy_actions,
)
from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.replay import ReplayBuffer
from breakout_rl.replay_gpu import GPUReplayBuffer
from breakout_rl.replay_tensors import (
    PreallocatedReplayBatchTransfer,
    ReplayTensorBatch,
    replay_batch_to_tensors,
)
from breakout_rl.targets import hard_update
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.diagnostics import (
    ATARI_ACTION_NAMES,
    collect_runtime_metadata,
    replay_occupancy,
)
from breakout_rl.training.dqn_trainer import (
    DQNTrainingStepResult,
    NonFiniteTrainingError,
    _StageProfiler,
    _ensure_optimizer_excludes_target,
    _require_finite,
    _training_reward,
    dqn_training_step,
    resolve_device,
    seed_everything,
)
from breakout_rl.training.metrics import MetricsLogger


VectorScheduleEventKind = Literal["optimizer_update", "target_sync"]


@dataclass(frozen=True)
class VectorizedTrainingStepSnapshot:
    """Inspectable values for one environment transition in a vector step."""

    global_step: int
    vector_iteration: int
    environment_index: int
    episode: int
    action: int
    action_source: str
    epsilon: float
    raw_reward: float
    current_raw_episode_return: float
    terminated: bool
    truncated: bool
    replay_size: int
    optimizer_updated: bool
    optimizer_updates: int
    target_sync_count: int


VectorizedTrainingStepCallback = Callable[[VectorizedTrainingStepSnapshot], None]


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be greater than zero")
    return int(parsed)


def crossed_transition_boundaries(
    previous_step: int,
    current_step: int,
    interval: int,
) -> tuple[int, ...]:
    """Return interval boundaries crossed by ``(previous_step, current_step]``."""

    previous = operator.index(previous_step)
    current = operator.index(current_step)
    parsed_interval = _positive_int(interval, name="interval")
    if previous < 0 or current < previous:
        raise ValueError("current_step must be >= previous_step >= 0")
    first_boundary = ((previous // parsed_interval) + 1) * parsed_interval
    return tuple(range(first_boundary, current + 1, parsed_interval))


def _schedule_events(
    previous_step: int,
    current_step: int,
    *,
    train_frequency: int,
    target_update_interval: int,
) -> tuple[tuple[int, int, VectorScheduleEventKind], ...]:
    events: list[tuple[int, int, VectorScheduleEventKind]] = []
    events.extend(
        (boundary, 0, "optimizer_update")
        for boundary in crossed_transition_boundaries(
            previous_step,
            current_step,
            train_frequency,
        )
    )
    events.extend(
        (boundary, 1, "target_sync")
        for boundary in crossed_transition_boundaries(
            previous_step,
            current_step,
            target_update_interval,
        )
    )
    return tuple(sorted(events))


def _validate_observation_batch(
    observations: np.ndarray,
    *,
    num_envs: int,
    observation_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(observations)
    expected_shape = (num_envs, *observation_shape)
    if array.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} must have dtype uint8")
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; "
            f"received {tuple(array.shape)}"
        )
    return np.ascontiguousarray(array)


def _validate_vector_rewards(
    rewards: np.ndarray,
    *,
    num_envs: int,
) -> np.ndarray:
    array = np.asarray(rewards)
    if array.ndim != 1 or int(array.shape[0]) != num_envs:
        raise ValueError(f"rewards must have shape ({num_envs},)")
    if array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("rewards must contain numeric values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_vector_flags(
    flags: np.ndarray,
    *,
    num_envs: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(flags)
    if array.ndim != 1 or int(array.shape[0]) != num_envs:
        raise ValueError(f"{name} must have shape ({num_envs},)")
    if array.dtype.kind == "b":
        return np.ascontiguousarray(array, dtype=np.bool_)
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain boolean or 0/1 values")
    if array.size and not np.all((array == 0) | (array == 1)):
        raise TypeError(f"{name} must contain boolean or 0/1 values")
    return np.ascontiguousarray(array, dtype=np.bool_)


def _info_at(
    infos: Mapping[str, Any] | Any,
    key: str,
    index: int,
    default: Any,
) -> Any:
    if not isinstance(infos, Mapping) or key not in infos:
        return default
    values = infos[key]
    if values is None:
        return default
    if isinstance(values, Mapping):
        return values.get(index, default)
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return values.item()
        return values[index] if index < len(values) else default
    if isinstance(values, (list, tuple)):
        return values[index] if index < len(values) else default
    return values


def _final_observation_for(
    next_observations: np.ndarray,
    infos: Mapping[str, Any] | Any,
    index: int,
) -> Any:
    final_mask = _info_at(infos, "_final_obs", index, True)
    if final_mask is not None and not bool(final_mask):
        return next_observations[index]
    return _info_at(infos, "final_obs", index, next_observations[index])


def _final_observation_batch(
    next_observations: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    infos: Mapping[str, Any] | Any,
    *,
    num_envs: int,
    observation_shape: tuple[int, ...],
) -> np.ndarray:
    """Preserve final observations even for same-step autoreset APIs."""

    final_batch = np.array(next_observations, copy=True)
    done = np.logical_or(terminated, truncated)
    for index in np.flatnonzero(done):
        candidate = _final_observation_for(next_observations, infos, int(index))
        final_batch[index] = np.asarray(
            _validate_single_observation(
                candidate,
                expected_shape=observation_shape,
                name=f"final_observation[{index}]",
            )
        )
    return _validate_observation_batch(
        final_batch,
        num_envs=num_envs,
        observation_shape=observation_shape,
        name="final_next_observations",
    )


def _validate_single_observation(
    observation: Any,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(observation)
    if array.dtype != np.dtype(np.uint8) or tuple(array.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape} and dtype uint8; "
            f"received shape {tuple(array.shape)} dtype {array.dtype}"
        )
    return np.ascontiguousarray(array)


def _vector_num_envs(env: Any) -> int:
    raw_num_envs = getattr(env, "num_envs", None)
    if raw_num_envs is None:
        raw_envs = getattr(env, "envs", None)
        raw_num_envs = len(raw_envs) if raw_envs is not None else None
    if raw_num_envs is None:
        raise ValueError("vector environment must expose num_envs")
    return _positive_int(raw_num_envs, name="env.num_envs")


def _space_for(env: Any, *, kind: str) -> Any:
    single_name = f"single_{kind}_space"
    space = getattr(env, single_name, None)
    if space is None:
        space = getattr(env, f"{kind}_space", None)
    if space is None:
        raise ValueError(f"vector environment must expose {single_name}")
    return space


class VectorizedDQNTrainer:
    """Train vanilla DQN from multiple environments in one model batch."""

    def __init__(
        self,
        env: Any,
        config: DQNConfig,
        *,
        run_dir: str | Path,
        online_network: nn.Module | None = None,
        target_network: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        resume_from: str | Path | None = None,
        on_step: VectorizedTrainingStepCallback | None = None,
    ) -> None:
        if not isinstance(config, DQNConfig):
            raise TypeError("config must be a DQNConfig")
        if on_step is not None and not callable(on_step):
            raise TypeError("on_step must be callable or None")

        self.env = env
        self.config = config
        self.on_step = on_step
        self.num_envs = _vector_num_envs(env)
        if config.num_envs != self.num_envs:
            raise ValueError(
                f"config.num_envs ({config.num_envs}) must match env.num_envs "
                f"({self.num_envs})"
            )
        if config.cpu_threads is not None:
            torch.set_num_threads(config.cpu_threads)

        seed_everything(config.seed)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.requested_device = config.requested_device
        self.device = resolve_device(self.requested_device)
        if config.replay_backend == "gpu" and self.device.type != "cuda":
            raise RuntimeError(
                "replay_backend='gpu' requires CUDA; refusing to fall back to CPU"
            )
        if config.replay_backend == "gpu" and config.replay_transfer != "direct":
            raise ValueError(
                "replay_transfer must be direct when replay_backend='gpu'"
            )
        self._stage_profiler = _StageProfiler(
            enabled=config.profile_stages,
            device=self.device,
        )
        self._transfer_timing_merged = False
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.reset_peak_memory_stats()

        action_space = _space_for(env, kind="action")
        raw_action_count = getattr(action_space, "n", None)
        try:
            action_count = operator.index(raw_action_count)
        except TypeError as error:
            raise ValueError("single_action_space.n must be a positive integer") from error
        if action_count < 1:
            raise ValueError("single_action_space.n must be a positive integer")
        self.action_count = int(action_count)

        observation_space = _space_for(env, kind="observation")
        observation_shape = getattr(observation_space, "shape", None)
        if observation_shape is None:
            raise ValueError("single_observation_space.shape is required")
        try:
            self.observation_shape = tuple(operator.index(dimension) for dimension in observation_shape)
        except TypeError as error:
            raise ValueError("single_observation_space.shape must contain integers") from error
        if not self.observation_shape or any(dimension < 1 for dimension in self.observation_shape):
            raise ValueError("single_observation_space.shape must contain positive dimensions")

        if online_network is None:
            if len(self.observation_shape) != 3:
                raise ValueError("default DQNNetwork requires a three-dimensional observation")
            online_network = DQNNetwork(
                self.action_count,
                input_shape=self.observation_shape,  # type: ignore[arg-type]
            )
        self.online_network = online_network.to(self.device)
        if target_network is None:
            target_network = copy.deepcopy(self.online_network)
        self.target_network = target_network.to(self.device)
        hard_update(self.target_network, self.online_network)
        self.target_network.eval()
        self.optimizer = optimizer or torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )
        _ensure_optimizer_excludes_target(self.optimizer, self.target_network)

        if config.replay_backend == "gpu":
            self.replay: ReplayBuffer | GPUReplayBuffer = GPUReplayBuffer(
                config.replay_capacity,
                observation_shape=self.observation_shape,
                device=self.device,
            )
        else:
            self.replay = ReplayBuffer(
                config.replay_capacity,
                observation_shape=self.observation_shape,
            )
        self._replay_transfer = (
            PreallocatedReplayBatchTransfer(
                batch_size=config.batch_size,
                observation_shape=self.observation_shape,
                device=self.device,
                profile_stages=config.profile_stages,
            )
            if config.replay_backend == "cpu" and config.replay_transfer == "preallocated"
            else None
        )

        self.schedule = LinearEpsilonSchedule(
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay_steps,
        )
        self.rng = np.random.default_rng(config.seed)
        self.global_step = 0
        self.vector_iterations = 0
        self.physical_environment_steps = 0
        self.optimizer_updates = 0
        self.target_sync_count = 1
        self.last_target_sync_step = 0
        self._resume_rewarm_steps_remaining = 0
        self._last_checkpoint: Path | None = None
        self._last_result: DQNTrainingStepResult | None = None
        self._episode_counts = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_training_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.episode = 0
        self._action_counts = [0 for _ in range(self.action_count)]
        self._random_decision_count = 0
        self._greedy_decision_count = 0
        self.action_inference_batches = 0
        self.action_inference_transitions = 0
        self.replay_insertion_calls = 0
        self.replay_insertion_transitions = 0
        self._started_at = time.perf_counter()

        environment_spec = getattr(env, "spec", None)
        if environment_spec is None:
            raw_envs = getattr(env, "envs", None)
            if raw_envs:
                environment_spec = getattr(raw_envs[0], "spec", None)
        self._environment_id = getattr(environment_spec, "id", None) or "unavailable"
        self.metrics = MetricsLogger(
            self.run_dir,
            config,
            metadata={
                "environment_id": self._environment_id,
                "vectorized": True,
                "num_envs": self.num_envs,
                "observation_shape": list(self.observation_shape),
                "action_count": self.action_count,
                "requested_device": self.requested_device,
                "resolved_device": self._resolved_device_name(),
                "precision": self.config.precision,
                "diagnostics_interval": self.config.diagnostics_interval,
                "metrics_flush_interval": self.config.metrics_flush_interval,
                "metrics_row_cadence": 1,
                "global_step_definition": "accepted environment transitions",
                "replay_transfer": self.config.replay_transfer,
                "replay_backend": self.config.replay_backend,
                "replay_storage_device": str(self.device)
                if self.config.replay_backend == "gpu"
                else "cpu",
                "replay_bytes": int(self.replay.allocated_bytes),
            },
        )

        if resume_from is not None:
            self.load_checkpoint(resume_from)

    def _resolved_device_name(self) -> str:
        if self.device.type != "cuda":
            return str(self.device)
        index = 0 if self.device.index is None else self.device.index
        return f"cuda:{index}"

    def _reset_result_observation(self, result: Any) -> np.ndarray:
        if isinstance(result, tuple) and len(result) == 2:
            observations = result[0]
        else:
            observations = result
        return _validate_observation_batch(
            observations,
            num_envs=self.num_envs,
            observation_shape=self.observation_shape,
            name="reset observations",
        )

    def _reset_done(self, done: np.ndarray) -> np.ndarray | None:
        if not bool(done.any()):
            return None
        reset_mask = np.asarray(done, dtype=np.bool_).copy()
        try:
            result = self.env.reset(options={"reset_mask": reset_mask})
        except (AssertionError, NotImplementedError, TypeError) as error:
            reset_done = getattr(self.env, "reset_done", None)
            if not callable(reset_done):
                raise RuntimeError(
                    "vector environment must support reset(options={'reset_mask': mask}) "
                    "or reset_done(mask) to preserve per-environment episodes"
                ) from error
            result = reset_done(reset_mask)
        return self._reset_result_observation(result)

    def _select_actions(
        self,
        observations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_tensor = observation_to_tensor(observations, device=self.device)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        if not isinstance(q_values, torch.Tensor):
            raise TypeError("online_network must return a torch.Tensor")
        expected_shape = (self.num_envs, self.action_count)
        if tuple(q_values.shape) != expected_shape:
            raise ValueError(
                "online_network must return shape "
                f"{expected_shape} for a vectorized state batch; "
                f"received {tuple(q_values.shape)}"
            )
        if not q_values.is_floating_point():
            raise TypeError("online_network output must be a floating-point tensor")
        _require_finite(q_values, name="online Q-values")
        epsilons = np.asarray(
            [self.schedule.value(self.global_step + index) for index in range(self.num_envs)],
            dtype=np.float64,
        )
        actions, sources = select_epsilon_greedy_actions(
            q_values,
            epsilons,
            action_space_n=self.action_count,
            rng=self.rng,
        )
        return actions, sources.astype(object), epsilons

    def _executed_actions(
        self,
        requested_actions: np.ndarray,
        action_sources: np.ndarray,
        infos: Mapping[str, Any] | Any,
        *,
        active_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        executed_actions = np.asarray(requested_actions, dtype=np.int64).copy()
        sources = np.asarray(action_sources, dtype=object).copy()
        for index in range(active_count):
            requested = int(requested_actions[index])
            raw_executed = _info_at(
                infos,
                "fire_reset_executed_action",
                index,
                requested,
            )
            try:
                executed = operator.index(raw_executed)
            except TypeError as error:
                raise ValueError(
                    "fire_reset_executed_action must be an integer"
                ) from error
            if not 0 <= executed < self.action_count:
                raise ValueError(
                    f"environment executed illegal action {executed}; "
                    f"expected 0 <= action < {self.action_count}"
                )
            executed_actions[index] = int(executed)
            if str(sources[index]) == "random":
                self._random_decision_count += 1
            elif str(sources[index]) == "greedy":
                self._greedy_decision_count += 1
            if bool(_info_at(infos, "fire_reset_auto", index, False)):
                sources[index] = "fire_reset"
            self._action_counts[int(executed)] += 1
        return executed_actions, sources

    def _update_once(self) -> DQNTrainingStepResult:
        if self.config.replay_backend == "gpu":
            tensor_batch = self._stage_profiler.measure_cuda(
                "gpu_replay_gather_cast",
                lambda: self.replay.sample(self.config.batch_size),  # type: ignore[union-attr]
            )
        else:
            batch = self._stage_profiler.measure(
                "replay_sample",
                lambda: self.replay.sample(self.config.batch_size, self.rng),  # type: ignore[union-attr]
            )
            if self._replay_transfer is None:
                tensor_batch = self._stage_profiler.measure_cuda(
                    "replay_transfer",
                    lambda: replay_batch_to_tensors(batch, device=self.device),
                )
            else:
                tensor_batch = self._stage_profiler.measure_cuda(
                    "replay_transfer",
                    lambda: self._replay_transfer.transfer(batch),
                )
        next_update = self.optimizer_updates + 1
        collect_diagnostics = (
            next_update % self.config.diagnostics_interval == 0
            or self.global_step % self.config.checkpoint_interval == 0
            or self.global_step >= self.config.total_steps
        )
        result = self._stage_profiler.measure_cuda(
            "dqn_update",
            lambda: dqn_training_step(
                self.online_network,
                self.target_network,
                self.optimizer,
                tensor_batch,
                gamma=self.config.gamma,
                gradient_clip_norm=self.config.gradient_clip_norm,
                collect_diagnostics=collect_diagnostics,
                stage_measure=self._stage_profiler.measure_cuda,
            ),
        )
        self.optimizer_updates += 1
        self._last_result = result
        return result

    def _run_scheduled_events(
        self,
        previous_step: int,
        current_step: int,
        *,
        rewarm_steps_before: int,
    ) -> DQNTrainingStepResult | None:
        last_result: DQNTrainingStepResult | None = None
        warmup_end_step = previous_step + rewarm_steps_before
        for boundary, _priority, kind in _schedule_events(
            previous_step,
            current_step,
            train_frequency=self.config.train_frequency,
            target_update_interval=self.config.target_update_interval,
        ):
            if kind == "optimizer_update":
                if (
                    boundary >= self.config.learning_starts
                    and boundary > warmup_end_step
                    and len(self.replay) >= self.config.batch_size
                    and self._resume_rewarm_steps_remaining == 0
                ):
                    last_result = self._update_once()
            else:
                hard_update(self.target_network, self.online_network)
                self.target_network.eval()
                self.target_sync_count += 1
                self.last_target_sync_step = boundary
        return last_result

    def _metric_row(
        self,
        *,
        global_step: int,
        environment_index: int,
        vector_iteration: int,
        action: int,
        action_source: str,
        epsilon: float,
        raw_reward: float,
        training_reward: float,
        current_raw_episode_return: float,
        completed_return: float | None,
        completed_length: int | None,
        terminated: bool,
        truncated: bool,
        transition_batch_size: int,
        action_inference_batch_size: int,
        result: DQNTrainingStepResult | None,
    ) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self._started_at, 1e-9)
        environment_sps = float(self.global_step / elapsed)
        vector_sps = float(self.vector_iterations / elapsed)
        return {
            "global_step": global_step,
            "episode": self.episode,
            "raw_episode_return": completed_return,
            "episode_length": completed_length,
            "current_raw_episode_return": current_raw_episode_return,
            "current_training_episode_return": float(
                self._episode_training_returns[environment_index]
            ),
            "epsilon": epsilon,
            "loss": None if result is None else result.loss,
            "q_mean": None if result is None else result.q_mean,
            "q_max": None if result is None else result.q_max,
            "q_min": None if result is None else result.q_min,
            "target_mean": None if result is None else result.target_mean,
            "target_max": None if result is None else result.target_max,
            "td_error_mean_abs": None if result is None else result.td_error_mean_abs,
            "td_error_max_abs": None if result is None else result.td_error_max_abs,
            "gradient_norm": None if result is None else result.gradient_norm,
            "replay_size": len(self.replay),
            "replay_capacity": self.config.replay_capacity,
            "replay_occupancy": len(self.replay) / self.config.replay_capacity,
            "steps_per_second": environment_sps,
            "sps": environment_sps,
            "optimizer_updates": self.optimizer_updates,
            "optimizer_updated": result is not None,
            "target_sync_count": self.target_sync_count,
            "last_target_sync_step": self.last_target_sync_step,
            "raw_reward": raw_reward,
            "training_reward": training_reward,
            "action": action,
            "action_name": ATARI_ACTION_NAMES.get(action, f"ACTION_{action}"),
            "action_source": action_source,
            "noop_count": self._action_counts[0]
            if len(self._action_counts) > 0
            else 0,
            "fire_count": self._action_counts[1]
            if len(self._action_counts) > 1
            else 0,
            "right_count": self._action_counts[2]
            if len(self._action_counts) > 2
            else 0,
            "left_count": self._action_counts[3]
            if len(self._action_counts) > 3
            else 0,
            "random_decision_count": self._random_decision_count,
            "greedy_decision_count": self._greedy_decision_count,
            "random_decision_ratio": (
                self._random_decision_count / self.global_step
                if self.global_step
                else 0.0
            ),
            "environment_index": environment_index,
            "vector_iteration": vector_iteration,
            "num_envs": self.num_envs,
            "transition_batch_size": transition_batch_size,
            "environment_transitions_per_second": environment_sps,
            "vector_iterations_per_second": vector_sps,
            "action_inference_batch_size": action_inference_batch_size,
            "replay_insert_batch_size": transition_batch_size,
        }

    def _runtime_metadata(self, elapsed: float) -> dict[str, Any]:
        if (
            self.config.profile_stages
            and not self._transfer_timing_merged
            and self._replay_transfer is not None
        ):
            for name, timing in self._replay_transfer.timing_summary().items():
                self._stage_profiler.merge_cuda_stage(
                    name,
                    calls=int(timing.get("calls", 0)),
                    wall_seconds=float(timing.get("wall_seconds", 0.0)),
                    cpu_seconds=float(timing.get("cpu_seconds", 0.0)),
                    gpu_seconds=float(timing.get("gpu_seconds", 0.0)),
                )
            self._transfer_timing_merged = True
        return collect_runtime_metadata(
            seed=self.config.seed,
            device=self._resolved_device_name(),
            requested_device=self.requested_device,
            precision=self.config.precision,
            run_dir=self.run_dir,
            extra={
                "environment_id": self._environment_id,
                "vectorized": True,
                "num_envs": self.num_envs,
                "observation_shape": list(self.observation_shape),
                "action_count": self.action_count,
                "wall_clock_seconds": float(elapsed),
                "steps_per_second": float(self.global_step / elapsed),
                "environment_transitions_per_second": float(self.global_step / elapsed),
                "physical_environment_steps": self.physical_environment_steps,
                "physical_environment_steps_per_second": float(
                    self.physical_environment_steps / elapsed
                ),
                "vector_iterations": self.vector_iterations,
                "vector_iterations_per_second": float(self.vector_iterations / elapsed),
                "action_inference_batches": self.action_inference_batches,
                "action_inference_transitions": self.action_inference_transitions,
                "action_inference_transitions_per_second": float(
                    self.action_inference_transitions / elapsed
                ),
                "replay_insertion_calls": self.replay_insertion_calls,
                "replay_insertion_transitions": self.replay_insertion_transitions,
                "replay_insertion_transitions_per_second": float(
                    self.replay_insertion_transitions / elapsed
                ),
                "optimizer_updates_per_second": float(
                    self.optimizer_updates / elapsed
                ),
                "training_samples_per_second": float(
                    self.optimizer_updates * self.config.batch_size / elapsed
                ),
                "diagnostics_interval": self.config.diagnostics_interval,
                "metrics_flush_interval": self.config.metrics_flush_interval,
                "metrics_row_cadence": 1,
                "global_step_definition": "accepted environment transitions",
                "configured_cpu_threads": self.config.cpu_threads,
                "replay_transfer": self.config.replay_transfer,
                "replay_backend": self.config.replay_backend,
                "replay_storage_device": str(self.device)
                if self.config.replay_backend == "gpu"
                else "cpu",
                "replay_bytes": int(self.replay.allocated_bytes),
                "replay_rewarm_steps_remaining": self._resume_rewarm_steps_remaining,
                "stage_timings": self._stage_profiler.summary(),
            },
        )

    def _summary(self, *, status: str = "completed", **extra: Any) -> dict[str, Any]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = max(time.perf_counter() - self._started_at, 1e-9)
        summary: dict[str, Any] = {
            "status": status,
            "trainer": "vectorized_dqn",
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "seed": self.config.seed,
            "num_envs": self.num_envs,
            "total_steps": self.global_step,
            "total_transitions": self.global_step,
            "physical_environment_steps": self.physical_environment_steps,
            "vector_iterations": self.vector_iterations,
            "episodes": self.episode,
            "per_environment_episode_counts": self._episode_counts.tolist(),
            "optimizer_updates": self.optimizer_updates,
            "target_sync_count": self.target_sync_count,
            "last_target_sync_step": self.last_target_sync_step,
            "replay_backend": self.config.replay_backend,
            "replay_transfer": self.config.replay_transfer,
            "replay_bytes": int(self.replay.allocated_bytes),
            "replay_rewarm_steps_remaining": self._resume_rewarm_steps_remaining,
            "replay_size": len(self.replay),
            "replay_occupancy": replay_occupancy(
                len(self.replay),
                self.config.replay_capacity,
            ),
            "steps_per_second": float(self.global_step / elapsed),
            "environment_transitions_per_second": float(self.global_step / elapsed),
            "physical_environment_steps_per_second": float(
                self.physical_environment_steps / elapsed
            ),
            "vector_iterations_per_second": float(self.vector_iterations / elapsed),
            "action_inference_batches": self.action_inference_batches,
            "action_inference_transitions": self.action_inference_transitions,
            "action_inference_transitions_per_second": float(
                self.action_inference_transitions / elapsed
            ),
            "replay_insertion_calls": self.replay_insertion_calls,
            "replay_insertion_transitions": self.replay_insertion_transitions,
            "replay_insertion_transitions_per_second": float(
                self.replay_insertion_transitions / elapsed
            ),
            "optimizer_updates_per_second": float(self.optimizer_updates / elapsed),
            "training_samples_per_second": float(
                self.optimizer_updates * self.config.batch_size / elapsed
            ),
            "runtime": self._runtime_metadata(elapsed),
            "last_loss": None if self._last_result is None else self._last_result.loss,
            "last_q_mean": None if self._last_result is None else self._last_result.q_mean,
            "last_q_max": None if self._last_result is None else self._last_result.q_max,
            "last_q_min": None if self._last_result is None else self._last_result.q_min,
            "last_target_mean": None
            if self._last_result is None
            else self._last_result.target_mean,
            "last_target_max": None
            if self._last_result is None
            else self._last_result.target_max,
            "last_td_error_mean_abs": None
            if self._last_result is None
            else self._last_result.td_error_mean_abs,
            "last_td_error_max_abs": None
            if self._last_result is None
            else self._last_result.td_error_max_abs,
            "action_distribution": {
                ATARI_ACTION_NAMES.get(index, f"ACTION_{index}"): count
                for index, count in enumerate(self._action_counts)
            },
            "random_decision_count": self._random_decision_count,
            "greedy_decision_count": self._greedy_decision_count,
            "random_decision_ratio": (
                self._random_decision_count / self.global_step
                if self.global_step
                else 0.0
            ),
            "last_checkpoint": None
            if self._last_checkpoint is None
            else str(self._last_checkpoint),
        }
        summary.update(extra)
        return summary

    def _rng_state(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "action_rng": self.rng.bit_generator.state,
        }

    def _restore_rng_state(self, state: Mapping[str, Any]) -> None:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy_global" in state:
            np.random.set_state(state["numpy_global"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        if "action_rng" in state:
            self.rng.bit_generator.state = state["action_rng"]

    def save_checkpoint(self, *, suffix: str | None = None) -> Path:
        """Save model, optimizer, schedule, and per-environment counters."""

        self.metrics.flush()
        filename = f"step-{self.global_step:08d}"
        if suffix:
            filename += f"-{suffix}"
        path = self.checkpoint_dir / f"{filename}.pt"
        temporary_path = path.with_suffix(".tmp")
        payload = {
            "format_version": 1,
            "trainer": "vectorized_dqn",
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "vector_iterations": self.vector_iterations,
            "physical_environment_steps": self.physical_environment_steps,
            "optimizer_updates": self.optimizer_updates,
            "episode": self.episode,
            "episode_counts": self._episode_counts.tolist(),
            "episode_returns": self._episode_returns.tolist(),
            "episode_training_returns": self._episode_training_returns.tolist(),
            "episode_lengths": self._episode_lengths.tolist(),
            "target_sync_count": self.target_sync_count,
            "last_target_sync_step": self.last_target_sync_step,
            "action_counts": list(self._action_counts),
            "random_decision_count": self._random_decision_count,
            "greedy_decision_count": self._greedy_decision_count,
            "action_inference_batches": self.action_inference_batches,
            "action_inference_transitions": self.action_inference_transitions,
            "replay_insertion_calls": self.replay_insertion_calls,
            "replay_insertion_transitions": self.replay_insertion_transitions,
            "config": self.config.to_dict(),
            "rng_state": self._rng_state(),
            "replay_saved": False,
            "replay_rewarm_steps_remaining": self._resume_rewarm_steps_remaining,
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        self._last_checkpoint = path
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore state while warming a fresh replay buffer before updates."""

        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must contain a mapping")
        saved_num_envs = payload.get("num_envs")
        config_payload = payload.get("config")
        if saved_num_envs is None and isinstance(config_payload, Mapping):
            saved_num_envs = config_payload.get("num_envs")
        if saved_num_envs is not None and int(saved_num_envs) != self.num_envs:
            raise ValueError("checkpoint num_envs does not match vector environment")

        self.online_network.load_state_dict(payload["online_network"])
        self.target_network.load_state_dict(payload["target_network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload["global_step"])
        self.vector_iterations = int(payload.get("vector_iterations", 0))
        self.physical_environment_steps = int(
            payload.get("physical_environment_steps", self.global_step)
        )
        self.optimizer_updates = int(payload["optimizer_updates"])
        self.episode = int(payload.get("episode", 0))
        self.target_sync_count = int(payload["target_sync_count"])
        self.last_target_sync_step = int(payload["last_target_sync_step"])
        for key, target in (
            ("episode_counts", self._episode_counts),
            ("episode_returns", self._episode_returns),
            ("episode_training_returns", self._episode_training_returns),
            ("episode_lengths", self._episode_lengths),
        ):
            values = payload.get(key)
            if values is not None:
                array = np.asarray(values)
                if array.shape != target.shape:
                    raise ValueError(f"checkpoint {key} shape does not match num_envs")
                target[...] = array
        saved_action_counts = payload.get("action_counts")
        if isinstance(saved_action_counts, list) and len(saved_action_counts) == self.action_count:
            self._action_counts = [int(count) for count in saved_action_counts]
        self._random_decision_count = int(payload.get("random_decision_count", 0))
        self._greedy_decision_count = int(payload.get("greedy_decision_count", 0))
        self.action_inference_batches = int(payload.get("action_inference_batches", 0))
        self.action_inference_transitions = int(
            payload.get("action_inference_transitions", 0)
        )
        self.replay_insertion_calls = int(payload.get("replay_insertion_calls", 0))
        self.replay_insertion_transitions = int(
            payload.get("replay_insertion_transitions", 0)
        )
        saved_rewarm = payload.get("replay_rewarm_steps_remaining")
        if isinstance(saved_rewarm, int) and saved_rewarm > 0:
            self._resume_rewarm_steps_remaining = saved_rewarm
        else:
            self._resume_rewarm_steps_remaining = self.config.learning_starts
        self.target_network.eval()
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, Mapping):
            self._restore_rng_state(rng_state)
        self._last_checkpoint = checkpoint_path

    def train(self) -> dict[str, Any]:
        """Run until ``config.total_steps`` accepted transitions are stored."""

        reset_seed = self.config.seed if self.global_step == 0 else None
        observations = self._reset_result_observation(self.env.reset(seed=reset_seed))

        try:
            while self.global_step < self.config.total_steps:
                previous_step = self.global_step
                active_count = min(self.num_envs, self.config.total_steps - previous_step)
                current_observations = np.array(observations, copy=True)
                vector_iteration = self.vector_iterations + 1
                actions, action_sources, epsilons = self._stage_profiler.measure_cuda(
                    "batched_action_inference",
                    lambda: self._select_actions(current_observations),
                )
                self.action_inference_batches += 1
                self.action_inference_transitions += self.num_envs

                step_result = self._stage_profiler.measure(
                    "env_step",
                    lambda: self.env.step(actions),
                )
                if not isinstance(step_result, tuple) or len(step_result) != 5:
                    raise ValueError(
                        "vector environment step must return "
                        "(observations, rewards, terminated, truncated, infos)"
                    )
                next_observations, raw_rewards, terminated, truncated, infos = step_result
                next_observations = _validate_observation_batch(
                    next_observations,
                    num_envs=self.num_envs,
                    observation_shape=self.observation_shape,
                    name="next observations",
                )
                raw_rewards = _validate_vector_rewards(raw_rewards, num_envs=self.num_envs)
                terminated = _validate_vector_flags(
                    terminated,
                    num_envs=self.num_envs,
                    name="terminated",
                )
                truncated = _validate_vector_flags(
                    truncated,
                    num_envs=self.num_envs,
                    name="truncated",
                )
                final_next_observations = _final_observation_batch(
                    next_observations,
                    terminated,
                    truncated,
                    infos,
                    num_envs=self.num_envs,
                    observation_shape=self.observation_shape,
                )
                executed_actions, action_sources = self._executed_actions(
                    actions,
                    action_sources,
                    infos,
                    active_count=active_count,
                )
                training_rewards = np.asarray(
                    [
                        _training_reward(float(reward), clip=self.config.reward_clip)
                        for reward in raw_rewards
                    ],
                    dtype=np.float32,
                )

                active_slice = slice(0, active_count)
                insert_measure = (
                    self._stage_profiler.measure_cuda
                    if self.config.replay_backend == "gpu"
                    else self._stage_profiler.measure
                )
                insert_measure(
                    "batched_replay_insert",
                    lambda: self.replay.add_batch(  # type: ignore[union-attr]
                        current_observations[active_slice],
                        executed_actions[active_slice],
                        training_rewards[active_slice],
                        final_next_observations[active_slice],
                        terminated[active_slice],
                        truncated[active_slice],
                    ),
                )
                self.replay_insertion_calls += 1
                self.replay_insertion_transitions += active_count
                self.vector_iterations = vector_iteration
                self.physical_environment_steps += self.num_envs
                self.global_step += active_count

                self._episode_returns[:active_count] += raw_rewards[active_slice]
                self._episode_training_returns[:active_count] += training_rewards[active_slice]
                self._episode_lengths[:active_count] += 1
                done = np.logical_or(terminated, truncated)
                active_done = done.copy()
                active_done[active_count:] = False
                completed_returns = self._episode_returns.copy()
                completed_lengths = self._episode_lengths.copy()
                self.episode += int(active_done.sum())
                self._episode_counts[active_done] += 1
                self._episode_returns[active_done] = 0.0
                self._episode_training_returns[active_done] = 0.0
                self._episode_lengths[active_done] = 0

                reset_observations = self._reset_done(done)
                observations = np.array(final_next_observations, copy=True)
                if reset_observations is not None:
                    observations[done] = reset_observations[done]

                rewarm_before = self._resume_rewarm_steps_remaining
                self._resume_rewarm_steps_remaining = max(
                    0,
                    rewarm_before - active_count,
                )
                result = self._run_scheduled_events(
                    previous_step,
                    self.global_step,
                    rewarm_steps_before=rewarm_before,
                )

                for index in range(active_count):
                    completed_return = (
                        float(completed_returns[index])
                        if active_done[index]
                        else None
                    )
                    completed_length = (
                        int(completed_lengths[index])
                        if active_done[index]
                        else None
                    )
                    row = self._metric_row(
                        global_step=previous_step + index + 1,
                        environment_index=index,
                        vector_iteration=vector_iteration,
                        action=int(executed_actions[index]),
                        action_source=str(action_sources[index]),
                        epsilon=float(epsilons[index]),
                        raw_reward=float(raw_rewards[index]),
                        training_reward=float(training_rewards[index]),
                        current_raw_episode_return=float(self._episode_returns[index]),
                        completed_return=completed_return,
                        completed_length=completed_length,
                        terminated=bool(terminated[index]),
                        truncated=bool(truncated[index]),
                        transition_batch_size=active_count,
                        action_inference_batch_size=self.num_envs,
                        result=result,
                    )
                    self._stage_profiler.measure(
                        "metrics_write",
                        lambda row=row: self.metrics.write(row),
                    )
                    if self.on_step is not None:
                        self.on_step(
                            VectorizedTrainingStepSnapshot(
                                global_step=previous_step + index + 1,
                                vector_iteration=vector_iteration,
                                environment_index=index,
                                episode=int(self._episode_counts[index]),
                                action=int(executed_actions[index]),
                                action_source=str(action_sources[index]),
                                epsilon=float(epsilons[index]),
                                raw_reward=float(raw_rewards[index]),
                                current_raw_episode_return=float(
                                    self._episode_returns[index]
                                ),
                                terminated=bool(terminated[index]),
                                truncated=bool(truncated[index]),
                                replay_size=len(self.replay),
                                optimizer_updated=result is not None,
                                optimizer_updates=self.optimizer_updates,
                                target_sync_count=self.target_sync_count,
                            )
                        )

                if self.global_step % self.config.checkpoint_interval == 0:
                    self._stage_profiler.measure("checkpoint", self.save_checkpoint)

            if self._last_checkpoint is None or self._last_checkpoint.stem != (
                f"step-{self.global_step:08d}"
            ):
                self._stage_profiler.measure("checkpoint", self.save_checkpoint)
            summary = self._summary()
            self.metrics.update_runtime_metadata(summary["runtime"])
            self.metrics.write_summary(summary)
            return summary
        except NonFiniteTrainingError as error:
            diagnostic_checkpoint = self.save_checkpoint(suffix="diagnostic")
            summary = self._summary(
                status="failed_non_finite",
                error=str(error),
                diagnostic_checkpoint=str(diagnostic_checkpoint),
            )
            self.metrics.update_runtime_metadata(summary["runtime"])
            self.metrics.write_summary(summary)
            raise
        finally:
            self.metrics.close()


__all__ = [
    "VectorScheduleEventKind",
    "VectorizedDQNTrainer",
    "VectorizedTrainingStepCallback",
    "VectorizedTrainingStepSnapshot",
    "crossed_transition_boundaries",
]
