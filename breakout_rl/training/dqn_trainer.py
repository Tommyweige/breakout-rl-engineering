"""The reusable DQN update and the environment interaction trainer."""

from __future__ import annotations

import copy
import math
import operator
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn

from breakout_rl.exploration import LinearEpsilonSchedule, select_epsilon_greedy_action
from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.replay import ReplayBuffer
from breakout_rl.replay_tensors import ReplayTensorBatch, replay_batch_to_tensors
from breakout_rl.targets import hard_update, should_update_target
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.diagnostics import (
    ATARI_ACTION_NAMES,
    collect_runtime_metadata,
    replay_occupancy,
)
from breakout_rl.training.metrics import MetricsLogger


class NonFiniteTrainingError(RuntimeError):
    """Raised when a training value becomes NaN or infinity."""


@dataclass(frozen=True)
class DQNTrainingStepResult:
    """Inspectable values produced by one online-network optimizer update."""

    loss: float | None
    selected_q_values: torch.Tensor | None
    targets: torch.Tensor | None
    q_mean: float | None
    q_max: float | None
    q_min: float | None
    target_mean: float | None
    target_max: float | None
    td_error_mean_abs: float | None
    td_error_max_abs: float | None
    gradient_norm: float | None

    @property
    def td_loss(self) -> float | None:
        """Descriptive alias for the scalar Huber loss."""

        return self.loss

    @property
    def q_selected(self) -> torch.Tensor | None:
        """Descriptive alias matching the Bellman-update notation."""

        return self.selected_q_values


@dataclass(frozen=True)
class TrainingStepSnapshot:
    """Runtime values exposed after one real environment interaction."""

    global_step: int
    episode: int
    action: int
    action_source: str
    epsilon: float
    raw_reward: float
    current_raw_episode_return: float
    terminated: bool
    truncated: bool
    replay_size: int
    warmup_complete: bool
    optimizer_updated: bool
    optimizer_updates: int
    target_sync_count: int


TrainingStepCallback = Callable[
    [TrainingStepSnapshot, np.ndarray | None],
    None,
]


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def _require_finite(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise NonFiniteTrainingError(f"{name} contains non-finite values")


def _validate_update_batch(batch: ReplayTensorBatch) -> int:
    if not isinstance(batch, ReplayTensorBatch):
        raise TypeError("batch must be a ReplayTensorBatch")

    batch_size = int(batch.states.shape[0]) if batch.states.ndim >= 1 else 0
    if batch.states.ndim < 1 or batch.next_states.ndim < 1 or batch_size < 1:
        raise ValueError("states and next_states must contain a non-empty batch")
    if int(batch.next_states.shape[0]) != batch_size:
        raise ValueError("states and next_states must share batch size")

    for name, tensor in (
        ("actions", batch.actions),
        ("rewards", batch.rewards),
        ("terminated", batch.terminated),
        ("truncated", batch.truncated),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
            raise ValueError(f"{name} must have shape (B,)")
        if int(tensor.shape[0]) != batch_size:
            raise ValueError(f"{name} must share batch size with states")

    if not _is_integer_tensor(batch.actions):
        raise TypeError("actions must be an integer tensor")
    if batch.states.device != batch.next_states.device:
        raise ValueError("states and next_states must share a device")
    for name, tensor in (
        ("actions", batch.actions),
        ("rewards", batch.rewards),
        ("terminated", batch.terminated),
        ("truncated", batch.truncated),
    ):
        if tensor.device != batch.states.device:
            raise ValueError(f"{name} must share the state device")
    return batch_size


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared_norms: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        _require_finite(gradient, name="gradient")
        squared_norms.append(torch.sum(gradient.float() * gradient.float()))

    if not squared_norms:
        return 0.0
    total = torch.sqrt(torch.stack(squared_norms).sum())
    _require_finite(total, name="gradient norm")
    return float(total.item())


def _validate_gradient_clip_norm(value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("gradient_clip_norm must be a positive finite number or None")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("gradient_clip_norm must be a positive finite number or None")


def _ensure_optimizer_excludes_target(
    optimizer: torch.optim.Optimizer,
    target_network: nn.Module,
) -> None:
    target_parameter_ids = {id(parameter) for parameter in target_network.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if target_parameter_ids & optimizer_parameter_ids:
        raise ValueError("optimizer must update online parameters, not target parameters")


def dqn_training_step(
    online_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: ReplayTensorBatch,
    *,
    gamma: float,
    gradient_clip_norm: float | None,
    loss_fn: nn.Module | None = None,
    collect_diagnostics: bool = True,
) -> DQNTrainingStepResult:
    """Perform one vanilla-DQN Huber-loss update.

    The online network predicts every action, then ``gather`` selects the Q
    value for the action actually recorded in each transition. The target
    network supplies the detached vanilla-DQN Bellman target, so its
    parameters are not part of the backward pass or optimizer update.

    ``gradient_norm`` is the total norm before clipping when clipping is
    enabled. This makes the metric useful for spotting exploding gradients.
    """

    if not isinstance(online_network, nn.Module):
        raise TypeError("online_network must be a torch.nn.Module")
    if not isinstance(target_network, nn.Module):
        raise TypeError("target_network must be a torch.nn.Module")
    if online_network is target_network:
        raise ValueError("online_network and target_network must be distinct objects")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    _ensure_optimizer_excludes_target(optimizer, target_network)
    _validate_gradient_clip_norm(gradient_clip_norm)
    batch_size = _validate_update_batch(batch)

    all_q_values = online_network(batch.states)
    if not isinstance(all_q_values, torch.Tensor):
        raise TypeError("online_network must return a torch.Tensor")
    if all_q_values.ndim != 2 or int(all_q_values.shape[0]) != batch_size:
        raise ValueError("online_network output must have shape (B, action_count)")
    if int(all_q_values.shape[1]) < 1:
        raise ValueError("online_network must return at least one action value")
    if not all_q_values.is_floating_point():
        raise TypeError("online_network output must be a floating-point tensor")
    if collect_diagnostics:
        _require_finite(all_q_values, name="online Q-values")

    actions = batch.actions.to(dtype=torch.long)
    if collect_diagnostics and actions.numel() and (
        int(actions.min().item()) < 0
        or int(actions.max().item()) >= int(all_q_values.shape[1])
    ):
        raise ValueError("actions must be valid indices for the network output")
    selected_q_values = all_q_values.gather(1, actions[:, None]).squeeze(1)

    from breakout_rl.targets import compute_dqn_targets

    targets = compute_dqn_targets(
        batch.rewards,
        batch.next_states,
        batch.terminated,
        target_network,
        gamma,
    )
    if collect_diagnostics:
        _require_finite(targets, name="Bellman targets")

    criterion = loss_fn if loss_fn is not None else nn.SmoothL1Loss()
    loss = criterion(selected_q_values, targets)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("loss_fn must return a scalar tensor")
    if collect_diagnostics:
        _require_finite(loss, name="loss")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm: float | None = None
    if collect_diagnostics:
        gradient_norm = _gradient_norm(online_network.parameters())

    if gradient_clip_norm is not None:
        returned_norm = nn.utils.clip_grad_norm_(
            list(online_network.parameters()),
            max_norm=float(gradient_clip_norm),
        )
        if collect_diagnostics:
            _require_finite(returned_norm, name="gradient norm")
            # ``clip_grad_norm_`` returns the norm before clipping. Keep the
            # explicit value above as the metric contract even if PyTorch changes
            # the return scalar's dtype or device.
            gradient_norm = float(returned_norm.detach().item())

    optimizer.step()
    if collect_diagnostics:
        for parameter in online_network.parameters():
            _require_finite(parameter.data, name="online parameters")

        detached_selected_q_values = selected_q_values.detach()
        detached_targets = targets.detach()
        absolute_td_errors = (detached_targets - detached_selected_q_values).abs()
        return DQNTrainingStepResult(
            loss=float(loss.detach().item()),
            selected_q_values=detached_selected_q_values.clone(),
            targets=detached_targets.clone(),
            q_mean=float(all_q_values.detach().mean().item()),
            q_max=float(all_q_values.detach().max().item()),
            q_min=float(all_q_values.detach().min().item()),
            target_mean=float(detached_targets.mean().item()),
            target_max=float(detached_targets.max().item()),
            td_error_mean_abs=float(absolute_td_errors.mean().item()),
            td_error_max_abs=float(absolute_td_errors.max().item()),
            gradient_norm=gradient_norm,
        )

    return DQNTrainingStepResult(
        loss=None,
        selected_q_values=None,
        targets=None,
        q_mean=None,
        q_max=None,
        q_min=None,
        target_mean=None,
        target_max=None,
        td_error_mean_abs=None,
        td_error_max_abs=None,
        gradient_norm=None,
    )


def _as_uint8_observation(observation: Any, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(observation)
    if array.dtype != np.uint8:
        raise TypeError(
            "environment observations must have dtype uint8; "
            "normalization belongs at the model boundary"
        )
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"environment observation must have shape {expected_shape}; "
            f"received {tuple(array.shape)}"
        )
    return np.ascontiguousarray(array)


def _training_reward(raw_reward: float, *, clip: bool) -> float:
    if not math.isfinite(float(raw_reward)):
        raise NonFiniteTrainingError("environment reward is non-finite")
    if clip:
        return float(np.sign(raw_reward))
    return float(raw_reward)


def seed_everything(seed: int) -> None:
    """Seed the RNGs used before and during a CPU/GPU training run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` while refusing an unavailable explicit CUDA request."""

    if not isinstance(requested_device, str) or not requested_device.strip():
        raise ValueError("requested_device must be a non-empty string")
    request = requested_device.strip().lower()
    if request == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        index = 0
    elif request.startswith("cuda:") and request[5:].isdigit():
        index = int(request[5:])
    else:
        raise ValueError(
            "requested_device must be one of auto, cpu, cuda, or cuda:<index>"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({requested_device}), but it is not available; "
            "refusing to fall back to CPU."
        )
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} was requested, but only "
            f"{torch.cuda.device_count()} device(s) are available."
        )
    return torch.device(f"cuda:{index}")


class DQNTrainer:
    """Connect Breakout interaction, replay, updates, and checkpoints."""

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
        on_step: TrainingStepCallback | None = None,
    ) -> None:
        if not isinstance(config, DQNConfig):
            raise TypeError("config must be a DQNConfig")
        self.env = env
        self.config = config
        if on_step is not None and not callable(on_step):
            raise TypeError("on_step must be callable or None")
        self.on_step = on_step
        if config.cpu_threads is not None:
            torch.set_num_threads(config.cpu_threads)
        # Seed before constructing the default network and optimizer. When a
        # checkpoint is loaded below, its saved RNG states take precedence.
        seed_everything(config.seed)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.requested_device = config.requested_device
        self.device = resolve_device(self.requested_device)
        if self.device.type == "cuda":
            # This Windows/CUDA build rejects an explicit argument after
            # ``torch.manual_seed``. Switch the current device, then use the
            # no-argument form so peak-memory collection remains reliable.
            with torch.cuda.device(self.device):
                torch.cuda.reset_peak_memory_stats()

        raw_action_count = getattr(getattr(env, "action_space", None), "n", None)
        try:
            action_count = operator.index(raw_action_count)
        except TypeError as error:
            raise ValueError("env.action_space.n must be a positive integer") from error
        if action_count < 1:
            raise ValueError("env.action_space.n must be a positive integer")
        self.action_count = int(action_count)

        observation_shape = getattr(getattr(env, "observation_space", None), "shape", None)
        if observation_shape is None:
            raise ValueError("env.observation_space.shape is required")
        self.observation_shape = tuple(int(dimension) for dimension in observation_shape)
        if not self.observation_shape or any(dimension < 1 for dimension in self.observation_shape):
            raise ValueError("env.observation_space.shape must contain positive dimensions")

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

        self.replay = ReplayBuffer(
            config.replay_capacity,
            observation_shape=self.observation_shape,
        )
        self.schedule = LinearEpsilonSchedule(
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay_steps,
        )
        self.rng = np.random.default_rng(config.seed)
        self.episode = 0
        self.global_step = 0
        self.optimizer_updates = 0
        # Count the initial synchronization so the summary describes the
        # complete target-network lifecycle. Subsequent values are env steps.
        self.target_sync_count = 1
        self.last_target_sync_step = 0
        self._last_checkpoint: Path | None = None
        self._current_raw_episode_return = 0.0
        self._current_training_episode_return = 0.0
        self._current_episode_length = 0
        self._last_result: DQNTrainingStepResult | None = None
        self._action_counts = [0 for _ in range(self.action_count)]
        self._random_decision_count = 0
        self._greedy_decision_count = 0
        self._started_at = time.perf_counter()
        environment_spec = getattr(env, "spec", None)
        self._environment_id = getattr(environment_spec, "id", None) or "unavailable"
        self.metrics = MetricsLogger(
            self.run_dir,
            config,
            metadata={
                "environment_id": self._environment_id,
                "observation_shape": list(self.observation_shape),
                "action_count": self.action_count,
                "requested_device": self.requested_device,
                "resolved_device": self._resolved_device_name(),
                "precision": self.config.precision,
                "diagnostics_interval": self.config.diagnostics_interval,
                "metrics_flush_interval": self.config.metrics_flush_interval,
                "metrics_row_cadence": 1,
                "configured_cpu_threads": self.config.cpu_threads,
            },
        )

        if resume_from is not None:
            self.load_checkpoint(resume_from)

    def _select_action(self, observation: np.ndarray, epsilon: float) -> tuple[int, str]:
        state_tensor = observation_to_tensor(observation, device=self.device)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        if not isinstance(q_values, torch.Tensor) or q_values.shape != (1, self.action_count):
            raise ValueError("online_network must return shape (1, action_count) for one state")
        _require_finite(q_values, name="online Q-values")
        action, source = select_epsilon_greedy_action(
            q_values[0],
            epsilon,
            action_space_n=self.action_count,
            rng=self.rng,
        )
        return action, source

    def _update_once(self) -> DQNTrainingStepResult:
        batch = self.replay.sample(self.config.batch_size, self.rng)
        tensor_batch = replay_batch_to_tensors(batch, device=self.device)
        next_update = self.optimizer_updates + 1
        collect_diagnostics = (
            next_update % self.config.diagnostics_interval == 0
            or self.global_step >= self.config.total_steps
        )
        result = dqn_training_step(
            self.online_network,
            self.target_network,
            self.optimizer,
            tensor_batch,
            gamma=self.config.gamma,
            gradient_clip_norm=self.config.gradient_clip_norm,
            collect_diagnostics=collect_diagnostics,
        )
        self.optimizer_updates += 1
        self._last_result = result
        return result

    def _sync_target_if_due(self) -> None:
        if not should_update_target(
            self.global_step,
            self.config.target_update_interval,
        ):
            return
        if self.global_step == self.last_target_sync_step:
            return
        hard_update(self.target_network, self.online_network)
        self.target_network.eval()
        self.target_sync_count += 1
        self.last_target_sync_step = self.global_step

    def _render_callback_frame(self) -> np.ndarray | None:
        """Read one raw rendered frame only when a callback requested it."""

        if self.on_step is None:
            return None
        render = getattr(self.env, "render", None)
        if not callable(render):
            return None
        frame = render()
        if frame is None:
            return None
        frame_array = np.asarray(frame)
        if frame_array.ndim not in {2, 3}:
            raise ValueError(
                "env.render() must return a grayscale or color image; "
                f"received shape {tuple(frame_array.shape)}"
            )
        if frame_array.dtype != np.uint8:
            raise TypeError("env.render() must return a uint8 image")
        return np.ascontiguousarray(frame_array)

    def _notify_step_callback(
        self,
        *,
        action: int,
        action_source: str,
        epsilon: float,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
        result: DQNTrainingStepResult | None,
    ) -> None:
        if self.on_step is None:
            return
        snapshot = TrainingStepSnapshot(
            global_step=self.global_step,
            # The trainer increments episode after an ended transition. The
            # callback should still label the frame with the episode on screen.
            episode=self.episode + 1,
            action=action,
            action_source=action_source,
            epsilon=epsilon,
            raw_reward=raw_reward,
            current_raw_episode_return=self._current_raw_episode_return,
            terminated=terminated,
            truncated=truncated,
            replay_size=len(self.replay),
            warmup_complete=len(self.replay) >= self.config.learning_starts,
            optimizer_updated=result is not None,
            optimizer_updates=self.optimizer_updates,
            target_sync_count=self.target_sync_count,
        )
        self.on_step(snapshot, self._render_callback_frame())

    def _metric_row(
        self,
        *,
        action: int,
        epsilon: float,
        action_source: str,
        raw_reward: float,
        training_reward: float,
        completed_return: float | None,
        completed_length: int | None,
        result: DQNTrainingStepResult | None,
    ) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self._started_at, 1e-9)
        sps = float(self.global_step / elapsed)
        return {
            "global_step": self.global_step,
            "episode": self.episode,
            "raw_episode_return": completed_return,
            "episode_length": completed_length,
            "current_raw_episode_return": self._current_raw_episode_return,
            "current_training_episode_return": self._current_training_episode_return,
            "epsilon": epsilon,
            "loss": None if result is None else result.loss,
            "q_mean": None if result is None else result.q_mean,
            "q_max": None if result is None else result.q_max,
            "q_min": None if result is None else result.q_min,
            "target_mean": None if result is None else result.target_mean,
            "target_max": None if result is None else result.target_max,
            "td_error_mean_abs": (
                None if result is None else result.td_error_mean_abs
            ),
            "td_error_max_abs": None if result is None else result.td_error_max_abs,
            "gradient_norm": None if result is None else result.gradient_norm,
            "replay_size": len(self.replay),
            "replay_capacity": self.config.replay_capacity,
            "replay_occupancy": len(self.replay) / self.config.replay_capacity,
            "steps_per_second": sps,
            "sps": sps,
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
        }

    def _resolved_device_name(self) -> str:
        if self.device.type != "cuda":
            return str(self.device)
        index = 0 if self.device.index is None else self.device.index
        return f"cuda:{index}"

    def _runtime_metadata(self, elapsed: float) -> dict[str, Any]:
        return collect_runtime_metadata(
            seed=self.config.seed,
            device=self._resolved_device_name(),
            requested_device=self.requested_device,
            precision=self.config.precision,
            run_dir=self.run_dir,
            extra={
                "environment_id": self._environment_id,
                "observation_shape": list(self.observation_shape),
                "action_count": self.action_count,
                "wall_clock_seconds": float(elapsed),
                "steps_per_second": float(self.global_step / elapsed),
                "diagnostics_interval": self.config.diagnostics_interval,
                "metrics_flush_interval": self.config.metrics_flush_interval,
                "metrics_row_cadence": 1,
                "configured_cpu_threads": self.config.cpu_threads,
            },
        )

    def _summary(self, *, status: str = "completed", **extra: Any) -> dict[str, Any]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = max(time.perf_counter() - self._started_at, 1e-9)
        summary: dict[str, Any] = {
            "status": status,
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "seed": self.config.seed,
            "total_steps": self.global_step,
            "episodes": self.episode,
            "optimizer_updates": self.optimizer_updates,
            "target_sync_count": self.target_sync_count,
            "last_target_sync_step": self.last_target_sync_step,
            "replay_size": len(self.replay),
            "replay_occupancy": replay_occupancy(
                len(self.replay),
                self.config.replay_capacity,
            ),
            "steps_per_second": float(self.global_step / elapsed),
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

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
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
        """Save model/optimizer/RNG state without serializing the replay arrays."""

        self.metrics.flush()
        filename = f"step-{self.global_step:08d}"
        if suffix:
            filename += f"-{suffix}"
        path = self.checkpoint_dir / f"{filename}.pt"
        temporary_path = path.with_suffix(".tmp")
        payload = {
            "format_version": 1,
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "optimizer_updates": self.optimizer_updates,
            "episode": self.episode,
            "target_sync_count": self.target_sync_count,
            "last_target_sync_step": self.last_target_sync_step,
            "action_counts": list(self._action_counts),
            "random_decision_count": self._random_decision_count,
            "greedy_decision_count": self._greedy_decision_count,
            "config": self.config.to_dict(),
            "rng_state": self._rng_state(),
            "replay_saved": False,
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        self._last_checkpoint = path
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model state; replay is intentionally empty and warms up again."""

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
            # Keep compatibility with older PyTorch builds without the keyword.
            payload = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must contain a mapping")

        self.online_network.load_state_dict(payload["online_network"])
        self.target_network.load_state_dict(payload["target_network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload["global_step"])
        self.optimizer_updates = int(payload["optimizer_updates"])
        self.episode = int(payload["episode"])
        self.target_sync_count = int(payload["target_sync_count"])
        self.last_target_sync_step = int(payload["last_target_sync_step"])
        saved_action_counts = payload.get("action_counts")
        if isinstance(saved_action_counts, list) and len(saved_action_counts) == self.action_count:
            self._action_counts = [int(count) for count in saved_action_counts]
        self._random_decision_count = int(payload.get("random_decision_count", 0))
        self._greedy_decision_count = int(payload.get("greedy_decision_count", 0))
        self.target_network.eval()
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, dict):
            self._restore_rng_state(rng_state)
        self._last_checkpoint = checkpoint_path

    def train(self) -> dict[str, Any]:
        """Run environment interaction until ``config.total_steps``."""

        reset_seed = self.config.seed if self.global_step == 0 else None
        observation, _ = self.env.reset(seed=reset_seed)
        current_observation = _as_uint8_observation(
            observation,
            expected_shape=self.observation_shape,
        )

        try:
            while self.global_step < self.config.total_steps:
                epsilon = self.schedule.value(self.global_step)
                action, action_source = self._select_action(
                    current_observation,
                    epsilon,
                )
                self._action_counts[action] += 1
                if action_source == "random":
                    self._random_decision_count += 1
                elif action_source == "greedy":
                    self._greedy_decision_count += 1
                next_observation, raw_reward, terminated, truncated, _ = self.env.step(action)
                next_observation_array = _as_uint8_observation(
                    next_observation,
                    expected_shape=self.observation_shape,
                )
                raw_reward = float(raw_reward)
                training_reward = _training_reward(
                    raw_reward,
                    clip=self.config.reward_clip,
                )
                terminated = bool(terminated)
                truncated = bool(truncated)
                self.replay.add(
                    current_observation,
                    action,
                    training_reward,
                    next_observation_array,
                    terminated,
                    truncated,
                )

                self.global_step += 1
                self._current_raw_episode_return += raw_reward
                self._current_training_episode_return += training_reward
                self._current_episode_length += 1

                result: DQNTrainingStepResult | None = None
                if (
                    self.global_step >= self.config.learning_starts
                    and self.global_step % self.config.train_frequency == 0
                    and len(self.replay) >= self.config.batch_size
                ):
                    result = self._update_once()

                self._sync_target_if_due()

                self._notify_step_callback(
                    action=action,
                    action_source=action_source,
                    epsilon=epsilon,
                    raw_reward=raw_reward,
                    terminated=terminated,
                    truncated=truncated,
                    result=result,
                )

                completed_return: float | None = None
                completed_length: int | None = None
                if terminated or truncated:
                    completed_return = self._current_raw_episode_return
                    completed_length = self._current_episode_length
                    self.episode += 1
                    self._current_raw_episode_return = 0.0
                    self._current_training_episode_return = 0.0
                    self._current_episode_length = 0
                    reset_observation, _ = self.env.reset()
                    current_observation = _as_uint8_observation(
                        reset_observation,
                        expected_shape=self.observation_shape,
                    )
                else:
                    current_observation = next_observation_array

                self.metrics.write(
                    self._metric_row(
                        action=action,
                        epsilon=epsilon,
                        action_source=action_source,
                        raw_reward=raw_reward,
                        training_reward=training_reward,
                        completed_return=completed_return,
                        completed_length=completed_length,
                        result=result,
                    )
                )

                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()

            if self._last_checkpoint is None or self._last_checkpoint.stem != (
                f"step-{self.global_step:08d}"
            ):
                self.save_checkpoint()
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
    "DQNTrainer",
    "DQNTrainingStepResult",
    "NonFiniteTrainingError",
    "TrainingStepCallback",
    "TrainingStepSnapshot",
    "resolve_device",
    "seed_everything",
    "dqn_training_step",
]
