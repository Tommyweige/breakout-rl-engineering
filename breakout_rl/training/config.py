"""Validated configuration for the Day 12 DQN training loop."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from numbers import Integral, Real
from typing import Any, Mapping

from breakout_rl.models.factory import SUPPORTED_ARCHITECTURES, normalize_architecture
from breakout_rl.training.reward_shaping import validate_life_loss_penalty


SUPPORTED_ALGORITHMS = ("dqn", "double_dqn")


def _validated_int(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must not be negative")
        if minimum == 1:
            raise ValueError(f"{name} must be greater than zero")
        raise ValueError(f"{name} must be at least {minimum}")
    return int(parsed)


def _finite_real(value: float, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and parsed <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")
    return parsed


def _probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number between 0 and 1")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return parsed


def _device_request(value: str, *, name: str) -> str:
    """Validate the user-facing device request without resolving hardware."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be one of auto, cpu, or cuda")
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "cuda"}:
        return normalized
    if normalized.startswith("cuda:") and normalized[5:].isdigit():
        return normalized
    raise ValueError(
        f"{name} must be one of auto, cpu, cuda, or cuda:<index>"
    )


def _replay_transfer_request(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be direct or preallocated")
    normalized = value.strip().lower()
    if normalized not in {"direct", "preallocated"}:
        raise ValueError(f"{name} must be direct or preallocated")
    return normalized


def _replay_backend_request(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be cpu or gpu")
    normalized = value.strip().lower()
    if normalized not in {"cpu", "gpu"}:
        raise ValueError(f"{name} must be cpu or gpu")
    return normalized


def _replay_sampling_request(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be uniform or prioritized")
    normalized = value.strip().lower()
    if normalized not in {"uniform", "prioritized"}:
        raise ValueError(f"{name} must be uniform or prioritized")
    return normalized


def _algorithm_request(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be one of {', '.join(SUPPORTED_ALGORITHMS)}"
        )
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"{name} must be one of {', '.join(SUPPORTED_ALGORITHMS)}"
        )
    return normalized


def normalize_algorithm(value: str) -> str:
    """Normalize one supported DQN-family algorithm name."""

    return _algorithm_request(value, name="algorithm")


@dataclass(frozen=True)
class DQNConfig:
    """Development defaults for one reproducible DQN training run.

    These values are deliberately a runnable baseline, not a claim about the
    best Breakout hyperparameters. Later experiments can vary them while the
    training-loop semantics stay fixed.
    """

    total_steps: int = 10_000
    seed: int = 42
    algorithm: str = "dqn"
    architecture: str = "standard"
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    replay_capacity: int = 10_000
    learning_starts: int = 1_000
    train_frequency: int = 4
    target_update_interval: int = 1_000
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    gradient_clip_norm: float | None = 10.0
    reward_clip: bool = True
    life_loss_penalty: float = 0.0
    device: str = "cpu"
    precision: str = "float32"
    checkpoint_interval: int = 1_000
    checkpoint_steps: tuple[int, ...] = ()
    diagnostics_interval: int = 1
    metrics_flush_interval: int = 1
    cpu_threads: int | None = None
    replay_transfer: str = "direct"
    replay_backend: str = "cpu"
    replay_sampling: str = "uniform"
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_transitions: int = 500_000
    priority_epsilon: float = 1e-6
    profile_stages: bool = False
    num_envs: int = 1
    strict_action_selection_parity: bool = False
    contract_id: str | None = None
    contract_path: str | None = None

    def __post_init__(self) -> None:
        _validated_int(self.total_steps, name="total_steps", minimum=1)
        _validated_int(self.seed, name="seed", minimum=0)
        object.__setattr__(
            self,
            "algorithm",
            normalize_algorithm(self.algorithm),
        )
        object.__setattr__(
            self,
            "architecture",
            normalize_architecture(self.architecture),
        )

        _probability(self.gamma, name="gamma")

        _finite_real(self.learning_rate, name="learning_rate", minimum=0.0)
        batch_size = _validated_int(self.batch_size, name="batch_size", minimum=1)
        replay_capacity = _validated_int(
            self.replay_capacity,
            name="replay_capacity",
            minimum=1,
        )
        learning_starts = _validated_int(
            self.learning_starts,
            name="learning_starts",
            minimum=1,
        )
        if replay_capacity < batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if learning_starts < batch_size:
            raise ValueError("learning_starts must be at least batch_size")

        _validated_int(self.train_frequency, name="train_frequency", minimum=1)
        _validated_int(
            self.target_update_interval,
            name="target_update_interval",
            minimum=1,
        )
        _probability(self.epsilon_start, name="epsilon_start")
        _probability(self.epsilon_end, name="epsilon_end")
        _validated_int(self.epsilon_decay_steps, name="epsilon_decay_steps", minimum=1)

        if self.gradient_clip_norm is not None:
            _finite_real(
                self.gradient_clip_norm,
                name="gradient_clip_norm",
                minimum=0.0,
            )

        if not isinstance(self.reward_clip, bool):
            raise TypeError("reward_clip must be a boolean")
        object.__setattr__(
            self,
            "life_loss_penalty",
            validate_life_loss_penalty(self.life_loss_penalty),
        )
        object.__setattr__(self, "device", _device_request(self.device, name="device"))
        if not isinstance(self.precision, str) or not self.precision.strip():
            raise ValueError("precision must be a non-empty string")
        precision = self.precision.strip().lower()
        if precision == "fp32":
            precision = "float32"
        if precision != "float32":
            raise ValueError(
                "precision must be float32; mixed-precision training is not implemented"
            )
        object.__setattr__(self, "precision", precision)
        _validated_int(
            self.diagnostics_interval,
            name="diagnostics_interval",
            minimum=1,
        )
        _validated_int(
            self.metrics_flush_interval,
            name="metrics_flush_interval",
            minimum=1,
        )
        if self.cpu_threads is not None:
            _validated_int(self.cpu_threads, name="cpu_threads", minimum=1)
        if not isinstance(self.profile_stages, bool):
            raise TypeError("profile_stages must be a boolean")
        num_envs = _validated_int(self.num_envs, name="num_envs", minimum=1)
        checkpoint_steps = self.checkpoint_steps
        if isinstance(checkpoint_steps, (str, bytes)) or not isinstance(
            checkpoint_steps,
            Sequence,
        ):
            raise TypeError("checkpoint_steps must be a sequence of transition counts")
        normalized_checkpoint_steps: list[int] = []
        for step in checkpoint_steps:
            parsed_step = _validated_int(
                step,
                name="checkpoint_steps entries",
                minimum=1,
            )
            if parsed_step > self.total_steps:
                raise ValueError("checkpoint_steps entries cannot exceed total_steps")
            if parsed_step % num_envs != 0:
                raise ValueError(
                    "checkpoint_steps entries must align with complete vector steps"
                )
            normalized_checkpoint_steps.append(parsed_step)
        object.__setattr__(
            self,
            "checkpoint_steps",
            tuple(sorted(set(normalized_checkpoint_steps))),
        )
        if not isinstance(self.strict_action_selection_parity, bool):
            raise TypeError("strict_action_selection_parity must be a boolean")
        for name in ("contract_id", "contract_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        object.__setattr__(
            self,
            "replay_transfer",
            _replay_transfer_request(self.replay_transfer, name="replay_transfer"),
        )
        object.__setattr__(
            self,
            "replay_backend",
            _replay_backend_request(self.replay_backend, name="replay_backend"),
        )
        object.__setattr__(
            self,
            "replay_sampling",
            _replay_sampling_request(self.replay_sampling, name="replay_sampling"),
        )
        object.__setattr__(self, "per_alpha", _probability(self.per_alpha, name="per_alpha"))
        beta_start = _probability(self.per_beta_start, name="per_beta_start")
        beta_end = _probability(self.per_beta_end, name="per_beta_end")
        if beta_end < beta_start:
            raise ValueError("per_beta_end must be greater than or equal to per_beta_start")
        object.__setattr__(self, "per_beta_start", beta_start)
        object.__setattr__(self, "per_beta_end", beta_end)
        _validated_int(
            self.per_beta_anneal_transitions,
            name="per_beta_anneal_transitions",
            minimum=1,
        )
        object.__setattr__(
            self,
            "priority_epsilon",
            _finite_real(self.priority_epsilon, name="priority_epsilon", minimum=0.0),
        )
        if self.replay_backend == "gpu" and self.replay_transfer != "direct":
            raise ValueError("replay_transfer must be direct when replay_backend='gpu'")
        if self.replay_sampling == "prioritized" and self.replay_backend != "gpu":
            raise ValueError("prioritized replay currently requires replay_backend='gpu'")
        _validated_int(
            self.checkpoint_interval,
            name="checkpoint_interval",
            minimum=1,
        )

    @classmethod
    def smoke(
        cls,
        *,
        total_steps: int = 1_000,
        device: str = "cpu",
        algorithm: str = "dqn",
        architecture: str = "standard",
    ) -> "DQNConfig":
        """Return a small preset that still executes the real update order."""

        return cls(
            total_steps=total_steps,
            algorithm=algorithm,
            architecture=architecture,
            batch_size=8,
            replay_capacity=256,
            learning_starts=32,
            train_frequency=4,
            target_update_interval=100,
            epsilon_decay_steps=max(total_steps, 1_000),
            device=device,
            checkpoint_interval=max(100, min(total_steps, 500)),
        )

    @classmethod
    def debug(
        cls,
        *,
        total_steps: int = 10_000,
        device: str = "cuda",
        algorithm: str = "dqn",
        architecture: str = "standard",
    ) -> "DQNConfig":
        """Return the CUDA-first diagnostic run with frequent checkpoints.

        CPU remains an explicit portability override for tests and small
        sanity checks; the formal Day 13 debug preset targets CUDA.
        """

        return cls(
            total_steps=total_steps,
            algorithm=algorithm,
            architecture=architecture,
            batch_size=32,
            replay_capacity=10_000,
            learning_starts=1_000,
            train_frequency=4,
            target_update_interval=500,
            epsilon_decay_steps=max(total_steps, 10_000),
            device=device,
            checkpoint_interval=500,
        )

    @classmethod
    def day17_smoke(
        cls,
        *,
        total_steps: int = 10_000,
        device: str = "cuda",
        algorithm: str = "double_dqn",
        architecture: str = "standard",
    ) -> "DQNConfig":
        """Return the Day 17 canonical N=2 CUDA/GPU-Replay smoke config."""

        return cls(
            total_steps=total_steps,
            seed=42,
            algorithm=algorithm,
            architecture=architecture,
            gamma=0.99,
            learning_rate=1e-4,
            batch_size=32,
            replay_capacity=10_000,
            learning_starts=1_000,
            train_frequency=4,
            target_update_interval=500,
            epsilon_start=0.9,
            epsilon_end=0.05,
            epsilon_decay_steps=10_000,
            gradient_clip_norm=10.0,
            reward_clip=True,
            device=device,
            precision="float32",
            checkpoint_interval=max(1, total_steps),
            diagnostics_interval=100,
            metrics_flush_interval=500,
            cpu_threads=2,
            replay_transfer="direct",
            replay_backend="gpu",
            profile_stages=True,
            num_envs=2,
            strict_action_selection_parity=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of the configuration fields."""

        payload = asdict(self)
        payload["checkpoint_steps"] = list(self.checkpoint_steps)
        return payload

    @property
    def requested_device(self) -> str:
        """Return the hardware request before runtime resolution."""

        return self.device

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DQNConfig":
        """Reconstruct a config while ignoring future metadata fields."""

        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping")
        names = {field.name for field in fields(cls)}
        return cls(**{name: values[name] for name in names if name in values})

    def with_overrides(self, **overrides: Any) -> "DQNConfig":
        """Return a validated copy with selected command-line overrides."""

        return replace(self, **overrides)


__all__ = [
    "DQNConfig",
    "SUPPORTED_ALGORITHMS",
    "SUPPORTED_ARCHITECTURES",
    "normalize_algorithm",
    "normalize_architecture",
]
