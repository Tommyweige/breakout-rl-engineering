"""Validated configuration for the Day 12 DQN training loop."""

from __future__ import annotations

import math
import operator
from dataclasses import asdict, dataclass, fields, replace
from numbers import Integral, Real
from typing import Any, Mapping


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


@dataclass(frozen=True)
class DQNConfig:
    """Development defaults for one reproducible DQN training run.

    These values are deliberately a runnable baseline, not a claim about the
    best Breakout hyperparameters. Later experiments can vary them while the
    training-loop semantics stay fixed.
    """

    total_steps: int = 10_000
    seed: int = 42
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
    device: str = "cpu"
    precision: str = "float32"
    checkpoint_interval: int = 1_000
    diagnostics_interval: int = 1
    metrics_flush_interval: int = 1
    cpu_threads: int | None = None
    replay_transfer: str = "direct"
    replay_backend: str = "cpu"

    def __post_init__(self) -> None:
        _validated_int(self.total_steps, name="total_steps", minimum=1)
        _validated_int(self.seed, name="seed", minimum=0)

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
        if self.replay_backend == "gpu" and self.replay_transfer != "direct":
            raise ValueError("replay_transfer must be direct when replay_backend='gpu'")
        _validated_int(
            self.checkpoint_interval,
            name="checkpoint_interval",
            minimum=1,
        )

    @classmethod
    def smoke(cls, *, total_steps: int = 1_000, device: str = "cpu") -> "DQNConfig":
        """Return a small preset that still executes the real update order."""

        return cls(
            total_steps=total_steps,
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
    def debug(cls, *, total_steps: int = 10_000, device: str = "cuda") -> "DQNConfig":
        """Return the CUDA-first diagnostic run with frequent checkpoints.

        CPU remains an explicit portability override for tests and small
        sanity checks; the formal Day 13 debug preset targets CUDA.
        """

        return cls(
            total_steps=total_steps,
            batch_size=32,
            replay_capacity=10_000,
            learning_starts=1_000,
            train_frequency=4,
            target_update_interval=500,
            epsilon_decay_steps=max(total_steps, 10_000),
            device=device,
            checkpoint_interval=500,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of the configuration fields."""

        return asdict(self)

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


__all__ = ["DQNConfig"]
