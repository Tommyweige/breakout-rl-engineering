"""Reusable checks and summaries for debugging DQN training runs."""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

    from breakout_rl.replay_tensors import ReplayTensorBatch


ATARI_ACTION_NAMES: dict[int, str] = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
}


class NonFiniteDiagnosticError(ValueError):
    """Raised when a diagnostic input contains NaN or infinity."""


@dataclass(frozen=True)
class FiniteCheck:
    """Result of checking whether a scalar, array, or tensor is finite."""

    name: str
    is_finite: bool
    non_finite_count: int
    total_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_finite": self.is_finite,
            "non_finite_count": self.non_finite_count,
            "total_count": self.total_count,
        }


@dataclass(frozen=True)
class FixedBatchOverfitResult:
    """Loss trace from repeatedly training on one fixed mini-batch."""

    losses: tuple[float, ...]

    @property
    def updates(self) -> int:
        return len(self.losses)

    @property
    def initial_loss(self) -> float:
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    @property
    def loss_reduction(self) -> float:
        return self.initial_loss - self.final_loss

    @property
    def loss_reduction_ratio(self) -> float:
        if self.initial_loss == 0.0:
            return 0.0
        return self.loss_reduction / abs(self.initial_loss)

    @property
    def passed(self) -> bool:
        """Whether the fixed target became easier to fit during the run."""

        return self.final_loss < self.initial_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_reduction": self.loss_reduction,
            "loss_reduction_ratio": self.loss_reduction_ratio,
            "passed": self.passed,
            "losses": list(self.losses),
        }


def _is_torch_tensor(value: Any) -> bool:
    """Detect tensors lazily so CSV-only analysis does not load PyTorch."""

    if value.__class__.__module__.split(".", maxsplit=1)[0] != "torch":
        return False
    import torch

    return isinstance(value, torch.Tensor)


def _numeric_array(value: Any, *, name: str) -> np.ndarray:
    if _is_torch_tensor(value):
        import torch

        if value.is_complex() or value.dtype == torch.bool:
            raise TypeError(f"{name} must be a real-valued tensor")
        return value.detach().float().cpu().numpy().reshape(-1)

    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    return array.astype(np.float64, copy=False).reshape(-1)


def check_finite(value: Any, *, name: str = "value") -> FiniteCheck:
    """Count non-finite values without hiding where a problem occurred."""

    if _is_torch_tensor(value):
        import torch

        if value.is_complex() or value.dtype == torch.bool:
            raise TypeError(f"{name} must be a real-valued tensor")
        finite = torch.isfinite(value)
        total_count = int(value.numel())
        non_finite_count = int((~finite).sum().item())
    else:
        array = _numeric_array(value, name=name)
        finite = np.isfinite(array)
        total_count = int(array.size)
        non_finite_count = int((~finite).sum())

    return FiniteCheck(
        name=name,
        is_finite=non_finite_count == 0,
        non_finite_count=non_finite_count,
        total_count=total_count,
    )


def require_finite(value: Any, *, name: str = "value") -> FiniteCheck:
    """Validate finiteness and raise a useful error when it is violated."""

    result = check_finite(value, name=name)
    if not result.is_finite:
        raise NonFiniteDiagnosticError(
            f"{name} contains {result.non_finite_count} non-finite values"
        )
    return result


def numeric_stats(values: Any, *, name: str = "value") -> dict[str, Any]:
    """Return finite-aware count, range, and average statistics."""

    array = _numeric_array(values, name=name)
    finite_values = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "count": int(array.size),
        "finite_count": int(finite_values.size),
        "non_finite_count": int(array.size - finite_values.size),
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
    }
    if finite_values.size:
        result.update(
            {
                "mean": float(np.mean(finite_values)),
                "median": float(np.median(finite_values)),
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
            }
        )
    return result


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Return the total L2 norm of finite parameter gradients."""

    import torch

    squared_norms: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        require_finite(parameter.grad, name="gradient")
        gradient = parameter.grad.detach().float()
        squared_norms.append(torch.sum(gradient * gradient))

    if not squared_norms:
        return 0.0
    total = torch.sqrt(torch.stack(squared_norms).sum())
    require_finite(total, name="gradient norm")
    return float(total.item())


def td_error_stats(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    """Summarize absolute temporal-difference errors independently of loss."""

    if not _is_torch_tensor(predictions) or not _is_torch_tensor(targets):
        raise TypeError("predictions and targets must be torch.Tensor values")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must share shape")
    errors = (targets.detach() - predictions.detach()).abs()
    return numeric_stats(errors, name="absolute TD error")


def replay_occupancy(size: int, capacity: int) -> dict[str, Any]:
    """Describe how full a replay buffer is without assuming a target ratio."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if size > capacity:
        raise ValueError("size cannot exceed capacity")
    return {
        "size": size,
        "capacity": capacity,
        "ratio": float(size / capacity),
    }


def action_counts(
    actions: Iterable[int],
    *,
    action_names: Mapping[int, str] = ATARI_ACTION_NAMES,
) -> dict[str, int]:
    """Count actions while retaining zero-count known actions for comparison."""

    counts = {str(name): 0 for name in action_names.values()}
    for action in actions:
        index = int(action)
        name = str(action_names.get(index, f"ACTION_{index}"))
        counts[name] = counts.get(name, 0) + 1
    return counts


def action_distribution(
    actions: Iterable[int],
    *,
    action_names: Mapping[int, str] = ATARI_ACTION_NAMES,
) -> dict[str, Any]:
    """Return action counts plus the observed total."""

    counts = action_counts(actions, action_names=action_names)
    return {"counts": counts, "total": int(sum(counts.values()))}


def decision_distribution(action_sources: Iterable[str]) -> dict[str, Any]:
    """Count random/greedy decisions and report their relative frequency."""

    random_count = 0
    greedy_count = 0
    other_count = 0
    for source in action_sources:
        if source == "random":
            random_count += 1
        elif source == "greedy":
            greedy_count += 1
        else:
            other_count += 1
    total = random_count + greedy_count + other_count
    return {
        "random": random_count,
        "greedy": greedy_count,
        "other": other_count,
        "total": total,
        "random_ratio": float(random_count / total) if total else None,
        "greedy_ratio": float(greedy_count / total) if total else None,
    }


def episode_return_trend(returns: Sequence[float]) -> dict[str, Any]:
    """Summarize completed episode returns without treating loss as reward."""

    stats = numeric_stats(returns, name="episode returns")
    finite_values = [float(value) for value in returns if math.isfinite(float(value))]
    stats.update(
        {
            "first": finite_values[0] if finite_values else None,
            "last": finite_values[-1] if finite_values else None,
            "delta": (
                finite_values[-1] - finite_values[0] if len(finite_values) >= 2 else None
            ),
        }
    )
    return stats


def steps_per_second(steps: int, elapsed_seconds: float) -> float:
    """Compute SPS with a safe zero-time boundary for short tests."""

    if steps < 0 or elapsed_seconds < 0:
        raise ValueError("steps and elapsed_seconds must be non-negative")
    if elapsed_seconds == 0:
        return 0.0
    return float(steps / elapsed_seconds)


def parse_numeric_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _series(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [
        number
        for row in rows
        if (number := parse_numeric_value(row.get(field))) is not None
    ]


def _final_numeric_row_value(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float | None:
    for row in reversed(rows):
        value = parse_numeric_value(row.get(field))
        if value is not None and math.isfinite(value):
            return value
    return None


def aggregate_training_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics CSV rows into a JSON-serializable diagnostic report."""

    materialized = [dict(row) for row in rows]
    step_values = [value for value in _series(materialized, "global_step") if math.isfinite(value)]
    episode_values = [value for value in _series(materialized, "episode") if math.isfinite(value)]
    returns = _series(materialized, "raw_episode_return")
    actions = [
        int(value)
        for value in _series(materialized, "action")
        if math.isfinite(value)
    ]
    if actions:
        action_report = action_distribution(actions)
    else:
        cumulative_counts = {
            "NOOP": _final_numeric_row_value(materialized, "noop_count"),
            "FIRE": _final_numeric_row_value(materialized, "fire_count"),
            "RIGHT": _final_numeric_row_value(materialized, "right_count"),
            "LEFT": _final_numeric_row_value(materialized, "left_count"),
        }
        counts = {
            name: int(value) if value is not None else 0
            for name, value in cumulative_counts.items()
        }
        action_report = {"counts": counts, "total": int(sum(counts.values()))}

    sources = [
        str(row["action_source"])
        for row in materialized
        if row.get("action_source") not in (None, "")
    ]
    decision_report = decision_distribution(sources)
    if not sources:
        random_count = _final_numeric_row_value(materialized, "random_decision_count")
        greedy_count = _final_numeric_row_value(materialized, "greedy_decision_count")
        if random_count is not None or greedy_count is not None:
            random_value = int(random_count or 0)
            greedy_value = int(greedy_count or 0)
            total = random_value + greedy_value
            decision_report = {
                "random": random_value,
                "greedy": greedy_value,
                "other": 0,
                "total": total,
                "random_ratio": float(random_value / total) if total else None,
                "greedy_ratio": float(greedy_value / total) if total else None,
            }

    non_finite_count = 0
    for row in materialized:
        for value in row.values():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                non_finite_count += 1

    replay_values = _series(materialized, "replay_size")
    replay_capacity_values = _series(materialized, "replay_capacity")
    replay_occupancy_values = _series(materialized, "replay_occupancy")
    sps_values = _series(materialized, "sps") or _series(
        materialized, "steps_per_second"
    )
    epsilon_values = _series(materialized, "epsilon")
    completed_episode_count = int(max(episode_values, default=len(returns)))

    q_summary = {
        field: numeric_stats(_series(materialized, field), name=field)
        for field in ("q_mean", "q_max", "q_min", "target_mean", "target_max")
    }
    return {
        "step_range": [
            int(min(step_values)) if step_values else None,
            int(max(step_values)) if step_values else None,
        ],
        "episodes_completed": completed_episode_count,
        "return_summary": episode_return_trend(returns),
        "loss_summary": numeric_stats(_series(materialized, "loss"), name="loss"),
        "q_value_summary": q_summary,
        "td_error_summary": {
            field: numeric_stats(_series(materialized, field), name=field)
            for field in ("td_error_mean_abs", "td_error_max_abs")
        },
        "gradient_summary": numeric_stats(
            _series(materialized, "gradient_norm"),
            name="gradient norm",
        ),
        "epsilon_range": [
            float(min(epsilon_values)) if epsilon_values else None,
            float(max(epsilon_values)) if epsilon_values else None,
        ],
        "replay_size": {
            "final": int(replay_values[-1]) if replay_values else None,
            "max": int(max(replay_values)) if replay_values else None,
            "capacity": (
                int(replay_capacity_values[-1]) if replay_capacity_values else None
            ),
            "occupancy": (
                float(replay_occupancy_values[-1])
                if replay_occupancy_values
                else None
            ),
        },
        "replay_occupancy": (
            {
                "size": int(replay_values[-1]),
                "capacity": int(replay_capacity_values[-1]),
                "ratio": float(replay_occupancy_values[-1]),
            }
            if replay_values and replay_capacity_values and replay_occupancy_values
            else None
        ),
        "sps": {
            "mean": float(np.mean(sps_values)) if sps_values else None,
            "final": float(sps_values[-1]) if sps_values else None,
            "max": float(max(sps_values)) if sps_values else None,
        },
        "action_distribution": action_report,
        "decision_distribution": decision_report,
        "non_finite_count": int(non_finite_count),
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
    except Exception:
        return "unavailable"


def _git_commit_sha(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    value = completed.stdout.strip()
    return value or "unavailable"


def _nvidia_smi_sample(device_index: int | None) -> dict[str, Any]:
    if device_index is None:
        return {
            "gpu_utilization_percent": None,
            "gpu_memory_total_bytes": None,
            "gpu_utilization_source": "unavailable",
        }
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "gpu_utilization_percent": None,
            "gpu_memory_total_bytes": None,
            "gpu_utilization_source": "nvidia-smi-unavailable",
        }
    try:
        completed = subprocess.run(
            [
                executable,
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if completed.returncode != 0:
            raise RuntimeError("nvidia-smi returned a non-zero status")
        fields = [field.strip() for field in completed.stdout.split(",")]
        if len(fields) < 2:
            raise ValueError("nvidia-smi returned an incomplete sample")
        return {
            "gpu_utilization_percent": float(fields[0]),
            "gpu_memory_total_bytes": int(float(fields[1]) * 1024 * 1024),
            "gpu_utilization_source": "nvidia-smi",
        }
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return {
            "gpu_utilization_percent": None,
            "gpu_memory_total_bytes": None,
            "gpu_utilization_source": "unavailable",
        }


def collect_runtime_metadata(
    *,
    seed: int,
    device: str,
    run_dir: str | Path,
    requested_device: str | None = None,
    precision: str = "float32",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect best-effort version and runtime context for a training run."""

    import torch

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    device_request = str(requested_device or device).strip().lower()
    if device_request == "auto":
        resolved_device = torch.device("cuda:0" if cuda_available else "cpu")
    else:
        resolved_device = torch.device(device)
    cuda_device_index: int | None = None
    if resolved_device.type == "cuda":
        cuda_device_index = 0 if resolved_device.index is None else int(resolved_device.index)

    cuda_device_name: str | None = None
    if cuda_available and cuda_device_index is not None:
        try:
            cuda_device_name = str(torch.cuda.get_device_name(cuda_device_index))
        except Exception:
            cuda_device_name = "unavailable"

    cuda_allocated_bytes: int | None = None
    cuda_peak_allocated_bytes: int | None = None
    cuda_reserved_bytes: int | None = None
    cuda_peak_reserved_bytes: int | None = None
    if cuda_available and cuda_device_index is not None:
        try:
            cuda_allocated_bytes = int(torch.cuda.memory_allocated(cuda_device_index))
            cuda_peak_allocated_bytes = int(
                torch.cuda.max_memory_allocated(cuda_device_index)
            )
            cuda_reserved_bytes = int(torch.cuda.memory_reserved(cuda_device_index))
            cuda_peak_reserved_bytes = int(
                torch.cuda.max_memory_reserved(cuda_device_index)
            )
        except Exception:
            cuda_allocated_bytes = None
            cuda_peak_allocated_bytes = None
            cuda_reserved_bytes = None
            cuda_peak_reserved_bytes = None

    gpu_sample = _nvidia_smi_sample(cuda_device_index if cuda_available else None)
    try:
        cpu_thread_count = int(torch.get_num_threads())
    except Exception:
        cpu_thread_count = None
    try:
        cpu_interop_thread_count = int(torch.get_num_interop_threads())
    except Exception:
        cpu_interop_thread_count = None

    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "gymnasium_version": _package_version("gymnasium"),
        "ale_version": _package_version("ale-py"),
        "numpy_version": np.__version__,
        "cpu_logical_count": os.cpu_count(),
        "cpu_thread_count": cpu_thread_count,
        "cpu_interop_thread_count": cpu_interop_thread_count,
        "device": str(resolved_device),
        "requested_device": device_request,
        "resolved_device": str(resolved_device),
        "precision": str(precision),
        "cuda_device_index": cuda_device_index,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "gpu_name": cuda_device_name,
        "gpu_model": cuda_device_name,
        "cuda_allocated_bytes": cuda_allocated_bytes,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_reserved_bytes": cuda_reserved_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
        **gpu_sample,
        "wall_clock_seconds": None,
        "steps_per_second": None,
        "seed": int(seed),
        "git_commit_sha": _git_commit_sha(Path(run_dir)),
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def _validate_fixed_batch(
    model: nn.Module,
    batch: ReplayTensorBatch,
    fixed_targets: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    import torch
    from torch import nn

    from breakout_rl.replay_tensors import ReplayTensorBatch

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(batch, ReplayTensorBatch):
        raise TypeError("batch must be a ReplayTensorBatch")
    if batch.states.ndim < 2 or batch.actions.ndim != 1:
        raise ValueError("batch states and actions have incompatible shapes")
    batch_size = int(batch.states.shape[0])
    if batch_size < 1 or int(batch.actions.shape[0]) != batch_size:
        raise ValueError("batch states and actions must share a non-empty batch size")
    if not isinstance(fixed_targets, torch.Tensor) or fixed_targets.ndim != 1:
        raise ValueError("fixed_targets must have shape (B,)")
    if int(fixed_targets.shape[0]) != batch_size:
        raise ValueError("fixed_targets must share batch size with states")
    if fixed_targets.device != batch.states.device:
        raise ValueError("fixed_targets must share the state device")
    actions = batch.actions.to(dtype=torch.long)
    return batch_size, actions


def run_fixed_batch_overfit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: ReplayTensorBatch,
    fixed_targets: torch.Tensor,
    *,
    updates: int = 200,
    loss_fn: nn.Module | None = None,
) -> FixedBatchOverfitResult:
    """Train repeatedly on one batch while keeping its targets unchanged."""

    import torch
    from torch import nn

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer")
    batch_size, actions = _validate_fixed_batch(model, batch, fixed_targets)
    criterion = loss_fn if loss_fn is not None else nn.SmoothL1Loss()
    targets = fixed_targets.detach().clone()
    losses: list[float] = []

    for _ in range(updates):
        q_values = model(batch.states)
        if not isinstance(q_values, torch.Tensor):
            raise TypeError("model must return a torch.Tensor")
        if q_values.ndim != 2 or int(q_values.shape[0]) != batch_size:
            raise ValueError("model output must have shape (B, action_count)")
        if actions.numel() and (
            int(actions.min().item()) < 0
            or int(actions.max().item()) >= int(q_values.shape[1])
        ):
            raise ValueError("batch actions must be valid model output indices")
        require_finite(q_values, name="fixed-batch Q-values")
        selected_q_values = q_values.gather(1, actions[:, None]).squeeze(1)
        loss = criterion(selected_q_values, targets)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        require_finite(loss, name="fixed-batch loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm(model.parameters())
        optimizer.step()
        for parameter in model.parameters():
            require_finite(parameter.data, name="fixed-batch parameters")
        losses.append(float(loss.detach().item()))

    return FixedBatchOverfitResult(losses=tuple(losses))


__all__ = [
    "ATARI_ACTION_NAMES",
    "FiniteCheck",
    "FixedBatchOverfitResult",
    "NonFiniteDiagnosticError",
    "action_counts",
    "action_distribution",
    "aggregate_training_metrics",
    "check_finite",
    "collect_runtime_metadata",
    "decision_distribution",
    "episode_return_trend",
    "gradient_norm",
    "numeric_stats",
    "parse_numeric_value",
    "replay_occupancy",
    "require_finite",
    "run_fixed_batch_overfit",
    "steps_per_second",
    "td_error_stats",
]
