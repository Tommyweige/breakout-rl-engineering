"""Shared GPU PER sampling and priority-update stages for DQN trainers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from breakout_rl.prioritized_replay import (
    PrioritizedReplaySample,
    beta_for_transition,
)
from breakout_rl.replay_gpu import GPUReplayBuffer
from breakout_rl.training.config import DQNConfig


CudaStageMeasure = Callable[[str, Callable[[], Any]], Any]


@dataclass(frozen=True)
class PrioritizedUpdateSample:
    sample: PrioritizedReplaySample
    beta: float
    host_dispatch_seconds: float


def sample_prioritized_update(
    replay: GPUReplayBuffer,
    config: DQNConfig,
    *,
    transitions: int,
    measure_cuda: CudaStageMeasure,
) -> PrioritizedUpdateSample:
    """Sample one device-resident PER batch at the current transition count."""

    beta = beta_for_transition(
        transitions,
        beta_start=config.per_beta_start,
        beta_end=config.per_beta_end,
        anneal_transitions=config.per_beta_anneal_transitions,
    )
    started = time.perf_counter()
    sample = measure_cuda(
        "per_replay_sample",
        lambda: replay.sample_prioritized(
            config.batch_size,
            alpha=config.per_alpha,
            beta=beta,
        ),
    )
    return PrioritizedUpdateSample(
        sample=sample,
        beta=beta,
        host_dispatch_seconds=time.perf_counter() - started,
    )


def update_priorities_after_optimizer(
    replay: GPUReplayBuffer,
    update_sample: PrioritizedUpdateSample,
    absolute_td_errors: torch.Tensor | None,
    config: DQNConfig,
    *,
    collect_diagnostics: bool,
    measure_cuda: CudaStageMeasure,
) -> dict[str, Any]:
    """Apply detached TD priorities and gather sparse health/timing metrics."""

    if absolute_td_errors is None:
        raise RuntimeError("PER training update did not return per-sample TD errors")
    started = time.perf_counter()
    measure_cuda(
        "per_priority_update",
        lambda: replay.update_priorities(
            update_sample.sample.indices,
            absolute_td_errors,
        ),
    )
    update_dispatch_seconds = time.perf_counter() - started
    metrics: dict[str, Any] = {
        "per_alpha": config.per_alpha,
        "per_beta": update_sample.beta,
        "per_sampling_dispatch_seconds": update_sample.host_dispatch_seconds,
        "per_priority_update_dispatch_seconds": update_dispatch_seconds,
    }
    if not collect_diagnostics:
        return metrics

    metrics.update(
        measure_cuda(
            "per_diagnostics",
            lambda: replay.priority_statistics(alpha=config.per_alpha),
        )
    )
    sample = update_sample.sample
    sample_statistics = torch.stack(
        (
            sample.importance_weights.mean(),
            sample.importance_weights.min(),
            sample.importance_weights.max(),
            sample.probabilities.min(),
            sample.probabilities.max(),
            sample.effective_sample_size,
        )
    ).detach().cpu().tolist()
    metrics.update(
        {
            "importance_weight_mean": float(sample_statistics[0]),
            "importance_weight_min": float(sample_statistics[1]),
            "importance_weight_max": float(sample_statistics[2]),
            "sample_probability_batch_min": float(sample_statistics[3]),
            "sample_probability_batch_max": float(sample_statistics[4]),
            "sampling_effective_sample_size": float(sample_statistics[5]),
        }
    )
    return metrics


__all__ = [
    "PrioritizedUpdateSample",
    "sample_prioritized_update",
    "update_priorities_after_optimizer",
]
