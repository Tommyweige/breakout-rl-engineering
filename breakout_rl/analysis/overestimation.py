"""Reproducible toy experiments for max-selection overestimation."""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(parsed)


def _finite_non_negative(value: float, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def generate_noisy_estimates(
    *,
    actions: int,
    trials: int,
    noise_std: float,
    seed: int,
    true_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent noisy estimators with a shared true value."""

    action_count = _positive_int(actions, name="actions")
    trial_count = _positive_int(trials, name="trials")
    standard_deviation = _finite_non_negative(noise_std, name="noise_std")
    actual_value = float(true_value)
    if not np.isfinite(actual_value):
        raise ValueError("true_value must be finite")
    try:
        parsed_seed = operator.index(seed)
    except TypeError as error:
        raise ValueError("seed must be an integer") from error

    rng = np.random.default_rng(parsed_seed)
    shape = (trial_count, action_count)
    estimator_a = actual_value + rng.normal(0.0, standard_deviation, size=shape)
    estimator_b = actual_value + rng.normal(0.0, standard_deviation, size=shape)
    return estimator_a.astype(np.float64), estimator_b.astype(np.float64)


def simulate_overestimation(
    *,
    actions: int,
    trials: int,
    noise_std: float,
    seed: int,
    true_value: float = 0.0,
) -> dict[str, Any]:
    """Run one deterministic trial group and summarize its selection bias."""

    estimator_a, estimator_b = generate_noisy_estimates(
        actions=actions,
        trials=trials,
        noise_std=noise_std,
        seed=seed,
        true_value=true_value,
    )
    selected_actions = estimator_a.argmax(axis=1)
    row_indices = np.arange(estimator_a.shape[0])
    vanilla_values = estimator_a.max(axis=1)
    decoupled_values = estimator_b[row_indices, selected_actions]
    actual_value = float(true_value)
    counts = np.bincount(selected_actions, minlength=estimator_a.shape[1])

    vanilla_mean = float(np.mean(vanilla_values))
    decoupled_mean = float(np.mean(decoupled_values))
    return {
        "actions": int(estimator_a.shape[1]),
        "trials": int(estimator_a.shape[0]),
        "noise_std": float(noise_std),
        "seed": int(seed),
        "true_value": actual_value,
        "single_estimate_mean": float(np.mean(estimator_a)),
        "vanilla_max_mean": vanilla_mean,
        "decoupled_mean": decoupled_mean,
        "vanilla_bias": vanilla_mean - actual_value,
        "decoupled_bias": decoupled_mean - actual_value,
        "estimated_bias": vanilla_mean - actual_value,
        "selected_action_counts": {
            str(index): int(count) for index, count in enumerate(counts)
        },
    }


def run_noise_sweep(
    *,
    actions: int,
    trials: int,
    noise_stds: Iterable[float],
    seed: int,
    true_value: float = 0.0,
) -> list[dict[str, Any]]:
    """Run the same experiment for several noise scales."""

    scales = tuple(float(value) for value in noise_stds)
    if not scales:
        raise ValueError("noise_stds must contain at least one value")
    return [
        simulate_overestimation(
            actions=actions,
            trials=trials,
            noise_std=scale,
            seed=seed,
            true_value=true_value,
        )
        for scale in scales
    ]


def plot_overestimation_bias(
    rows: Iterable[dict[str, Any]],
    output: str | Path,
) -> Path:
    """Plot means and biases calculated by :func:`run_noise_sweep`."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = list(rows)
    if not values:
        raise ValueError("rows must contain at least one result")
    required = {
        "noise_std",
        "single_estimate_mean",
        "vanilla_max_mean",
        "decoupled_mean",
        "true_value",
    }
    if any(not required.issubset(row) for row in values):
        raise ValueError("rows are missing plot fields")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    x = [float(row["noise_std"]) for row in values]
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.plot(x, [row["single_estimate_mean"] for row in values], "o-", label="single estimate mean")
    axis.plot(x, [row["vanilla_max_mean"] for row in values], "o-", label="Vanilla max")
    axis.plot(x, [row["decoupled_mean"] for row in values], "o-", label="decoupled selection/evaluation")
    axis.plot(x, [row["true_value"] for row in values], "k--", label="true value")
    axis.set_title("Max selection bias from noisy estimates")
    axis.set_xlabel("noise std")
    axis.set_ylabel("mean estimate")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def write_sweep_json(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    """Write the machine-readable source used by the bias plot."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"results": list(rows)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "generate_noisy_estimates",
    "plot_overestimation_bias",
    "run_noise_sweep",
    "simulate_overestimation",
    "write_sweep_json",
]
