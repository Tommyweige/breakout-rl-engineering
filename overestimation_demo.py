"""Simulate the max-selection overestimation mechanism with NumPy."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _finite_non_negative(value: float, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def run_simulation(
    *,
    seed: int,
    trials: int,
    true_action_values: Sequence[float],
    noise_stds: Sequence[float],
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    if trials < 1 or chunk_size < 1:
        raise ValueError("trials and chunk_size must be positive")
    true_values = np.asarray(true_action_values, dtype=np.float64)
    if true_values.ndim != 1 or true_values.size < 2:
        raise ValueError("true_action_values must contain at least two actions")
    if not np.isfinite(true_values).all():
        raise ValueError("true_action_values must be finite")
    parsed_stds = tuple(
        _finite_non_negative(value, name="noise_std") for value in noise_stds
    )
    if not parsed_stds:
        raise ValueError("noise_stds must contain at least one value")

    rng = np.random.default_rng(seed)
    true_best = float(true_values.max())
    results: list[dict[str, Any]] = []
    for noise_std in parsed_stds:
        vanilla_values: list[np.ndarray] = []
        decoupled_values: list[np.ndarray] = []
        remaining = trials
        while remaining:
            count = min(remaining, chunk_size)
            selection_noise = rng.normal(
                0.0,
                noise_std,
                size=(count, true_values.size),
            )
            evaluation_noise = rng.normal(
                0.0,
                noise_std,
                size=(count, true_values.size),
            )
            noisy_values = true_values[None, :] + selection_noise
            selected_actions = np.argmax(noisy_values, axis=1)
            vanilla_values.append(np.max(noisy_values, axis=1))
            decoupled_values.append(
                true_values[selected_actions]
                + evaluation_noise[np.arange(count), selected_actions]
            )
            remaining -= count

        vanilla = np.concatenate(vanilla_values)
        decoupled = np.concatenate(decoupled_values)
        results.append(
            {
                "noise_std": noise_std,
                "true_best_value": true_best,
                "vanilla_max_mean": float(vanilla.mean()),
                "vanilla_max_std": float(vanilla.std()),
                "vanilla_max_bias": float(vanilla.mean() - true_best),
                "decoupled_estimator_mean": float(decoupled.mean()),
                "decoupled_estimator_std": float(decoupled.std()),
                "decoupled_estimator_bias": float(decoupled.mean() - true_best),
            }
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "purpose": "Toy simulation of max-selection Q-value overestimation",
        "source": "NumPy Monte Carlo simulation; not a Breakout trajectory",
        "seed": int(seed),
        "trials": int(trials),
        "chunk_size": int(chunk_size),
        "true_action_values": [float(value) for value in true_values],
        "true_best_value": true_best,
        "noise_stds": list(parsed_stds),
        "results": results,
        "interpretation_boundary": (
            "The simulation isolates selection noise. It demonstrates a mechanism, "
            "not the magnitude of bias in a trained Breakout checkpoint."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible toy data for Q-value overestimation."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=500_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--true-action-values",
        nargs="+",
        type=float,
        default=[1.0, 1.0, 1.0, 1.0],
    )
    parser.add_argument(
        "--noise-stds",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day16/overestimation-bias.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must not be negative")
    payload = run_simulation(
        seed=args.seed,
        trials=args.trials,
        true_action_values=args.true_action_values,
        noise_stds=args.noise_stds,
        chunk_size=args.chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Simulation report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
