"""Show max-selection overestimation with a reproducible noisy estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from breakout_rl.analysis.overestimation import (
    plot_overestimation_bias,
    run_noise_sweep,
    write_sweep_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare noisy max selection with decoupled selection/evaluation."
    )
    parser.add_argument("--actions", type=int, default=4)
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--noise-stds", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--true-value", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.noise_stds is not None and args.noise_std is not None:
        raise SystemExit("choose either --noise-std or --noise-stds")
    noise_stds = (
        args.noise_stds
        if args.noise_stds is not None
        else [args.noise_std]
        if args.noise_std is not None
        else [0.1, 0.5, 1.0]
    )
    rows = run_noise_sweep(
        actions=args.actions,
        trials=args.trials,
        noise_stds=noise_stds,
        seed=args.seed,
        true_value=args.true_value,
    )
    print("noise_std  single_mean  vanilla_max  decoupled  vanilla_bias  decoupled_bias")
    for row in rows:
        print(
            f"{row['noise_std']:9.3f}  "
            f"{row['single_estimate_mean']:11.6f}  "
            f"{row['vanilla_max_mean']:11.6f}  "
            f"{row['decoupled_mean']:9.6f}  "
            f"{row['vanilla_bias']:12.6f}  "
            f"{row['decoupled_bias']:14.6f}"
        )
    if args.output is not None:
        write_sweep_json(rows, args.output)
    if args.plot_output is not None:
        plot_overestimation_bias(rows, args.plot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
