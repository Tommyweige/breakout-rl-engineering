"""Plot the measured toy Q-value overestimation simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt


def _records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = payload.get("results")
    if not isinstance(values, list) or not values:
        raise ValueError("simulation report must contain a non-empty results list")
    records = [value for value in values if isinstance(value, Mapping)]
    if len(records) != len(values):
        raise ValueError("simulation results must contain objects")
    return sorted(records, key=lambda value: float(value["noise_std"]))


def plot_overestimation(
    payload: Mapping[str, Any],
    *,
    output: Path,
) -> None:
    records = _records(payload)
    noise = [float(record["noise_std"]) for record in records]
    true_value = float(payload["true_best_value"])
    vanilla = [float(record["vanilla_max_mean"]) for record in records]
    decoupled = [float(record["decoupled_estimator_mean"]) for record in records]
    vanilla_bias = [float(record["vanilla_max_bias"]) for record in records]
    decoupled_bias = [float(record["decoupled_estimator_bias"]) for record in records]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(noise, [true_value] * len(noise), marker="o", label="True best value")
    axes[0].plot(noise, vanilla, marker="o", label="Vanilla max")
    axes[0].plot(noise, decoupled, marker="o", label="Decoupled estimator")
    axes[0].set_xlabel("Independent estimate noise std")
    axes[0].set_ylabel("Estimated value")
    axes[0].set_title("Measured estimates")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].plot(noise, vanilla_bias, marker="o", label="Vanilla max bias")
    axes[1].plot(noise, decoupled_bias, marker="o", label="Decoupled bias")
    axes[1].set_xlabel("Independent estimate noise std")
    axes[1].set_ylabel("Mean estimate − true best value")
    axes[1].set_title("Bias measured from the simulation")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    trials = int(payload.get("trials", 0))
    fig.suptitle(f"Q-value overestimation toy simulation ({trials:,} trials)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot overestimation simulation data")
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day16/overestimation-bias.png"),
    )
    args = parser.parse_args(argv)
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("simulation report must contain a JSON object")
    plot_overestimation(payload, output=args.output)
    print(f"Wrote overestimation figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
