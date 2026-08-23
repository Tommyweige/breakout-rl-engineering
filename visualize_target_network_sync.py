"""Plot real online/target DQN outputs across a hard-sync sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_OUTPUT = Path("assets/day11/target-network-sync.png")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse visualization options."""

    parser = argparse.ArgumentParser(
        description=(
            "Visualize actual online and target DQN outputs before and after "
            "an online optimizer step and hard target sync."
        )
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="forward device; cpu is the reproducible default (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for model initialization and synthetic input (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path (default: assets/day11/target-network-sync.png)",
    )
    return parser.parse_args(argv)


def _write_metadata(
    output: Path,
    *,
    inspection: dict[str, object],
    command: str,
) -> Path:
    """Persist the real values and command used to create the figure."""

    metadata: dict[str, object] = {
        "generated_by": "visualize_target_network_sync.py",
        "source": "inspect_target_network.py runtime JSON",
        "reproduction_command": command,
    }
    metadata.update(inspection)
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def build_visualization(
    *,
    seed: int,
    device_name: str,
    output: Path,
) -> dict[str, object]:
    """Generate the target-sync figure from actual model outputs."""

    inspection_script = Path(__file__).with_name("inspect_target_network.py")
    with tempfile.TemporaryDirectory(prefix="day11-target-network-") as temporary_directory:
        inspection_path = Path(temporary_directory) / "inspection.json"
        command = [
            sys.executable,
            str(inspection_script),
            "--device",
            device_name,
            "--seed",
            str(seed),
            "--json-output",
            str(inspection_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))

    phase_names = tuple(str(value) for value in inspection["phase_names"])
    num_actions = int(inspection["num_actions"])
    phases = np.arange(len(phase_names))
    online_values = np.asarray(inspection["online_sample_outputs"], dtype=np.float64)
    target_values = np.asarray(inspection["target_sample_outputs"], dtype=np.float64)
    differences = np.asarray(inspection["max_abs_diffs"], dtype=np.float64)

    figure, (q_axis, diff_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        height_ratios=(2.2, 1.0),
        constrained_layout=True,
    )

    colors = plt.get_cmap("tab10").colors
    for action_index in range(num_actions):
        color = colors[action_index % len(colors)]
        q_axis.plot(
            phases,
            online_values[:, action_index],
            color=color,
            marker="o",
            linewidth=2,
        )
        q_axis.plot(
            phases,
            target_values[:, action_index],
            color=color,
            marker="o",
            linestyle="--",
            alpha=0.65,
        )

    q_axis.set_title("Real DQN output for synthetic sample 0")
    q_axis.set_ylabel("Q-value")
    q_axis.set_xticks(phases, phase_names)
    q_axis.grid(axis="y", alpha=0.25)
    q_axis.legend(
        handles=[
            Line2D([0], [0], color=colors[index], linewidth=2, label=f"action {index}")
            for index in range(num_actions)
        ]
        + [
            Line2D([0], [0], color="#333333", linewidth=2, label="online (solid)"),
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=2,
                linestyle="--",
                label="target (dashed)",
            ),
        ],
        ncol=3,
        fontsize=8,
        loc="best",
    )

    bars = diff_axis.bar(
        phases,
        differences,
        color=("#4c78a8", "#f58518", "#54a24b"),
        width=0.58,
    )
    diff_axis.set_title("Maximum absolute output difference")
    diff_axis.set_ylabel("max |online − target|")
    diff_axis.set_xticks(phases, phase_names)
    diff_axis.set_ylim(bottom=0.0, top=max(float(differences.max()) * 1.25, 1e-6))
    diff_axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, differences, strict=True):
        diff_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.suptitle(
        "Day 11 Target Network: online changes first, target changes at hard sync",
        fontsize=15,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    command = (
        "conda run --name breakout-rl-engineering python "
        "visualize_target_network_sync.py "
        f"--device {device_name} --seed {seed}"
    )
    _write_metadata(output, inspection=inspection, command=command)
    return inspection


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        inspection = build_visualization(
            seed=args.seed,
            device_name=args.device,
            output=args.output,
        )
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error

    print(f"Saved figure: {args.output}")
    print(f"Saved metadata: {args.output.with_suffix('.json')}")
    print(
        "max abs diffs: "
        + ", ".join(f"{value:.8f}" for value in inspection["max_abs_diffs"])
    )


if __name__ == "__main__":
    main()
