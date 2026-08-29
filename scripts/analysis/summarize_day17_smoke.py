"""Compare Day 17 Vanilla and Double DQN smoke-run measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(payload)


def _metric(summary: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in summary:
            return summary[name]
    return None


def build_comparison(vanilla_path: str | Path, double_path: str | Path) -> dict[str, Any]:
    """Build a comparison from two real trainer summaries."""

    vanilla = _read_mapping(Path(vanilla_path))
    double = _read_mapping(Path(double_path))
    if vanilla.get("algorithm") != "dqn":
        raise ValueError("vanilla summary must have algorithm=dqn")
    if double.get("algorithm") != "double_dqn":
        raise ValueError("Double summary must have algorithm=double_dqn")
    invariant_names = (
        "seed",
        "num_envs",
        "total_steps",
        "replay_backend",
        "replay_transfer",
    )
    invariants = {
        name: vanilla.get(name) for name in invariant_names
    }
    mismatches = {
        name: (vanilla.get(name), double.get(name))
        for name in invariant_names
        if vanilla.get(name) != double.get(name)
    }
    if mismatches:
        raise ValueError(f"smoke runs do not share control variables: {mismatches}")

    rows: list[dict[str, Any]] = []
    for payload in (vanilla, double):
        runtime = payload.get("runtime", {})
        if not isinstance(runtime, Mapping):
            runtime = {}
        timings = runtime.get("stage_timings", {})
        if not isinstance(timings, Mapping):
            timings = {}
        target_timing = timings.get("target_forward", {})
        if not isinstance(target_timing, Mapping):
            target_timing = {}
        rows.append(
            {
                "algorithm": payload["algorithm"],
                "steps_per_second": _metric(
                    payload,
                    "steps_per_second",
                    "environment_transitions_per_second",
                ),
                "optimizer_updates_per_second": runtime.get(
                    "optimizer_updates_per_second",
                    payload.get("optimizer_updates_per_second"),
                ),
                "optimizer_updates": payload.get("optimizer_updates"),
                "wall_clock_seconds": runtime.get(
                    "wall_clock_seconds",
                    payload.get("wall_clock_seconds"),
                ),
                "peak_vram_bytes": runtime.get("cuda_peak_allocated_bytes"),
                "gpu_utilization_percent": runtime.get("gpu_utilization_percent"),
                "target_forward_gpu_seconds": target_timing.get("gpu_seconds"),
            }
        )
    return {
        "schema_version": 1,
        "control_variables": invariants,
        "interpretation": (
            "Double DQN performs an additional online next-state forward; "
            "these runs measure overhead only and are not a model-quality comparison."
        ),
        "runs": rows,
    }


def plot_comparison(payload: Mapping[str, Any], output: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("comparison must contain two run rows")
    labels = [str(run["algorithm"]) for run in runs]
    metrics = (
        ("steps_per_second", "transitions/s"),
        ("optimizer_updates_per_second", "optimizer updates/s"),
        ("target_forward_gpu_seconds", "target forward GPU seconds"),
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    for axis, (key, title) in zip(axes, metrics, strict=True):
        values = [float(run[key]) if run.get(key) is not None else 0.0 for run in runs]
        bars = axis.bar(labels, values, color=("#6b7280", "#2563eb"))
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )
    figure.suptitle("Day 17 smoke measurements: DQN vs Double DQN")
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize two same-config Day 17 smoke runs."
    )
    parser.add_argument("--vanilla-summary", type=Path, required=True)
    parser.add_argument("--double-summary", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day17/smoke-performance.json"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("assets/day17/smoke-performance.png"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_comparison(args.vanilla_summary, args.double_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_comparison(payload, args.plot_output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
