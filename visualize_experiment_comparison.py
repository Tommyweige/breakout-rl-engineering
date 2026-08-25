"""Render Day 14 comparison figures directly from a manifest and run CSVs."""

from __future__ import annotations

import argparse
import multiprocessing
import queue
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.experiments import load_manifest_run_paths, read_metrics, write_json_object


METRIC_SPECS: dict[str, tuple[str, str, str]] = {
    "return": ("raw_episode_return", "Episode return", "Raw episode return"),
    "loss": ("loss", "Loss", "Huber loss"),
    "q": ("q_mean", "Q-value mean", "Q mean"),
    "epsilon": ("epsilon", "Exploration", "Epsilon"),
    "sps": ("sps", "Throughput", "Steps per second"),
}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _paired_series(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        x_value = _number(row.get("global_step"))
        y_value = _number(row.get(field))
        if x_value is None or y_value is None:
            continue
        x_values.append(x_value)
        y_values.append(y_value)
    return x_values, y_values


def _metadata_payload(
    manifest_path: str | Path,
    destination: Path,
    metadata_destination: Path,
    selected: Sequence[str],
    entries: Sequence[tuple[Mapping[str, Any], Path]],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "command": list(sys.argv),
        "manifest": str(Path(manifest_path).resolve()),
        "output": str(destination.resolve()),
        "metadata_output": str(metadata_destination.resolve()),
        "metrics": list(selected),
        "runs": [
            {
                "label": entry.get("label", run_dir.name),
                "run_dir": str(run_dir),
                "seed": entry.get("seed"),
                "requested_device": entry.get("requested_device"),
                "resolved_device": entry.get("resolved_device"),
            }
            for entry, run_dir in entries
        ],
    }


def _render_local(
    entries: Sequence[tuple[Mapping[str, Any], Path]],
    destination: Path,
    *,
    selected: Sequence[str],
    metadata_destination: Path,
    manifest_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(selected),
        1,
        figsize=(8.5, max(4.2, 3.2 * len(selected))),
        dpi=160,
        squeeze=False,
    )
    axes_list = [axis for row in axes for axis in row]
    plotted_labels: list[str] = []
    rows_by_run = [(entry, run_dir, read_metrics(run_dir)) for entry, run_dir in entries]
    for axis, metric in zip(axes_list, selected):
        field, title, ylabel = METRIC_SPECS[metric]
        plotted = False
        for entry, run_dir, rows in rows_by_run:
            x_values, y_values = _paired_series(rows, field)
            label = str(entry.get("label", run_dir.name))
            seed = entry.get("seed")
            resolved_device = entry.get("resolved_device") or entry.get(
                "requested_device", "device?"
            )
            label = f"{label} (seed {seed}, {resolved_device})"
            if y_values:
                axis.plot(
                    x_values,
                    y_values,
                    linewidth=1.2,
                    marker="o" if metric == "return" else None,
                    markersize=2.5,
                    alpha=0.9,
                    label=label,
                )
                plotted = True
                if label not in plotted_labels:
                    plotted_labels.append(label)
        axis.set_title(title)
        axis.set_xlabel("Environment step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        if not plotted:
            axis.text(
                0.5,
                0.5,
                "No finite samples in the selected run artifacts",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    if plotted_labels:
        axes_list[0].legend(loc="best", fontsize=8)
    figure.suptitle("Day 14 controlled experiment comparison", y=0.995)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="png")
    plt.close(figure)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_object(
        metadata_destination,
        _metadata_payload(
            manifest_path,
            destination,
            metadata_destination,
            selected,
            entries,
        ),
    )


def _render_worker(
    serialized_entries: Sequence[tuple[dict[str, Any], str]],
    destination: str,
    selected: Sequence[str],
    metadata_destination: str,
    manifest_path: str,
    result_queue: Any,
) -> None:
    try:
        entries = [(entry, Path(run_dir)) for entry, run_dir in serialized_entries]
        _render_local(
            entries,
            Path(destination),
            selected=selected,
            metadata_destination=Path(metadata_destination),
            manifest_path=manifest_path,
        )
        result_queue.put({"ok": True})
    except Exception as error:  # pragma: no cover - exercised in child process
        result_queue.put({"ok": False, "error": f"{type(error).__name__}: {error}"})


def render_comparison(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    metrics: Sequence[str] = ("return",),
    metadata_path: str | Path | None = None,
) -> Path:
    """Render one or more aligned metrics; no values are entered by hand."""

    selected = list(metrics)
    if not selected or any(metric not in METRIC_SPECS for metric in selected):
        raise ValueError(f"metrics must be chosen from {', '.join(METRIC_SPECS)}")
    entries = load_manifest_run_paths(manifest_path)

    destination = Path(output_path)
    metadata_destination = (
        Path(metadata_path)
        if metadata_path is not None
        else destination.with_suffix(".json")
    )
    if "torch" not in sys.modules:
        _render_local(
            entries,
            destination,
            selected=selected,
            metadata_destination=metadata_destination,
            manifest_path=manifest_path,
        )
        return destination

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_render_worker,
        args=(
            [(dict(entry), str(run_dir)) for entry, run_dir in entries],
            str(destination),
            selected,
            str(metadata_destination),
            str(Path(manifest_path).resolve()),
            result_queue,
        ),
    )
    process.start()
    process.join()
    try:
        result = result_queue.get(timeout=5)
    except queue.Empty:
        result = None
    result_queue.close()
    if process.exitcode != 0 or not isinstance(result, dict) or not result.get("ok"):
        error = result.get("error") if isinstance(result, dict) else "unknown plotting error"
        raise RuntimeError(f"plot worker failed: {error}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot real Day 14 experiment metrics aligned by environment step."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day14/experiment-return-comparison.png"),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRIC_SPECS),
        default=["return"],
    )
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = render_comparison(
            args.manifest,
            args.output,
            metrics=args.metrics,
            metadata_path=args.metadata_output,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"Unable to render experiment comparison: {error}")
        return 2
    print(f"Wrote comparison figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
