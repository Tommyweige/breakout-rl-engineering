"""Summarize a DQN run and render evidence-bearing diagnostic curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from breakout_rl.training.diagnostics import (
    aggregate_training_metrics,
    parse_numeric_value,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _paired_series(
    rows: Sequence[Mapping[str, Any]],
    x_field: str,
    y_field: str,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        x_value = parse_numeric_value(row.get(x_field))
        y_value = parse_numeric_value(row.get(y_field))
        if x_value is None or y_value is None:
            continue
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        x_values.append(x_value)
        y_values.append(y_value)
    return x_values, y_values


def _plot_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[tuple[str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int) -> ImageFont.ImageFont:
        for candidate in (
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ):
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    canvas = Image.new("RGB", (1200, 675), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = font(24)
    label_font = font(15)
    small_font = font(12)
    plot_left, plot_top, plot_right, plot_bottom = 95, 72, 1160, 585
    series: list[tuple[str, list[float], list[float]]] = []
    for field, label in fields:
        x_values, y_values = _paired_series(rows, "global_step", field)
        if y_values:
            series.append((label, x_values, y_values))

    draw.text((plot_left, 22), title, font=title_font, fill=(17, 24, 39))
    draw.text((plot_left, plot_bottom + 38), "Environment step", font=label_font, fill=(55, 65, 81))
    draw.text((18, plot_top), ylabel, font=label_font, fill=(55, 65, 81))
    if not series:
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(156, 163, 175), width=2)
        message = "No finite samples in metrics.csv"
        bbox = draw.textbbox((0, 0), message, font=label_font)
        draw.text(
            ((plot_left + plot_right - bbox[2]) // 2, (plot_top + plot_bottom - bbox[3]) // 2),
            message,
            font=label_font,
            fill=(75, 85, 99),
        )
        canvas.save(output_path, format="PNG", optimize=True)
        return

    all_x = [value for _label, x_values, _y_values in series for value in x_values]
    all_y = [value for _label, _x_values, y_values in series for value in y_values]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    y_padding = max((y_max - y_min) * 0.08, 1e-6)
    y_min -= y_padding
    y_max += y_padding
    if y_min <= 0.0 <= y_max:
        zero_y = plot_bottom - (0.0 - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
    else:
        zero_y = None
    for index in range(6):
        fraction = index / 5
        y = int(plot_top + fraction * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill=(229, 231, 235), width=1)
        tick = y_max - fraction * (y_max - y_min)
        draw.text((plot_left - 75, y - 8), f"{tick:.2g}", font=small_font, fill=(107, 114, 128))
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(107, 114, 128), width=2)
    if zero_y is not None:
        draw.line((plot_left, int(zero_y), plot_right, int(zero_y)), fill=(156, 163, 175), width=1)
    colors = ((37, 99, 235), (220, 38, 38), (22, 163, 74), (124, 58, 237), (234, 88, 12))
    for series_index, (label, x_values, y_values) in enumerate(series):
        points = []
        for x_value, y_value in zip(x_values, y_values):
            x = plot_left + (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left)
            y = plot_bottom - (y_value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            points.append((int(x), int(y)))
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[series_index % len(colors)])
        else:
            draw.line(points, fill=colors[series_index % len(colors)], width=2, joint="curve")
        legend_x = plot_left + series_index * 180
        draw.rectangle((legend_x, plot_top - 35, legend_x + 14, plot_top - 21), fill=colors[series_index % len(colors)])
        draw.text((legend_x + 20, plot_top - 38), label, font=small_font, fill=(55, 65, 81))
    draw.text((plot_left - 5, plot_bottom + 8), f"{x_min:.0f}", font=small_font, fill=(107, 114, 128))
    draw.text((plot_right - 35, plot_bottom + 8), f"{x_max:.0f}", font=small_font, fill=(107, 114, 128))
    canvas.save(output_path, format="PNG", optimize=True)


def _write_plots_local(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Render the required Day 13 plots from metrics rows."""

    materialized = [dict(row) for row in rows]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plot_specs = {
        "return_curve": (
            (("raw_episode_return", "Raw episode return"),),
            "Episode return over training",
            "Raw episode return",
            "return-curve.png",
        ),
        "loss_curve": (
            (("loss", "Huber loss"),),
            "Training loss over environment steps",
            "Loss",
            "loss-curve.png",
        ),
        "q_values": (
            (
                ("q_mean", "Q mean"),
                ("q_max", "Q max"),
                ("q_min", "Q min"),
                ("target_mean", "Target mean"),
                ("target_max", "Target max"),
            ),
            "Q-values and Bellman targets",
            "Value",
            "q-values.png",
        ),
        "gradient_norm": (
            (("gradient_norm", "Gradient norm"),),
            "Gradient norm before clipping",
            "L2 norm",
            "gradient-norm.png",
        ),
        "epsilon_curve": (
            (("epsilon", "Epsilon"),),
            "Exploration probability over training",
            "Epsilon",
            "epsilon-curve.png",
        ),
    }
    paths: dict[str, str] = {}
    for key, (fields, title, ylabel, filename) in plot_specs.items():
        output_path = destination / filename
        _plot_metric(
            materialized,
            fields=fields,
            title=title,
            ylabel=ylabel,
            output_path=output_path,
        )
        paths[key] = str(output_path)
    return paths


def write_plots(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Render evidence plots without importing a second numerical runtime."""

    materialized = [dict(row) for row in rows]
    return _write_plots_local(materialized, Path(output_dir))


def analyze_run(
    run_dir: str | Path,
    *,
    plots_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read a run's three durable artifacts, summarize them, and plot them."""

    path = Path(run_dir)
    config = _read_json(path / "config.json")
    rows = _read_metrics(path / "metrics.csv")
    summary = _read_json(path / "summary.json")
    report = aggregate_training_metrics(rows)
    if summary.get("status") == "failed_non_finite":
        # The trainer checkpoints and summarizes before the failed row can be
        # appended, so the CSV alone may not contain the offending value.
        report["non_finite_count"] = max(int(report["non_finite_count"]), 1)
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    report.update(
        {
            "run_id": str(config.get("run_id", path.name)),
            "resolved_device": runtime.get("resolved_device", "unavailable"),
            "cuda_device_index": runtime.get("cuda_device_index"),
            "metadata": runtime,
            "run_summary": summary,
            "plots": write_plots(rows, plots_dir or path / "plots"),
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize and plot a DQN training run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="where to write PNGs (default: <run_dir>/plots)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_run(args.run_dir, plots_dir=args.plots_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
