"""Render Day 14 comparison figures directly from a manifest and run CSVs."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.experiments import (
    load_manifest_run_paths,
    read_metrics,
    relative_path,
    write_json_object,
)
from breakout_rl.training.diagnostics import parse_numeric_value


METRIC_SPECS: dict[str, tuple[str, str, str]] = {
    "return": ("raw_episode_return", "Episode return", "Raw episode return"),
    "loss": ("loss", "Loss", "Huber loss"),
    "q": ("q_mean", "Q-value mean", "Q mean"),
    "target": ("target_mean", "Target mean", "Target mean"),
    "gradient": ("gradient_norm", "Gradient norm", "Gradient norm"),
    "epsilon": ("epsilon", "Exploration", "Epsilon"),
    "sps": ("sps", "Throughput", "Steps per second"),
}


def _paired_series(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        x_value = parse_numeric_value(row.get("global_step"))
        y_value = parse_numeric_value(row.get(field))
        if (
            x_value is None
            or y_value is None
            or not math.isfinite(x_value)
            or not math.isfinite(y_value)
        ):
            continue
        x_values.append(x_value)
        y_values.append(y_value)
    return x_values, y_values


def _rolling_return_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    window: int = 20,
) -> tuple[list[float], list[float]]:
    episodes = [
        (step, value)
        for step, value in zip(
            *_paired_series(rows, "raw_episode_return")
        )
    ]
    if len(episodes) < window:
        return [], []
    x_values = [step for step, _ in episodes[window - 1 :]]
    y_values = [
        sum(value for _, value in episodes[index - window + 1 : index + 1]) / window
        for index in range(window - 1, len(episodes))
    ]
    return x_values, y_values


def _metadata_payload(
    manifest_path: str | Path,
    destination: Path,
    metadata_destination: Path,
    selected: Sequence[str],
    entries: Sequence[tuple[Mapping[str, Any], Path]],
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "command": list(sys.argv),
        "manifest": relative_path(manifest_source, start=Path.cwd()),
        "output": relative_path(destination, start=Path.cwd()),
        "metadata_output": relative_path(metadata_destination, start=Path.cwd()),
        "metrics": list(selected),
        "runs": [
            {
                "label": entry.get("label", run_dir.name),
                "run_dir": relative_path(run_dir, start=manifest_source.parent),
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

    title_font = font(22)
    panel_font = font(18)
    label_font = font(13)
    small_font = font(11)
    panel_height = 275
    canvas = Image.new(
        "RGB",
        (1280, 150 + panel_height * len(selected)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 24), "Day 14 controlled experiment comparison", font=title_font, fill=(17, 24, 39))
    rows_by_run = [(entry, run_dir, read_metrics(run_dir)) for entry, run_dir in entries]
    colors = ((37, 99, 235), (220, 38, 38), (22, 163, 74), (124, 58, 237), (234, 88, 12))
    for metric_index, metric in enumerate(selected):
        field, title, ylabel = METRIC_SPECS[metric]
        panel_left, panel_top = 35, 95 + metric_index * panel_height
        panel_right, panel_bottom = 1245, panel_top + panel_height - 25
        draw.rounded_rectangle(
            (panel_left, panel_top, panel_right, panel_bottom),
            radius=10,
            fill=(250, 250, 249),
            outline=(203, 213, 225),
            width=2,
        )
        draw.text((panel_left + 18, panel_top + 14), f"{title} — {ylabel}", font=panel_font, fill=(17, 24, 39))
        plot_left, plot_top = panel_left + 85, panel_top + 58
        plot_right, plot_bottom = panel_right - 22, panel_bottom - 45
        series: list[tuple[str, list[float], list[float], tuple[int, int, int]]] = []
        for entry, run_dir, rows in rows_by_run:
            x_values, y_values = _paired_series(rows, field)
            label = str(entry.get("label", run_dir.name))
            seed = entry.get("seed")
            resolved_device = entry.get("resolved_device") or entry.get(
                "requested_device", "device?"
            )
            label = f"{label} (seed {seed}, {resolved_device})"
            if y_values and metric == "return":
                color = colors[len(series) % len(colors)]
                series.append((f"{label} raw", x_values, y_values, tuple(min(255, value + 100) for value in color)))
                rolling_x, rolling_y = _rolling_return_series(rows)
                if rolling_y:
                    series.append((f"{label} rolling20", rolling_x, rolling_y, color))
            elif y_values:
                series.append((label, x_values, y_values, colors[len(series) % len(colors)]))

        draw.text((plot_left, panel_bottom + 8), "Environment step", font=label_font, fill=(55, 65, 81))
        draw.text((panel_left + 10, plot_top), ylabel, font=label_font, fill=(55, 65, 81))
        if not series:
            draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(156, 163, 175), width=2)
            draw.text((plot_left + 240, plot_top + 70), "No finite samples in the selected run artifacts", font=label_font, fill=(75, 85, 99))
            continue
        all_x = [value for _label, x_values, _y_values, _color in series for value in x_values]
        all_y = [value for _label, _x_values, y_values, _color in series for value in y_values]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        y_padding = max((y_max - y_min) * 0.08, 1e-6)
        y_min -= y_padding
        y_max += y_padding
        for tick_index in range(6):
            fraction = tick_index / 5
            y = int(plot_top + fraction * (plot_bottom - plot_top))
            draw.line((plot_left, y, plot_right, y), fill=(229, 231, 235), width=1)
            tick = y_max - fraction * (y_max - y_min)
            draw.text((plot_left - 62, y - 7), f"{tick:.2g}", font=small_font, fill=(107, 114, 128))
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(107, 114, 128), width=2)
        for series_index, (label, x_values, y_values, color) in enumerate(series):
            points = [
                (
                    int(plot_left + (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left)),
                    int(plot_bottom - (y_value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)),
                )
                for x_value, y_value in zip(x_values, y_values)
            ]
            if len(points) == 1:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            else:
                draw.line(points, fill=color, width=2, joint="curve")
            legend_x = plot_left + (series_index % 4) * 270
            legend_y = panel_top + 18 + (series_index // 4) * 18
            draw.rectangle((legend_x, legend_y + 3, legend_x + 12, legend_y + 15), fill=color)
            draw.text((legend_x + 17, legend_y), label[:38], font=small_font, fill=(55, 65, 81))
        draw.text((plot_left - 4, plot_bottom + 6), f"{x_min:.0f}", font=small_font, fill=(107, 114, 128))
        draw.text((plot_right - 32, plot_bottom + 6), f"{x_max:.0f}", font=small_font, fill=(107, 114, 128))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)
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
    _render_local(
        entries,
        destination,
        selected=selected,
        metadata_destination=metadata_destination,
        manifest_path=manifest_path,
    )
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
