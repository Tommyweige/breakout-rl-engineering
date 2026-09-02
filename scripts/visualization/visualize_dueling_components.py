"""Plot real Value/Advantage/Q outputs for one Contract v2 Breakout state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from breakout_rl.analysis.dueling import (
    DuelingInspection,
    collect_dueling_inspection,
    inspection_payload,
)


DEFAULT_OUTPUT = Path("assets/day19/dueling-value-advantage-q.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="metadata destination; defaults to the PNG path with a .json suffix",
    )
    return parser


def _values(inspection: DuelingInspection) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    value = float(inspection.value[0, 0].detach().cpu().item())
    raw_advantage = inspection.advantage[0].detach().cpu().numpy().astype(float)
    centered_advantage = (
        inspection.centered_advantage[0].detach().cpu().numpy().astype(float)
    )
    q_values = inspection.q_values[0].detach().cpu().numpy().astype(float)
    return value, raw_advantage, centered_advantage, q_values


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    content: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (31, 41, 55),
) -> None:
    draw.text(position, content, font=font, fill=fill)


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    *,
    title_font: ImageFont.ImageFont,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill=(250, 250, 249), outline=(203, 213, 225), width=2)
    _text(draw, (left + 22, top + 16), title, font=title_font, fill=(17, 24, 39))
    return left + 24, top + 58, right - 24, bottom - 24


def _vertical_bars(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    labels: list[str],
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    *,
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = plot
    all_values = np.concatenate([values for _name, values, _color in series])
    scale = max(float(np.max(np.abs(all_values))), 1e-6) * 1.2
    zero_y = (top + bottom) // 2
    half_height = (bottom - top) // 2
    draw.line((left, zero_y, right, zero_y), fill=(75, 85, 99), width=2)
    count = len(labels)
    group_width = (right - left) / max(count, 1)
    bar_width = max(8, int(group_width / (len(series) + 1.5)))
    for series_index, (name, values, color) in enumerate(series):
        for index, value in enumerate(values):
            center_x = left + (index + 0.5) * group_width
            offset = (series_index - (len(series) - 1) / 2) * bar_width
            x0 = int(center_x + offset - bar_width / 2)
            x1 = int(center_x + offset + bar_width / 2)
            height = int(abs(float(value)) / scale * half_height)
            if value >= 0:
                y0, y1 = zero_y - height, zero_y
            else:
                y0, y1 = zero_y, zero_y + height
            draw.rectangle((x0, y0, x1, y1), fill=color)
            value_y = y0 - 22 if value >= 0 else y1 + 3
            _text(draw, (x0 - 8, value_y), f"{value:+.3f}", font=small_font, fill=color)
    for index, label in enumerate(labels):
        center_x = int(left + (index + 0.5) * group_width)
        bbox = draw.textbbox((0, 0), label, font=body_font)
        _text(draw, (center_x - (bbox[2] - bbox[0]) // 2, bottom + 7), label, font=body_font)
    legend_x = left
    legend_y = top - 28
    for name, _values_array, color in series:
        draw.rectangle((legend_x, legend_y + 4, legend_x + 13, legend_y + 17), fill=color)
        _text(draw, (legend_x + 18, legend_y), name, font=small_font)
        legend_x += 165 if len(series) > 1 else 130


def _single_value_bar(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    value: float,
    *,
    body_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = plot
    center_x = (left + right) // 2
    scale = max(abs(value), 1e-6) * 1.2
    draw.line((center_x, top, center_x, bottom), fill=(75, 85, 99), width=2)
    width = int(abs(value) / scale * ((right - left) // 2))
    if value >= 0:
        x0, x1 = center_x, center_x + width
    else:
        x0, x1 = center_x - width, center_x
    draw.rounded_rectangle((x0, top + 55, x1, bottom - 35), radius=8, fill=(124, 58, 237))
    _text(draw, (left, bottom + 4), "negative", font=body_font, fill=(75, 85, 99))
    _text(draw, (right - 72, bottom + 4), "positive", font=body_font, fill=(75, 85, 99))
    _text(draw, (center_x - 42, top + 15), f"{value:+.5f}", font=body_font, fill=(91, 33, 182))


def create_figure(
    inspection: DuelingInspection,
    *,
    seed: int,
    output: str | Path,
    metadata_path: str | Path | None = None,
    command: str,
) -> tuple[Path, Path]:
    """Render actual model components and persist their source metadata."""

    destination = Path(output)
    metadata_destination = (
        Path(metadata_path)
        if metadata_path is not None
        else destination.with_suffix(".json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)

    value, raw_advantage, centered_advantage, q_values = _values(inspection)
    labels = list(inspection.action_meanings)
    canvas = Image.new("RGB", (1500, 930), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30)
    section_font = _font(21)
    body_font = _font(15)
    small_font = _font(12)
    _text(
        draw,
        (42, 25),
        "Day 19 — 同一個 state 中，V(s)、A(s,a) 與 Q(s,a) 如何重新組合",
        font=title_font,
        fill=(17, 24, 39),
    )
    _text(
        draw,
        (44, 70),
        "真實 Contract v2 observation + DuelingDQNNetwork forward output（不是 mock / 裝飾數字）",
        font=body_font,
        fill=(75, 85, 99),
    )

    frame_box = (38, 112, 690, 455)
    value_box = (762, 112, 1462, 455)
    advantage_box = (38, 490, 870, 895)
    q_box = (910, 490, 1462, 895)
    frame_plot = _panel(draw, frame_box, "同一個 state 的真實最新 frame", title_font=section_font)
    frame = Image.fromarray(inspection.observation[-1], mode="L").convert("RGB")
    frame_size = min(
        frame_plot[2] - frame_plot[0],
        frame_plot[3] - frame_plot[1] - 30,
        300,
    )
    frame = frame.resize((frame_size, frame_size), Image.Resampling.NEAREST)
    frame_x = frame_plot[0] + (frame_plot[2] - frame_plot[0] - frame.width) // 2
    frame_y = frame_plot[1] + 8
    canvas.paste(frame, (frame_x, frame_y))
    _text(draw, (frame_plot[0], frame_plot[3] - 18), "shape=(4,84,84)；圖中顯示最後一幀", font=body_font, fill=(75, 85, 99))

    value_plot = _panel(draw, value_box, f"State Value：V(s)", title_font=section_font)
    _single_value_bar(draw, value_plot, value, body_font=body_font)

    advantage_plot = _panel(draw, advantage_box, "Action Advantage：raw 與 mean-centered", title_font=section_font)
    _vertical_bars(
        draw,
        (advantage_plot[0], advantage_plot[1] + 12, advantage_plot[2], advantage_plot[3] - 25),
        labels,
        [
            ("raw A(s,a)", raw_advantage, (245, 158, 11)),
            ("A − mean(A)", centered_advantage, (14, 165, 233)),
        ],
        body_font=body_font,
        small_font=small_font,
    )

    q_plot = _panel(draw, q_box, "Final Q(s,a) = V(s) + centered A(s,a)", title_font=section_font)
    _vertical_bars(
        draw,
        (q_plot[0], q_plot[1] + 12, q_plot[2], q_plot[3] - 55),
        labels,
        [("Q(s,a)", q_values, (22, 163, 74))],
        body_font=body_font,
        small_font=small_font,
    )
    _text(
        draw,
        (q_plot[0], q_plot[3] - 34),
        f"reconstruction max abs error={inspection.reconstruction_max_abs_error:.2e}  "
        f"argmax={labels[inspection.greedy_action_index]}",
        font=small_font,
        fill=(22, 101, 52),
    )
    canvas.save(destination, format="PNG", optimize=True)

    payload = inspection_payload(inspection, seed=seed)
    payload.update(
        {
            "generation_command": command,
            "source_script": "scripts/visualization/visualize_dueling_components.py",
            "figure_question": (
                "How do the real Value and centered Advantage outputs combine into Q-values?"
            ),
            "figure_interpretation": (
                "The four bars use one seeded real observation and the same Dueling forward pass; "
                "they verify the representation equation, not policy quality."
            ),
        }
    )
    metadata_destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination, metadata_destination


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inspection = collect_dueling_inspection(
        seed=args.seed,
        device_name=args.device,
        contract_path=args.contract,
        checkpoint=args.checkpoint,
    )
    command = (
        "python -m scripts.visualization.visualize_dueling_components "
        f"--device {args.device} --seed {args.seed} "
        f"--contract {args.contract.as_posix()}"
    )
    if args.checkpoint is not None:
        command += f" --checkpoint {args.checkpoint.as_posix()}"
    output, metadata = create_figure(
        inspection,
        seed=args.seed,
        output=args.output,
        metadata_path=args.metadata,
        command=command,
    )
    print(f"Saved figure: {output}")
    print(f"Saved metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
