"""Create an evidence figure from a real Breakout observation and CNN forward."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from inspect_cnn_dimensions import CnnInspection, collect_cnn_inspection

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_OUTPUT = Path("artifacts/day07/cnn-dimensions.png")


def parse_args() -> argparse.Namespace:
    """Parse options for the reproducible Day 7 figure."""

    parser = argparse.ArgumentParser(
        description="Plot runtime-derived Atari CNN tensor dimensions."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="forward device; auto selects CUDA when available (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used for the Breakout reset and model initialization (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PNG output path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def _shape_elements(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64))


def _write_metadata(
    output: Path,
    *,
    inspection: CnnInspection,
    seed: int,
    command: str,
) -> Path:
    """Write machine-readable provenance beside the PNG."""

    metadata_path = output.with_suffix(".json")
    payload = {
        "command": command,
        "seed": seed,
        "device": str(inspection.device),
        "observation": {
            "shape": list(inspection.observation.shape),
            "dtype": str(inspection.observation.dtype),
            "min": int(inspection.observation.min()),
            "max": int(inspection.observation.max()),
        },
        "model_input": {
            "shape": list(inspection.model_input.shape),
            "dtype": str(inspection.model_input.dtype),
            "min": float(inspection.model_input.min()),
            "max": float(inspection.model_input.max()),
        },
        "runtime_shapes": {
            name: list(shape) for name, shape in inspection.shapes.items()
        },
        "feature_dim": int(inspection.features.shape[-1]),
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable local font while keeping the script portable."""

    for font_path in (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _activation_image(activation: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Convert one real activation channel into a readable heatmap."""

    minimum = float(activation.min())
    maximum = float(activation.max())
    if maximum == minimum:
        normalized = np.zeros_like(activation, dtype=np.uint8)
    else:
        normalized = np.round(
            (activation - minimum) / (maximum - minimum) * 255
        ).astype(np.uint8)

    grayscale = Image.fromarray(normalized, mode="L")
    grayscale = grayscale.resize(size, Image.Resampling.NEAREST)
    return ImageOps.colorize(
        grayscale,
        black=(35, 35, 90),
        white=(240, 210, 60),
    )


def create_figure(
    inspection: CnnInspection,
    *,
    seed: int,
    output: Path,
    command: str,
) -> Path:
    """Render runtime observation, activation, and shape evidence with Pillow."""

    output.parent.mkdir(parents=True, exist_ok=True)

    shape_labels = (
        "Environment state",
        "Model input",
        "Conv1",
        "Conv2",
        "Conv3",
        "Flatten",
    )
    shapes = [
        tuple(inspection.observation.shape),
        inspection.shapes["input"],
        inspection.shapes["conv1"],
        inspection.shapes["conv2"],
        inspection.shapes["conv3"],
        inspection.shapes["flatten"],
    ]
    element_counts = [_shape_elements(shape) for shape in shapes]

    canvas = Image.new("RGB", (1500, 980), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(26)
    section_font = _font(18)
    body_font = _font(15)
    small_font = _font(13)

    draw.text(
        (55, 28),
        "Day 7 — Atari CNN tensor dimensions from one real forward pass",
        fill=(25, 25, 25),
        font=title_font,
    )

    frame_box = (70, 95, 390, 415)
    frame = Image.fromarray(inspection.observation[0], mode="L").resize(
        (320, 320),
        Image.Resampling.NEAREST,
    )
    canvas.paste(frame.convert("RGB"), (frame_box[0], frame_box[1]))
    draw.rectangle(frame_box, outline=(30, 30, 30), width=2)
    draw.text(
        (70, 425),
        "Real Breakout frame 1 of state=(4, 84, 84) | dtype=uint8",
        fill=(25, 25, 25),
        font=body_font,
    )

    conv3_activation = (
        inspection.activations["conv3"][0, 0].detach().cpu().numpy()
    )
    activation_box = (470, 95, 790, 415)
    activation = _activation_image(conv3_activation, (320, 320))
    canvas.paste(activation, (activation_box[0], activation_box[1]))
    draw.rectangle(activation_box, outline=(30, 30, 30), width=2)
    draw.text(
        (470, 425),
        "Real Conv3 channel 0 | shape=(1, 64, 7, 7)",
        fill=(25, 25, 25),
        font=body_font,
    )
    draw.text(
        (850, 120),
        "Question this figure answers",
        fill=(25, 25, 25),
        font=section_font,
    )
    draw.text(
        (850, 155),
        "How does one real (4, 84, 84) state become\n"
        "a 3,136-value feature vector?",
        fill=(45, 45, 45),
        font=body_font,
        spacing=8,
    )
    draw.text(
        (850, 255),
        "The heatmap is an untrained model output.\n"
        "It proves the forward path ran; it does not\n"
        "prove the random CNN has learned useful features.",
        fill=(80, 80, 80),
        font=body_font,
        spacing=8,
    )

    chart_left = 70
    chart_right = 1430
    chart_top = 500
    bar_left = 280
    bar_right = 930
    draw.text(
        (chart_left, chart_top),
        "Runtime shape evidence (bar length uses a log scale for element count)",
        fill=(25, 25, 25),
        font=section_font,
    )
    min_log = min(math.log10(count) for count in element_counts)
    max_log = max(math.log10(count) for count in element_counts)
    bar_colors = (
        (120, 120, 120),
        (76, 120, 168),
        (245, 133, 24),
        (84, 162, 75),
        (228, 87, 86),
        (114, 183, 178),
    )

    for index, (label, shape, count, color) in enumerate(
        zip(shape_labels, shapes, element_counts, bar_colors)
    ):
        y = chart_top + 55 + index * 48
        fraction = (math.log10(count) - min_log) / (max_log - min_log)
        width = max(4, int((bar_right - bar_left) * fraction))
        draw.text((chart_left, y + 3), label, fill=(35, 35, 35), font=body_font)
        draw.rectangle(
            (bar_left, y, bar_left + width, y + 28),
            fill=color,
        )
        draw.text(
            (bar_right + 20, y + 3),
            f"shape={shape} | {count:,} elements",
            fill=(35, 35, 35),
            font=body_font,
        )

    axis_y = chart_top + 55 + len(shape_labels) * 48 + 22
    draw.line((bar_left, axis_y, bar_right, axis_y), fill=(80, 80, 80), width=1)
    for tick in (10**3, 10**4, 10**5):
        if min(element_counts) <= tick <= max(element_counts):
            fraction = (math.log10(tick) - min_log) / (max_log - min_log)
            x = bar_left + int((bar_right - bar_left) * fraction)
            draw.line((x, axis_y - 5, x, axis_y + 5), fill=(80, 80, 80), width=1)
            draw.text((x - 18, axis_y + 8), f"{tick:,}", fill=(70, 70, 70), font=small_font)
    draw.text(
        (chart_left, axis_y + 35),
        f"seed={seed} | device={inspection.device} | source: make_breakout_env() + AtariFeatureExtractor",
        fill=(80, 80, 80),
        font=small_font,
    )

    canvas.save(output, format="PNG", optimize=True)
    _write_metadata(output, inspection=inspection, seed=seed, command=command)
    return output


def main() -> None:
    """Generate the Day 7 evidence figure."""

    args = parse_args()
    try:
        inspection = collect_cnn_inspection(
            seed=args.seed,
            device_name=args.device,
        )
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error

    output = create_figure(
        inspection,
        seed=args.seed,
        output=args.output,
        command=(
            "python visualize_cnn_dimensions.py "
            f"--device {args.device} --seed {args.seed}"
        ),
    )
    print(f"Saved figure: {output}")
    print(f"Saved metadata: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
