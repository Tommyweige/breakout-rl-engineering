"""Visualize raw Q-values from one real untrained DQN forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from inspect_dqn_network import DQNInspection, collect_dqn_inspection


DEFAULT_OUTPUT = Path("assets/day08/dqn-q-values.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize real runtime Q-values from an untrained DQN."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _write_metadata(
    output: Path,
    *,
    inspection: DQNInspection,
    seed: int,
    command: str,
) -> Path:
    metadata_path = output.with_suffix(".json")
    q_values = inspection.q_values[0].detach().cpu().tolist()
    payload = {
        "command": command,
        "seed": seed,
        "device": str(inspection.device),
        "observation_shape": list(inspection.observation.shape),
        "model_input_shape": list(inspection.model_input.shape),
        "feature_shape": list(inspection.features.shape),
        "output_shape": list(inspection.q_values.shape),
        "action_meanings": list(inspection.action_meanings),
        "q_values": q_values,
        "greedy_action_index": inspection.greedy_action_index,
        "greedy_action_meaning": inspection.action_meanings[
            inspection.greedy_action_index
        ],
        "parameter_count": inspection.parameter_count,
        "state_dict_roundtrip_max_abs_diff": (
            inspection.state_dict_roundtrip_max_abs_diff
        ),
        "trained": False,
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def create_figure(
    inspection: DQNInspection,
    *,
    seed: int,
    output: Path,
    command: str,
) -> Path:
    """Draw the real input frame, feature/Q shapes, and raw Q-values."""

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1450, 880), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    section_font = _font(20)
    body_font = _font(16)
    small_font = _font(13)

    draw.text(
        (55, 30),
        "Day 8 — One real Breakout state → four raw DQN Q-values",
        fill=(25, 25, 25),
        font=title_font,
    )

    frame = Image.fromarray(inspection.observation[-1], mode="L").resize(
        (330, 330), Image.Resampling.NEAREST
    )
    canvas.paste(frame.convert("RGB"), (60, 105))
    draw.rectangle((60, 105, 390, 435), outline=(35, 35, 35), width=2)
    draw.text(
        (60, 450),
        "Latest real frame from state=(4, 84, 84)",
        fill=(35, 35, 35),
        font=body_font,
    )

    draw.text((455, 110), "Forward path", fill=(25, 25, 25), font=section_font)
    path_lines = (
        f"model input   {tuple(inspection.model_input.shape)}",
        "       ↓",
        f"CNN features  {tuple(inspection.features.shape)}",
        "       ↓",
        "Linear 3136 → 512 + ReLU",
        "       ↓",
        f"raw Q-values  {tuple(inspection.q_values.shape)}",
    )
    y = 155
    for line in path_lines:
        draw.text((455, y), line, fill=(45, 45, 45), font=body_font)
        y += 42

    q_values = inspection.q_values[0].detach().cpu().tolist()
    max_abs = max(max(abs(value) for value in q_values), 1e-6)
    chart_left = 65
    zero_x = 580
    chart_top = 555
    half_width = 390

    draw.text(
        (chart_left, chart_top - 45),
        "Raw Q-values (not probabilities; no softmax)",
        fill=(25, 25, 25),
        font=section_font,
    )
    draw.line((zero_x, chart_top, zero_x, chart_top + 215), fill=(90, 90, 90), width=2)

    for index, (meaning, value) in enumerate(zip(inspection.action_meanings, q_values)):
        y = chart_top + index * 52
        width = int(abs(value) / max_abs * half_width)
        if value >= 0:
            x0, x1 = zero_x, zero_x + width
        else:
            x0, x1 = zero_x - width, zero_x
        draw.rectangle((x0, y, x1, y + 30), fill=(100, 135, 165))
        marker = "  ← argmax" if index == inspection.greedy_action_index else ""
        draw.text(
            (70, y + 3),
            f"{index}: {meaning:<6} {value:+.6f}{marker}",
            fill=(35, 35, 35),
            font=body_font,
        )

    draw.text((1030, 555), "Important", fill=(25, 25, 25), font=section_font)
    draw.text(
        (1030, 595),
        "These values come from random\n"
        "initial weights. argmax only tells\n"
        "which random output is largest;\n"
        "it does not mean the agent has\n"
        "learned to choose that action.",
        fill=(60, 60, 60),
        font=body_font,
        spacing=7,
    )

    draw.text(
        (55, 825),
        f"seed={seed} | device={inspection.device} | parameters={inspection.parameter_count:,} | state_dict max diff={inspection.state_dict_roundtrip_max_abs_diff:.8f}",
        fill=(85, 85, 85),
        font=small_font,
    )

    canvas.save(output, format="PNG", optimize=True)
    _write_metadata(output, inspection=inspection, seed=seed, command=command)
    return output


def main() -> None:
    args = parse_args()
    try:
        inspection = collect_dqn_inspection(seed=args.seed, device_name=args.device)
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error

    command = (
        "python visualize_dqn_network.py "
        f"--device {args.device} --seed {args.seed}"
    )
    output = create_figure(
        inspection,
        seed=args.seed,
        output=args.output,
        command=command,
    )
    print(f"Saved figure: {output}")
    print(f"Saved metadata: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
