"""Plot the real Day 30 Browser score comparison from captured JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("assets/day30/final-browser-score-comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("assets/day30/final-browser-score-distribution.png"))
    parser.add_argument("--metadata", type=Path, default=Path("assets/day30/final-browser-score-distribution.json"))
    return parser.parse_args()


def read_comparison(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifactType") != "day30_browser_policy_score_comparison":
        raise ValueError(f"{path}: unexpected artifactType")
    for backend in ("wasm", "webgpu"):
        episodes = payload.get("results", {}).get(backend, {}).get("episodes")
        if not isinstance(episodes, list) or len(episodes) < 30:
            raise ValueError(f"{path}: {backend} must contain at least 30 episodes")
        if any(not isinstance(episode.get("episodeReturn"), (int, float)) for episode in episodes):
            raise ValueError(f"{path}: {backend} contains a non-numeric episode return")
        unique_seed_count = len({episode.get("seed") for episode in episodes})
        required_unique_seed_count = 50 if backend == "webgpu" else 30
        if unique_seed_count < required_unique_seed_count:
            raise ValueError(f"{path}: {backend} must contain at least {required_unique_seed_count} unique seeds; got {unique_seed_count}")
    return payload


def main() -> None:
    args = parse_args()
    payload = read_comparison(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    scores = {backend: [float(episode["episodeReturn"]) for episode in payload["results"][backend]["episodes"]] for backend in ("wasm", "webgpu")}

    image = Image.new("RGB", (1400, 820), "#f5f1e8")
    draw = ImageDraw.Draw(image)
    title_font = _font(32, bold=True)
    label_font = _font(22)
    small_font = _font(17)
    draw.text((90, 42), "Day 30 Browser policy score distribution", fill="#101826", font=title_font)
    draw.text((90, 88), f"n={len(scores['wasm'])} WASM / n={len(scores['webgpu'])} WebGPU · 50 unique WebGPU seeds · raw ALE episode return", fill="#526277", font=small_font)

    plot_left, plot_top, plot_right, plot_bottom = 150, 160, 1320, 680
    all_scores = scores["wasm"] + scores["webgpu"]
    lower, upper = min(all_scores), max(all_scores)
    if lower == upper:
        lower -= 1
        upper += 1
    padding = max(0.5, (upper - lower) * 0.12)
    lower -= padding
    upper += padding

    def y(value: float) -> int:
        return round(plot_bottom - (value - lower) / (upper - lower) * (plot_bottom - plot_top))

    for tick in _ticks(lower, upper):
        y_tick = y(tick)
        draw.line((plot_left, y_tick, plot_right, y_tick), fill="#d4d0c7", width=1)
        draw.text((plot_left - 55, y_tick - 10), _number(tick), fill="#526277", font=small_font)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#101826", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#101826", width=2)
    draw.text((35, plot_top + 170), "return", fill="#526277", font=small_font)

    centers = {"wasm": 480, "webgpu": 980}
    fills = {"wasm": "#dbe5ff", "webgpu": "#d7ff5f"}
    points = {"wasm": "#557dff", "webgpu": "#263f88"}
    for backend in ("wasm", "webgpu"):
        values = sorted(scores[backend])
        center = centers[backend]
        q1, median, q3 = _percentile(values, 0.25), _percentile(values, 0.5), _percentile(values, 0.75)
        mean = sum(values) / len(values)
        draw.line((center, y(values[0]), center, y(values[-1])), fill="#526277", width=4)
        draw.line((center - 48, y(values[0]), center + 48, y(values[0])), fill="#526277", width=4)
        draw.line((center - 48, y(values[-1]), center + 48, y(values[-1])), fill="#526277", width=4)
        draw.rectangle((center - 80, y(q3), center + 80, y(q1)), fill=fills[backend], outline="#101826", width=3)
        draw.line((center - 80, y(median), center + 80, y(median)), fill="#ff725e", width=5)
        draw.regular_polygon((center, y(mean), 10), n_sides=4, rotation=45, fill="#101826")
        for index, value in enumerate(values):
            jitter = ((index * 17) % 81) - 40
            draw.ellipse((center + jitter - 5, y(value) - 5, center + jitter + 5, y(value) + 5), fill=points[backend])
        draw.text((center - 47, plot_bottom + 28), backend.upper(), fill="#101826", font=label_font)

    contract = payload.get("contract", {})
    draw.text((90, 118), f"Contract={contract.get('contractId')} · model={payload['modelSha256'][:12]}… · source=real Browser ALE evaluation", fill="#526277", font=small_font)
    image.save(args.output)

    metadata = {
        "artifactType": "day30_browser_policy_score_distribution",
        "sourceArtifact": args.input.as_posix(),
        "output": args.output.as_posix(),
        "generationCommand": "python -m scripts.visualization.visualize_day30_browser_scores --input assets/day30/final-browser-score-comparison.json --output assets/day30/final-browser-score-distribution.png",
        "backends": ["wasm", "webgpu"],
        "episodeCountPerBackend": {backend: len(values) for backend, values in scores.items()},
        "uniqueSeedCountPerBackend": {
            backend: len({episode["seed"] for episode in payload["results"][backend]["episodes"]})
            for backend in ("wasm", "webgpu")
        },
        "aggregatePerBackend": {
            backend: payload["results"][backend].get("aggregate")
            for backend in ("wasm", "webgpu")
        },
        "modelSha256": payload["modelSha256"],
        "contract": contract,
    }
    args.metadata.write_text(f"{json.dumps(metadata, indent=2, ensure_ascii=False)}\n", encoding="utf-8")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf")
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    weight = position - lower
    return values[lower] + (values[upper] - values[lower]) * weight


def _ticks(lower: float, upper: float) -> list[float]:
    step = 5 if upper - lower <= 100 else 10
    start = math.ceil(lower / step) * step
    stop = math.floor(upper / step) * step
    return [float(value) for value in range(start, stop + step, step)]


def _number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


if __name__ == "__main__":
    main()
