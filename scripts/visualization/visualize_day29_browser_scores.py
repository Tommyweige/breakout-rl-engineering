"""Plot the real Day 29 browser policy score distribution from captured JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("assets/day29/browser-policy-score-comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("assets/day29/browser-policy-score-distribution.png"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("assets/day29/browser-policy-score-distribution.json"),
    )
    return parser.parse_args()


def read_comparison(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("artifactType") != "day29_browser_policy_score_comparison":
        raise ValueError(f"{path}: unexpected artifactType")
    for backend in ("wasm", "webgpu"):
        result = payload.get("results", {}).get(backend, {})
        episodes = result.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"{path}: {backend} episodes are missing")
        if any(not isinstance(episode.get("episodeReturn"), (int, float)) for episode in episodes):
            raise ValueError(f"{path}: {backend} contains a non-numeric episode return")
    return payload


def main() -> None:
    args = parse_args()
    payload = read_comparison(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    scores = {
        backend: [float(episode["episodeReturn"]) for episode in payload["results"][backend]["episodes"]]
        for backend in ("wasm", "webgpu")
    }
    image = Image.new("RGB", (1400, 820), "#f6f1e7")
    draw = ImageDraw.Draw(image)
    title_font = _font(32, bold=True)
    label_font = _font(22)
    small_font = _font(17)
    draw.text((90, 42), "Day 29 Browser policy score distribution", fill="#17212b", font=title_font)
    draw.text((90, 88), f"n={len(scores['wasm'])} per backend · raw ALE episode return", fill="#506070", font=small_font)

    plot_left, plot_top, plot_right, plot_bottom = 150, 160, 1320, 680
    all_scores = scores["wasm"] + scores["webgpu"]
    lower = min(all_scores)
    upper = max(all_scores)
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
        draw.line((plot_left, y_tick, plot_right, y_tick), fill="#ded8cd", width=1)
        draw.text((plot_left - 55, y_tick - 10), _number(tick), fill="#506070", font=small_font)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#17212b", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#17212b", width=2)
    draw.text((35, plot_top + 170), "return", fill="#506070", font=small_font)

    centers = {"wasm": 480, "webgpu": 980}
    colors = {"wasm": "#dce7ff", "webgpu": "#b7f34b"}
    point_colors = {"wasm": "#2857e8", "webgpu": "#183aa8"}
    for backend in ("wasm", "webgpu"):
        values = sorted(scores[backend])
        center = centers[backend]
        q1, median, q3 = _percentile(values, 0.25), _percentile(values, 0.5), _percentile(values, 0.75)
        mean = sum(values) / len(values)
        draw.line((center, y(values[0]), center, y(values[-1])), fill="#506070", width=4)
        draw.line((center - 48, y(values[0]), center + 48, y(values[0])), fill="#506070", width=4)
        draw.line((center - 48, y(values[-1]), center + 48, y(values[-1])), fill="#506070", width=4)
        draw.rectangle((center - 80, y(q3), center + 80, y(q1)), fill=colors[backend], outline="#17212b", width=3)
        draw.line((center - 80, y(median), center + 80, y(median)), fill="#f16a45", width=5)
        draw.regular_polygon((center, y(mean), 10), n_sides=4, rotation=45, fill="#17212b")
        for index, value in enumerate(values):
            jitter = ((index * 17) % 81) - 40
            draw.ellipse((center + jitter - 6, y(value) - 6, center + jitter + 6, y(value) + 6), fill=point_colors[backend])
        draw.text((center - 47, plot_bottom + 28), backend.upper(), fill="#17212b", font=label_font)

    draw.text((plot_left + 25, plot_top + 20), f"parity={payload['environmentContract']['status'].upper()} · WebGPU smoke={payload['gameplaySmoke']['completedEpisode']}", fill="#506070", font=small_font)
    draw.text((plot_left + 25, plot_top + 52), f"model={payload['modelSha256'][:12]}… · source=real Browser ALE evaluation", fill="#506070", font=small_font)
    image.save(args.output)

    metadata = {
        "artifactType": "day29_browser_policy_score_distribution",
        "sourceArtifact": str(args.input).replace("\\", "/"),
        "output": str(args.output).replace("\\", "/"),
        "generationCommand": "python -m scripts.visualization.visualize_day29_browser_scores --input assets/day29/browser-policy-score-comparison.json --output assets/day29/browser-policy-score-distribution.png",
        "backends": ["wasm", "webgpu"],
        "episodeCountPerBackend": {backend: len(values) for backend, values in scores.items()},
        "modelSha256": payload["modelSha256"],
        "environmentParity": payload["environmentContract"],
    }
    args.metadata.write_text(f"{json.dumps(metadata, indent=2)}\n", encoding="utf-8")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    )
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
    start = int(lower // 1)
    stop = int(upper // 1) + 1
    return [float(value) for value in range(start, stop + 1)]


def _number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


if __name__ == "__main__":
    main()
