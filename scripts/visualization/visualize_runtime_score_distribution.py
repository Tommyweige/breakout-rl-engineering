"""Render the real Day 26 multi-episode runtime score artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.artifacts import repository_relative_path, repository_root, sha256_file
from breakout_rl.runtime_scores import validate_runtime_score_artifact


DEFAULT_INPUT = Path("assets/day26/runtime-score-comparison.json")
DEFAULT_OUTPUT = Path("assets/day26/runtime-score-distribution.png")

RUNTIME_LABELS = {
    "pytorch_cuda_fp32": "PyTorch FP32",
    "onnx_cuda_fp32": "ORT FP32",
    "tensorrt_cuda_fp32": "TensorRT FP32",
    "tensorrt_cuda_fp16": "TensorRT FP16",
}
RUNTIME_COLORS = {
    "pytorch_cuda_fp32": "#64748b",
    "onnx_cuda_fp32": "#0f766e",
    "tensorrt_cuda_fp32": "#2563eb",
    "tensorrt_cuda_fp16": "#f97316",
}
PAIRING_LABELS = {
    "onnx_fp32_minus_pytorch_fp32": "ORT - PT\nFP32",
    "tensorrt_fp32_minus_pytorch_fp32": "TRT - PT\nFP32",
    "tensorrt_fp32_minus_onnx_fp32": "TRT - ORT\nFP32",
    "tensorrt_fp16_minus_pytorch_fp32": "TRT FP16 -\nPT FP32",
}
PAIRING_CANDIDATES = {
    "onnx_fp32_minus_pytorch_fp32": "onnx_cuda_fp32",
    "tensorrt_fp32_minus_pytorch_fp32": "tensorrt_cuda_fp32",
    "tensorrt_fp32_minus_onnx_fp32": "tensorrt_cuda_fp32",
    "tensorrt_fp16_minus_pytorch_fp32": "tensorrt_cuda_fp16",
}


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)


def _relative(root: Path, path: Path) -> str:
    return repository_relative_path(path, root=root)


def load_runtime_score_visualization_data(path: str | Path) -> dict[str, Any]:
    """Load and validate the formal JSON used by the score distribution plot."""

    source = Path(path)
    payload = _json_object(source)
    validated = validate_runtime_score_artifact(payload)
    targets = tuple(validated["targets"])
    scores = {
        runtime: [
            float(row["score"])
            for row in validated["rows_by_runtime"][runtime]
            if not row["failure"]
        ]
        for runtime in targets
    }
    paired = {
        name: [
            float(row["difference"])
            for row in payload["paired_score_differences"][name]["differences"]
            if row["difference"] is not None
        ]
        for name in validated["paired_score_differences"]
    }
    return {
        "source": source,
        "payload": payload,
        "episode_seeds": validated["episode_seeds"],
        "targets": targets,
        "scores": scores,
        "paired_differences": paired,
    }


def _configure_font() -> None:
    """Use an installed Traditional Chinese font when available on Windows."""

    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["axes.unicode_minus"] = False
    for candidate in (
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/mingliu.ttc"),
        Path("C:/Windows/Fonts/NotoSansTC-VF.ttf"),
    ):
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            matplotlib.rcParams["font.family"] = [
                font_manager.FontProperties(fname=str(candidate)).get_name()
            ]
            return


def render_runtime_score_distribution(
    source: str | Path,
    destination: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Render score distributions and same-seed paired differences."""

    data = load_runtime_score_visualization_data(source)
    destination_path = Path(destination).resolve()
    if destination_path.exists() and not force:
        raise FileExistsError(f"output already exists: {destination_path}; use --force")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _configure_font()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.5),
        dpi=180,
        gridspec_kw={"width_ratios": (1.2, 1.0)},
    )
    distribution_axis, paired_axis = axes

    targets = tuple(data["targets"])
    runtime_values = [data["scores"][runtime] for runtime in targets]
    positions = list(range(1, len(targets) + 1))
    boxplot = distribution_axis.boxplot(
        runtime_values,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": "#334155", "linewidth": 1.0},
        whiskerprops={"color": "#475569", "linewidth": 1.0},
        capprops={"color": "#475569", "linewidth": 1.0},
        medianprops={"color": "#111827", "linewidth": 1.7},
    )
    for patch, runtime in zip(boxplot["boxes"], targets):
        patch.set_facecolor(RUNTIME_COLORS[runtime])
        patch.set_alpha(0.35)
    for position, runtime, values in zip(positions, targets, runtime_values):
        if len(values) == 1:
            offsets = [0.0]
        else:
            offsets = [
                -0.18 + 0.36 * index / (len(values) - 1) for index in range(len(values))
            ]
        distribution_axis.scatter(
            [position + offset for offset in offsets],
            values,
            s=21,
            color=RUNTIME_COLORS[runtime],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        distribution_axis.scatter(
            [position],
            [sum(values) / len(values)],
            marker="D",
            s=38,
            color=RUNTIME_COLORS[runtime],
            edgecolors="#111827",
            linewidths=0.7,
            zorder=4,
        )
    distribution_axis.set_xticks(
        positions,
        [RUNTIME_LABELS[runtime] for runtime in targets],
        rotation=12,
    )
    distribution_axis.set_ylabel("Raw Atari episode score")
    distribution_axis.set_title("30 fixed-seed score distributions")
    distribution_axis.grid(axis="y", alpha=0.25)
    distribution_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#475569",
                markeredgecolor="#111827",
                markersize=6,
                label="Mean",
            ),
            Line2D(
                [0],
                [0],
                color="#111827",
                linewidth=1.8,
                label="Median",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#475569",
                markerfacecolor="#475569",
                markersize=5,
                linestyle="none",
                label="Episode score",
            ),
        ],
        loc="best",
        fontsize=8,
    )

    pair_names = tuple(data["paired_differences"])
    pair_values = [data["paired_differences"][name] for name in pair_names]
    pair_positions = list(range(1, len(pair_names) + 1))
    pair_boxplot = paired_axis.boxplot(
        pair_values,
        positions=pair_positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": "#334155", "linewidth": 1.0},
        whiskerprops={"color": "#475569", "linewidth": 1.0},
        capprops={"color": "#475569", "linewidth": 1.0},
        medianprops={"color": "#111827", "linewidth": 1.7},
    )
    for patch, name in zip(pair_boxplot["boxes"], pair_names):
        patch.set_facecolor(RUNTIME_COLORS[PAIRING_CANDIDATES[name]])
        patch.set_alpha(0.35)
    for position, values in zip(pair_positions, pair_values):
        if len(values) == 1:
            offsets = [0.0]
        else:
            offsets = [
                -0.16 + 0.32 * index / (len(values) - 1) for index in range(len(values))
            ]
        paired_axis.scatter(
            [position + offset for offset in offsets],
            values,
            s=19,
            color="#475569",
            alpha=0.68,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
    paired_axis.axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    paired_axis.set_xticks(
        pair_positions,
        [PAIRING_LABELS.get(name, name) for name in pair_names],
    )
    paired_axis.set_ylabel("Same-seed score difference (left - right)")
    paired_axis.set_title("Paired score differences")
    paired_axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Day 26 multi-episode runtime score evaluation",
        fontsize=14,
        y=0.99,
    )
    figure.text(
        0.5,
        0.01,
        f"每個 runtime {len(data['episode_seeds'])} 局；資料直接來自 runtime-score-comparison.json",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination_path, format="png")
    plt.close(figure)
    return destination_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repository_root()
    source = (
        (root / args.input).resolve() if not args.input.is_absolute() else args.input
    )
    output = (
        (root / args.output).resolve() if not args.output.is_absolute() else args.output
    )
    try:
        rendered = render_runtime_score_distribution(source, output, force=args.force)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Runtime score visualization failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": _relative(root, rendered),
                "output_sha256": sha256_file(rendered),
                "source": _relative(root, source),
                "source_sha256": sha256_file(source),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
