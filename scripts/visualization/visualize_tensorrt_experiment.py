"""Render the real Day 26 TensorRT evidence or its preflight blocker summary."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from breakout_rl.artifacts import repository_relative_path, repository_root, sha256_file


DEFAULT_PREFLIGHT = Path("assets/day26/tensorrt-preflight.json")
DEFAULT_COMPARISON = Path("assets/day26/tensorrt-comparison.json")
DEFAULT_OUTPUT = Path("assets/day26/tensorrt-comparison.png")
DEFAULT_BLOCKED_OUTPUT = Path("assets/day26/tensorrt-blocker-summary.png")


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)


def _resolve(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _relative(root: Path, path: Path) -> str:
    return repository_relative_path(path, root=root)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _render_comparison(result: Mapping[str, Any], output: Path) -> None:
    parity = _mapping(result.get("parity"), label="parity")
    candidates = _mapping(parity.get("candidates"), label="parity.candidates")
    evaluation = _mapping(result.get("evaluation"), label="evaluation")
    eval_candidates = _mapping(
        evaluation.get("candidates"), label="evaluation.candidates"
    )
    benchmark = _mapping(result.get("benchmark"), label="benchmark")
    latency = _mapping(
        benchmark.get("batch1_end_to_end"), label="benchmark.batch1_end_to_end"
    )
    sizes = _mapping(
        benchmark.get("artifact_sizes_bytes"), label="benchmark.artifact_sizes_bytes"
    )

    targets = ["tensorrt_cuda_fp32", "tensorrt_cuda_fp16"]
    labels = ["TensorRT FP32", "TensorRT FP16"]
    colors = ["#1f77b4", "#ff7f0e"]
    parity_abs = [
        float(
            _mapping(candidates[target], label=target)["metrics"]["max_absolute_error"]
        )
        for target in targets
    ]
    parity_agreement = [
        float(
            _mapping(candidates[target], label=target)["metrics"][
                "action_agreement_rate"
            ]
        )
        for target in targets
    ]
    eval_agreement = [
        float(
            _mapping(eval_candidates[target], label=target)["metrics"][
                "action_agreement_rate"
            ]
        )
        for target in targets
    ]
    latency_p50 = [
        float(_mapping(latency[target], label=target)["p50_ms"])
        for target in ["tensorrt_cuda_fp32", "tensorrt_cuda_fp16"]
    ]
    latency_p95 = [
        float(_mapping(latency[target], label=target)["p95_ms"])
        for target in ["tensorrt_cuda_fp32", "tensorrt_cuda_fp16"]
    ]
    size_values = [
        int(sizes[target])
        for target in [
            "onnx_fp32",
            "onnx_fp16",
            "tensorrt_cuda_fp32",
            "tensorrt_cuda_fp16",
        ]
    ]
    size_labels = ["ONNX FP32", "ONNX FP16", "TRT FP32", "TRT FP16"]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure.suptitle(
        "Day 26 TensorRT experiment — native NVIDIA GPU only\n"
        f"{result.get('runtime', {}).get('gpu_model')} · validation={result.get('validation_status')}",
        fontsize=14,
    )

    axis = axes[0, 0]
    x = np.arange(len(targets))
    axis.bar(x, parity_abs, color=colors)
    axis.set_xticks(x, labels)
    axis.set_yscale("log")
    axis.set_ylabel("max absolute Q error (log scale)")
    axis.set_title("Probe parity against live PyTorch FP32")
    for index, value in enumerate(parity_abs):
        axis.text(index, value, f"{value:.2g}", ha="center", va="bottom", fontsize=9)

    axis = axes[0, 1]
    width = 0.36
    axis.bar(
        x - width / 2, parity_agreement, width, label="60 probe states", color="#4c78a8"
    )
    axis.bar(
        x + width / 2,
        eval_agreement,
        width,
        label="15 fixed-seed episodes",
        color="#f58518",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("action agreement rate")
    axis.set_title("Decision agreement")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    axis.bar(x - width / 2, latency_p50, width, label="P50", color="#59a14f")
    axis.bar(x + width / 2, latency_p95, width, label="P95", color="#e15759")
    axis.set_xticks(x, labels)
    axis.set_ylabel("milliseconds")
    axis.set_title("Batch=1 end-to-end latency")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    size_mb = [value / (1024 * 1024) for value in size_values]
    x_size = np.arange(len(size_values))
    axis.bar(x_size, size_mb, color=["#9c755f", "#bab0ab", "#1f77b4", "#ff7f0e"])
    axis.set_xticks(x_size, size_labels, rotation=15)
    axis.set_ylabel("MiB")
    axis.set_title("Serialized model/engine size")
    for index, value in enumerate(size_mb):
        axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _render_blocked(preflight: Mapping[str, Any], output: Path) -> None:
    software = _mapping(preflight.get("software"), label="software")
    tensorrt = _mapping(software.get("tensorrt"), label="software.tensorrt")
    baseline = _mapping(preflight.get("baseline"), label="baseline")
    gpu = _mapping(
        _mapping(preflight.get("host"), label="host").get("gpu"), label="host.gpu"
    )
    rows = [
        ("preflight status", str(preflight.get("status"))),
        ("GPU", str(gpu.get("gpu_model") or "unavailable")),
        ("compute capability", str(gpu.get("compute_capability") or "unavailable")),
        ("PyTorch CUDA baseline", str(baseline.get("status"))),
        ("TensorRT import", str(tensorrt.get("imported"))),
        ("ONNX parser", str(tensorrt.get("onnx_parser_available"))),
        ("trtexec", str(software.get("trtexec_path") or "not found")),
        (
            "blockers",
            "; ".join(str(value) for value in preflight.get("blockers", [])) or "none",
        ),
    ]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.axis("off")
    axis.set_title(
        "Day 26 TensorRT preflight — blocked/diagnostic result\n"
        "No TensorRT performance numbers are fabricated",
        fontsize=14,
        loc="left",
    )
    table = axis.table(
        cellText=[[label, value] for label, value in rows],
        colLabels=["Observed check", "Real preflight value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.28, 0.68],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4c566a")
        elif row % 2 == 0:
            cell.set_facecolor("#eceff4")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def render(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    preflight_path = _resolve(root, args.preflight)
    preflight = _json_object(preflight_path)
    comparison_path: Path | None = None
    comparison: dict[str, Any] | None = None
    if args.comparison is not None:
        candidate = (
            (root / args.comparison).resolve()
            if not args.comparison.is_absolute()
            else args.comparison.resolve()
        )
        if candidate.is_file():
            comparison_path = candidate
            comparison = _json_object(candidate)
    preflight_ready = preflight.get("status") in {"READY", "CLI_ONLY"}
    comparison_preflight = (
        comparison.get("preflight") if isinstance(comparison, Mapping) else None
    )
    comparison_matches_preflight = (
        comparison is not None
        and isinstance(comparison_preflight, Mapping)
        and comparison_preflight.get("sha256") == sha256_file(preflight_path)
    )
    if (
        comparison is not None
        and comparison.get("status") == "completed"
        and preflight_ready
        and comparison_matches_preflight
    ):
        output = (
            (root / args.output).resolve()
            if not args.output.is_absolute()
            else args.output.resolve()
        )
        _render_comparison(comparison, output)
        output_kind = "comparison"
    else:
        output = (
            (root / args.blocked_output).resolve()
            if not args.blocked_output.is_absolute()
            else args.blocked_output.resolve()
        )
        _render_blocked(preflight, output)
        output_kind = "blocker_summary"
    metadata_path = output.with_name(output.stem + "-visualization.json")
    metadata = {
        "schema_version": 1,
        "artifact_type": "day26_tensorrt_visualization",
        "kind": output_kind,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "output": {"path": _relative(root, output), "sha256": sha256_file(output)},
        "sources": {
            "preflight": {
                "path": _relative(root, preflight_path),
                "sha256": sha256_file(preflight_path),
            },
            "comparison": (
                {
                    "path": _relative(root, comparison_path),
                    "sha256": sha256_file(comparison_path),
                }
                if comparison_path is not None
                else None
            ),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "matplotlib_version": matplotlib.__version__,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blocked-output", type=Path, default=DEFAULT_BLOCKED_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    metadata = render(build_parser().parse_args(argv))
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
