"""Plot the Day 25 FP32/FP16 correctness, latency, and size evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from breakout_rl.artifacts import repository_relative_path, repository_root, sha256_file


DEFAULT_INPUT = Path("assets/day25/precision-comparison.json")
DEFAULT_OUTPUT = Path("assets/day25/fp32-vs-fp16.png")


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)


def _relative(root: Path, path: Path) -> str:
    return repository_relative_path(path, root=root)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _benchmark_result(summary: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    results = summary.get("results")
    if not isinstance(results, list):
        raise ValueError("benchmark summary has no results list")
    for result in results:
        if (
            isinstance(result, Mapping)
            and result.get("target") == target
            and result.get("scope") == "end_to_end"
            and int(result.get("batch_size", -1)) == 1
        ):
            return result
    raise ValueError(f"missing batch=1 end-to-end result for {target}")


def _load_sources(root: Path, input_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    comparison = _json_object(input_path)
    if comparison.get("artifact_type") != "day25_precision_comparison":
        raise ValueError("comparison artifact has an unexpected artifact_type")
    benchmark = _require_mapping(comparison.get("benchmark"), label="comparison.benchmark")
    summary_path = (root / str(benchmark["summary_path"])).resolve()
    raw_path = (root / str(benchmark["raw_samples_path"])).resolve()
    summary_hash = sha256_file(summary_path)
    raw_hash = sha256_file(raw_path)
    if summary_hash != benchmark.get("summary_sha256"):
        raise ValueError("benchmark summary hash does not match comparison artifact")
    if raw_hash != benchmark.get("raw_samples_sha256"):
        raise ValueError("benchmark raw-sample hash does not match comparison artifact")
    summary = _json_object(summary_path)
    if summary.get("benchmark_id") != benchmark.get("benchmark_id"):
        raise ValueError("benchmark summary id does not match comparison artifact")
    sources = {
        "comparison": _relative(root, input_path),
        "comparison_sha256": sha256_file(input_path),
        "benchmark_summary": _relative(root, summary_path),
        "benchmark_summary_sha256": summary_hash,
        "benchmark_raw_samples": _relative(root, raw_path),
        "benchmark_raw_samples_sha256": raw_hash,
    }
    return comparison, summary, sources


def _precision_arrays(comparison: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    parity = _require_mapping(comparison.get("parity"), label="comparison.parity")
    candidates = _require_mapping(parity.get("precision_pairs"), label="comparison.parity.precision_pairs")
    target_order = [
        "pytorch_cuda_fp16_vs_pytorch_cuda_fp32",
        "onnx_cuda_fp16_vs_onnx_cuda_fp32",
    ]
    labels = [
        "PyTorch CUDA\nFP16 vs FP32",
        "ORT CUDA\nFP16 vs FP32",
    ]
    errors: list[float] = []
    agreements: list[float] = []
    for target in target_order:
        candidate = _require_mapping(candidates.get(target), label=f"candidate {target}")
        metrics = _require_mapping(candidate.get("metrics"), label=f"candidate {target} metrics")
        errors.append(float(metrics["max_absolute_error"]))
        agreements.append(float(metrics["action_agreement_rate"]))
    return labels, np.asarray(errors), np.asarray(agreements)


def _latency_arrays(summary: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    target_order = [
        "pytorch_cuda_fp32",
        "pytorch_cuda_fp16",
        "onnx_cuda_fp32",
        "onnx_cuda_fp16",
    ]
    labels = ["PyTorch\nFP32", "PyTorch\nFP16", "ORT\nFP32", "ORT\nFP16"]
    p50: list[float] = []
    p95: list[float] = []
    for target in target_order:
        result = _benchmark_result(summary, target)
        latency = _require_mapping(result.get("latency"), label=f"{target} latency")
        p50.append(float(latency["p50_ms"]))
        p95.append(float(latency["p95_ms"]))
    return labels, np.asarray(p50), np.asarray(p95)


def _model_size_arrays(comparison: Mapping[str, Any], root: Path) -> tuple[list[str], np.ndarray]:
    lineage = _require_mapping(comparison.get("lineage"), label="comparison.lineage")
    onnx_path = (root / str(lineage["onnx_model"]["path"])).resolve()
    fp16_path = (root / str(lineage["fp16_model"]["path"])).resolve()
    sizes = np.asarray([onnx_path.stat().st_size, fp16_path.stat().st_size], dtype=np.float64)
    declared = _require_mapping(
        _require_mapping(comparison["benchmark"], label="comparison.benchmark").get("model_size_bytes"),
        label="comparison.benchmark.model_size_bytes",
    )
    if int(declared["fp32"]) != int(sizes[0]) or int(declared["fp16"]) != int(sizes[1]):
        raise ValueError("model file sizes do not match the comparison artifact")
    return ["FP32\nONNX", "FP16\nONNX"], sizes


def _annotate_bars(axis: Any, bars: Any, values: Sequence[float], *, format_string: str) -> None:
    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            format_string.format(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def visualize_precision_comparison(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Read real Day 25 artifacts and render one quantitative comparison figure."""

    root = repository_root()
    input_file = (root / Path(input_path)).resolve() if not Path(input_path).is_absolute() else Path(input_path).resolve()
    output_file = (root / Path(output_path)).resolve() if not Path(output_path).is_absolute() else Path(output_path).resolve()
    comparison, summary, sources = _load_sources(root, input_file)
    precision_labels, errors, agreements = _precision_arrays(comparison)
    latency_labels, p50, p95 = _latency_arrays(summary)
    size_labels, sizes = _model_size_arrays(comparison, root)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
        "Day 25 — FP32 vs FP16 on NVIDIA CUDA\n"
        "Correctness first, then batch=1 end-to-end latency",
        fontsize=16,
        fontweight="bold",
    )
    palette = ["#2563eb", "#dc2626", "#64748b"]

    error_axis = axes[0, 0]
    error_bars = error_axis.bar(precision_labels, errors, color=palette)
    error_axis.set_yscale("log")
    error_axis.set_ylabel("Maximum absolute Q-value error")
    error_axis.set_title("Numerical error vs PyTorch CUDA FP32")
    error_axis.grid(axis="y", alpha=0.25)
    _annotate_bars(error_axis, error_bars, errors, format_string="{:.3g}")

    agreement_axis = axes[0, 1]
    agreement_bars = agreement_axis.bar(precision_labels, agreements, color=palette)
    agreement_axis.axhline(1.0, color="#111827", linestyle="--", linewidth=1, label="configured target = 1.0")
    agreement_axis.set_ylim(0.0, 1.05)
    agreement_axis.set_ylabel("Action agreement rate")
    agreement_axis.set_title("Does argmax choose the same action?")
    agreement_axis.grid(axis="y", alpha=0.25)
    _annotate_bars(agreement_axis, agreement_bars, agreements, format_string="{:.2%}")

    latency_axis = axes[1, 0]
    x = np.arange(len(latency_labels), dtype=np.float64)
    width = 0.35
    p50_bars = latency_axis.bar(x - width / 2.0, p50, width, label="P50", color="#0ea5e9")
    p95_bars = latency_axis.bar(x + width / 2.0, p95, width, label="P95", color="#f97316")
    latency_axis.set_xticks(x, latency_labels)
    latency_axis.set_ylabel("Latency (ms)")
    latency_axis.set_title("Batch=1 end-to-end decision latency")
    latency_axis.legend(frameon=False)
    latency_axis.grid(axis="y", alpha=0.25)
    _annotate_bars(latency_axis, p50_bars, p50, format_string="{:.2f}")
    _annotate_bars(latency_axis, p95_bars, p95, format_string="{:.2f}")

    size_axis = axes[1, 1]
    size_bars = size_axis.bar(size_labels, sizes / (1024**2), color=["#64748b", "#16a34a"])
    size_axis.set_ylabel("Model file size (MiB)")
    size_axis.set_title("Independent ONNX artifact size")
    size_axis.grid(axis="y", alpha=0.25)
    _annotate_bars(size_axis, size_bars, sizes / (1024**2), format_string="{:.2f}")

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelsize=9)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "schema_version": 1,
        "artifact_type": "day25_precision_comparison_visualization",
        "technical_question": "Does internal FP16 preserve policy decisions and improve native batch=1 latency?",
        "command": "python -m scripts.visualization.visualize_precision_comparison",
        "output": _relative(root, output_file),
        "output_sha256": sha256_file(output_file),
        "sources": sources,
        "source_model_paths": {
            "fp32_onnx": _relative(root, (root / str(comparison["lineage"]["onnx_model"]["path"])).resolve()),
            "fp16_onnx": _relative(root, (root / str(comparison["lineage"]["fp16_model"]["path"])).resolve()),
        },
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib.__version__,
    }
    metadata_path = output_file.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = visualize_precision_comparison(
            input_path=args.input,
            output_path=args.output,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"Day 25 precision visualization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": result["output"], "output_sha256": result["output_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
