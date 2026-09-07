"""Render Day 24 latency figures directly from raw samples and summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from breakout_rl.artifacts import repository_relative_path, repository_root, sha256_file
from breakout_rl.benchmarking import load_benchmark_artifacts, summarize_samples


DEFAULT_INPUT_ROOT = Path("assets/day24/benchmarks")
DEFAULT_BATCH1_OUTPUT = Path("assets/day24/batch1-latency.png")
DEFAULT_DISTRIBUTION_OUTPUT = Path("assets/day24/latency-distribution.png")


def _primary_results(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("benchmark summary has no results list")
    selected = [
        result
        for result in results
        if isinstance(result, Mapping)
        and result.get("batch_size") == 1
        and result.get("thread_setting") == "default"
    ]
    expected = {
        ("PyTorch", "cpu"),
        ("PyTorch", "cuda"),
        ("ONNX Runtime", "cpu"),
        ("ONNX Runtime", "cuda"),
    }
    observed = {
        (str(result.get("runtime")), str(result.get("requested_provider")))
        for result in selected
    }
    if not expected.issubset(observed):
        raise ValueError(
            "batch=1 default-thread summary is missing required runtime matrix: "
            f"observed={sorted(observed)}"
        )
    return selected


def _check_raw_samples(
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    raw_samples = payload.get("raw_samples")
    if not isinstance(raw_samples, list):
        raise ValueError("benchmark artifact has no raw samples")
    selected = [
        sample
        for sample in raw_samples
        if isinstance(sample, Mapping)
        and sample.get("result_key") == result.get("result_key")
    ]
    if len(selected) != int(result.get("sample_count", -1)):
        raise ValueError(
            f"raw sample count does not match summary for {result.get('result_key')}"
        )
    reconstructed = summarize_samples(
        [int(sample["latency_ns"]) for sample in selected],
        batch_size=int(result["batch_size"]),
    )
    summary_latency = result.get("latency")
    if not isinstance(summary_latency, Mapping):
        raise ValueError("benchmark result has no latency summary")
    for field in ("p50_ns", "p95_ns", "mean_ns", "min_ns", "max_ns"):
        if not np.isclose(float(summary_latency[field]), float(reconstructed[field])):
            raise ValueError(
                f"summary field {field} cannot be reconstructed from raw samples"
            )
    return selected


def _label(result: Mapping[str, Any]) -> str:
    runtime = str(result["runtime"])
    provider = str(result["requested_provider"]).upper()
    actual = str(result["actual_provider"])
    return f"{runtime}\n{provider} FP32\n{actual}"


def render_batch1_latency(
    payload: Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Plot P50/P95 for the primary batch=1 runtime matrix."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = _primary_results(payload)
    for result in selected:
        _check_raw_samples(payload, result)
    order = [
        ("PyTorch", "cpu"),
        ("PyTorch", "cuda"),
        ("ONNX Runtime", "cpu"),
        ("ONNX Runtime", "cuda"),
    ]
    ordered = sorted(
        [
            result
            for result in selected
            if result.get("scope") in {"model_only", "end_to_end"}
        ],
        key=lambda result: (
            0 if result.get("scope") == "model_only" else 1,
            order.index((result.get("runtime"), result.get("requested_provider"))),
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(16.5, 7.5), constrained_layout=True)
    colors = {"P50": "#2f6f9f", "P95": "#c55a11"}
    for axis, scope in zip(axes, ("model_only", "end_to_end"), strict=True):
        values = [
            result
            for result in ordered
            if result.get("scope") == scope
        ]
        if len(values) != 4:
            raise ValueError(f"expected four batch=1 {scope} results")
        positions = np.arange(len(values), dtype=np.float64)
        width = 0.34
        p50 = [float(result["latency"]["p50_ms"]) for result in values]
        p95 = [float(result["latency"]["p95_ms"]) for result in values]
        axis.bar(positions - width / 2, p50, width, label="P50", color=colors["P50"])
        axis.bar(positions + width / 2, p95, width, label="P95", color=colors["P95"])
        axis.set_xticks(positions, [_label(result) for result in values])
        axis.set_ylabel("latency (ms)")
        axis.set_title(
            "Model call only" if scope == "model_only" else "End-to-end policy decision"
        )
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False)
        for position, p50_value, p95_value in zip(positions, p50, p95, strict=True):
            axis.text(position - width / 2, p50_value, f"{p50_value:.3f}", ha="center", va="bottom", fontsize=8)
            axis.text(position + width / 2, p95_value, f"{p95_value:.3f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle(
        "Day 24 — Batch=1 native inference latency\nP50 and P95 from steady-state raw samples",
        fontsize=15,
        fontweight="bold",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    return output_path


def render_latency_distribution(
    payload: Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Plot raw end-to-end batch=1 distributions with P50/P95 markers."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        result
        for result in _primary_results(payload)
        if result.get("scope") == "end_to_end"
    ]
    order = [
        ("PyTorch", "cpu"),
        ("PyTorch", "cuda"),
        ("ONNX Runtime", "cpu"),
        ("ONNX Runtime", "cuda"),
    ]
    selected.sort(
        key=lambda result: order.index(
            (result.get("runtime"), result.get("requested_provider"))
        )
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, result in zip(axes.ravel(), selected, strict=True):
        raw_samples = _check_raw_samples(payload, result)
        values_ms = np.asarray(
            [int(sample["latency_ns"]) / 1_000_000.0 for sample in raw_samples],
            dtype=np.float64,
        )
        axis.hist(values_ms, bins=min(30, max(8, len(values_ms) // 5)), color="#4f81bd", alpha=0.78)
        p50 = float(result["latency"]["p50_ms"])
        p95 = float(result["latency"]["p95_ms"])
        axis.axvline(p50, color="#2f6f9f", linewidth=1.7, label=f"P50 {p50:.3f} ms")
        axis.axvline(p95, color="#c55a11", linewidth=1.7, linestyle="--", label=f"P95 {p95:.3f} ms")
        axis.set_title(_label(result))
        axis.set_xlabel("end-to-end latency (ms)")
        axis.set_ylabel("sample count")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Day 24 — Raw batch=1 end-to-end latency distributions",
        fontsize=15,
        fontweight="bold",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    return output_path


def _write_sidecar(
    figure: Path,
    *,
    benchmark_dir: Path,
    root: Path,
    command: str,
    technical_question: str,
) -> Path:
    sidecar = figure.with_suffix(".json")
    summary = benchmark_dir / "summary.json"
    raw = benchmark_dir / "raw-samples.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "day24_inference_benchmark_visualization",
                "technical_question": technical_question,
                "benchmark_id": benchmark_dir.name,
                "source_summary": {
                    "path": repository_relative_path(summary, root=root),
                    "sha256": sha256_file(summary),
                },
                "source_raw_samples": {
                    "path": repository_relative_path(raw, root=root),
                    "sha256": sha256_file(raw),
                },
                "output": repository_relative_path(figure, root=root),
                "command": command,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecar


def render_from_benchmark_id(
    *,
    benchmark_id: str,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    batch1_output: str | Path = DEFAULT_BATCH1_OUTPUT,
    distribution_output: str | Path = DEFAULT_DISTRIBUTION_OUTPUT,
) -> tuple[Path, Path, Path, Path]:
    root = repository_root()
    benchmark_dir = (root / input_root / benchmark_id).resolve()
    payload = load_benchmark_artifacts(benchmark_dir)
    if payload.get("benchmark_id") != benchmark_id:
        raise ValueError("benchmark directory and artifact benchmark_id do not match")
    batch1_path = (root / batch1_output).resolve()
    distribution_path = (root / distribution_output).resolve()
    batch1 = render_batch1_latency(payload, batch1_path)
    distribution = render_latency_distribution(payload, distribution_path)
    command_base = (
        "python -m scripts.visualization.visualize_inference_benchmarks "
        f"--benchmark-id {benchmark_id}"
    )
    batch1_sidecar = _write_sidecar(
        batch1,
        benchmark_dir=benchmark_dir,
        root=root,
        command=command_base,
        technical_question=(
            "How do the steady-state P50 and P95 decision costs compare across "
            "runtime/provider and model-only versus end-to-end scope?"
        ),
    )
    distribution_sidecar = _write_sidecar(
        distribution,
        benchmark_dir=benchmark_dir,
        root=root,
        command=command_base,
        technical_question=(
            "Does batch=1 end-to-end latency have a tail that a mean or fastest "
            "sample would hide?"
        ),
    )
    return batch1, distribution, batch1_sidecar, distribution_sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--batch1-output", type=Path, default=DEFAULT_BATCH1_OUTPUT)
    parser.add_argument(
        "--distribution-output",
        type=Path,
        default=DEFAULT_DISTRIBUTION_OUTPUT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = render_from_benchmark_id(
            benchmark_id=args.benchmark_id,
            input_root=args.input_root,
            batch1_output=args.batch1_output,
            distribution_output=args.distribution_output,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 24 inference visualization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"outputs": [str(path) for path in outputs]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
