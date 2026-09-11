"""Render the Day 28 WASM/WebGPU latency figure from the real browser artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_INPUT = Path("assets/day28/web-benchmark.json")
DEFAULT_OUTPUT = Path("assets/day28/wasm-vs-webgpu-latency.png")


def _read_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"benchmark artifact must be an object: {path}")
    return value


def _percentile(samples: Sequence[float], probability: float) -> float:
    if not samples:
        raise ValueError("latency samples must not be empty")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _backend_samples(payload: Mapping[str, Any], backend: str) -> list[float]:
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("benchmark artifact has no results object")
    result = results.get(backend)
    if not isinstance(result, Mapping):
        raise ValueError(f"benchmark artifact has no {backend} result")
    if result.get("requestedBackend") != backend or result.get("actualBackend") != backend:
        raise ValueError(f"{backend} requested/actual backend is not truthful")
    samples = result.get("rawLatencyMs")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{backend} rawLatencyMs must be a non-empty list")
    values = [float(sample) for sample in samples]
    if any(value < 0 for value in values):
        raise ValueError(f"{backend} latency samples must be non-negative")
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"{backend} result has no summary")
    checks = {
        "p50Ms": _percentile(values, 0.5),
        "p95Ms": _percentile(values, 0.95),
        "meanMs": sum(values) / len(values),
        "minMs": min(values),
        "maxMs": max(values),
    }
    for field, expected in checks.items():
        observed = float(summary[field])
        tolerance = max(1e-9, abs(expected) * 1e-6)
        if abs(observed - expected) > tolerance:
            raise ValueError(f"{backend} {field} cannot be reconstructed from raw samples")
    return values


def load_and_validate(path: str | Path) -> tuple[Mapping[str, Any], dict[str, list[float]]]:
    input_path = Path(path)
    payload = _read_object(input_path)
    if payload.get("artifactType") != "day28_web_benchmark":
        raise ValueError("unexpected Day 28 benchmark artifact type")
    if payload.get("scope") != "prepared_float32_to_q_values_ready":
        raise ValueError("unexpected Day 28 benchmark scope")
    if payload.get("sessionInitialization") != "excluded_from_latency_samples":
        raise ValueError("benchmark must exclude session initialization from timed samples")
    webgpu_support = payload.get("webgpuSupport")
    if not isinstance(webgpu_support, Mapping) or webgpu_support.get("supported") is not True:
        raise ValueError("formal Day 28 benchmark must record WebGPU support")
    samples = {backend: _backend_samples(payload, backend) for backend in ("wasm", "webgpu")}
    return payload, samples


def render_latency_figure(
    payload: Mapping[str, Any],
    samples: Mapping[str, Sequence[float]],
    output: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    backends = ["WASM", "WebGPU"]
    keys = ["wasm", "webgpu"]
    p50 = [_percentile(samples[key], 0.5) for key in keys]
    p95 = [_percentile(samples[key], 0.95) for key in keys]
    positions = np.arange(len(backends), dtype=np.float64)
    width = 0.32

    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    bars_p50 = axis.bar(positions - width / 2, p50, width, label="P50", color="#2857e8")
    bars_p95 = axis.bar(positions + width / 2, p95, width, label="P95", color="#f16a45")
    axis.set_xticks(positions, backends)
    axis.set_ylabel("latency (ms)")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    for bars in (bars_p50, bars_p95):
        for bar in bars:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle(
        "Day 28 — Browser batch=1 inference latency\nprepared Float32Array → q_values ready",
        fontsize=15,
        y=0.96,
    )
    figure.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.78)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)
    return output_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_from_paths(input_path: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    source = Path(input_path)
    payload, samples = load_and_validate(source)
    figure = render_latency_figure(payload, samples, output_path)
    sidecar = figure.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "day28_webgpu_latency_visualization",
                "technical_question": "在同一個 Browser 與同一份模型下，WASM 與 WebGPU 的 steady-state batch=1 P50/P95 延遲如何比較？",
                "source_artifact": {"path": source.as_posix(), "sha256": _sha256(source)},
                "model_sha256": payload.get("modelSha256"),
                "browser": payload.get("browser"),
                "requested_backends": payload.get("requestedBackends"),
                "actual_backends": {
                    backend: payload["results"][backend]["actualBackend"]
                    for backend in ("wasm", "webgpu")
                },
                "raw_sample_counts": {backend: len(samples[backend]) for backend in samples},
                "output": figure.as_posix(),
                "command": "conda run -n breakout-rl-engineering python -m scripts.visualization.visualize_day28_webgpu --input assets/day28/web-benchmark.json --output assets/day28/wasm-vs-webgpu-latency.png",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return figure, sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        figure, sidecar = render_from_paths(args.input, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Day 28 WebGPU visualization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"figure": str(figure), "sidecar": str(sidecar)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
