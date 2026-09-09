"""Reusable, scope-aware inference benchmark measurement helpers."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


BENCHMARK_SCHEMA_VERSION = 1
_BREAKDOWN_FIELDS = (
    "input_prepare_ns",
    "runtime_inference_ns",
    "output_materialization_ns",
    "argmax_ns",
)


class BenchmarkBlockedError(RuntimeError):
    """Raised when a required Day 24 runtime cannot be verified."""


def _noop() -> None:
    return None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Steady-state measurement settings shared by every backend."""

    warmup_iterations: int = 25
    iterations: int = 100
    batch_sizes: tuple[int, ...] = (1, 4, 16, 32)

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if not self.batch_sizes or any(batch < 1 for batch in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive integers")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must not contain duplicates")


@dataclass(frozen=True)
class InferenceBackend:
    """A timing seam around one already-constructed inference runtime.

    ``run_model_input`` is the public validated call when no prevalidated seam
    is available. ``run_prevalidated_model_input`` is used by both benchmark
    scopes after the input has already been prepared and validated outside the
    timed loop. The end-to-end scope still measures input preparation,
    transfers, output materialization, and argmax. This keeps per-call
    validation work from being mistaken for neural-network/runtime execution
    cost.
    """

    runtime: str
    requested_provider: str
    actual_provider: str
    precision: str
    cpu_threads: int | None
    thread_setting: str
    input_ownership: str
    output_ownership: str
    prepare_model_input: Callable[[np.ndarray], Any]
    run_model_input: Callable[[Any], Any]
    materialize_output: Callable[[Any], np.ndarray]
    run_prevalidated_model_input: Callable[[Any], Any] | None = None
    synchronize: Callable[[], None] = _noop
    output_materialization_scope: str = "separate"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def require_cuda_matrix(
    *,
    torch_cuda_available: bool,
    onnxruntime_providers: Sequence[str],
) -> None:
    """Fail closed when either required CUDA benchmark side is unavailable."""

    missing: list[str] = []
    if not torch_cuda_available:
        missing.append("PyTorch CUDA")
    if "CUDAExecutionProvider" not in {
        str(provider) for provider in onnxruntime_providers
    }:
        missing.append("ONNX Runtime CUDAExecutionProvider")
    if missing:
        raise BenchmarkBlockedError(
            "Day 24 is blocked: required CUDA matrix entries are unavailable: "
            + ", ".join(missing)
            + ". No CPU fallback is permitted."
        )


def build_benchmark_id(identity: Mapping[str, Any]) -> str:
    """Build a stable id from the complete benchmark identity/configuration."""

    serialized = json.dumps(
        dict(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"day24-{hashlib.sha256(serialized).hexdigest()[:12]}"


def summarize_samples(
    samples_ns: Sequence[int | float],
    *,
    batch_size: int,
) -> dict[str, float | int]:
    """Summarize raw nanosecond samples without discarding their distribution."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = np.asarray(samples_ns, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError(
            "samples_ns must be a finite, non-empty one-dimensional sequence"
        )
    if (values <= 0).any():
        raise ValueError("latency samples must be positive")
    mean_ns = float(values.mean())
    return {
        "count": int(values.size),
        "min_ns": int(values.min()),
        "max_ns": int(values.max()),
        "p50_ns": float(np.percentile(values, 50, method="linear")),
        "p95_ns": float(np.percentile(values, 95, method="linear")),
        "mean_ns": mean_ns,
        "std_ns": float(values.std()),
        "min_ms": float(values.min() / 1_000_000.0),
        "max_ms": float(values.max() / 1_000_000.0),
        "p50_ms": float(np.percentile(values, 50, method="linear") / 1_000_000.0),
        "p95_ms": float(np.percentile(values, 95, method="linear") / 1_000_000.0),
        "mean_ms": mean_ns / 1_000_000.0,
        "std_ms": float(values.std() / 1_000_000.0),
        "throughput_per_second": float(batch_size / (mean_ns / 1_000_000_000.0)),
    }


def _validate_observations(observations: np.ndarray) -> None:
    if not isinstance(observations, np.ndarray):
        raise TypeError("observations must be a numpy.ndarray")
    if observations.dtype != np.dtype("uint8"):
        raise TypeError("observations must have dtype uint8")
    if observations.ndim != 4 or tuple(observations.shape[1:]) != (4, 84, 84):
        raise ValueError("observations must have shape (N, 4, 84, 84)")
    if observations.shape[0] < 1:
        raise ValueError("observations must contain at least one sample")


def _batch(observations: np.ndarray, *, start: int, batch_size: int) -> np.ndarray:
    indices = (np.arange(batch_size, dtype=np.int64) + start) % observations.shape[0]
    return np.ascontiguousarray(observations[indices])


def _result_key(
    backend: InferenceBackend,
    *,
    scope: str,
    batch_size: int,
) -> str:
    runtime = backend.runtime.strip().lower().replace(" ", "-")
    provider = backend.requested_provider.strip().lower().replace(" ", "-")
    thread = backend.thread_setting.strip().lower().replace(" ", "-")
    return f"{runtime}-{provider}-threads-{thread}-{scope}-batch{batch_size}"


def _model_only_runner(backend: InferenceBackend) -> Callable[[Any], Any]:
    return backend.run_prevalidated_model_input or backend.run_model_input


def _run_model_only_sample(
    backend: InferenceBackend,
    model_input: Any,
) -> tuple[Any, int]:
    backend.synchronize()
    started = time.perf_counter_ns()
    output = _model_only_runner(backend)(model_input)
    backend.synchronize()
    return output, time.perf_counter_ns() - started


def _run_end_to_end_sample(
    backend: InferenceBackend,
    raw_batch: np.ndarray,
) -> tuple[dict[str, int], np.ndarray]:
    backend.synchronize()
    total_started = time.perf_counter_ns()

    input_started = time.perf_counter_ns()
    model_input = backend.prepare_model_input(raw_batch)
    backend.synchronize()
    input_prepare_ns = time.perf_counter_ns() - input_started

    runtime_started = time.perf_counter_ns()
    output = _model_only_runner(backend)(model_input)
    backend.synchronize()
    runtime_inference_ns = time.perf_counter_ns() - runtime_started

    output_started = time.perf_counter_ns()
    materialized = backend.materialize_output(output)
    backend.synchronize()
    output_materialization_ns = time.perf_counter_ns() - output_started
    if backend.output_materialization_scope != "separate":
        output_materialization_ns = 0

    argmax_started = time.perf_counter_ns()
    actions = np.argmax(materialized, axis=1).astype(np.int64, copy=False)
    argmax_ns = time.perf_counter_ns() - argmax_started

    backend.synchronize()
    total_ns = time.perf_counter_ns() - total_started
    if actions.shape != (raw_batch.shape[0],):
        raise ValueError(
            "materialized output must contain one action row per observation: "
            f"observed={actions.shape}, expected={(raw_batch.shape[0],)}"
        )
    return (
        {
            "input_prepare_ns": input_prepare_ns,
            "runtime_inference_ns": runtime_inference_ns,
            "output_materialization_ns": output_materialization_ns,
            "argmax_ns": argmax_ns,
            "latency_ns": total_ns,
        },
        materialized,
    )


def _breakdown_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> dict[str, dict[str, float | int]]:
    return {
        field: summarize_samples(
            [int(sample[field]) for sample in samples],
            batch_size=batch_size,
        )
        for field in _BREAKDOWN_FIELDS
        if any(int(sample[field]) > 0 for sample in samples)
    }


def run_benchmark_target(
    backend: InferenceBackend,
    observations: np.ndarray,
    *,
    config: BenchmarkConfig,
    initialization_ns: int = 0,
) -> dict[str, Any]:
    """Measure both scopes for one runtime/provider/thread target."""

    _validate_observations(observations)
    raw_samples: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for batch_size in config.batch_sizes:
        raw_batch = _batch(observations, start=0, batch_size=batch_size)
        model_input = backend.prepare_model_input(raw_batch)
        backend.synchronize()

        # Model-only starts from a prepared input. Validation belongs outside
        # this timed loop so the result remains a runtime/model diagnostic.
        for _ in range(config.warmup_iterations):
            _run_model_only_sample(backend, model_input)
        model_only_samples: list[dict[str, Any]] = []
        for sample_index in range(config.iterations):
            _, latency_ns = _run_model_only_sample(backend, model_input)
            sample = {
                "result_key": _result_key(
                    backend,
                    scope="model_only",
                    batch_size=batch_size,
                ),
                "runtime": backend.runtime,
                "requested_provider": backend.requested_provider,
                "actual_provider": backend.actual_provider,
                "thread_setting": backend.thread_setting,
                "scope": "model_only",
                "batch_size": batch_size,
                "sample_index": sample_index,
                "latency_ns": int(latency_ns),
                "input_prepare_ns": 0,
                "runtime_inference_ns": int(latency_ns),
                "output_materialization_ns": 0,
                "argmax_ns": 0,
            }
            model_only_samples.append(sample)
            raw_samples.append(sample)

        model_only_result = _result_metadata(
            backend,
            scope="model_only",
            batch_size=batch_size,
            samples=model_only_samples,
            initialization_ns=initialization_ns,
        )
        results.append(model_only_result)

        # End-to-end keeps the production data path, but uses the same
        # prevalidated runtime seam after ``prepare_model_input``.  The
        # preparation step already validates the normalized CUDA tensor;
        # repeating GPU finite/range checks inside the timed runtime call
        # would measure validation overhead rather than the model path.
        for _ in range(config.warmup_iterations):
            _run_end_to_end_sample(backend, raw_batch)
        end_to_end_samples: list[dict[str, Any]] = []
        for sample_index in range(config.iterations):
            timings, _materialized = _run_end_to_end_sample(backend, raw_batch)
            sample = {
                "result_key": _result_key(
                    backend,
                    scope="end_to_end",
                    batch_size=batch_size,
                ),
                "runtime": backend.runtime,
                "requested_provider": backend.requested_provider,
                "actual_provider": backend.actual_provider,
                "thread_setting": backend.thread_setting,
                "scope": "end_to_end",
                "batch_size": batch_size,
                "sample_index": sample_index,
                **{key: int(value) for key, value in timings.items()},
            }
            end_to_end_samples.append(sample)
            raw_samples.append(sample)

        end_to_end_result = _result_metadata(
            backend,
            scope="end_to_end",
            batch_size=batch_size,
            samples=end_to_end_samples,
            initialization_ns=initialization_ns,
        )
        results.append(end_to_end_result)

    return {"summary": {"results": results}, "raw_samples": raw_samples}


def _result_metadata(
    backend: InferenceBackend,
    *,
    scope: str,
    batch_size: int,
    samples: Sequence[Mapping[str, Any]],
    initialization_ns: int,
) -> dict[str, Any]:
    latency = summarize_samples(
        [int(sample["latency_ns"]) for sample in samples],
        batch_size=batch_size,
    )
    if scope == "model_only":
        timing_semantics = (
            "prevalidated_runtime_call"
            if backend.run_prevalidated_model_input is not None
            else "runtime_call_with_backend_validation"
        )
    else:
        timing_semantics = "production_policy_decision_path"
    return {
        "result_key": _result_key(backend, scope=scope, batch_size=batch_size),
        "runtime": backend.runtime,
        "requested_provider": backend.requested_provider,
        "actual_provider": backend.actual_provider,
        "precision": backend.precision,
        "cpu_threads": backend.cpu_threads,
        "thread_setting": backend.thread_setting,
        "input_ownership": backend.input_ownership,
        "output_ownership": backend.output_ownership,
        "scope": scope,
        "timing_semantics": timing_semantics,
        "batch_size": batch_size,
        "sample_count": len(samples),
        "initialization_ns": int(initialization_ns),
        "initialization_ms": float(initialization_ns / 1_000_000.0),
        "output_materialization_scope": backend.output_materialization_scope,
        "latency": latency,
        "breakdown": _breakdown_summary(samples, batch_size=batch_size),
        "metadata": dict(backend.metadata),
    }


def write_benchmark_artifacts(
    output_dir: str | Path, payload: Mapping[str, Any]
) -> None:
    """Write summary and raw samples as separate auditable JSON artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = payload.get("summary")
    raw_samples = payload.get("raw_samples")
    if not isinstance(summary, Mapping) or not isinstance(raw_samples, Sequence):
        raise TypeError(
            "payload must contain a summary mapping and raw_samples sequence"
        )
    artifact_type = str(payload.get("artifact_type", "day24_inference_benchmark"))
    if not artifact_type.strip():
        raise ValueError("payload artifact_type must be a non-empty string")
    summary_payload = dict(summary)
    summary_payload.setdefault("schema_version", BENCHMARK_SCHEMA_VERSION)
    summary_payload["artifact_type"] = f"{artifact_type}_summary"
    summary_payload["benchmark_id"] = payload.get("benchmark_id")
    for key in (
        "generated_at_utc",
        "lineage",
        "configuration",
        "host",
        "source_observations",
    ):
        if key in payload:
            summary_payload[key] = payload[key]
    raw_payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "artifact_type": f"{artifact_type}_raw_samples",
        "benchmark_id": payload.get("benchmark_id"),
        "samples": list(raw_samples),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (destination / "raw-samples.json").write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_benchmark_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Load and minimally validate one Day 24 summary/raw artifact pair."""

    source = Path(output_dir)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((source / "raw-samples.json").read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping) or not isinstance(raw, Mapping):
        raise ValueError("benchmark artifacts must contain JSON objects")
    if summary.get("benchmark_id") != raw.get("benchmark_id"):
        raise ValueError("summary and raw benchmark ids do not match")
    samples = raw.get("samples")
    if not isinstance(samples, list):
        raise ValueError("raw benchmark artifact has no samples list")
    return {**dict(summary), "raw_samples": samples}


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkBlockedError",
    "BenchmarkConfig",
    "InferenceBackend",
    "build_benchmark_id",
    "load_benchmark_artifacts",
    "require_cuda_matrix",
    "run_benchmark_target",
    "summarize_samples",
    "write_benchmark_artifacts",
]
