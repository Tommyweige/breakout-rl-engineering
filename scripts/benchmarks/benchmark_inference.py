"""Benchmark native Day 24 inference latency for the canonical final model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from breakout_rl.artifacts import (
    repository_relative_path,
    repository_root,
    sha256_file,
)
from breakout_rl.benchmarking import (
    BenchmarkBlockedError,
    BenchmarkConfig,
    InferenceBackend,
    build_benchmark_id,
    require_cuda_matrix,
    run_benchmark_target,
    write_benchmark_artifacts,
)
from breakout_rl.inference import (
    ONNXRuntimePolicy,
    PyTorchPolicy,
    load_inference_spec,
    prepare_model_input,
)
from scripts.deployment.export_onnx import (
    _load_deployment_model,
    _source_identity,
    _validate_environment_contract,
)


DEFAULT_SOURCE_MODEL = Path("assets/day21/models/final_model/model.pt")
DEFAULT_SOURCE_METADATA = Path("assets/day21/models/final_model/metadata.json")
DEFAULT_ONNX_MODEL = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_ONNX_METADATA = Path(
    "assets/day22/models/final_model/model.onnx.metadata.json"
)
DEFAULT_SPEC = Path("configs/inference/inference_spec.json")
DEFAULT_PARITY = Path("configs/inference/parity_validation.json")
DEFAULT_PROBE_STATES = Path("assets/day22/inference/probe_states.npz")
DEFAULT_OUTPUT_ROOT = Path("assets/day24/benchmarks")


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)


def _resolve(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _parse_thread_settings(value: str) -> tuple[str, ...]:
    settings = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not settings:
        raise ValueError("--cpu-threads must contain at least one setting")
    for setting in settings:
        if setting == "default":
            continue
        try:
            parsed = int(setting)
        except ValueError as error:
            raise ValueError(
                "--cpu-threads entries must be positive integers or default"
            ) from error
        if parsed < 1:
            raise ValueError(
                "--cpu-threads entries must be positive integers or default"
            )
    if len(set(settings)) != len(settings):
        raise ValueError("--cpu-threads entries must not be duplicated")
    return settings


def _load_probe_observations(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as fixture:
        if "observations" not in fixture.files:
            raise ValueError(f"{path}: fixture has no observations array")
        observations = np.ascontiguousarray(fixture["observations"])
    if observations.dtype != np.dtype("uint8"):
        raise ValueError(f"{path}: observations must have dtype uint8")
    if observations.ndim != 4 or tuple(observations.shape[1:]) != (4, 84, 84):
        raise ValueError(f"{path}: observations must have shape (N, 4, 84, 84)")
    if observations.shape[0] < 1:
        raise ValueError(f"{path}: observations must not be empty")
    return observations


def _verify_lineage(
    *,
    root: Path,
    source_model: Path,
    source_metadata: Path,
    onnx_model: Path,
    onnx_metadata: Path,
    spec_path: Path,
    parity_path: Path,
    probe_states: Path,
) -> tuple[Any, dict[str, Any]]:
    spec = load_inference_spec(spec_path)
    _validate_environment_contract(spec, root=root)
    identity = _source_identity(source_model, source_metadata, root=root)

    onnx_meta = _json_object(onnx_metadata)
    observed_onnx_hash = sha256_file(onnx_model)
    if onnx_meta.get("model_sha256") != observed_onnx_hash:
        raise ValueError("Day 22 ONNX hash does not match its metadata")
    if onnx_meta.get("source_model_sha256") != identity["model_sha256"]:
        raise ValueError("Day 22 ONNX artifact is not derived from Day 21 model")
    if onnx_meta.get("source_checkpoint_sha256") != identity[
        "source_checkpoint_sha256"
    ]:
        raise ValueError("Day 22 ONNX checkpoint provenance does not match Day 21")

    parity = _json_object(parity_path)
    lineage = parity.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("Day 23 parity config has no lineage object")
    expected_lineage = {
        "canonical_source_model_sha256": identity["model_sha256"],
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "onnx_model_sha256": observed_onnx_hash,
        "inference_spec_sha256": sha256_file(spec_path, normalize_text=True),
        "probe_states_sha256": sha256_file(probe_states),
    }
    for field, expected in expected_lineage.items():
        if lineage.get(field) != expected:
            raise ValueError(
                f"Day 23 lineage field {field} does not match the loaded artifact"
            )

    return spec, {
        "canonical_source_model": {
            "path": repository_relative_path(source_model, root=root),
            "sha256": identity["model_sha256"],
            "source_run_id": identity["source_run_id"],
            "source_stage": identity["source_stage"],
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "source_checkpoint_step": identity["source_checkpoint_step"],
        },
        "canonical_source_metadata": {
            "path": repository_relative_path(source_metadata, root=root),
            "sha256": sha256_file(source_metadata),
        },
        "onnx_model": {
            "path": repository_relative_path(onnx_model, root=root),
            "sha256": observed_onnx_hash,
        },
        "onnx_metadata": {
            "path": repository_relative_path(onnx_metadata, root=root),
            "sha256": sha256_file(onnx_metadata),
        },
        "inference_spec": {
            "path": repository_relative_path(spec_path, root=root),
            "sha256": sha256_file(spec_path, normalize_text=True),
            "contract_id": spec.contract_id,
        },
        "parity_validation": {
            "path": repository_relative_path(parity_path, root=root),
            "sha256": sha256_file(parity_path),
        },
        "probe_states": {
            "path": repository_relative_path(probe_states, root=root),
            "sha256": sha256_file(probe_states),
        },
    }


def _sync_for(device: torch.device):
    if device.type != "cuda":
        return lambda: None

    def synchronize() -> None:
        torch.cuda.synchronize(device)

    return synchronize


def _pytorch_backend(
    policy: PyTorchPolicy,
    *,
    requested_provider: str,
    device: torch.device,
    thread_setting: str,
    cpu_threads: int,
) -> InferenceBackend:
    actual_provider = f"torch.{device.type}"
    if device.index is not None:
        actual_provider += f":{device.index}"

    def run_prevalidated_model_input(model_input: torch.Tensor) -> torch.Tensor:
        # The model-only diagnostic validates/prepares its input before timing.
        # Keep the timed body to the actual PyTorch forward call.
        policy.model.eval()
        with torch.inference_mode():
            return policy.model(model_input)

    return InferenceBackend(
        runtime="PyTorch",
        requested_provider=requested_provider,
        actual_provider=actual_provider,
        precision="float32",
        cpu_threads=cpu_threads,
        thread_setting=thread_setting,
        input_ownership=f"numpy.uint8/cpu → torch.float32/{device}",
        output_ownership=f"torch.float32/{device} → numpy.float32/cpu",
        prepare_model_input=lambda observations: prepare_model_input(
            observations,
            device=device,
            spec=policy.spec,
        ),
        run_model_input=policy.predict_model_input,
        run_prevalidated_model_input=run_prevalidated_model_input,
        materialize_output=lambda output: np.ascontiguousarray(
            output.detach().cpu().numpy(),
            dtype=np.float32,
        ),
        synchronize=_sync_for(device),
        metadata={
            "framework": "PyTorch",
            "device": str(device),
            "pytorch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cpu_thread_semantics": "torch.set_num_threads",
            "model_only_input_validation": "outside_timed_loop",
        },
    )


def _onnx_backend(
    policy: ONNXRuntimePolicy,
    *,
    requested_provider: str,
    thread_setting: str,
    cpu_threads: int,
) -> InferenceBackend:
    runtime_metadata = policy.runtime_metadata
    actual_provider = str(runtime_metadata["actual_provider"])
    session_inputs = list(policy.session.get_inputs())
    session_outputs = list(policy.session.get_outputs())
    if len(session_inputs) != 1 or len(session_outputs) != 1:
        raise ValueError("Day 24 ONNX benchmark expects exactly one model input/output")
    input_name = str(session_inputs[0].name)
    output_name = str(session_outputs[0].name)

    def run_prevalidated_model_input(model_input: np.ndarray) -> np.ndarray:
        # session.run is intentionally called directly here. The input was
        # prepared/validated before timing; production policy validation stays
        # in the end-to-end path through policy.predict_model_input().
        outputs = policy.session.run([output_name], {input_name: model_input})
        return np.asarray(outputs[0])

    return InferenceBackend(
        runtime="ONNX Runtime",
        requested_provider=requested_provider,
        actual_provider=actual_provider,
        precision="float32",
        cpu_threads=cpu_threads,
        thread_setting=thread_setting,
        input_ownership="numpy.uint8/cpu → numpy.float32/cpu",
        output_ownership="numpy.float32/cpu",
        prepare_model_input=lambda observations: np.ascontiguousarray(
            prepare_model_input(observations, device="cpu", spec=policy.spec).numpy(),
            dtype=np.float32,
        ),
        run_model_input=policy.predict_model_input,
        run_prevalidated_model_input=run_prevalidated_model_input,
        materialize_output=lambda output: np.ascontiguousarray(
            output,
            dtype=np.float32,
        ),
        output_materialization_scope="included_in_runtime_api",
        metadata={
            "framework": "ONNX Runtime",
            "onnxruntime_version": runtime_metadata.get("onnxruntime_version"),
            "onnxruntime_device": runtime_metadata.get("onnxruntime_device"),
            "active_providers": runtime_metadata.get("active_providers"),
            "available_providers": runtime_metadata.get("available_providers"),
            "session_provider_options": runtime_metadata.get(
                "session_provider_options"
            ),
            "graph_assignment": runtime_metadata.get("graph_assignment"),
            "intra_op_num_threads": runtime_metadata.get("intra_op_num_threads"),
            "inter_op_num_threads": runtime_metadata.get("inter_op_num_threads"),
            "cpu_thread_semantics": "onnxruntime SessionOptions.intra_op_num_threads",
            "cuda_version": runtime_metadata.get("cuda_version"),
            "gpu_model": runtime_metadata.get("gpu_model"),
            "model_only_input_validation": "outside_timed_loop",
        },
    )


def _host_metadata(*, default_torch_threads: int, ort_providers: Sequence[str]) -> dict[str, Any]:
    gpu_model = None
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": platform.processor() or platform.uname().processor or "unknown",
        "cpu_logical_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "gpu_model": gpu_model,
        "gpu_device_index": 0,
        "torch_default_threads": default_torch_threads,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "onnxruntime_available_providers": list(ort_providers),
    }


def _run_target(
    *,
    backend: InferenceBackend,
    observations: np.ndarray,
    config: BenchmarkConfig,
    initialization_ns: int,
    results: list[dict[str, Any]],
    raw_samples: list[dict[str, Any]],
) -> None:
    measured = run_benchmark_target(
        backend,
        observations,
        config=config,
        initialization_ns=initialization_ns,
    )
    results.extend(measured["summary"]["results"])
    raw_samples.extend(measured["raw_samples"])


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    source_model = _resolve(root, args.source_model)
    source_metadata = _resolve(root, args.source_metadata)
    onnx_model = _resolve(root, args.onnx_model)
    onnx_metadata = _resolve(root, args.onnx_metadata)
    spec_path = _resolve(root, args.spec)
    parity_path = _resolve(root, args.parity)
    probe_states = _resolve(root, args.probe_states)
    observations = _load_probe_observations(probe_states)

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise BenchmarkBlockedError(
            "Day 24 is blocked: ONNX Runtime CUDAExecutionProvider cannot be "
            "verified because the pinned ONNX Runtime package is unavailable."
        ) from error

    require_cuda_matrix(
        torch_cuda_available=bool(torch.cuda.is_available()),
        onnxruntime_providers=ort.get_available_providers(),
    )
    spec, lineage = _verify_lineage(
        root=root,
        source_model=source_model,
        source_metadata=source_metadata,
        onnx_model=onnx_model,
        onnx_metadata=onnx_metadata,
        spec_path=spec_path,
        parity_path=parity_path,
        probe_states=probe_states,
    )

    cpu_thread_settings = _parse_thread_settings(args.cpu_threads)
    default_torch_threads = torch.get_num_threads()
    actual_cpu_threads = tuple(
        default_torch_threads if setting == "default" else int(setting)
        for setting in cpu_thread_settings
    )
    config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        iterations=args.iterations,
        batch_sizes=tuple(args.batch_sizes),
    )
    identity = {
        "lineage": lineage,
        "warmup_iterations": config.warmup_iterations,
        "iterations": config.iterations,
        "batch_sizes": config.batch_sizes,
        "cpu_thread_settings": cpu_thread_settings,
        "device_index": args.device_index,
        "model_only_semantics": "prevalidated_runtime_call",
    }
    benchmark_id = args.benchmark_id or build_benchmark_id(identity)
    output_dir = (root / args.output_root / benchmark_id).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"benchmark directory already contains artifacts: {output_dir}; "
            "choose --benchmark-id, --output-root, or --force"
        )

    results: list[dict[str, Any]] = []
    raw_samples: list[dict[str, Any]] = []
    initializations: dict[str, int] = {}
    pytorch_cuda_memory: dict[str, int] = {}

    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(default_torch_threads)
        started = time.perf_counter_ns()
        cpu_model, _payload, _model_config = _load_deployment_model(
            source_model,
            device=torch.device("cpu"),
            identity=_source_identity(source_model, source_metadata, root=root),
            spec=spec,
        )
        cpu_load_ns = time.perf_counter_ns() - started
        cpu_policy = PyTorchPolicy(cpu_model, device="cpu", spec=spec)
        initializations["pytorch_cpu"] = cpu_load_ns

        for setting, threads in zip(cpu_thread_settings, actual_cpu_threads, strict=True):
            torch.set_num_threads(threads)
            _run_target(
                backend=_pytorch_backend(
                    cpu_policy,
                    requested_provider="cpu",
                    device=torch.device("cpu"),
                    thread_setting=setting,
                    cpu_threads=threads,
                ),
                observations=observations,
                config=config,
                initialization_ns=cpu_load_ns,
                results=results,
                raw_samples=raw_samples,
            )

        torch.set_num_threads(default_torch_threads)
        started = time.perf_counter_ns()
        cuda_device = torch.device(f"cuda:{args.device_index}")
        cuda_model, _payload, _model_config = _load_deployment_model(
            source_model,
            device=cuda_device,
            identity=_source_identity(source_model, source_metadata, root=root),
            spec=spec,
        )
        cuda_load_ns = time.perf_counter_ns() - started
        cuda_policy = PyTorchPolicy(cuda_model, device=cuda_device, spec=spec)
        initializations["pytorch_cuda"] = cuda_load_ns
        torch.cuda.reset_peak_memory_stats(cuda_device)
        _run_target(
            backend=_pytorch_backend(
                cuda_policy,
                requested_provider="cuda",
                device=cuda_device,
                thread_setting="default",
                cpu_threads=default_torch_threads,
            ),
            observations=observations,
            config=config,
            initialization_ns=cuda_load_ns,
            results=results,
            raw_samples=raw_samples,
        )
        pytorch_cuda_memory = {
            "current_allocated_bytes": int(torch.cuda.memory_allocated(cuda_device)),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(cuda_device)
            ),
            "current_reserved_bytes": int(torch.cuda.memory_reserved(cuda_device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_device)),
        }

        for setting in cpu_thread_settings:
            threads = 0 if setting == "default" else int(setting)
            started = time.perf_counter_ns()
            ort_policy = ONNXRuntimePolicy(
                onnx_model,
                provider="cpu",
                intra_op_num_threads=None if setting == "default" else threads,
                spec=spec,
            )
            session_ns = time.perf_counter_ns() - started
            initializations[f"onnx_cpu_{setting}"] = session_ns
            _run_target(
                backend=_onnx_backend(
                    ort_policy,
                    requested_provider="cpu",
                    thread_setting=setting,
                    cpu_threads=0 if setting == "default" else threads,
                ),
                observations=observations,
                config=config,
                initialization_ns=session_ns,
                results=results,
                raw_samples=raw_samples,
            )

        started = time.perf_counter_ns()
        ort_cuda_policy = ONNXRuntimePolicy(
            onnx_model,
            provider="cuda",
            device_index=args.device_index,
            spec=spec,
        )
        ort_cuda_session_ns = time.perf_counter_ns() - started
        initializations["onnx_cuda"] = ort_cuda_session_ns
        _run_target(
            backend=_onnx_backend(
                ort_cuda_policy,
                requested_provider="cuda",
                thread_setting="default",
                cpu_threads=0,
            ),
            observations=observations,
            config=config,
            initialization_ns=ort_cuda_session_ns,
            results=results,
            raw_samples=raw_samples,
        )
    finally:
        torch.set_num_threads(original_threads)

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    host = _host_metadata(
        default_torch_threads=default_torch_threads,
        ort_providers=ort.get_available_providers(),
    )
    host["pytorch_cuda_memory"] = pytorch_cuda_memory
    payload = {
        "artifact_type": "day24_inference_benchmark",
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "generated_at_utc": generated_at,
        "lineage": lineage,
        "configuration": {
            "warmup_iterations": config.warmup_iterations,
            "iterations": config.iterations,
            "batch_sizes": list(config.batch_sizes),
            "primary_batch_size": 1,
            "scopes": ["model_only", "end_to_end"],
            "model_only_semantics": "prevalidated_runtime_call",
            "end_to_end_semantics": "production_policy_decision_path",
            "cpu_thread_settings": list(cpu_thread_settings),
            "actual_cpu_thread_counts": list(actual_cpu_threads),
            "device_index": args.device_index,
            "precision": "float32",
            "initialization_ns_by_target": initializations,
        },
        "host": host,
        "source_observations": {
            "path": repository_relative_path(probe_states, root=root),
            "sha256": sha256_file(probe_states),
            "count": int(observations.shape[0]),
            "shape": list(observations.shape[1:]),
            "dtype": str(observations.dtype),
        },
        "summary": {"results": results},
        "raw_samples": raw_samples,
    }
    write_benchmark_artifacts(output_dir, payload)
    payload["output_dir"] = repository_relative_path(output_dir, root=root)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--onnx-metadata", type=Path, default=DEFAULT_ONNX_METADATA)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-id")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--cpu-threads", default="1,2,4,default")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_benchmark(args)
    except (
        BenchmarkBlockedError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Day 24 inference benchmark failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "benchmark_id": payload["benchmark_id"],
                "output_dir": payload["output_dir"],
                "result_count": len(payload["summary"]["results"]),
                "sample_count": len(payload["raw_samples"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
