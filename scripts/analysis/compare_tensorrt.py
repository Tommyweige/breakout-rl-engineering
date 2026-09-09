"""Run the native-NVIDIA Day 26 TensorRT correctness and latency experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from breakout_env import make_breakout_env
from breakout_rl.artifacts import repository_relative_path, repository_root, sha256_file
from breakout_rl.benchmarking import (
    BenchmarkConfig,
    InferenceBackend,
    run_benchmark_target,
    write_benchmark_artifacts,
)
from breakout_rl.deployment import (
    load_deployment_model,
    source_identity,
    validate_environment_contract,
)
from breakout_rl.evaluation import evaluate_policy, load_evaluation_config
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.inference import (
    ONNXRuntimePolicy,
    load_inference_spec,
    prepare_model_input,
)
from breakout_rl.onnx_parity import compare_q_values
from breakout_rl.precision import (
    PrecisionThresholds,
    compare_evaluation_parity,
    require_cuda_precision_matrix,
)
from breakout_rl.tensorrt import (
    TENSORRT_STATUS_CLI_ONLY,
    TENSORRT_STATUS_READY,
    TensorRTBlockedError,
    TensorRTSession,
    require_gpu_baseline,
)


DEFAULT_CONFIG = Path("configs/inference/tensorrt_experiment.json")
DEFAULT_PREFLIGHT = Path("assets/day26/tensorrt-preflight.json")
DEFAULT_SOURCE_MODEL = Path("assets/day21/models/final_model/model.pt")
DEFAULT_SOURCE_METADATA = Path("assets/day21/models/final_model/metadata.json")
DEFAULT_ONNX_FP32 = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_ONNX_FP16 = Path("assets/day25/models/model.fp16.onnx")
DEFAULT_DAY25_COMPARISON = Path("assets/day25/precision-comparison.json")
DEFAULT_PARITY_CONFIG = Path("configs/inference/parity_validation.json")
DEFAULT_SPEC = Path("configs/inference/inference_spec.json")
DEFAULT_PROBE_STATES = Path("assets/day22/inference/probe_states.npz")
DEFAULT_REFERENCE = Path("assets/day22/inference/pytorch_reference.npz")
DEFAULT_CONTRACT = Path("configs/eval/breakout_contract_v2.json")
DEFAULT_EVALUATION_CONFIG = Path("configs/eval/breakout_eval.json")
DEFAULT_OUTPUT = Path("assets/day26/tensorrt-comparison.json")
DEFAULT_REPORT = Path("reports/day26-tensorrt-experiment.md")
DEFAULT_BENCHMARK_ROOT = Path("assets/day26/benchmarks")
DEFAULT_EVALUATIONS_ROOT = Path("assets/day26/evaluations")
DEFAULT_ENGINE_FP32 = Path("assets/day26/models/model.v2.fp32.engine")
DEFAULT_ENGINE_FP32_METADATA = Path(
    "assets/day26/models/model.v2.fp32.engine.metadata.json"
)
DEFAULT_ENGINE_FP16 = Path("assets/day26/models/model.v2.fp16.engine")
DEFAULT_ENGINE_FP16_METADATA = Path(
    "assets/day26/models/model.v2.fp16.engine.metadata.json"
)


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def _relative(root: Path, path: Path) -> str:
    return repository_relative_path(path, root=root)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _load_config(
    path: Path,
) -> tuple[
    dict[str, Any],
    PrecisionThresholds,
    PrecisionThresholds,
    BenchmarkConfig,
    float,
    int,
]:
    config = _json_object(path)
    _equal("TensorRT config schema_version", config.get("schema_version"), 1)
    _equal(
        "TensorRT config artifact_type",
        config.get("artifact_type"),
        "day26_tensorrt_experiment",
    )
    parity = _mapping(config.get("parity"), label="parity")
    raw_thresholds = _mapping(parity.get("thresholds"), label="parity.thresholds")
    fp32_thresholds = PrecisionThresholds.from_mapping(
        _mapping(
            raw_thresholds.get("tensorrt_fp32"), label="parity.thresholds.tensorrt_fp32"
        )
    )
    fp16_thresholds = PrecisionThresholds.from_mapping(
        _mapping(
            raw_thresholds.get("tensorrt_fp16"), label="parity.thresholds.tensorrt_fp16"
        )
    )
    relative_epsilon = float(parity.get("relative_error_epsilon"))
    if not np.isfinite(relative_epsilon) or relative_epsilon <= 0.0:
        raise ValueError("parity.relative_error_epsilon must be positive and finite")
    build = _mapping(config.get("build"), label="build")
    _equal("build network typing", build.get("network_typing"), "strongly_typed")
    _equal("build TF32 policy", build.get("tf32_enabled"), False)
    required_headroom_bytes = int(build.get("required_headroom_bytes"))
    if required_headroom_bytes < 0:
        raise ValueError("build.required_headroom_bytes must not be negative")
    benchmark = _mapping(config.get("benchmark"), label="benchmark")
    batch_sizes = tuple(int(value) for value in benchmark.get("batch_sizes", ()))
    benchmark_config = BenchmarkConfig(
        warmup_iterations=int(benchmark.get("warmup_iterations")),
        iterations=int(benchmark.get("iterations")),
        batch_sizes=batch_sizes,
    )
    _equal("primary benchmark batch size", benchmark.get("primary_batch_size"), 1)
    _equal(
        "benchmark scopes",
        tuple(str(value) for value in benchmark.get("scopes", ())),
        ("model_only", "end_to_end"),
    )
    evaluation = _mapping(config.get("evaluation"), label="evaluation")
    eval_thresholds = _mapping(
        evaluation.get("thresholds"), label="evaluation.thresholds"
    )
    for field in (
        "action_agreement_rate",
        "episode_return_match_rate",
        "episode_length_match_rate",
    ):
        value = float(eval_thresholds.get(field))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"evaluation.thresholds.{field} must be between 0 and 1")
    browser = _mapping(config.get("browser_decision"), label="browser_decision")
    _equal("browser baseline precision", browser.get("baseline_precision"), "float32")
    _equal("TensorRT web serving policy", browser.get("tensorrt_is_web_serving"), False)
    return (
        config,
        fp32_thresholds,
        fp16_thresholds,
        benchmark_config,
        relative_epsilon,
        required_headroom_bytes,
    )


def _load_cuda_baseline_thresholds(path: Path) -> PrecisionThresholds:
    payload = _json_object(path)
    thresholds = _mapping(
        payload.get("thresholds"), label="parity validation thresholds"
    )
    return PrecisionThresholds.from_mapping(
        _mapping(thresholds.get("cuda"), label="parity validation thresholds.cuda")
    )


def _load_observations(
    path: Path, *, expected_shape: tuple[int, int, int]
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        observations = np.ascontiguousarray(archive["observations"])
    if (
        observations.dtype != np.dtype("uint8")
        or observations.ndim != 4
        or tuple(observations.shape[1:]) != expected_shape
        or observations.shape[0] < 1
    ):
        raise ValueError(
            "probe states must be non-empty uint8 (N, 4, 84, 84) observations"
        )
    return observations


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        observations = np.ascontiguousarray(archive["observations"])
        q_values = np.ascontiguousarray(archive["q_values"])
        actions = np.ascontiguousarray(archive["greedy_actions"])
    if observations.dtype != np.dtype("uint8") or q_values.dtype != np.dtype("float32"):
        raise ValueError("Day 22 reference fixture has unexpected dtypes")
    if actions.dtype != np.dtype("int64"):
        raise ValueError("Day 22 reference actions must have dtype int64")
    return observations, q_values, actions


def _verify_engine_metadata(
    *,
    root: Path,
    engine: Path,
    metadata_path: Path,
    precision: str,
    preflight: Path,
    expected_onnx_hash: str,
) -> dict[str, Any]:
    metadata = _json_object(metadata_path)
    _equal(
        "engine metadata artifact_type",
        metadata.get("artifact_type"),
        "day26_tensorrt_engine",
    )
    _equal("engine metadata precision", metadata.get("precision"), precision)
    _equal(
        "engine metadata network typing",
        metadata.get("network_typing"),
        "strongly_typed",
    )
    _equal("engine metadata TF32 policy", metadata.get("tf32_enabled"), False)
    _equal("engine metadata path", metadata.get("engine_path"), _relative(root, engine))
    _equal("engine SHA256", metadata.get("engine_sha256"), sha256_file(engine))
    software = _mapping(metadata.get("software"), label=f"{precision} engine software")
    _equal(
        f"{precision} engine preflight SHA256",
        software.get("preflight_sha256"),
        sha256_file(preflight),
    )
    lineage = _mapping(metadata.get("lineage"), label=f"{precision} engine lineage")
    onnx = _mapping(lineage.get("onnx"), label=f"{precision} engine ONNX lineage")
    _equal(f"{precision} engine ONNX SHA256", onnx.get("sha256"), expected_onnx_hash)
    compatibility = _mapping(
        metadata.get("engine_compatibility"),
        label=f"{precision} engine compatibility",
    )
    _equal(
        f"{precision} engine hardware specificity",
        compatibility.get("hardware_specific"),
        True,
    )
    return {
        "path": _relative(root, engine),
        "sha256": sha256_file(engine),
        "metadata_path": _relative(root, metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "payload": metadata,
    }


def _verify_lineage(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    preflight: Mapping[str, Any],
    preflight_path: Path,
    source_model: Path,
    source_metadata: Path,
    onnx_fp32: Path,
    onnx_fp16: Path,
    day25_comparison: Path,
    parity_config_path: Path,
    spec_path: Path,
    probe_states: Path,
    reference: Path,
    contract_path: Path,
    evaluation_config_path: Path,
    engine_fp32: Path,
    engine_fp32_metadata: Path,
    engine_fp16: Path,
    engine_fp16_metadata: Path,
) -> tuple[Any, Any, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _mapping(config.get("source"), label="source")
    identity = source_identity(source_model, source_metadata, root=root)
    _equal(
        "Day 21 model SHA256",
        identity["model_sha256"],
        source.get("canonical_source_model_sha256"),
    )
    _equal(
        "Day 21 checkpoint SHA256",
        identity["source_checkpoint_sha256"],
        source.get("source_checkpoint_sha256"),
    )
    _equal(
        "Day 21 checkpoint step",
        identity["source_checkpoint_step"],
        source.get("source_checkpoint_step"),
    )
    _equal(
        "Day 22 FP32 ONNX SHA256",
        sha256_file(onnx_fp32),
        source.get("onnx_fp32_sha256"),
    )
    _equal(
        "Day 25 FP16 ONNX SHA256",
        sha256_file(onnx_fp16),
        source.get("onnx_fp16_sha256"),
    )
    _equal(
        "Day 25 comparison SHA256",
        sha256_file(day25_comparison),
        source.get("day25_comparison_sha256"),
    )
    day25 = _json_object(day25_comparison)
    _equal(
        "Day 25 comparison type",
        day25.get("artifact_type"),
        "day25_precision_comparison",
    )
    _equal("Day 25 comparison status", day25.get("status"), "completed")
    parity_config = _json_object(parity_config_path)
    _equal(
        "Day 23 parity config type",
        parity_config.get("artifact_type"),
        "day23_onnx_runtime_parity_validation",
    )

    spec = load_inference_spec(spec_path)
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    validate_environment_contract(spec, root=root)
    _equal(
        "inference spec Contract v2 hash",
        sha256_file(contract_path),
        spec.environment_contract_sha256,
    )
    _equal(
        "evaluation environment id",
        load_evaluation_config(evaluation_config_path).environment_id,
        contract.environment_id,
    )
    evaluation_config = load_evaluation_config(evaluation_config_path)
    _equal("evaluation epsilon", evaluation_config.epsilon, contract.evaluation_epsilon)
    _equal(
        "evaluation episode seeds",
        tuple(contract.concrete_episode_seeds),
        tuple(
            seed + offset
            for seed in evaluation_config.seeds
            for offset in range(evaluation_config.episodes_per_seed)
        ),
    )

    observations = _load_observations(
        probe_states,
        expected_shape=spec.source_observation_shape,
    )
    fixture_observations, fixture_q_values, fixture_actions = _load_reference(reference)
    if not np.array_equal(observations, fixture_observations):
        raise ValueError("Day 22 probe states differ from the Day 22 reference fixture")
    expected_source_paths = {
        "canonical_source_model_path": _relative(root, source_model),
        "onnx_fp32_path": _relative(root, onnx_fp32),
        "onnx_fp16_path": _relative(root, onnx_fp16),
        "day25_comparison_path": _relative(root, day25_comparison),
        "inference_spec_path": _relative(root, spec_path),
        "probe_states_path": _relative(root, probe_states),
        "pytorch_reference_path": _relative(root, reference),
    }
    for key, expected in expected_source_paths.items():
        if key.endswith("_path") and key in source:
            _equal(f"configured source.{key}", source.get(key), expected)
    preflight_lineage = _mapping(preflight.get("lineage"), label="preflight.lineage")
    preflight_config = _mapping(
        preflight_lineage.get("config"), label="preflight.lineage.config"
    )
    _equal(
        "preflight config path",
        preflight_config.get("path"),
        _relative(root, config_path),
    )
    _equal(
        "preflight config SHA256",
        preflight_config.get("sha256"),
        sha256_file(config_path, normalize_text=True),
    )
    preflight_contract = _mapping(
        preflight_lineage.get("environment_contract"),
        label="preflight.lineage.environment_contract",
    )
    _equal(
        "preflight Contract v2 path",
        preflight_contract.get("path"),
        spec.environment_contract_path,
    )
    _equal(
        "preflight Contract v2 hash",
        preflight_contract.get("sha256"),
        spec.environment_contract_sha256,
    )
    preflight_capabilities = _mapping(
        preflight.get("capabilities"), label="preflight.capabilities"
    )
    _equal(
        "preflight network typing",
        preflight_capabilities.get("network_typing"),
        "strongly_typed",
    )
    _equal(
        "preflight TF32 policy",
        preflight_capabilities.get("tf32_enabled"),
        False,
    )
    if preflight.get("status") == TENSORRT_STATUS_READY:
        _equal(
            "preflight strongly typed network capability",
            preflight_capabilities.get("strongly_typed_network_available"),
            True,
        )
    # The preflight is allowed to change its timestamp; engine metadata binds
    # the exact preflight file, while the config hash above binds the inputs.
    engine_lineage = {
        "canonical_source_model": {
            "path": _relative(root, source_model),
            "sha256": identity["model_sha256"],
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "source_checkpoint_step": identity["source_checkpoint_step"],
        },
        "onnx_fp32": {
            "path": _relative(root, onnx_fp32),
            "sha256": sha256_file(onnx_fp32),
        },
        "onnx_fp16": {
            "path": _relative(root, onnx_fp16),
            "sha256": sha256_file(onnx_fp16),
        },
        "day25_comparison": {
            "path": _relative(root, day25_comparison),
            "sha256": sha256_file(day25_comparison),
        },
        "day23_parity_config": {
            "path": _relative(root, parity_config_path),
            "sha256": sha256_file(parity_config_path, normalize_text=True),
        },
        "inference_spec": {
            "path": _relative(root, spec_path),
            "sha256": sha256_file(spec_path, normalize_text=True),
        },
        "environment_contract": {
            "path": _relative(root, contract_path),
            "sha256": sha256_file(contract_path),
            "contract_id": contract.contract_id,
        },
        "evaluation_config": {
            "path": _relative(root, evaluation_config_path),
            "sha256": sha256_file(evaluation_config_path),
        },
        "probe_states": {
            "path": _relative(root, probe_states),
            "sha256": sha256_file(probe_states),
            "count": int(observations.shape[0]),
        },
        "pytorch_reference": {
            "path": _relative(root, reference),
            "sha256": sha256_file(reference),
        },
        "config": {
            "path": _relative(root, config_path),
            "sha256": sha256_file(config_path, normalize_text=True),
        },
        "preflight": {
            "path": _relative(root, preflight_path),
            "sha256": sha256_file(preflight_path),
        },
    }
    engine_fp32_info = _verify_engine_metadata(
        root=root,
        engine=engine_fp32,
        metadata_path=engine_fp32_metadata,
        precision="float32",
        preflight=preflight_path,
        expected_onnx_hash=sha256_file(onnx_fp32),
    )
    engine_fp16_info = _verify_engine_metadata(
        root=root,
        engine=engine_fp16,
        metadata_path=engine_fp16_metadata,
        precision="float16",
        preflight=preflight_path,
        expected_onnx_hash=sha256_file(onnx_fp16),
    )
    engine_lineage["engine_fp32"] = {
        "path": engine_fp32_info["path"],
        "sha256": engine_fp32_info["sha256"],
        "metadata_path": engine_fp32_info["metadata_path"],
        "metadata_sha256": engine_fp32_info["metadata_sha256"],
    }
    engine_lineage["engine_fp16"] = {
        "path": engine_fp16_info["path"],
        "sha256": engine_fp16_info["sha256"],
        "metadata_path": engine_fp16_info["metadata_path"],
        "metadata_sha256": engine_fp16_info["metadata_sha256"],
    }
    return (
        spec,
        contract,
        observations,
        fixture_q_values,
        fixture_actions,
        {
            "identity": identity,
            "lineage": engine_lineage,
            "evaluation_config": evaluation_config,
            "engine_fp32": engine_fp32_info,
            "engine_fp16": engine_fp16_info,
        },
    )


def _sync_for(device: torch.device):
    def synchronize() -> None:
        torch.cuda.synchronize(device)

    return synchronize


def _validate_q_values(values: Any, *, count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype not in (
        np.dtype("float16"),
        np.dtype("float32"),
        np.dtype("float64"),
    ):
        raise TypeError(f"Q-values must be floating point, observed {array.dtype}")
    if array.shape != (count, 4) or not np.isfinite(array).all():
        raise ValueError(f"Q-values must be finite with shape ({count}, 4)")
    return np.ascontiguousarray(array, dtype=np.float32)


def _run_pytorch_reference(
    model: nn.Module,
    observations: np.ndarray,
    *,
    spec: Any,
    device: torch.device,
) -> np.ndarray:
    model_input = prepare_model_input(observations, device=device, spec=spec)
    with torch.inference_mode():
        output = model(model_input)
    torch.cuda.synchronize(device)
    if not isinstance(output, torch.Tensor):
        raise TypeError("PyTorch FP32 baseline did not return a tensor")
    return _validate_q_values(output.detach().cpu().numpy(), count=len(observations))


def _compare(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    thresholds: PrecisionThresholds,
    label: str,
    relative_epsilon: float,
) -> dict[str, Any]:
    metrics = compare_q_values(
        reference,
        candidate,
        sample_ids=range(len(reference)),
        relative_epsilon=relative_epsilon,
    )
    checks = thresholds.check(metrics)
    return {
        "label": label,
        "metrics": metrics,
        "thresholds": checks,
        "passed": bool(checks["passed"]),
    }


def _run_tensorrt_probe(
    session: TensorRTSession,
    model_input: np.ndarray,
) -> np.ndarray:
    """Run a probe set in profile-sized chunks while preserving row order."""

    max_batch = int(session.profile_shapes[2][0])
    outputs = [
        session.predict_model_input(model_input[start : start + max_batch])
        for start in range(0, len(model_input), max_batch)
    ]
    return np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.float32)


def _torch_backend(
    model: nn.Module,
    *,
    spec: Any,
    device: torch.device,
    initialization_ns: int,
) -> InferenceBackend:
    def prepare(observations: np.ndarray) -> torch.Tensor:
        return prepare_model_input(observations, device=device, spec=spec)

    def run(model_input: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return model(model_input)

    return InferenceBackend(
        runtime="PyTorch",
        requested_provider="cuda",
        actual_provider=f"torch.cuda:{device.index or 0}",
        precision="float32",
        cpu_threads=None,
        thread_setting="default",
        input_ownership=f"numpy.uint8/cpu → torch.float32/cuda:{device.index or 0}",
        output_ownership=f"torch.float32/cuda:{device.index or 0} → numpy.float32/cpu",
        prepare_model_input=prepare,
        run_model_input=run,
        run_prevalidated_model_input=run,
        materialize_output=lambda output: np.ascontiguousarray(
            output.detach().to(device="cpu", dtype=torch.float32).numpy(),
            dtype=np.float32,
        ),
        synchronize=_sync_for(device),
        metadata={
            "framework": "PyTorch",
            "device": str(device),
            "precision": "float32",
            "pytorch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "initialization_ns": initialization_ns,
        },
    )


def _onnx_backend(
    policy: ONNXRuntimePolicy,
    *,
    initialization_ns: int,
) -> InferenceBackend:
    inputs = list(policy.session.get_inputs())
    outputs = list(policy.session.get_outputs())
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Day 26 ONNX baseline expects exactly one input and output")
    input_name = str(inputs[0].name)
    output_name = str(outputs[0].name)

    def prepare(observations: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            prepare_model_input(observations, device="cpu", spec=policy.spec).numpy(),
            dtype=np.float32,
        )

    def run_prevalidated(model_input: np.ndarray) -> np.ndarray:
        values = policy.session.run([output_name], {input_name: model_input})
        return np.asarray(values[0])

    runtime = policy.runtime_metadata
    return InferenceBackend(
        runtime="ONNX Runtime",
        requested_provider="cuda",
        actual_provider=str(runtime["actual_provider"]),
        precision="float32",
        cpu_threads=0,
        thread_setting="default",
        input_ownership="numpy.uint8/cpu → numpy.float32/cpu → CUDAExecutionProvider",
        output_ownership="numpy.float32/cpu",
        prepare_model_input=prepare,
        run_model_input=policy.predict_model_input,
        run_prevalidated_model_input=run_prevalidated,
        materialize_output=lambda output: np.ascontiguousarray(
            output, dtype=np.float32
        ),
        synchronize=lambda: None,
        output_materialization_scope="included_in_runtime_api",
        metadata={
            "framework": "ONNX Runtime",
            "precision": "float32",
            "io_policy": "float32 input / float32 output",
            "onnxruntime_version": runtime.get("onnxruntime_version"),
            "active_providers": runtime.get("active_providers"),
            "available_providers": runtime.get("available_providers"),
            "graph_assignment": runtime.get("graph_assignment"),
            "session_provider_options": runtime.get("session_provider_options"),
            "cuda_version": runtime.get("cuda_version"),
            "gpu_model": runtime.get("gpu_model"),
            "cuda_device_index": runtime.get("cuda_device_index"),
            "initialization_ns": initialization_ns,
        },
    )


def _tensorrt_backend(
    session: TensorRTSession,
    *,
    spec: Any,
    precision: str,
    initialization_ns: int,
    engine_metadata: Mapping[str, Any] | None = None,
) -> InferenceBackend:
    def prepare(observations: np.ndarray) -> torch.Tensor:
        return prepare_model_input(observations, device=session.device, spec=spec)

    return InferenceBackend(
        runtime="TensorRT",
        requested_provider="cuda",
        actual_provider="TensorRT",
        precision=precision,
        cpu_threads=None,
        thread_setting="CUDA stream 0",
        input_ownership=f"numpy.uint8/cpu → torch.float32/cuda:{session.device_index}",
        output_ownership=f"TensorRT float32/cuda:{session.device_index} → numpy.float32/cpu",
        prepare_model_input=prepare,
        run_model_input=session.execute,
        run_prevalidated_model_input=session.execute_prevalidated,
        materialize_output=lambda output: np.ascontiguousarray(
            output.detach().to(device="cpu", dtype=torch.float32).numpy(),
            dtype=np.float32,
        ),
        synchronize=_sync_for(session.device),
        metadata={
            **session.runtime_metadata,
            "network_typing": (
                engine_metadata.get("network_typing")
                if engine_metadata is not None
                else None
            ),
            "tf32_enabled": (
                engine_metadata.get("tf32_enabled")
                if engine_metadata is not None
                else None
            ),
            "engine_precision": precision,
            "io_precision": "float32",
            "initialization_ns": initialization_ns,
        },
    )


class _TorchEvaluationModule(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.action_log: list[int] = []

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        output = self.model(model_input)
        if not isinstance(output, torch.Tensor):
            raise TypeError("PyTorch evaluation model did not return a tensor")
        self.action_log.extend(torch.argmax(output, dim=1).detach().cpu().tolist())
        return output


class _ONNXEvaluationModule(nn.Module):
    def __init__(self, policy: ONNXRuntimePolicy) -> None:
        super().__init__()
        self.policy = policy
        self.action_log: list[int] = []

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        output = self.policy.predict_model_input(
            np.ascontiguousarray(model_input.detach().cpu().numpy(), dtype=np.float32)
        )
        result = torch.from_numpy(output).to(device=model_input.device)
        self.action_log.extend(torch.argmax(result, dim=1).detach().cpu().tolist())
        return result


class _TensorRTEvaluationModule(nn.Module):
    def __init__(self, session: TensorRTSession) -> None:
        super().__init__()
        self.session = session
        self.action_log: list[int] = []

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        output = self.session.execute(model_input)
        torch.cuda.synchronize(self.session.device)
        self.action_log.extend(torch.argmax(output, dim=1).detach().cpu().tolist())
        return output


def _split_action_log(
    action_log: Sequence[int], lengths: Sequence[int]
) -> list[list[int]]:
    sequences: list[list[int]] = []
    offset = 0
    for length in lengths:
        end = offset + int(length)
        sequence = [int(value) for value in action_log[offset:end]]
        if len(sequence) != int(length):
            raise ValueError(
                "evaluation action log length does not match episode result"
            )
        sequences.append(sequence)
        offset = end
    if offset != len(action_log):
        raise ValueError("evaluation action log contains extra decisions")
    return sequences


def _run_evaluation(
    *,
    root: Path,
    contract: Any,
    evaluation_config: Any,
    target: str,
    model: nn.Module,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    env_factory = lambda: make_breakout_env(**breakout_environment_kwargs(contract))
    started = time.perf_counter()
    result = evaluate_policy(
        model,
        episodes=evaluation_config.episodes_per_seed,
        seeds=evaluation_config.seeds,
        device=device,
        epsilon=evaluation_config.epsilon,
        env_factory=env_factory,
        model_id=f"day26-{target}",
        evaluation_id=f"day26-{target}",
        metadata={
            "experiment": "day26_tensorrt_optimization",
            "target": target,
            "contract_id": contract.contract_id,
        },
    )
    elapsed = time.perf_counter() - started
    result_payload = result.to_dict()
    lengths = [int(episode.episode_length) for episode in result.episodes]
    returns = [float(episode.episode_return) for episode in result.episodes]
    action_sequences = _split_action_log(model.action_log, lengths)  # type: ignore[attr-defined]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "day26_tensorrt_evaluation",
        "target": target,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "elapsed_seconds": elapsed,
        "result": result_payload,
        "requested_action_sequences": action_sequences,
        "episode_returns": returns,
        "episode_lengths": lengths,
        "contract_id": contract.contract_id,
    }
    destination = output_root / f"{target}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["path"] = _relative(root, destination)
    payload["sha256"] = sha256_file(destination)
    return payload


def _evaluation_checks(
    metrics: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "action_agreement_rate": {
            "observed": float(metrics["action_agreement_rate"]),
            "limit": float(thresholds["action_agreement_rate"]),
            "passed": float(metrics["action_agreement_rate"])
            >= float(thresholds["action_agreement_rate"]),
        },
        "episode_return_match_rate": {
            "observed": float(metrics["episode_return_match_rate"]),
            "limit": float(thresholds["episode_return_match_rate"]),
            "passed": float(metrics["episode_return_match_rate"])
            >= float(thresholds["episode_return_match_rate"]),
        },
        "episode_length_match_rate": {
            "observed": float(metrics["episode_length_match_rate"]),
            "limit": float(thresholds["episode_length_match_rate"]),
            "passed": float(metrics["episode_length_match_rate"])
            >= float(thresholds["episode_length_match_rate"]),
        },
    }
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def _find_benchmark_result(
    summary: Mapping[str, Any], target: str, scope: str
) -> Mapping[str, Any]:
    results = _mapping(summary, label="benchmark summary").get("results")
    if not isinstance(results, Sequence):
        raise ValueError("benchmark summary has no results list")
    for result in results:
        if (
            isinstance(result, Mapping)
            and result.get("target") == target
            and result.get("scope") == scope
            and int(result.get("batch_size", -1)) == 1
        ):
            return result
    raise ValueError(
        f"benchmark result not found: target={target}, scope={scope}, batch=1"
    )


def _primary_latency(
    summary: Mapping[str, Any], target: str, scope: str = "end_to_end"
) -> dict[str, float]:
    result = _find_benchmark_result(summary, target, scope)
    latency = _mapping(result.get("latency"), label=f"{target} latency")
    return {"p50_ms": float(latency["p50_ms"]), "p95_ms": float(latency["p95_ms"])}


def _improvement(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "p50_rate": float(
            (float(reference["p50_ms"]) - float(candidate["p50_ms"]))
            / float(reference["p50_ms"])
        ),
        "p95_rate": float(
            (float(reference["p95_ms"]) - float(candidate["p95_ms"]))
            / float(reference["p95_ms"])
        ),
    }


def _render_report(result: Mapping[str, Any]) -> str:
    if result.get("status") != "completed":
        preflight = _mapping(result.get("preflight"), label="preflight")
        decision = _mapping(result.get("decision"), label="decision")
        blockers = preflight.get("blockers", [])
        blocker_text = "; ".join(str(value) for value in blockers) or "none recorded"
        return "\n".join(
            [
                "# Day 26 TensorRT optional experiment",
                "",
                "這次 native NVIDIA TensorRT 路徑沒有產生可宣稱的完整 performance comparison。",
                f"- experiment status: `{result.get('status')}`",
                f"- validation status: `{result.get('validation_status')}`",
                f"- preflight status: `{preflight.get('status')}`",
                f"- blockers: {blocker_text}",
                f"- decision: {decision.get('reason')}",
                "",
                "Browser path remains FP32 ONNX → ORT Web; no TensorRT engine is used for web serving.",
                "",
            ]
        )
    preflight = _mapping(result.get("preflight"), label="preflight")
    parity = _mapping(result.get("parity"), label="parity")
    benchmark = _mapping(result.get("benchmark"), label="benchmark")
    decision = _mapping(result.get("decision"), label="decision")
    preflight_baseline = _mapping(preflight.get("baseline"), label="preflight.baseline")
    pytorch_cuda_status = (
        "passed" if preflight_baseline.get("status") == "passed" else "blocked"
    )
    lines = [
        "# Day 26 TensorRT optional experiment",
        "",
        "這份報告只評估 native NVIDIA GPU 上的 TensorRT；TensorRT 不屬於 Browser/production web serving path。",
        "Browser 仍使用 Day 22 FP32 ONNX → ORT Web WASM/WebGPU。",
        "",
        f"- experiment status: `{result.get('status')}`",
        f"- validation status: `{result.get('validation_status')}`",
        f"- preflight status: `{preflight.get('status')}`",
        f"- GPU: `{result.get('runtime', {}).get('gpu_model')}` (compute capability `{result.get('runtime', {}).get('compute_capability')}`)",
        f"- TensorRT: `{result.get('runtime', {}).get('tensorrt_version')}`",
        f"- benchmark scope: `{benchmark.get('scope_note')}`",
        "",
        "## Preflight gate",
        "",
        "| check | observed | status |",
        "| --- | --- | :---: |",
        f"| PyTorch CUDA | `{pytorch_cuda_status}` | `{preflight.get('baseline', {}).get('status')}` |",
        f"| ORT CUDA provider / graph assignment | `{preflight.get('baseline', {}).get('onnxruntime_cuda', {}).get('runtime', {}).get('actual_provider')}` / `{preflight.get('baseline', {}).get('onnxruntime_cuda', {}).get('runtime', {}).get('graph_assignment', {}).get('status')}` | `{preflight.get('baseline', {}).get('status')}` |",
        f"| TensorRT Python parser | `{preflight.get('software', {}).get('tensorrt', {}).get('imported')}` / `{preflight.get('software', {}).get('tensorrt', {}).get('onnx_parser_available')}` | `{preflight.get('status')}` |",
        f"| Strongly typed network | `{preflight.get('capabilities', {}).get('network_typing')}` / `{preflight.get('capabilities', {}).get('strongly_typed_network_available')}` | `{preflight.get('status')}` |",
        f"| TF32 policy | `{preflight.get('capabilities', {}).get('tf32_enabled')}` | `{preflight.get('status')}` |",
        f"| trtexec | `{preflight.get('software', {}).get('trtexec_path') or 'not found'}` | informational |",
        f"| FP16 capability | `{preflight.get('capabilities', {}).get('fp16_supported')}` | `{preflight.get('status')}` |",
        f"| Driver / CUDA compatibility | `{preflight.get('compatibility', {}).get('driver_status')}` / `{preflight.get('compatibility', {}).get('cuda_status')}` | `{preflight.get('status')}` |",
        f"| Workspace + headroom | `{preflight.get('memory_policy', {}).get('workspace_budget_status')}` | `{preflight.get('status')}` |",
        "",
        "## Q-value correctness and action agreement",
        "",
        "`action agreement` 是 Q-values 取 argmax 後的 action 一致率；`smallest margin` 是樣本中最佳與次佳 action 的最小 Q 差距，能指出最容易受數值誤差影響的決策邊界。",
        "",
        "| candidate | max abs | mean abs | max rel | action agreement | smallest margin | passed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    candidates = _mapping(parity.get("candidates"), label="parity.candidates")
    for target, payload in candidates.items():
        if not isinstance(payload, Mapping):
            continue
        metrics = _mapping(payload.get("metrics"), label=f"{target} parity metrics")
        margin = _mapping(
            metrics.get("candidate_top_2_q_margin"), label=f"{target} margin"
        )
        lines.append(
            f"| {target} | {float(metrics['max_absolute_error']):.9g} | {float(metrics['mean_absolute_error']):.9g} | "
            f"{float(metrics['max_relative_error']):.9g} | {float(metrics['action_agreement_rate']):.6f} | "
            f"{float(margin['min']):.9g} | {payload.get('passed')} |"
        )
    lines.extend(
        [
            "",
            "Direct Day 22 golden-fixture checks are stored separately so the live PyTorch CUDA reference and the saved fixture cannot be confused:",
            "",
            "| candidate | golden action agreement | golden max abs | passed |",
            "| --- | ---: | ---: | :---: |",
        ]
    )
    fixture_candidates = _mapping(
        parity.get("fixture_candidates"), label="parity.fixture_candidates"
    )
    for target, payload in fixture_candidates.items():
        if not isinstance(payload, Mapping):
            continue
        metrics = _mapping(payload.get("metrics"), label=f"{target} fixture metrics")
        lines.append(
            f"| {target} | {float(metrics['action_agreement_rate']):.6f} | "
            f"{float(metrics['max_absolute_error']):.9g} | {payload.get('passed')} |"
        )
    lines.extend(
        [
            "",
            "## Fixed Contract v2 evaluation",
            "",
            "| candidate | action agreement | return match | length match | action disagreements | passed |",
            "| --- | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    eval_candidates = _mapping(
        _mapping(result.get("evaluation"), label="evaluation").get("candidates"),
        label="evaluation.candidates",
    )
    for target, payload in eval_candidates.items():
        if not isinstance(payload, Mapping):
            continue
        metrics = _mapping(payload.get("metrics"), label=f"{target} evaluation metrics")
        lines.append(
            f"| {target} | {float(metrics['action_agreement_rate']):.6f} | "
            f"{float(metrics['episode_return_match_rate']):.6f} | "
            f"{float(metrics['episode_length_match_rate']):.6f} | "
            f"{int(metrics['action_disagreement_count'])} | {payload.get('passed')} |"
        )
    lines.extend(
        [
            "",
            "## Batch latency",
            "",
            "P50/P95 使用同一批 prepared inputs 與 Day 24 的 `model_only`、`end_to_end` 定義；本次 TensorRT 的兩個 scope 都在 `prepare_model_input` 後使用 `execute_prevalidated`，end-to-end 仍包含 uint8→float32 normalization、CPU→GPU transfer、engine execution、GPU→CPU output materialization 與 argmax。",
            "",
        ]
    )
    model_only_latencies = _mapping(
        benchmark.get("batch1_model_only"), label="benchmark.batch1_model_only"
    )
    lines.extend(
        [
            "### Model-only (batch=1)",
            "",
            "| target | P50 ms | P95 ms |",
            "| --- | ---: | ---: |",
        ]
    )
    for target, label in (
        ("pytorch_cuda_fp32", "PyTorch CUDA FP32"),
        ("onnx_cuda_fp32", "ORT CUDA FP32"),
        ("tensorrt_cuda_fp32", "TensorRT FP32"),
        ("tensorrt_cuda_fp16", "TensorRT FP16"),
    ):
        latency = _mapping(
            model_only_latencies.get(target), label=f"{target} model-only latency"
        )
        lines.append(
            f"| {label} | {float(latency['p50_ms']):.6f} | "
            f"{float(latency['p95_ms']):.6f} |"
        )
    lines.extend(
        [
            "",
            "### End-to-end (batch=1)",
            "",
            "| target | engine/model size | batch=1 P50 ms | batch=1 P95 ms | vs ORT FP32 P95 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    latencies = _mapping(
        benchmark.get("batch1_end_to_end"), label="benchmark.batch1_end_to_end"
    )
    sizes = _mapping(
        benchmark.get("artifact_sizes_bytes"), label="benchmark.artifact_sizes_bytes"
    )
    reference_latency = _mapping(
        latencies.get("onnx_cuda_fp32"), label="ORT FP32 latency"
    )
    for target, label in (
        ("pytorch_cuda_fp32", "PyTorch CUDA FP32"),
        ("onnx_cuda_fp32", "ORT CUDA FP32"),
        ("tensorrt_cuda_fp32", "TensorRT FP32"),
        ("tensorrt_cuda_fp16", "TensorRT FP16"),
    ):
        latency = _mapping(latencies.get(target), label=f"{target} latency")
        improvement = _improvement(reference_latency, latency)
        size_value = sizes.get(target)
        size_text = "runtime" if not size_value else f"{int(size_value):,} bytes"
        lines.append(
            f"| {label} | {size_text} | {float(latency['p50_ms']):.6f} | "
            f"{float(latency['p95_ms']):.6f} | {improvement['p95_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- TensorRT FP32: **{decision.get('tensorrt_fp32_recommendation')}**",
            f"- TensorRT FP16: **{decision.get('tensorrt_fp16_recommendation')}**",
            f"- reason: {decision.get('reason')}",
            f"- Browser baseline remains `{decision.get('browser_baseline_precision')}`; `tensorrt_is_web_serving` is `{decision.get('tensorrt_is_web_serving')}`.",
            "",
            "### Portability and maintenance limits",
            "",
            "Serialized TensorRT engines are hardware/software specific. The engine metadata records the GPU, compute capability, CUDA, TensorRT version, source ONNX hash, optimization profile, workspace budget, and exact preflight hash. A different GPU or software stack must rebuild and revalidate the engine.",
            "",
            "TensorRT 11 的強型別 network 讓 FP16 precision 由 source ONNX tensor types 決定；因此 FP16 engine 使用 Day 25 的 derived typed ONNX，而不是假設仍存在舊版 builder FP16 flag。",
            "",
            "### Installation complexity",
            "",
            "本次 base environment 沒有被修改；TensorRT 需要額外的 isolated optional environment/site-packages，且 `trtexec` 不在 PATH，因此正式 parity 與 Python end-to-end benchmark 依賴 TensorRT Python API。這是額外的安裝與維護成本，不應被模型大小的下降掩蓋。",
            "",
            "Official references:",
            "- https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html",
            "- https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html",
            "- https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html",
            "",
            "The decision is deliberately native-only; it does not authorize uploading an engine or adding a TensorRT server to the web demo.",
        ]
    )
    return "\n".join(lines) + "\n"


def _blocked_result(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    preflight_path: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "day26_tensorrt_comparison",
        "technical_question": config.get("technical_question"),
        "status": "blocked",
        "validation_status": "blocked",
        "preflight": dict(preflight),
        "runtime": {},
        "parity": {"candidates": {}, "precision_pairs": {}},
        "benchmark": {"status": "not_run", "reason": "preflight BLOCKED"},
        "evaluation": {"status": "not_run", "reason": "preflight BLOCKED"},
        "decision": {
            "tensorrt_fp32_recommendation": "blocked",
            "tensorrt_fp16_recommendation": "blocked",
            "browser_baseline_precision": "float32",
            "tensorrt_is_web_serving": False,
            "reason": "; ".join(str(value) for value in preflight.get("blockers", [])),
        },
        "lineage": {
            "config": {
                "path": _relative(root, config_path),
                "sha256": sha256_file(config_path, normalize_text=True),
            },
            "preflight": {
                "path": _relative(root, preflight_path),
                "sha256": sha256_file(preflight_path),
            },
        },
        "generation": {"repository_relative_paths_only": True},
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    config_path = _resolve(root, args.config)
    (
        config,
        fp32_thresholds,
        fp16_thresholds,
        benchmark_config,
        relative_epsilon,
        required_headroom_bytes,
    ) = _load_config(config_path)
    parity_config_path = _resolve(root, args.parity_config)
    cuda_baseline_thresholds = _load_cuda_baseline_thresholds(parity_config_path)
    preflight_path = _resolve(root, args.preflight)
    preflight = _json_object(preflight_path)
    if preflight.get("artifact_type") != "day26_tensorrt_preflight":
        raise ValueError("preflight artifact has an unexpected type")
    if preflight.get("status") != TENSORRT_STATUS_READY:
        result = _blocked_result(
            root=root,
            config=config,
            config_path=config_path,
            preflight_path=preflight_path,
            preflight=preflight,
        )
        if preflight.get("status") == TENSORRT_STATUS_CLI_ONLY:
            result["decision"]["reason"] = (
                "CLI_ONLY cannot prove the required strongly typed Python network; "
                "the formal Day 26 experiment is blocked."
            )
            result["benchmark"]["reason"] = (
                "CLI_ONLY preflight: formal comparison is blocked because strong "
                "typing was not verified by the TensorRT Python API."
            )
        return _write_result(root=root, args=args, result=result)

    source_model = _resolve(root, args.source_model)
    source_metadata = _resolve(root, args.source_metadata)
    onnx_fp32 = _resolve(root, args.onnx_fp32)
    onnx_fp16 = _resolve(root, args.onnx_fp16)
    day25_comparison = _resolve(root, args.day25_comparison)
    spec_path = _resolve(root, args.spec)
    probe_states = _resolve(root, args.probe_states)
    reference = _resolve(root, args.reference)
    contract_path = _resolve(root, args.contract)
    evaluation_config_path = _resolve(root, args.evaluation_config)
    engine_fp32 = _resolve(root, args.engine_fp32)
    engine_fp32_metadata = _resolve(root, args.engine_fp32_metadata)
    engine_fp16 = _resolve(root, args.engine_fp16)
    engine_fp16_metadata = _resolve(root, args.engine_fp16_metadata)
    spec, contract, observations, fixture_q_values, _fixture_actions, verified = (
        _verify_lineage(
            root=root,
            config=config,
            config_path=config_path,
            preflight=preflight,
            preflight_path=preflight_path,
            source_model=source_model,
            source_metadata=source_metadata,
            onnx_fp32=onnx_fp32,
            onnx_fp16=onnx_fp16,
            day25_comparison=day25_comparison,
            parity_config_path=parity_config_path,
            spec_path=spec_path,
            probe_states=probe_states,
            reference=reference,
            contract_path=contract_path,
            evaluation_config_path=evaluation_config_path,
            engine_fp32=engine_fp32,
            engine_fp32_metadata=engine_fp32_metadata,
            engine_fp16=engine_fp16,
            engine_fp16_metadata=engine_fp16_metadata,
        )
    )
    if not torch.cuda.is_available() or args.device_index >= torch.cuda.device_count():
        raise TensorRTBlockedError(
            "Day 26 requires the verified NVIDIA CUDA device from preflight"
        )
    device = torch.device(f"cuda:{args.device_index}")
    baseline = _mapping(preflight.get("baseline"), label="preflight.baseline")
    ort_baseline = _mapping(
        baseline.get("onnxruntime_cuda"), label="preflight.baseline.onnxruntime_cuda"
    )
    ort_runtime = _mapping(ort_baseline.get("runtime"), label="preflight ORT runtime")
    require_gpu_baseline(
        torch_cuda_available=bool(
            preflight.get("host", {}).get("gpu", {}).get("cuda_available")
        ),
        ort_actual_provider=ort_runtime.get("actual_provider"),
        ort_graph_assignment_status=_mapping(
            ort_runtime.get("graph_assignment"), label="preflight graph assignment"
        ).get("status"),
    )
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - environment gate
        raise TensorRTBlockedError("Day 26 requires the ORT CUDA baseline") from error
    require_cuda_precision_matrix(
        torch_cuda_available=bool(torch.cuda.is_available()),
        onnxruntime_providers=ort.get_available_providers(),
    )
    source_model_identity = verified["identity"]
    model, _payload, _model_config = load_deployment_model(
        source_model,
        device=device,
        identity=source_model_identity,
        spec=spec,
    )
    model.eval()
    live_reference_q_values = _run_pytorch_reference(
        model,
        observations,
        spec=spec,
        device=device,
    )
    fixture_metrics = _compare(
        fixture_q_values,
        live_reference_q_values,
        thresholds=cuda_baseline_thresholds,
        label="day22_fixture_vs_live_pytorch_cuda_fp32",
        relative_epsilon=relative_epsilon,
    )
    if not fixture_metrics["passed"]:
        raise ValueError(
            "live PyTorch CUDA FP32 baseline diverges from the Day 22 fixture"
        )

    ort_started = time.perf_counter_ns()
    ort_policy = ONNXRuntimePolicy(
        onnx_fp32,
        provider="cuda",
        device_index=args.device_index,
        spec=spec,
    )
    ort_initialization_ns = time.perf_counter_ns() - ort_started
    ort_runtime_metadata = ort_policy.runtime_metadata
    require_gpu_baseline(
        torch_cuda_available=bool(torch.cuda.is_available()),
        ort_actual_provider=ort_runtime_metadata.get("actual_provider"),
        ort_graph_assignment_status=_mapping(
            ort_runtime_metadata.get("graph_assignment"),
            label="live ORT graph assignment",
        ).get("status"),
    )
    ort_q_values = _validate_q_values(
        ort_policy.predict_q_values(observations), count=len(observations)
    )

    trt_site = args.tensorrt_site_packages
    trt_fp32_started = time.perf_counter_ns()
    trt_fp32 = TensorRTSession(
        engine_fp32,
        device_index=args.device_index,
        tensorrt_site_packages=trt_site,
        precision="float32",
    )
    trt_fp32_initialization_ns = time.perf_counter_ns() - trt_fp32_started
    trt_fp16_started = time.perf_counter_ns()
    trt_fp16 = TensorRTSession(
        engine_fp16,
        device_index=args.device_index,
        tensorrt_site_packages=trt_site,
        precision="float16",
    )
    trt_fp16_initialization_ns = time.perf_counter_ns() - trt_fp16_started
    normalized_probe = np.ascontiguousarray(
        prepare_model_input(observations, device="cpu", spec=spec).numpy(),
        dtype=np.float32,
    )
    trt_fp32_q_values = _validate_q_values(
        _run_tensorrt_probe(trt_fp32, normalized_probe), count=len(observations)
    )
    trt_fp16_q_values = _validate_q_values(
        _run_tensorrt_probe(trt_fp16, normalized_probe), count=len(observations)
    )

    parity_candidates = {
        "onnx_cuda_fp32": _compare(
            live_reference_q_values,
            ort_q_values,
            thresholds=cuda_baseline_thresholds,
            label="pytorch_cuda_fp32_reference",
            relative_epsilon=relative_epsilon,
        ),
        "tensorrt_cuda_fp32": _compare(
            live_reference_q_values,
            trt_fp32_q_values,
            thresholds=fp32_thresholds,
            label="pytorch_cuda_fp32_reference",
            relative_epsilon=relative_epsilon,
        ),
        "tensorrt_cuda_fp16": _compare(
            live_reference_q_values,
            trt_fp16_q_values,
            thresholds=fp16_thresholds,
            label="pytorch_cuda_fp32_reference",
            relative_epsilon=relative_epsilon,
        ),
    }
    fixture_candidates = {
        "pytorch_cuda_fp32": fixture_metrics,
        "onnx_cuda_fp32": _compare(
            fixture_q_values,
            ort_q_values,
            thresholds=cuda_baseline_thresholds,
            label="day22_saved_fixture",
            relative_epsilon=relative_epsilon,
        ),
        "tensorrt_cuda_fp32": _compare(
            fixture_q_values,
            trt_fp32_q_values,
            thresholds=fp32_thresholds,
            label="day22_saved_fixture",
            relative_epsilon=relative_epsilon,
        ),
        "tensorrt_cuda_fp16": _compare(
            fixture_q_values,
            trt_fp16_q_values,
            thresholds=fp16_thresholds,
            label="day22_saved_fixture",
            relative_epsilon=relative_epsilon,
        ),
    }
    precision_pairs = {
        "tensorrt_cuda_fp16_vs_tensorrt_cuda_fp32": _compare(
            trt_fp32_q_values,
            trt_fp16_q_values,
            thresholds=fp16_thresholds,
            label="tensorrt_cuda_fp32",
            relative_epsilon=relative_epsilon,
        ),
        "tensorrt_cuda_fp32_vs_onnx_cuda_fp32": _compare(
            ort_q_values,
            trt_fp32_q_values,
            thresholds=fp32_thresholds,
            label="onnx_cuda_fp32",
            relative_epsilon=relative_epsilon,
        ),
    }

    benchmark_id_payload = {
        "experiment": "day26_tensorrt",
        "lineage": verified["lineage"],
        "device_index": args.device_index,
        "targets": [
            "pytorch_cuda_fp32",
            "onnx_cuda_fp32",
            "tensorrt_cuda_fp32",
            "tensorrt_cuda_fp16",
        ],
        "warmup_iterations": benchmark_config.warmup_iterations,
        "iterations": benchmark_config.iterations,
        "batch_sizes": list(benchmark_config.batch_sizes),
    }
    benchmark_id = (
        "day26-tensorrt-v2-"
        + hashlib.sha256(
            json.dumps(
                benchmark_id_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:12]
    )
    benchmark_dir = (root / args.benchmark_root / benchmark_id).resolve()
    if benchmark_dir.exists() and any(benchmark_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"benchmark directory already contains artifacts: {benchmark_dir}; use --force"
        )
    backends = {
        "pytorch_cuda_fp32": _torch_backend(
            model, spec=spec, device=device, initialization_ns=0
        ),
        "onnx_cuda_fp32": _onnx_backend(
            ort_policy, initialization_ns=ort_initialization_ns
        ),
        "tensorrt_cuda_fp32": _tensorrt_backend(
            trt_fp32,
            spec=spec,
            precision="float32",
            initialization_ns=trt_fp32_initialization_ns,
            engine_metadata=verified["engine_fp32"]["payload"],
        ),
        "tensorrt_cuda_fp16": _tensorrt_backend(
            trt_fp16,
            spec=spec,
            precision="float16",
            initialization_ns=trt_fp16_initialization_ns,
            engine_metadata=verified["engine_fp16"]["payload"],
        ),
    }
    benchmark_results: list[dict[str, Any]] = []
    benchmark_samples: list[dict[str, Any]] = []
    for target, backend in backends.items():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if int(free_bytes) < required_headroom_bytes:
            raise TensorRTBlockedError(
                "Day 26 benchmark stopped because configured VRAM headroom is unavailable"
            )
        measured = run_benchmark_target(
            backend,
            observations,
            config=benchmark_config,
            initialization_ns=int(backend.metadata.get("initialization_ns", 0)),
        )
        for summary in measured["summary"]["results"]:
            summary["target"] = target
        for sample in measured["raw_samples"]:
            sample["target"] = target
        benchmark_results.extend(measured["summary"]["results"])
        benchmark_samples.extend(measured["raw_samples"])
    benchmark_payload = {
        "artifact_type": "day26_tensorrt_benchmark",
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "lineage": verified["lineage"],
        "configuration": {
            "warmup_iterations": benchmark_config.warmup_iterations,
            "iterations": benchmark_config.iterations,
            "batch_sizes": list(benchmark_config.batch_sizes),
            "primary_batch_size": 1,
            "scopes": ["model_only", "end_to_end"],
            "scope_note": "TensorRT end_to_end includes host/device transfers and output materialization; model_only starts from a prepared CUDA tensor.",
            "device_index": args.device_index,
            "targets": list(backends),
        },
        "host": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "onnxruntime_version": str(ort.__version__),
            "tensorrt_version": trt_fp32.runtime_metadata.get("tensorrt_version"),
            "gpu_model": torch.cuda.get_device_name(device),
            "cuda_device_index": args.device_index,
            "gpu_memory_total_bytes": int(total_bytes),
            "gpu_memory_free_before_last_target_bytes": int(free_bytes),
        },
        "source_observations": {
            "path": _relative(root, probe_states),
            "sha256": sha256_file(probe_states),
            "count": int(observations.shape[0]),
            "shape": list(observations.shape[1:]),
            "dtype": str(observations.dtype),
        },
        "summary": {"results": benchmark_results},
        "raw_samples": benchmark_samples,
    }
    write_benchmark_artifacts(benchmark_dir, benchmark_payload)
    benchmark_summary_path = benchmark_dir / "summary.json"
    benchmark_raw_path = benchmark_dir / "raw-samples.json"
    benchmark_summary = _json_object(benchmark_summary_path)
    batch1_latency = {
        target: _primary_latency(benchmark_summary, target) for target in backends
    }
    batch1_model_only = {
        target: _primary_latency(benchmark_summary, target, "model_only")
        for target in backends
    }
    artifact_sizes = {
        "onnx_fp32": onnx_fp32.stat().st_size,
        "onnx_fp16": onnx_fp16.stat().st_size,
        "tensorrt_cuda_fp32": engine_fp32.stat().st_size,
        "tensorrt_cuda_fp16": engine_fp16.stat().st_size,
        "pytorch_cuda_fp32": source_model.stat().st_size,
    }

    evaluation_config = verified["evaluation_config"]
    eval_root = (root / args.evaluations_root).resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    if args.force:
        for old_path in eval_root.glob("day26-*.json"):
            old_path.unlink()
    eval_modules: dict[str, nn.Module] = {
        "pytorch_cuda_fp32": _TorchEvaluationModule(model),
        "onnx_cuda_fp32": _ONNXEvaluationModule(ort_policy),
        "tensorrt_cuda_fp32": _TensorRTEvaluationModule(trt_fp32),
        "tensorrt_cuda_fp16": _TensorRTEvaluationModule(trt_fp16),
    }
    evaluations: dict[str, Any] = {}
    for target, module in eval_modules.items():
        evaluations[target] = _run_evaluation(
            root=root,
            contract=contract,
            evaluation_config=evaluation_config,
            target=target,
            model=module,
            device=device,
            output_root=eval_root,
        )
    raw_eval_thresholds = _mapping(
        _mapping(config.get("evaluation"), label="evaluation").get("thresholds"),
        label="evaluation.thresholds",
    )
    reference_target = "pytorch_cuda_fp32"
    eval_candidates: dict[str, Any] = {}
    for target in ("onnx_cuda_fp32", "tensorrt_cuda_fp32", "tensorrt_cuda_fp16"):
        candidate = evaluations[target]
        reference_eval = evaluations[reference_target]
        metrics = compare_evaluation_parity(
            reference_action_sequences=reference_eval["requested_action_sequences"],
            candidate_action_sequences=candidate["requested_action_sequences"],
            reference_returns=reference_eval["episode_returns"],
            candidate_returns=candidate["episode_returns"],
            reference_lengths=reference_eval["episode_lengths"],
            candidate_lengths=candidate["episode_lengths"],
        )
        checks = _evaluation_checks(metrics, raw_eval_thresholds)
        eval_candidates[target] = {
            "reference_target": reference_target,
            "metrics": metrics,
            "thresholds": checks,
            "passed": bool(checks["passed"]),
            "artifact_path": candidate["path"],
            "artifact_sha256": candidate["sha256"],
        }

    ort_reference = batch1_latency["onnx_cuda_fp32"]
    fp32_improvement = _improvement(ort_reference, batch1_latency["tensorrt_cuda_fp32"])
    fp16_improvement = _improvement(ort_reference, batch1_latency["tensorrt_cuda_fp16"])
    fp32_correct = bool(
        parity_candidates["tensorrt_cuda_fp32"]["passed"]
        and eval_candidates["tensorrt_cuda_fp32"]["passed"]
    )
    fp16_correct = bool(
        parity_candidates["tensorrt_cuda_fp16"]["passed"]
        and precision_pairs["tensorrt_cuda_fp16_vs_tensorrt_cuda_fp32"]["passed"]
        and eval_candidates["tensorrt_cuda_fp16"]["passed"]
    )
    fp32_recommendation = (
        "adopt for native NVIDIA inference"
        if fp32_correct and fp32_improvement["p95_rate"] > 0.0
        else "do not adopt"
    )
    fp16_recommendation = (
        "adopt for native NVIDIA inference"
        if fp16_correct and fp16_improvement["p95_rate"] > 0.0
        else "do not adopt"
    )
    result = {
        "schema_version": 1,
        "artifact_type": "day26_tensorrt_comparison",
        "technical_question": config.get("technical_question"),
        "status": "completed",
        "validation_status": "passed" if fp32_correct and fp16_correct else "failed",
        "formal_matrix": [
            "PyTorch CUDA FP32",
            "ONNX Runtime CUDA FP32",
            "TensorRT FP32",
            "TensorRT FP16",
        ],
        "runtime": {
            "gpu_model": torch.cuda.get_device_name(device),
            "compute_capability": f"{torch.cuda.get_device_properties(device).major}.{torch.cuda.get_device_properties(device).minor}",
            "cuda_device_index": args.device_index,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "onnxruntime_version": str(ort.__version__),
            "tensorrt_version": trt_fp32.runtime_metadata.get("tensorrt_version"),
            "required_headroom_bytes": required_headroom_bytes,
            "preflight_status": preflight.get("status"),
        },
        "preflight": {
            "status": preflight.get("status"),
            "path": _relative(root, preflight_path),
            "sha256": sha256_file(preflight_path),
            "compatibility": preflight.get("host", {}).get("compatibility"),
            "baseline": preflight.get("baseline"),
            "software": preflight.get("software"),
            "capabilities": preflight.get("capabilities"),
            "memory_policy": preflight.get("memory_policy"),
            "blockers": preflight.get("blockers", []),
        },
        "lineage": verified["lineage"],
        "reference": {
            "target": "pytorch_cuda_fp32",
            "fixture_path": _relative(root, reference),
            "fixture_sha256": sha256_file(reference),
            "live_fixture_metrics": fixture_metrics,
        },
        "parity": {
            "sample_count": int(len(observations)),
            "relative_error_epsilon": relative_epsilon,
            "action_meanings": list(spec.action_meanings),
            "candidates": parity_candidates,
            "fixture_candidates": fixture_candidates,
            "precision_pairs": precision_pairs,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "summary_path": _relative(root, benchmark_summary_path),
            "summary_sha256": sha256_file(benchmark_summary_path),
            "raw_samples_path": _relative(root, benchmark_raw_path),
            "raw_samples_sha256": sha256_file(benchmark_raw_path),
            "scope_note": benchmark_payload["configuration"]["scope_note"],
            "required_headroom_bytes": required_headroom_bytes,
            "batch1_model_only": batch1_model_only,
            "batch1_end_to_end": batch1_latency,
            "artifact_sizes_bytes": artifact_sizes,
        },
        "evaluation": {
            "reference_target": reference_target,
            "targets": {
                target: {
                    "path": payload["path"],
                    "sha256": payload["sha256"],
                    "episode_count": len(payload["episode_returns"]),
                }
                for target, payload in evaluations.items()
            },
            "candidates": eval_candidates,
        },
        "operational_evaluation": {
            "installation_complexity": {
                "base_environment_modified": False,
                "isolated_optional_environment_required": True,
                "tensorrt_version": trt_fp32.runtime_metadata.get("tensorrt_version"),
                "python_api_used": True,
                "trtexec_available": bool(
                    preflight.get("software", {}).get("trtexec_path")
                ),
                "assessment": "additional isolated TensorRT dependency and explicit site-packages wiring",
            },
            "portability": {
                "hardware_specific_engine": True,
                "cross_gpu_architecture": False,
                "cross_platform": False,
                "assessment": "rebuild and revalidate for a different GPU or software stack",
            },
            "maintenance": {
                "browser_serving": False,
                "engine_rebuild_on_runtime_change": True,
                "metadata_lineage_recorded": True,
                "assessment": "keep as an optional native artifact; do not add it to the Browser deployment path",
            },
        },
        "decision": {
            "tensorrt_fp32_recommendation": fp32_recommendation,
            "tensorrt_fp16_recommendation": fp16_recommendation,
            "tensorrt_fp32_correctness_gate": fp32_correct,
            "tensorrt_fp16_correctness_gate": fp16_correct,
            "tensorrt_fp32_vs_ort_fp32_p50_improvement_rate": fp32_improvement[
                "p50_rate"
            ],
            "tensorrt_fp32_vs_ort_fp32_p95_improvement_rate": fp32_improvement[
                "p95_rate"
            ],
            "tensorrt_fp16_vs_ort_fp32_p50_improvement_rate": fp16_improvement[
                "p50_rate"
            ],
            "tensorrt_fp16_vs_ort_fp32_p95_improvement_rate": fp16_improvement[
                "p95_rate"
            ],
            "browser_baseline_precision": "float32",
            "tensorrt_is_web_serving": False,
            "reason": "Native TensorRT adoption requires the fixed-seed correctness gates and a measured batch=1 end-to-end P95 improvement over the ORT CUDA FP32 baseline; model size is secondary.",
        },
        "generation": {
            "command": f"python -m scripts.analysis.compare_tensorrt --config {args.config.as_posix()} --preflight {args.preflight.as_posix()} --device-index {args.device_index} --force",
            "python_version": platform.python_version(),
            "repository_relative_paths_only": True,
        },
    }
    return _write_result(root=root, args=args, result=result)


def _write_result(
    *, root: Path, args: argparse.Namespace, result: dict[str, Any]
) -> dict[str, Any]:
    output_path = (
        (root / args.output).resolve()
        if not args.output.is_absolute()
        else args.output.resolve()
    )
    report_path = (
        (root / args.report).resolve()
        if not args.report.is_absolute()
        else args.report.resolve()
    )
    if (output_path.exists() or report_path.exists()) and not args.force:
        raise FileExistsError("Day 26 output/report already exists; use --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(result), encoding="utf-8")
    result["output_path"] = _relative(root, output_path)
    result["report_path"] = _relative(root, report_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--onnx-fp32", type=Path, default=DEFAULT_ONNX_FP32)
    parser.add_argument("--onnx-fp16", type=Path, default=DEFAULT_ONNX_FP16)
    parser.add_argument(
        "--day25-comparison", type=Path, default=DEFAULT_DAY25_COMPARISON
    )
    parser.add_argument("--parity-config", type=Path, default=DEFAULT_PARITY_CONFIG)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG
    )
    parser.add_argument("--engine-fp32", type=Path, default=DEFAULT_ENGINE_FP32)
    parser.add_argument(
        "--engine-fp32-metadata", type=Path, default=DEFAULT_ENGINE_FP32_METADATA
    )
    parser.add_argument("--engine-fp16", type=Path, default=DEFAULT_ENGINE_FP16)
    parser.add_argument(
        "--engine-fp16-metadata", type=Path, default=DEFAULT_ENGINE_FP16_METADATA
    )
    parser.add_argument("--tensorrt-site-packages", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--evaluations-root", type=Path, default=DEFAULT_EVALUATIONS_ROOT
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_experiment(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TensorRTBlockedError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Day 26 TensorRT experiment failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "validation_status": result["validation_status"],
                "output": result["output_path"],
                "report": result["report_path"],
                "benchmark_id": result.get("benchmark", {}).get("benchmark_id"),
                "browser_baseline_precision": result["decision"][
                    "browser_baseline_precision"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
