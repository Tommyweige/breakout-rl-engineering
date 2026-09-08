"""Run the formal Day 25 CUDA FP32 versus FP16 inference experiment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
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
from breakout_rl.evaluation import evaluate_policy
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs,
    expand_concrete_episode_seeds,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.inference import (
    ONNXRuntimePolicy,
    load_inference_spec,
    prepare_model_input,
)
from breakout_rl.onnx_artifacts import inspect_onnx_model
from breakout_rl.onnx_parity import compare_q_values
from breakout_rl.precision import (
    PrecisionBlockedError,
    PrecisionThresholds,
    build_precision_benchmark_id,
    compare_evaluation_parity,
    require_cuda_precision_matrix,
)


DEFAULT_CONFIG = Path("configs/inference/precision_validation.json")
DEFAULT_SOURCE_MODEL = Path("assets/day21/models/final_model/model.pt")
DEFAULT_SOURCE_METADATA = Path("assets/day21/models/final_model/metadata.json")
DEFAULT_ONNX_MODEL = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_ONNX_METADATA = Path(
    "assets/day22/models/final_model/model.onnx.metadata.json"
)
DEFAULT_FP16_MODEL = Path("assets/day25/models/model.fp16.onnx")
DEFAULT_FP16_METADATA = Path("assets/day25/models/model.fp16.onnx.metadata.json")
DEFAULT_SPEC = Path("configs/inference/inference_spec.json")
DEFAULT_PARITY = Path("configs/inference/parity_validation.json")
DEFAULT_PARITY_RESULT = Path("assets/day23/onnx-runtime-parity.json")
DEFAULT_PROBE_STATES = Path("assets/day22/inference/probe_states.npz")
DEFAULT_REFERENCE = Path("assets/day22/inference/pytorch_reference.npz")
DEFAULT_EVALUATION_CONFIG = Path("configs/eval/breakout_eval.json")
DEFAULT_CONTRACT = Path("configs/eval/breakout_contract_v2.json")
DEFAULT_OUTPUT = Path("assets/day25/precision-comparison.json")
DEFAULT_REPORT = Path("reports/day25-fp32-vs-fp16.md")
DEFAULT_BENCHMARK_ROOT = Path("assets/day25/benchmarks")
DEFAULT_EVALUATIONS_ROOT = Path("assets/day25/evaluations")
ONE_GIB = 1024**3


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


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, PrecisionThresholds]]:
    config = _json_object(path)
    _require_equal("precision config schema_version", config.get("schema_version"), 1)
    _require_equal(
        "precision config artifact_type",
        config.get("artifact_type"),
        "day25_precision_validation",
    )
    parity = _require_mapping(config.get("parity"), label="parity")
    raw_thresholds = _require_mapping(parity.get("thresholds"), label="parity.thresholds")
    thresholds: dict[str, PrecisionThresholds] = {}
    for key in ("pytorch_cuda_fp16", "onnx_cuda_fp16"):
        thresholds[key] = PrecisionThresholds.from_mapping(
            _require_mapping(raw_thresholds.get(key), label=f"parity.thresholds.{key}")
        )
    relative_epsilon = float(parity.get("relative_error_epsilon"))
    if not np.isfinite(relative_epsilon) or relative_epsilon <= 0.0:
        raise ValueError("parity.relative_error_epsilon must be positive and finite")

    benchmark = _require_mapping(config.get("benchmark"), label="benchmark")
    batch_sizes = tuple(int(value) for value in benchmark.get("batch_sizes", ()))
    benchmark_config = BenchmarkConfig(
        warmup_iterations=int(benchmark.get("warmup_iterations")),
        iterations=int(benchmark.get("iterations")),
        batch_sizes=batch_sizes,
    )
    if int(benchmark.get("primary_batch_size")) != 1:
        raise ValueError("Day 25 primary benchmark batch size must be 1")
    if tuple(str(value) for value in benchmark.get("scopes", ())) != (
        "model_only",
        "end_to_end",
    ):
        raise ValueError("Day 25 benchmark must preserve model_only and end_to_end scopes")

    evaluation = _require_mapping(config.get("evaluation"), label="evaluation")
    evaluation_thresholds = _require_mapping(
        evaluation.get("thresholds"),
        label="evaluation.thresholds",
    )
    for field in (
        "evaluation_action_agreement_rate",
        "episode_return_match_rate",
        "episode_length_match_rate",
    ):
        value = float(evaluation_thresholds.get(field))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"evaluation.thresholds.{field} must be between 0 and 1")
    decision = _require_mapping(config.get("decision"), label="decision")
    minimum_gain = float(decision.get("minimum_batch1_end_to_end_p95_improvement"))
    if not np.isfinite(minimum_gain) or not 0.0 <= minimum_gain <= 1.0:
        raise ValueError("decision minimum latency improvement must be between 0 and 1")
    config["_benchmark_config"] = benchmark_config
    config["_relative_error_epsilon"] = relative_epsilon
    config["_threshold_objects"] = thresholds
    return config, thresholds


def _assert_configured_path(root: Path, actual: Path, expected: Any, *, label: str) -> None:
    if not isinstance(expected, str):
        raise ValueError(f"lineage.{label} must be a repository-relative path")
    _require_equal(f"configured {label}", _relative(root, actual), expected)


def _verify_lineage(
    *,
    root: Path,
    config: Mapping[str, Any],
    source_model: Path,
    source_metadata: Path,
    onnx_model: Path,
    onnx_metadata: Path,
    fp16_model: Path,
    fp16_metadata: Path,
    spec_path: Path,
    parity_config_path: Path,
    parity_result_path: Path,
    probe_states: Path,
    reference: Path,
    contract_path: Path,
) -> tuple[Any, dict[str, Any], np.ndarray]:
    lineage = _require_mapping(config.get("lineage"), label="lineage")
    fp16_config = _require_mapping(config.get("fp16_artifact"), label="fp16_artifact")
    _assert_configured_path(root, source_model, lineage.get("canonical_source_model_path"), label="canonical_source_model_path")
    _assert_configured_path(root, source_metadata, lineage.get("canonical_source_metadata_path"), label="canonical_source_metadata_path")
    _assert_configured_path(root, onnx_model, lineage.get("onnx_model_path"), label="onnx_model_path")
    _assert_configured_path(root, onnx_metadata, lineage.get("onnx_metadata_path"), label="onnx_metadata_path")
    _assert_configured_path(root, spec_path, lineage.get("inference_spec_path"), label="inference_spec_path")
    _assert_configured_path(root, parity_config_path, "configs/inference/parity_validation.json", label="parity_validation_path")
    _assert_configured_path(root, parity_result_path, lineage.get("day23_parity_result_path"), label="day23_parity_result_path")
    _assert_configured_path(root, probe_states, lineage.get("probe_states_path"), label="probe_states_path")
    _assert_configured_path(root, reference, lineage.get("pytorch_reference_path"), label="pytorch_reference_path")
    _assert_configured_path(root, contract_path, lineage.get("environment_contract_path"), label="environment_contract_path")
    _assert_configured_path(root, fp16_model, fp16_config.get("model_path"), label="fp16_model_path")
    _assert_configured_path(root, fp16_metadata, fp16_config.get("metadata_path"), label="fp16_metadata_path")

    identity = source_identity(source_model, source_metadata, root=root)
    _require_equal(
        "canonical source model SHA256",
        identity["model_sha256"],
        lineage.get("canonical_source_model_sha256"),
    )
    _require_equal(
        "source checkpoint SHA256",
        identity["source_checkpoint_sha256"],
        lineage.get("source_checkpoint_sha256"),
    )
    _require_equal("source run id", identity["source_run_id"], lineage.get("source_run_id"))
    _require_equal("source stage", identity["source_stage"], lineage.get("source_stage"))
    _require_equal(
        "source checkpoint step",
        identity["source_checkpoint_step"],
        lineage.get("source_checkpoint_step"),
    )

    observed_hashes = {
        "onnx_model_sha256": sha256_file(onnx_model),
        "inference_spec_sha256": sha256_file(spec_path, normalize_text=True),
        "environment_contract_sha256": sha256_file(contract_path),
        "probe_states_sha256": sha256_file(probe_states),
        "pytorch_reference_sha256": sha256_file(reference),
    }
    for field, observed in observed_hashes.items():
        _require_equal(f"lineage.{field}", observed, lineage.get(field))

    onnx_metadata_payload = _json_object(onnx_metadata)
    _require_equal("Day 22 ONNX metadata model hash", onnx_metadata_payload.get("model_sha256"), observed_hashes["onnx_model_sha256"])
    _require_equal("Day 22 ONNX source model hash", onnx_metadata_payload.get("source_model_sha256"), identity["model_sha256"])
    _require_equal("Day 22 ONNX checkpoint hash", onnx_metadata_payload.get("source_checkpoint_sha256"), identity["source_checkpoint_sha256"])
    spec = load_inference_spec(spec_path)
    validate_environment_contract(spec, root=root)
    _require_equal("configured contract path", _relative(root, contract_path), spec.environment_contract_path)
    _require_equal("configured contract hash", sha256_file(contract_path), spec.environment_contract_sha256)
    evaluation_config_path = (root / str(_require_mapping(config.get("evaluation"), label="evaluation")["config_path"])).resolve()
    evaluation_config = _json_object(evaluation_config_path)
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    _require_equal("evaluation environment_id", evaluation_config.get("environment_id"), contract.environment_id)
    evaluation_seeds = tuple(int(value) for value in evaluation_config.get("seeds", ()))
    episodes_per_seed = int(evaluation_config.get("episodes_per_seed"))
    _require_equal(
        "evaluation concrete seeds",
        tuple(expand_concrete_episode_seeds(evaluation_seeds, episodes_per_seed=episodes_per_seed)),
        contract.concrete_episode_seeds,
    )
    _require_equal("evaluation epsilon", float(evaluation_config.get("epsilon")), contract.evaluation_epsilon)

    parity = _json_object(parity_result_path)
    _require_equal("Day 23 parity status", parity.get("status"), "passed")
    providers = _require_mapping(parity.get("providers"), label="Day 23 providers")
    cuda_result = _require_mapping(providers.get("cuda"), label="Day 23 CUDA result")
    _require_equal("Day 23 CUDA parity", cuda_result.get("passed"), True)
    day24_id = lineage.get("day24_benchmark_id")
    if not isinstance(day24_id, str) or not day24_id:
        raise ValueError("lineage.day24_benchmark_id must be non-empty")
    day24_dir = root / "assets" / "day24" / "benchmarks" / day24_id
    day24_summary_path = day24_dir / "summary.json"
    day24_raw_path = day24_dir / "raw-samples.json"
    day24_summary = _json_object(day24_summary_path)
    _require_equal("Day 24 benchmark id", day24_summary.get("benchmark_id"), day24_id)
    _require_equal("Day 24 benchmark precision", day24_summary.get("configuration", {}).get("precision"), "float32")
    if not day24_raw_path.is_file():
        raise FileNotFoundError(day24_raw_path)

    fp16_metadata_payload = _json_object(fp16_metadata)
    fp16_hash = sha256_file(fp16_model)
    _require_equal("FP16 model metadata hash", fp16_metadata_payload.get("model_sha256"), fp16_hash)
    fp16_source = _require_mapping(fp16_metadata_payload.get("source"), label="FP16 source")
    fp16_source_onnx = _require_mapping(fp16_source.get("onnx_model"), label="FP16 source ONNX")
    _require_equal("FP16 source ONNX hash", fp16_source_onnx.get("sha256"), observed_hashes["onnx_model_sha256"])
    _require_equal("FP16 source model hash", fp16_metadata_payload.get("source", {}).get("source_model_sha256"), identity["model_sha256"])
    _require_equal("FP16 source checkpoint hash", fp16_metadata_payload.get("source", {}).get("source_checkpoint_sha256"), identity["source_checkpoint_sha256"])
    _require_equal("FP16 metadata config hash", fp16_metadata_payload.get("lineage_config", {}).get("sha256"), sha256_file(root / "configs/inference/precision_validation.json", normalize_text=True))
    fp16_graph = inspect_onnx_model(fp16_model, check=True)
    expected_input = ["N", *spec.source_observation_shape]
    expected_output = ["N", len(spec.action_meanings)]
    if fp16_graph["inputs"][0]["name"] != spec.input_name or fp16_graph["inputs"][0]["dtype"] != "float32" or fp16_graph["inputs"][0]["shape"] != expected_input:
        raise ValueError("FP16 model input does not preserve the inference contract")
    if fp16_graph["outputs"][0]["name"] != spec.output_name or fp16_graph["outputs"][0]["dtype"] != "float32" or fp16_graph["outputs"][0]["shape"] != expected_output:
        raise ValueError("FP16 model output does not preserve the inference contract")

    with np.load(probe_states, allow_pickle=False) as probe_archive:
        observations = np.ascontiguousarray(probe_archive["observations"])
    if observations.dtype != np.dtype("uint8") or observations.ndim != 4 or tuple(observations.shape[1:]) != spec.source_observation_shape:
        raise ValueError("probe states do not match the inference contract")
    with np.load(reference, allow_pickle=False) as reference_archive:
        reference_observations = np.ascontiguousarray(reference_archive["observations"])
        reference_q_values = np.ascontiguousarray(reference_archive["q_values"])
        reference_actions = np.ascontiguousarray(reference_archive["greedy_actions"])
    if not np.array_equal(observations, reference_observations):
        raise ValueError("probe states differ from the Day 22 PyTorch reference")
    if reference_q_values.dtype != np.dtype("float32") or reference_actions.dtype != np.dtype("int64"):
        raise ValueError("Day 22 reference dtypes do not match the inference contract")
    return spec, {
        "canonical_source_model": {
            "path": _relative(root, source_model),
            "sha256": identity["model_sha256"],
            "source_run_id": identity["source_run_id"],
            "source_stage": identity["source_stage"],
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "source_checkpoint_step": identity["source_checkpoint_step"],
        },
        "onnx_model": {"path": _relative(root, onnx_model), "sha256": observed_hashes["onnx_model_sha256"]},
        "fp16_model": {"path": _relative(root, fp16_model), "sha256": fp16_hash, "metadata_path": _relative(root, fp16_metadata)},
        "inference_spec": {"path": _relative(root, spec_path), "sha256": observed_hashes["inference_spec_sha256"], "contract_id": spec.contract_id},
        "environment_contract": {"path": _relative(root, contract_path), "sha256": observed_hashes["environment_contract_sha256"], "contract_id": spec.environment_contract_id},
        "evaluation_config": {"path": _relative(root, evaluation_config_path), "sha256": sha256_file(evaluation_config_path)},
        "probe_states": {"path": _relative(root, probe_states), "sha256": observed_hashes["probe_states_sha256"], "count": int(observations.shape[0])},
        "pytorch_reference": {"path": _relative(root, reference), "sha256": observed_hashes["pytorch_reference_sha256"]},
        "day23_parity": {"path": _relative(root, parity_result_path), "sha256": sha256_file(parity_result_path), "status": parity.get("status")},
        "day24_benchmark": {"id": day24_id, "summary_path": _relative(root, day24_summary_path), "summary_sha256": sha256_file(day24_summary_path), "raw_samples_path": _relative(root, day24_raw_path), "raw_samples_sha256": sha256_file(day24_raw_path)},
    }, observations, evaluation_config


def _validate_finite_outputs(q_values: np.ndarray, *, expected_count: int) -> np.ndarray:
    values = np.asarray(q_values)
    if values.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"precision output must be a floating-point array, observed {values.dtype}")
    if values.shape != (expected_count, 4) or not np.isfinite(values).all():
        raise ValueError("precision output must be finite with shape (N, 4)")
    return np.ascontiguousarray(values, dtype=np.float32)


def _run_torch_outputs(
    model: nn.Module,
    observations: np.ndarray,
    *,
    spec: Any,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    model_input = prepare_model_input(observations, device=device, spec=spec)
    if precision == "float16":
        model_input = model_input.to(dtype=torch.float16)
    elif precision != "float32":
        raise ValueError(f"unsupported PyTorch precision: {precision}")
    with torch.inference_mode():
        outputs = model(model_input)
    torch.cuda.synchronize(device)
    if not isinstance(outputs, torch.Tensor):
        raise TypeError("PyTorch precision model must return a tensor")
    return _validate_finite_outputs(outputs.detach().cpu().numpy(), expected_count=len(observations))


def _load_torch_model(
    source_model: Path,
    source_metadata: Path,
    *,
    identity: Mapping[str, Any],
    spec: Any,
    device: torch.device,
    precision: str,
) -> tuple[nn.Module, int]:
    started = time.perf_counter_ns()
    model, _payload, _model_config = load_deployment_model(
        source_model,
        device=device,
        identity=identity,
        spec=spec,
    )
    if precision == "float16":
        model.half()
    elif precision != "float32":
        raise ValueError(f"unsupported PyTorch precision: {precision}")
    model.eval()
    return model, time.perf_counter_ns() - started


def _sync_for(device: torch.device):
    def synchronize() -> None:
        torch.cuda.synchronize(device)

    return synchronize


def _torch_backend(
    model: nn.Module,
    *,
    spec: Any,
    device: torch.device,
    precision: str,
    initialization_ns: int,
) -> InferenceBackend:
    model_dtype = torch.float16 if precision == "float16" else torch.float32

    def prepare(observations: np.ndarray) -> torch.Tensor:
        model_input = prepare_model_input(observations, device=device, spec=spec)
        return model_input.to(dtype=model_dtype)

    def run(model_input: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return model(model_input)

    return InferenceBackend(
        runtime="PyTorch",
        requested_provider="cuda",
        actual_provider=f"torch.cuda:{device.index or 0}",
        precision=precision,
        cpu_threads=None,
        thread_setting="default",
        input_ownership=f"numpy.uint8/cpu → torch.{precision}/cuda:{device.index or 0}",
        output_ownership=f"torch.{precision}/cuda:{device.index or 0} → numpy.float32/cpu",
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
            "precision": precision,
            "input_dtype": str(model_dtype).replace("torch.", ""),
            "output_dtype": str(model_dtype).replace("torch.", ""),
            "required_cast_in_end_to_end": precision == "float16",
            "pytorch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "initialization_ns": initialization_ns,
        },
    )


def _onnx_backend(
    policy: ONNXRuntimePolicy,
    *,
    precision: str,
    initialization_ns: int,
) -> InferenceBackend:
    inputs = list(policy.session.get_inputs())
    outputs = list(policy.session.get_outputs())
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Day 25 ONNX benchmark expects exactly one input and output")
    input_name = str(inputs[0].name)
    output_name = str(outputs[0].name)
    runtime = policy.runtime_metadata

    def prepare(observations: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            prepare_model_input(observations, device="cpu", spec=policy.spec).numpy(),
            dtype=np.float32,
        )

    def run_prevalidated(model_input: np.ndarray) -> np.ndarray:
        values = policy.session.run([output_name], {input_name: model_input})
        return np.asarray(values[0])

    return InferenceBackend(
        runtime="ONNX Runtime",
        requested_provider="cuda",
        actual_provider=str(runtime["actual_provider"]),
        precision=precision,
        cpu_threads=0,
        thread_setting="default",
        input_ownership="numpy.uint8/cpu → numpy.float32/cpu → CUDAExecutionProvider",
        output_ownership="numpy.float32/cpu",
        prepare_model_input=prepare,
        run_model_input=policy.predict_model_input,
        run_prevalidated_model_input=run_prevalidated,
        materialize_output=lambda output: np.ascontiguousarray(output, dtype=np.float32),
        synchronize=lambda: None,
        output_materialization_scope="included_in_runtime_api",
        metadata={
            "framework": "ONNX Runtime",
            "precision": precision,
            "internal_precision": "float16" if precision == "float16" else "float32",
            "io_policy": "float32 input / float32 output",
            "required_cast_in_end_to_end": False,
            "onnxruntime_version": runtime.get("onnxruntime_version"),
            "onnxruntime_device": runtime.get("onnxruntime_device"),
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


def _nvidia_smi_metadata() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,cuda_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return {"status": "unavailable"}
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "status": "available" if rows else "unavailable",
        "query": "name,driver_version,cuda_version,memory.total,memory.free",
        "rows": rows,
    }


def _host_metadata(*, device: torch.device, ort_version: str) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "onnxruntime_version": ort_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_index": int(device.index or 0),
        "gpu_model": torch.cuda.get_device_name(device),
        "gpu_memory_total_bytes": int(total_bytes),
        "gpu_memory_free_before_bytes": int(free_bytes),
        "required_headroom_bytes": ONE_GIB,
        "nvidia_smi": _nvidia_smi_metadata(),
    }


def _check_headroom(device: torch.device) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if int(free_bytes) < ONE_GIB:
        raise PrecisionBlockedError(
            "Day 25 is blocked: CUDA free memory is below the required 1 GiB headroom "
            f"({int(free_bytes)} bytes free)."
        )
    return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}


def _peak_memory(device: torch.device) -> dict[str, Any]:
    return {
        "status": "available",
        "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


class _TorchEvaluationModule(nn.Module):
    def __init__(self, model: nn.Module, *, precision: str) -> None:
        super().__init__()
        self.model = model
        self.precision = precision
        self.action_log: list[int] = []

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        if self.precision == "float16":
            output = self.model(model_input.to(dtype=torch.float16)).float()
        else:
            output = self.model(model_input)
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


def _split_action_log(action_log: Sequence[int], episode_lengths: Sequence[int]) -> list[list[int]]:
    sequences: list[list[int]] = []
    offset = 0
    for length in episode_lengths:
        end = offset + int(length)
        sequence = [int(value) for value in action_log[offset:end]]
        if len(sequence) != int(length):
            raise ValueError("evaluation action log length does not match episode result")
        sequences.append(sequence)
        offset = end
    if offset != len(action_log):
        raise ValueError("evaluation action log contains extra decisions")
    return sequences


def _run_evaluation(
    *,
    root: Path,
    config: Mapping[str, Any],
    contract: Any,
    name: str,
    model: nn.Module,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    evaluation = _require_mapping(config["_evaluation_protocol"], label="evaluation protocol")
    env_factory = lambda: make_breakout_env(**breakout_environment_kwargs(contract))
    started = time.perf_counter()
    result = evaluate_policy(
        model,
        episodes=int(evaluation["episodes_per_seed"]),
        seeds=tuple(int(value) for value in evaluation["seeds"]),
        device=device,
        epsilon=float(evaluation["epsilon"]),
        env_factory=env_factory,
        model_id=f"day25-{name}",
        evaluation_id=f"day25-{name}",
        metadata={
            "experiment": "day25_fp32_vs_fp16",
            "precision_target": name,
            "contract_id": contract.contract_id,
        },
    )
    elapsed = time.perf_counter() - started
    episode_rows = result.to_dict()
    episode_lengths = [int(episode.episode_length) for episode in result.episodes]
    episode_returns = [float(episode.episode_return) for episode in result.episodes]
    action_sequences = _split_action_log(model.action_log, episode_lengths)  # type: ignore[attr-defined]
    payload = {
        "schema_version": 1,
        "artifact_type": "day25_precision_evaluation",
        "target": name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": elapsed,
        "result": episode_rows,
        "requested_action_sequences": action_sequences,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "contract_id": contract.contract_id,
    }
    destination = output_root / f"{name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["path"] = _relative(root, destination)
    payload["sha256"] = sha256_file(destination)
    return payload


def _evaluation_checks(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "evaluation_action_agreement_rate": {
            "observed": float(metrics["action_agreement_rate"]),
            "limit": float(thresholds["evaluation_action_agreement_rate"]),
            "passed": float(metrics["action_agreement_rate"]) >= float(thresholds["evaluation_action_agreement_rate"]),
        },
        "episode_return_match_rate": {
            "observed": float(metrics["episode_return_match_rate"]),
            "limit": float(thresholds["episode_return_match_rate"]),
            "passed": float(metrics["episode_return_match_rate"]) >= float(thresholds["episode_return_match_rate"]),
        },
        "episode_length_match_rate": {
            "observed": float(metrics["episode_length_match_rate"]),
            "limit": float(thresholds["episode_length_match_rate"]),
            "passed": float(metrics["episode_length_match_rate"]) >= float(thresholds["episode_length_match_rate"]),
        },
    }
    return {"passed": all(bool(value["passed"]) for value in checks.values()), "checks": checks}


def _find_benchmark_result(summary: Mapping[str, Any], target: str, scope: str) -> Mapping[str, Any]:
    results = _require_mapping(summary, label="benchmark summary").get("results")
    if not isinstance(results, Sequence):
        raise ValueError("benchmark summary has no results list")
    for result in results:
        if isinstance(result, Mapping) and result.get("target") == target and result.get("scope") == scope and int(result.get("batch_size", -1)) == 1:
            return result
    raise ValueError(f"benchmark result not found: target={target}, scope={scope}, batch=1")


def _latency_comparison(summary: Mapping[str, Any], runtime: str) -> dict[str, Any]:
    fp32_target = f"{runtime}_cuda_fp32"
    fp16_target = f"{runtime}_cuda_fp16"
    fp32 = _find_benchmark_result(summary, fp32_target, "end_to_end")
    fp16 = _find_benchmark_result(summary, fp16_target, "end_to_end")
    fp32_latency = _require_mapping(fp32.get("latency"), label="FP32 latency")
    fp16_latency = _require_mapping(fp16.get("latency"), label="FP16 latency")
    p95_fp32 = float(fp32_latency["p95_ms"])
    p95_fp16 = float(fp16_latency["p95_ms"])
    p50_fp32 = float(fp32_latency["p50_ms"])
    p50_fp16 = float(fp16_latency["p50_ms"])
    return {
        "runtime": runtime,
        "fp32": {"p50_ms": p50_fp32, "p95_ms": p95_fp32},
        "fp16": {"p50_ms": p50_fp16, "p95_ms": p95_fp16},
        "p50_improvement_rate": float((p50_fp32 - p50_fp16) / p50_fp32),
        "p95_improvement_rate": float((p95_fp32 - p95_fp16) / p95_fp32),
    }


def _render_report(result: Mapping[str, Any]) -> str:
    parity = _require_mapping(result.get("parity"), label="parity")
    benchmark = _require_mapping(result.get("benchmark"), label="benchmark")
    decision = _require_mapping(result.get("decision"), label="decision")
    lines = [
        "# Day 25 FP32 vs FP16 precision report",
        "",
        "這份 report 只回答 native NVIDIA CUDA 上的 inference optimization 問題：",
        "把 Day 21 Final Model 的 Day 22 FP32 ONNX baseline 改成內部 FP16 後，",
        "Q-values、greedy actions、固定 Contract v2 evaluation 與 batch=1 latency 是否仍可接受？",
        "",
        f"- experiment status: `{result.get('status')}`",
        f"- precision validation: `{result.get('validation_status')}`",
        f"- GPU: `{result.get('runtime', {}).get('gpu_model')}`",
        f"- CUDA device: `{result.get('runtime', {}).get('cuda_device_index')}`",
        f"- headroom requirement: `{result.get('runtime', {}).get('required_headroom_bytes')}` bytes",
        "",
        "## Direct FP16 precision parity",
        "",
        "| precision pair | max absolute error | mean absolute error | action agreement | passed |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for target, payload in parity.get("precision_pairs", {}).items():
        if not isinstance(payload, Mapping):
            continue
        metrics = _require_mapping(payload.get("metrics"), label=f"{target} metrics")
        lines.append(
            f"| {target} | {float(metrics['max_absolute_error']):.9g} | "
            f"{float(metrics['mean_absolute_error']):.9g} | "
            f"{float(metrics['action_agreement_rate']):.6f} | {payload.get('passed')} |"
        )
    lines.extend(
        [
            "",
            "ORT FP32 對 PyTorch CUDA FP32 的 provider baseline 仍保存在 comparison JSON；"
            "上表則只回答同一 runtime 內 FP32 到 FP16 的精度變化。",
            "",
            "`action agreement` 是四個 Q-values 做 argmax 後選到同一個 action 的比例；"
            "它和數值誤差一起看，避免只因誤差很小就忽略決策邊界。",
            "",
            "## Fixed-seed evaluation parity",
            "",
            "| target | action agreement | return match | length match | passed |",
            "| --- | ---: | ---: | ---: | :---: |",
        ]
    )
    for target, payload in result.get("evaluation", {}).get("candidates", {}).items():
        if not isinstance(payload, Mapping):
            continue
        metrics = _require_mapping(payload.get("metrics"), label=f"{target} evaluation metrics")
        lines.append(
            f"| {target} | {float(metrics['action_agreement_rate']):.6f} | "
            f"{float(metrics['episode_return_match_rate']):.6f} | "
            f"{float(metrics['episode_length_match_rate']):.6f} | {payload.get('passed')} |"
        )
    lines.extend(
        [
            "",
            "這裡的 evaluation parity 是相同固定 seed 下的 requested action trace、episode return "
            "與 episode length 比較；它不是把新的遊戲分數硬套成數值相等的理論保證。",
            "",
            "## Batch=1 end-to-end latency",
            "",
            "| runtime | FP32 P50 | FP16 P50 | P50 change | FP32 P95 | FP16 P95 | P95 change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for runtime, values in benchmark.get("batch1_end_to_end", {}).items():
        if not isinstance(values, Mapping):
            continue
        lines.append(
            f"| {runtime} | {values['fp32']['p50_ms']:.6f} ms | {values['fp16']['p50_ms']:.6f} ms | "
            f"{values['p50_improvement_rate']:.2%} | {values['fp32']['p95_ms']:.6f} ms | "
            f"{values['fp16']['p95_ms']:.6f} ms | {values['p95_improvement_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"ONNX model size: FP32 `{benchmark['model_size_bytes']['fp32']}` bytes; "
            f"FP16 `{benchmark['model_size_bytes']['fp16']}` bytes; "
            f"reduction `{benchmark['model_size_bytes']['reduction_rate']:.2%}`.",
            "",
            "## Decision",
            "",
            f"- PyTorch CUDA: `{decision.get('pytorch_cuda_recommendation')}`",
            f"- ONNX Runtime CUDA: `{decision.get('onnx_cuda_recommendation')}`",
            f"- browser baseline precision: `{decision.get('browser_baseline_precision')}`",
            f"- reason: {decision.get('reason')}",
            "",
            "## Reproduction",
            "",
            "```powershell",
            str(result.get("generation", {}).get("command", "")),
            "```",
            "The result is native CUDA evidence for this model and machine. It does not establish ORT Web FP16 parity or justify changing the Cloudflare Browser baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def run_precision_experiment(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    config_path = _resolve(root, args.config)
    config, threshold_objects = _load_config(config_path)
    source_model = _resolve(root, args.source_model)
    source_metadata = _resolve(root, args.source_metadata)
    onnx_model = _resolve(root, args.onnx_model)
    onnx_metadata = _resolve(root, args.onnx_metadata)
    fp16_model = _resolve(root, args.fp16_model)
    fp16_metadata = _resolve(root, args.fp16_metadata)
    spec_path = _resolve(root, args.spec)
    parity_config_path = _resolve(root, args.parity)
    parity_result_path = _resolve(root, args.parity_result)
    probe_states = _resolve(root, args.probe_states)
    reference = _resolve(root, args.reference)
    contract_path = _resolve(root, args.contract)
    spec, lineage, observations, evaluation_protocol = _verify_lineage(
        root=root,
        config=config,
        source_model=source_model,
        source_metadata=source_metadata,
        onnx_model=onnx_model,
        onnx_metadata=onnx_metadata,
        fp16_model=fp16_model,
        fp16_metadata=fp16_metadata,
        spec_path=spec_path,
        parity_config_path=parity_config_path,
        parity_result_path=parity_result_path,
        probe_states=probe_states,
        reference=reference,
        contract_path=contract_path,
    )
    config["_evaluation_protocol"] = evaluation_protocol
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise PrecisionBlockedError("Day 25 is blocked: onnxruntime-gpu is unavailable") from error
    require_cuda_precision_matrix(
        torch_cuda_available=bool(torch.cuda.is_available()),
        onnxruntime_providers=ort.get_available_providers(),
    )
    if args.device_index < 0 or args.device_index >= torch.cuda.device_count():
        raise PrecisionBlockedError(f"CUDA device index {args.device_index} is unavailable")
    device = torch.device(f"cuda:{args.device_index}")
    headroom = _check_headroom(device)
    source_model_identity = source_identity(source_model, source_metadata, root=root)
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)

    fp32_model, fp32_init_ns = _load_torch_model(
        source_model,
        source_metadata,
        identity=source_model_identity,
        spec=spec,
        device=device,
        precision="float32",
    )
    fp16_torch_model, fp16_init_ns = _load_torch_model(
        source_model,
        source_metadata,
        identity=source_model_identity,
        spec=spec,
        device=device,
        precision="float16",
    )
    reference_q_values = _run_torch_outputs(
        fp32_model,
        observations,
        spec=spec,
        device=device,
        precision="float32",
    )
    with np.load(reference, allow_pickle=False) as reference_archive:
        saved_reference_q_values = np.ascontiguousarray(reference_archive["q_values"])
        saved_reference_actions = np.ascontiguousarray(reference_archive["greedy_actions"])
    reference_fixture_metrics = compare_q_values(
        saved_reference_q_values,
        reference_q_values,
        reference_actions=saved_reference_actions,
        candidate_actions=np.argmax(reference_q_values, axis=1),
        sample_ids=range(len(observations)),
    )
    if reference_fixture_metrics["action_agreement_rate"] != 1.0 or reference_fixture_metrics["max_absolute_error"] > 1e-5:
        raise ValueError("live PyTorch CUDA FP32 reference diverges from the Day 22 fixture")

    fp16_torch_q_values = _run_torch_outputs(
        fp16_torch_model,
        observations,
        spec=spec,
        device=device,
        precision="float16",
    )
    ort_fp32_started = time.perf_counter_ns()
    ort_fp32_policy = ONNXRuntimePolicy(
        onnx_model,
        provider="cuda",
        device_index=args.device_index,
        spec=spec,
    )
    ort_fp32_init_ns = time.perf_counter_ns() - ort_fp32_started
    ort_fp16_started = time.perf_counter_ns()
    ort_fp16_policy = ONNXRuntimePolicy(
        fp16_model,
        provider="cuda",
        device_index=args.device_index,
        spec=spec,
    )
    ort_fp16_init_ns = time.perf_counter_ns() - ort_fp16_started
    ort_fp32_q_values = np.ascontiguousarray(ort_fp32_policy.predict_q_values(observations), dtype=np.float32)
    ort_fp16_q_values = np.ascontiguousarray(ort_fp16_policy.predict_q_values(observations), dtype=np.float32)

    fp16_metadata_payload = _json_object(fp16_metadata)
    fp16_metadata_payload["formal_cuda_validation"] = {
        "device_index": args.device_index,
        "gpu_model": torch.cuda.get_device_name(device),
        "pytorch_version": torch.__version__,
        "onnxruntime_version": str(ort.__version__),
        "cuda_version": torch.version.cuda,
        "precision": "float16",
        "provider": "CUDAExecutionProvider",
        "model_sha256": sha256_file(fp16_model),
        "graph_assignment": ort_fp16_policy.runtime_metadata.get("graph_assignment"),
    }
    fp16_metadata.write_text(
        json.dumps(fp16_metadata_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    candidate_arrays = {
        "pytorch_cuda_fp16": fp16_torch_q_values,
        "onnx_cuda_fp32": ort_fp32_q_values,
        "onnx_cuda_fp16": ort_fp16_q_values,
    }
    candidate_thresholds = {
        "pytorch_cuda_fp16": threshold_objects["pytorch_cuda_fp16"],
        "onnx_cuda_fp32": PrecisionThresholds.from_mapping(
            _require_mapping(
                _require_mapping(_json_object(parity_config_path).get("thresholds"), label="Day 23 thresholds").get("cuda"),
                label="Day 23 CUDA thresholds",
            )
        ),
        "onnx_cuda_fp16": threshold_objects["onnx_cuda_fp16"],
    }
    parity_candidates: dict[str, Any] = {}
    for target, candidate in candidate_arrays.items():
        metrics = compare_q_values(
            reference_q_values,
            candidate,
            sample_ids=range(len(observations)),
            relative_epsilon=float(config["_relative_error_epsilon"]),
        )
        checks = candidate_thresholds[target].check(metrics)
        parity_candidates[target] = {
            "metrics": metrics,
            "thresholds": checks,
            "passed": bool(checks["passed"]),
            "runtime": (
                {
                    "framework": "PyTorch",
                    "precision": "float16",
                    "device": str(device),
                    "cuda_device_index": args.device_index,
                    "gpu_model": torch.cuda.get_device_name(device),
                    "pytorch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                }
                if target == "pytorch_cuda_fp16"
                else (
                    dict(ort_fp32_policy.runtime_metadata)
                    if target == "onnx_cuda_fp32"
                    else dict(ort_fp16_policy.runtime_metadata)
                )
            ),
        }

    precision_pairs: dict[str, Any] = {}
    pair_definitions = {
        "pytorch_cuda_fp16_vs_pytorch_cuda_fp32": (
            reference_q_values,
            fp16_torch_q_values,
            threshold_objects["pytorch_cuda_fp16"],
            "pytorch_cuda_fp32",
        ),
        "onnx_cuda_fp16_vs_onnx_cuda_fp32": (
            ort_fp32_q_values,
            ort_fp16_q_values,
            threshold_objects["onnx_cuda_fp16"],
            "onnx_cuda_fp32",
        ),
    }
    for pair_name, (pair_reference, pair_candidate, thresholds, reference_target) in pair_definitions.items():
        metrics = compare_q_values(
            pair_reference,
            pair_candidate,
            sample_ids=range(len(observations)),
            relative_epsilon=float(config["_relative_error_epsilon"]),
        )
        checks = thresholds.check(metrics)
        precision_pairs[pair_name] = {
            "reference_target": reference_target,
            "candidate_target": pair_name.split("_vs_", 1)[0],
            "metrics": metrics,
            "thresholds": checks,
            "passed": bool(checks["passed"]),
        }

    benchmark_config: BenchmarkConfig = config["_benchmark_config"]
    benchmark_identity = {
        "lineage": lineage,
        "source_model_sha256": source_model_identity["model_sha256"],
        "onnx_fp32_sha256": sha256_file(onnx_model),
        "onnx_fp16_sha256": sha256_file(fp16_model),
        "warmup_iterations": benchmark_config.warmup_iterations,
        "iterations": benchmark_config.iterations,
        "batch_sizes": list(benchmark_config.batch_sizes),
        "primary_batch_size": 1,
        "device_index": args.device_index,
        "targets": ["pytorch_cuda_fp32", "pytorch_cuda_fp16", "onnx_cuda_fp32", "onnx_cuda_fp16"],
    }
    benchmark_id = args.benchmark_id or build_precision_benchmark_id(benchmark_identity)
    benchmark_root = (root / args.output_root / benchmark_id).resolve()
    if benchmark_root.exists() and any(benchmark_root.iterdir()) and not args.force:
        raise FileExistsError(f"benchmark directory already contains artifacts: {benchmark_root}; use --force")

    target_backends = {
        "pytorch_cuda_fp32": (_torch_backend(fp32_model, spec=spec, device=device, precision="float32", initialization_ns=fp32_init_ns), fp32_model),
        "pytorch_cuda_fp16": (_torch_backend(fp16_torch_model, spec=spec, device=device, precision="float16", initialization_ns=fp16_init_ns), fp16_torch_model),
        "onnx_cuda_fp32": (_onnx_backend(ort_fp32_policy, precision="float32", initialization_ns=ort_fp32_init_ns), ort_fp32_policy),
        "onnx_cuda_fp16": (_onnx_backend(ort_fp16_policy, precision="float16", initialization_ns=ort_fp16_init_ns), ort_fp16_policy),
    }
    benchmark_results: list[dict[str, Any]] = []
    benchmark_samples: list[dict[str, Any]] = []
    peak_vram: dict[str, Any] = {}
    for target, (backend, _runtime) in target_backends.items():
        _check_headroom(device)
        if target.startswith("pytorch"):
            torch.cuda.reset_peak_memory_stats(device)
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
        if target.startswith("pytorch"):
            peak_vram[target] = _peak_memory(device)
        else:
            peak_vram[target] = {
                "status": "unavailable",
                "reason": "ONNX Runtime CUDA allocator is not exposed through PyTorch memory statistics",
            }

    benchmark_payload = {
        "artifact_type": "day25_precision_benchmark",
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lineage": lineage,
        "configuration": {
            "warmup_iterations": benchmark_config.warmup_iterations,
            "iterations": benchmark_config.iterations,
            "batch_sizes": list(benchmark_config.batch_sizes),
            "primary_batch_size": 1,
            "scopes": ["model_only", "end_to_end"],
            "end_to_end_semantics": "production_policy_decision_path_with_precision_casts_and_transfers",
            "device_index": args.device_index,
            "targets": list(target_backends),
            "precision": "mixed_fp32_fp16_by_target",
            "initialization_ns_by_target": {
                target: int(backend.metadata.get("initialization_ns", 0))
                for target, (backend, _runtime) in target_backends.items()
            },
        },
        "host": _host_metadata(device=device, ort_version=str(ort.__version__)),
        "headroom_check": headroom,
        "peak_vram_by_target": peak_vram,
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
    write_benchmark_artifacts(benchmark_root, benchmark_payload)
    benchmark_summary_path = benchmark_root / "summary.json"
    benchmark_raw_path = benchmark_root / "raw-samples.json"
    benchmark_summary = _json_object(benchmark_summary_path)
    batch1_latency = {
        "pytorch": _latency_comparison(benchmark_summary, "pytorch"),
        "onnx": _latency_comparison(benchmark_summary, "onnx"),
    }
    fp32_size = onnx_model.stat().st_size
    fp16_size = fp16_model.stat().st_size
    model_size = {
        "fp32": fp32_size,
        "fp16": fp16_size,
        "reduction_rate": float((fp32_size - fp16_size) / fp32_size),
    }

    evaluation_root = (root / args.evaluations_root).resolve()
    if evaluation_root.exists() and args.force:
        for old_path in evaluation_root.glob("day25-*.json"):
            old_path.unlink()
    eval_wrappers = {
        "pytorch_cuda_fp32": _TorchEvaluationModule(fp32_model, precision="float32"),
        "pytorch_cuda_fp16": _TorchEvaluationModule(fp16_torch_model, precision="float16"),
        "onnx_cuda_fp32": _ONNXEvaluationModule(ort_fp32_policy),
        "onnx_cuda_fp16": _ONNXEvaluationModule(ort_fp16_policy),
    }
    evaluations: dict[str, Any] = {}
    for target, wrapper in eval_wrappers.items():
        evaluations[target] = _run_evaluation(
            root=root,
            config=config,
            contract=contract,
            name=target,
            model=wrapper,
            device=device,
            output_root=evaluation_root,
        )
    eval_thresholds = _require_mapping(
        _require_mapping(config["evaluation"], label="evaluation").get("thresholds"),
        label="evaluation.thresholds",
    )
    evaluation_pairs = {
        "pytorch_cuda_fp16": "pytorch_cuda_fp32",
        "onnx_cuda_fp16": "onnx_cuda_fp32",
    }
    evaluation_candidates: dict[str, Any] = {}
    for target, reference_target in evaluation_pairs.items():
        reference_eval = evaluations[reference_target]
        candidate_eval = evaluations[target]
        metrics = compare_evaluation_parity(
            reference_action_sequences=reference_eval["requested_action_sequences"],
            candidate_action_sequences=candidate_eval["requested_action_sequences"],
            reference_returns=reference_eval["episode_returns"],
            candidate_returns=candidate_eval["episode_returns"],
            reference_lengths=reference_eval["episode_lengths"],
            candidate_lengths=candidate_eval["episode_lengths"],
        )
        checks = _evaluation_checks(metrics, eval_thresholds)
        evaluation_candidates[target] = {
            "reference_target": reference_target,
            "metrics": metrics,
            "thresholds": checks,
            "passed": bool(checks["passed"]),
            "artifact_path": candidate_eval["path"],
            "artifact_sha256": candidate_eval["sha256"],
        }

    correctness_passed = (
        bool(parity_candidates["onnx_cuda_fp32"]["passed"])
        and all(bool(value["passed"]) for value in precision_pairs.values())
        and all(bool(value["passed"]) for value in evaluation_candidates.values())
    )
    minimum_gain = float(
        _require_mapping(config["decision"], label="decision")[
            "minimum_batch1_end_to_end_p95_improvement"
        ]
    )
    pytorch_gain = batch1_latency["pytorch"]["p95_improvement_rate"]
    onnx_gain = batch1_latency["onnx"]["p95_improvement_rate"]
    if not correctness_passed:
        pytorch_recommendation = "not recommended: correctness threshold failed"
        onnx_recommendation = "not recommended: correctness threshold failed"
        decision_reason = "FP16 cannot be recommended when numeric, action, or fixed-seed evaluation parity fails."
    else:
        pytorch_recommendation = (
            "worth testing further: batch=1 P95 meets the configured improvement gate"
            if pytorch_gain >= minimum_gain
            else "not worth adopting for this batch=1 workload"
        )
        onnx_recommendation = (
            "worth testing further: batch=1 P95 meets the configured improvement gate"
            if onnx_gain >= minimum_gain
            else "not worth adopting for this batch=1 workload"
        )
        decision_reason = (
            "The recommendation is based on fixed-seed correctness first and measured batch=1 end-to-end P95 second; "
            "model-size reduction is recorded as a secondary result."
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "day25_precision_comparison",
        "technical_question": config.get("technical_question"),
        "status": "completed",
        "validation_status": "passed" if correctness_passed else "failed",
        "formal_matrix": [
            "PyTorch CUDA FP32",
            "PyTorch CUDA FP16",
            "ONNX Runtime CUDA FP32",
            "ONNX Runtime CUDA FP16",
        ],
        "runtime": {
            "gpu_model": torch.cuda.get_device_name(device),
            "cuda_device_index": args.device_index,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "onnxruntime_version": str(ort.__version__),
            "required_headroom_bytes": ONE_GIB,
            "headroom_before_experiment": headroom,
        },
        "lineage": lineage,
        "config": {
            "path": _relative(root, config_path),
            "sha256": sha256_file(config_path, normalize_text=True),
        },
        "reference": {
            "target": "pytorch_cuda_fp32",
            "fixture_path": _relative(root, reference),
            "fixture_sha256": sha256_file(reference),
            "live_fixture_metrics": reference_fixture_metrics,
        },
        "parity": {
            "sample_count": int(len(observations)),
            "action_meanings": list(spec.action_meanings),
            "candidates": parity_candidates,
            "precision_pairs": precision_pairs,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "summary_path": _relative(root, benchmark_summary_path),
            "summary_sha256": sha256_file(benchmark_summary_path),
            "raw_samples_path": _relative(root, benchmark_raw_path),
            "raw_samples_sha256": sha256_file(benchmark_raw_path),
            "batch1_end_to_end": batch1_latency,
            "model_size_bytes": model_size,
            "peak_vram_by_target": peak_vram,
        },
        "evaluation": {
            "reference_targets": {
                target: reference_target
                for target, reference_target in evaluation_pairs.items()
            },
            "targets": {
                target: {
                    "path": payload["path"],
                    "sha256": payload["sha256"],
                    "episode_count": len(payload["episode_returns"]),
                }
                for target, payload in evaluations.items()
            },
            "candidates": evaluation_candidates,
        },
        "decision": {
            "pytorch_cuda_recommendation": pytorch_recommendation,
            "onnx_cuda_recommendation": onnx_recommendation,
            "browser_baseline_precision": "float32",
            "minimum_batch1_end_to_end_p95_improvement": minimum_gain,
            "pytorch_p95_improvement_rate": pytorch_gain,
            "onnx_p95_improvement_rate": onnx_gain,
            "reason": decision_reason,
            "model_size_is_secondary": True,
        },
        "generation": {
            "command": (
                "python -m scripts.analysis.compare_precision "
                f"--config {args.config.as_posix()} --device-index {args.device_index} --force"
            ),
            "python_version": platform.python_version(),
            "repository_relative_paths_only": True,
        },
    }
    output_path = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    report_path = (root / args.report).resolve() if not args.report.is_absolute() else args.report.resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"comparison artifact already exists: {output_path}; use --force")
    if report_path.exists() and not args.force:
        raise FileExistsError(f"report already exists: {report_path}; use --force")
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
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--onnx-metadata", type=Path, default=DEFAULT_ONNX_METADATA)
    parser.add_argument("--fp16-model", type=Path, default=DEFAULT_FP16_MODEL)
    parser.add_argument("--fp16-metadata", type=Path, default=DEFAULT_FP16_METADATA)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--parity", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--parity-result", type=Path, default=DEFAULT_PARITY_RESULT)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--evaluations-root", type=Path, default=DEFAULT_EVALUATIONS_ROOT)
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_precision_experiment(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        PrecisionBlockedError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Day 25 precision experiment failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "validation_status": result["validation_status"],
                "benchmark_id": result["benchmark"]["benchmark_id"],
                "output": result["output_path"],
                "report": result["report_path"],
                "browser_baseline_precision": result["decision"]["browser_baseline_precision"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
