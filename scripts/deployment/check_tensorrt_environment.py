"""Preflight the optional TensorRT path without changing the main environment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from breakout_rl.artifacts import repository_relative_path, repository_root, sha256_file
from breakout_rl.deployment import (
    load_deployment_model,
    source_identity,
    validate_environment_contract,
)
from breakout_rl.inference import (
    ONNXRuntimePolicy,
    load_inference_spec,
    prepare_model_input,
)
from breakout_rl.onnx_parity import compare_q_values
from breakout_rl.precision import PrecisionThresholds
from breakout_rl.tensorrt import (
    TENSORRT_STATUS_BLOCKED,
    TensorRTBlockedError,
    classify_tensorrt_status,
    create_strongly_typed_network,
    require_gpu_baseline,
    validate_workspace_budget,
)


DEFAULT_OUTPUT = Path("assets/day26/tensorrt-preflight.json")
DEFAULT_CONFIG = Path("configs/inference/tensorrt_experiment.json")
DEFAULT_DAY25_COMPARISON = Path("assets/day25/precision-comparison.json")
DEFAULT_SOURCE_MODEL = Path("assets/day21/models/final_model/model.pt")
DEFAULT_SOURCE_METADATA = Path("assets/day21/models/final_model/metadata.json")
DEFAULT_ONNX_MODEL = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_SPEC = Path("configs/inference/inference_spec.json")
DEFAULT_PROBE_STATES = Path("assets/day22/inference/probe_states.npz")
DEFAULT_REFERENCE = Path("assets/day22/inference/pytorch_reference.npz")
DEFAULT_DEVICE_INDEX = 0

OFFICIAL_REFERENCES = [
    {
        "title": "TensorRT Prerequisites",
        "url": "https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html",
        "checked_claims": [
            "Windows driver r537 or later",
            "Python 3.10-3.14 recommended",
            "CUDA 13.0 update 2 is a supported CUDA 13.x target",
        ],
    },
    {
        "title": "TensorRT Support Matrix",
        "url": "https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html",
        "checked_claims": [
            "Windows x64 and Python wheel support",
            "serialized engines are hardware/software specific",
        ],
    },
    {
        "title": "TensorRT pip installation",
        "url": "https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html",
        "checked_claims": [
            "pip installation uses a CUDA 13.x TensorRT variant by default",
        ],
    },
]


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


def _resolve(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _nvidia_smi() -> dict[str, Any]:
    query = "name,driver_version,memory.total,memory.free,memory.used"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {"status": "unavailable", "error": str(error)}
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "status": "available" if rows else "unavailable",
        "query": query,
        "rows": rows,
    }


def _driver_cuda_compatibility(
    *, gpu: Mapping[str, Any], nvidia_smi: Mapping[str, Any]
) -> dict[str, Any]:
    """Turn observed driver/CUDA facts into an explicit fail-closed gate."""

    parsed_driver: float | None = None
    driver_text: str | None = None
    rows = nvidia_smi.get("rows", [])
    if isinstance(rows, Sequence) and rows:
        fields = [field.strip() for field in str(rows[0]).split(",")]
        if len(fields) >= 2:
            driver_text = fields[1] or None
            try:
                parsed_driver = float(driver_text) if driver_text else None
            except ValueError:
                parsed_driver = None
    required_driver = 537.0 if platform.system() == "Windows" else None
    driver_passed = parsed_driver is not None and (
        required_driver is None or parsed_driver >= required_driver
    )
    cuda_version = gpu.get("torch_cuda_version")
    cuda_passed = bool(
        gpu.get("cuda_available") and isinstance(cuda_version, str) and cuda_version
    )
    return {
        "driver_version": driver_text,
        "minimum_driver_version": required_driver,
        "driver_status": "passed" if driver_passed else "failed",
        "torch_cuda_version": cuda_version,
        "cuda_status": "passed" if cuda_passed else "failed",
    }


def _load_optional_tensorrt(
    site_packages: Path | None,
) -> tuple[Any | None, dict[str, Any]]:
    if site_packages is not None:
        resolved = site_packages.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        sys.path.insert(0, str(resolved))
    try:
        import tensorrt as trt
    except Exception as error:  # pragma: no cover - host-dependent optional package
        return None, {
            "imported": False,
            "version": None,
            "import_error": f"{type(error).__name__}: {error}",
            "onnx_parser_available": False,
            "builder_available": False,
            "strongly_typed_network_available": False,
            "strongly_typed_network_error": "TensorRT Python API was not imported",
        }
    parser_available = callable(getattr(trt, "OnnxParser", None))
    builder_available = callable(getattr(trt, "Builder", None))
    platform_has_fast_fp16: bool | None = None
    strongly_typed_network_available = False
    strongly_typed_network_error: str | None = None
    if builder_available:
        try:
            logger = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
        except Exception as error:
            platform_has_fast_fp16 = None
            strongly_typed_network_error = f"{type(error).__name__}: {error}"
        else:
            try:
                platform_has_fast_fp16 = bool(builder.platform_has_fast_fp16)
            except Exception:
                platform_has_fast_fp16 = None
            try:
                create_strongly_typed_network(builder, trt)
                strongly_typed_network_available = True
            except Exception as error:
                strongly_typed_network_error = f"{type(error).__name__}: {error}"
    return trt, {
        "imported": True,
        "version": str(getattr(trt, "__version__", "unknown")),
        "import_error": None,
        "onnx_parser_available": parser_available,
        "builder_available": builder_available,
        "platform_has_fast_fp16": platform_has_fast_fp16,
        "strongly_typed_network_available": strongly_typed_network_available,
        "strongly_typed_network_error": strongly_typed_network_error,
    }


def _gpu_metadata(device_index: int) -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    metadata: dict[str, Any] = {
        "cuda_available": available,
        "device_index": device_index,
        "gpu_model": None,
        "compute_capability": None,
        "total_memory_bytes": None,
        "free_memory_bytes": None,
        "torch_cuda_version": torch.version.cuda,
    }
    if not available or device_index < 0 or device_index >= torch.cuda.device_count():
        return metadata
    properties = torch.cuda.get_device_properties(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    metadata.update(
        {
            "gpu_model": torch.cuda.get_device_name(device_index),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(total_bytes),
            "free_memory_bytes": int(free_bytes),
            "torch_device_count": torch.cuda.device_count(),
        }
    )
    return metadata


def _load_probe(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        observations = np.ascontiguousarray(archive["observations"])
    if (
        observations.dtype != np.dtype("uint8")
        or observations.ndim != 4
        or tuple(observations.shape[1:]) != (4, 84, 84)
    ):
        raise ValueError("probe states must have shape (N, 4, 84, 84) and dtype uint8")
    return observations


def _run_gpu_baseline(
    *,
    root: Path,
    source_model: Path,
    source_metadata: Path,
    onnx_model: Path,
    spec: Any,
    observations: np.ndarray,
    reference: Path,
    device_index: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "blocked", "reason": "PyTorch CUDA is unavailable"}
    device = torch.device(f"cuda:{device_index}")
    identity = source_identity(source_model, source_metadata, root=root)
    model, _payload, _config = load_deployment_model(
        source_model,
        device=device,
        identity=identity,
        spec=spec,
    )
    model_input = prepare_model_input(observations, device=device, spec=spec)
    with torch.inference_mode():
        pytorch_output = model(model_input)
    torch.cuda.synchronize(device)
    if not isinstance(pytorch_output, torch.Tensor):
        raise TypeError("PyTorch baseline did not return a tensor")
    pytorch_q = np.ascontiguousarray(
        pytorch_output.detach().cpu().numpy(), dtype=np.float32
    )
    pytorch_actions = np.argmax(pytorch_q, axis=1).astype(np.int64)
    ort_policy = ONNXRuntimePolicy(
        onnx_model,
        provider="cuda",
        device_index=device_index,
        spec=spec,
    )
    ort_q = np.ascontiguousarray(
        ort_policy.predict_q_values(observations), dtype=np.float32
    )
    ort_actions = np.argmax(ort_q, axis=1).astype(np.int64)
    require_gpu_baseline(
        torch_cuda_available=True,
        ort_actual_provider=ort_policy.runtime_metadata.get("actual_provider"),
        ort_graph_assignment_status=ort_policy.runtime_metadata.get(
            "graph_assignment", {}
        ).get("status"),
    )
    with np.load(reference, allow_pickle=False) as archive:
        golden_q = np.ascontiguousarray(archive["q_values"], dtype=np.float32)
        golden_actions = np.ascontiguousarray(archive["greedy_actions"], dtype=np.int64)
    parity_config = _json_object(root / "configs/inference/parity_validation.json")
    thresholds = PrecisionThresholds.from_mapping(parity_config["thresholds"]["cuda"])
    pytorch_metrics = compare_q_values(
        golden_q,
        pytorch_q,
        reference_actions=golden_actions,
        candidate_actions=pytorch_actions,
        sample_ids=range(len(observations)),
    )
    ort_metrics = compare_q_values(
        golden_q,
        ort_q,
        reference_actions=golden_actions,
        candidate_actions=ort_actions,
        sample_ids=range(len(observations)),
    )
    pytorch_checks = thresholds.check(pytorch_metrics)
    ort_checks = thresholds.check(ort_metrics)
    return {
        "status": (
            "passed"
            if pytorch_checks["passed"]
            and ort_checks["passed"]
            and ort_policy.runtime_metadata["graph_assignment"]["status"] == "verified"
            else "failed"
        ),
        "device": {
            "index": device_index,
            "gpu_model": torch.cuda.get_device_name(device_index),
        },
        "pytorch_cuda": {
            "precision": "float32",
            "metrics": pytorch_metrics,
            "thresholds": pytorch_checks,
        },
        "onnxruntime_cuda": {
            "precision": "float32",
            "runtime": ort_policy.runtime_metadata,
            "metrics": ort_metrics,
            "thresholds": ort_checks,
        },
        "source_model_sha256": identity["model_sha256"],
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    output = (
        (root / args.output).resolve()
        if not args.output.is_absolute()
        else args.output.resolve()
    )
    if output.exists() and not args.force:
        raise FileExistsError(
            f"preflight artifact already exists: {output}; use --force"
        )
    config_path = _resolve(root, args.config)
    config = _json_object(config_path)
    if config.get("artifact_type") != "day26_tensorrt_experiment":
        raise ValueError("TensorRT config has an unexpected artifact_type")
    source_config = _require_mapping(config.get("source"), label="config.source")
    build_config = _require_mapping(config.get("build"), label="config.build")
    if build_config.get("network_typing") != "strongly_typed":
        raise ValueError("config.build.network_typing must be strongly_typed")
    if build_config.get("tf32_enabled") is not False:
        raise ValueError("config.build.tf32_enabled must be false")
    workspace_limit_bytes = int(build_config.get("workspace_limit_bytes"))
    required_headroom_bytes = int(build_config.get("required_headroom_bytes"))
    optimization_profile = _require_mapping(
        build_config.get("optimization_profile"),
        label="config.build.optimization_profile",
    )
    if workspace_limit_bytes < 0 or required_headroom_bytes < 0:
        raise ValueError("TensorRT workspace and headroom must not be negative")
    day25_path = _resolve(root, args.day25_comparison)
    day25 = _json_object(day25_path)
    if (
        day25.get("artifact_type") != "day25_precision_comparison"
        or day25.get("status") != "completed"
    ):
        raise ValueError(
            "Day 25 comparison artifact is not a completed source artifact"
        )
    source_model = _resolve(root, args.source_model)
    source_metadata = _resolve(root, args.source_metadata)
    onnx_model = _resolve(root, args.onnx_model)
    spec_path = _resolve(root, args.spec)
    probe_states = _resolve(root, args.probe_states)
    reference = _resolve(root, args.reference)
    spec = load_inference_spec(spec_path)
    validate_environment_contract(spec, root=root)
    if source_config.get("inference_spec_path") != _relative(root, spec_path):
        raise ValueError("inference spec path does not match TensorRT config")
    if source_config.get("environment_contract_path") != spec.environment_contract_path:
        raise ValueError("Contract v2 path does not match the inference spec")
    source_identity_payload = source_identity(source_model, source_metadata, root=root)
    if source_identity_payload["model_sha256"] != source_config.get(
        "canonical_source_model_sha256"
    ):
        raise ValueError("Day 21 source model hash does not match TensorRT config")
    if source_config.get("onnx_fp32_sha256") != sha256_file(onnx_model):
        raise ValueError("Day 22 ONNX hash does not match TensorRT config")
    if source_config.get("day25_comparison_sha256") != sha256_file(day25_path):
        raise ValueError("Day 25 comparison hash does not match TensorRT config")
    expected_onnx = _require_mapping(day25["lineage"], label="Day 25 lineage")[
        "onnx_model"
    ]
    if expected_onnx.get("sha256") != sha256_file(onnx_model):
        raise ValueError("Day 22 ONNX hash does not match the completed Day 25 source")
    observations = _load_probe(probe_states)
    gpu = _gpu_metadata(args.device_index)
    nvidia_smi = _nvidia_smi()
    compatibility = _driver_cuda_compatibility(gpu=gpu, nvidia_smi=nvidia_smi)
    trt, tensorrt = _load_optional_tensorrt(args.tensorrt_site_packages)
    trtexec_path = shutil.which("trtexec")
    parser_available = bool(tensorrt["onnx_parser_available"] or trtexec_path)
    fp16_supported = False
    if gpu["compute_capability"] is not None:
        major, minor = (int(value) for value in gpu["compute_capability"].split(".", 1))
        fp16_supported = (major, minor) >= (5, 3)
    if tensorrt.get("platform_has_fast_fp16") is True:
        fp16_supported = True
    baseline = _run_gpu_baseline(
        root=root,
        source_model=source_model,
        source_metadata=source_metadata,
        onnx_model=onnx_model,
        spec=spec,
        observations=observations,
        reference=reference,
        device_index=args.device_index,
    )
    blockers: list[str] = []
    if not gpu["cuda_available"]:
        blockers.append("PyTorch CUDA is unavailable")
    if baseline.get("status") != "passed":
        blockers.append(
            "PyTorch/ORT CUDA baseline did not pass strict provider validation"
        )
    if not tensorrt["imported"] and not trtexec_path:
        blockers.append("TensorRT Python API and trtexec are unavailable")
    if not tensorrt["builder_available"] and not trtexec_path:
        blockers.append("TensorRT builder API and trtexec are unavailable")
    if (
        tensorrt["imported"]
        and tensorrt["builder_available"]
        and not tensorrt["strongly_typed_network_available"]
    ):
        blockers.append(
            "TensorRT Python API could not create the required strongly typed network"
        )
    if not parser_available:
        blockers.append("TensorRT ONNX parser is unavailable")
    if gpu["compute_capability"] is None:
        blockers.append("NVIDIA compute capability could not be observed")
    if gpu["compute_capability"] is not None and not fp16_supported:
        blockers.append("GPU does not advertise the required FP16 capability")
    if compatibility["driver_status"] != "passed":
        blockers.append("NVIDIA driver version could not pass the platform requirement")
    if compatibility["cuda_status"] != "passed":
        blockers.append("CUDA runtime version/availability could not be verified")
    try:
        validate_workspace_budget(
            workspace_bytes=workspace_limit_bytes,
            free_bytes=int(gpu["free_memory_bytes"]),
            headroom_bytes=required_headroom_bytes,
        )
        workspace_budget_status = "passed"
        workspace_budget_reason = None
    except (TypeError, ValueError, TensorRTBlockedError) as error:
        workspace_budget_status = "failed"
        workspace_budget_reason = str(error)
        blockers.append("configured TensorRT workspace would violate VRAM headroom")
    status = classify_tensorrt_status(
        tensorrt_imported=bool(tensorrt["imported"]),
        onnx_parser_available=parser_available,
        builder_available=bool(tensorrt["builder_available"]),
        strongly_typed_network_available=bool(
            tensorrt["strongly_typed_network_available"]
        ),
        trtexec_available=bool(trtexec_path),
    )
    if blockers:
        status = TENSORRT_STATUS_BLOCKED
    result = {
        "schema_version": 1,
        "artifact_type": "day26_tensorrt_preflight",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "host": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "gpu": gpu,
            "compatibility": compatibility,
            "nvidia_smi": nvidia_smi,
            "nvcc_path": shutil.which("nvcc"),
        },
        "software": {
            "pytorch_version": torch.__version__,
            "pytorch_cuda_version": torch.version.cuda,
            "onnxruntime_version": baseline.get("onnxruntime_cuda", {})
            .get("runtime", {})
            .get("onnxruntime_version"),
            "tensorrt": tensorrt,
            "trtexec_path": trtexec_path,
            "onnx_parser_available": parser_available,
            "driver_version": compatibility["driver_version"],
        },
        "capabilities": {
            "fp16_supported": fp16_supported,
            "builder_api_available": bool(tensorrt["builder_available"]),
            "strongly_typed_network_available": bool(
                tensorrt["strongly_typed_network_available"]
            ),
            "network_typing": build_config["network_typing"],
            "tf32_enabled": bool(build_config["tf32_enabled"]),
            "python_api_build_path": bool(
                tensorrt["imported"]
                and tensorrt["builder_available"]
                and parser_available
                and tensorrt["strongly_typed_network_available"]
            ),
            "cli_build_path": bool(trtexec_path and parser_available),
            "driver_compatible": compatibility["driver_status"] == "passed",
            "cuda_runtime_compatible": compatibility["cuda_status"] == "passed",
        },
        "memory_policy": {
            "required_headroom_bytes": required_headroom_bytes,
            "observed_free_bytes": gpu["free_memory_bytes"],
            "workspace_limit_default_bytes": workspace_limit_bytes,
            "workspace_budget_status": workspace_budget_status,
            "workspace_budget_reason": workspace_budget_reason,
            "optimization_profile": dict(optimization_profile),
        },
        "baseline": baseline,
        "lineage": {
            "day25_comparison": {
                "path": _relative(root, day25_path),
                "sha256": sha256_file(day25_path),
            },
            "canonical_source_model": {
                "path": _relative(root, source_model),
                "sha256": source_identity_payload["model_sha256"],
                "source_checkpoint_sha256": source_identity_payload[
                    "source_checkpoint_sha256"
                ],
            },
            "onnx_model": {
                "path": _relative(root, onnx_model),
                "sha256": sha256_file(onnx_model),
            },
            "inference_spec": {
                "path": _relative(root, spec_path),
                "sha256": sha256_file(spec_path, normalize_text=True),
            },
            "environment_contract": {
                "path": spec.environment_contract_path,
                "sha256": spec.environment_contract_sha256,
                "contract_id": spec.environment_contract_id,
            },
            "probe_states": {
                "path": _relative(root, probe_states),
                "sha256": sha256_file(probe_states),
                "count": int(len(observations)),
            },
            "config": {
                "path": _relative(root, config_path),
                "sha256": sha256_file(config_path, normalize_text=True),
            },
        },
        "official_references": OFFICIAL_REFERENCES,
        "blockers": blockers,
        "reproduction": {
            "command": "python -m scripts.deployment.check_tensorrt_environment --force",
            "optional_dependency_isolation": "TensorRT may be supplied through an isolated optional environment/site-packages path; the Day 22-25 base environment is not modified.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["output"] = _relative(root, output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--day25-comparison", type=Path, default=DEFAULT_DAY25_COMPARISON
    )
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--device-index", type=int, default=DEFAULT_DEVICE_INDEX)
    parser.add_argument("--tensorrt-site-packages", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_preflight(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"TensorRT preflight failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": result["output"],
                "blockers": result["blockers"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
