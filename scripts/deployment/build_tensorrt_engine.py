"""Build a hardware-specific TensorRT engine for the canonical Breakout ONNX model."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from breakout_rl.artifacts import repository_relative_path, repository_root, sha256_file
from breakout_rl.deployment import source_identity, validate_environment_contract
from breakout_rl.inference import load_inference_spec
from breakout_rl.tensorrt import (
    TENSORRT_STATUS_READY,
    TensorRTBlockedError,
    create_strongly_typed_network,
    disable_tf32,
    validate_workspace_budget,
)


DEFAULT_CONFIG = Path("configs/inference/tensorrt_experiment.json")
DEFAULT_PREFLIGHT = Path("assets/day26/tensorrt-preflight.json")
DEFAULT_SOURCE_MODEL = Path("assets/day21/models/final_model/model.pt")
DEFAULT_SOURCE_METADATA = Path("assets/day21/models/final_model/metadata.json")
DEFAULT_ONNX_FP32 = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_ONNX_FP16 = Path("assets/day25/models/model.fp16.onnx")
DEFAULT_PRECISION = "float32"


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)


def _relative(root: Path, path: Path) -> str:
    return repository_relative_path(path, root=root)


def _resolve(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _load_tensorrt(site_packages: Path | None) -> Any:
    if site_packages is not None:
        resolved = site_packages.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        sys.path.insert(0, str(resolved))
    try:
        import tensorrt as trt
    except Exception as error:  # pragma: no cover - optional environment
        raise TensorRTBlockedError(
            "TensorRT Python API is unavailable; use a preflight with CLI_ONLY "
            "or install TensorRT in an isolated optional environment"
        ) from error
    return trt


def _verify_inputs(
    *,
    root: Path,
    config: Mapping[str, Any],
    precision: str,
    source_model: Path,
    source_metadata: Path,
    onnx_model: Path,
    day25_comparison: Path,
) -> dict[str, Any]:
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("TensorRT config is missing source")
    expected_model = str(source["canonical_source_model_sha256"])
    identity = source_identity(source_model, source_metadata, root=root)
    if identity["model_sha256"] != expected_model:
        raise ValueError("Day 21 source model hash does not match TensorRT config")
    expected_onnx = str(
        source["onnx_fp32_sha256"]
        if precision == "float32"
        else source["onnx_fp16_sha256"]
    )
    if sha256_file(onnx_model) != expected_onnx:
        raise ValueError(f"{precision} source ONNX hash does not match TensorRT config")
    day25 = _json_object(day25_comparison)
    if sha256_file(day25_comparison) != source["day25_comparison_sha256"]:
        raise ValueError("Day 25 comparison hash does not match TensorRT config")
    if day25.get("status") != "completed":
        raise ValueError("Day 25 comparison is not a completed source artifact")
    return {
        "canonical_source_model": {
            "path": _relative(root, source_model),
            "sha256": identity["model_sha256"],
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "source_checkpoint_step": identity["source_checkpoint_step"],
        },
        "onnx": {
            "path": _relative(root, onnx_model),
            "sha256": sha256_file(onnx_model),
            "precision": precision,
        },
        "day25_comparison": {
            "path": _relative(root, day25_comparison),
            "sha256": sha256_file(day25_comparison),
        },
    }


def _build_with_python_api(
    *,
    trt: Any,
    onnx_model: Path,
    output: Path,
    precision: str,
    workspace_bytes: int,
    profile: Mapping[str, int],
) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = create_strongly_typed_network(builder, trt)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_model.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parser failed: " + " | ".join(errors))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError(
            "TensorRT network must expose one input and one output: "
            f"inputs={network.num_inputs}, outputs={network.num_outputs}"
        )
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    config = builder.create_builder_config()
    tf32_enabled = disable_tf32(config, trt)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    optimization_profile = builder.create_optimization_profile()
    shape_min = (profile["min_batch"], 4, 84, 84)
    shape_opt = (profile["opt_batch"], 4, 84, 84)
    shape_max = (profile["max_batch"], 4, 84, 84)
    # TensorRT 11's Python binding mutates the profile and returns ``None``;
    # older bindings may return a truthy success value.  Treat an exception as
    # rejection instead of interpreting the normal ``None`` return as failure.
    optimization_profile.set_shape(
        input_tensor.name,
        shape_min,
        shape_opt,
        shape_max,
    )
    config.add_optimization_profile(optimization_profile)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT builder returned no serialized engine")
    engine_bytes = bytes(serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(engine_bytes)
    return {
        "build_api": "TensorRT Python API",
        "engine_size_bytes": len(engine_bytes),
        "build_seconds": build_seconds,
        "network_typing": "strongly_typed",
        "tf32_enabled": tf32_enabled,
        "parser": {
            "input_name": input_tensor.name,
            "input_dtype": str(input_tensor.dtype),
            "input_shape": [int(value) for value in input_tensor.shape],
            "output_name": output_tensor.name,
            "output_dtype": str(output_tensor.dtype),
            "output_shape": [int(value) for value in output_tensor.shape],
        },
        "precision_control": (
            "source ONNX tensor types; TensorRT 11 strongly typed network"
            if precision == "float16"
            else "strict FP32 source ONNX tensor types; TF32 disabled"
        ),
    }


def _build_with_trtexec(
    *,
    executable: str,
    onnx_model: Path,
    output: Path,
    precision: str,
    workspace_bytes: int,
    profile: Mapping[str, int],
    log_path: Path,
) -> dict[str, Any]:
    raise TensorRTBlockedError(
        "trtexec-only builds cannot prove the required strongly typed network; "
        "formal Day 26 engines require the verified TensorRT Python API"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def shape(batch: int) -> str:
        return f"{batch}x4x84x84"

    command = [
        executable,
        f"--onnx={onnx_model}",
        f"--saveEngine={output}",
        f"--minShapes=observation:{shape(profile['min_batch'])}",
        f"--optShapes=observation:{shape(profile['opt_batch'])}",
        f"--maxShapes=observation:{shape(profile['max_batch'])}",
        f"--memPoolSize=workspace:{workspace_bytes // (1024 * 1024)}",
        "--skipInference",
    ]
    if precision == "float16":
        command.append("--fp16")
    command.append("--noTF32")
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "COMMAND: "
        + " ".join(command)
        + "\n\n"
        + completed.stdout
        + "\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"trtexec failed with exit code {completed.returncode}; see {log_path}"
        )
    return {
        "build_api": "trtexec CLI",
        "engine_size_bytes": output.stat().st_size,
        "build_seconds": elapsed,
        "network_typing": "strongly_typed",
        "tf32_enabled": False,
        "command": " ".join(command),
        "log_path": log_path.as_posix(),
        "precision_control": (
            "trtexec --fp16; TF32 disabled"
            if precision == "float16"
            else "strict FP32; TF32 disabled"
        ),
    }


def build_engine(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    config_path = _resolve(root, args.config)
    config = _json_object(config_path)
    if config.get("artifact_type") != "day26_tensorrt_experiment":
        raise ValueError("TensorRT config has an unexpected artifact_type")
    if args.precision not in {"float32", "float16"}:
        raise ValueError("precision must be float32 or float16")
    source_config = config.get("source")
    if not isinstance(source_config, Mapping):
        raise ValueError("TensorRT config is missing source")
    spec_path = _resolve(root, Path(str(source_config["inference_spec_path"])))
    spec = load_inference_spec(spec_path)
    validate_environment_contract(spec, root=root)
    if source_config.get("environment_contract_path") != spec.environment_contract_path:
        raise ValueError(
            "TensorRT config Contract v2 path does not match the inference spec"
        )
    build_config = config["build"]
    profile = build_config["optimization_profile"]
    if build_config.get("network_typing") != "strongly_typed":
        raise TensorRTBlockedError(
            "TensorRT config must require a strongly typed network"
        )
    if build_config.get("tf32_enabled") is not False:
        raise TensorRTBlockedError("TensorRT config must disable TF32")
    preflight_path = _resolve(root, args.preflight)
    preflight = _json_object(preflight_path)
    status = preflight.get("status")
    if status != TENSORRT_STATUS_READY:
        raise TensorRTBlockedError(
            "Formal TensorRT engine build requires a READY preflight with a verified strongly typed Python API: "
            f"status={status!r}, blockers={preflight.get('blockers', [])}"
        )
    capabilities = preflight.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise ValueError("preflight is missing capabilities")
    if capabilities.get("strongly_typed_network_available") is not True:
        raise TensorRTBlockedError(
            "preflight did not verify the required strongly typed network"
        )
    if capabilities.get("network_typing") != "strongly_typed":
        raise TensorRTBlockedError(
            "preflight network typing policy is not strongly_typed"
        )
    if capabilities.get("tf32_enabled") is not False:
        raise TensorRTBlockedError("preflight TF32 policy is not disabled")
    preflight_lineage = preflight.get("lineage")
    if not isinstance(preflight_lineage, Mapping):
        raise ValueError("preflight is missing lineage")
    preflight_config = preflight_lineage.get("config")
    if not isinstance(preflight_config, Mapping):
        raise ValueError("preflight is missing config lineage")
    if preflight_config.get("path") != _relative(root, config_path):
        raise ValueError("preflight was generated from a different TensorRT config")
    if preflight_config.get("sha256") != sha256_file(config_path, normalize_text=True):
        raise ValueError(
            "preflight config hash does not match the current TensorRT config"
        )
    source_model = _resolve(root, args.source_model)
    source_metadata = _resolve(root, args.source_metadata)
    onnx_model = _resolve(
        root, args.onnx_fp32 if args.precision == "float32" else args.onnx_fp16
    )
    day25_comparison = _resolve(root, args.day25_comparison)
    lineage = _verify_inputs(
        root=root,
        config=config,
        precision=args.precision,
        source_model=source_model,
        source_metadata=source_metadata,
        onnx_model=onnx_model,
        day25_comparison=day25_comparison,
    )
    lineage["inference_spec"] = {
        "path": _relative(root, spec_path),
        "sha256": sha256_file(spec_path, normalize_text=True),
    }
    lineage["environment_contract"] = {
        "path": spec.environment_contract_path,
        "sha256": spec.environment_contract_sha256,
        "contract_id": spec.environment_contract_id,
    }
    if not torch.cuda.is_available():
        raise TensorRTBlockedError(
            "TensorRT engine build requires an NVIDIA CUDA device"
        )
    if args.device_index < 0 or args.device_index >= torch.cuda.device_count():
        raise TensorRTBlockedError(
            f"CUDA device index {args.device_index} is unavailable"
        )
    gpu = torch.cuda.get_device_properties(args.device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(args.device_index)
    workspace_bytes = int(args.workspace_bytes or build_config["workspace_limit_bytes"])
    headroom_bytes = int(build_config["required_headroom_bytes"])
    validate_workspace_budget(
        workspace_bytes=workspace_bytes,
        free_bytes=int(free_bytes),
        headroom_bytes=headroom_bytes,
    )
    output_config_key = (
        "fp32_engine_path" if args.precision == "float32" else "fp16_engine_path"
    )
    metadata_config_key = (
        "fp32_metadata_path" if args.precision == "float32" else "fp16_metadata_path"
    )
    output = (root / str(build_config[output_config_key])).resolve()
    metadata_path = (root / str(build_config[metadata_config_key])).resolve()
    if (output.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError(
            f"TensorRT engine artifact already exists: {output}; use --force"
        )
    build_info: dict[str, Any]
    trt = _load_tensorrt(args.tensorrt_site_packages)
    build_info = _build_with_python_api(
        trt=trt,
        onnx_model=onnx_model,
        output=output,
        precision=args.precision,
        workspace_bytes=workspace_bytes,
        profile=profile,
    )
    trt_version = str(getattr(trt, "__version__", "unknown"))
    metadata = {
        "schema_version": 1,
        "artifact_type": "day26_tensorrt_engine",
        "engine_path": _relative(root, output),
        "engine_sha256": sha256_file(output),
        "engine_size_bytes": output.stat().st_size,
        "precision": args.precision,
        "network_typing": build_info["network_typing"],
        "tf32_enabled": bool(build_info["tf32_enabled"]),
        "lineage": lineage,
        "hardware": {
            "device_index": args.device_index,
            "gpu_model": torch.cuda.get_device_name(args.device_index),
            "compute_capability": f"{gpu.major}.{gpu.minor}",
            "total_memory_bytes": int(total_bytes),
            "free_memory_before_build_bytes": int(free_bytes),
        },
        "software": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "tensorrt_version": trt_version,
            "preflight_path": _relative(root, preflight_path),
            "preflight_sha256": sha256_file(preflight_path),
        },
        "memory_policy": {
            "workspace_limit_bytes": workspace_bytes,
            "required_headroom_bytes": headroom_bytes,
            "free_memory_after_build_bytes": int(
                torch.cuda.mem_get_info(args.device_index)[0]
            ),
        },
        "optimization_profile": {
            "min": [profile["min_batch"], 4, 84, 84],
            "opt": [profile["opt_batch"], 4, 84, 84],
            "max": [profile["max_batch"], 4, 84, 84],
        },
        "engine_compatibility": {
            "hardware_specific": True,
            "portable_across_gpu_architectures": False,
            "portable_across_platforms": False,
            "source_precision_contract": (
                "Day 22 FP32 ONNX"
                if args.precision == "float32"
                else "Day 25 derived typed FP16 ONNX because TensorRT 11 is strongly typed"
            ),
        },
        "build": build_info,
        "metadata_path": _relative(root, metadata_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"engine": output, "metadata": metadata_path, "metadata_payload": metadata}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--onnx-fp32", type=Path, default=DEFAULT_ONNX_FP32)
    parser.add_argument("--onnx-fp16", type=Path, default=DEFAULT_ONNX_FP16)
    parser.add_argument(
        "--day25-comparison",
        type=Path,
        default=Path("assets/day25/precision-comparison.json"),
    )
    parser.add_argument(
        "--precision", choices=("float32", "float16"), default=DEFAULT_PRECISION
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--workspace-bytes", type=int, default=None)
    parser.add_argument("--tensorrt-site-packages", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_engine(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TensorRTBlockedError,
        TypeError,
        ValueError,
    ) as error:
        print(f"TensorRT engine build failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "engine": str(result["engine"]),
                "metadata": str(result["metadata"]),
                "precision": result["metadata_payload"]["precision"],
                "engine_sha256": result["metadata_payload"]["engine_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
