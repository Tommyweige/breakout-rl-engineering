"""Export the Day 21 final policy to a checked, browser-ready ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from breakout_rl.analysis.q_values import load_probe_states
from breakout_rl.evaluation_contract import (
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.inference import (
    InferenceSpec,
    load_inference_spec,
    prepare_model_input,
    q_values_to_action,
    validate_action_meanings,
)
from breakout_rl.models.factory import build_q_network, checkpoint_architecture
from breakout_rl.onnx_artifacts import inspect_onnx_model


DEFAULT_SOURCE_MODEL = Path("assets/day21/models/final_model/model.pt")
DEFAULT_SOURCE_METADATA = Path("assets/day21/models/final_model/metadata.json")
DEFAULT_SPEC = Path("configs/inference/inference_spec.json")
DEFAULT_PROBE_STATES = Path("assets/day22/inference/probe_states.npz")
DEFAULT_REFERENCE = Path("assets/day22/inference/pytorch_reference.npz")
DEFAULT_OUTPUT = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_OUTPUT_METADATA = Path(
    "assets/day22/models/final_model/model.onnx.metadata.json"
)
DEFAULT_FIXTURE_METADATA = Path("assets/day22/inference/metadata.json")
DEFAULT_OPSET_VERSION = 17


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of one artifact without changing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: expected a JSON object")
    return dict(value)


def _repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "eval" / "breakout_contract_v2.json").is_file():
            return candidate
    raise FileNotFoundError(
        "could not locate repository root from the current working directory"
    )


def _relative_reference(path: str | Path, *, root: Path) -> str:
    source = Path(path)
    resolved = source.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "deployment metadata only permits repository-relative artifact paths: "
            f"{source}"
        ) from error


def _source_identity(
    source_model: Path,
    source_metadata: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    metadata = _json_object(source_metadata)
    model_hash = sha256_file(source_model)
    declared_hash = metadata.get("model_sha256")
    if declared_hash != model_hash:
        raise ValueError(
            "source model hash does not match Day 21 metadata: "
            f"declared={declared_hash!r}, observed={model_hash}"
        )
    if metadata.get("artifact_type") != "canonical_final_model":
        raise ValueError("source metadata is not the Day 21 canonical final model")
    source_checkpoint = metadata.get("source_checkpoint")
    if not isinstance(source_checkpoint, Mapping):
        raise ValueError("Day 21 metadata is missing source checkpoint provenance")
    checkpoint_hash = source_checkpoint.get("sha256")
    if (
        not isinstance(checkpoint_hash, str)
        or len(checkpoint_hash) != 64
        or any(
            character not in "0123456789abcdef" for character in checkpoint_hash.lower()
        )
    ):
        raise ValueError("Day 21 metadata has no valid source checkpoint SHA256")

    identity = {
        "model_path": _relative_reference(source_model, root=root),
        "model_sha256": model_hash,
        "algorithm": metadata.get("algorithm"),
        "architecture": metadata.get("architecture"),
        "hidden_dim": metadata.get("hidden_dim"),
        "observation_shape": metadata.get("observation_shape"),
        "num_actions": metadata.get("num_actions"),
        "parameter_count": metadata.get("parameter_count"),
        "training_seed": metadata.get("training_seed"),
        "training_transitions": metadata.get("training_transitions"),
        "source_run_id": metadata.get("source_run_id"),
        "source_stage": metadata.get("source_stage"),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_checkpoint_step": source_checkpoint.get("step"),
    }
    required = (
        "algorithm",
        "architecture",
        "hidden_dim",
        "observation_shape",
        "num_actions",
        "parameter_count",
        "source_run_id",
    )
    missing = [name for name in required if identity.get(name) is None]
    if missing:
        raise ValueError(
            "Day 21 source metadata is missing model identity fields: "
            + ", ".join(missing)
        )
    return identity


def _validate_environment_contract(spec: InferenceSpec, *, root: Path) -> Path:
    contract_path = (root / spec.environment_contract_path).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    observed_hash = sha256_file(contract_path)
    if observed_hash != spec.environment_contract_sha256:
        raise ValueError(
            "inference spec is bound to a different Contract v2 artifact: "
            f"declared={spec.environment_contract_sha256}, observed={observed_hash}"
        )
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    if contract.contract_id != spec.environment_contract_id:
        raise ValueError(
            "inference spec environment contract id does not match the loaded contract"
        )
    return contract_path


def _load_deployment_model(
    source_model: Path,
    *,
    device: torch.device,
    identity: Mapping[str, Any],
    spec: InferenceSpec,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(source_model, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Day 21 model checkpoint must contain a mapping")
    architecture = checkpoint_architecture(payload)
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("Day 21 model checkpoint is missing model_config")
    input_shape = tuple(int(value) for value in model_config.get("input_shape", ()))
    hidden_dim = int(model_config.get("hidden_dim", 0))
    num_actions = int(model_config.get("num_actions", 0))
    if input_shape != tuple(spec.source_observation_shape):
        raise ValueError(
            "source model input shape does not match inference spec: "
            f"model={input_shape}, spec={spec.source_observation_shape}"
        )
    if num_actions != len(spec.action_meanings):
        raise ValueError(
            "source model action count does not match inference spec: "
            f"model={num_actions}, spec={len(spec.action_meanings)}"
        )
    if architecture != identity["architecture"] or hidden_dim != int(
        identity["hidden_dim"]
    ):
        raise ValueError("source model architecture does not match Day 21 metadata")
    if model_config.get("architecture") != architecture:
        raise ValueError(
            "source model_config architecture conflicts with checkpoint metadata"
        )
    if payload.get("algorithm") != identity["algorithm"]:
        raise ValueError("source model algorithm does not match Day 21 metadata")
    if payload.get("global_step") != identity["source_checkpoint_step"]:
        raise ValueError("source model checkpoint step does not match Day 21 metadata")
    if payload.get("contract_id") != spec.environment_contract_id:
        raise ValueError(
            "source model Contract v2 id does not match the inference spec"
        )
    payload_metadata = payload.get("metadata")
    if isinstance(payload_metadata, Mapping):
        payload_source_checkpoint = payload_metadata.get("source_checkpoint")
        if (
            isinstance(payload_source_checkpoint, Mapping)
            and payload_source_checkpoint.get("sha256")
            != identity["source_checkpoint_sha256"]
        ):
            raise ValueError(
                "source checkpoint provenance conflicts inside the model payload"
            )
    state_dict = payload.get("online_network")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Day 21 model checkpoint is missing online_network")
    model = build_q_network(
        architecture,
        num_actions=num_actions,
        input_shape=input_shape,  # type: ignore[arg-type]
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    observed_parameters = sum(parameter.numel() for parameter in model.parameters())
    if observed_parameters != int(identity["parameter_count"]):
        raise ValueError(
            "source model parameter count does not match Day 21 metadata: "
            f"model={observed_parameters}, metadata={identity['parameter_count']}"
        )
    return model, dict(payload), dict(model_config)


def _cuda_runtime(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "formal Day 22 export is blocked: CUDA source-model validation is required"
        )
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    torch.cuda.set_device(index)
    return {
        "framework": "PyTorch",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": True,
        "device": f"cuda:{index}",
        "cuda_device_index": index,
        "gpu_model": torch.cuda.get_device_name(index),
    }


def _run_cuda_reference(
    model: torch.nn.Module,
    observations: np.ndarray,
    *,
    spec: InferenceSpec,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_inputs = prepare_model_input(observations, device=device, spec=spec)
    with torch.inference_mode():
        outputs = model(model_inputs)
    torch.cuda.synchronize(device)
    if not isinstance(outputs, torch.Tensor):
        raise TypeError("source model must return a torch.Tensor")
    if outputs.dtype != torch.float32:
        raise TypeError("source model output must have dtype torch.float32")
    expected_shape = (int(observations.shape[0]), len(spec.action_meanings))
    if tuple(outputs.shape) != expected_shape:
        raise ValueError(
            "source model output does not match inference spec: "
            f"observed={tuple(outputs.shape)}, expected={expected_shape}"
        )
    if not torch.isfinite(outputs).all().item():
        raise ValueError("source model output contains non-finite Q-values")
    inputs_array = np.ascontiguousarray(model_inputs.detach().cpu().numpy())
    q_values = np.ascontiguousarray(outputs.detach().cpu().numpy())
    actions = np.asarray(
        q_values_to_action(q_values, spec=spec),
        dtype=np.int64,
    )
    return inputs_array, q_values, actions


def _reference_metadata(
    *,
    identity: Mapping[str, Any],
    spec: InferenceSpec,
    spec_hash: str,
    spec_file: Path,
    probes: Path,
    probe_hash: str,
    observations: np.ndarray,
    model_inputs: np.ndarray,
    q_values: np.ndarray,
    actions: np.ndarray,
    runtime: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "pytorch_cuda_golden_reference",
        "model_identity": dict(identity),
        "source_model_sha256": identity["model_sha256"],
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "inference_spec": {
            "path": _relative_reference(spec_file, root=root),
            "sha256": spec_hash,
            "contract_id": spec.contract_id,
        },
        "probe_states": {
            "path": _relative_reference(probes, root=root),
            "sha256": probe_hash,
            "count": int(observations.shape[0]),
            "shape": list(observations.shape[1:]),
            "dtype": str(observations.dtype),
        },
        "runtime": dict(runtime),
        "input": {
            "name": spec.input_name,
            "shape": list(model_inputs.shape),
            "dtype": str(model_inputs.dtype),
            "normalization_divisor": spec.normalization_divisor,
        },
        "output": {
            "name": spec.output_name,
            "shape": list(q_values.shape),
            "dtype": str(q_values.dtype),
            "action_meanings": list(spec.action_meanings),
            "greedy_rule": spec.greedy_rule,
        },
        "arrays": {
            "model_inputs_sha256": _sha256_array(model_inputs),
            "q_values_sha256": _sha256_array(q_values),
            "greedy_actions_sha256": _sha256_array(actions),
        },
    }


def _write_reference(
    path: Path,
    *,
    observations: np.ndarray,
    model_inputs: np.ndarray,
    q_values: np.ndarray,
    actions: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
    np.savez_compressed(
        path,
        observations=np.ascontiguousarray(observations),
        model_inputs=np.ascontiguousarray(model_inputs, dtype=np.float32),
        q_values=np.ascontiguousarray(q_values, dtype=np.float32),
        greedy_actions=np.ascontiguousarray(actions, dtype=np.int64),
        metadata_json=np.array(metadata_json),
    )


def _load_and_verify_reference(
    path: Path,
    *,
    expected_metadata: Mapping[str, Any],
    observations: np.ndarray,
    model_inputs: np.ndarray,
    q_values: np.ndarray,
    actions: np.ndarray,
) -> tuple[dict[str, Any], float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "observations",
            "model_inputs",
            "q_values",
            "greedy_actions",
            "metadata_json",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"golden reference is missing arrays: {', '.join(sorted(missing))}"
            )
        raw_metadata = archive["metadata_json"].item()
        if not isinstance(raw_metadata, str):
            raise ValueError("golden reference metadata_json must be a JSON string")
        metadata = json.loads(raw_metadata)
        if not isinstance(metadata, Mapping):
            raise ValueError("golden reference metadata_json must be an object")
        metadata = dict(metadata)
        for key in ("source_model_sha256", "source_checkpoint_sha256"):
            if metadata.get(key) != expected_metadata.get(key):
                raise ValueError(f"golden reference provenance mismatch: {key}")
        expected_spec = expected_metadata["inference_spec"]
        expected_probe = expected_metadata["probe_states"]
        if metadata.get("inference_spec") != expected_spec:
            raise ValueError("golden reference is bound to a different inference spec")
        if metadata.get("probe_states") != expected_probe:
            raise ValueError("golden reference is bound to different probe states")
        runtime = metadata.get("runtime")
        if not isinstance(runtime, Mapping) or not runtime.get("cuda_available"):
            raise ValueError("golden reference is not CUDA-backed")
        stored_observations = archive["observations"]
        stored_inputs = archive["model_inputs"]
        stored_q_values = archive["q_values"]
        stored_actions = archive["greedy_actions"]
        if stored_observations.dtype != np.uint8 or not np.array_equal(
            stored_observations,
            observations,
        ):
            raise ValueError("golden reference observations do not match probe states")
        if stored_inputs.dtype != np.float32 or not np.array_equal(
            stored_inputs,
            model_inputs,
        ):
            raise ValueError(
                "golden reference normalized inputs do not match the adapter"
            )
        if (
            stored_q_values.dtype != np.float32
            or stored_q_values.shape != q_values.shape
        ):
            raise ValueError("golden reference Q-value shape/dtype does not match")
        if stored_actions.dtype != np.int64 or not np.array_equal(
            stored_actions, actions
        ):
            raise ValueError("golden reference greedy actions do not match")
        max_abs_diff = float(np.max(np.abs(stored_q_values - q_values)))
        if not np.allclose(stored_q_values, q_values, rtol=1e-5, atol=1e-6):
            raise ValueError(
                "golden reference Q-values do not match the CUDA source model: "
                f"max_abs_diff={max_abs_diff}"
            )
    return metadata, max_abs_diff


def _validate_graph_contract(
    graph: Mapping[str, Any],
    *,
    spec: InferenceSpec,
) -> None:
    inputs = graph.get("inputs")
    outputs = graph.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("ONNX graph inspection did not return input/output lists")
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("ONNX graph must expose exactly one input and one output")
    input_info = inputs[0]
    output_info = outputs[0]
    expected_input_shape = list(spec.input_shape)
    expected_output_shape = list(spec.output_shape)
    if (
        input_info.get("name") != spec.input_name
        or input_info.get("dtype") != spec.input_dtype
        or input_info.get("shape") != expected_input_shape
    ):
        raise ValueError(
            "ONNX input contract mismatch: "
            f"observed={input_info}, expected={{'name': {spec.input_name!r}, "
            f"'dtype': {spec.input_dtype!r}, 'shape': {expected_input_shape!r}}}"
        )
    if (
        output_info.get("name") != spec.output_name
        or output_info.get("dtype") != spec.output_dtype
        or output_info.get("shape") != expected_output_shape
    ):
        raise ValueError(
            "ONNX output contract mismatch: "
            f"observed={output_info}, expected={{'name': {spec.output_name!r}, "
            f"'dtype': {spec.output_dtype!r}, 'shape': {expected_output_shape!r}}}"
        )


def _export_graph(
    model: torch.nn.Module,
    model_inputs: np.ndarray,
    *,
    output: Path,
    spec: InferenceSpec,
    opset_version: int,
    display_path: str,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp.onnx")
    if temporary_output.exists():
        temporary_output.unlink()
    dummy_input = torch.from_numpy(model_inputs[:1]).to(next(model.parameters()).device)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (dummy_input,),
                str(temporary_output),
                input_names=[spec.input_name],
                output_names=[spec.output_name],
                opset_version=opset_version,
                dynamic_axes={
                    spec.input_name: {0: "N"},
                    spec.output_name: {0: "N"},
                },
                export_params=True,
                do_constant_folding=True,
                keep_initializers_as_inputs=False,
                dynamo=False,
                external_data=False,
            )
        graph = inspect_onnx_model(
            temporary_output,
            check=True,
            display_path=display_path,
        )
        _validate_graph_contract(graph, spec=spec)
        temporary_output.replace(output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()
    return graph


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_final_model(
    *,
    source_model: str | Path = DEFAULT_SOURCE_MODEL,
    source_metadata: str | Path = DEFAULT_SOURCE_METADATA,
    spec_path: str | Path = DEFAULT_SPEC,
    probe_states: str | Path = DEFAULT_PROBE_STATES,
    reference_path: str | Path = DEFAULT_REFERENCE,
    output: str | Path = DEFAULT_OUTPUT,
    output_metadata: str | Path = DEFAULT_OUTPUT_METADATA,
    fixture_metadata: str | Path = DEFAULT_FIXTURE_METADATA,
    device_index: int = 0,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> dict[str, Any]:
    """Run the formal CUDA reference, ONNX export, and host graph checks."""

    if device_index < 0:
        raise ValueError("device_index must not be negative")
    if opset_version < 1:
        raise ValueError("opset_version must be positive")
    root = _repository_root()
    source_model_path = Path(source_model).resolve()
    source_metadata_path = Path(source_metadata).resolve()
    spec_file = Path(spec_path).resolve()
    probe_file = Path(probe_states).resolve()
    reference_file = Path(reference_path).resolve()
    output_file = Path(output).resolve()
    output_metadata_file = Path(output_metadata).resolve()
    fixture_metadata_file = Path(fixture_metadata).resolve()
    spec = load_inference_spec(spec_file)
    spec_hash = sha256_file(spec_file)
    _validate_environment_contract(spec, root=root)
    identity = _source_identity(source_model_path, source_metadata_path, root=root)
    observations, probe_metadata = load_probe_states(probe_file)
    probe_hash = sha256_file(probe_file)
    if probe_metadata.get("contract_id") != spec.environment_contract_id:
        raise ValueError(
            "probe states are not generated under the declared environment contract"
        )
    if probe_metadata.get("contract_sha256") != spec.environment_contract_sha256:
        raise ValueError("probe states are bound to a different Contract v2 hash")
    raw_action_meanings = probe_metadata.get("action_meanings")
    if not isinstance(raw_action_meanings, Sequence):
        raise ValueError("probe states are missing runtime ALE action meanings")
    validate_action_meanings(raw_action_meanings, spec=spec)

    device = torch.device(f"cuda:{device_index}")
    runtime = _cuda_runtime(device)
    source_model_hash_before = sha256_file(source_model_path)
    model, payload, model_config = _load_deployment_model(
        source_model_path,
        device=device,
        identity=identity,
        spec=spec,
    )
    pre_inputs, pre_q_values, pre_actions = _run_cuda_reference(
        model,
        observations,
        spec=spec,
        device=device,
    )
    reference_metadata = _reference_metadata(
        identity=identity,
        spec=spec,
        spec_hash=spec_hash,
        spec_file=spec_file,
        probes=probe_file,
        probe_hash=probe_hash,
        observations=observations,
        model_inputs=pre_inputs,
        q_values=pre_q_values,
        actions=pre_actions,
        runtime=runtime,
        root=root,
    )
    reference_created = not reference_file.is_file()
    if reference_created:
        _write_reference(
            reference_file,
            observations=observations,
            model_inputs=pre_inputs,
            q_values=pre_q_values,
            actions=pre_actions,
            metadata=reference_metadata,
        )
    reference_metadata, reference_max_abs_diff = _load_and_verify_reference(
        reference_file,
        expected_metadata=reference_metadata,
        observations=observations,
        model_inputs=pre_inputs,
        q_values=pre_q_values,
        actions=pre_actions,
    )
    reference_hash_before = sha256_file(reference_file)

    graph = _export_graph(
        model,
        pre_inputs,
        output=output_file,
        spec=spec,
        opset_version=opset_version,
        display_path=_relative_reference(output_file, root=root),
    )
    post_inputs, post_q_values, post_actions = _run_cuda_reference(
        model,
        observations,
        spec=spec,
        device=device,
    )
    source_post_export_max_abs_diff = float(
        np.max(np.abs(pre_q_values - post_q_values))
    )
    if (
        not np.array_equal(pre_inputs, post_inputs)
        or not np.array_equal(
            pre_actions,
            post_actions,
        )
        or not np.allclose(pre_q_values, post_q_values, rtol=0.0, atol=0.0)
    ):
        raise RuntimeError(
            "source-model CUDA reference changed across export: "
            f"max_abs_diff={source_post_export_max_abs_diff}"
        )
    source_model_hash_after = sha256_file(source_model_path)
    if source_model_hash_before != source_model_hash_after:
        raise RuntimeError("source checkpoint/model changed during ONNX export")
    identity_after = _source_identity(
        source_model_path,
        source_metadata_path,
        root=root,
    )
    if identity_after != identity:
        raise RuntimeError("source checkpoint provenance changed during ONNX export")
    if sha256_file(spec_file) != spec_hash:
        raise RuntimeError("inference spec changed during ONNX export")
    if (
        sha256_file(root / spec.environment_contract_path)
        != spec.environment_contract_sha256
    ):
        raise RuntimeError("Contract v2 artifact changed during ONNX export")
    if sha256_file(probe_file) != probe_hash:
        raise RuntimeError("probe-state fixture changed during ONNX export")
    if sha256_file(reference_file) != reference_hash_before:
        raise RuntimeError("CUDA golden reference fixture changed during ONNX export")

    batch_validation = []
    for batch_size in (1, 4):
        batch_validation.append(
            {
                "batch_size": batch_size,
                "source_model_input_shape": list(pre_inputs[:batch_size].shape),
                "source_model_output_shape": list(pre_q_values[:batch_size].shape),
                "q_values_sha256": _sha256_array(pre_q_values[:batch_size]),
                "greedy_actions_sha256": _sha256_array(pre_actions[:batch_size]),
            }
        )
    source_validation = {
        "backend": "PyTorch",
        "purpose": "formal source-model reference; CUDA is mandatory",
        "runtime": runtime,
        "before_export": {
            "input_shape": list(pre_inputs.shape),
            "q_values_shape": list(pre_q_values.shape),
            "q_values_sha256": _sha256_array(pre_q_values),
            "greedy_actions_sha256": _sha256_array(pre_actions),
        },
        "after_export": {
            "input_shape": list(post_inputs.shape),
            "q_values_shape": list(post_q_values.shape),
            "q_values_sha256": _sha256_array(post_q_values),
            "greedy_actions_sha256": _sha256_array(post_actions),
        },
        "max_abs_diff_before_vs_after": source_post_export_max_abs_diff,
        "actions_match": bool(np.array_equal(pre_actions, post_actions)),
        "provenance_revalidated_after_export": True,
    }
    onnx_hash = sha256_file(output_file)
    onnx_size = output_file.stat().st_size
    output_metadata_value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "breakout_onnx_export_metadata",
        "model_path": _relative_reference(output_file, root=root),
        "model_sha256": onnx_hash,
        "model_size_bytes": onnx_size,
        "source_model": dict(identity),
        "source_model_sha256": identity["model_sha256"],
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "source_checkpoint_step": identity["source_checkpoint_step"],
        "inference_spec": {
            "path": _relative_reference(spec_file, root=root),
            "sha256": spec_hash,
            "contract_id": spec.contract_id,
        },
        "probe_states": {
            "path": _relative_reference(probe_file, root=root),
            "sha256": probe_hash,
            "count": int(observations.shape[0]),
        },
        "pytorch_reference": {
            "path": _relative_reference(reference_file, root=root),
            "sha256": reference_hash_before,
            "status": "created_or_validated_before_export",
            "max_abs_diff_to_current_cuda_reference": reference_max_abs_diff,
        },
        "source_model_validation": source_validation,
        "onnx_export": {
            "exporter": "torch.onnx.export",
            "dynamo": False,
            "opset_version": opset_version,
            "external_data": False,
            "input_names": [spec.input_name],
            "output_names": [spec.output_name],
            "dynamic_axes": {
                spec.input_name: {"0": "N"},
                spec.output_name: {"0": "N"},
            },
            "source_model_config": model_config,
        },
        "dynamic_batch_validation": {
            "dimension": "N",
            "declared_input_shape": list(spec.input_shape),
            "declared_output_shape": list(spec.output_shape),
            "tested_batch_sizes": [1, 4],
            "source_model_batches": batch_validation,
            "onnx_runtime_execution": "deferred to Day 23",
        },
        "host_graph_validation": {
            "backend": "CPU host-side graph inspection",
            "checker": "onnx.checker.check_model",
            "checker_passed": True,
            "graph": graph,
        },
        "golden_fixture_provenance": {
            "source_model_sha256": identity["model_sha256"],
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "probe_states_sha256": probe_hash,
            "pytorch_reference_sha256": reference_hash_before,
        },
        "generation": {
            "command": (
                "python -m scripts.deployment.export_onnx "
                f"--source-model {_relative_reference(source_model_path, root=root)} "
                f"--source-metadata {_relative_reference(source_metadata_path, root=root)}"
            ),
            "repository_relative_paths_only": True,
        },
    }
    _write_json(output_metadata_file, output_metadata_value)

    fixture_metadata_value = {
        "schema_version": 1,
        "artifact_type": "day22_inference_fixture_manifest",
        "inference_spec": output_metadata_value["inference_spec"],
        "source_model": output_metadata_value["source_model"],
        "probe_states": output_metadata_value["probe_states"],
        "pytorch_reference": output_metadata_value["pytorch_reference"],
        "onnx_model": {
            "path": output_metadata_value["model_path"],
            "sha256": onnx_hash,
            "size_bytes": onnx_size,
        },
        "source_model_validation": source_validation,
        "host_graph_validation": output_metadata_value["host_graph_validation"],
        "dynamic_batch_validation": output_metadata_value["dynamic_batch_validation"],
        "action_contract": {
            "meanings": list(spec.action_meanings),
            "greedy_rule": spec.greedy_rule,
        },
        "deployment": {
            "runtime": "browser client-side",
            "model_relative_url": spec.model_relative_url,
            "same_origin_relative_url": True,
        },
    }
    _write_json(fixture_metadata_file, fixture_metadata_value)
    return output_metadata_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Day 21 final model on CUDA and export a checked FP32 ONNX graph."
        )
    )
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-metadata", type=Path, default=DEFAULT_OUTPUT_METADATA)
    parser.add_argument(
        "--fixture-metadata", type=Path, default=DEFAULT_FIXTURE_METADATA
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metadata = export_final_model(
            source_model=args.source_model,
            source_metadata=args.source_metadata,
            spec_path=args.spec,
            probe_states=args.probe_states,
            reference_path=args.reference,
            output=args.output,
            output_metadata=args.output_metadata,
            fixture_metadata=args.fixture_metadata,
            device_index=args.device_index,
            opset_version=args.opset,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 22 ONNX export failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "model": metadata["model_path"],
                "model_sha256": metadata["model_sha256"],
                "checker_passed": metadata["host_graph_validation"]["checker_passed"],
                "source_cuda_device": metadata["source_model_validation"]["runtime"][
                    "device"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
