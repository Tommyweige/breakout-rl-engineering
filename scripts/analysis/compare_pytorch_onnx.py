"""Validate PyTorch CUDA golden outputs against native ONNX Runtime providers."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from breakout_rl.artifacts import (
    repository_relative_path as _relative_path,
    repository_root as _repository_root,
    sha256_file as _sha256_file,
)
from breakout_rl.inference import (
    EXPECTED_ACTION_MEANINGS,
    ONNXRuntimePolicy,
    load_inference_spec,
    prepare_model_input,
    q_values_to_action,
    validate_action_meanings,
)
from breakout_rl.onnx_parity import compare_q_values


DEFAULT_SOURCE_MODEL = Path("assets/day21/models/final_model/model.pt")
DEFAULT_SOURCE_METADATA = Path("assets/day21/models/final_model/metadata.json")
DEFAULT_ONNX_MODEL = Path("assets/day22/models/final_model/model.onnx")
DEFAULT_ONNX_METADATA = Path(
    "assets/day22/models/final_model/model.onnx.metadata.json"
)
DEFAULT_SPEC = Path("configs/inference/inference_spec.json")
DEFAULT_FIXTURE_METADATA = Path("assets/day22/inference/metadata.json")
DEFAULT_PROBE_STATES = Path("assets/day22/inference/probe_states.npz")
DEFAULT_REFERENCE = Path("assets/day22/inference/pytorch_reference.npz")
DEFAULT_CONFIG = Path("configs/inference/parity_validation.json")
DEFAULT_OUTPUT = Path("assets/day23/onnx-runtime-parity.json")
DEFAULT_REPORT = Path("reports/day23-onnx-runtime-parity.md")


def _verify_declared_file_hash(
    path: Path,
    expected: Any,
    *,
    label: str,
) -> str:
    """Match a declared hash, tolerating only text checkout line endings."""

    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} must declare a SHA256 hex digest")
    observed = path.read_bytes()
    raw_hash = hashlib.sha256(observed).hexdigest()
    if raw_hash == expected:
        return raw_hash
    normalized_hash = hashlib.sha256(
        observed.replace(bytes([13, 10]), bytes([10]))
    ).hexdigest()
    if normalized_hash == expected:
        return normalized_hash
    raise ValueError(
        f"{label} SHA256 mismatch: observed_raw={raw_hash!r}, "
        f"observed_lf_normalized={normalized_hash!r}, expected={expected!r}"
    )


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def _embedded_json(npz: Any, *, path: Path) -> dict[str, Any]:
    value = npz.get("metadata_json")
    if value is None or np.asarray(value).shape != ():
        raise ValueError(f"{path}: missing scalar metadata_json array")
    try:
        parsed = json.loads(str(np.asarray(value).item()))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid embedded metadata_json") from error
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{path}: embedded metadata_json must be an object")
    return dict(parsed)


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_validation_config(path: Path) -> dict[str, Any]:
    config = _json_object(path)
    _require_equal("parity config schema_version", config.get("schema_version"), 1)
    _require_equal(
        "parity config artifact_type",
        config.get("artifact_type"),
        "day23_onnx_runtime_parity_validation",
    )
    providers = _require_mapping(config.get("providers"), label="providers")
    for key, expected in (
        ("cpu", "CPUExecutionProvider"),
        ("cuda", "CUDAExecutionProvider"),
    ):
        _require_equal(f"providers.{key}", providers.get(key), expected)
    raw_batch_sizes = config.get("batch_sizes")
    if (
        isinstance(raw_batch_sizes, (str, bytes))
        or not isinstance(raw_batch_sizes, Sequence)
        or not raw_batch_sizes
    ):
        raise ValueError("batch_sizes must be a non-empty sequence")
    batch_sizes = [int(value) for value in raw_batch_sizes]
    if any(value < 1 for value in batch_sizes) or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch_sizes must contain unique positive integers")
    reference = _require_mapping(config.get("reference"), label="reference")
    _require_equal("reference.runtime", reference.get("runtime"), "PyTorch")
    _require_equal("reference.provider", reference.get("provider"), "CUDA")
    _require_equal("reference.precision", reference.get("precision"), "float32")
    lineage = _require_mapping(config.get("lineage"), label="lineage")
    for field in (
        "canonical_source_model_path",
        "canonical_source_metadata_path",
        "canonical_source_model_sha256",
        "source_run_id",
        "source_stage",
        "source_checkpoint_sha256",
        "source_checkpoint_step",
        "onnx_model_path",
        "onnx_metadata_path",
        "onnx_model_sha256",
        "inference_spec_path",
        "inference_spec_sha256",
        "environment_contract_path",
        "environment_contract_sha256",
        "probe_states_path",
        "probe_states_sha256",
        "pytorch_reference_path",
        "pytorch_reference_sha256",
        "fixture_metadata_path",
    ):
        if lineage.get(field) is None:
            raise ValueError(f"lineage.{field} is required")
    thresholds = _require_mapping(config.get("thresholds"), label="thresholds")
    for key in ("cpu", "cuda"):
        values = _require_mapping(thresholds.get(key), label=f"thresholds.{key}")
        for metric in (
            "max_absolute_error",
            "mean_absolute_error",
            "max_relative_error",
            "mean_relative_error",
            "action_agreement_rate",
        ):
            try:
                parsed = float(values.get(metric))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"thresholds.{key}.{metric} must be finite"
                ) from error
            if not np.isfinite(parsed) or parsed < 0.0:
                raise ValueError(
                    f"thresholds.{key}.{metric} must be finite and non-negative"
                )
    try:
        epsilon = float(config.get("relative_error_epsilon"))
    except (TypeError, ValueError) as error:
        raise ValueError("relative_error_epsilon must be a positive finite number") from error
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("relative_error_epsilon must be a positive finite number")
    try:
        preprocessing_tolerance = float(config.get("preprocessing_max_absolute_error"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "preprocessing_max_absolute_error must be a positive finite number"
        ) from error
    if not np.isfinite(preprocessing_tolerance) or preprocessing_tolerance <= 0.0:
        raise ValueError(
            "preprocessing_max_absolute_error must be a positive finite number"
        )
    return config


def _check_metric_thresholds(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "max_absolute_error": {
            "observed": float(metrics["max_absolute_error"]),
            "limit": float(thresholds["max_absolute_error"]),
            "passed": float(metrics["max_absolute_error"])
            <= float(thresholds["max_absolute_error"]),
        },
        "mean_absolute_error": {
            "observed": float(metrics["mean_absolute_error"]),
            "limit": float(thresholds["mean_absolute_error"]),
            "passed": float(metrics["mean_absolute_error"])
            <= float(thresholds["mean_absolute_error"]),
        },
        "max_relative_error": {
            "observed": float(metrics["max_relative_error"]),
            "limit": float(thresholds["max_relative_error"]),
            "passed": float(metrics["max_relative_error"])
            <= float(thresholds["max_relative_error"]),
        },
        "mean_relative_error": {
            "observed": float(metrics["mean_relative_error"]),
            "limit": float(thresholds["mean_relative_error"]),
            "passed": float(metrics["mean_relative_error"])
            <= float(thresholds["mean_relative_error"]),
        },
        "action_agreement_rate": {
            "observed": float(metrics["action_agreement_rate"]),
            "limit": float(thresholds["action_agreement_rate"]),
            "passed": float(metrics["action_agreement_rate"])
            >= float(thresholds["action_agreement_rate"]),
        },
    }
    return {
        "passed": all(bool(value["passed"]) for value in checks.values()),
        "checks": checks,
    }


def _nvidia_smi_metadata() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,cuda_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"status": "unavailable"}
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "status": "available" if rows else "unavailable",
        "query": "name,driver_version,cuda_version,memory.total",
        "rows": rows,
    }


def _load_fixture_bundle(
    *,
    root: Path,
    source_model_path: Path,
    source_metadata_path: Path,
    onnx_model_path: Path,
    onnx_metadata_path: Path,
    spec_path: Path,
    fixture_metadata_path: Path,
    probe_states_path: Path,
    reference_path: Path,
    preprocessing_max_absolute_error: float,
    lineage_config: Mapping[str, Any],
) -> tuple[Any, ...]:
    source_metadata = _json_object(source_metadata_path)
    onnx_metadata = _json_object(onnx_metadata_path)
    fixture_metadata = _json_object(fixture_metadata_path)
    spec = load_inference_spec(spec_path)
    source_model_hash = _sha256_file(source_model_path)
    onnx_model_hash = _sha256_file(onnx_model_path)
    spec_hash = _verify_declared_file_hash(
        spec_path,
        fixture_metadata["inference_spec"]["sha256"],
        label="inference spec",
    )
    contract_path = (root / spec.environment_contract_path).resolve()
    contract_hash = _verify_declared_file_hash(
        contract_path,
        spec.environment_contract_sha256,
        label="Contract v2",
    )
    probe_hash = _sha256_file(probe_states_path)
    reference_hash = _sha256_file(reference_path)

    expected_paths = {
        "canonical source model path": (
            source_model_path,
            lineage_config["canonical_source_model_path"],
        ),
        "canonical source metadata path": (
            source_metadata_path,
            lineage_config["canonical_source_metadata_path"],
        ),
        "ONNX model path": (onnx_model_path, lineage_config["onnx_model_path"]),
        "ONNX metadata path": (
            onnx_metadata_path,
            lineage_config["onnx_metadata_path"],
        ),
        "inference spec path": (spec_path, lineage_config["inference_spec_path"]),
        "Contract v2 path": (
            contract_path,
            lineage_config["environment_contract_path"],
        ),
        "probe states path": (
            probe_states_path,
            lineage_config["probe_states_path"],
        ),
        "PyTorch reference path": (
            reference_path,
            lineage_config["pytorch_reference_path"],
        ),
        "fixture metadata path": (
            fixture_metadata_path,
            lineage_config["fixture_metadata_path"],
        ),
    }
    for label, (path, expected_path) in expected_paths.items():
        _require_equal(
            f"configured {label}",
            _relative_path(path, root=root),
            expected_path,
        )

    _require_equal(
        "canonical source artifact_type",
        source_metadata.get("artifact_type"),
        "canonical_final_model",
    )
    _require_equal(
        "canonical source model path",
        source_metadata.get("model_path"),
        _relative_path(source_model_path, root=root),
    )
    _require_equal(
        "canonical source model SHA256",
        source_metadata.get("model_sha256"),
        source_model_hash,
    )
    _require_equal(
        "configured canonical source model SHA256",
        source_model_hash,
        lineage_config["canonical_source_model_sha256"],
    )
    _require_equal(
        "configured source run id",
        source_metadata.get("source_run_id"),
        lineage_config["source_run_id"],
    )
    _require_equal(
        "configured source stage",
        source_metadata.get("source_stage"),
        lineage_config["source_stage"],
    )
    source_checkpoint = _require_mapping(
        source_metadata.get("source_checkpoint"),
        label="canonical source_checkpoint",
    )
    source_checkpoint_hash = str(source_checkpoint.get("sha256"))
    source_checkpoint_step = int(source_checkpoint.get("step"))
    _require_equal(
        "configured source checkpoint SHA256",
        source_checkpoint_hash,
        lineage_config["source_checkpoint_sha256"],
    )
    _require_equal(
        "configured source checkpoint step",
        source_checkpoint_step,
        lineage_config["source_checkpoint_step"],
    )

    _require_equal(
        "inference spec contract id",
        spec.contract_id,
        fixture_metadata["inference_spec"]["contract_id"],
    )
    _require_equal(
        "inference spec SHA256",
        spec_hash,
        fixture_metadata["inference_spec"]["sha256"],
    )
    _require_equal(
        "configured inference spec SHA256",
        spec_hash,
        lineage_config["inference_spec_sha256"],
    )
    _require_equal("Contract v2 SHA256", contract_hash, spec.environment_contract_sha256)
    _require_equal(
        "configured Contract v2 SHA256",
        contract_hash,
        lineage_config["environment_contract_sha256"],
    )
    _require_equal("ONNX model SHA256", onnx_model_hash, onnx_metadata.get("model_sha256"))
    _require_equal(
        "configured ONNX model SHA256",
        onnx_model_hash,
        lineage_config["onnx_model_sha256"],
    )
    _require_equal(
        "fixture ONNX model SHA256",
        onnx_model_hash,
        fixture_metadata["onnx_model"]["sha256"],
    )
    _require_equal(
        "ONNX source model SHA256",
        onnx_metadata.get("source_model_sha256"),
        source_model_hash,
    )
    _require_equal(
        "ONNX source checkpoint SHA256",
        onnx_metadata.get("source_checkpoint_sha256"),
        source_checkpoint_hash,
    )
    _require_equal(
        "fixture source model SHA256",
        fixture_metadata["source_model"]["model_sha256"],
        source_model_hash,
    )
    _require_equal(
        "fixture source checkpoint SHA256",
        fixture_metadata["source_model"]["source_checkpoint_sha256"],
        source_checkpoint_hash,
    )
    _require_equal(
        "fixture probe states SHA256",
        probe_hash,
        fixture_metadata["probe_states"]["sha256"],
    )
    _require_equal(
        "configured probe states SHA256",
        probe_hash,
        lineage_config["probe_states_sha256"],
    )
    _require_equal(
        "fixture reference SHA256",
        reference_hash,
        fixture_metadata["pytorch_reference"]["sha256"],
    )
    _require_equal(
        "configured PyTorch reference SHA256",
        reference_hash,
        lineage_config["pytorch_reference_sha256"],
    )

    with np.load(probe_states_path, allow_pickle=False) as probe_npz:
        observations = np.asarray(probe_npz["observations"])
        probe_metadata = _embedded_json(probe_npz, path=probe_states_path)
    with np.load(reference_path, allow_pickle=False) as reference_npz:
        reference_observations = np.asarray(reference_npz["observations"])
        model_inputs = np.asarray(reference_npz["model_inputs"])
        q_values = np.asarray(reference_npz["q_values"])
        greedy_actions = np.asarray(reference_npz["greedy_actions"])
        reference_metadata = _embedded_json(reference_npz, path=reference_path)

    expected_observation_shape = tuple(spec.source_observation_shape)
    if observations.dtype != np.dtype("uint8") or observations.ndim != 4:
        raise ValueError("probe observations must be a uint8 BCHW array")
    if tuple(observations.shape[1:]) != expected_observation_shape:
        raise ValueError(
            "probe observations do not match the inference contract: "
            f"observed={observations.shape[1:]}, expected={expected_observation_shape}"
        )
    if not np.array_equal(observations, reference_observations):
        raise ValueError("probe observations differ from the PyTorch reference fixture")
    expected_inputs = prepare_model_input(observations, device="cpu", spec=spec).numpy()
    input_difference = np.abs(
        model_inputs.astype(np.float64) - expected_inputs.astype(np.float64)
    )
    if (
        model_inputs.dtype != np.dtype("float32")
        or float(np.max(input_difference)) > preprocessing_max_absolute_error
    ):
        raise ValueError("PyTorch reference model_inputs do not match shared preprocessing")
    if (
        q_values.dtype != np.dtype("float32")
        or q_values.shape != (len(observations), len(EXPECTED_ACTION_MEANINGS))
    ):
        raise ValueError("PyTorch reference q_values do not match the output contract")
    if greedy_actions.dtype != np.dtype("int64") or greedy_actions.shape != (len(observations),):
        raise ValueError("PyTorch reference greedy_actions do not match the output contract")
    if not np.array_equal(greedy_actions, q_values_to_action(q_values, spec=spec)):
        raise ValueError("PyTorch reference greedy_actions are not argmax(q_values)")
    for label, array, expected in (
        (
            "reference model_inputs",
            model_inputs,
            reference_metadata["arrays"]["model_inputs_sha256"],
        ),
        ("reference q_values", q_values, reference_metadata["arrays"]["q_values_sha256"]),
        (
            "reference greedy_actions",
            greedy_actions,
            reference_metadata["arrays"]["greedy_actions_sha256"],
        ),
    ):
        _require_equal(f"{label} SHA256", _sha256_array(array), expected)
    _require_equal(
        "reference source model SHA256",
        reference_metadata["source_model_sha256"],
        source_model_hash,
    )
    _require_equal(
        "reference source checkpoint SHA256",
        reference_metadata["source_checkpoint_sha256"],
        source_checkpoint_hash,
    )
    _require_equal(
        "reference probe states SHA256",
        reference_metadata["probe_states"]["sha256"],
        probe_hash,
    )
    _require_equal(
        "probe Contract v2 id",
        probe_metadata["contract_id"],
        spec.environment_contract_id,
    )
    _require_equal("probe contract SHA256", probe_metadata["contract_sha256"], contract_hash)
    validate_action_meanings(reference_metadata["output"]["action_meanings"], spec=spec)
    validate_action_meanings(probe_metadata["action_meanings"], spec=spec)
    reference_runtime = _require_mapping(
        reference_metadata.get("runtime"),
        label="reference runtime",
    )
    if reference_runtime.get("cuda_available") is not True or not str(
        reference_runtime.get("device", "")
    ).startswith("cuda"):
        raise ValueError("PyTorch golden reference is not a CUDA reference")

    lineage = {
        "canonical_source_model": {
            "path": _relative_path(source_model_path, root=root),
            "sha256": source_model_hash,
            "source_run_id": source_metadata.get("source_run_id"),
            "source_stage": source_metadata.get("source_stage"),
            "training_seed": source_metadata.get("training_seed"),
            "training_transitions": source_metadata.get("training_transitions"),
        },
        "source_checkpoint": {
            "sha256": source_checkpoint_hash,
            "step": source_checkpoint_step,
            "provenance": "declared by the canonical Day 21 metadata",
        },
        "onnx_model": {
            "path": _relative_path(onnx_model_path, root=root),
            "sha256": onnx_model_hash,
        },
        "inference_spec": {
            "path": _relative_path(spec_path, root=root),
            "sha256": spec_hash,
            "contract_id": spec.contract_id,
            "environment_contract_path": _relative_path(contract_path, root=root),
            "environment_contract_sha256": contract_hash,
        },
        "probe_states": {
            "path": _relative_path(probe_states_path, root=root),
            "sha256": probe_hash,
            "count": int(len(observations)),
        },
        "pytorch_reference": {
            "path": _relative_path(reference_path, root=root),
            "sha256": reference_hash,
            "runtime": dict(reference_runtime),
            "model_input_preparation": {
                "comparison": "CUDA golden input vs shared CPU-boundary preparation",
                "max_absolute_error": float(np.max(input_difference)),
                "mean_absolute_error": float(np.mean(input_difference)),
                "tolerance": preprocessing_max_absolute_error,
            },
        },
    }
    return (
        spec,
        observations,
        model_inputs,
        q_values,
        greedy_actions,
        lineage,
        source_metadata,
        onnx_metadata,
        fixture_metadata,
    )


def _run_in_batches(
    policy: ONNXRuntimePolicy,
    observations: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    outputs: list[np.ndarray] = []
    calls = 0
    for start in range(0, len(observations), batch_size):
        batch = observations[start : start + batch_size]
        if len(batch) != batch_size:
            raise ValueError(
                f"probe count {len(observations)} is not divisible by batch size {batch_size}"
            )
        outputs.append(policy.predict_q_values(batch))
        calls += 1
    return np.concatenate(outputs, axis=0), calls


def _provider_result(
    *,
    provider_key: str,
    provider_name: str,
    policy: ONNXRuntimePolicy,
    observations: np.ndarray,
    reference_q_values: np.ndarray,
    reference_actions: np.ndarray,
    batch_sizes: Sequence[int],
    thresholds: Mapping[str, Any],
    relative_error_epsilon: float,
) -> dict[str, Any]:
    batches: dict[str, Any] = {}
    for batch_size in batch_sizes:
        candidate_q_values, calls = _run_in_batches(
            policy,
            observations,
            batch_size=batch_size,
        )
        candidate_actions = np.asarray(
            q_values_to_action(candidate_q_values),
            dtype=np.int64,
        )
        metrics = compare_q_values(
            reference_q_values,
            candidate_q_values,
            reference_actions=reference_actions,
            candidate_actions=candidate_actions,
            sample_ids=range(len(observations)),
            relative_epsilon=relative_error_epsilon,
        )
        checks = _check_metric_thresholds(metrics, thresholds)
        batches[str(batch_size)] = {
            "batch_size": batch_size,
            "calls": calls,
            "input_shape_per_call": [batch_size, *observations.shape[1:]],
            "output_shape": list(candidate_q_values.shape),
            "metrics": metrics,
            "thresholds": checks,
        }
    return {
        "provider_key": provider_key,
        "provider_name": provider_name,
        "runtime": policy.runtime_metadata,
        "batches": batches,
        "passed": all(value["thresholds"]["passed"] for value in batches.values()),
    }


def _command_line(
    *,
    provider: str,
    root: Path,
    config: Path,
    output: Path,
    report: Path,
) -> str:
    return (
        "python -m scripts.analysis.compare_pytorch_onnx "
        f"--provider {provider} --config {_relative_path(config, root=root)} "
        f"--output {_relative_path(output, root=root)} "
        f"--report {_relative_path(report, root=root)}"
    )


def render_report(result: Mapping[str, Any]) -> str:
    """Render the source-backed Day 23 report from comparison JSON."""

    lineage = _require_mapping(result.get("lineage"), label="lineage")
    lines = [
        "# Day 23 ONNX Runtime parity report",
        "",
        (
            "這份 report 回答一個具體問題：同一批 Day 22 固定 probe states，"
            "交給 Day 21 canonical Final Model 的 PyTorch CUDA reference 與 native "
            "ONNX Runtime 後，Q-values 和 greedy action 是否仍一致？"
        ),
        "",
        f"- status: `{result.get('status')}`",
        (
            "- selected providers: `"
            f"{', '.join(str(value) for value in result.get('selected_providers', []))}`"
        ),
        f"- probe count: `{lineage.get('probe_states', {}).get('count')}`",
        "",
        "## Frozen lineage",
        "",
    ]
    for label in (
        "canonical_source_model",
        "source_checkpoint",
        "onnx_model",
        "inference_spec",
        "probe_states",
        "pytorch_reference",
    ):
        value = lineage.get(label)
        if isinstance(value, Mapping):
            path = value.get("path", "declared by metadata")
            digest = value.get("sha256") or value.get("environment_contract_sha256")
            lines.append(f"- {label}: `{path}` / `{digest}`")
    lines.extend(
        [
            "",
            (
                "PyTorch reference 的來源是 Day 21 frozen model；ONNX 是 Day 22 export。"
                "這裡沒有重新挑選 Day 20 comparison checkpoint，也沒有改寫 "
                "preprocessing contract。"
            ),
            "",
            "## Provider results",
            "",
        ]
    )
    providers = result.get("providers")
    if not isinstance(providers, Mapping):
        raise ValueError("result is missing provider results")
    for key, provider in providers.items():
        if not isinstance(provider, Mapping):
            raise ValueError(f"provider result {key} is malformed")
        runtime = _require_mapping(
            provider.get("runtime"),
            label=f"provider {key} runtime",
        )
        lines.extend(
            [
                f"### {key}",
                "",
                f"- requested: `{runtime.get('requested_provider')}`",
                f"- available: `{runtime.get('available_providers')}`",
                f"- active: `{runtime.get('active_providers')}`",
                f"- actual primary: `{runtime.get('actual_provider')}`",
                f"- ONNX Runtime: `{runtime.get('onnxruntime_version')}`",
                f"- GPU model: `{runtime.get('gpu_model')}`",
                (
                    f"- CUDA / cuDNN: `{runtime.get('cuda_version')}` / "
                    f"`{runtime.get('cudnn_version')}`"
                ),
                f"- graph assignment: `{runtime.get('graph_assignment')}`",
                "",
                (
                    "| batch N | max abs. error | mean abs. error | max relative error | "
                    "action agreement | passed |"
                ),
                "| ---: | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        batches = _require_mapping(
            provider.get("batches"),
            label=f"provider {key} batches",
        )
        for batch_size, batch in batches.items():
            if not isinstance(batch, Mapping):
                raise ValueError(f"provider {key} batch {batch_size} is malformed")
            metrics = _require_mapping(batch.get("metrics"), label="batch metrics")
            checks = _require_mapping(batch.get("thresholds"), label="batch thresholds")
            lines.append(
                "| "
                f"{batch_size} | {float(metrics['max_absolute_error']):.9g} | "
                f"{float(metrics['mean_absolute_error']):.9g} | "
                f"{float(metrics['max_relative_error']):.9g} | "
                f"{float(metrics['action_agreement_rate']):.6f} | "
                f"{checks.get('passed')} |"
            )
        first_batch = next(iter(batches.values()))
        first_metrics = (
            first_batch.get("metrics", {})
            if isinstance(first_batch, Mapping)
            else {}
        )
        lines.extend(
            [
                "",
                (
                    "disagreement sample ids: `"
                    f"{first_metrics.get('disagreement_sample_ids', [])}`"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## How to read the two correctness metrics",
            "",
            (
                "數值 parity 衡量的是每個 action 的 Q-value 差多少；action agreement "
                "衡量的是把四個 Q-values 做 argmax 後，最後選出的 action index 是否相同。"
                "前者能抓出輸出數值被改寫，後者能抓出即使誤差很小、卻剛好跨過兩個 "
                "action 的決策邊界。兩者都通過，才表示這批 state 在數值和決策兩層都保持一致。"
            ),
            "",
            (
                "top-2 Q margin 是最大與次大的 Q-value 差距；margin 越小，越容易被浮點誤差 "
                "推過決策邊界。它是解讀 action agreement 的診斷資訊，不是替代 agreement "
                "的通行證。"
            ),
            "",
            "## Reproduction",
            "",
            "```powershell",
            str(result.get("generation", {}).get("command", "")),
            "```",
            "",
            (
                "這個結果只證明固定 probe states 上的 native CPU/CUDA ORT parity；它不等於 "
                "瀏覽器 WebAssembly/WebGPU parity，也不等於完整遊戲分數相同。下一步才是把 "
                "同一個 input/action contract 帶進瀏覽器 runtime。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def compare_artifacts(
    *,
    provider: str,
    source_model: str | Path = DEFAULT_SOURCE_MODEL,
    source_metadata: str | Path = DEFAULT_SOURCE_METADATA,
    onnx_model: str | Path = DEFAULT_ONNX_MODEL,
    onnx_metadata: str | Path = DEFAULT_ONNX_METADATA,
    spec_path: str | Path = DEFAULT_SPEC,
    fixture_metadata: str | Path = DEFAULT_FIXTURE_METADATA,
    probe_states: str | Path = DEFAULT_PROBE_STATES,
    reference: str | Path = DEFAULT_REFERENCE,
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path = DEFAULT_OUTPUT,
    report: str | Path = DEFAULT_REPORT,
    device_index: int = 0,
) -> dict[str, Any]:
    root = _repository_root()
    config_file = Path(config_path).resolve()
    _require_equal(
        "parity config path",
        _relative_path(config_file, root=root),
        DEFAULT_CONFIG.as_posix(),
    )
    config = _load_validation_config(config_file)
    lineage_config = _require_mapping(config["lineage"], label="lineage")
    source_model_path = Path(source_model).resolve()
    source_metadata_path = Path(source_metadata).resolve()
    onnx_model_path = Path(onnx_model).resolve()
    onnx_metadata_path = Path(onnx_metadata).resolve()
    spec_file = Path(spec_path).resolve()
    fixture_metadata_path = Path(fixture_metadata).resolve()
    probe_states_path = Path(probe_states).resolve()
    reference_path = Path(reference).resolve()
    preprocessing_tolerance = float(config["preprocessing_max_absolute_error"])
    (
        spec,
        observations,
        _model_inputs,
        reference_q_values,
        reference_actions,
        lineage,
        _source_metadata,
        _onnx_metadata,
        _fixture_metadata,
    ) = _load_fixture_bundle(
        root=root,
        source_model_path=source_model_path,
        source_metadata_path=source_metadata_path,
        onnx_model_path=onnx_model_path,
        onnx_metadata_path=onnx_metadata_path,
        spec_path=spec_file,
        fixture_metadata_path=fixture_metadata_path,
        probe_states_path=probe_states_path,
        reference_path=reference_path,
        preprocessing_max_absolute_error=preprocessing_tolerance,
        lineage_config=lineage_config,
    )
    if provider not in {"cpu", "cuda", "both"}:
        raise ValueError("provider must be one of: cpu, cuda, both")
    selected = ["cpu", "cuda"] if provider == "both" else [provider]
    providers_config = _require_mapping(config["providers"], label="providers")
    thresholds_config = _require_mapping(config["thresholds"], label="thresholds")
    batch_sizes = [int(value) for value in config["batch_sizes"]]
    relative_error_epsilon = float(config["relative_error_epsilon"])
    provider_results: dict[str, Any] = {}
    for key in selected:
        provider_name = str(providers_config[key])
        policy = ONNXRuntimePolicy(
            onnx_model_path,
            provider=provider_name,
            device_index=device_index,
            spec=spec,
        )
        provider_results[key] = _provider_result(
            provider_key=key,
            provider_name=provider_name,
            policy=policy,
            observations=observations,
            reference_q_values=reference_q_values,
            reference_actions=reference_actions,
            batch_sizes=batch_sizes,
            thresholds=_require_mapping(
                thresholds_config[key],
                label=f"thresholds.{key}",
            ),
            relative_error_epsilon=relative_error_epsilon,
        )
        provider_results[key]["runtime"]["nvidia_smi"] = _nvidia_smi_metadata()
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "day23_onnx_runtime_parity_result",
        "technical_question": (
            "Do PyTorch CUDA golden Q-values and greedy actions remain unchanged "
            "when the same ONNX policy runs in native ONNX Runtime?"
        ),
        "status": (
            "passed"
            if all(value["passed"] for value in provider_results.values())
            else "failed"
        ),
        "selected_providers": selected,
        "reference": {
            "runtime": "PyTorch",
            "provider": "CUDAExecutionProvider",
            "precision": "float32",
            "q_values_shape": list(reference_q_values.shape),
            "action_meanings": list(EXPECTED_ACTION_MEANINGS),
        },
        "batch_sizes": batch_sizes,
        "lineage": lineage,
        "validation_config": {
            "path": _relative_path(config_file, root=root),
            "sha256": _sha256_file(config_file, normalize_text=True),
        },
        "providers": provider_results,
        "generation": {
            "command": _command_line(
                provider=provider,
                root=root,
                config=config_file,
                output=Path(output),
                report=Path(report),
            ),
            "python_version": platform.python_version(),
            "repository_relative_paths_only": True,
        },
    }
    output_path = Path(output)
    report_path = Path(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("cpu", "cuda", "both"),
        default="both",
        help="provider to validate; both requires the formal CUDA check",
    )
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--onnx-metadata", type=Path, default=DEFAULT_ONNX_METADATA)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--fixture-metadata", type=Path, default=DEFAULT_FIXTURE_METADATA)
    parser.add_argument("--probe-states", type=Path, default=DEFAULT_PROBE_STATES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device-index", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compare_artifacts(
            provider=args.provider,
            source_model=args.source_model,
            source_metadata=args.source_metadata,
            onnx_model=args.onnx_model,
            onnx_metadata=args.onnx_metadata,
            spec_path=args.spec,
            fixture_metadata=args.fixture_metadata,
            probe_states=args.probe_states,
            reference=args.reference,
            config_path=args.config,
            output=args.output,
            report=args.report,
            device_index=args.device_index,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Day 23 parity validation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_providers": result["selected_providers"],
                "output": str(args.output),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
