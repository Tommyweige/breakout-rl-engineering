"""Evaluate Day 26 runtimes by fixed-seed multi-episode score distributions."""

from __future__ import annotations

import argparse
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
from breakout_rl.inference import ONNXRuntimePolicy, load_inference_spec
from breakout_rl.runtime_scores import (
    PAIRING_DEFINITIONS,
    RUNTIME_SCORE_REQUIRED_TARGETS,
    RuntimeScoreConfig,
    aggregate_runtime_scores,
    paired_score_differences,
)
from breakout_rl.tensorrt import (
    TENSORRT_STATUS_READY,
    TensorRTBlockedError,
    TensorRTSession,
    require_gpu_baseline,
)


DEFAULT_CONFIG = Path("configs/eval/day26_runtime_score_evaluation.json")
DEFAULT_TENSORRT_CONFIG = Path("configs/inference/tensorrt_experiment.json")
DEFAULT_PREFLIGHT = Path("assets/day26/tensorrt-preflight.json")
DEFAULT_COMPARISON = Path("assets/day26/tensorrt-comparison.json")
DEFAULT_SOURCE_METADATA = Path("assets/day21/models/final_model/metadata.json")
DEFAULT_DEVICE_INDEX = 0


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


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _relative(root: Path, path: Path) -> str:
    return repository_relative_path(path, root=root)


def _engine_metadata(
    *,
    root: Path,
    engine: Path,
    metadata_path: Path,
    precision: str,
    preflight: Path,
    expected_onnx_hash: str,
) -> dict[str, Any]:
    metadata = _json_object(metadata_path)
    if metadata.get("artifact_type") != "day26_tensorrt_engine":
        raise ValueError(f"{metadata_path}: unexpected TensorRT engine artifact type")
    if metadata.get("precision") != precision:
        raise ValueError(f"{metadata_path}: precision does not match {precision}")
    if metadata.get("network_typing") != "strongly_typed":
        raise TensorRTBlockedError(f"{metadata_path}: engine is not strongly typed")
    if metadata.get("tf32_enabled") is not False:
        raise TensorRTBlockedError(f"{metadata_path}: TF32 is not disabled")
    if metadata.get("engine_path") != _relative(root, engine):
        raise ValueError(f"{metadata_path}: engine path does not match")
    if metadata.get("engine_sha256") != sha256_file(engine):
        raise ValueError(f"{metadata_path}: engine SHA256 does not match")
    software = _mapping(metadata.get("software"), name=f"{metadata_path}.software")
    if software.get("preflight_sha256") != sha256_file(preflight):
        raise ValueError(f"{metadata_path}: preflight SHA256 does not match")
    lineage = _mapping(metadata.get("lineage"), name=f"{metadata_path}.lineage")
    onnx = _mapping(lineage.get("onnx"), name=f"{metadata_path}.lineage.onnx")
    if onnx.get("sha256") != expected_onnx_hash:
        raise ValueError(f"{metadata_path}: source ONNX SHA256 does not match")
    return {
        "path": _relative(root, engine),
        "sha256": sha256_file(engine),
        "metadata_path": _relative(root, metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "precision": precision,
        "network_typing": metadata["network_typing"],
        "tf32_enabled": metadata["tf32_enabled"],
        "size_bytes": engine.stat().st_size,
    }


class _ONNXScoreModule(nn.Module):
    def __init__(self, policy: ONNXRuntimePolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        values = self.policy.predict_model_input(
            np.ascontiguousarray(model_input.detach().cpu().numpy(), dtype=np.float32)
        )
        return torch.from_numpy(values).to(device=model_input.device)


class _TensorRTScoreModule(nn.Module):
    def __init__(self, session: TensorRTSession) -> None:
        super().__init__()
        self.session = session

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        output = self.session.execute(model_input)
        torch.cuda.synchronize(self.session.device)
        return output


def _failure_row(runtime: str, seed: int, error: BaseException) -> dict[str, Any]:
    return {
        "runtime": runtime,
        "seed": int(seed),
        "evaluation_seed": int(seed),
        "episode_index": 1,
        "episode_return": None,
        "score": None,
        "episode_length": None,
        "termination_reason": "runtime_error",
        "time_limit_source": None,
        "terminated": False,
        "truncated": False,
        "runtime_error": f"{type(error).__name__}: {error}",
        "failure": True,
    }


def _run_runtime(
    *,
    runtime: str,
    module: nn.Module,
    config: RuntimeScoreConfig,
    contract: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_metadata: list[dict[str, Any]] = []
    for seed_group in config.seed_groups:
        try:
            result = evaluate_policy(
                module,
                episodes=config.episodes_per_seed,
                seeds=[seed_group],
                device=device,
                epsilon=config.evaluation_epsilon,
                env_factory=lambda: make_breakout_env(
                    **breakout_environment_kwargs(contract)
                ),
                model_id=f"day26-runtime-score-{runtime}",
                evaluation_id=f"{config.evaluation_id}-{runtime}-group-{seed_group}",
                metadata={
                    "experiment": "day26_multi_episode_runtime_score",
                    "runtime": runtime,
                    "seed_selection_rule": config.seed_selection_rule,
                },
            )
        except Exception as error:
            group_metadata.append(
                {
                    "seed_group": seed_group,
                    "status": "failed",
                    "runtime_error": f"{type(error).__name__}: {error}",
                }
            )
            rows.extend(
                _failure_row(runtime, seed_group + offset, error)
                for offset in range(config.episodes_per_seed)
            )
            continue
        group_metadata.append(
            {
                "seed_group": seed_group,
                "status": "completed",
                **dict(result.runtime),
            }
        )
        for episode in result.episodes:
            rows.append(
                {
                    "runtime": runtime,
                    "seed": int(episode.episode_seed),
                    "evaluation_seed": int(episode.evaluation_seed),
                    "episode_index": int(episode.episode_index),
                    "episode_return": float(episode.episode_return),
                    "score": float(episode.episode_return),
                    "episode_length": int(episode.episode_length),
                    "termination_reason": episode.stop_reason,
                    "time_limit_source": episode.time_limit_source,
                    "terminated": bool(episode.terminated),
                    "truncated": bool(episode.truncated),
                    "runtime_error": None,
                    "failure": False,
                }
            )
    return rows, {
        "group_runs": group_metadata,
        "group_count": len(group_metadata),
        "evaluation_steps": sum(
            int(group.get("evaluation_steps", 0)) for group in group_metadata
        ),
        "wall_clock_seconds": sum(
            float(group.get("wall_clock_seconds", 0.0)) for group in group_metadata
        ),
    }


def _fixed_state_evidence(
    *,
    root: Path,
    comparison_path: Path,
    comparison: Mapping[str, Any],
    preflight_path: Path,
    preflight_status: str,
) -> dict[str, Any]:
    parity = _mapping(comparison.get("parity"), name="comparison.parity")
    if parity.get("sample_count") != 60:
        raise ValueError("fixed-state evidence must contain exactly 60 samples")
    candidates = _mapping(parity.get("candidates"), name="comparison.parity.candidates")
    lineage = _mapping(comparison.get("lineage"), name="comparison.lineage")
    recorded_preflight = _mapping(
        lineage.get("preflight"), name="comparison.lineage.preflight"
    )
    if recorded_preflight.get("path") != _relative(root, preflight_path):
        raise ValueError("fixed-state evidence was generated from another preflight")
    if recorded_preflight.get("sha256") != sha256_file(preflight_path):
        raise ValueError("fixed-state evidence preflight hash does not match")
    evidence: dict[str, Any] = {
        "path": _relative(root, comparison_path),
        "sha256": sha256_file(comparison_path),
        "benchmark_id": _mapping(
            comparison.get("benchmark"), name="comparison.benchmark"
        ).get("benchmark_id"),
        "sample_count": parity.get("sample_count"),
        "preflight": {
            "path": _relative(root, preflight_path),
            "sha256": sha256_file(preflight_path),
            "status": preflight_status,
        },
        "candidates": {},
    }
    fixed_state_targets = [
        "onnx_cuda_fp32",
        "tensorrt_cuda_fp32",
    ]
    if "tensorrt_cuda_fp16" in candidates:
        fixed_state_targets.append("tensorrt_cuda_fp16")
    for target in fixed_state_targets:
        payload = _mapping(candidates.get(target), name=f"comparison.{target}")
        metrics = _mapping(payload.get("metrics"), name=f"comparison.{target}.metrics")
        evidence["candidates"][target] = {
            "passed": bool(payload.get("passed")),
            "action_agreement_rate": metrics.get("action_agreement_rate"),
            "max_absolute_error": metrics.get("max_absolute_error"),
            "mean_absolute_error": metrics.get("mean_absolute_error"),
            "max_relative_error": metrics.get("max_relative_error"),
            "mean_relative_error": metrics.get("mean_relative_error"),
            "candidate_top_2_q_margin_min": _mapping(
                metrics.get("candidate_top_2_q_margin"),
                name=f"comparison.{target}.margin",
            ).get("min"),
        }
    evidence["batch1_end_to_end"] = _mapping(
        _mapping(comparison.get("benchmark"), name="comparison.benchmark").get(
            "batch1_end_to_end"
        ),
        name="comparison.benchmark.batch1_end_to_end",
    )
    return evidence


def _score_distribution_decision(
    *,
    config: RuntimeScoreConfig,
    aggregates: Mapping[str, Mapping[str, Any]],
    paired: Mapping[str, Mapping[str, Any]],
    fixed_state: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _mapping(
        aggregates.get("onnx_cuda_fp32"), name="aggregate.onnx_cuda_fp32"
    )
    candidate = _mapping(
        aggregates.get("tensorrt_cuda_fp32"),
        name="aggregate.tensorrt_cuda_fp32",
    )

    def delta(field: str) -> float | None:
        candidate_value = candidate.get(field)
        baseline_value = baseline.get(field)
        if candidate_value is None or baseline_value is None:
            return None
        return float(candidate_value) - float(baseline_value)

    def relative_drop(field: str) -> float:
        delta_value = delta(field)
        baseline_raw = baseline.get(field)
        if delta_value is None or baseline_raw is None:
            return 1.0
        baseline_value = float(baseline_raw)
        if baseline_value == 0.0:
            return 0.0 if float(candidate[field]) >= 0.0 else 1.0
        return max(0.0, -delta_value / abs(baseline_value))

    def relative_change(field: str) -> float:
        delta_value = delta(field)
        baseline_raw = baseline.get(field)
        if delta_value is None or baseline_raw is None:
            return 1.0
        baseline_value = float(baseline_raw)
        if baseline_value == 0.0:
            return 0.0 if float(candidate[field]) == 0.0 else 1.0
        return abs(delta_value) / abs(baseline_value)

    paired_trt_ort = _mapping(
        paired.get("tensorrt_fp32_minus_onnx_fp32"),
        name="paired.tensorrt_fp32_minus_onnx_fp32",
    )
    differences = [
        float(row["difference"])
        for row in paired_trt_ort.get("differences", [])
        if row.get("difference") is not None
    ]
    worse_fraction = (
        sum(value < 0.0 for value in differences) / len(differences)
        if differences
        else 1.0
    )
    policy = config.adoption_policy
    latency = _mapping(fixed_state["batch1_end_to_end"], name="fixed_state.latency")
    baseline_p95_ms = float(latency["onnx_cuda_fp32"]["p95_ms"])
    candidate_p95_ms = float(latency["tensorrt_cuda_fp32"]["p95_ms"])
    latency_improvement = (
        (baseline_p95_ms - candidate_p95_ms) / baseline_p95_ms
        if baseline_p95_ms > 0.0
        else None
    )
    distribution = {
        "baseline_runtime": "onnx_cuda_fp32",
        "candidate_runtime": "tensorrt_cuda_fp32",
        "mean_delta": delta("mean"),
        "median_delta": delta("median"),
        "p10_delta": delta("p10"),
        "p90_delta": delta("p90"),
        "relative_score_drop": {
            "mean": relative_drop("mean"),
            "median": relative_drop("median"),
            "p10": relative_drop("p10"),
            "p90": relative_drop("p90"),
            "min": relative_drop("min"),
        },
        "std_relative_change": relative_change("std"),
        "distribution_deltas": {
            field: delta(field)
            for field in ("mean", "median", "std", "p10", "p90", "min", "max")
        },
        "candidate_worse_fraction": worse_fraction,
        "batch1_p95_improvement_fraction": latency_improvement,
        "paired_statistics": paired_trt_ort.get("statistics"),
        "paired_bootstrap_mean_ci": paired_trt_ort.get("bootstrap_mean_ci"),
    }
    checks = {
        "all_runtime_episodes_complete": all(
            int(aggregate.get("successful_episode_count", 0))
            == len(config.concrete_episode_seeds)
            for aggregate in aggregates.values()
        ),
        "fixed_state_sanity": bool(
            _mapping(
                _mapping(
                    fixed_state.get("candidates"), name="fixed_state.candidates"
                ).get("tensorrt_cuda_fp32"),
                name="fixed_state.tensorrt_cuda_fp32",
            ).get("passed")
        ),
        "mean_distribution_not_materially_lower": distribution["relative_score_drop"][
            "mean"
        ]
        <= float(policy["max_mean_relative_score_drop"]),
        "median_distribution_not_materially_lower": distribution["relative_score_drop"][
            "median"
        ]
        <= float(policy["max_median_relative_score_drop"]),
        "p10_low_tail_not_materially_lower": distribution["relative_score_drop"]["p10"]
        <= float(policy["max_p10_relative_score_drop"]),
        "p90_upper_distribution_not_materially_lower": distribution[
            "relative_score_drop"
        ]["p90"]
        <= float(policy["max_p90_relative_score_drop"]),
        "minimum_score_not_materially_lower": distribution["relative_score_drop"]["min"]
        <= float(policy["max_min_relative_score_drop"]),
        "score_spread_not_materially_changed": distribution["std_relative_change"]
        <= float(policy["max_std_relative_change"]),
        "candidate_worse_fraction_acceptable": worse_fraction
        <= float(policy["max_candidate_worse_fraction"]),
        "batch1_end_to_end_p95_improved": bool(
            not policy["require_batch1_p95_improvement"]
            or (
                latency_improvement is not None
                and latency_improvement
                >= float(policy["min_batch1_p95_improvement_fraction"])
            )
        ),
        "maintenance_cost_accepted": bool(policy["maintenance_cost_accepted"]),
    }
    recommendation = (
        "native deployment-worthy" if all(checks.values()) else "native not worthwhile"
    )
    return {
        "distribution": distribution,
        "checks": checks,
        "recommendation": recommendation,
        "adoption_gate": "score distribution and latency; no gameplay trace equality gate",
    }


def run_score_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    config_path = _resolve(root, args.config)
    score_config_payload = _json_object(config_path)
    score_config = RuntimeScoreConfig.from_mapping(score_config_payload)
    active_targets = tuple(score_config.runtime_targets)
    if not set(RUNTIME_SCORE_REQUIRED_TARGETS).issubset(active_targets):
        raise ValueError("runtime score evaluation is missing a required runtime")
    output_path = _resolve_output(root, score_config.output_path, args.force)

    tensorrt_config_path = _resolve(root, args.tensorrt_config)
    tensorrt_config = _json_object(tensorrt_config_path)
    if tensorrt_config.get("artifact_type") != "day26_tensorrt_experiment":
        raise ValueError("TensorRT config has an unexpected artifact_type")
    source = _mapping(tensorrt_config.get("source"), name="tensorrt_config.source")
    build = _mapping(tensorrt_config.get("build"), name="tensorrt_config.build")
    if (
        build.get("network_typing") != "strongly_typed"
        or build.get("tf32_enabled") is not False
    ):
        raise TensorRTBlockedError(
            "TensorRT config is not strongly typed with TF32 disabled"
        )

    contract_path = _resolve(root, score_config.environment_contract_path)
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    if score_config.environment_id != contract.environment_id:
        raise ValueError("runtime score config environment_id differs from Contract v2")
    if score_config.evaluation_epsilon != contract.evaluation_epsilon:
        raise ValueError("runtime score config epsilon differs from Contract v2")
    if set(contract.concrete_episode_seeds) != set(
        score_config.source_protocol["base_concrete_episode_seeds"]
    ):
        raise ValueError(
            "runtime score source protocol does not retain Contract v2 seeds"
        )

    base_config_path = _resolve(root, str(score_config.source_protocol["config_path"]))
    base_config = load_evaluation_config(base_config_path)
    if (
        tuple(base_config.seeds) != tuple(score_config.source_protocol["seed_groups"])
        or base_config.episodes_per_seed
        != int(score_config.source_protocol["base_episodes_per_seed"])
        or base_config.environment_id != contract.environment_id
        or base_config.epsilon != contract.evaluation_epsilon
    ):
        raise ValueError(
            "runtime score source protocol differs from breakout_eval.json"
        )

    spec_path = _resolve(root, str(source["inference_spec_path"]))
    spec = load_inference_spec(spec_path)
    validate_environment_contract(spec, root=root)
    if spec.environment_contract_path != score_config.environment_contract_path:
        raise ValueError("runtime score contract path differs from inference spec")

    preflight_path = _resolve(root, args.preflight)
    preflight = _json_object(preflight_path)
    if preflight.get("artifact_type") != "day26_tensorrt_preflight":
        raise ValueError("preflight artifact has an unexpected type")
    if preflight.get("status") != TENSORRT_STATUS_READY:
        raise TensorRTBlockedError(
            f"runtime score evaluation requires READY preflight, got {preflight.get('status')!r}"
        )
    capabilities = _mapping(
        preflight.get("capabilities"), name="preflight.capabilities"
    )
    if capabilities.get("strongly_typed_network_available") is not True:
        raise TensorRTBlockedError("preflight did not verify strongly typed TensorRT")
    if capabilities.get("network_typing") != "strongly_typed":
        raise TensorRTBlockedError(
            "preflight network typing policy is not strongly_typed"
        )
    if capabilities.get("tf32_enabled") is not False:
        raise TensorRTBlockedError("preflight TF32 policy is not disabled")
    preflight_lineage = _mapping(preflight.get("lineage"), name="preflight.lineage")
    preflight_config = _mapping(
        preflight_lineage.get("config"), name="preflight.lineage.config"
    )
    if preflight_config.get("path") != _relative(root, tensorrt_config_path):
        raise ValueError("preflight was generated from a different TensorRT config")
    if preflight_config.get("sha256") != sha256_file(
        tensorrt_config_path, normalize_text=True
    ):
        raise ValueError("preflight TensorRT config hash does not match")

    source_model = _resolve(root, str(source["canonical_source_model_path"]))
    source_metadata = _resolve(root, args.source_metadata)
    identity = source_identity(source_model, source_metadata, root=root)
    if identity["model_sha256"] != source.get("canonical_source_model_sha256"):
        raise ValueError("canonical source model hash does not match TensorRT config")
    onnx_fp32 = _resolve(root, str(source["onnx_fp32_path"]))
    if sha256_file(onnx_fp32) != source.get("onnx_fp32_sha256"):
        raise ValueError("FP32 ONNX hash does not match TensorRT config")
    onnx_fp16: Path | None = None
    if "tensorrt_cuda_fp16" in active_targets:
        onnx_fp16 = _resolve(root, str(source["onnx_fp16_path"]))
        if sha256_file(onnx_fp16) != source.get("onnx_fp16_sha256"):
            raise ValueError("FP16 ONNX hash does not match TensorRT config")

    engine_fp32 = _resolve_output(
        root,
        str(build["fp32_engine_path"]),
        must_exist=True,
    )
    metadata_fp32 = _resolve(root, str(build["fp32_metadata_path"]))
    engine_records = {
        "engine_fp32": _engine_metadata(
            root=root,
            engine=engine_fp32,
            metadata_path=metadata_fp32,
            precision="float32",
            preflight=preflight_path,
            expected_onnx_hash=sha256_file(onnx_fp32),
        ),
    }
    engine_fp16: Path | None = None
    if "tensorrt_cuda_fp16" in active_targets:
        if onnx_fp16 is None:
            raise ValueError("FP16 runtime requires an FP16 ONNX artifact")
        engine_fp16 = _resolve_output(
            root,
            str(build["fp16_engine_path"]),
            must_exist=True,
        )
        metadata_fp16 = _resolve(root, str(build["fp16_metadata_path"]))
        engine_records["engine_fp16"] = _engine_metadata(
            root=root,
            engine=engine_fp16,
            metadata_path=metadata_fp16,
            precision="float16",
            preflight=preflight_path,
            expected_onnx_hash=sha256_file(onnx_fp16),
        )

    comparison_path = _resolve(root, args.comparison)
    comparison = _json_object(comparison_path)
    if comparison.get("artifact_type") != "day26_tensorrt_comparison":
        raise ValueError("Day 26 comparison artifact has an unexpected type")
    if comparison.get("status") != "completed":
        raise ValueError("Day 26 comparison artifact is not completed")
    fixed_state = _fixed_state_evidence(
        root=root,
        comparison_path=comparison_path,
        comparison=comparison,
        preflight_path=preflight_path,
        preflight_status=str(preflight.get("status")),
    )

    if not torch.cuda.is_available():
        raise TensorRTBlockedError("multi-episode runtime evaluation requires CUDA")
    if args.device_index < 0 or args.device_index >= torch.cuda.device_count():
        raise TensorRTBlockedError(
            f"CUDA device index {args.device_index} is unavailable"
        )
    device = torch.device(f"cuda:{args.device_index}")
    model, _payload, _model_config = load_deployment_model(
        source_model,
        device=device,
        identity=identity,
        spec=spec,
    )
    ort_policy = ONNXRuntimePolicy(
        onnx_fp32,
        provider="cuda",
        device_index=args.device_index,
        spec=spec,
    )
    require_gpu_baseline(
        torch_cuda_available=True,
        ort_actual_provider=ort_policy.runtime_metadata.get("actual_provider"),
        ort_graph_assignment_status=_mapping(
            ort_policy.runtime_metadata.get("graph_assignment"),
            name="ORT graph assignment",
        ).get("status"),
    )
    trt_site = args.tensorrt_site_packages
    trt_fp32 = TensorRTSession(
        engine_fp32,
        device_index=args.device_index,
        tensorrt_site_packages=trt_site,
        precision="float32",
    )
    modules: dict[str, nn.Module] = {
        "pytorch_cuda_fp32": model,
        "onnx_cuda_fp32": _ONNXScoreModule(ort_policy),
        "tensorrt_cuda_fp32": _TensorRTScoreModule(trt_fp32),
    }
    if "tensorrt_cuda_fp16" in active_targets:
        if engine_fp16 is None:
            raise ValueError("FP16 runtime requires a TensorRT FP16 engine")
        trt_fp16 = TensorRTSession(
            engine_fp16,
            device_index=args.device_index,
            tensorrt_site_packages=trt_site,
            precision="float16",
        )
        modules["tensorrt_cuda_fp16"] = _TensorRTScoreModule(trt_fp16)
    bootstrap = {
        "enabled": score_config.bootstrap_enabled,
        "seed": score_config.bootstrap_seed,
        "resamples": score_config.bootstrap_resamples,
        "confidence_level": score_config.bootstrap_confidence_level,
    }
    per_runtime: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for runtime in active_targets:
        rows, runtime_metadata = _run_runtime(
            runtime=runtime,
            module=modules[runtime],
            config=score_config,
            contract=contract,
            device=device,
        )
        aggregate = aggregate_runtime_scores(
            rows,
            runtime=runtime,
            expected_seeds=score_config.concrete_episode_seeds,
            bootstrap=bootstrap,
        )
        per_runtime[runtime] = {
            "runtime_metadata": runtime_metadata,
            "episodes": rows,
            "aggregate": aggregate,
        }
    elapsed = time.perf_counter() - started

    rows_by_runtime = {
        runtime: per_runtime[runtime]["episodes"] for runtime in active_targets
    }
    paired = {
        name: paired_score_differences(
            rows_by_runtime[left],
            rows_by_runtime[right],
            left_runtime=left,
            right_runtime=right,
            expected_seeds=score_config.concrete_episode_seeds,
            bootstrap={
                **bootstrap,
                "seed": score_config.bootstrap_seed + index + 1,
            },
        )
        for index, (name, (left, right)) in enumerate(
            (
                item
                for item in PAIRING_DEFINITIONS.items()
                if set(item[1]).issubset(active_targets)
            )
        )
    }
    aggregates = {
        runtime: per_runtime[runtime]["aggregate"] for runtime in active_targets
    }
    decision = _score_distribution_decision(
        config=score_config,
        aggregates=aggregates,
        paired=paired,
        fixed_state=fixed_state,
    )
    failures = {
        runtime: int(per_runtime[runtime]["aggregate"]["failure_count"])
        for runtime in active_targets
    }
    output_payload = {
        "schema_version": 1,
        "artifact_type": "day26_runtime_score_comparison",
        "status": "completed",
        "evaluation_status": (
            "completed_with_runtime_failures" if any(failures.values()) else "completed"
        ),
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "evaluation_id": score_config.evaluation_id,
        "seed_manifest": {
            "episode_seeds": list(score_config.concrete_episode_seeds),
            "count": len(score_config.concrete_episode_seeds),
            "unique": len(set(score_config.concrete_episode_seeds))
            == len(score_config.concrete_episode_seeds),
            "seed_groups": list(score_config.seed_groups),
            "episodes_per_seed": score_config.episodes_per_seed,
            "selection_rule": "predeclared_before_runtime_results",
            "source_protocol": dict(score_config.source_protocol),
        },
        "evaluation_protocol": {
            "environment_id": contract.environment_id,
            "environment_contract_id": contract.contract_id,
            "environment_contract_path": _relative(root, contract_path),
            "environment_contract_sha256": sha256_file(contract_path),
            "evaluation_epsilon": score_config.evaluation_epsilon,
            "device_index": args.device_index,
            "runtime_targets": list(active_targets),
            "adoption_policy": dict(score_config.adoption_policy),
            "action_trace_equality_is_adoption_gate": False,
        },
        "runtime_failures": failures,
        "elapsed_seconds": elapsed,
        "source": {
            "evaluation_config": {
                "path": _relative(root, config_path),
                "sha256": sha256_file(config_path, normalize_text=True),
            },
            "source_protocol_config": {
                "path": _relative(root, base_config_path),
                "sha256": sha256_file(base_config_path, normalize_text=True),
            },
            "tensorrt_config": {
                "path": _relative(root, tensorrt_config_path),
                "sha256": sha256_file(tensorrt_config_path, normalize_text=True),
            },
            "preflight": {
                "path": _relative(root, preflight_path),
                "sha256": sha256_file(preflight_path),
            },
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
            **(
                {
                    "onnx_fp16": {
                        "path": _relative(root, onnx_fp16),
                        "sha256": sha256_file(onnx_fp16),
                    }
                }
                if onnx_fp16 is not None
                else {}
            ),
            **engine_records,
            "prior_fixed_state_and_latency_evidence": fixed_state,
        },
        "bootstrap": bootstrap,
        "per_runtime": per_runtime,
        "aggregates": aggregates,
        "paired_score_differences": paired,
        "decision": decision,
        "visualization": {
            "path": score_config.figure_path,
            "source": "per_runtime episode rows and paired_score_differences in this JSON",
        },
        "generation": {
            "command": " ".join(sys.argv),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_model": torch.cuda.get_device_name(args.device_index),
            "repository_relative_paths_only": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_payload["output_path"] = _relative(root, output_path)
    return output_payload


def _resolve_output(
    root: Path,
    value: str | Path,
    force: bool = False,
    *,
    must_exist: bool = False,
) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    if not must_exist and resolved.exists() and not force:
        raise FileExistsError(f"output already exists: {resolved}; use --force")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tensorrt-config", type=Path, default=DEFAULT_TENSORRT_CONFIG)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--device-index", type=int, default=DEFAULT_DEVICE_INDEX)
    parser.add_argument("--tensorrt-site-packages", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_score_evaluation(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Day 26 runtime score evaluation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": result["output_path"],
                "episode_count": result["seed_manifest"]["count"],
                "runtime_failures": result["runtime_failures"],
                "recommendation": result["decision"]["recommendation"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
