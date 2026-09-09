"""Shared validation and statistics for multi-episode runtime score evidence."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from breakout_rl.evaluation_contract import expand_concrete_episode_seeds


RUNTIME_SCORE_SCHEMA_VERSION = 1
RUNTIME_SCORE_REQUIRED_TARGETS = (
    "pytorch_cuda_fp32",
    "onnx_cuda_fp32",
    "tensorrt_cuda_fp32",
)
RUNTIME_SCORE_TARGETS = RUNTIME_SCORE_REQUIRED_TARGETS + ("tensorrt_cuda_fp16",)
PAIRING_DEFINITIONS = {
    "onnx_fp32_minus_pytorch_fp32": (
        "onnx_cuda_fp32",
        "pytorch_cuda_fp32",
    ),
    "tensorrt_fp32_minus_pytorch_fp32": (
        "tensorrt_cuda_fp32",
        "pytorch_cuda_fp32",
    ),
    "tensorrt_fp32_minus_onnx_fp32": (
        "tensorrt_cuda_fp32",
        "onnx_cuda_fp32",
    ),
    "tensorrt_fp16_minus_pytorch_fp32": (
        "tensorrt_cuda_fp16",
        "pytorch_cuda_fp32",
    ),
}


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(parsed)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _seed_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of non-negative integers")
    parsed = tuple(_integer(item, name=f"{name} item") for item in value)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must contain unique values")
    return parsed


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class RuntimeScoreConfig:
    """The predeclared multi-episode score protocol."""

    evaluation_id: str
    environment_contract_path: str
    environment_id: str
    evaluation_epsilon: float
    seed_selection_rule: str
    seed_groups: tuple[int, ...]
    episodes_per_seed: int
    concrete_episode_seeds: tuple[int, ...]
    runtime_targets: tuple[str, ...]
    bootstrap_enabled: bool
    bootstrap_seed: int
    bootstrap_resamples: int
    bootstrap_confidence_level: float
    adoption_policy: Mapping[str, Any]
    output_path: str
    figure_path: str
    source_protocol: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeScoreConfig":
        if payload.get("schema_version") != RUNTIME_SCORE_SCHEMA_VERSION:
            raise ValueError("runtime score config has an unexpected schema_version")
        if payload.get("artifact_type") != "day26_runtime_score_evaluation_config":
            raise ValueError("runtime score config has an unexpected artifact_type")

        evaluation_id = payload.get("evaluation_id")
        contract_path = payload.get("environment_contract_path")
        environment_id = payload.get("environment_id")
        output_path = payload.get("output_path")
        figure_path = payload.get("figure_path")
        for value, name in (
            (evaluation_id, "evaluation_id"),
            (contract_path, "environment_contract_path"),
            (environment_id, "environment_id"),
            (output_path, "output_path"),
            (figure_path, "figure_path"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if payload.get("seed_selection_rule") != "predeclared_before_runtime_results":
            raise ValueError(
                "runtime score seeds must be predeclared before runtime results"
            )
        seed_groups = _seed_tuple(payload.get("seed_groups"), name="seed_groups")
        episodes_per_seed = _integer(
            payload.get("episodes_per_seed"),
            name="episodes_per_seed",
            minimum=1,
        )
        concrete_seeds = _seed_tuple(
            payload.get("concrete_episode_seeds"),
            name="concrete_episode_seeds",
        )
        expanded_seeds = expand_concrete_episode_seeds(
            seed_groups,
            episodes_per_seed=episodes_per_seed,
        )
        if concrete_seeds != expanded_seeds:
            raise ValueError(
                "concrete_episode_seeds must exactly expand seed_groups and "
                "episodes_per_seed"
            )
        if len(concrete_seeds) < 30:
            raise ValueError("runtime score evaluation requires at least 30 episodes")
        if payload.get("total_episodes") != len(concrete_seeds):
            raise ValueError("total_episodes does not match concrete_episode_seeds")

        source_protocol = _mapping(
            payload.get("source_protocol"), name="source_protocol"
        )
        source_groups = _seed_tuple(
            source_protocol.get("seed_groups"),
            name="source_protocol.seed_groups",
        )
        source_episodes = _integer(
            source_protocol.get("base_episodes_per_seed"),
            name="source_protocol.base_episodes_per_seed",
            minimum=1,
        )
        source_concrete = _seed_tuple(
            source_protocol.get("base_concrete_episode_seeds"),
            name="source_protocol.base_concrete_episode_seeds",
        )
        if source_concrete != expand_concrete_episode_seeds(
            source_groups,
            episodes_per_seed=source_episodes,
        ):
            raise ValueError("source protocol concrete seeds do not match its groups")
        if not set(source_concrete).issubset(concrete_seeds):
            raise ValueError("expanded protocol must retain all source protocol seeds")
        if source_groups != seed_groups:
            raise ValueError("source protocol and runtime score seed groups differ")

        raw_targets = payload.get("runtime_targets")
        targets = tuple(str(value) for value in raw_targets or ())
        if targets not in (RUNTIME_SCORE_REQUIRED_TARGETS, RUNTIME_SCORE_TARGETS):
            raise ValueError(
                "runtime_targets must include the three required Day 26 runtimes "
                "and may include optional TensorRT FP16"
            )
        bootstrap = _mapping(payload.get("bootstrap"), name="bootstrap")
        enabled = bootstrap.get("enabled")
        if not isinstance(enabled, bool):
            raise TypeError("bootstrap.enabled must be a boolean")
        confidence = _finite_float(
            bootstrap.get("confidence_level"),
            name="bootstrap.confidence_level",
        )
        if not 0.0 < confidence < 1.0:
            raise ValueError("bootstrap.confidence_level must be between 0 and 1")
        adoption_policy = _mapping(
            payload.get("adoption_policy"), name="adoption_policy"
        )
        for field in (
            "max_mean_relative_score_drop",
            "max_median_relative_score_drop",
            "max_p10_relative_score_drop",
            "max_p90_relative_score_drop",
            "max_min_relative_score_drop",
            "max_std_relative_change",
        ):
            value = _finite_float(
                adoption_policy.get(field), name=f"adoption_policy.{field}"
            )
            if value < 0.0:
                raise ValueError(f"adoption_policy.{field} must not be negative")
        worse_fraction = _finite_float(
            adoption_policy.get("max_candidate_worse_fraction"),
            name="adoption_policy.max_candidate_worse_fraction",
        )
        if not 0.0 <= worse_fraction <= 1.0:
            raise ValueError(
                "adoption_policy.max_candidate_worse_fraction must be between 0 and 1"
            )
        for field in ("require_batch1_p95_improvement", "maintenance_cost_accepted"):
            if not isinstance(adoption_policy.get(field), bool):
                raise TypeError(f"adoption_policy.{field} must be a boolean")
        minimum_latency_improvement = _finite_float(
            adoption_policy.get("min_batch1_p95_improvement_fraction"),
            name="adoption_policy.min_batch1_p95_improvement_fraction",
        )
        if not 0.0 <= minimum_latency_improvement <= 1.0:
            raise ValueError(
                "adoption_policy.min_batch1_p95_improvement_fraction must be "
                "between 0 and 1"
            )
        evaluation_epsilon = _finite_float(
            payload.get("evaluation_epsilon"),
            name="evaluation_epsilon",
        )
        if not 0.0 <= evaluation_epsilon <= 1.0:
            raise ValueError("evaluation_epsilon must be between 0 and 1")
        return cls(
            evaluation_id=evaluation_id,
            environment_contract_path=contract_path,
            environment_id=environment_id,
            evaluation_epsilon=evaluation_epsilon,
            seed_selection_rule="predeclared_before_runtime_results",
            seed_groups=seed_groups,
            episodes_per_seed=episodes_per_seed,
            concrete_episode_seeds=concrete_seeds,
            runtime_targets=targets,
            bootstrap_enabled=enabled,
            bootstrap_seed=_integer(
                bootstrap.get("seed"), name="bootstrap.seed", minimum=0
            ),
            bootstrap_resamples=_integer(
                bootstrap.get("resamples"),
                name="bootstrap.resamples",
                minimum=1,
            ),
            bootstrap_confidence_level=confidence,
            adoption_policy=dict(adoption_policy),
            output_path=output_path,
            figure_path=figure_path,
            source_protocol=dict(source_protocol),
        )


def score_statistics(scores: Sequence[float]) -> dict[str, float | int]:
    """Return the requested distribution summary for finite episode scores."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite, non-empty one-dimensional sequence")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "p10": float(np.percentile(values, 10, method="linear")),
        "p90": float(np.percentile(values, 90, method="linear")),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, float | int | str]:
    """Compute a deterministic percentile bootstrap interval for a mean."""

    parsed_seed = _integer(seed, name="bootstrap seed", minimum=0)
    parsed_resamples = _integer(resamples, name="bootstrap resamples", minimum=1)
    confidence = _finite_float(confidence_level, name="bootstrap confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence_level must be between 0 and 1")
    parsed = np.asarray(values, dtype=np.float64)
    if parsed.ndim != 1 or parsed.size < 1 or not np.isfinite(parsed).all():
        raise ValueError("bootstrap values must be finite and non-empty")
    generator = np.random.default_rng(parsed_seed)
    indices = generator.integers(
        0,
        parsed.size,
        size=(parsed_resamples, parsed.size),
    )
    means = parsed[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "percentile_bootstrap_mean",
        "seed": parsed_seed,
        "resamples": parsed_resamples,
        "confidence_level": confidence,
        "mean": float(parsed.mean()),
        "lower": float(np.percentile(means, 100.0 * alpha, method="linear")),
        "upper": float(np.percentile(means, 100.0 * (1.0 - alpha), method="linear")),
    }


def validate_runtime_episode_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    runtime: str,
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Validate one runtime's one-row-per-seed result table."""

    expected = _seed_tuple(expected_seeds, name="expected_seeds")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("episode rows must be a sequence")
    by_seed: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, name=f"episode row {index}")
        if row.get("runtime") != runtime:
            raise ValueError(f"episode row {index} has an unexpected runtime")
        seed = _integer(row.get("seed"), name=f"episode row {index} seed")
        if seed not in expected:
            raise ValueError(f"episode row {index} has an unexpected seed {seed}")
        if seed in by_seed:
            raise ValueError(f"runtime {runtime} has duplicate seed {seed}")
        error = row.get("runtime_error")
        failure = row.get("failure", error is not None)
        if not isinstance(failure, bool):
            raise TypeError(f"episode row {index} failure must be a boolean")
        normalized = dict(row)
        normalized["seed"] = seed
        normalized["failure"] = failure
        if failure:
            if not isinstance(error, str) or not error.strip():
                raise ValueError(
                    f"failed episode row {index} must record runtime_error"
                )
            normalized["runtime_error"] = error
            normalized["score"] = None
        else:
            if error not in (None, ""):
                raise ValueError(
                    f"successful episode row {index} cannot record runtime_error"
                )
            score = _finite_float(
                row.get("score", row.get("episode_return")),
                name=f"episode row {index} score",
            )
            if row.get("episode_return") is not None:
                return_value = _finite_float(
                    row["episode_return"],
                    name=f"episode row {index} episode_return",
                )
                if return_value != score:
                    raise ValueError(
                        f"episode row {index} score and episode_return differ"
                    )
            length = _integer(
                row.get("episode_length"),
                name=f"episode row {index} episode_length",
                minimum=1,
            )
            reason = row.get("termination_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"successful episode row {index} needs termination_reason"
                )
            normalized.update(
                {
                    "score": score,
                    "episode_return": score,
                    "episode_length": length,
                    "runtime_error": None,
                }
            )
        by_seed[seed] = normalized
    if set(by_seed) != set(expected):
        raise ValueError(
            f"runtime {runtime} seeds do not match the declared evaluation set"
        )
    return [by_seed[seed] for seed in expected]


def aggregate_runtime_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    runtime: str,
    expected_seeds: Sequence[int],
    bootstrap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild one runtime's aggregate from its raw episode rows."""

    normalized = validate_runtime_episode_rows(
        rows,
        runtime=runtime,
        expected_seeds=expected_seeds,
    )
    successful = [row for row in normalized if not row["failure"]]
    scores = [float(row["score"]) for row in successful]
    aggregate: dict[str, Any] = {
        "runtime": runtime,
        "episode_count": len(normalized),
        "successful_episode_count": len(successful),
        "failure_count": len(normalized) - len(successful),
        "failure_seeds": [int(row["seed"]) for row in normalized if row["failure"]],
        "statistics_scope": "successful_episodes",
    }
    if scores:
        aggregate.update(score_statistics(scores))
        aggregate["episode_length_statistics"] = score_statistics(
            [float(row["episode_length"]) for row in successful]
        )
        if bootstrap is not None and bool(bootstrap.get("enabled", False)):
            aggregate["bootstrap_mean_ci"] = bootstrap_mean_ci(
                scores,
                seed=_integer(bootstrap["seed"], name="bootstrap seed"),
                resamples=_integer(
                    bootstrap["resamples"], name="bootstrap resamples", minimum=1
                ),
                confidence_level=_finite_float(
                    bootstrap["confidence_level"],
                    name="bootstrap confidence_level",
                ),
            )
    else:
        aggregate.update(
            {
                key: None
                for key in (
                    "count",
                    "mean",
                    "median",
                    "std",
                    "p10",
                    "p90",
                    "min",
                    "max",
                )
            }
        )
        aggregate["episode_length_statistics"] = None
    return aggregate


def paired_score_differences(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    left_runtime: str,
    right_runtime: str,
    expected_seeds: Sequence[int],
    bootstrap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pair scores by the same declared seed and compute left-minus-right."""

    left = validate_runtime_episode_rows(
        left_rows,
        runtime=left_runtime,
        expected_seeds=expected_seeds,
    )
    right = validate_runtime_episode_rows(
        right_rows,
        runtime=right_runtime,
        expected_seeds=expected_seeds,
    )
    left_by_seed = {int(row["seed"]): row for row in left}
    right_by_seed = {int(row["seed"]): row for row in right}
    if set(left_by_seed) != set(right_by_seed):
        raise ValueError("paired runtimes must use the same seeds")
    differences: list[dict[str, Any]] = []
    values: list[float] = []
    excluded: list[int] = []
    for seed in _seed_tuple(expected_seeds, name="expected_seeds"):
        left_score = left_by_seed[seed].get("score")
        right_score = right_by_seed[seed].get("score")
        if left_score is None or right_score is None:
            difference = None
            excluded.append(seed)
        else:
            difference = float(left_score) - float(right_score)
            values.append(difference)
        differences.append(
            {
                "seed": seed,
                "left_score": left_score,
                "right_score": right_score,
                "difference": difference,
            }
        )
    payload: dict[str, Any] = {
        "left_runtime": left_runtime,
        "right_runtime": right_runtime,
        "definition": "left_score_minus_right_score",
        "declared_seed_count": len(differences),
        "paired_episode_count": len(values),
        "excluded_seeds": excluded,
        "differences": differences,
        "statistics": score_statistics(values) if values else None,
    }
    if values and bootstrap is not None and bool(bootstrap.get("enabled", False)):
        payload["bootstrap_mean_ci"] = bootstrap_mean_ci(
            values,
            seed=_integer(bootstrap["seed"], name="bootstrap seed"),
            resamples=_integer(
                bootstrap["resamples"], name="bootstrap resamples", minimum=1
            ),
            confidence_level=_finite_float(
                bootstrap["confidence_level"],
                name="bootstrap confidence_level",
            ),
        )
    return payload


def validate_runtime_score_artifact(
    payload: Mapping[str, Any],
    *,
    expected_targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate a generated artifact and return rows ready for plotting."""

    if payload.get("schema_version") != RUNTIME_SCORE_SCHEMA_VERSION:
        raise ValueError("runtime score artifact has an unexpected schema_version")
    if payload.get("artifact_type") != "day26_runtime_score_comparison":
        raise ValueError("runtime score artifact has an unexpected artifact_type")
    if payload.get("status") != "completed":
        raise ValueError("runtime score artifact is not completed")
    manifest = _mapping(payload.get("seed_manifest"), name="seed_manifest")
    seeds = _seed_tuple(manifest.get("episode_seeds"), name="episode_seeds")
    if len(seeds) < 30:
        raise ValueError("runtime score artifact needs at least 30 episode seeds")
    if manifest.get("count") != len(seeds) or manifest.get("unique") is not True:
        raise ValueError("runtime score artifact seed manifest is inconsistent")
    if manifest.get("selection_rule") != "predeclared_before_runtime_results":
        raise ValueError("runtime score artifact seed selection was not predeclared")
    per_runtime = _mapping(payload.get("per_runtime"), name="per_runtime")
    if expected_targets is None:
        protocol = payload.get("evaluation_protocol")
        protocol_mapping = protocol if isinstance(protocol, Mapping) else {}
        raw_targets = protocol_mapping.get("runtime_targets")
        targets = tuple(str(value) for value in raw_targets or tuple(per_runtime))
    else:
        targets = tuple(expected_targets)
    if targets not in (RUNTIME_SCORE_REQUIRED_TARGETS, RUNTIME_SCORE_TARGETS):
        raise ValueError("runtime score artifact has an unsupported target set")
    if set(per_runtime) != set(targets) or len(per_runtime) != len(targets):
        raise ValueError("runtime score artifact targets do not match the expected set")
    rows_by_runtime: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    for runtime in targets:
        runtime_payload = _mapping(
            per_runtime.get(runtime), name=f"per_runtime.{runtime}"
        )
        rows = runtime_payload.get("episodes")
        normalized = validate_runtime_episode_rows(
            rows,
            runtime=runtime,
            expected_seeds=seeds,
        )
        stored_aggregate = _mapping(
            runtime_payload.get("aggregate"),
            name=f"per_runtime.{runtime}.aggregate",
        )
        recomputed = aggregate_runtime_scores(
            normalized,
            runtime=runtime,
            expected_seeds=seeds,
        )
        for field in (
            "episode_count",
            "successful_episode_count",
            "failure_count",
            "mean",
            "median",
            "std",
            "p10",
            "p90",
            "min",
            "max",
            "episode_length_statistics",
        ):
            if stored_aggregate.get(field) != recomputed.get(field):
                raise ValueError(
                    f"stored aggregate disagrees with raw rows: {runtime}.{field}"
                )
        rows_by_runtime[runtime] = normalized
        aggregates[runtime] = dict(stored_aggregate)
    raw_paired = _mapping(
        payload.get("paired_score_differences"),
        name="paired_score_differences",
    )
    paired_payloads: dict[str, Mapping[str, Any]] = {}
    for name, (left_runtime, right_runtime) in PAIRING_DEFINITIONS.items():
        if left_runtime not in targets or right_runtime not in targets:
            continue
        stored = _mapping(raw_paired.get(name), name=f"paired_score_differences.{name}")
        recomputed = paired_score_differences(
            rows_by_runtime[left_runtime],
            rows_by_runtime[right_runtime],
            left_runtime=left_runtime,
            right_runtime=right_runtime,
            expected_seeds=seeds,
        )
        for field in (
            "left_runtime",
            "right_runtime",
            "declared_seed_count",
            "paired_episode_count",
            "excluded_seeds",
            "differences",
            "statistics",
        ):
            if stored.get(field) != recomputed.get(field):
                raise ValueError(
                    f"stored paired comparison disagrees with raw rows: {name}.{field}"
                )
        paired_payloads[name] = stored
    return {
        "episode_seeds": seeds,
        "targets": targets,
        "rows_by_runtime": rows_by_runtime,
        "aggregates": aggregates,
        "paired_score_differences": paired_payloads,
    }


__all__ = [
    "PAIRING_DEFINITIONS",
    "RUNTIME_SCORE_SCHEMA_VERSION",
    "RUNTIME_SCORE_REQUIRED_TARGETS",
    "RUNTIME_SCORE_TARGETS",
    "RuntimeScoreConfig",
    "aggregate_runtime_scores",
    "bootstrap_mean_ci",
    "paired_score_differences",
    "score_statistics",
    "validate_runtime_episode_rows",
    "validate_runtime_score_artifact",
]
