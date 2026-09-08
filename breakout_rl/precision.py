"""Public contracts and metrics for the Day 25 precision experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from breakout_rl.onnx_parity import compare_q_values


class PrecisionBlockedError(RuntimeError):
    """Raised when the formal Day 25 CUDA matrix cannot be verified."""


@dataclass(frozen=True)
class PrecisionThresholds:
    """Pre-declared acceptance limits for one Q-value comparison."""

    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    mean_relative_error: float
    action_agreement_rate: float

    def __post_init__(self) -> None:
        for name in (
            "max_absolute_error",
            "mean_absolute_error",
            "max_relative_error",
            "mean_relative_error",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        agreement = float(self.action_agreement_rate)
        if not np.isfinite(agreement) or not 0.0 <= agreement <= 1.0:
            raise ValueError("action_agreement_rate must be between 0 and 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PrecisionThresholds":
        """Build thresholds from the machine-readable validation config."""

        if not isinstance(values, Mapping):
            raise TypeError("precision thresholds must be a mapping")
        fields = (
            "max_absolute_error",
            "mean_absolute_error",
            "max_relative_error",
            "mean_relative_error",
            "action_agreement_rate",
        )
        missing = [field for field in fields if field not in values]
        if missing:
            raise ValueError(
                "precision thresholds are missing: " + ", ".join(missing)
            )
        try:
            return cls(**{field: float(values[field]) for field in fields})
        except (TypeError, ValueError) as error:
            raise ValueError("precision thresholds must contain numeric values") from error

    def check(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Return auditable pass/fail checks without changing the metrics."""

        checks = {
            "max_absolute_error": {
                "observed": float(metrics["max_absolute_error"]),
                "limit": self.max_absolute_error,
                "passed": float(metrics["max_absolute_error"])
                <= self.max_absolute_error,
            },
            "mean_absolute_error": {
                "observed": float(metrics["mean_absolute_error"]),
                "limit": self.mean_absolute_error,
                "passed": float(metrics["mean_absolute_error"])
                <= self.mean_absolute_error,
            },
            "max_relative_error": {
                "observed": float(metrics["max_relative_error"]),
                "limit": self.max_relative_error,
                "passed": float(metrics["max_relative_error"])
                <= self.max_relative_error,
            },
            "mean_relative_error": {
                "observed": float(metrics["mean_relative_error"]),
                "limit": self.mean_relative_error,
                "passed": float(metrics["mean_relative_error"])
                <= self.mean_relative_error,
            },
            "action_agreement_rate": {
                "observed": float(metrics["action_agreement_rate"]),
                "limit": self.action_agreement_rate,
                "passed": float(metrics["action_agreement_rate"])
                >= self.action_agreement_rate,
            },
        }
        return {
            "passed": all(bool(value["passed"]) for value in checks.values()),
            "checks": checks,
        }


def require_cuda_precision_matrix(
    *,
    torch_cuda_available: bool,
    onnxruntime_providers: Sequence[str],
) -> None:
    """Fail closed instead of labelling a CPU fallback as a CUDA experiment."""

    missing: list[str] = []
    if not torch_cuda_available:
        missing.append("PyTorch CUDA")
    if "CUDAExecutionProvider" not in {str(provider) for provider in onnxruntime_providers}:
        missing.append("ONNX Runtime CUDAExecutionProvider")
    if missing:
        raise PrecisionBlockedError(
            "Day 25 is blocked: required CUDA precision paths are unavailable: "
            + ", ".join(missing)
            + ". No CPU fallback is permitted."
        )


def compare_precision_outputs(
    reference_q_values: Any,
    candidate_q_values: Any,
    *,
    thresholds: PrecisionThresholds,
    sample_ids: Sequence[int] | None = None,
    relative_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compare one precision candidate and attach its fixed threshold checks."""

    metrics = compare_q_values(
        reference_q_values,
        candidate_q_values,
        sample_ids=sample_ids,
        relative_epsilon=relative_epsilon,
    )
    return {"metrics": metrics, "thresholds": thresholds.check(metrics)}


def compare_evaluation_parity(
    *,
    reference_action_sequences: Sequence[Sequence[int]],
    candidate_action_sequences: Sequence[Sequence[int]],
    reference_returns: Sequence[float],
    candidate_returns: Sequence[float],
    reference_lengths: Sequence[int],
    candidate_lengths: Sequence[int],
) -> dict[str, Any]:
    """Compare fixed-seed gameplay traces without hiding episode differences."""

    if len(reference_action_sequences) != len(candidate_action_sequences):
        raise ValueError("evaluation action sequence counts must match")
    if len(reference_returns) != len(candidate_returns):
        raise ValueError("evaluation return counts must match")
    if len(reference_lengths) != len(candidate_lengths):
        raise ValueError("evaluation length counts must match")
    if not reference_action_sequences:
        raise ValueError("evaluation results must contain at least one episode")
    reference_actions = [
        int(action) for sequence in reference_action_sequences for action in sequence
    ]
    candidate_actions = [
        int(action) for sequence in candidate_action_sequences for action in sequence
    ]
    matched_actions = 0
    action_disagreement_count = 0
    comparable_action_count = 0
    for reference_sequence, candidate_sequence in zip(
        reference_action_sequences,
        candidate_action_sequences,
        strict=True,
    ):
        common_length = min(len(reference_sequence), len(candidate_sequence))
        if common_length:
            reference_prefix = np.asarray(reference_sequence[:common_length], dtype=np.int64)
            candidate_prefix = np.asarray(candidate_sequence[:common_length], dtype=np.int64)
            prefix_matches = np.equal(reference_prefix, candidate_prefix)
            matched_actions += int(np.count_nonzero(prefix_matches))
            action_disagreement_count += int(np.count_nonzero(~prefix_matches))
            comparable_action_count += common_length
        action_disagreement_count += abs(len(reference_sequence) - len(candidate_sequence))
    total_action_count = max(len(reference_actions), len(candidate_actions))
    reference_return_array = np.asarray(reference_returns, dtype=np.float64)
    candidate_return_array = np.asarray(candidate_returns, dtype=np.float64)
    reference_length_array = np.asarray(reference_lengths, dtype=np.int64)
    candidate_length_array = np.asarray(candidate_lengths, dtype=np.int64)
    if not (
        np.isfinite(reference_return_array).all()
        and np.isfinite(candidate_return_array).all()
    ):
        raise ValueError("evaluation returns must be finite")
    return_matches = np.equal(reference_return_array, candidate_return_array)
    length_matches = np.equal(reference_length_array, candidate_length_array)
    return_differences = np.abs(reference_return_array - candidate_return_array)
    return {
        "episode_count": len(reference_returns),
        "reference_action_count": len(reference_actions),
        "candidate_action_count": len(candidate_actions),
        "comparable_action_count": comparable_action_count,
        "action_agreement_rate": float(matched_actions / max(total_action_count, 1)),
        "action_disagreement_count": action_disagreement_count,
        "episode_return_match_rate": float(np.mean(return_matches)),
        "episode_length_match_rate": float(np.mean(length_matches)),
        "max_absolute_return_difference": float(np.max(return_differences)),
        "mean_absolute_return_difference": float(np.mean(return_differences)),
        "return_disagreement_episode_indices": [
            int(index) for index in np.flatnonzero(~return_matches)
        ],
        "length_disagreement_episode_indices": [
            int(index) for index in np.flatnonzero(~length_matches)
        ],
    }


def build_precision_benchmark_id(identity: Mapping[str, Any]) -> str:
    """Build a stable Day 25 benchmark id from lineage and settings."""

    serialized = json.dumps(
        dict(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"day25-{hashlib.sha256(serialized).hexdigest()[:12]}"


__all__ = [
    "PrecisionBlockedError",
    "PrecisionThresholds",
    "build_precision_benchmark_id",
    "compare_evaluation_parity",
    "compare_precision_outputs",
    "require_cuda_precision_matrix",
]
