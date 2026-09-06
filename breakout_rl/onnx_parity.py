"""Numerical and greedy-action parity metrics for exported Breakout policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from breakout_rl.inference import q_values_to_action


def _q_values(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N, action_count)")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _actions(
    values: np.ndarray,
    provided: Any,
    *,
    name: str,
) -> np.ndarray:
    if provided is None:
        return np.asarray(q_values_to_action(values), dtype=np.int64)
    array = np.asarray(provided)
    if array.shape != (values.shape[0],):
        raise ValueError(f"{name} must have shape ({values.shape[0]},)")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer action indices")
    result = np.ascontiguousarray(array, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= values.shape[1]):
        raise ValueError(f"{name} contains an action outside the Q-value columns")
    return result


def _margin_summary(values: np.ndarray, sample_ids: np.ndarray) -> dict[str, Any]:
    if values.shape[1] == 1:
        margins = np.zeros(values.shape[0], dtype=np.float64)
    else:
        ordered = np.sort(values, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
    minimum = float(np.min(margins))
    return {
        "min": minimum,
        "mean": float(np.mean(margins)),
        "max": float(np.max(margins)),
        "per_sample": [float(value) for value in margins],
        "smallest_sample_ids": [
            int(sample_ids[index])
            for index, value in enumerate(margins)
            if value == minimum
        ],
    }


def compare_q_values(
    reference_q_values: Any,
    candidate_q_values: Any,
    *,
    reference_actions: Any = None,
    candidate_actions: Any = None,
    sample_ids: Sequence[int] | None = None,
    relative_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compare one candidate output matrix against an independent reference.

    The returned values are JSON-serializable and retain per-sample errors so a
    later visualization can be rebuilt from the comparison artifact itself.
    """

    reference = _q_values(reference_q_values, name="reference_q_values")
    candidate = _q_values(candidate_q_values, name="candidate_q_values")
    if reference.shape != candidate.shape:
        raise ValueError(
            "reference_q_values and candidate_q_values must have the same shape; "
            f"received {reference.shape} and {candidate.shape}"
        )
    try:
        parsed_epsilon = float(relative_epsilon)
    except (TypeError, ValueError) as error:
        raise TypeError("relative_epsilon must be a positive finite number") from error
    if not np.isfinite(parsed_epsilon) or parsed_epsilon <= 0.0:
        raise ValueError("relative_epsilon must be a positive finite number")

    if sample_ids is None:
        ids = np.arange(reference.shape[0], dtype=np.int64)
    else:
        ids = np.asarray(list(sample_ids), dtype=np.int64)
        if ids.shape != (reference.shape[0],):
            raise ValueError(
                f"sample_ids must contain {reference.shape[0]} values"
            )
    reference_action_array = _actions(
        reference,
        reference_actions,
        name="reference_actions",
    )
    candidate_action_array = _actions(
        candidate,
        candidate_actions,
        name="candidate_actions",
    )
    absolute_error = np.abs(reference - candidate)
    relative_error = absolute_error / np.maximum(np.abs(reference), parsed_epsilon)
    action_matches = reference_action_array == candidate_action_array
    disagreement_indices = np.flatnonzero(~action_matches)
    reference_margins = _margin_summary(reference, ids)
    candidate_margins = _margin_summary(candidate, ids)
    disagreement_details = [
        {
            "sample_id": int(ids[index]),
            "reference_action": int(reference_action_array[index]),
            "candidate_action": int(candidate_action_array[index]),
            "reference_q_values": [float(value) for value in reference[index]],
            "candidate_q_values": [float(value) for value in candidate[index]],
            "reference_top_2_q_margin": reference_margins["per_sample"][index],
        }
        for index in disagreement_indices
    ]
    return {
        "sample_count": int(reference.shape[0]),
        "action_count": int(reference.shape[1]),
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "max_relative_error": float(np.max(relative_error)),
        "mean_relative_error": float(np.mean(relative_error)),
        "per_sample_max_absolute_error": [
            float(value) for value in np.max(absolute_error, axis=1)
        ],
        "per_sample_mean_absolute_error": [
            float(value) for value in np.mean(absolute_error, axis=1)
        ],
        "per_sample_max_relative_error": [
            float(value) for value in np.max(relative_error, axis=1)
        ],
        "absolute_error_values": [float(value) for value in absolute_error.ravel()],
        "reference_q_values": [
            [float(value) for value in row] for row in reference
        ],
        "candidate_q_values": [
            [float(value) for value in row] for row in candidate
        ],
        "action_agreement_rate": float(np.mean(action_matches)),
        "disagreement_sample_ids": [
            int(ids[index]) for index in disagreement_indices
        ],
        "reference_actions": [int(value) for value in reference_action_array],
        "candidate_actions": [int(value) for value in candidate_action_array],
        "reference_top_2_q_margin": reference_margins,
        "candidate_top_2_q_margin": candidate_margins,
        "disagreement_details": disagreement_details,
    }


__all__ = ["compare_q_values"]
