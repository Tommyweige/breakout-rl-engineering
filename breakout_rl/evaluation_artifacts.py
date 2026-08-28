"""Validation and aggregation helpers for evaluation JSON artifacts."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Mapping, Sequence


TIME_LIMIT_SUMMARY_FIELDS = (
    "finished_episode_count",
    "terminated_count",
    "truncated_count",
    "time_limit_truncated_count",
    "truncation_rate",
    "mean_return_terminated",
    "mean_return_truncated",
    "mean_length_terminated",
    "mean_length_truncated",
)


def read_evaluation_results(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: evaluation results must be a JSON object")
    if not isinstance(payload.get("per_episode"), list):
        raise ValueError(f"{source}: per_episode must be an array")
    return payload


def summarize_returns(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize all episode returns as a population, preserving spread."""

    if not values:
        raise ValueError("at least one episode return is required")
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError("episode returns must be finite")
    return {
        "count": len(parsed),
        "mean_return": float(fmean(parsed)),
        "median_return": float(median(parsed)),
        "std_return": float(pstdev(parsed)),
        "min_return": float(min(parsed)),
        "max_return": float(max(parsed)),
    }


def validate_episode_rows(
    payload: Mapping[str, Any],
    *,
    source: str | Path,
    expected_seeds: Sequence[int] | None = None,
    expected_episodes_per_seed: int | None = None,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Validate per-episode identity and derive completion from env flags.

    The stored ``complete`` field is checked when present, but never trusted
    as the source of truth. A formal result must contain exactly one row for
    each configured ``(evaluation_seed, episode_index)`` pair.
    """

    source_path = Path(source)
    raw_rows = payload.get("per_episode")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{source_path}: per_episode must be a non-empty array")
    parsed_seeds = (
        tuple(int(seed) for seed in expected_seeds)
        if expected_seeds is not None
        else None
    )
    if parsed_seeds is not None and len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError(f"{source_path}: expected evaluation seeds must be unique")
    if expected_episodes_per_seed is not None and expected_episodes_per_seed < 1:
        raise ValueError("expected_episodes_per_seed must be positive")

    rows: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{source_path}: every episode must be an object")
        raw_return = raw_row.get("episode_return", raw_row.get("return"))
        raw_episode_seed = raw_row.get("episode_seed", raw_row.get("seed"))
        try:
            evaluation_seed = int(raw_row["evaluation_seed"])
            episode_index = int(raw_row["episode_index"])
            episode_seed = int(raw_episode_seed)
            episode_return = float(raw_return)
            episode_length = int(raw_row["episode_length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source_path}: malformed episode identity or value") from error
        if not math.isfinite(episode_return) or episode_length < 1:
            raise ValueError(f"{source_path}: episode return/length must be valid")
        identity = (evaluation_seed, episode_index)
        if identity in identities:
            raise ValueError(f"{source_path}: duplicate episode identity {identity}")
        identities.add(identity)
        if parsed_seeds is not None:
            if evaluation_seed not in parsed_seeds:
                raise ValueError(f"{source_path}: unexpected evaluation seed {evaluation_seed}")
            if expected_episodes_per_seed is not None and not 1 <= episode_index <= expected_episodes_per_seed:
                raise ValueError(f"{source_path}: invalid episode index {episode_index}")
            expected_episode_seed = evaluation_seed + episode_index - 1
            if episode_seed != expected_episode_seed:
                raise ValueError(
                    f"{source_path}: episode seed {episode_seed} does not match "
                    f"seed group/index {identity}"
                )
        raw_terminated = raw_row.get("terminated", False)
        raw_truncated = raw_row.get("truncated", False)
        if not isinstance(raw_terminated, bool) or not isinstance(raw_truncated, bool):
            raise ValueError(f"{source_path}: termination flags must be booleans")
        terminated = raw_terminated
        truncated = raw_truncated
        if terminated and truncated:
            raise ValueError(f"{source_path}: terminated and truncated cannot both be true")
        raw_time_limit = raw_row.get("time_limit", False)
        if not isinstance(raw_time_limit, bool):
            raise ValueError(f"{source_path}: time_limit must be a boolean")
        time_limit = raw_time_limit
        if time_limit and not truncated:
            raise ValueError(f"{source_path}: time_limit requires truncated=true")
        complete = terminated or truncated
        if "complete" in raw_row:
            stored_complete = raw_row["complete"]
            if not isinstance(stored_complete, bool):
                raise ValueError(f"{source_path}: complete must be a boolean")
            if stored_complete != complete:
                raise ValueError(f"{source_path}: complete disagrees with termination flags")
        expected_stop_reason = (
            "time_limit"
            if time_limit
            else "terminated"
            if terminated
            else "truncated"
            if truncated
            else "incomplete"
        )
        if "stop_reason" in raw_row and raw_row["stop_reason"] != expected_stop_reason:
            raise ValueError(f"{source_path}: stop_reason disagrees with termination flags")
        if require_complete and not complete:
            raise ValueError(
                f"{source_path}: episode {identity} has neither terminated nor truncated"
            )
        rows.append(
            {
                "evaluation_seed": evaluation_seed,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "episode_return": episode_return,
                "episode_length": episode_length,
                "terminated": terminated,
                "truncated": truncated,
                "time_limit": time_limit,
                "complete": complete,
                "stop_reason": expected_stop_reason,
            }
        )

    if parsed_seeds is not None and expected_episodes_per_seed is not None:
        expected_identities = {
            (seed, episode_index)
            for seed in parsed_seeds
            for episode_index in range(1, expected_episodes_per_seed + 1)
        }
        if identities != expected_identities:
            missing = sorted(expected_identities - identities)
            unexpected = sorted(identities - expected_identities)
            raise ValueError(
                f"{source_path}: episode identities do not match config; "
                f"missing={missing}, unexpected={unexpected}"
            )
    return rows


def summary_from_episode_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one episode row is required")
    summary = summarize_returns([float(row["episode_return"]) for row in rows])
    terminated_rows = [row for row in rows if bool(row["terminated"])]
    truncated_rows = [row for row in rows if bool(row["truncated"])]
    time_limit_rows = [row for row in rows if bool(row.get("time_limit", False))]

    def _mean(field: str, selected: Sequence[Mapping[str, Any]]) -> float | None:
        if not selected:
            return None
        return float(fmean([float(row[field]) for row in selected]))

    summary["mean_episode_length"] = float(
        fmean([int(row["episode_length"]) for row in rows])
    )
    summary["complete_episodes"] = sum(bool(row["complete"]) for row in rows)
    summary.update(
        {
            "finished_episode_count": sum(bool(row["complete"]) for row in rows),
            "terminated_count": len(terminated_rows),
            "truncated_count": len(truncated_rows),
            "time_limit_truncated_count": len(time_limit_rows),
            "truncation_rate": float(len(truncated_rows) / len(rows)),
            "mean_return_terminated": _mean("episode_return", terminated_rows),
            "mean_return_truncated": _mean("episode_return", truncated_rows),
            "mean_length_terminated": _mean("episode_length", terminated_rows),
            "mean_length_truncated": _mean("episode_length", truncated_rows),
        }
    )
    requested_action_counts: Counter[str] = Counter()
    executed_action_counts: Counter[str] = Counter()
    auto_fire_reason_counts: Counter[str] = Counter()
    auto_fire_count = 0
    has_action_provenance = False
    for row in rows:
        raw_requested = row.get("requested_action_distribution")
        raw_executed = row.get(
            "executed_action_distribution",
            row.get("action_distribution"),
        )
        if isinstance(raw_requested, Mapping):
            has_action_provenance = True
            requested_action_counts.update(
                {str(name): int(count) for name, count in raw_requested.items()}
            )
        if isinstance(raw_executed, Mapping):
            has_action_provenance = True
            executed_action_counts.update(
                {str(name): int(count) for name, count in raw_executed.items()}
            )
        raw_reason_counts = row.get("auto_fire_reason_counts")
        if isinstance(raw_reason_counts, Mapping):
            auto_fire_reason_counts.update(
                {str(name): int(count) for name, count in raw_reason_counts.items()}
            )
        try:
            auto_fire_count += int(row.get("auto_fire_count", 0))
        except (TypeError, ValueError):
            raise ValueError("auto_fire_count must be an integer when present") from None
    if has_action_provenance:
        if not requested_action_counts:
            requested_action_counts.update(executed_action_counts)
        summary.update(
            {
                "action_distribution_semantics": "executed/wrapper-resolved action",
                "action_distribution": dict(sorted(executed_action_counts.items())),
                "requested_action_distribution": dict(
                    sorted(requested_action_counts.items())
                ),
                "executed_action_distribution": dict(
                    sorted(executed_action_counts.items())
                ),
                "auto_fire_count": auto_fire_count,
                "auto_fire_reason_counts": dict(
                    sorted(auto_fire_reason_counts.items())
                ),
            }
        )
    return summary


def validate_embedded_summary(
    payload: Mapping[str, Any],
    computed: Mapping[str, Any],
    *,
    source: str | Path,
    require_time_limit_fields: bool = False,
) -> None:
    embedded = payload.get("summary")
    if not isinstance(embedded, Mapping):
        raise ValueError(f"{source}: summary is required")
    fields = (
        "count",
        "mean_return",
        "median_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_episode_length",
        "complete_episodes",
    )
    if require_time_limit_fields:
        fields += TIME_LIMIT_SUMMARY_FIELDS
    for field in fields:
        if field not in embedded:
            raise ValueError(f"{source}: summary is missing {field}")
        if computed[field] is None:
            if embedded[field] is not None:
                raise ValueError(
                    f"{source}: summary.{field} does not match per_episode artifacts"
                )
            continue
        if not math.isclose(
            float(embedded[field]),
            float(computed[field]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"{source}: summary.{field} does not match per_episode artifacts"
            )


__all__ = [
    "read_evaluation_results",
    "summarize_returns",
    "summary_from_episode_rows",
    "TIME_LIMIT_SUMMARY_FIELDS",
    "validate_embedded_summary",
    "validate_episode_rows",
]
