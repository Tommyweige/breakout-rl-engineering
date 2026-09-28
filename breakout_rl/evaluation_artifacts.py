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
EVALUATION_ARTIFACT_SCHEMA_VERSION = 3
SUPPORTED_EVALUATION_ARTIFACT_SCHEMA_VERSIONS = (
    1,
    2,
    EVALUATION_ARTIFACT_SCHEMA_VERSION,
)
ACTION_DISTRIBUTION_SEMANTICS = "executed/wrapper-resolved action"
COMPLETION_EPISODE_FIELDS = (
    "cleared",
    "clear_agent_step",
    "clear_emulator_frame",
    "clear_score",
    "lives_remaining_at_clear",
    "completion_detection_source",
)
COMPLETION_SUMMARY_FIELDS = (
    "total_episodes",
    "clear_detection_available",
    "clear_detection_unknown_count",
    "clear_count",
    "clear_rate",
    "per_seed_clears",
    "failure_stop_reasons",
    "clear_time_sample_count",
    "clear_emulator_frame_sample_count",
    "best_clear_steps",
    "median_clear_steps",
    "mean_clear_steps",
    "p90_clear_steps",
    "best_clear_emulator_frame",
    "median_clear_emulator_frame",
    "mean_clear_emulator_frame",
    "p90_clear_emulator_frame",
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
    schema_version = payload.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(f"{source}: schema_version must be an integer")
    if schema_version not in SUPPORTED_EVALUATION_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(
            f"{source}: unsupported evaluation artifact schema_version="
            f"{schema_version}"
        )
    if not isinstance(payload.get("per_episode"), list):
        raise ValueError(f"{source}: per_episode must be an array")
    if schema_version in (2, EVALUATION_ARTIFACT_SCHEMA_VERSION):
        required = (
            "action_distribution_semantics",
            "requested_action_distribution",
            "executed_action_distribution",
            "auto_fire_count",
            "auto_fire_reason_counts",
        )
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(
                f"{source}: schema v{schema_version} is missing "
                + ", ".join(missing)
            )
        if payload["action_distribution_semantics"] != ACTION_DISTRIBUTION_SEMANTICS:
            raise ValueError(
                f"{source}: schema v2 action_distribution_semantics must be "
                f"{ACTION_DISTRIBUTION_SEMANTICS!r}"
            )
        for field in (
            "requested_action_distribution",
            "executed_action_distribution",
            "auto_fire_reason_counts",
        ):
            if not isinstance(payload[field], Mapping):
                raise ValueError(f"{source}: schema v2 {field} must be an object")
        if isinstance(payload["auto_fire_count"], bool) or not isinstance(
            payload["auto_fire_count"], int
        ):
            raise ValueError(f"{source}: schema v2 auto_fire_count must be an integer")
        for index, row in enumerate(payload["per_episode"]):
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}: per_episode[{index}] must be an object")
            missing_row = [field for field in required if field not in row]
            if missing_row:
                raise ValueError(
                    f"{source}: schema v2 per_episode[{index}] is missing "
                    + ", ".join(missing_row)
                )
            if row["action_distribution_semantics"] != ACTION_DISTRIBUTION_SEMANTICS:
                raise ValueError(
                    f"{source}: schema v2 per_episode[{index}] has invalid "
                    "action_distribution_semantics"
                )
    if schema_version == EVALUATION_ARTIFACT_SCHEMA_VERSION:
        required = (
            "evaluation_status",
            "completion_detector",
            "source_provenance",
            "contract_provenance",
            "verified_clears",
        )
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(
                f"{source}: schema v3 is missing " + ", ".join(missing)
            )
        if payload["evaluation_status"] != "completed":
            raise ValueError(
                f"{source}: schema v3 evaluation_status must be 'completed'"
            )
        for field in (
            "completion_detector",
            "source_provenance",
            "contract_provenance",
        ):
            if not isinstance(payload[field], Mapping):
                raise ValueError(f"{source}: schema v3 {field} must be an object")
        if not isinstance(payload["verified_clears"], list):
            raise ValueError(f"{source}: schema v3 verified_clears must be an array")
        derived_clears: list[Mapping[str, Any]] = []
        for index, row in enumerate(payload["per_episode"]):
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}: per_episode[{index}] must be an object")
            missing_row = [
                field for field in COMPLETION_EPISODE_FIELDS if field not in row
            ]
            if missing_row:
                raise ValueError(
                    f"{source}: schema v3 per_episode[{index}] is missing "
                    + ", ".join(missing_row)
                )
            if row["cleared"] is True and not isinstance(
                row.get("completion_provenance"), Mapping
            ):
                raise ValueError(
                    f"{source}: schema v3 per_episode[{index}] clear is missing "
                    "completion_provenance"
                )
            if row["cleared"] is True:
                provenance = row["completion_provenance"]
                if (
                    provenance.get("evaluation_seed") != row.get("evaluation_seed")
                    or provenance.get("episode_seed") != row.get("episode_seed")
                    or provenance.get("episode_index") != row.get("episode_index")
                    or provenance.get("clear_agent_step") != row.get("clear_agent_step")
                    or provenance.get("clear_emulator_frame")
                    != row.get("clear_emulator_frame")
                ):
                    raise ValueError(
                        f"{source}: schema v3 per_episode[{index}] completion "
                        "provenance disagrees with its episode"
                    )
                derived_clears.append(provenance)
        if list(payload["verified_clears"]) != derived_clears:
            raise ValueError(
                f"{source}: schema v3 verified_clears disagrees with per_episode"
            )
        normalized_rows = validate_episode_rows(
            payload,
            source=source,
            require_complete=False,
        )
        computed_summary = summary_from_episode_rows(normalized_rows)
        validate_embedded_summary(
            payload,
            computed_summary,
            source=source,
            require_completion_fields=True,
        )
    else:
        # Older artifacts did not distinguish clears from normal episode ends.
        # Add explicit unknowns so `complete=true` is never inferred as a clear.
        for row in payload["per_episode"]:
            if isinstance(row, dict):
                row.setdefault("cleared", None)
                for field in COMPLETION_EPISODE_FIELDS[1:]:
                    row.setdefault(field, None)
    return payload


def summarize_returns(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize all episode returns as a population, preserving spread."""

    if not values:
        raise ValueError("at least one episode return is required")
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError("episode returns must be finite")
    ordered = sorted(parsed)

    def _percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)

    return {
        "count": len(parsed),
        "mean_return": float(fmean(parsed)),
        "median_return": float(median(parsed)),
        "std_return": float(pstdev(parsed)),
        "p10_return": _percentile(0.10),
        "p90_return": _percentile(0.90),
        "min_return": float(min(parsed)),
        "max_return": float(max(parsed)),
    }


def _optional_finite_float(value: Any, *, name: str, source: str | Path) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {name} must be finite when present") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{source}: {name} must be finite when present")
    return parsed


def _optional_nonnegative_int(
    value: Any,
    *,
    name: str,
    source: str | Path,
    minimum: int = 0,
) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{source}: {name} must be an integer when present")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {name} must be an integer when present") from error
    if isinstance(value, float) and (not math.isfinite(value) or value != parsed):
        raise ValueError(f"{source}: {name} must be an integer when present")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise ValueError(f"{source}: {name} must be an integer when present")
    if parsed < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{source}: {name} must be a {qualifier} integer")
    return parsed


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
        raw_life_loss_count = raw_row.get("life_loss_count", 0)
        if isinstance(raw_life_loss_count, bool):
            raise ValueError(f"{source_path}: life_loss_count must be a non-negative integer")
        try:
            life_loss_count = int(raw_life_loss_count)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{source_path}: life_loss_count must be a non-negative integer"
            ) from error
        if life_loss_count < 0:
            raise ValueError(f"{source_path}: life_loss_count must be a non-negative integer")
        if isinstance(raw_life_loss_count, str):
            if str(life_loss_count) != raw_life_loss_count.strip():
                raise ValueError(
                    f"{source_path}: life_loss_count must be a non-negative integer"
                )
        elif isinstance(raw_life_loss_count, float):
            if not math.isfinite(raw_life_loss_count) or raw_life_loss_count != life_loss_count:
                raise ValueError(
                    f"{source_path}: life_loss_count must be a non-negative integer"
                )
        elif not isinstance(raw_life_loss_count, int):
            raise ValueError(f"{source_path}: life_loss_count must be a non-negative integer")
        score_per_life = _optional_finite_float(
            raw_row.get("score_per_life"),
            name="score_per_life",
            source=source_path,
        )
        expected_score_per_life = episode_return / max(1, life_loss_count)
        if score_per_life is None:
            score_per_life = expected_score_per_life
        elif not math.isclose(score_per_life, expected_score_per_life):
            raise ValueError(f"{source_path}: score_per_life disagrees with raw score")
        frames_between_life_losses = _optional_finite_float(
            raw_row.get("frames_between_life_losses"),
            name="frames_between_life_losses",
            source=source_path,
        )
        time_to_first_life_loss = raw_row.get("time_to_first_life_loss")
        if time_to_first_life_loss not in (None, ""):
            try:
                time_to_first_life_loss = int(time_to_first_life_loss)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{source_path}: time_to_first_life_loss must be an integer"
                ) from error
            if time_to_first_life_loss < 1 or time_to_first_life_loss > episode_length:
                raise ValueError(
                    f"{source_path}: time_to_first_life_loss is outside episode bounds"
                )
        else:
            time_to_first_life_loss = None
        life_losses_per_1000_steps = _optional_finite_float(
            raw_row.get("life_losses_per_1000_steps"),
            name="life_losses_per_1000_steps",
            source=source_path,
        )
        expected_life_loss_rate = life_loss_count / episode_length * 1000.0
        if life_losses_per_1000_steps is None:
            life_losses_per_1000_steps = expected_life_loss_rate
        elif not math.isclose(
            life_losses_per_1000_steps,
            expected_life_loss_rate,
        ):
            raise ValueError(
                f"{source_path}: life_losses_per_1000_steps disagrees with life-loss count"
            )
        raw_cleared = raw_row.get("cleared")
        if raw_cleared is not None and not isinstance(raw_cleared, bool):
            raise ValueError(f"{source_path}: cleared must be a boolean or null")
        clear_agent_step = _optional_nonnegative_int(
            raw_row.get("clear_agent_step"),
            name="clear_agent_step",
            source=source_path,
            minimum=1,
        )
        clear_emulator_frame = _optional_nonnegative_int(
            raw_row.get("clear_emulator_frame"),
            name="clear_emulator_frame",
            source=source_path,
        )
        clear_score = _optional_finite_float(
            raw_row.get("clear_score"),
            name="clear_score",
            source=source_path,
        )
        lives_remaining_at_clear = _optional_nonnegative_int(
            raw_row.get("lives_remaining_at_clear"),
            name="lives_remaining_at_clear",
            source=source_path,
        )
        completion_detection_source = raw_row.get("completion_detection_source")
        if completion_detection_source in (None, ""):
            completion_detection_source = None
        elif not isinstance(completion_detection_source, str):
            raise ValueError(
                f"{source_path}: completion_detection_source must be a string or null"
            )
        if raw_cleared is True:
            if clear_agent_step is None or clear_score is None:
                raise ValueError(
                    f"{source_path}: a verified clear requires clear_agent_step "
                    "and clear_score"
                )
            if clear_agent_step > episode_length:
                raise ValueError(
                    f"{source_path}: clear_agent_step is outside episode bounds"
                )
            if clear_score > episode_return and not math.isclose(
                clear_score, episode_return, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"{source_path}: clear_score cannot exceed the episode raw score"
                )
            if completion_detection_source is None:
                raise ValueError(
                    f"{source_path}: a verified clear requires a detection source"
                )
        elif any(
            value is not None
            for value in (
                clear_agent_step,
                clear_emulator_frame,
                clear_score,
                lives_remaining_at_clear,
            )
        ):
            raise ValueError(
                f"{source_path}: clear timing and score fields require cleared=true"
            )
        raw_completion_outcome = raw_row.get("completion_outcome")
        if raw_completion_outcome is not None and not isinstance(
            raw_completion_outcome, str
        ):
            raise ValueError(f"{source_path}: completion_outcome must be a string or null")
        if raw_completion_outcome not in (None, ""):
            allowed_outcomes = {
                "cleared",
                "clear_status_unavailable",
                "game_over",
                "time_limit",
                "truncated",
                "terminated",
                "incomplete",
            }
            if raw_completion_outcome not in allowed_outcomes:
                raise ValueError(f"{source_path}: invalid completion_outcome")
            if raw_cleared is True and raw_completion_outcome != "cleared":
                raise ValueError(
                    f"{source_path}: cleared=true requires completion_outcome='cleared'"
                )
            if (
                raw_cleared is None
                and raw_completion_outcome != "clear_status_unavailable"
            ):
                raise ValueError(
                    f"{source_path}: cleared=null requires completion_outcome="
                    "'clear_status_unavailable'"
                )
            if raw_cleared is False and raw_completion_outcome in {
                "cleared",
                "clear_status_unavailable",
            }:
                raise ValueError(
                    f"{source_path}: completion_outcome disagrees with cleared=false"
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
                "life_loss_count": life_loss_count,
                "score_per_life": score_per_life,
                "frames_between_life_losses": frames_between_life_losses,
                "time_to_first_life_loss": time_to_first_life_loss,
                "life_losses_per_1000_steps": life_losses_per_1000_steps,
                "cleared": raw_cleared,
                "clear_agent_step": clear_agent_step,
                "clear_emulator_frame": clear_emulator_frame,
                "clear_score": clear_score,
                "lives_remaining_at_clear": lives_remaining_at_clear,
                "completion_detection_source": completion_detection_source,
                "completion_outcome": raw_completion_outcome,
                "completion_provenance": raw_row.get("completion_provenance"),
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
    life_loss_counts = [int(row.get("life_loss_count", 0)) for row in rows]
    summary["life_loss_count"] = int(sum(life_loss_counts))
    summary["mean_life_loss_count"] = float(fmean(life_loss_counts))
    summary["mean_score_per_life"] = float(
        fmean(float(row.get("score_per_life", 0.0)) for row in rows)
    )
    frames_between = [
        float(row["frames_between_life_losses"])
        for row in rows
        if row.get("frames_between_life_losses") is not None
    ]
    first_loss_times = [
        float(row["time_to_first_life_loss"])
        for row in rows
        if row.get("time_to_first_life_loss") is not None
    ]
    summary["mean_frames_between_life_losses"] = (
        float(fmean(frames_between)) if frames_between else None
    )
    summary["mean_time_to_first_life_loss"] = (
        float(fmean(first_loss_times)) if first_loss_times else None
    )
    summary["life_losses_per_1000_steps"] = float(
        sum(life_loss_counts)
        / max(1, sum(int(row["episode_length"]) for row in rows))
        * 1000.0
    )
    summary["complete_episodes"] = sum(bool(row["complete"]) for row in rows)
    known_clear_rows = [row for row in rows if row.get("cleared") is not None]
    clear_rows = [row for row in rows if row.get("cleared") is True]
    clear_detection_available = len(known_clear_rows) == len(rows)
    summary.update(
        {
            "total_episodes": len(rows),
            "clear_detection_available": clear_detection_available,
            "clear_detection_unknown_count": len(rows) - len(known_clear_rows),
            "clear_count": len(clear_rows),
            "clear_rate": (
                float(len(clear_rows) / len(rows))
                if clear_detection_available
                else None
            ),
        }
    )

    per_seed_rows: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        seed_value = row.get("evaluation_seed")
        if seed_value is None:
            continue
        try:
            seed = int(seed_value)
        except (TypeError, ValueError):
            continue
        per_seed_rows.setdefault(seed, []).append(row)
    summary["per_seed_clears"] = [
        {
            "evaluation_seed": seed,
            "total_episodes": len(seed_rows),
            "clear_count": sum(item.get("cleared") is True for item in seed_rows),
            "clear_rate": (
                float(
                    sum(item.get("cleared") is True for item in seed_rows)
                    / len(seed_rows)
                )
                if all(item.get("cleared") is not None for item in seed_rows)
                else None
            ),
        }
        for seed, seed_rows in sorted(per_seed_rows.items())
    ]

    failure_stop_reasons: Counter[str] = Counter()
    for row in rows:
        if row.get("cleared") is not False:
            continue
        outcome = row.get("completion_outcome")
        if outcome in (None, ""):
            outcome = row.get("stop_reason", "unknown")
        failure_stop_reasons[str(outcome)] += 1
    summary["failure_stop_reasons"] = dict(sorted(failure_stop_reasons.items()))

    def _clear_timing_stats(
        field: str,
    ) -> tuple[int, int | None, float | None, float | None, float | None]:
        values = sorted(
            int(row[field])
            for row in clear_rows
            if row.get(field) is not None
        )
        if not values:
            return 0, None, None, None, None
        # P90 uses the nearest-rank definition: ceil(0.9 * n) - 1.
        p90_index = math.ceil(0.9 * len(values)) - 1
        return (
            len(values),
            values[0],
            float(median(values)),
            float(fmean(values)),
            float(values[p90_index]),
        )

    step_stats = _clear_timing_stats("clear_agent_step")
    frame_stats = _clear_timing_stats("clear_emulator_frame")
    summary.update(
        {
            "clear_time_sample_count": step_stats[0],
            "clear_emulator_frame_sample_count": frame_stats[0],
            "best_clear_steps": step_stats[1],
            "median_clear_steps": step_stats[2],
            "mean_clear_steps": step_stats[3],
            "p90_clear_steps": step_stats[4],
            "best_clear_emulator_frame": frame_stats[1],
            "median_clear_emulator_frame": frame_stats[2],
            "mean_clear_emulator_frame": frame_stats[3],
            "p90_clear_emulator_frame": frame_stats[4],
        }
    )
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
                "action_distribution_semantics": ACTION_DISTRIBUTION_SEMANTICS,
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
    require_completion_fields: bool = False,
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
    if require_completion_fields:
        fields += COMPLETION_SUMMARY_FIELDS
    for field in fields:
        if field not in embedded:
            raise ValueError(f"{source}: summary is missing {field}")
        if computed[field] is None:
            if embedded[field] is not None:
                raise ValueError(
                    f"{source}: summary.{field} does not match per_episode artifacts"
                )
            continue
        expected = computed[field]
        if isinstance(expected, (bool, dict, list, str)):
            matches = embedded[field] == expected
        else:
            try:
                matches = math.isclose(
                    float(embedded[field]),
                    float(expected),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            except (TypeError, ValueError):
                matches = False
        if not matches:
            raise ValueError(
                f"{source}: summary.{field} does not match per_episode artifacts"
            )


__all__ = [
    "ACTION_DISTRIBUTION_SEMANTICS",
    "EVALUATION_ARTIFACT_SCHEMA_VERSION",
    "read_evaluation_results",
    "summarize_returns",
    "summary_from_episode_rows",
    "SUPPORTED_EVALUATION_ARTIFACT_SCHEMA_VERSIONS",
    "TIME_LIMIT_SUMMARY_FIELDS",
    "validate_embedded_summary",
    "validate_episode_rows",
]
