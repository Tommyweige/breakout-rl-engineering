"""Offline association classifier for the Issue 27 RGB dropout audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def classify_dropout_events(
    trace: Sequence[Mapping[str, Any]],
    *,
    window_frames: int = 30,
    minimum_dropout_frames: int = 5,
) -> dict[str, Any]:
    """Classify wrapper life-loss events using only the completed visual trace.

    Trace rows describe the pre-action observation at ``observation_ale_frame``
    and the action result at ``action_result_ale_frame``. The wrapper event is
    timestamped at the latter boundary. A qualifying dropout begins on the
    frame immediately after a directly detected descending observation whose
    predicted paddle-contact horizon is in [0, 30].
    """

    if window_frames < 1 or minimum_dropout_frames < 1:
        raise ValueError("window_frames and minimum_dropout_frames must be positive")
    episode_groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in trace:
        episode_groups.setdefault(int(row.get("episode_seed", 0)), []).append(row)
    if len(episode_groups) > 1:
        combined_events: list[dict[str, Any]] = []
        for seed, episode_rows in sorted(episode_groups.items()):
            per_episode = classify_dropout_events(
                episode_rows,
                window_frames=window_frames,
                minimum_dropout_frames=minimum_dropout_frames,
            )
            combined_events.extend(
                {**event, "episode_seed": seed} for event in per_episode["events"]
            )
        event_count = len(combined_events)
        ambiguous_count = sum(event["status"] == "ambiguous" for event in combined_events)
        associated_count = sum(bool(event["associated"]) for event in combined_events)
        associated_seeds = {
            int(event["episode_seed"])
            for event in combined_events
            if event["associated"]
        }
        fraction = associated_count / event_count if event_count else None
        ambiguous_fraction = ambiguous_count / event_count if event_count else None
        if (
            event_count >= 6
            and ambiguous_fraction is not None
            and ambiguous_fraction <= 0.10
            and fraction is not None
            and fraction >= 2 / 3
            and len(associated_seeds) >= 2
        ):
            classification = "PROMOTED"
        elif (
            event_count >= 6
            and ambiguous_fraction is not None
            and ambiguous_fraction <= 0.10
            and fraction is not None
            and fraction <= 1 / 6
        ):
            classification = "REJECTED"
        else:
            classification = "INCONCLUSIVE"
        counts: dict[str, int] = {}
        for seed in sorted(associated_seeds):
            counts[str(seed)] = sum(
                event["associated"] and event["episode_seed"] == seed
                for event in combined_events
            )
        return {
            "classification": classification,
            "all_wrapper_reported_life_loss_events": event_count,
            "associated_life_loss_events": associated_count,
            "classifiable_life_loss_events": event_count - ambiguous_count,
            "ambiguous_life_loss_events": ambiguous_count,
            "associated_life_loss_fraction": fraction,
            "ambiguous_life_loss_fraction": ambiguous_fraction,
            "associated_events_by_episode_seed": counts,
            "events": combined_events,
        }
    rows = sorted(trace, key=lambda row: int(row["observation_ale_frame"]))
    by_frame: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        frame = int(row["observation_ale_frame"])
        if frame in by_frame:
            raise ValueError(f"duplicate observation ALE frame {frame}")
        by_frame[frame] = row

    events = [
        row for row in rows if bool(row.get("wrapper_life_loss_event", False))
    ]
    event_results: list[dict[str, Any]] = []
    for event in events:
        event_frame = int(event["action_result_ale_frame"])
        window_start = event_frame - window_frames
        window = [
            (frame, row)
            for frame, row in sorted(by_frame.items())
            if window_start <= frame < event_frame
        ]
        window_is_complete = all(
            right[0] == left[0] + 1 for left, right in zip(window, window[1:])
        )
        if (
            not window
            or not window_is_complete
            or not bool(event.get("trace_semantics_valid", True))
        ):
            event_results.append(
                {"event_ale_frame": event_frame, "status": "ambiguous", "associated": False}
            )
            continue

        candidates: list[dict[str, Any]] = []
        index = 0
        while index < len(window):
            frame, row = window[index]
            if bool(row.get("ball_directly_detected", False)):
                index += 1
                continue
            run_start = frame
            run: list[tuple[int, Mapping[str, Any]]] = []
            previous_frame: int | None = None
            while index < len(window):
                current_frame, current = window[index]
                if bool(current.get("ball_directly_detected", False)):
                    break
                if previous_frame is not None and current_frame != previous_frame + 1:
                    break
                run.append((current_frame, current))
                previous_frame = current_frame
                index += 1
            # The beginning of a miss run must be exactly one native frame
            # after the qualifying directly observed descending estimate.
            preceding = by_frame.get(run_start - 1)
            if len(run) >= minimum_dropout_frames and preceding is not None:
                vy = preceding.get("last_directly_observed_vy")
                horizon = preceding.get("predicted_paddle_intercept_horizon_frames")
                try:
                    vy_value = float(vy)
                    horizon_value = float(horizon)
                except (TypeError, ValueError):
                    vy_value = horizon_value = float("nan")
                if (
                    bool(preceding.get("ball_directly_detected", False))
                    and vy_value > 0.0
                    and 0.0 <= horizon_value <= window_frames
                ):
                    candidates.append(
                        {
                            "start_ale_frame": run_start,
                            "end_ale_frame": run[-1][0],
                            "length_frames": len(run),
                            "preceding_vy": vy_value,
                            "preceding_intercept_horizon_frames": horizon_value,
                        }
                    )
            if index < len(window) and window[index][0] == run_start:
                index += 1

        associated = bool(candidates)
        event_results.append(
            {
                "event_ale_frame": event_frame,
                "status": "associated" if associated else "not_associated",
                "associated": associated,
                "qualifying_dropout_runs": candidates,
            }
        )

    event_count = len(event_results)
    ambiguous_count = sum(row["status"] == "ambiguous" for row in event_results)
    associated_count = sum(bool(row["associated"]) for row in event_results)
    classifiable_count = event_count - ambiguous_count
    fraction = associated_count / event_count if event_count else None
    ambiguous_fraction = ambiguous_count / event_count if event_count else None
    episode_counts: dict[int, int] = {}
    for row, event in zip(event_results, events, strict=True):
        if row["associated"]:
            seed = int(event["episode_seed"])
            episode_counts[seed] = episode_counts.get(seed, 0) + 1

    if (
        event_count >= 6
        and ambiguous_fraction is not None
        and ambiguous_fraction <= 0.10
        and fraction is not None
        and fraction >= 2 / 3
        and len(episode_counts) >= 2
    ):
        classification = "PROMOTED"
    elif (
        event_count >= 6
        and ambiguous_fraction is not None
        and ambiguous_fraction <= 0.10
        and fraction is not None
        and fraction <= 1 / 6
    ):
        classification = "REJECTED"
    else:
        classification = "INCONCLUSIVE"

    return {
        "classification": classification,
        "all_wrapper_reported_life_loss_events": event_count,
        "associated_life_loss_events": associated_count,
        "classifiable_life_loss_events": classifiable_count,
        "ambiguous_life_loss_events": ambiguous_count,
        "associated_life_loss_fraction": fraction,
        "ambiguous_life_loss_fraction": ambiguous_fraction,
        "associated_events_by_episode_seed": {
            str(seed): count for seed, count in sorted(episode_counts.items())
        },
        "events": event_results,
    }


__all__ = ["classify_dropout_events"]
