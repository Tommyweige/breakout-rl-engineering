"""Offline Issue #33 intercept-target revision analysis for the frozen #29 trace."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TRACE_SHA256 = "36f6c52d0814e873431eae46cd48061b367164725158513417e35c7eba03d8fe"
RESULTS_SHA256 = "638fb46ce23ffd425ea766c85d71a71db058717f3a5adfab933f34d2b6e87e50"
SEEDS = (101, 202, 303)
MIN_GAP = 5
MAX_GAP = 12
MAX_HORIZON = 30.0
THRESHOLD_PX = 8.0
REQUIRED_ROW_FIELDS = (
    "episode_seed", "episode_ordinal", "observation_ale_frame", "ball_directly_detected",
    "ball_y", "ball_vy", "predicted_intercept_x", "predicted_paddle_intercept_horizon_frames",
)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_horizon(row: dict[str, Any]) -> bool:
    horizon = row.get("predicted_paddle_intercept_horizon_frames")
    return finite(horizon) and 0 <= horizon <= MAX_HORIZON


def analyze_episode(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return eligible events, excluded dropout runs, and episode integrity issues."""
    rows = sorted(rows, key=lambda row: (
        0 if isinstance(row.get("observation_ale_frame"), int) and not isinstance(row.get("observation_ale_frame"), bool) else 1,
        row.get("observation_ale_frame") if isinstance(row.get("observation_ale_frame"), int) and not isinstance(row.get("observation_ale_frame"), bool) else 0,
    ))
    issues: list[str] = []
    frames = [row.get("observation_ale_frame") for row in rows]
    if len(set(frames)) != len(frames):
        issues.append("duplicate observation_ale_frame")
    if any(not isinstance(frame, int) or isinstance(frame, bool) for frame in frames) or any(
        b != a + 1 for a, b in zip(frames, frames[1:])
    ):
        issues.append("missing or invalid observation frame")

    events: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    i = 0
    while i < len(rows):
        if rows[i].get("ball_directly_detected") is not False:
            i += 1
            continue
        start = i
        while i < len(rows) and rows[i].get("ball_directly_detected") is False:
            i += 1
        end = i
        gap = end - start
        previous = rows[start - 1] if start else None
        last_stale = rows[end - 1]
        reacquisition = rows[end] if end < len(rows) else None
        seed = (previous or last_stale).get("episode_seed")
        record: dict[str, Any] = {
            "episode_seed": seed,
            "episode_ordinal": (previous or last_stale).get("episode_ordinal"),
            "preceding_direct_frame": frames[start - 1] if previous else None,
            "gap_start_frame": frames[start],
            "last_non_direct_frame": frames[end - 1],
            "gap_length": gap,
            "reacquisition_frame": frames[end] if reacquisition else None,
        }
        reason = None
        if previous is None or previous.get("ball_directly_detected") is not True:
            reason = "no immediately preceding direct detection"
        elif reacquisition is None or reacquisition.get("ball_directly_detected") is not True:
            reason = "no immediately following direct reacquisition"
        elif gap < MIN_GAP or gap > MAX_GAP:
            reason = "gap length outside 5-12"
        elif not finite(previous.get("ball_y")) or not finite(previous.get("ball_vy")):
            reason = "preceding direct ball_y or ball_vy is not finite"
        elif previous["ball_vy"] <= 0:
            reason = "preceding direct ball_vy is not descending"
        elif not _valid_horizon(previous):
            reason = "preceding direct horizon is not finite in [0, 30]"
        elif not finite(last_stale.get("predicted_intercept_x")):
            reason = "last non-direct predicted_intercept_x is not finite"
        elif not _valid_horizon(last_stale):
            reason = "last non-direct horizon is not finite in [0, 30]"
        elif not finite(reacquisition.get("predicted_intercept_x")):
            reason = "reacquisition predicted_intercept_x is not finite"
        elif not _valid_horizon(reacquisition):
            reason = "reacquisition horizon is not finite in [0, 30]"

        if reason:
            excluded.append({**record, "reason": reason})
            continue
        stale_target = last_stale["predicted_intercept_x"]
        reacq_target = reacquisition["predicted_intercept_x"]
        revision = abs(stale_target - reacq_target)
        events.append({
            **record,
            "last_non_direct_predicted_intercept_x": stale_target,
            "reacquisition_predicted_intercept_x": reacq_target,
            "last_non_direct_horizon_frames": last_stale["predicted_paddle_intercept_horizon_frames"],
            "reacquisition_horizon_frames": reacquisition["predicted_paddle_intercept_horizon_frames"],
            "intercept_target_revision_px": revision,
            "over_8px": revision > THRESHOLD_PX,
        })
    return events, excluded, issues


def classify(events: list[dict[str, Any]], invalid: bool = False) -> tuple[str, float | None, int]:
    n = len(events)
    coverage = len({event["episode_seed"] for event in events})
    fraction = sum(bool(event["over_8px"]) for event in events) / n if n else None
    if invalid or n < 6 or coverage < 2:
        return "INCONCLUSIVE", fraction, coverage
    if fraction >= 2 / 3:
        return "PROMOTED", fraction, coverage
    if fraction <= 1 / 6:
        return "REJECTED", fraction, coverage
    return "INCONCLUSIVE", fraction, coverage


def _read_groups(trace: Path) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], list[str]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    issues: list[str] = []
    for line_no, line in enumerate(trace.open(), 1):
        row = json.loads(line)
        if row.get("arm") != "candidate" or row.get("max_missing_frames") != 12:
            continue
        seed = row.get("episode_seed")
        if seed not in SEEDS:
            continue
        if not isinstance(row.get("episode_ordinal"), int):
            issues.append(f"line {line_no}: invalid episode_ordinal")
            continue
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            issues.append(f"line {line_no}: missing required fields {', '.join(missing)}")
        groups[(seed, row["episode_ordinal"])].append(row)
    return groups, issues


def run(trace: Path, results: Path, out: Path) -> dict[str, Any]:
    trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    results_hash = hashlib.sha256(results.read_bytes()).hexdigest()
    if trace_hash != TRACE_SHA256 or results_hash != RESULTS_SHA256:
        raise ValueError(f"frozen input hash mismatch: trace={trace_hash}, results={results_hash}")
    analyzer_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    groups, issues = _read_groups(trace)
    expected = {(seed, ordinal) for seed, ordinal in groups}
    if {seed for seed, _ in expected} != set(SEEDS):
        issues.append("candidate max_missing_frames=12 trace does not contain all three frozen seeds")
    events: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for seed, ordinal in sorted(groups):
        episode_events, episode_excluded, episode_issues = analyze_episode(groups[(seed, ordinal)])
        events.extend(episode_events)
        excluded.extend(episode_excluded)
        issues.extend(f"seed {seed} episode {ordinal}: {issue}" for issue in episode_issues)
    classification, fraction, coverage = classify(events, bool(issues))
    revisions = sorted(event["intercept_target_revision_px"] for event in events)
    result: dict[str, Any] = {
        "issue": 33,
        "classification": classification,
        "hypothesis": "At least two-thirds of eligible direct reacquisitions after 5-12 non-direct observations revise the controller intercept target by strictly more than 8 px.",
        "provenance": {
            "frozen_base_branch": "research/issue-14-phase1",
            "frozen_base_sha": "9e75303f7df87c74f84613badb54e4129816b9ae",
            "source_lineage": {
                "prior_diagnostic_original_base_sha": "f8311b55a05f4f901d63e70a89a49d6dc39a02c8",
                "issue_31_source_in_frozen_base_sha": "9e75303f7df87c74f84613badb54e4129816b9ae",
                "issue_33_analyzer_source": "scripts/analysis/analyze_intercept_revision.py",
                "issue_33_source_rebase": "none; created directly from the frozen base",
            },
            "trace_path": str(trace), "trace_sha256": trace_hash,
            "results_path": str(results), "results_sha256": results_hash,
            "analyzer_sha256": analyzer_hash,
            "command": "python -m scripts.analysis.analyze_intercept_revision --output-dir outputs/issue-33-intercept-revision",
            "episode_seeds": list(SEEDS), "arm": "candidate", "max_missing_frames": 12,
            "new_episodes": 0, "new_ale_native_frames": 0, "training": 0,
        },
        "primary_metric": {
            "eligible_reacquisitions": len(events),
            "revisions_over_8px": sum(event["over_8px"] for event in events),
            "high_intercept_revision_fraction": fraction,
            "seed_coverage": coverage,
            "sample_floor_met": len(events) >= 6 and coverage >= 2,
        },
        "secondary_metrics": {
            "eligible_by_seed": dict(sorted(Counter(str(e["episode_seed"]) for e in events).items())),
            "over_8px_by_seed": dict(sorted(Counter(str(e["episode_seed"]) for e in events if e["over_8px"]).items())),
            "gap_length_counts": dict(sorted(Counter(str(e["gap_length"]) for e in events).items())),
            "revision_px_min_median_max": {
                "min": revisions[0] if revisions else None,
                "median": revisions[len(revisions) // 2] if revisions else None,
                "max": revisions[-1] if revisions else None,
            },
            "excluded_or_ambiguous_event_count": len(excluded),
            "integrity_issue_count": len(issues),
        },
        "events": events,
        "excluded_events": excluded,
        "analysis_issues": issues,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    event_fields = [
        "episode_seed", "episode_ordinal", "preceding_direct_frame", "gap_start_frame",
        "last_non_direct_frame", "gap_length", "reacquisition_frame",
        "last_non_direct_predicted_intercept_x", "reacquisition_predicted_intercept_x",
        "last_non_direct_horizon_frames", "reacquisition_horizon_frames",
        "intercept_target_revision_px", "over_8px",
    ]
    with (out / "events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(events)
    pct = "n/a" if fraction is None else f"{fraction:.3f} ({fraction:.1%})"
    report = f'''# Issue #33: Intercept-target revision after dropouts

## Hypothesis Result

**{classification}** — high intercept-target revision fraction {pct}; frozen criteria applied.

## Baseline

Issue #29 candidate arm, seeds 101/202/303, `max_missing_frames=12`; Issue #29 remains INCONCLUSIVE (paired survival deltas −888, −575, +607 frames). Issue #31 remains INCONCLUSIVE (15/23 vertical forecast errors >8 px). No controller or champion was changed.

## Experiment

Offline diagnostic using the frozen Issue #29 trace. Each event requires a direct row, 5–12 consecutive non-direct rows, then a direct reacquisition. The measured change is the absolute difference between `predicted_intercept_x` on the last non-direct row and on the reacquisition row. Both horizons must be finite in [0, 30] frames. The preceding direct row must have finite y and vy, descending vy, and a finite horizon in [0, 30].

The no-bounce constant-velocity sanity test advances an exact straight trajectory by one observation step; the helper's intercept target changes by approximately zero. Threshold is strictly greater than 8 px. Zero new episodes, ALE-native frames, and training.

## Primary Result

Eligible reacquisitions: {len(events)}; revisions >8 px: {result['primary_metric']['revisions_over_8px']}; fraction: {pct}; seed coverage: {coverage}; sample floor met: {result['primary_metric']['sample_floor_met']}.

## Secondary Metrics

Eligible by seed: {result['secondary_metrics']['eligible_by_seed']}; >8 px by seed: {result['secondary_metrics']['over_8px_by_seed']}; gap lengths: {result['secondary_metrics']['gap_length_counts']}; revision min/median/max (px): {result['secondary_metrics']['revision_px_min_median_max']}; excluded/ambiguous events: {len(excluded)}; integrity issues: {len(issues)}. Event-level values and exclusions are in `events.csv` and `result.json`.

## Failure Analysis

This measures controller target changes in the frozen artifact set. It does not establish that a target revision caused a life loss. The stale target already includes extrapolation and side-wall reflection; exclusions and episode integrity issues are retained in `result.json`.

## Reproducibility

Frozen base: `research/issue-14-phase1` at `{result['provenance']['frozen_base_sha']}`. Issue #31's recorded original diagnostic base was `{result['provenance']['source_lineage']['prior_diagnostic_original_base_sha']}`; its source is present in the frozen base. Issue #33 source was created directly from the frozen base; no source rebase was performed. Trace SHA-256: `{trace_hash}`; Issue #29 results SHA-256: `{results_hash}`. Analyzer SHA-256: `{analyzer_hash}`. Exact command: `{result['provenance']['command']}`. Candidate seeds: 101, 202, 303; max_missing_frames=12.

## Tests

Pure unit tests cover dropout boundaries, eligibility, strict threshold, classification/sample floor, and constant-velocity no-bounce target invariance. Python compilation and `git diff --check` were run. No ALE imports or environment launches.

## Remaining Uncertainty

The eligible event count and seed coverage limit inference to this fixed artifact set. No causal effect on life loss is established, and this diagnostic does not change the inconclusive status of Issues #29 or #31.

## Recommended Next Decision

{('Prioritize reacquisition-induced horizontal target changes for a later controlled Phase 1 experiment; this result does not promote a controller or satisfy Issue #14.' if classification == 'PROMOTED' else 'Do not prioritize on this evidence; preserve the finding as ' + classification + '.')}
'''
    (out / "report.md").write_text(report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("outputs/issue-29-track-retention/trace.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("outputs/issue-29-track-retention/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/issue-33-intercept-revision"))
    args = parser.parse_args()
    result = run(args.trace, args.results, args.output_dir)
    print(json.dumps({"classification": result["classification"], "primary_metric": result["primary_metric"]}, indent=2))


if __name__ == "__main__":
    main()
