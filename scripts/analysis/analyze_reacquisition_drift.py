"""Offline vertical reacquisition-error analysis for the frozen Issue #29 trace."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

TRACE_SHA256 = "36f6c52d0814e873431eae46cd48061b367164725158513417e35c7eba03d8fe"
RESULTS_SHA256 = "638fb46ce23ffd425ea766c85d71a71db058717f3a5adfab933f34d2b6e87e50"
SEEDS = (101, 202, 303)
THRESHOLD_PX = 8.0


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def analyze_episode(rows: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    rows = sorted(rows, key=lambda r: r["observation_ale_frame"])
    issues: list[str] = []
    frames = [r.get("observation_ale_frame") for r in rows]
    if len(set(frames)) != len(frames):
        issues.append("duplicate observation_ale_frame")
    if any(not isinstance(f, int) for f in frames) or any(b != a + 1 for a, b in zip(frames, frames[1:])):
        issues.append("missing or invalid observation frame")
    events: list[dict] = []
    excluded: list[dict] = []
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
        prev = rows[start - 1] if start else None
        reacq = rows[end] if end < len(rows) else None
        seed = (prev or rows[start]).get("episode_seed")
        rec = {"episode_seed": seed, "gap_start_frame": frames[start], "gap_end_frame": frames[end - 1],
               "gap_length": gap, "reacquisition_frame": frames[end] if reacq else None}
        reason = None
        if prev is None or not prev.get("ball_directly_detected"):
            reason = "no immediately preceding direct detection"
        elif reacq is None or not reacq.get("ball_directly_detected"):
            reason = "no direct reacquisition row"
        elif gap < 5 or gap > 12:
            reason = "gap length outside 5-12"
        elif not finite(prev.get("ball_y")) or not finite(prev.get("ball_vy")):
            reason = "invalid preceding direct y/vy"
            issues.append(reason)
        elif prev["ball_vy"] <= 0:
            reason = "preceding vertical velocity is not descending"
        elif not finite(prev.get("predicted_paddle_intercept_horizon_frames")):
            reason = "invalid predicted paddle-intercept horizon"
            issues.append(reason)
        elif not 0 <= prev["predicted_paddle_intercept_horizon_frames"] <= 30:
            reason = "predicted paddle-intercept horizon outside 0-30"
        elif not finite(reacq.get("ball_y")):
            reason = "invalid reacquired direct y"
            issues.append(reason)
        if reason:
            excluded.append({**rec, "reason": reason})
        else:
            delta = reacq["observation_ale_frame"] - prev["observation_ale_frame"]
            forecast_y = prev["ball_y"] + prev["ball_vy"] * delta
            error = abs(forecast_y - reacq["ball_y"])
            events.append({**rec, "frame_delta": delta, "forecast_y": forecast_y,
                           "reacquired_y": reacq["ball_y"], "vertical_error_px": error,
                           "over_8px": error > THRESHOLD_PX})
    return events, excluded, issues


def classify(events: list[dict], seeds: list[int], invalid: bool = False) -> tuple[str, float | None]:
    n = len(events)
    coverage = len({e["episode_seed"] for e in events})
    fraction = sum(bool(e["over_8px"]) for e in events) / n if n else None
    if invalid or n < 6 or coverage < 2:
        return "INCONCLUSIVE", fraction
    if fraction >= 2 / 3:
        return "PROMOTED", fraction
    if fraction <= 1 / 6:
        return "REJECTED", fraction
    return "INCONCLUSIVE", fraction


def run(trace: Path, results: Path, out: Path) -> dict:
    trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    results_hash = hashlib.sha256(results.read_bytes()).hexdigest()
    if trace_hash != TRACE_SHA256 or results_hash != RESULTS_SHA256:
        raise ValueError(f"frozen input hash mismatch: trace={trace_hash}, results={results_hash}")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    groups: dict[int, list[dict]] = {s: [] for s in SEEDS}
    for line_no, line in enumerate(trace.open(), 1):
        row = json.loads(line)
        if row.get("arm") == "candidate" and row.get("max_missing_frames") == 12 and row.get("episode_seed") in groups:
            groups[row["episode_seed"]].append(row)
    events, excluded, issues = [], [], []
    for seed in SEEDS:
        es, xs, problems = analyze_episode(groups[seed])
        events.extend(es); excluded.extend(xs); issues.extend(f"seed {seed}: {p}" for p in problems)
    classification, fraction = classify(events, list(SEEDS), bool(issues))
    errors = sorted(e["vertical_error_px"] for e in events)
    summary = {
      "issue": 31, "classification": classification,
      "hypothesis": "At least two-thirds of eligible direct reacquisitions have vertical forecast error >8 px.",
      "provenance": {"trace_path": str(trace), "trace_sha256": trace_hash, "results_path": str(results),
                     "results_sha256": results_hash, "analyzer_sha256": source_hash, "base_sha": "f8311b55a05f4f901d63e70a89a49d6dc39a02c8",
                     "command": "python -m scripts.analysis.analyze_reacquisition_drift --output-dir outputs/issue-31-reacquisition-drift",
                     "episode_seeds": list(SEEDS), "arm": "candidate", "max_missing_frames": 12},
      "primary_metric": {"eligible_reacquisitions": len(events), "reacquisitions_over_8px": sum(e["over_8px"] for e in events),
                         "high_vertical_error_fraction": fraction, "seed_coverage": len({e["episode_seed"] for e in events})},
      "secondary_metrics": {"eligible_by_seed": dict(Counter(str(e['episode_seed']) for e in events)),
                            "over_8px_by_seed": dict(Counter(str(e['episode_seed']) for e in events if e['over_8px'])),
                            "gap_length_counts": dict(Counter(str(e['gap_length']) for e in events)),
                            "error_quantiles_px": {"min": errors[0] if errors else None, "median": errors[len(errors)//2] if errors else None,
                                                    "max": errors[-1] if errors else None},
                            "excluded_event_count": len(excluded), "invalid_or_ambiguous_count": len(issues)},
      "events": events, "excluded_events": excluded, "analysis_issues": issues,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = ["episode_seed", "gap_start_frame", "gap_end_frame", "gap_length", "reacquisition_frame", "frame_delta", "forecast_y", "reacquired_y", "vertical_error_px", "over_8px"]
    with (out / "events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(events)
    pct = "n/a" if fraction is None else f"{fraction:.3f} ({fraction:.1%})"
    report = f'''# Issue #31: Reacquisition drift diagnostic\n\n## Hypothesis Result\n\n**{classification}** — high vertical forecast error fraction {pct}; frozen classification criteria applied.\n\n## Baseline\n\nIssue #29 candidate arm, seeds 101/202/303, max_missing_frames=12; Issue #29 remains INCONCLUSIVE. No champion or controller was changed.\n\n## Experiment\n\nOffline only; zero new ALE frames, episodes, or training. At each eligible direct reacquisition after a 5–12 frame dropout, forecast from the preceding direct y/vy over the exact frame delta. Threshold: strictly greater than 8 px.\n\n## Primary Result\n\nEligible events: {len(events)}; over 8 px: {summary['primary_metric']['reacquisitions_over_8px']}; fraction: {pct}; seed coverage: {summary['primary_metric']['seed_coverage']}.\n\n## Secondary Metrics\n\nEligible by seed: {summary['secondary_metrics']['eligible_by_seed']}; over 8 px by seed: {summary['secondary_metrics']['over_8px_by_seed']}; gap counts: {summary['secondary_metrics']['gap_length_counts']}; error min/median/max: {summary['secondary_metrics']['error_quantiles_px']}. Excluded events: {len(excluded)}; invalid/ambiguous conditions: {len(issues)}. See events.csv and result.json for event-level details.\n\n## Failure Analysis\n\nNo causal inference is made from forecast error; sticky-resolved physical actions are unknown. Excluded events and data-integrity issues are retained in result.json.\n\n## Reproducibility\n\nTrace SHA-256: `{trace_hash}`; #29 results SHA-256: `{results_hash}`. Command: `python -m scripts.analysis.analyze_reacquisition_drift --output-dir outputs/issue-31-reacquisition-drift`. Analyzer SHA-256: `{source_hash}`. Base SHA: `f8311b55a05f4f901d63e70a89a49d6dc39a02c8`. Source: `scripts/analysis/analyze_reacquisition_drift.py`.\n\n## Tests\n\nPure Python unit tests; Python compilation and git diff --check. No ALE imports or environment launches.\n\n## Remaining Uncertainty\n\nThis diagnostic measures a stale linear vertical forecast on an existing artifact set; it does not show dropout caused a life loss or explain Issue #29 survival differences.\n\n## Recommended Next Decision\n\n{('Prioritize stale vertical extrapolation error for a future Phase 1 experiment.' if classification == 'PROMOTED' else 'Do not prioritize on this evidence; retain the diagnostic as ' + classification + '.') }\n'''
    (out / "report.md").write_text(report)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=Path("outputs/issue-29-track-retention/trace.jsonl"))
    p.add_argument("--results", type=Path, default=Path("outputs/issue-29-track-retention/results.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/issue-31-reacquisition-drift"))
    a = p.parse_args()
    result = run(a.trace, a.results, a.output_dir)
    print(json.dumps({"classification": result["classification"], "primary_metric": result["primary_metric"]}, indent=2))

if __name__ == "__main__":
    main()
