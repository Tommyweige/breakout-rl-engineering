# Issue #31: Reacquisition drift diagnostic

## Hypothesis Result

**INCONCLUSIVE** — high vertical forecast error fraction 0.652 (65.2%); frozen classification criteria applied.

## Baseline

Issue #29 candidate arm, seeds 101/202/303, max_missing_frames=12; Issue #29 remains INCONCLUSIVE. No champion or controller was changed.

## Experiment

Offline only; zero new ALE frames, episodes, or training. At each eligible direct reacquisition after a 5–12 frame dropout, forecast from the preceding direct y/vy over the exact frame delta. Threshold: strictly greater than 8 px.

## Primary Result

Eligible events: 23; over 8 px: 15; fraction: 0.652 (65.2%); seed coverage: 3.

## Secondary Metrics

Eligible by seed: {'101': 4, '202': 10, '303': 9}; over 8 px by seed: {'101': 3, '202': 4, '303': 8}; gap counts: {'8': 1, '12': 5, '11': 5, '7': 3, '5': 9}; error min/median/max: {'min': 2.5, 'median': 18.0, 'max': 114.0}. Excluded events: 631; invalid/ambiguous conditions: 0. See events.csv and result.json for event-level details.

## Failure Analysis

No causal inference is made from forecast error; sticky-resolved physical actions are unknown. Excluded events and data-integrity issues are retained in result.json.

## Reproducibility

Trace SHA-256: `36f6c52d0814e873431eae46cd48061b367164725158513417e35c7eba03d8fe`; #29 results SHA-256: `638fb46ce23ffd425ea766c85d71a71db058717f3a5adfab933f34d2b6e87e50`. Command: `python -m scripts.analysis.analyze_reacquisition_drift --output-dir outputs/issue-31-reacquisition-drift`. Analyzer SHA-256: `526b7c326b259bc4c3e01710bf6f6f053d7ee6122f79954352fa20797d025a39`. Base SHA: `f8311b55a05f4f901d63e70a89a49d6dc39a02c8`. Source: `scripts/analysis/analyze_reacquisition_drift.py`.

## Tests

Pure Python unit tests; Python compilation and git diff --check. No ALE imports or environment launches.

## Remaining Uncertainty

This diagnostic measures a stale linear vertical forecast on an existing artifact set; it does not show dropout caused a life loss or explain Issue #29 survival differences.

## Recommended Next Decision

Do not prioritize on this evidence; retain the diagnostic as INCONCLUSIVE.
