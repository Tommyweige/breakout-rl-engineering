# Issue #33: Intercept-target revision after dropouts

## Hypothesis Result

**INCONCLUSIVE** — high intercept-target revision fraction 0.000 (0.0%); frozen criteria applied.

## Baseline

Issue #29 candidate arm, seeds 101/202/303, `max_missing_frames=12`; Issue #29 remains INCONCLUSIVE (paired survival deltas −888, −575, +607 frames). Issue #31 remains INCONCLUSIVE (15/23 vertical forecast errors >8 px). No controller or champion was changed.

## Experiment

Offline diagnostic using the frozen Issue #29 trace. Each event requires a direct row, 5–12 consecutive non-direct rows, then a direct reacquisition. The measured change is the absolute difference between `predicted_intercept_x` on the last non-direct row and on the reacquisition row. Both horizons must be finite in [0, 30] frames. The preceding direct row must have finite y and vy, descending vy, and a finite horizon in [0, 30].

The no-bounce constant-velocity sanity test advances an exact straight trajectory by one observation step; the helper's intercept target changes by approximately zero. Threshold is strictly greater than 8 px. Zero new episodes, ALE-native frames, and training.

## Primary Result

Eligible reacquisitions: 5; revisions >8 px: 0; fraction: 0.000 (0.0%); seed coverage: 1; sample floor met: False.

## Secondary Metrics

Eligible by seed: {'202': 5}; >8 px by seed: {}; gap lengths: {'5': 5}; revision min/median/max (px): {'min': 6.656976744186039, 'median': 6.656976744186039, 'max': 6.656976744186068}; excluded/ambiguous events: 649; integrity issues: 0. Event-level values and exclusions are in `events.csv` and `result.json`.

## Failure Analysis

This measures controller target changes in the frozen artifact set. It does not establish that a target revision caused a life loss. The stale target already includes extrapolation and side-wall reflection; exclusions and episode integrity issues are retained in `result.json`.

## Reproducibility

Frozen source and PR base: `research/issue-14-phase1` at `9e75303f7df87c74f84613badb54e4129816b9ae`. Issue #33 analyzer source was authored directly on this frozen base; the run and submitted source use the same analyzer bytes, and no Issue #33 rebase occurred. For upstream Issue #31 provenance, its original analyzer source commit was `6b29ef00c5e1deddf1014399e047beab22ff2de4` on original base `f8311b55a05f4f901d63e70a89a49d6dc39a02c8`; its rebased PR source commit was `ad138f503b9df04cbf37e3cbd8c380ea5a81756a` on base `1f9c31f39c786f8db801f1b2d086e5cabdc2b73d`, with the same analyzer SHA-256 `526b7c326b259bc4c3e01710bf6f6f053d7ee6122f79954352fa20797d025a39`. That prior analysis was not rerun after rebase. Trace SHA-256: `36f6c52d0814e873431eae46cd48061b367164725158513417e35c7eba03d8fe`; Issue #29 results SHA-256: `638fb46ce23ffd425ea766c85d71a71db058717f3a5adfab933f34d2b6e87e50`. Issue #33 analyzer SHA-256: `cae2a28a96efd11712292c8e59aa067364db0dadadfab3d35f0bda2658069732`. Exact command: `python -m scripts.analysis.analyze_intercept_revision --output-dir outputs/issue-33-intercept-revision` (0.53 seconds wall time). Candidate seeds: 101, 202, 303; max_missing_frames=12.

## Tests

Pure tests: `python -m unittest tests.test_intercept_revision` (9 passed). Compilation: `python -m py_compile scripts/analysis/analyze_intercept_revision.py tests/test_intercept_revision.py`. `git diff --check` passed. No ALE imports or environment launches.

## Remaining Uncertainty

The eligible event count and seed coverage limit inference to this fixed artifact set. No causal effect on life loss is established, and this diagnostic does not change the inconclusive status of Issues #29 or #31.

## Recommended Next Decision

Do not prioritize on this evidence; preserve the finding as INCONCLUSIVE.
