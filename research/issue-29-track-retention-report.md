# Issue 29: paired track-retention comparison

## Hypothesis Result

The frozen hypothesis was not established. Increasing `max_missing_frames` from 4 to 12 added at least 500 ALE-native game-over frames on one of three paired seeds, below the required two. No pair lost more than 1,000 frames. The result is **INCONCLUSIVE** under the predeclared criteria.

## Baseline

The baseline arm used the reviewed RGB predictive controller with Contract v3 and `max_missing_frames=4`, run in this experiment on seeds 101, 202, and 303. The candidate arm reused the same protocol and changed only that setting to 12. Historical context from the frozen issue: the 15-seed baseline had 0/15 canonical clear detections; the #27 diagnostic subset associated dropouts with 14/15 life-loss events but did not establish causality.

## Experiment

Ran the exact six episodes, baseline first and candidate second, with one episode for each seed per arm. All six ended at game over; none cleared or hit the 10,000-step cap. Contract v3 runtime semantics, RGB-only control, the canonical clear detector, and one ALE-native frame per agent action were verified. Total counted ALE-native frames were 45,768; wall time was 126.34 seconds.

| `max_missing_frames` | Seed | Game-over frame | Raw score | Life-loss frames | Direct-ball detection | Longest dropout | Stop |
|---:|---:|---:|---:|---|---:|---:|---|
| 4 | 101 | 8,657 | 337 | 2,842; 4,642; 5,914; 7,822; 8,657 | 86.1% | 191 | game over |
| 4 | 202 | 7,164 | 201 | 3,442; 5,111; 5,208; 6,773; 7,164 | 89.5% | 34 | game over |
| 4 | 303 | 7,491 | 341 | 3,226; 4,044; 4,404; 6,963; 7,491 | 83.3% | 181 | game over |
| 12 | 101 | 7,769 | 284 | 2,842; 4,642; 5,592; 6,400; 7,769 | 88.0% | 181 | game over |
| 12 | 202 | 6,589 | 302 | 2,058; 2,613; 4,562; 4,996; 6,589 | 85.4% | 113 | game over |
| 12 | 303 | 8,098 | 290 | 3,226; 4,044; 6,890; 7,334; 8,098 | 87.6% | 63 | game over |

## Primary Result

`paired_survival_gain_seed_count = 1 / 3`, with the frozen 500-frame threshold applied only to completed game-over pairs.

## Secondary Metrics

| Seed | Candidate minus baseline game-over frames | Baseline requested / ALE-input action counts (NOOP, LEFT, RIGHT, FIRE) | Candidate requested / ALE-input action counts (NOOP, LEFT, RIGHT, FIRE) |
|---:|---:|---|---|
| 101 | -888 | 5,612/5,602; 1,527/1,527; 1,518/1,518; 0/10 | 5,108/5,098; 1,353/1,353; 1,308/1,308; 0/10 |
| 202 | -575 | 4,502/4,489; 1,345/1,345; 1,317/1,317; 0/13 | 4,112/4,099; 1,301/1,301; 1,176/1,176; 0/13 |
| 303 | +607 | 4,994/4,983; 1,261/1,261; 1,236/1,236; 0/11 | 5,270/5,259; 1,466/1,466; 1,362/1,362; 0/11 |

The counts above list requested/ALE-input actions, in the stated action order. Direct-ball detection, dropout length, score, and life-loss frames are shown per episode above. ALE-input FIRE counts reflect the environment-owned serve wrapper; requested FIRE count stayed zero. The sticky-resolved physical action remains unknown.

## Delta vs Baseline

The candidate arm reduced survival by 888 frames on seed 101 and 575 frames on seed 202, while adding 607 frames on seed 303. Candidate scores were lower on seeds 101 and 303 and higher on 202; score is diagnostic only and does not affect the decision. The primary criterion counts only seed 303.

## Failure Analysis

The predeclared promotion threshold was not met because only one pair gained at least 500 frames. The rejection threshold was not met because the primary count was not zero. No regression exceeded 1,000 frames, but that does not change the inconclusive classification. There was no canonical clear; survival remains a proxy and does not satisfy Issue 14 Phase 1.

## Reproducibility

- Issue: #29; experiment branch: `codex/issue-29-track-retention`; base: `f8311b55a05f4f901d63e70a89a49d6dc39a02c8`.
- Tested source commit: `7f94c43006161bd87d3eb70673779c21cd505cf6`; working tree was clean at run start.
- Exact command: `timeout --signal=TERM --kill-after=5s 600s python -m scripts.analysis.compare_vision_dropout_retention --config configs/eval/breakout_vision_controller_v1.json --episode-seeds 101,202,303 --baseline-max-missing-frames 4 --candidate-max-missing-frames 12 --max-steps-per-episode 10000 --output-dir outputs/issue-29-track-retention`.
- Config SHA-256: `25ddff6756747a6539ddd6a1eb570e7ef4c274f3e700ae011e68c87cae37c724`; Contract v3 SHA-256: `d97b7fcb5758cba495857b3972a119ed3eb7db74fb0145d151e9d82725c5a786`.
- Run source digest: `750bef23cd6b5ff87938a860caf150ea2f77c268d0e35bc931dd50cf12aa8ac2`.
- Runtime: Python 3.12.14, NumPy 2.3.5, Gymnasium 1.3.0, ALE-Py 0.12.0, OpenCV 5.0.0.93. Canonical completion detector support was true; ROM SHA-256: `376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5`.
- Runtime verification: Breakout v3, `frame_skip=1`, `frame_stack=4`, sticky-action probability 0.25, environment-owned FIRE reset, no terminal-on-life-loss, raw rewards, one native frame per agent action, RGB-only controller observations, exact frozen arm/seed pairs, and canonical completion checks.
- Full ignored run artifacts and per-frame traces are preserved under `outputs/issue-29-track-retention/`: `results.json`, `episodes.csv`, `trace.jsonl`, `report.md`, and `resume.json`. The trace has 45,768 action rows; the resume record confirms all six pairs complete and no further episode is scheduled.

## Tests

Passed: Python compilation, `git diff --check`, runner `--help`, and 35 selected new, vision-controller, and completion unit/regression tests. The selected tests did not launch ALE episodes. No separate ALE smoke episode was run. The broader evaluation-completion pipeline tests were not verified because Torch is not installed in the active runtime; they failed at import before test execution.

## Remaining Uncertainty

There are only three paired seeds. Sticky-action-resolved physical actions are not exposed, and action trajectories diverged between arms after decisions changed. The result does not show that tracking dropouts are harmless or establish a causal mechanism beyond this parameter intervention.

## Recommended Next Decision

Keep the result **INCONCLUSIVE**. Do not promote the 12-frame setting, reject the broader dropout hypothesis, extend the compute budget, or enter Phase 2 from this result. Any further experiment needs a separately frozen design.
