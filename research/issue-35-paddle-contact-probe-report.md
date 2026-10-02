# Issue #35 Paddle Contact Probe Report

## Hypothesis Result

Classification: **INCONCLUSIVE**.
The specified RGB predictive controller was probed at requested impact offsets -12, 0, and +12 px.

## Baseline

Contract v3 and the default predictive vision controller config remain unchanged. Baseline #26 recorded 0/15 canonical clear detections; raw score comparison is out of scope.

## Experiment

Scheduled design: exactly nine episodes in fixed seed-major order (101, 202, 303; offsets -12, 0, +12). Completed 9/9 arms. Each episode used raw RGB action selection and stopped at first confirmed bounce plus 8 native frames, termination/life loss, or 5,000 native frames (or the aggregate budget).
Contract hash `d97b7fcb5758cba495857b3972a119ed3eb7db74fb0145d151e9d82725c5a786`; controller config hash `25ddff6756747a6539ddd6a1eb570e7ef4c274f3e700ae011e68c87cae37c724`.
ROM SHA-256 `376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5`; ALE-Py `0.12.0`; Gymnasium `1.3.0`; Python `3.12.14`; NumPy `2.3.5`; OpenCV `5.0.0`.

## Primary Result

`positive_response_seed_count = 0` / 3. Classification: **INCONCLUSIVE**.

Seed-level fits use outgoing vx regressed on observed impact offset:

| Seed | Valid triplet | Strict offset order | Span (px) | OLS slope | Residuals | Positive |
|---:|:---:|:---:|---:|---:|---|:---:|
| 101 | False | False | n/a | n/a | n/a | False |
| 202 | False | False | n/a | n/a | n/a | False |
| 303 | False | False | n/a | n/a | n/a | False |

## Secondary Metrics

| Seed | Offset | Observed impact offset | Outgoing vx | Bounce | Direct coverage | Frames | Stop | Clears | Requested counts | ALE input counts | Exclusion |
|---:|---:|---:|---:|:---:|---:|---:|---|---:|---|---|---|
| 101 | -12 | n/a | n/a | no | 0.863 | 285 | life_loss | False | `{'NOOP': 155, 'LEFT': 69, 'RIGHT': 61}` | `{'FIRE': 2, 'LEFT': 69, 'NOOP': 153, 'RIGHT': 61}` | bounce_not_confirmed |
| 101 | 0 | 2.500 | n/a | yes | 0.961 | 75 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 55, 'LEFT': 20}` | `{'FIRE': 2, 'LEFT': 20, 'NOOP': 53}` | wall_or_brick_collision |
| 101 | 12 | 2.500 | n/a | yes | 0.961 | 75 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 55, 'LEFT': 20}` | `{'FIRE': 2, 'LEFT': 20, 'NOOP': 53}` | wall_or_brick_collision |
| 202 | -12 | n/a | n/a | no | 0.905 | 472 | life_loss | False | `{'NOOP': 268, 'LEFT': 102, 'RIGHT': 102}` | `{'FIRE': 5, 'LEFT': 102, 'NOOP': 263, 'RIGHT': 102}` | bounce_not_confirmed |
| 202 | 0 | 7.500 | 1.000 | yes | 0.924 | 78 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 28, 'LEFT': 28, 'RIGHT': 22}` | `{'FIRE': 5, 'LEFT': 28, 'NOOP': 23, 'RIGHT': 22}` | none |
| 202 | 12 | n/a | n/a | no | 0.640 | 100 | life_loss | False | `{'NOOP': 48, 'LEFT': 31, 'RIGHT': 21}` | `{'FIRE': 5, 'LEFT': 31, 'NOOP': 43, 'RIGHT': 21}` | bounce_not_confirmed |
| 303 | -12 | -2.375 | -2.000 | yes | 0.936 | 77 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 69, 'RIGHT': 8}` | `{'FIRE': 3, 'RIGHT': 8, 'NOOP': 66}` | none |
| 303 | 0 | -2.375 | -2.000 | yes | 0.936 | 77 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 69, 'RIGHT': 8}` | `{'FIRE': 3, 'RIGHT': 8, 'NOOP': 66}` | none |
| 303 | 12 | n/a | n/a | no | 0.653 | 98 | life_loss | False | `{'NOOP': 51, 'RIGHT': 26, 'LEFT': 21}` | `{'FIRE': 3, 'RIGHT': 26, 'NOOP': 48, 'LEFT': 21}` | bounce_not_confirmed |

Action requests and post-FIRE ALE inputs are recorded separately in JSON/CSV traces; sticky-resolved physical actions are unknown. Bounce candidates require the projected intercept to overlap the directly detected paddle span expanded by the configured 2 px ball radius; the ascending confirmation must be directly detected within ±5 px of paddle top. Wall/brick collision exclusions use observed velocity reversals and side-wall proximity.

## Delta vs Baseline

No clear-rate or raw-score delta is claimed because this probe measures first-bounce steering and uses only nine short episodes.

## Failure Analysis

Valid manipulation/measurement-floor seed triplets: 0/3. Per-run exclusions are listed above.

## Reproducibility

Command: `python -m scripts.evaluation.run_issue35_paddle_contact_probe --config configs/eval/issue35_paddle_contact_probe_v1.json --output-dir outputs/issue-35-paddle-contact-probe`.
Source branch `codex/issue-35-paddle-contact-probe`, commit `5ae213f32552fd7cf4f48ba1084750271a2d45f3` (clean tree: True); source SHA-256 `58aac661ad6e3f0d79efe53ad1b5454e691bca479e03a2b29f6b7ece6d6c66fc`. Exact completed order: `[{'seed': 101, 'offset_px': -12}, {'seed': 101, 'offset_px': 0}, {'seed': 101, 'offset_px': 12}, {'seed': 202, 'offset_px': -12}, {'seed': 202, 'offset_px': 0}, {'seed': 202, 'offset_px': 12}, {'seed': 303, 'offset_px': -12}, {'seed': 303, 'offset_px': 0}, {'seed': 303, 'offset_px': 12}]`. Probe native frames: 1337; predeclared ALE regression-test frame reserve: 350; combined conservative native-frame total: 1687; probe wall time: 3.961s.
Combined wall accounting: 4.0s reserved for pre-run tests/compilation, 3.961s measured through environment setup and probe, and at most 60.0s reserved for artifact/report generation (600s hard ceiling).
Measured end-to-end command time including the validation reserve and first artifact/report generation pass: 8.013s. The final artifact rewrite completed within the reserved 60s. Frozen experiment base: `362b5a271fb94af4ce9ccb4c74489cde2c36bd85`.

Artifacts are preserved under ignored `outputs/issue-35-paddle-contact-probe/`: `results.json`, `episodes.csv`, `report.md`, and nine per-episode RGB/controller traces under `traces/`.
Review copies are included in [the research artifact bundle](issue-35-paddle-contact-probe-artifacts/): [results.json](issue-35-paddle-contact-probe-artifacts/results.json), [episodes.csv](issue-35-paddle-contact-probe-artifacts/episodes.csv), [generated report](issue-35-paddle-contact-probe-artifacts/report.md), and the nine trace CSVs under `traces/`.

## Tests

Focused command: `python -m unittest tests.test_issue35_paddle_contact_probe -v` (12 tests, passed). Regression command: `python -m unittest tests.test_vision_controller.PredictiveVisionPerceptionTests tests.test_vision_controller.PredictiveControlMathTests tests.test_vision_controller.VisionEvaluationSemanticsTests tests.test_evaluation_contract.Day15ContractTests.test_contract_v3_requires_explicit_precision_opt_in tests.test_completion.ALETransitionFrameCounterTests tests.test_completion.BreakoutCompletionDetectorTests tests.test_completion.CompletionSummaryTests tests.test_evaluation_completion_pipeline.EvaluationCompletionPipelineTests -v`. The RGB controller and completion tests passed; the selected Contract v3 and evaluation-completion modules could not load because their imports require PyTorch, which is not installed. Three completion regression fixtures reset ALE; a conservative 350-native-frame reserve is included in the aggregate budget. Python compilation and `git diff --check` passed.

## Remaining Uncertainty

Sticky action resolution is hidden; observations estimate ball motion after any sticky substitution. Three seeds and one bounce per run provide limited evidence. A collision before three ascending velocity estimates excludes that run.

## Recommended Next Decision

Follow the frozen decision rule only: INCONCLUSIVE. PROMOTED permits considering one later controlled outgoing-angle experiment; it does not establish clear improvement or satisfy Issue #14.
