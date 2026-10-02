# Issue #37 Reachable Paddle Contact Probe Report

## Hypothesis Result

Classification: **INCONCLUSIVE**.
The specified RGB predictive controller was probed at requested impact offsets -6, 0, and +6 px.

## Baseline

Contract v3 and the default predictive vision controller config remain unchanged. Baseline #26 recorded 0/15 canonical clear detections; raw score comparison is out of scope.

## Experiment

Scheduled design: exactly nine episodes in fixed seed-major order (404, 505, 606; offsets -6, 0, +6). Completed 9/9 arms. Each episode used raw RGB action selection and stopped at first confirmed bounce plus 8 native frames, termination/life loss, or 5,000 native frames (or the aggregate budget).
Contract hash `d97b7fcb5758cba495857b3972a119ed3eb7db74fb0145d151e9d82725c5a786`; controller config hash `25ddff6756747a6539ddd6a1eb570e7ef4c274f3e700ae011e68c87cae37c724`.
ROM SHA-256 `376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5`; ALE-Py `0.12.0`; Gymnasium `1.3.0`; Python `3.12.14`; NumPy `2.3.5`; OpenCV `5.0.0`.

## Primary Result

`positive_response_seed_count = 0` / 3. Classification: **INCONCLUSIVE**.

Seed-level fits use outgoing vx regressed on observed impact offset:

| Seed | Valid triplet | Strict offset order | Span (px) | OLS slope | Residuals | Positive |
|---:|:---:|:---:|---:|---:|---|:---:|
| 404 | True | False | 2.000 | -0.00000 | 0.0000, 0.0000, 0.0000 | False |
| 505 | False | False | n/a | n/a | n/a | False |
| 606 | False | False | n/a | n/a | n/a | False |

## Secondary Metrics

| Seed | Offset | Observed impact offset | Outgoing vx | Bounce | Direct coverage | Frames | Stop | Clears | Requested counts | ALE input counts | Exclusion |
|---:|---:|---:|---:|:---:|---:|---:|---|---:|---|---|---|
| 404 | -6 | 0.500 | -2.000 | yes | 0.936 | 77 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 69, 'RIGHT': 8}` | `{'FIRE': 3, 'RIGHT': 8, 'NOOP': 66}` | none |
| 404 | 0 | 0.500 | -2.000 | yes | 0.936 | 77 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 68, 'RIGHT': 9}` | `{'FIRE': 3, 'RIGHT': 9, 'NOOP': 65}` | none |
| 404 | 6 | 2.500 | -2.000 | yes | 0.936 | 77 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 28, 'RIGHT': 29, 'LEFT': 20}` | `{'FIRE': 3, 'RIGHT': 29, 'NOOP': 25, 'LEFT': 20}` | none |
| 505 | -6 | -9.500 | n/a | yes | 0.923 | 77 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 29, 'LEFT': 32, 'RIGHT': 16}` | `{'FIRE': 2, 'LEFT': 32, 'NOOP': 27, 'RIGHT': 16}` | wall_or_brick_collision |
| 505 | 0 | 2.500 | n/a | yes | 0.961 | 75 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 57, 'LEFT': 18}` | `{'FIRE': 2, 'LEFT': 18, 'NOOP': 55}` | wall_or_brick_collision |
| 505 | 6 | 2.500 | n/a | yes | 0.961 | 75 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 57, 'LEFT': 18}` | `{'FIRE': 2, 'LEFT': 18, 'NOOP': 55}` | wall_or_brick_collision |
| 606 | -6 | n/a | n/a | no | 0.649 | 97 | life_loss | False | `{'NOOP': 49, 'LEFT': 31, 'RIGHT': 17}` | `{'FIRE': 2, 'LEFT': 31, 'NOOP': 47, 'RIGHT': 17}` | bounce_not_confirmed |
| 606 | 0 | 2.500 | n/a | yes | 0.961 | 75 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 57, 'LEFT': 18}` | `{'FIRE': 2, 'LEFT': 18, 'NOOP': 55}` | wall_or_brick_collision |
| 606 | 6 | 2.500 | n/a | yes | 0.961 | 75 | first_paddle_bounce_plus_8_frames | False | `{'NOOP': 57, 'LEFT': 18}` | `{'FIRE': 2, 'LEFT': 18, 'NOOP': 55}` | wall_or_brick_collision |

Action requests and post-FIRE ALE inputs are recorded separately in JSON/CSV traces; sticky-resolved physical actions are unknown. Bounce candidates require the projected intercept to overlap the directly detected paddle span expanded by the configured 2 px ball radius; the ascending confirmation must be directly detected within ±5 px of paddle top. Wall/brick collision exclusions use observed velocity reversals and side-wall proximity.

## Delta vs Baseline

No clear-rate or raw-score delta is claimed because this probe measures first-bounce steering and uses only nine short episodes.

## Failure Analysis

Valid manipulation/measurement-floor seed triplets: 0/3. Per-run exclusions are listed above.

## Reproducibility

Command: `python -m scripts.evaluation.run_issue37_paddle_contact_probe --config configs/eval/issue37_paddle_contact_probe_v1.json --output-dir outputs/issue-37-paddle-contact-probe`.
Source branch `codex/issue-37-reachable-contact-probe`, commit `28e2415b13ac7bcfaf774464a43ad66b0baaf91f` (clean tree: True); source SHA-256 `e1f9ab533784f3054cfdb9ff43ccc9ae539f24b1d02d87f4f97d7b35026a7c7b`. Exact completed order: `[{'seed': 404, 'offset_px': -6}, {'seed': 404, 'offset_px': 0}, {'seed': 404, 'offset_px': 6}, {'seed': 505, 'offset_px': -6}, {'seed': 505, 'offset_px': 0}, {'seed': 505, 'offset_px': 6}, {'seed': 606, 'offset_px': -6}, {'seed': 606, 'offset_px': 0}, {'seed': 606, 'offset_px': 6}]`. Probe native frames: 705; predeclared ALE regression-test frame reserve: 350; combined conservative native-frame total: 1055; probe wall time: 2.589s.
Combined wall accounting: at most 20.0s for validation/tests/compile, 1.0s environment setup reserve, up to 519.0s evaluator runtime, and 60.0s report/artifact reserve (600s hard ceiling). Validation/tests/compile were charged against the frozen 20s reserve; aggregate test wall was not instrumented across shell calls. Evaluator wall: 2.589s. The combined budget accounting recorded by the runner was 22.620s (20s validation reserve plus measured environment, probe, and report generation); the shell command returned in 2.677s. All are below the 600s cap.
Conservative frame accounting: 705 formal native frames + 350 predeclared regression-test frame reserve = 1055 / 45,000; formal-run ceiling 44,650. Executor routing record: `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.

## Tests

Focused command: `python -m unittest tests.test_issue37_reachable_contact_probe -v` — 15 passed in the final pre-run test invocation (0.055s). The three target/baseline tests verify sign, clamping, unchanged zero-offset behavior at both side bounds, and baseline behavior before a descending intercept. Synthetic analysis verifies exactly 8 px qualifies at the OLS slope threshold and unordered/sub-8 px spans fail the floor. Regression command: `python -m unittest tests.test_vision_controller.PredictiveVisionPerceptionTests tests.test_vision_controller.PredictiveControlMathTests tests.test_vision_controller.VisionEvaluationSemanticsTests tests.test_evaluation_contract.Day15ContractTests.test_contract_v3_requires_explicit_precision_opt_in tests.test_completion.ALETransitionFrameCounterTests tests.test_completion.BreakoutCompletionDetectorTests tests.test_completion.CompletionSummaryTests tests.test_evaluation_completion_pipeline.EvaluationCompletionPipelineTests -v` — 33 passed; 2 selected modules had import errors. Exact runtime limitation: `ModuleNotFoundError: No module named 'torch'` importing `tests.test_evaluation_contract` through `breakout_rl/evaluation.py` and `tests.test_evaluation_completion_pipeline` through `breakout_rl/evaluation.py`. No dependencies were installed or runtime changed. Completion regression fixtures ran once; 350 native frames are reserved. Python compilation and `git diff --check` passed. Validation completed within the 20-second wall reserve.

## Remaining Uncertainty

Sticky action resolution is hidden; observations estimate ball motion after any sticky substitution. Three seeds and one bounce per run provide limited evidence. A collision before three ascending velocity estimates excludes that run.

### Post-run review and workflow record

The frozen formal command started before the Planner's requested pre-run checkpoint and source review had been delivered to this Executor. The checkpoint arrived after the command completed; this workflow deviation was discovered afterward. The nine-arm run was not repeated. The raw outputs under ignored `outputs/issue-37-paddle-contact-probe/` remain unchanged.

The run used source commit `28e2415b13ac7bcfaf774464a43ad66b0baaf91f`, clean at execution, with source digest `e1f9ab533784f3054cfdb9ff43ccc9ae539f24b1d02d87f4f97d7b35026a7c7b`. Post-run code review found the original collision guard checked sign changes only among `vy < 0` samples; a directly observed ascending-to-descending change could miss a brick collision. The corrected source is commit `202d5778225cd0590869653319391ef5cc5d43a5`, source digest `33eedcdc11e048f7160e2af635c1afb6dbbbc938cffd74ef9119cb95b1b2c384`. It checks reversals across every direct observation in the 8-frame window. This is a post-run correction only and was not used to rerun or rewrite any arm. No seed met the usability floor before the correction, so the frozen outcome remains **INCONCLUSIVE**.

Post-run pure/unit validation of the correction: `python -m unittest tests.test_issue37_reachable_contact_probe -v` — 16 passed in 0.057s, including the new synthetic ascending-to-descending collision test; no ALE-backed fixture or episode ran in this post-run check. `python -m py_compile ...` and `git diff --check` passed.

The pre-run selected regression command ran once: 33 tests passed; `tests.test_evaluation_contract` and `tests.test_evaluation_completion_pipeline` failed to import, both with the exact error `ModuleNotFoundError: No module named 'torch'` while importing `breakout_rl/evaluation.py`. The focused pre-run Issue #37 suite passed 15 tests. The ALE-backed completion regression fixtures ran once; their exact native-frame usage was not instrumented, so the full 350-frame reserve is charged. No dependency was installed and the runtime was not changed.

## Recommended Next Decision

Follow the frozen decision rule only: INCONCLUSIVE. PROMOTED permits considering one later controlled outgoing-angle experiment; it does not establish clear improvement or satisfy Issue #14.
