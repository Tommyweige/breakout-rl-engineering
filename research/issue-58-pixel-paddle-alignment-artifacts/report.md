# Issue #58: Calibrated Grayscale Paddle Alignment First-Clear Probe

## Hypothesis Result

**INCONCLUSIVE**; HVC `NO`; round status `CONTINUE_RESEARCH`; candidate hypothesis `INCONCLUSIVE`.

## Baseline

Unchanged ONNX raw `argmax(Q)` at every decision. Detector output is logged in both arms and cannot affect baseline actions.

## Primary Result

Has Verified Clear: **NO**; evaluation status `completed`; episodes `6`; stop `schedule_completed`.

## Delta vs Baseline

Matched diagnostics below are descriptive-only and non-adjudicating; they do not change HVC classification.

| Seed | Baseline frames | Candidate frames | Δ frames | Baseline score | Candidate score | Δ score |
|---:|---:|---:|---:|---:|---:|---:|
| 510 | 4706 | 3120 | -1586 | 44.0 | 24.0 | -20.0 |
| 511 | 3742 | 3917 | 175 | 29.0 | 36.0 | 7.0 |
| 512 | 4495 | 3422 | -1073 | 50.0 | 31.0 | -19.0 |

## Failure Analysis

No provenance-complete canonical clear in 6 collected episode(s); 6 episode(s) ended at zero lives. This is descriptive failure context only; the classification remains HVC-only and INCONCLUSIVE without a verified clear.
Per-decision detection/fallback results and evaluator trace are in `trajectory.jsonl`. No proxy metric can promote, reject, or update a champion.

## Reproducibility

Formal command: `timeout --signal=INT --kill-after=5s 600s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.evaluation.run_issue58_pixel_paddle_alignment --output-dir research/issue-58-pixel-paddle-alignment-artifacts`.

Source branch `codex/issue-58-pixel-paddle-alignment`, commit `d591af16faae00dc8286b13a0de2df17e2556a38`; source digest `e35f2524c6d71c84a3a5e24d0821bdc36c6a3b222b59f6347bc7aa9773605dae`; completion-source digest `9444a7955e65f4fe9e1568f5b321d8c9ffc0a105ae0d8549c5c76bc99c08f926`.
Model `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`; metadata `fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512`; spec `b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507` (metadata-declared historical spec `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`); Contract v2 `7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a`; completion audit `43d431e77a4a3b60dd720933ee399767d456a6c633ee93aab7265d92e4f1b64b`; Issue #50 index `757eab8a54327e0663e2d83e5156958b776c1c5e7a09521a3355a9af43746978`, results `6954370d9270b154de55c38c26d107b25347a82d3ecf86bc723cdf03b68cf582`, trajectory `48c3ad056962862ba8a80219909caa6f4349a704642cef65b3210f62b47a2398`.
Issue #50 calibration stack hashes `{"seed105_baseline_ep1.uint8": "78f506be322d2faa03aa5770468ae45729e963f8d4d47859d9e74e956fc9c9fa", "seed105_candidate_ep2.uint8": "78f506be322d2faa03aa5770468ae45729e963f8d4d47859d9e74e956fc9c9fa", "seed206_baseline_ep3.uint8": "e973922e1390fb6dc32f849b036968a92083c8c3b673dd3d911aba346760dc55", "seed206_candidate_ep4.uint8": "e973922e1390fb6dc32f849b036968a92083c8c3b673dd3d911aba346760dc55", "seed307_baseline_ep5.uint8": "27ad2a34d6be0b01ae69764eb7ea8a930d7e77decc22a0f9cd83e4fd79197a95", "seed307_candidate_ep6.uint8": "27ad2a34d6be0b01ae69764eb7ea8a930d7e77decc22a0f9cd83e4fd79197a95"}`.
Runtime Python `3.12.14`, gymnasium `1.3.0`, ALE `0.12.0`, ONNX Runtime `1.22.1`, provider `CPUExecutionProvider`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.
Frozen calibration scan from 300 indexed rows / 150 unique observations: selected T=80; activation counts {'100': 42, '105': 12, '110': 2, '115': 0, '120': 0, '130': 0, '148': 0, '200': 0, '80': 64, '90': 52, '95': 50}; this is a detector diagnostic, not object truth.
Metadata's historical spec SHA matches the earlier spec at commit 025d4bb; commit d3d235a changed only its embedded Contract v2 digest. Input, preprocessing, output, and action semantics are unchanged.

Focused test command `env PYTHONPATH=/tmp/issue41-onnxruntime python -m unittest tests.test_issue58_pixel_paddle_alignment -v` passed 11 tests in 0.399s. Compile took 0.040s; zero-frame CPU preflight took 0.238s with `ale_environment_created=false`.
Earlier validation work totaled 3.140980s, including the two listed failed assertions, whose measured durations sum to 0.911211s: `[{"ale_environment_created": false, "ale_native_frames": 0, "ale_resets": 0, "ale_steps": 0, "command": "env PYTHONPATH=/tmp/issue41-onnxruntime python -m unittest tests.test_issue58_pixel_paddle_alignment -v", "name": "threshold_fixture_assertion", "result": "failed one assertion: the fixture used 199 but expected it below T=80; corrected to 79", "test_count": 11, "wall_seconds": 0.511703}, {"ale_environment_created": false, "ale_native_frames": 0, "ale_resets": 0, "ale_steps": 0, "command": "env PYTHONPATH=/tmp/issue41-onnxruntime python -m unittest tests.test_issue58_pixel_paddle_alignment -v", "name": "calibration_json_key_normalization_assertion", "result": "failed one assertion comparing integer threshold keys to JSON string keys; comparison normalized through JSON round-trip", "test_count": 11, "wall_seconds": 0.399508}]`. These were focused-code checks only; none created, reset, or stepped ALE. Prior exact-head passing validations charged 3.722964s across [0.7378490000000002, 0.7522075269953348, 0.759421371, 0.752562087, 0.720924262]; this exact-head passing validation charged 0.676663s; cumulative validation cap charge is 7.540607s.

Contract v2 sticky-action probability is 0.25; sticky resolution can make the physical ALE action differ from the requested policy/ALE-input action, so controller effects retain this uncertainty.
Frozen caps: validation plus process-launch reserve and formal setup 20s; collection 560s; finalization 20s; total wall 600s; each episode 108,000 native frames / 27,000 decisions; six-episode aggregate 648,000 native frames / 162,000 decisions. The external timeout includes final manifest writing.

## Tests and Setup

No detector smoke fixture was used; setup/preflight consumed zero ALE-native frames.

## Remaining Uncertainty

This is current-frame horizontal alignment; it does not estimate ball velocity or a future intercept. No-clear remains INCONCLUSIVE under HVC-only evaluation.

## Recommended Next Decision

Continue research under HVC-only criteria; do not rank or promote from diagnostics.

Raw formal stdout: `formal.stdout` SHA-256 `e96d83e31300d3432278b7676afd23fe32522fd03ded3dfbaee008fe10f710d5`.
Raw formal stderr: `formal.stderr` SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Pre-run validation plus formal command and artifact packaging elapsed: 23.939s.

Runner elapsed through manifest 11.245s; runner finalization through manifest 0.093s. Wrapper timing artifact: `wrapper_timing.json` SHA-256 `d49659c71dd73cfce10ed1968e42030c1f2e8fa2f204e9a950d72cb31a52287b`; outer elapsed 11.402s, launch reserve 5.000s, combined validation/command 23.943s, finalization 0.097s.

## Post-run Timing Reconciliation

Results recorded actual post-run artifact packaging of `0.004048385s`; the manifest's provisional `0.000528090s` value was stale and has been corrected to the same measured scope. `wrapper_timing.json` preserves the earlier boundary; `terminal_wrapper_timing.json` records the final measured boundary through terminal report/results/manifest/sidecar writes and excludes writing this reconciliation copy. The original completion manifest SHA-256 `759fd665bbbd54a556ea0614391ae2d2bee73fd440946b828d3c693a4883a426` is preserved in the terminal timing artifact. No collection evidence changed.
