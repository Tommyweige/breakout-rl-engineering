# Issue #50: Pixel-Only Lower-Playfield Paddle Alignment Probe

## Hypothesis Result

**INCONCLUSIVE**; round status `CONTINUE_RESEARCH`; candidate hypothesis `INCONCLUSIVE`.

## Baseline

Unchanged ONNX raw `argmax(Q)` at every decision. Detector output is logged in both arms and cannot affect baseline actions.

## Primary Result

Has Verified Clear: **NO**; evaluation status `completed`; episodes `6`; stop `schedule_completed`.

## Delta vs Baseline

Matched diagnostics below are descriptive-only and non-adjudicating; they do not change HVC classification.

| Seed | Baseline frames | Candidate frames | Δ frames | Baseline score | Candidate score | Δ score |
|---:|---:|---:|---:|---:|---:|---:|
| 105 | 6287 | 6287 | 0 | 74.0 | 74.0 | 0.0 |
| 206 | 4873 | 4873 | 0 | 70.0 | 70.0 | 0.0 |
| 307 | 3541 | 3541 | 0 | 38.0 | 38.0 | 0.0 |

## Failure Analysis

No provenance-complete canonical clear in 6 collected episode(s); 6 episode(s) ended at zero lives. This is descriptive failure context only; the classification remains HVC-only and INCONCLUSIVE without a verified clear.
The frozen detector reported both ball and paddle absent on all 7,354 decisions. Candidate therefore used raw-greedy fallback on all 3,677 candidate decisions and made zero detector overrides; the candidate intervention was not activated in this schedule. This is a frozen-detector diagnostic, not an HVC proxy or a threshold-tuning basis.
Per-decision detection/fallback results and evaluator trace are in `trajectory.jsonl`. No proxy metric can promote, reject, or update a champion.

## Reproducibility

Formal command: `timeout --signal=INT --kill-after=5s 600s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.evaluation.run_issue50_pixel_paddle_alignment --output-dir research/issue-50-pixel-paddle-alignment-artifacts`.

Source commit `8867a89812a3d2e67924fdaf7afbb4ff2c211213`; source digest `e32be297ba474635708fefd7da311e5db37b6076439e759be2701d4aaf2fee2f`; completion-source digest `9444a7955e65f4fe9e1568f5b321d8c9ffc0a105ae0d8549c5c76bc99c08f926`.
Model `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`; metadata `fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512`; spec `b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507` (metadata-declared historical spec `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`); Contract v2 `7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a`; completion audit `43d431e77a4a3b60dd720933ee399767d456a6c633ee93aab7265d92e4f1b64b`; calibration provenance `5d89f8bc30fb1eba2acb21dc8355160addf0c17938ab054f310703401d5043dc`.
Runtime Python `3.12.14`, ONNX Runtime `1.22.1`, provider `CPUExecutionProvider`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.
Metadata's historical spec SHA matches the earlier spec at commit 025d4bb; commit d3d235a changed only its embedded Contract v2 digest. Input, preprocessing, output, and action semantics are unchanged.

Focused pure tests: 9 passed. Zero-frame ONNX/provider/completion-support preflight passed. See `manifest.json` for wall/frame accounting and artifact hashes.

Frozen caps: setup plus focused tests 20s; collection 560s; finalization 20s; total wall 600s; each episode 108,000 native frames / 27,000 steps; six-episode aggregate 648,000 native frames. The external timeout includes final manifest writing.

## Tests and Setup

No detector smoke fixture was used; setup/preflight consumed zero ALE-native frames.

## Remaining Uncertainty

This is current-frame horizontal alignment; it does not estimate ball velocity or a future intercept. Because no valid detections occurred, the schedule did not observe candidate actions selected by the pixel rule. No-clear remains INCONCLUSIVE under HVC-only evaluation.

## Recommended Next Decision

Continue research under HVC-only criteria; do not rank or promote from diagnostics.
