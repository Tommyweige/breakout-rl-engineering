# Issue #41: Existing ONNX DQN First-Clear Probe

## Result

**INCONCLUSIVE — Has Verified Clear: NO.** All three frozen episodes completed with game-over termination and no canonical score-864 clear. Issue #41 has no rejection threshold, so this result does not reject the ONNX policy family. No Contract v3 comparison was made.

| Episode seed | Stop reason | Agent steps | ALE-native frames | Raw score | Life losses | Clear status |
|---:|---|---:|---:|---:|---:|---|
| 101 | terminated | 1,300 | 5,199 | 73 | 5 | NO_CLEAR |
| 202 | terminated | 1,403 | 5,609 | 58 | 5 | NO_CLEAR |
| 303 | terminated | 912 | 3,647 | 35 | 5 | NO_CLEAR |
| **Total** |  | **3,615** | **14,455** | **166** | **15** | **0 verified clears** |

The primary metric, Has Verified Clear, is **NO**. No episode reached the canonical clear detector threshold (864), so there are no verified-clear provenance rows. The canonical detector was available for all episodes and cumulative raw reward agreed with ALE scoreboard RAM.

## Frozen protocol and runtime

The exact ONNX artifact was evaluated with Contract v2: ALE/Breakout-v5, mode 0, difficulty 0, sticky action probability 0.25, frame skip 4, four-frame stack, environment-owned FIRE reset, unmodified rewards, and ALE's 108,000-native-frame / 27,000-step cap. The three seeds ran once in frozen order: 101, 202, 303. The runner recorded 14,455 formal native frames; the 350-frame test reserve was unused. No exploratory episode, rerun, or budget extension occurred.

At each decision the policy received only the uint8 grayscale stack `[4,84,84]`, converted to float32 `[1,4,84,84]` by dividing by 255 once. It returned raw q-values `[1,4]`; greedy argmax used action order `NOOP,FIRE,RIGHT,LEFT`. RAM, score, lives, and completion state were evaluator-only. Requested actions and wrapper ALE-input actions were logged separately; sticky-resolved physical actions are unknown.

ONNX Runtime `1.22.1` used `CPUExecutionProvider` only. The ONNX input was `observation` float32 `[N,4,84,84]`; output was `q_values` float32 `[N,4]`. The zero-input preflight and environment action-order preflight passed before gameplay, with zero ALE frames used by preflight.

**MODEL_ROUTING_VERIFICATION: UNAVAILABLE.**

## Artifact identity and provenance

- ONNX checkpoint SHA-256: `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`.
- Metadata SHA-256: `fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512`.
- Current inference spec SHA-256: `b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507`.
- Metadata-declared older inference spec SHA-256: `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`.
- Contract v2 SHA-256: `7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a`.
- Completion audit SHA-256: `43d431e77a4a3b60dd720933ee399767d456a6c633ee93aab7265d92e4f1b64b`.
- Completion source digest, using the repository's `_capture_source_provenance` six-file procedure: `9444a7955e65f4fe9e1568f5b321d8c9ffc0a105ae0d8549c5c76bc99c08f926`.
- Evaluated source commit: `89457a82fd7e1c029cc9f4fbad84ef1a90a20f42`, clean at run time.

The metadata records Double DQN with dueling architecture, training seed 2022, 2,500,000 transitions, source model SHA `6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a`, and source checkpoint SHA `ab07c0a48202428ddbb377c81f4091b3c434ce95e19d19fb1ec335df79841c48`. The original `.pt` checkpoint is absent from this checkout, so those lineage values are metadata-declared and the original bytes could not be independently rehashed. The older inference-spec hash matches the historical spec before a later Contract v2 digest-only update; the inference input, preprocessing, output, and action sections remain unchanged. Neither metadata nor spec was edited.

## Reproduction

The formal command was run exactly once:

```sh
timeout --signal=TERM --kill-after=5s 560s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.evaluation.run_issue41_onnx_dqn_clear_eval --contract configs/eval/breakout_contract_v2.json --inference-spec configs/inference/inference_spec.json --model web/public/models/final_model/model.onnx --episode-seeds 101,202,303 --max-native-frames-per-episode 108000 --output-dir outputs/issue-41-onnx-dqn-clear
```

Focused pure tests: `PYTHONPATH=/tmp/issue41-onnxruntime python -m unittest tests.test_issue41_onnx_dqn_clear -v` — 5 passed. Python compilation and `git diff --cached --check` passed. No PyTorch-importing tests were run.

## Preserved evidence

Exact run outputs are in [`issue-41-onnx-dqn-clear-artifacts`](issue-41-onnx-dqn-clear-artifacts/): full per-step `trajectory.jsonl`, `results.json`, and the generated `run_report.md`.

- Results SHA-256: `5fe3b96631dc77963d02c962726f39b3efe01b52d2986cfc979e19588f6d46d6`.
- Trajectory SHA-256: `5d89f8bc30fb1eba2acb21dc8355160addf0c17938ab054f310703401d5043dc`.
- Generated report SHA-256: `72e2af91831680a24f3db17a5b5df311dbaa4ab8b5bc1355028ae68581548f7c`.

The result payload records 5.61 seconds of runner wall time. All three episodes and final report/artifact writes completed well within the 560-second command cap. The frozen decision is INCONCLUSIVE; it neither changes the current Contract v3 research champion nor satisfies the Issue #14 first-clear goal.
