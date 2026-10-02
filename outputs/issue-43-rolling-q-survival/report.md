# Issue #43: Rolling Q4 Survival Ablation

**INCONCLUSIVE** — Has Verified Clear: **NO**.

Status: `completed`; episodes: `6/6`; native frames: `19459`; wall seconds: `8.44`.

A verified clear reaches the Phase 1 milestone after independent Planner validation. If none was verified, the first-clear result is INCONCLUSIVE.
Survival, score, action-switch rate, and Q-vector differences are descriptive diagnostics only; they do not rank a champion or replace the clear gate.

Model `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`; metadata `fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512`; current inference spec `b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507` (metadata-declared older spec `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`); Contract v2 `7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a`; audit `43d431e77a4a3b60dd720933ee399767d456a6c633ee93aab7265d92e4f1b64b`.

ONNX Runtime `1.22.1`, providers `['CPUExecutionProvider']`. Source commit `cacbae2c596f5888fb1950db5f9dd24a68cf739a`, dirty `False`. Original PyTorch `.pt` bytes were absent and lineage remains metadata-declared.
Requested execution context: `gpt-6-luna` / `high`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE` (no identity introspection is available).

## Paired descriptive diagnostics

| Seed | Status | Raw clear status | Q4 clear status | Raw frames | Q4 frames | Difference | Difference % | Raw lives lost | Q4 lives lost | Raw score | Q4 score | Raw switch rate | Q4 switch rate | Raw median Q margin | Q4 median Q margin |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 102 | paired_complete | NO_CLEAR | NO_CLEAR | 4476 | 1373 | -3103 | -69.33% | 5 | 5 | 45.0 | 5.0 | 0.46332737030411447 | 0.22157434402332363 | 0.006571769714355469 | 0.006378293037414551 |
| 203 | paired_complete | NO_CLEAR | NO_CLEAR | 3089 | 2187 | -902 | -29.20% | 5 | 5 | 26.0 | 11.0 | 0.4585492227979275 | 0.2619047619047619 | 0.005383491516113281 | 0.005033969879150391 |
| 304 | paired_complete | NO_CLEAR | NO_CLEAR | 5590 | 2744 | -2846 | -50.91% | 5 | 5 | 60.0 | 18.0 | 0.4523979957050823 | 0.2569343065693431 | 0.006498098373413086 | 0.0055582523345947266 |

## Verified clear events

- None.

## Canonical clears without complete provenance

- None.

Requested and wrapper ALE-input actions are stored separately; sticky-resolved physical actions remain unknown. Raw and smoothed float32 Q-vectors and per-step trajectories are in `trajectory.jsonl`; full episode details are in `results.json`.

## Run accounting

Exact frozen command, executed once:

```sh
timeout --signal=TERM --kill-after=5s 560s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.evaluation.run_issue43_rolling_q_survival_eval --contract configs/eval/breakout_contract_v2.json --inference-spec configs/inference/inference_spec.json --model web/public/models/final_model/model.onnx --episode-seeds 102,203,304 --arm-order raw-greedy,rolling-q4 --smoothing-window 4 --max-native-frames-per-episode 108000 --output-dir outputs/issue-43-rolling-q-survival
```

Runner wall: 8.443 seconds. Setup, focused tests, compilation, diff checks and preflight used about 3.2 seconds of the 20 second allowance; ONNX Runtime 1.22.1 was already present in isolated `/tmp/issue41-onnxruntime` (no install). Setup/pre-run validation wall (~3.2 seconds) plus runner wall (8.443 seconds) totals about 11.64 seconds through evaluation. Post-run report/manifest finalization fit within the reserved 30 seconds. All six episodes completed without a cap stop. Native frames: 19,459 of 648,000 formal; the 350-frame test reserve was unused. Preflight used zero native frames. The 600 second overall limit, 560 second command limit, 530 second collection stop, and 30 second finalization reserve were respected.

Hashes and exact setup/runtime accounting are recorded in `run_manifest.json`.


## Artifact hashes and provenance

- `results.json` SHA-256: `2b0bf7285d9d203e496fa58f4f75506f0c71c9277d0f424adb53be8c648f03d5`
- `trajectory.jsonl` SHA-256: `29f052e311de8a0a37a8d9f7647198a10798ba8d03cb4a542ce14356316eb4b9`
- `run_manifest.json` records the final report hash and all run accounting.
- Executed source commit: `cacbae2c596f5888fb1950db5f9dd24a68cf739a`; working tree clean at evaluation.
- Completion source digest: `c3e7aa81739833eb6b2070516b842c1772449b64dc513164388b06022f677126`.
