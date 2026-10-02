# Issue #41: Existing ONNX DQN First-Clear Probe

**INCONCLUSIVE** — Has Verified Clear: **NO**.

Status: `completed`; seeds run: `[101, 202, 303]`; native frames: `14455`; wall seconds: `5.61`.

The ONNX policy received only uint8 stacked pixels converted once to float32 by `/255`. RAM, score, lives and completion state were evaluator-only. Contract v2 was used unchanged; this result is not compared with Contract v3.

Model SHA-256: `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`; metadata SHA-256: `fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512`; inference spec current SHA-256: `b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507` (metadata-declared older SHA-256: `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`); Contract v2 SHA-256: `7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a`; completion audit SHA-256: `43d431e77a4a3b60dd720933ee399767d456a6c633ee93aab7265d92e4f1b64b`.

ONNX Runtime `1.22.1`, providers `['CPUExecutionProvider']`. Source commit `89457a82fd7e1c029cc9f4fbad84ef1a90a20f42`, dirty `False`. Original PyTorch `.pt` bytes were absent and could not be independently rehashed; lineage is metadata-declared.

| Seed | Stop reason | Agent steps | ALE-native frames | Raw score | Life losses | Clear status |
|---:|---|---:|---:|---:|---:|---|
| 101 | terminated | 1300 | 5199 | 73.0 | 5 | NO_CLEAR |
| 202 | terminated | 1403 | 5609 | 58.0 | 5 | NO_CLEAR |
| 303 | terminated | 912 | 3647 | 35.0 | 5 | NO_CLEAR |

## Provenance

Full per-step action/evaluator rows and the complete result payload are in `trajectory.jsonl` and `results.json`. Requested and wrapper ALE-input actions are distinct; sticky-resolved physical actions are not observable.

## Decision

A verified clear is reported as GOAL_REACHED / PROMOTED after independent Planner validation. If all three seeds finish without one, the frozen decision is INCONCLUSIVE; there is no rejection threshold.
