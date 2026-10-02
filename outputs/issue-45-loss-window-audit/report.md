# Issue #45: Contract v2 Life-Loss Observation Windows

**INCONCLUSIVE** — Has Verified Clear: **NO**.

Status: `completed`; episodes: `3/3`; life-loss windows: `15`; native frames: `12511`; wall seconds: `5.53`.
Formal run source commit: `82b72e1a7e1af89a4b3223ee221ab10e9d9a4164`. This report was refreshed offline from the preserved results and window index after collection; no further ALE run occurred.
All predeclared seeds completed; none were skipped.
Secondary coverage: `15` evaluator-detected life losses; `15` loss windows captured.
If no clear has complete provenance, the primary result is INCONCLUSIVE. This runner records a provenance-complete candidate as pending Planner validation; only independent Planner review may set GOAL_REACHED.
Life-loss windows contain exact uint8 model-input stacks and per-step raw Q/action metadata. Contact sheets are diagnostic only and never clear evidence.
Model `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`; metadata `fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512`; current inference spec `b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507` (metadata-declared older spec `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`); Contract v2 `7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a`; audit `43d431e77a4a3b60dd720933ee399767d456a6c633ee93aab7265d92e4f1b64b`.
ONNX Runtime `1.22.1`, providers `['CPUExecutionProvider']`. Source commit `82b72e1a7e1af89a4b3223ee221ab10e9d9a4164`, dirty `False`. Original PyTorch `.pt` bytes were absent and lineage remains metadata-declared.
Requested execution context: `gpt-6-luna` / `high`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.

## Episode outcomes

| Seed | Raw score | Lives remaining | Native frames | Stop | Clear status |
|---:|---:|---:|---:|---|---|
| 103 | 58.0 | 0 | 4138 | terminated | NO_CLEAR |
| 204 | 37.0 | 0 | 3561 | terminated | NO_CLEAR |
| 305 | 48.0 | 0 | 4812 | terminated | NO_CLEAR |

## Life-loss windows

| Seed | Episode | Event | Loss step | Frame | Decisions captured | Post-step observation | Contact sheet |
|---:|---:|---:|---:|---:|---:|:---:|---|
| 103 | 1 | 1 | 137 | 570 | 8 | yes | [loss_window_seed103_event1.png](loss_window_seed103_event1.png) |
| 103 | 1 | 2 | 489 | 1978 | 8 | yes | [loss_window_seed103_event2.png](loss_window_seed103_event2.png) |
| 103 | 1 | 3 | 704 | 2838 | 8 | yes | [loss_window_seed103_event3.png](loss_window_seed103_event3.png) |
| 103 | 1 | 4 | 859 | 3458 | 8 | yes | [loss_window_seed103_event4.png](loss_window_seed103_event4.png) |
| 103 | 1 | 5 | 1035 | 4160 | 8 | yes | [loss_window_seed103_event5.png](loss_window_seed103_event5.png) |
| 204 | 2 | 1 | 171 | 713 | 8 | yes | [loss_window_seed204_event1.png](loss_window_seed204_event1.png) |
| 204 | 2 | 2 | 379 | 1545 | 8 | yes | [loss_window_seed204_event2.png](loss_window_seed204_event2.png) |
| 204 | 2 | 3 | 589 | 2385 | 8 | yes | [loss_window_seed204_event3.png](loss_window_seed204_event3.png) |
| 204 | 2 | 4 | 774 | 3125 | 8 | yes | [loss_window_seed204_event4.png](loss_window_seed204_event4.png) |
| 204 | 2 | 5 | 891 | 3590 | 8 | yes | [loss_window_seed204_event5.png](loss_window_seed204_event5.png) |
| 305 | 3 | 1 | 310 | 1244 | 8 | yes | [loss_window_seed305_event1.png](loss_window_seed305_event1.png) |
| 305 | 3 | 2 | 480 | 1924 | 8 | yes | [loss_window_seed305_event2.png](loss_window_seed305_event2.png) |
| 305 | 3 | 3 | 795 | 3184 | 8 | yes | [loss_window_seed305_event3.png](loss_window_seed305_event3.png) |
| 305 | 3 | 4 | 1106 | 4428 | 8 | yes | [loss_window_seed305_event4.png](loss_window_seed305_event4.png) |
| 305 | 3 | 5 | 1203 | 4816 | 8 | yes | [loss_window_seed305_event5.png](loss_window_seed305_event5.png) |

Lossless uint8 arrays: `loss_windows.npz`; `loss_windows_index.json` binds every array to seed, episode, life-loss event/count, loss step/frame, per-decision Q and both action values.
RAM, score, lives, completion, and screenshots are evaluator diagnostics only; sticky-resolved physical actions remain unknown.
