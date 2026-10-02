# Issue #48: Q-Margin Runner-Up First-Clear Probe

**INCONCLUSIVE**; round status `CONTINUE_RESEARCH`; Has Verified Clear: **NO**.

Status `completed`; stop `schedule_completed`; episodes `6`; native frames `26132`; collection `10.92s`.

## Result and interpretation

- **Hypothesis Result:** **INCONCLUSIVE**; no verified clear occurred, so the candidate hypothesis was not adjudicated.
- **Baseline:** Raw greedy argmax, paired with the candidate on the same seeds in seed-major order.
- **Primary Result:** Has Verified Clear (HVC): **NO** in either arm. A verified clear is the Phase 1 goal gate; scores and survival do not substitute for HVC.
- **Delta vs Baseline:** Descriptive only, candidate minus baseline. Seed 104: native frames −490, raw score −14; seed 205: −1,607 frames, score −24; seed 306: +1,601 frames, score +35. Totals: −496 native frames and −3 raw score. These diagnostics do not adjudicate the hypothesis and are not clear proxies.
- **Failure Analysis:** Both arms terminated with zero lives before a canonical clear. Why the candidate did not produce HVC is unknown.
- **Tests:** Focused pure suite passed 9/9 in 0.001s. Zero-frame model/environment preflight took 0.218s and consumed 0 ALE-native frames.
- **Remaining Uncertainty:** Three paired seeds with no verified clear leave the near-tie runner-up hypothesis unadjudicated; reliability and champion impact are unknown. Model routing identity was unavailable.
- **Recommended Next Decision:** Continue research under the Phase 1 HVC standard. Make no champion or Phase 2 change.

The policy received only Contract v2 stacked pixels. RAM, score, lives, and completion remained evaluator-only.

Frozen threshold `0.0027604103088378906` from calibration SHA `5d89f8bc30fb1eba2acb21dc8355160addf0c17938ab054f310703401d5043dc` (3,615 rows, linear q=.25).

Model `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`; metadata `fbb9868fa0fb0ad7ce511e59b368efbef984279178c554dede0ae7e950731512`; spec `b396989041eff00396789c0f878fa783b071b0d3327aa1638c891a25d3f84507`; Contract v2 `7eca5ae5262a28aeabf3be658f8275a68f3d2ae40cc74379400be2890dc31a2a`.

ONNX Runtime `1.22.1`, providers `['CPUExecutionProvider']`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.

| Schedule | Arm | Seed | Steps | Native frames | Raw score | Life losses | Clear status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | baseline | 104 | 1206 | 4822 | 47.0 | 5 | NO_CLEAR |
| 2 | candidate | 104 | 1083 | 4332 | 33.0 | 5 | NO_CLEAR |
| 3 | baseline | 205 | 1136 | 4542 | 45.0 | 5 | NO_CLEAR |
| 4 | candidate | 205 | 734 | 2935 | 21.0 | 5 | NO_CLEAR |
| 5 | baseline | 306 | 988 | 3950 | 34.0 | 5 | NO_CLEAR |
| 6 | candidate | 306 | 1388 | 5551 | 69.0 | 5 | NO_CLEAR |

Source commit `659ec73763693286ed58686c561ae7c74be15042`; Issue #48 source digest `f12e63339435fd85fff75483aba000d8d01e4a2b8e92b47ee536634b3b9b3932`.
Metadata-declared older inference spec SHA `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`. Metadata's older spec SHA is exact at commit 025d4bb; d3d235a changed only the embedded Contract v2 digest to the current Contract v2 hash. Input, preprocessing, output, and action sections are unchanged.
**Allowed experiment classification:** `PROMOTED`, `REJECTED`, or `INCONCLUSIVE`; this run is `INCONCLUSIVE`. **Round status:** `CONTINUE_RESEARCH`. These are distinct: `GOAL_REACHED` is the round state only when any arm produces a provenance-complete canonical HVC. Exact setup, collection, finalization, and total wall accounting is in `manifest.json` (caps: 560s, 20s, 600s). Focused test timing is listed there as a pre-run measurement.

## Reproducibility

- Command: `timeout --signal=INT --kill-after=5s 600s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.evaluation.run_issue48_qmargin_runner_up --output-dir outputs/issue48-qmargin-runner-up`
- Seeds/order: baseline104, candidate104, baseline205, candidate205, baseline306, candidate306.
- Run source commit: `659ec73763693286ed58686c561ae7c74be15042`; source digest: `f12e63339435fd85fff75483aba000d8d01e4a2b8e92b47ee536634b3b9b3932`.
- Artifact manifest: `research/issue48-qmargin-runner-up-artifacts/manifest.json`.

Full Q/action/evaluator trajectories and provenance are in `trajectory.jsonl` and `results.json`.
A verified clear stops the schedule immediately. No-clear runs are INCONCLUSIVE; diagnostics do not classify the hypothesis.

Measured finalization prep `0.006s`; total through report preparation `11.132s`.
