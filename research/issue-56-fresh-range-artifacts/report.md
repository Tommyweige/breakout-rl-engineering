# Issue 56: Contract v2 model-input grayscale range audit

Classification: **INCONCLUSIVE**; round status: **CONTINUE_RESEARCH**; HVC: **NO**. Diagnostic result: **PROMOTED**.

## Result summary

- Hypothesis: **PROMOTED for this sampled input window only**; overall Phase 1 outcome: **INCONCLUSIVE**.
- Baseline: no controller-performance baseline or performance comparison; the pinned ONNX policy followed unchanged raw-greedy.
- Primary diagnostic: `R/N = 0/87`; performance delta vs baseline: **N/A**.
- Failure/deviation analysis: no fixed-ROI pixel reached 200. The first preflight created/closed an ALE environment contrary to the no-create rule, with 0 resets, 0 steps, and 0 frames; its 0.605s validation-bundle elapsed remains in the cumulative 1.741s budget and it is not accepted as a successful preflight.
- Reproducibility: source commit `bf839c7395d24b021b5cbbe1fb3382e9b57a9332`, source digest `9f575455dd1a3959e22f6ac3327b90d9445dd380d4f34db62158c41fe6e2433e`; model/spec/metadata/contract hashes are recorded in `results.json`. Exact command: `timeout --signal=INT --kill-after=5s 90s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.evaluation.run_issue56_fresh_range --output-dir research/issue-56-fresh-range-artifacts`. Runtime: Python 3.12.14, NumPy 2.3.5, ONNX Runtime 1.22.1 CPUExecutionProvider.
- Validation: 12 focused tests passed against the recorded test file path/hash; compile and range diff checks passed; corrected CPU-only preflight created no ALE environment.
- Remaining uncertainty: this is one seed and one short fresh input window; a threshold crossing would not identify an object or establish detector quality.
- Recommended next decision: continue Phase 1 research. HVC remains **NO**; do not change the champion or begin Phase 2 from this diagnostic.

Seed 509: 0/87 decisions (0.0) had a channel-3 uint8 pixel >= 200 in either frozen ROI. This is sample reachability only.

Captured 87 pre-action stacks over 348 ALE-native frames; stop reason: `native_frame_cap`.

Channel-3 summaries (NumPy linear quantiles): `{"channel_3_ball_roi": {"max": 110, "min": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "quantile_method": "linear"}, "channel_3_full_plane": {"max": 148, "min": 0, "p50": 0.0, "p90": 142.0, "p95": 142.0, "p99": 148.0, "quantile_method": "linear"}, "channel_3_paddle_roi": {"max": 110, "min": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 110.0, "quantile_method": "linear"}}`

Pre-run validation: 12 focused tests passed in 0.280s; compile and diff checks passed in 0.029s and 0.003s; zero-frame preflight passed in 0.252s with 0 ALE frames, 0 resets and 0 steps. Setup/test cap: 20.0s; recorded pre-run use 1.741s plus formal setup 0.164s, leaving 18.095s.

Operational deviation: the initial preflight attempt was noncompliant because it created and closed an ALE environment. It made 0 resets, 0 steps and 0 native frames; it is retained as a deviation, not accepted as a successful preflight. Its 0.337s preflight and 0.605s validation-bundle elapsed remain included in the cumulative 1.741s pre-run cap budget. The corrected CPU-only preflight above passed without creating an ALE environment.

Model and spec hashes were verified before collection. Runtime: Python 3.12.14, ONNX Runtime 1.22.1, providers ['CPUExecutionProvider'].

Pre-write wall-cap guard: `{"artifact_write_time_included_in_wrapper_elapsed": true, "collection_cap_passed": true, "collection_cap_seconds": 60.0, "finalization_before_artifact_writes_seconds": 0.04298103699693456, "finalization_reserve_seconds": 10.0, "remaining_finalization_reserve_before_serialization_seconds": 9.957018963003065, "sampled_before_artifact_writes": true, "setup_test_cap_passed": true, "setup_test_cap_seconds": 20.0, "total_before_artifact_writes_passed": true, "total_before_artifact_writes_seconds": 0.4535207049921155, "total_cap_seconds": 90.0}`. Exact full-command elapsed and finalization including artifact writes are captured under `/tmp` and are authoritative for post-write caps.

Formal command wall: 0.644s external, 0.458s internal including artifact writes; finalization including artifact writes was 0.047s. The 60s collection, 10s finalization and 90s total caps passed. Captures: stdout `/tmp/issue56-formal.stdout` SHA-256 `dcffc0b84081d2ddfffda4c966f0e4bcb6d7283f48a53e5ee6c884ebbea74f2a`; stderr `/tmp/issue56-formal.stderr` SHA-256 `3042d2cd31292307b159eb83cfa49ef0b525584f115562f5f96ce8880ff653a3`.

Source commit `bf839c7395d24b021b5cbbe1fb3382e9b57a9332` on `codex/issue-56-fresh-range-observe`; branch commit `bf839c7395d24b021b5cbbe1fb3382e9b57a9332`; `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`.
## Post-run timing and capture accounting

Formal phase timing: setup/preflight 0.164360s; collection 0.246175s; finalization including artifact writes 0.047170s; internal total 0.457705s; external wrapper total 0.643623s. All caps passed (20s pre-run/setup, 60s collection, 10s finalization, 90s total).

Captured command stdout `research/issue-56-fresh-range-artifacts/formal.stdout` SHA-256 `dcffc0b84081d2ddfffda4c966f0e4bcb6d7283f48a53e5ee6c884ebbea74f2a`; stderr `research/issue-56-fresh-range-artifacts/formal.stderr` SHA-256 `3042d2cd31292307b159eb83cfa49ef0b525584f115562f5f96ce8880ff653a3`; wrapper timing record `research/issue-56-fresh-range-artifacts/formal_wrapper.json` SHA-256 `54ece6a861d5dfd3fc164a36d23eb2f5521364c4d1967134530c115e0969ebb9`. Original captures remain byte-identical under `/tmp`.
