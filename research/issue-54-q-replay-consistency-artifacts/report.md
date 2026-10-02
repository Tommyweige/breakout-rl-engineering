# Issue #54 Offline Q Replay Consistency

**PROMOTED** — replay consistency diagnostic only.

## Hypothesis and result

The exact Issue #50 uint8 observation stacks, normalized once with frozen Contract v2 preprocessing, reproduce the logged ONNX Q vectors on CPU ONNX Runtime 1.22.1 within a maximum absolute error of `1e-6`.

The sole primary metric was `max_i max_a |Q_replayed[i,a] - Q_logged[i,a]|`: **`0`** across **300** replayed observations (300 exact input joins). All integrity gates passed; max_abs_error 0 is at or below 1e-6.

## Integrity and provenance

All pinned input hashes passed. The exact Issue #50 index, trajectory, and six stack streams were checked; 300 rows joined uniquely on seed, arm, episode, step, and emulator frame, with matching observation hashes and four finite logged Q values each. Experiment branch: `codex/issue-54-q-replay-consistency`; source commit: `6b232266ad996e671c735054bb7dda88ab249db0`; research base: `a8641d99d6a0eb5b7226dae2b82ed3479f35e55f`; Issue #50 source digest: `e32be297ba474635708fefd7da311e5db37b6076439e759be2701d4aaf2fee2f`. Input hashes are recorded in `results.json`.

The metadata retains historical inference-spec SHA `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`; the pinned current spec only updates the embedded Contract v2 digest, with model input, preprocessing, output, and action semantics unchanged.

## Limits and decision

Issue #52's fresh-range probe failed before environment creation or collection (`N=0`), so it provides no fresh-input evidence. This replay covers only the selected 300 Issue #50 life-window stacks. It does not test a controller or establish object identity or a Breakout clear.

Has Verified Clear: **NO**. Phase 1: **INCONCLUSIVE**. Round status: `CONTINUE_RESEARCH`. No champion promotion or Phase 2 transition is implied. Continue research based only on replay consistency; do not infer controller performance from this audit.

`MODEL_ROUTING_VERIFICATION: UNAVAILABLE`


## Pre-run validation

focused test (11 tests): `env PYTHONPATH=/tmp/issue41-onnxruntime python -m unittest tests.test_issue54_q_replay_consistency -v` — passed, 0.258762s.

compile: `python -m py_compile breakout_rl/issue54_q_replay_consistency.py scripts/analysis/run_issue54_q_replay_consistency.py scripts/analysis/attach_issue54_capture.py scripts/analysis/run_issue54_once.py tests/test_issue54_q_replay_consistency.py` — passed, 0.033386s.

diff check: `git show --check --oneline HEAD` — passed, 0.003589s.

preflight: `timeout --signal=INT --kill-after=2s 20s env PYTHONPATH=/tmp/issue41-onnxruntime python -m scripts.analysis.run_issue54_q_replay_consistency --output-dir /tmp/issue54-preflight-6b23226 --preflight-only` — passed, 0.274068s.

Combined measured validation: 0.569805s.
Formal command wall: 0.536070s; failure-bundle finalization: 0.000117s.
The outer timeout is capped at remaining budget minus a 0.25s startup allowance; it includes guarded-run startup and final metadata writes. The reported measured-component total excludes pre-launch supervisor startup and final metadata refresh.

## Formal command capture

Stdout: `/tmp/issue54-formal.stdout` (preserved as `formal.stdout`) SHA-256 `da90c38623bb739de157518ee00651d0a70aa1a6f72f7de944b3cae496c5c0d9`.

Stderr: `/tmp/issue54-formal.stderr` (preserved as `formal.stderr`) SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Post-run capture/hash attachment: 0.002315s. Combined measured-component elapsed (excluding supervisor startup and final metadata refresh): 1.108307s (including failure-bundle and capture/hash finalization).
