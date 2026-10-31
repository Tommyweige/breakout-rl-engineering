# Unit tests on `main`

The suite is green. Up to the `main` cleanup this page listed 13 failures; they
are gone, and this page now records what was removed to get there, so the next
reader can tell a deliberate deletion from an accidental one.

```bash
python -m unittest discover -s tests -v
```

```text
Ran 340 tests in 133.641s

OK (skipped=5)
```

Five tests skip here, and the skip set depends on the machine, not on `main`:

- four `test_inference.ONNXRuntimePolicyTests` tests need the Day 22 ONNX model,
  which lives under the gitignored `assets/` tree;
- `test_tensors` keeps one negative test that skips when CUDA is present.

On a CPU-only host the CUDA-dependency tests skip instead. The allowlist in
[`../../scripts/smoke/known_failures.json`](../../scripts/smoke/known_failures.json)
is empty: any failure now fails the smoke.

## One test needs the git history

`test_per_experiment.test_day20_staged_replay_reset_is_rejected_as_primary_control`
audits the Day 20 baseline, and the audit reads the pinned manifest from commit
`aa2231a2` with `git show` and asserts that commit is an ancestor of `HEAD`. A
normal `git clone` has both; a `git clone --depth 1` has neither, so that one
test errors with `fatal: path 'experiments/day20-dqn-family/manifest.json' does
not exist`. The CI job checks out with `fetch-depth: 0` for this reason
([`ci.md`](ci.md)), and a shallow clone is the only checkout shape that fails.

## What the 13 failures were, and what happened to them

### 1. Tests imported scripts that `1e50f4b` deleted (10 failures)

Commit `1e50f4b` (`chore: slim main branch to maintained code`) removed
diagnostic, deployment, and visualization scripts from `main`, and left the
tests that imported them behind. Restoring the scripts was not an option — they
were deleted on purpose — so each test module was either deleted with its
subject or trimmed to the part that still tests maintained code.

| Test module | What it imported | Action |
| --- | --- | --- |
| `test_inspect_cnn_dimensions` | `scripts.analysis.inspect_cnn_dimensions` | deleted: the inspector is gone and nothing in `main` calls it |
| `test_inspect_dqn_network` | `scripts.analysis.inspect_dqn_network` | deleted, same reason |
| `test_inspect_dueling_network` | `scripts.analysis.inspect_dueling_network` | deleted, same reason |
| `test_onnx_artifacts` | `scripts.deployment` | deleted: the deployment package is gone, and this module also needs the untracked `assets/day22` model |
| `test_evaluation` | `scripts.analysis.generate_dqn_milestone_report` | trimmed: dropped the report and visualization test and the Day 14 manifest provenance test; the evaluator tests remain |
| `test_precision` | `scripts.deployment` | trimmed: dropped the fp16-conversion test; the precision audit tests remain |
| `test_runtime_scores` | `scripts.visualization.visualize_runtime_score_distribution` | trimmed: dropped the four tests that read the deleted `configs/eval/day26_runtime_score_evaluation.json` and the visualization test; the aggregation, statistics, and duplicate-seed tests remain |
| `test_tensorrt` | `scripts.deployment.build_tensorrt_engine`, `scripts.analysis.compare_tensorrt` | trimmed: dropped the two tests that build or compare engines; the build-seam and status tests remain |

The three `inspect_*` modules and `test_onnx_artifacts` had no maintained
subject left: `breakout_rl/` has no counterpart for those inspectors, and the
Day 22 assets they read are not in `main`. Keeping the tests would have meant
re-creating deleted scripts, so they were deleted with a note in the commit
message.

### 2. ONNX inference tests needed an untracked asset (3 failures, now skips)

`tests/test_inference.py` points at `assets/day22/models/final_model/model.onnx`.
`assets/` is gitignored and is not part of `main`, so `ONNXRuntimePolicyTests`
now carries a second `skipUnless` on that path, next to the existing
`onnxruntime` guard. When the asset is present the four tests run unchanged;
`MODEL` is no longer re-declared per test.

### 3. A recorded contract hash came from a CRLF checkout (1 failure)

`tests/test_per_experiment.py` asserts the number of failed checks reported by
`audit_baseline(configs/per_experiment.json)`. One of those checks compares the
Contract v2 SHA-256 recorded in `configs/per_experiment.json` with the file. The
recorded value was the hash of the contract **as written by a Windows checkout
with `core.autocrlf=true`**, so it never matched the LF file. The value is now
the LF hash, and `.gitattributes` pins LF for every file whose hash is recorded
(see below). The audit now reports 19 failed checks: 18 staged-resume checks,
plus one historical record that is meant to fail.

The historical one is the Day 20 manifest, read from commit `aa2231a2`. It froze
the CRLF hash of that era, and no change to this checkout can make that record
match, so the assertion expects all 19 and the comment next to it says why.

## Hash-recorded files are pinned to LF

[`.gitattributes`](../../.gitattributes) sets `text eol=lf` for every text file
whose SHA-256 is recorded in an audit, a contract, a fixture, or
`configs/per_experiment.json`: the two eval contracts, the completion audit, the
browser-clear audit CSV and summaries, the Laya browser-QA records, and the
completion fixtures. `git checkout` then writes LF on every platform, so a
Windows worktree hashes the same bytes as CI.

A worktree that predates the change keeps the bytes it already has. The index
already holds LF, so deleting those paths and checking them out again
(`git checkout -- <path>`) rewrites them with LF and the provenance tests pass.

## Evaluation-config mismatch (not a unit test)

The checked-in evaluation config cannot be used as-is with
`evaluate_vectorized_dqn` on this commit.

`configs/eval/breakout_eval.json` contains:

```json
"seeds": [101, 202, 303],
"episodes_per_seed": 5
```

That expands to the contract's 15 concrete episode seeds, and the config
validation in `_validate_contract_for_config` accepts it. The later runtime
binding does not:

- `_contract_runtime_binding` requires
  `tuple(evaluation_seeds) == contract.concrete_episode_seeds` (15 raw seeds);
- it also requires `episodes_per_seed == 1`.

The CLI passes `evaluation_config.seeds` (3 values) and
`evaluation_config.episodes_per_seed` (5) to `evaluate_policy`, so the runtime
binding rejects it.

Exact reproduction:

```bash
python -m scripts.evaluation.evaluate_vectorized_dqn \
  --checkpoint /tmp/.../checkpoints/step-00000400.pt \
  --config configs/eval/breakout_eval.json \
  --contract configs/eval/breakout_contract_v2.json \
  --device cpu \
  --output-dir /tmp/eval-attempt \
  --evaluation-id smoke-eval
```

Output:

```text
A.L.E: Arcade Learning Environment (version 0.12.0+0706845)
[Powered by Stella]
Vectorized checkpoint evaluation failed: Breakout contract runtime validation failed; refusing to evaluate: The evaluation seed list does not match Contract v2.
```

Exit code: `2`.

The smoke works around this by generating an equivalent config with the 15
concrete seeds and `episodes_per_seed: 1`. It evaluates exactly the same 15
episode seeds. This is a workaround, not a fix: either the checked-in config or
the evaluator's seed expansion needs to change.

## `evaluate_dqn` cannot run from a clean checkout

The README lists `scripts/evaluation/evaluate_dqn.py`. For `--policy dqn` under
Contract v2 it loads Day 14 provenance:

```text
Evaluation failed: experiments/day14-final-frozen-100k/manifest.json
```

That path is not in `main`. Use `evaluate_vectorized_dqn` for checkpoints
written by `train_vectorized_dqn`; that is the path the smoke exercises.

## Lockfile portability

`environment.lock.yml` is a Windows export and cannot be solved on Linux; see
[`setup.md`](setup.md#do-not-use-environmentlockyml-on-linux). The smoke uses
the CPU pip set, not the lockfile.

## Training provenance outside a Git worktree

`VectorizedDQNTrainer` records `runtime.git_commit_sha` by running `git -C
<run-dir> rev-parse HEAD`. If `--run-dir` is outside the checkout (the smoke's
default temporary directory), the field is `"unavailable"`. The smoke manifest
records the actual commit either way; the documented `--output-dir
outputs/smoke` path also gives the training summary the commit.
