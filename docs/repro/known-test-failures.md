# Known failures on `main`

These are pre-existing problems in the checkout, not regressions introduced by
the smoke path or CI. They are documented here so the smoke can classify them
explicitly instead of silently skipping tests. The machine-readable allowlist is
[`../../scripts/smoke/known_failures.json`](../../scripts/smoke/known_failures.json).

The short version: a clean checkout runs 315 discoverable tests. 294 pass, 8
skip because CUDA is unavailable, and 13 fail for the reasons below. A few
test modules fail at import time, so their individual test methods are never
counted.

## Unit-test failures

Command:

```bash
python -m unittest discover -s tests -v
```

Result:

```text
Ran 315 tests in 23.911s

FAILED (failures=1, errors=12, skipped=8)
```

### 1. Tests import scripts deleted by the `main` cleanup (10 failures)

Commit `1e50f4b` (`chore: slim main branch to maintained code`) removed
diagnostic, deployment, and visualization scripts from `main`. The tests that
import them were left behind.

| Test module | Exact error |
| --- | --- |
| `test_evaluation` | `ModuleNotFoundError: No module named 'scripts.analysis.generate_dqn_milestone_report'` |
| `test_inspect_cnn_dimensions` | `ModuleNotFoundError: No module named 'scripts.analysis.inspect_cnn_dimensions'` |
| `test_inspect_dqn_network` | `ModuleNotFoundError: No module named 'scripts.analysis.inspect_dqn_network'` |
| `test_inspect_dueling_network` | `ModuleNotFoundError: No module named 'scripts.analysis.inspect_dueling_network'` |
| `test_onnx_artifacts` | `ModuleNotFoundError: No module named 'scripts.deployment'` |
| `test_precision` | `ModuleNotFoundError: No module named 'scripts.deployment'` |
| `test_runtime_scores` | `ModuleNotFoundError: No module named 'scripts.visualization.visualize_runtime_score_distribution'` |
| `test_tensorrt` (method 1) | `ModuleNotFoundError: No module named 'scripts.deployment.build_tensorrt_engine'` |
| `test_tensorrt` (method 2) | `ModuleNotFoundError: No module named 'scripts.analysis.compare_tensorrt'` |

`test_evaluation` fails while `unittest` imports the module, so none of its
individual tests run. The same applies to the three `inspect_*` modules,
`test_onnx_artifacts`, `test_precision`, and `test_runtime_scores`. This is why
the 315-test count is lower than the historical suite.

### 2. ONNX inference tests need an untracked asset (3 failures)

```text
FileNotFoundError: /.../assets/day22/models/final_model/model.onnx
```

`tests/test_inference.py` points at `assets/day22/models/final_model/model.onnx`.
`assets/` is gitignored and is not part of `main`. A same-shaped model exists at
`web/public/models/final_model/model.onnx`, but the test has its own path and
its own expected metadata; copying the web model into `assets/` would change
test semantics and is not part of this change.

Affected tests:

```text
test_inference.ONNXRuntimePolicyTests.test_cpu_policy_accepts_a_prepared_model_input
test_inference.ONNXRuntimePolicyTests.test_cpu_policy_validates_dynamic_batch_and_returns_contract_shapes
test_inference.ONNXRuntimePolicyTests.test_requested_cuda_provider_never_silently_falls_back_when_unavailable
```

### 3. PER audit count drifted (1 failure)

```text
FAIL: test_day20_staged_replay_reset_is_rejected_as_primary_control
  self.assertEqual(len(audit["failed_checks"]), 18)
AssertionError: 22 != 18
```

`audit_baseline(Path("configs/per_experiment.json"))` reports 22 failed checks;
the test expects 18. The failure occurs at the first count assertion, so the
remaining assertions in that test are not reached. The cause was not
investigated because `tests/` and `configs/` are outside this task's edit
scope.

## Why these were not fixed here

The requested change was limited to `.github/`, `docs/`, the README, and new
smoke scripts. Fixing the failures requires either restoring scripts that
`main` intentionally removed, or editing/deleting tests, or adding the missing
model asset. Those are owner decisions, not documentation tasks. The smoke
therefore runs every test, reports these 13 as known, and fails on anything
else.

To make them blocking again after a fix:

```bash
python -m scripts.smoke.run_smoke --units-only --strict-units
```

and remove the fixed entry from `scripts/smoke/known_failures.json`.

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
the evaluator's seed expansion needs to change. `configs/eval/` is owned by
another change and was not modified here.

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
default temp directory), the field is `"unavailable"`. The smoke manifest
records the actual commit instead. Pass `--output-dir` inside the checkout if
you need training-level provenance in `summary.json`.

## Gitignore exception for this directory

`.gitignore` ignored `docs/*` except `docs/final-results/`. This change adds
`!docs/repro/` so these documents can be committed. It does not unignore the
other historical material under `docs/`.
