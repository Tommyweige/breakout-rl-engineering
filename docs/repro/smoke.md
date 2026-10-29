# Smoke path

The smoke path is one command that exercises the unit-test suite and the
training-to-evaluation chain that the repository exists to demonstrate.

```bash
python -m scripts.smoke.run_smoke --output-dir outputs/smoke
```

Run it from the repository root with the Python environment from
[`setup.md`](setup.md). `outputs/` is gitignored.

## What it runs

| Stage | Command underneath | What it proves |
| --- | --- | --- |
| Unit tests | `python -m unittest` over every `tests/test_*.py` | the maintained Python modules import and their contracts hold |
| Training | `python -m scripts.training.train_vectorized_dqn --preset smoke --total-steps 400 --num-envs 1 --device cpu` | ALE environment construction, vectorized rollout, replay insertion, optimizer updates, checkpoint writing |
| Evaluation | `python -m scripts.evaluation.evaluate_vectorized_dqn --device cpu` with the checkpoint | checkpoint loading, Contract v2 runtime binding, 15 fixed seed episodes, emulator-frame accounting, result artifacts |

The evaluation config is generated inside the smoke run because the checked-in
`configs/eval/breakout_eval.json` is rejected by the current evaluator; see
[`known-test-failures.md`](known-test-failures.md). The generated config uses
the contract's 15 concrete seeds with `episodes_per_seed: 1`, which is what the
runtime binding requires.

The training stage is deliberately small: 400 agent steps, one environment,
CPU. It is not a performance benchmark. The evaluation stage still uses the
real 108,000-frame episode limit and the real 15-seed protocol; the 400-step
policy simply loses its lives quickly (2,115 agent steps and 8,421 emulator
frames across 15 episodes in the verified run).

## Verified output

From a clean clone at commit `b39294d` (the review-branch commit used for this
run; later documentation-only commits do not change the smoke behavior, and
the merge commit will have a different SHA):

```text
$ python -m scripts.smoke.run_smoke --output-dir outputs/smoke

Breakout RL reproducibility smoke
  repository: /tmp/breakout-accept
  commit:     b39294d6c6c4efad0c5d822d26df5b1c7f08847a
  describe:   day9-217-gb39294d
  python:     3.12.15 (Linux-...-aarch64-with-glibc2.34)

[1/2] Unit tests
Discovered 315 tests under /tmp/breakout-accept/tests
...
Ran 315 tests in 24.9s

FAILED (failures=1, errors=12, skipped=8)

Known pre-existing failures on main: 13
  - test_evaluation: ImportError: scripts.analysis.generate_dqn_milestone_report ...
  ...
Unit tests: PASSED (294 passed, 8 skipped, 13 failed in 24.9s)

[2/2] End-to-end smoke: 400 training steps + one 15-episode evaluation
...
Training:   PASS in 38.2s (400 agent steps, 1597 emulator frames)
Evaluation: PASS in 44.6s (15 episodes, mean return 0.733, 8421 emulator frames)
Artifacts:  /tmp/breakout-accept/outputs/smoke/smoke-20261029T075301Z
Manifest:   /tmp/breakout-accept/outputs/smoke/smoke-20261029T075301Z/smoke-manifest.json

Smoke: PASS in 109.9s
```

The total was 109.9 seconds on a 4-vCPU ARM host. Earlier runs on the same host
took 116.6 and 143.5 seconds while the machine was busier. A final clean-clone
run at `49c5f9a` took 80.1 seconds (20.1 s unit tests, 28.5 s training, 29.3 s
evaluation); timings vary with host load. Allow about five minutes on a slow
CI runner.

## What it asserts

Training (`train/smoke-train/summary.json`):

- `status == "completed"`, `total_agent_steps == 400`, `total_steps == 400`;
- one environment, `dqn` algorithm;
- `contract_id` is `day15-breakout-evaluation-v2-fire-reset`;
- `total_emulator_frames >= 400` (the native ALE counter is recorded
  separately from agent steps);
- `last_checkpoint` exists on disk.

Evaluation (`evaluation/results.json`):

- `evaluation_status == "completed"`;
- exactly 15 episodes, all 15 complete;
- positive agent-step and emulator-frame totals, with frames greater than agent
  steps;
- finite mean return;
- the result references the checkpoint written by the training stage;
- `contract_provenance.runtime_binding_validated == True`.

Any unexpected unit-test failure, non-zero CLI exit, missing artifact, or
assertion mismatch fails the smoke with a non-zero exit code.

## Unit-test classification

The full suite is executed, not a curated subset. On `main`, 13 tests fail for
reasons that predate this smoke path. Each failure is matched by module name
(for import errors) or full test id against
[`../../scripts/smoke/known_failures.json`](../../scripts/smoke/known_failures.json),
printed with its reason, and counted. A failure outside the allowlist fails
the smoke.

The eight skipped tests are CUDA-only and skip on a CPU host:

```text
test_gpu_replay ... skipped 'CUDA is required for storage integration'
test_prioritized_replay ... skipped 'CUDA is required for device-residency checks'
test_replay_tensors ... skipped 'CUDA is required for transfer profiling'
test_training_diagnostics ... skipped 'CUDA is not available in this environment'
test_training_trainer ... skipped 'CUDA is required for GPU replay integration'
test_training_trainer ... skipped 'CUDA is required for PER trainer integration'
test_vectorized_training ... skipped 'CUDA is required for PER training integration'
test_inference ... skipped 'CUDA is required for CUDA EP test'
```

Once the 13 known failures are fixed, run:

```bash
python -m scripts.smoke.run_smoke --units-only --strict-units
```

and remove the fixed entries from `known_failures.json`.

## Flags

```text
--units-only     run the unit tests and stop
--e2e-only       skip the unit tests and run training + evaluation
--strict-units   fail if a known pre-existing failure is still present
--output-dir     parent directory for artifacts (default: a fresh temp dir)
```

The default (no flags) runs both stages.

## Artifacts

```text
outputs/smoke/smoke-<UTC>/
├── eval-config.json          generated Contract v2 evaluation config
├── logs/
│   ├── train.log             full training CLI output
│   └── evaluate.log          full evaluation CLI output
├── train/smoke-train/
│   ├── config.json
│   ├── metrics.csv
│   ├── summary.json
│   └── checkpoints/step-00000400.pt
├── evaluation/
│   ├── results.json
│   └── episodes.csv
└── smoke-manifest.json       commit, describe, dirty flag, package versions, timings, results
```

The manifest is the machine-readable record for the run:

```json
{
  "repository": {
    "commit": "b39294d6c6c4efad0c5d822d26df5b1c7f08847a",
    "describe": "day9-217-gb39294d",
    "dirty": false
  },
  "packages": {"torch": "2.13.0+cpu", "ale-py": "0.12.0", "...": "..."},
  "unit_tests": {"tests_run": 315, "passed": 294, "failed": 13, "skipped": 8},
  "e2e": {"training": {"total_agent_steps": 400}, "evaluation": {"total_episodes": 15}}
}
```

The training summary records `runtime.git_commit_sha` when the run directory is
inside a Git worktree. The smoke's default temporary directory is outside the
repository, so that field is `"unavailable"` there. The documented
`--output-dir outputs/smoke` keeps the run inside the checkout and does record
the commit. The smoke manifest records the commit either way.

## CI

The same two stages run in GitHub Actions:

```text
python -m scripts.smoke.run_smoke --units-only
python -m scripts.smoke.run_smoke --e2e-only --output-dir outputs/smoke
```

See [`ci.md`](ci.md).
