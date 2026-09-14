# Repository instructions for coding agents

## Scope of `main`

`main` is the maintained code branch. Keep it focused on runnable implementation, tests, configs, and the browser application.

Do not reintroduce the 30-day article archive, article screenshots, historical experiment dumps, evaluation result archives, or generated reports into `main`. Historical material is preserved on `archive/full-history-before-main-cleanup`.

## Layout

```text
breakout_rl/   reusable Python implementation
configs/       task, training, evaluation, inference, deployment configs
scripts/       executable CLIs grouped by responsibility
tests/         regression and correctness tests
web/           browser application
```

`breakout_env.py` is a compatibility exception at repository root. Do not add new root-level Python CLIs.

## Code placement

- shared model/training/evaluation/inference logic -> `breakout_rl/`
- training entry points -> `scripts/training/`
- evaluation entry points -> `scripts/evaluation/`
- diagnostics/inspection -> `scripts/analysis/`
- performance/profiling -> `scripts/benchmarks/`
- export/deployment helpers -> `scripts/deployment/`

Keep scripts thin. Reusable logic belongs in `breakout_rl/`.

Run scripts from repository root using module syntax, e.g.:

```bash
python -m scripts.training.train_vectorized_dqn --help
python -m scripts.evaluation.evaluate_dqn --help
```

Do not use machine-specific `sys.path` hacks.

## Canonical Breakout contract

From Day 16 onward, task-defining environment/evaluation semantics come from:

```text
configs/eval/breakout_contract_v2.json
```

Do not silently drift frame skip/stack, sticky actions, FIRE ownership, life-loss handling, reward handling, seed lists, or episode limits between training and evaluation.

When the environment overrides a requested policy action with mandatory serve `FIRE`, Replay Buffer transitions must store the executed environment action.

## Change discipline

- preserve training/evaluation parity;
- update tests with behavior changes;
- avoid adding generated outputs to the repository;
- keep temporary runs, checkpoints, screenshots, benchmark dumps, and article artifacts ignored unless they are required runtime assets under `web/`.
