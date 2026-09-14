# Repository instructions for coding agents

## Scope of `main`

Keep `main` focused on maintained implementation, active configs, tests, and the browser application.

Do not add the 30-day articles, screenshots, generated reports, historical experiment dumps, or evaluation archives back to `main`. Reader-facing series material belongs on `docs-ironman-series`; the complete pre-cleanup snapshot is preserved on `archive/full-history-before-main-cleanup`.

## Layout

```text
breakout_rl/   reusable Python implementation
configs/       active task / training / evaluation / inference configuration
scripts/       maintained executable CLIs
tests/         correctness and regression tests
web/           browser application and required runtime assets
```

`breakout_env.py` remains at repository root for compatibility. Do not add new root-level Python CLIs.

## Code placement

- reusable model, replay, training, evaluation, and inference logic -> `breakout_rl/`
- training entry points -> `scripts/training/`
- evaluation entry points -> `scripts/evaluation/`
- reusable diagnostics -> `scripts/analysis/`
- reusable performance checks -> `scripts/benchmarks/`

Keep scripts thin; shared logic belongs in `breakout_rl/`.

Run tools from the repository root using module syntax, for example:

```bash
python -m scripts.training.train_vectorized_dqn --help
python -m scripts.evaluation.evaluate_dqn --help
```

Do not use machine-specific `sys.path` hacks.

## Canonical Breakout contract

Task-defining environment and evaluation semantics come from:

```text
configs/eval/breakout_contract_v2.json
```

Do not silently drift frame skip/stack, sticky actions, FIRE ownership, life-loss handling, reward handling, seed lists, or episode limits between training and evaluation.

When the environment overrides a requested policy action with mandatory serve `FIRE`, Replay Buffer transitions must store the executed environment action.

## Change discipline

- preserve training/evaluation parity;
- update tests with behavior changes;
- keep generated runs, checkpoints, screenshots, benchmark dumps, and article artifacts out of `main`;
- only commit large assets when they are required by the runtime under `web/`.
