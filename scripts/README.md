# Executable tooling

All command-line entry points live under `scripts/`. Reusable implementation belongs in `breakout_rl/`; these modules should stay thin and orchestrate library code, configs, artifacts, and reports.

Run commands from the repository root with module syntax:

```powershell
python -m scripts.training.train_vectorized_dqn --help
python -m scripts.evaluation.evaluate_dqn --help
python -m scripts.analysis.analyze_q_values --help
```

## Categories

- `training/` — training and controlled experiment runners.
- `evaluation/` — fixed-protocol policy evaluation and baselines.
- `analysis/` — diagnostics, inspection, summaries, probe generation, and report generation.
- `benchmarks/` — throughput, profiling, replay, and systems-performance experiments.
- `visualization/` — figures, plots, GIF/gameplay recording, and rendered evidence generation.
- `demos/` — small educational or interactive demonstrations that are not formal model-quality experiments.

## Canonical handoff

From Day 16 onward, Breakout task semantics come from:

```text
configs/eval/breakout_contract_v2.json
```

The selected training systems backend comes from:

```text
configs/training/day16-canonical-backend.json
```

A script may expose additional CLI flags, but formal Day 17+ experiments must validate these canonical sources instead of silently reconstructing the task or backend.

## Formal experiments vs. diagnostics

A benchmark, smoke run, probe, toy demo, or visualization is evidence about a specific mechanism; it is not automatically a model-quality comparison. Formal DQN-family comparisons must use the frozen environment contract, the canonical training backend, controlled training budgets/seeds, and the fixed evaluation protocol.

## Adding a new script

Choose the directory by responsibility, not by Day number. If logic becomes reusable or is imported by multiple scripts, move that logic into `breakout_rl/` and keep the CLI thin. Do not add new root-level Python CLIs unless there is a documented architectural reason.
