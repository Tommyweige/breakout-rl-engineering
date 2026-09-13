# Executable Tooling

All command-line entry points live under `scripts/`. Reusable implementation belongs in `breakout_rl/`; script modules should stay thin and orchestrate library code, configs, artifacts, and reports.

> Repository-wide placement rules: [`PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md)

Run commands from the repository root with module syntax:

```powershell
python -m scripts.training.train_vectorized_dqn --help
python -m scripts.evaluation.evaluate_dqn --help
python -m scripts.analysis.analyze_q_values --help
```

## Categories

| Directory | Responsibility |
| --- | --- |
| `training/` | Training jobs and controlled training runners |
| `evaluation/` | Fixed-protocol policy evaluation and baselines |
| `analysis/` | Diagnostics, inspection, summaries, probes, report generation |
| `benchmarks/` | Throughput, profiling, replay, and systems-performance experiments |
| `visualization/` | Figures, plots, GIF/gameplay recording, rendered evidence generation |
| `deployment/` | Export, packaging, parity checks, and deployment preparation |
| `demos/` | Small educational or interactive demonstrations |

Choose a directory by **responsibility**, not by Day number. A script named after a Day is acceptable when it is inherently tied to a one-off experiment, but shared behavior should move into `breakout_rl/`.

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

A benchmark, smoke run, probe, toy demo, or visualization is evidence about a specific mechanism; it is not automatically a model-quality comparison.

Formal DQN-family comparisons must use the frozen environment contract, canonical training backend, controlled training budgets/seeds, and fixed evaluation protocol. Preserve the resulting records under `experiments/` or `evaluations/` according to their role.

## Adding a new script

1. Pick the category above that matches what the executable does.
2. Keep argument parsing and orchestration in `scripts/`.
3. Move logic imported by multiple tools into `breakout_rl/`.
4. Load canonical configs rather than retyping task-defining constants.
5. Write disposable output to ignored local locations first; only commit selected reproducible evidence.
6. Do not add a new root-level Python CLI unless there is a documented architectural reason.

Do not repair imports with machine-specific paths or `sys.path.append(...)`. Use package/module imports.
