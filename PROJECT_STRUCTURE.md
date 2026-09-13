# Repository Structure

This repository contains three different kinds of material and they should stay visibly separated:

1. **Reusable implementation** — code that is imported and maintained as part of the RL system.
2. **Executable workflows** — thin command-line entry points that train, evaluate, inspect, benchmark, visualize, or deploy the system.
3. **Evidence and history** — committed experiment outputs, evaluation records, figures, reports, and 30-day articles.

The top-level layout is intentionally organized around those responsibilities.

```text
.
├─ breakout_rl/        # reusable Python package
├─ scripts/            # executable CLIs, grouped by responsibility
├─ configs/            # environment, training, experiment, and deployment contracts
├─ tests/              # regression and correctness tests
├─ docs/               # reader-facing 30-day articles
├─ assets/             # curated, reproducible evidence used by articles
├─ evaluations/        # committed policy-evaluation results
├─ experiments/        # committed training/system experiment records
├─ reports/            # derived summaries and engineering reports
├─ runs/               # local run location; ignored for new uncurated runs
├─ web/                # browser demo / deployment application
├─ design-system/      # UI design-system support for the web demo
├─ breakout_env.py     # temporary root-level environment exception
├─ environment.yml     # human-maintained Conda environment
├─ environment.lock.yml# locked environment snapshot
└─ AGENTS.md            # repository rules for coding agents
```

## Where new files should go

| You are adding… | Put it in… |
| --- | --- |
| Model, replay buffer, inference, environment, shared runtime logic | `breakout_rl/` |
| Training CLI | `scripts/training/` |
| Evaluation CLI | `scripts/evaluation/` |
| Diagnostics / model inspection / report generation | `scripts/analysis/` |
| Throughput / profiling / systems benchmark | `scripts/benchmarks/` |
| Figure, GIF, gameplay recording generator | `scripts/visualization/` |
| Small educational demonstration | `scripts/demos/` |
| Deployment/export helper CLI | `scripts/deployment/` |
| Canonical experiment/task settings | `configs/` |
| Reader-facing Ironman article | `docs/dayXX-*.md` |
| Curated evidence referenced by an article | `assets/dayXX/` |
| Formal evaluation result | `evaluations/` |
| Controlled experiment result | `experiments/` |
| Derived summary/report | `reports/` |
| Disposable local training output | `runs/` (ignored) |

## Important boundaries

### `breakout_rl/` is a library, not a notebook dump

If multiple scripts import the same logic, the logic belongs in `breakout_rl/`. Day-specific orchestration should not become a permanent public API just because it was useful while writing a particular article.

Existing files named after Days are historical implementation that can be refactored gradually. New reusable modules should be named by responsibility rather than by publication day.

### `scripts/` owns orchestration

Scripts should parse arguments, load configs, call reusable implementation, and write artifacts. They should not duplicate environment semantics or algorithm logic already implemented in the package.

Run scripts from the repository root using module syntax, for example:

```bash
python -m scripts.training.train_vectorized_dqn --help
python -m scripts.evaluation.evaluate_dqn --help
python -m scripts.analysis.analyze_q_values --help
```

### `assets/` is curated evidence; `experiments/` is history

`assets/dayXX/` should contain the evidence that a reader actually needs: selected charts, metadata, diagrams, small machine-readable traces, and article media.

`experiments/` may contain many controlled runs used to reach a conclusion. Do not copy an entire experiment directory into `assets/`; promote only the evidence needed to support the article.

### `evaluations/` is not another training-output directory

Only fixed-protocol policy evaluation belongs in `evaluations/`. Training curves, profiler dumps, sweep results, and backend-selection runs belong in `experiments/` unless they are promoted to curated `assets/`.

### `reports/` is derived

Reports summarize or compare existing evidence. A report should point back to the run/evaluation/config that produced its claims rather than becoming a second source of truth.

### `runs/` is disposable by default

The directory is ignored for new local runs. If a run becomes important enough to keep, move the minimal reproducible evidence into `experiments/`, `evaluations/`, or `assets/` instead of force-adding arbitrary local output.

## Canonical Breakout contract

From Day 16 onward, task semantics are defined by:

```text
configs/eval/breakout_contract_v2.json
```

Training, evaluation, gameplay recording, deployment parity checks, and DQN-family comparisons must not silently reconstruct or drift from that contract.

## Historical exception: `breakout_env.py`

`breakout_env.py` remains at the repository root because existing training/evaluation code and historical artifacts depend on it. It is an explicit migration exception, not a pattern for new root-level Python files.

When it is eventually migrated into `breakout_rl/`, do it as one compatibility-focused change with import updates and regression tests rather than as a cosmetic move.

## Cleanup rule

Before committing a new top-level directory, ask whether the content fits one of the categories above. If it does, use the existing directory. If it does not, document the architectural reason in the pull request and update this file and `AGENTS.md` together.
