# Configuration map

`configs/` contains machine-readable task, training, and experiment definitions. Treat canonical config paths as interfaces: do not move them casually just to make the tree look nicer.

## Task / environment contract

```text
configs/eval/breakout_contract_v2.json
```

From Day 16 onward this is the source of truth for the Breakout task semantics: environment id, frame skip/stack, sticky actions, FIRE reset ownership/confirmation, life-loss handling, TimeLimit semantics, evaluation epsilon, raw reward, and fixed evaluation seeds.

## Canonical training backend

```text
configs/training/day16-canonical-backend.json
```

This freezes the selected systems path for Day 17+: N=2 vectorized training, strict action-selection parity, CUDA/GPU Replay, float32, and the validated training-system settings/evidence lineage.

## Algorithm baselines

```text
configs/dqn_baseline.json
configs/double_dqn_baseline.json
configs/dueling_double_dqn_baseline.json
```

These are algorithm/architecture-facing baseline configs. `algorithm` selects the Bellman target rule (`dqn` or `double_dqn`), while `architecture` selects the Q-network representation (`standard` or `dueling`). Formal comparisons must keep the task/backend fixed and change only the intended variable.

## Experiment groups

- `screening/` — short health/sanity runs; not model-selection evidence by themselves.
- `experiments/` — controlled comparison variants.
- `batch-size/` — batch-size profiling candidates.
- `performance/` — throughput/profiling configurations.
- `final/` — frozen handoff/final-stage configs from completed experiments.
- `eval/` — evaluation protocol and environment contract.
- `training/` — canonical training-system manifests.

## Day 18 paired comparison

`experiments/day18-dqn-vs-double.json` freezes the Day 18 staged protocol:
100K screening, a seed-11 250K pilot, and a three-seed 500K main comparison.
The runner derives every stage config from the Day 16 backend manifest, so the
algorithm is the only within-pair DQN config variable.

## Day 20 DQN family comparison

`comparisons/dqn-family/manifest.json` freezes the three-family Day 20
comparison: DQN, Double DQN, and Dueling Double DQN share the Day 16 CUDA
backend, Contract v2, paired seeds, and staged 100K/250K/500K transition
milestones. The runner audits compatible Day 18 evidence before adding only
the missing family runs and can extend the aggregate-selected top two to 1M.

## Day 14 controlled batches

The Day 14 runner resolves each experiment file to a full `DQNConfig`, records the resolved values, and computes the changed-field diff against the baseline.

Run the 100K controlled batch from the repository root:

```powershell
conda run --name breakout-rl-engineering python -m scripts.training.run_experiments --require-cuda `
  --experiment-id day14-cuda-lr-100k-main `
  --runs-root assets/day14/experiment-runs `
  configs/dqn_baseline.json `
  configs/experiments/lr-low.json `
  configs/experiments/lr-high.json
```

The 10K screening batch remains separate:

```powershell
conda run --name breakout-rl-engineering python -m scripts.training.run_experiments --require-cuda --experiment-id day14-cuda-lr-10k-screening --runs-root assets/day14/experiment-runs configs/screening/dqn_baseline_10k.json configs/screening/lr-low-10k.json configs/screening/lr-high-10k.json
```

The profiling-driven batch-size stage is a systems experiment, not a model-quality comparison:

```powershell
conda run --name breakout-rl-engineering python -m scripts.benchmarks.profile_batch_size_experiment --experiment-id day14-batch-size-profiling-final --experiments-root experiments --runs-root assets/day14/batch-size-runs --samples-root assets/day14/batch-size-profiling --sample-interval 1 --gpu-index 0 --require-cuda configs/batch-size/bs32-10k.json configs/batch-size/bs64-10k.json configs/batch-size/bs128-10k.json
```

CPU-thread profiling is summarized by `scripts.analysis.summarize_thread_profiles`; the current Day 14 frozen config remains `configs/final/day14-vanilla-dqn.json`.
