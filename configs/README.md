# Day 14 experiment configs

`dqn_baseline.json` is the complete, CUDA-first **100K main comparison**
configuration. The files under `experiments/` inherit it and change exactly one
field. The files under `screening/` inherit it but override the budget to 10K;
their labels and stage explicitly identify them as short health screening, not
config-ranking evidence. The runner resolves each file to a full `DQNConfig`,
records the resolved values in the manifest and in each run directory, and
computes the changed-field diff against the baseline.

The budget levels are explicit: smoke is 1K–10K steps, short screening is
exactly 10K, the main Day 14 comparison is exactly 100K, and a longer pilot is
250K–1M. A config whose declared level does not contain `total_steps` is
rejected. The main batch is evidence for the workflow and a development signal,
not a multi-seed claim about the best DQN setting.

Run the controlled batch sequentially from the repository root:

```powershell
conda run --name breakout-rl-engineering python run_experiments.py --require-cuda `
  --experiment-id day14-cuda-lr-100k-main `
  --runs-root assets/day14/experiment-runs `
  configs/dqn_baseline.json `
  configs/experiments/lr-low.json `
  configs/experiments/lr-high.json
```

The same command on one line is:

```powershell
conda run --name breakout-rl-engineering python run_experiments.py --require-cuda --experiment-id day14-cuda-lr-100k-main --runs-root assets/day14/experiment-runs configs/dqn_baseline.json configs/experiments/lr-low.json configs/experiments/lr-high.json

The 10K screening batch is separate:

```powershell
conda run --name breakout-rl-engineering python run_experiments.py --require-cuda --experiment-id day14-cuda-lr-10k-screening --runs-root assets/day14/experiment-runs configs/screening/dqn_baseline_10k.json configs/screening/lr-low-10k.json configs/screening/lr-high-10k.json
```
```
