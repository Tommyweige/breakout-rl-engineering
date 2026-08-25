# Day 14 experiment configs

`dqn_baseline.json` is the complete, CUDA-first development configuration. The
files under `experiments/` inherit it and change exactly one field. The runner
resolves each file to a full `DQNConfig`, records the resolved values in the
manifest and in each run directory, and computes the changed-field diff against
the baseline.

The configs are deliberately short development runs. They are evidence for the
workflow and a first signal, not a multi-seed claim about the best DQN setting.

Run the controlled batch sequentially from the repository root:

```powershell
conda run --name breakout-rl-engineering python run_experiments.py --require-cuda `
  configs/dqn_baseline.json `
  configs/experiments/lr-low.json `
  configs/experiments/lr-high.json
```

The same command on one line is:

```powershell
conda run --name breakout-rl-engineering python run_experiments.py --require-cuda configs/dqn_baseline.json configs/experiments/lr-low.json configs/experiments/lr-high.json
```
