# Day 21 evidence

Day 21 freezes the Day 20 `dueling_double_dqn` winner and runs fresh seeds
through the predeclared 1M → 2.5M → 5M transition protocol under the same
Contract v2 and Day 16 CUDA/GPU-Replay backend.

The runner keeps selected trainers alive between milestones so the Replay
Buffer is continuous in the normal run. If a process is restarted, the
manifest records the checkpoint resume as non-exact when Replay was not
serialized.

## Reproduction

Plan the protocol without starting CUDA training:

```powershell
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.training.run_day21_final_training --stage plan
```

Run or resume the complete final-training protocol:

```powershell
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.training.run_day21_final_training --stage all --resume
```

Generate the source-backed report and figures after the run:

```powershell
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.analysis.generate_day21_report
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.visualization.visualize_final_long_training
```

The ignored `runs/` tree contains full CUDA checkpoints and raw metrics.
Reviewable compact metrics, evaluations, manifests, reports, and figures are
stored under the tracked Day 21 evidence paths.
