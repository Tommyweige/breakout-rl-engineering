# Day 13 evidence artifacts

These files are the preserved, machine-readable evidence for the Day 13 article.

## Source run

- Run id: `day13-debug-seed42`
- Environment: `ALE/Breakout-v5`
- Seed: `42`
- Device: `cpu`
- Training steps: `10,000`
- Optimizer updates: `2,251`
- Target synchronizations: `21`
- Source checkpoint: `runs/day13-debug-seed42/checkpoints/step-00010000.pt`
- Source Git commit recorded by the run: `39298d2a3ac7186d974869794ecb05abc061707f`

The compact `debug-run/` directory keeps `config.json`, `metrics.csv`, and
`summary.json`, so the plots can be regenerated without committing the large
training checkpoint:

```text
python analyze_training_run.py assets/day13/debug-run --plots-dir assets/day13
```

The PNGs in this directory were generated from that run's `metrics.csv`. The
training run itself was started with:

```text
python train_dqn.py --preset debug --total-steps 10000 --seed 42 --device cpu --run-dir runs --run-id day13-debug-seed42
```

`random-baseline.json` is the corresponding five-episode random-policy
collector output from the same environment preprocessing and seed.
