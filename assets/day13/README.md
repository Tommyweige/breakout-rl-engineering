# Day 13 evidence artifacts

These files are the preserved, machine-readable evidence for the Day 13 article.

## Source run

- Run id: `day13-debug-cuda-seed42-clean`
- Environment: `ALE/Breakout-v5`
- Seed: `42`
- Device: `cuda:0`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- PyTorch/CUDA: `2.13.0+cu130` / `13.0`
- Training steps: `10,000`
- Completed episodes: `43`
- Optimizer updates: `2,251`
- Target synchronizations: `21`
- Wall-clock: `68.6086` seconds
- SPS: `145.7543`
- Allocated / peak VRAM: `51,090,432` / `70,963,200` bytes
- Source checkpoint: `runs/day13-debug-cuda-seed42-clean/checkpoints/step-00010000.pt`
- Source Git commit recorded by the run: `5bd36a74a3234a2de659daecda709e3310cf8571`

The compact `debug-run/` directory keeps `config.json`, `metrics.csv`, and
`summary.json`, so the plots can be regenerated without committing the large
training checkpoint:

```text
python analyze_training_run.py assets/day13/debug-run --plots-dir assets/day13
```

The PNGs in this directory were generated from that run's `metrics.csv`. The
training run itself was started with:

```text
python train_dqn.py --preset debug --total-steps 10000 --seed 42 --device cuda --run-dir runs --run-id day13-debug-cuda-seed42-clean
```

`random-baseline.json` is the corresponding five-episode random-policy
collector output from the same environment preprocessing and seed.
