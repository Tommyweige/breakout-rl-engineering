# Day 13 evidence artifacts

These files are the preserved, machine-readable evidence for the Day 13 article.

## Source run

- Run id: `day13-debug-cuda-seed42-final`
- Environment: `ALE/Breakout-v5`
- Seed: `42`
- Device: `cuda:0`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- PyTorch/CUDA: `2.13.0+cu130` / `13.0`
- Training steps: `10,000`
- Completed episodes: `48`
- Optimizer updates: `2,251`
- Target synchronizations: `21`
- Wall-clock: `60.1647` seconds
- SPS: `166.2104`
- Allocated / peak VRAM: `51,090,432` / `70,963,200` bytes
- Source checkpoint: `runs/day13-debug-cuda-seed42-final/checkpoints/step-00010000.pt`
- Source Git commit recorded by the run: `6f20079770da339f6ed2b89560be2c93928f8b14`
- Article workflow: `technical-blog-writer` skill was read before drafting and the article was reviewed against its evidence/figure gates.

The compact `debug-run/` directory keeps `config.json`, `metrics.csv`, and
`summary.json`, so the plots can be regenerated without committing the large
training checkpoint:

```text
python analyze_training_run.py assets/day13/debug-run --plots-dir assets/day13
```

The PNGs in this directory were generated from that run's `metrics.csv`. The
training run itself was started with:

```text
python train_dqn.py --preset debug --total-steps 10000 --seed 42 --device cuda --run-dir runs --run-id day13-debug-cuda-seed42-final
```

`debugging-workflow.mmd` is the verified conceptual flow source rendered to
`debugging-workflow.png` with `@mermaid-js/mermaid-cli@11.16.0`:

```text
C:\Users\tommy\anaconda3\envs\breakout-rl-engineering\python.exe C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py --theme neutral --background-color white --width 1200 --scale 2.0 assets/day13/debugging-workflow.mmd assets/day13/debugging-workflow.png
```

`random-baseline.json` is the corresponding five-episode random-policy
collector output from the same environment preprocessing and seed.
