# Day 12 evidence metadata

## Training run

- Run ID: `day12-smoke-seed42-reproducible`
- Source metrics: `runs/day12-smoke-seed42-reproducible/metrics.csv`
- Configuration: `runs/day12-smoke-seed42-reproducible/config.json`
- Summary: `runs/day12-smoke-seed42-reproducible/summary.json`
- Seed: `42`
- Device: `cpu`
- Environment steps: `1,000`
- Optimizer updates: `243`
- Completed episodes: `4` (a fifth episode was still active at step 1,000)
- Target synchronizations: `11` (including the initial copy)

## Rebuild the run and plots

```powershell
python train_dqn.py --preset smoke --total-steps 1000 --device cpu --seed 42 --run-id day12-smoke-seed42-reproducible
python visualize_training_run.py --run-id day12-smoke-seed42-reproducible --runs-dir runs --output-dir assets/day12
```

The PNGs are generated from the CSV written by the first command. No metric
values are entered in the plotting script.

## Mermaid diagram

`dqn-training-loop.mmd` is a structural flow verified against
`breakout_rl/training/dqn_trainer.py`; it is rendered with:

```powershell
python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day12/dqn-training-loop.mmd assets/day12/dqn-training-loop.png
```
