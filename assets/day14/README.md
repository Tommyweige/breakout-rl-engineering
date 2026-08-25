# Day 14 evidence assets

The comparison figure answers one question: when the seed, environment-step
budget, precision, and CUDA device are fixed, how do the three learning-rate
variants' raw episode returns change over environment step?

The source artifacts are:

- `experiments/day14-cuda-lr-comparison-committed/manifest.json`;
- `assets/day14/experiment-runs/day14-cuda-lr-comparison-committed/*/config.json`;
- `assets/day14/experiment-runs/day14-cuda-lr-comparison-committed/*/metrics.csv`;
- `assets/day14/experiment-runs/day14-cuda-lr-comparison-committed/*/summary.json`.

The batch used seed `42`, `10,000` environment steps per run, `float32`,
`requested_device=cuda`, and the resolved device `cuda:0` (`NVIDIA GeForce RTX
4060 Laptop GPU`). The manifest and run summaries contain the exact PyTorch /
CUDA versions, SPS, wall-clock duration, checkpoint paths, and peak memory
metadata.

Recreate the batch with:

```powershell
conda run --name breakout-rl-engineering python run_experiments.py --require-cuda --experiment-id day14-cuda-lr-reproduction --runs-root assets/day14/experiment-runs configs/dqn_baseline.json configs/experiments/lr-low.json configs/experiments/lr-high.json
```

Recreate the PNG from a manifest without entering data manually:

```powershell
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/day14-cuda-lr-comparison-committed/manifest.json --output assets/day14/experiment-return-comparison.png --metrics return
```

The workflow diagram is structural and its editable source is
`experiment-workflow.mmd`. The rendered PNG was produced with:

```powershell
conda run --name breakout-rl-engineering python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day14/experiment-workflow.mmd assets/day14/experiment-workflow.png
```
