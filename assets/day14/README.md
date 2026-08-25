# Day 14 evidence assets

The comparison figure answers one question: after the Day 13 10K health check,
when the seed, 100K environment-step budget, precision, and CUDA device are
fixed, how do the three learning-rate variants' raw episode returns change over
the longer observation horizon?

The 10K screening source artifacts are retained separately:

- `experiments/day14-cuda-lr-comparison-committed/manifest.json`;
- `assets/day14/experiment-runs/day14-cuda-lr-comparison-committed/*/`.

The main Day 14 source artifacts are:

- `experiments/day14-cuda-lr-100k-main/manifest.json`;
- `experiments/day14-cuda-lr-100k-main/comparison.json`;
- `assets/day14/experiment-runs/day14-cuda-lr-100k-main/*/config.json`;
- `assets/day14/experiment-runs/day14-cuda-lr-100k-main/*/metrics.csv`;
- `assets/day14/experiment-runs/day14-cuda-lr-100k-main/*/summary.json`.

The main batch used seed `42`, `100,000` environment steps per run, `float32`,
`requested_device=cuda`, and the resolved device `cuda:0` (`NVIDIA GeForce RTX
4060 Laptop GPU`). The manifest and run summaries contain the exact PyTorch /
CUDA versions, SPS, wall-clock duration, 25K/50K/75K/100K checkpoint paths, and
peak memory metadata. The 10K batch is retained as screening evidence only and
must not be used to rank the configs.

Recreate the 100K main batch with:

```powershell
conda run --name breakout-rl-engineering python run_experiments.py --require-cuda --experiment-id day14-cuda-lr-100k-reproduction --runs-root assets/day14/experiment-runs configs/dqn_baseline.json configs/experiments/lr-low.json configs/experiments/lr-high.json
```

Recreate the 10K screening batch separately with:

```powershell
conda run --name breakout-rl-engineering python run_experiments.py --require-cuda --experiment-id day14-cuda-lr-10k-screening-reproduction --runs-root assets/day14/experiment-runs configs/screening/dqn_baseline_10k.json configs/screening/lr-low-10k.json configs/screening/lr-high-10k.json
```

Recreate the main return PNG from its manifest without entering data manually:

```powershell
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/day14-cuda-lr-100k-main/manifest.json --output assets/day14/experiment-return-comparison.png --metrics return
```

Recreate the diagnostic comparison PNG:

```powershell
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/day14-cuda-lr-100k-main/manifest.json --output assets/day14/experiment-diagnostics-comparison.png --metrics loss q target gradient epsilon sps
```

The workflow diagram is structural and its editable source is
`experiment-workflow.mmd`. The rendered PNG was produced with:

```powershell
conda run --name breakout-rl-engineering python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day14/experiment-workflow.mmd assets/day14/experiment-workflow.png
```
