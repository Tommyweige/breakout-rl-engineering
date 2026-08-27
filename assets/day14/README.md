# Day 14 evidence assets

The comparison figure answers one question: after the Day 13 10K health check,
when the seed, 100K environment-step budget, precision, and CUDA device are
fixed, how do the three learning-rate variants' raw episode returns change over
the longer observation horizon?

The 10K screening source artifacts are retained separately:

- `experiments/day14-cuda-lr-comparison-committed/manifest.json`;
- `assets/day14/experiment-runs/day14-cuda-lr-comparison-committed/*/`.

The main Day 14 source artifacts are:

- `experiments/day14-cuda-lr-100k-main-final/manifest.json`;
- `experiments/day14-cuda-lr-100k-main-final/comparison.json`;
- `assets/day14/experiment-runs/day14-cuda-lr-100k-main-final/*/config.json`;
- `assets/day14/experiment-runs/day14-cuda-lr-100k-main-final/*/metrics.csv`;
- `assets/day14/experiment-runs/day14-cuda-lr-100k-main-final/*/summary.json`.

The GPU-resident replay implementation and its controlled evidence are kept
separately from the earlier CPU-replay reference:

- `breakout_rl/replay_gpu.py`;
- `configs/final/day14-gpu-replay-vanilla-dqn.json`;
- `experiments/day14-gpu-replay-ab-profiled-v2/`;
- `experiments/day14-gpu-replay-lr-100k-v2/`;
- `experiments/day14-gpu-replay-ab-profiled-v2/batch-size-comparison.json`;
- `assets/day14/gpu-replay-lr-100k-v2-comparison.json`.

The latest profiled full-trainer A/B is recorded in:

- `experiments/day14-gpu-replay-ab-profiled-v5/manifest.json`;
- `experiments/day14-gpu-replay-ab-profiled-v5/batch-size-comparison.json`;
- `assets/day14/gpu-replay-ab-profiled-v5-profiling/*/runtime-samples-summary.json`.

Both paths use batch size `32`, `train_frequency=4`, `learning_starts=2048`,
the same model, seed, environment budget, and CUDA device. The checked-in
report contains the aggregate throughput and stage timing evidence; the raw
run directories and per-second CSV files remain local because they include
large checkpoints. Report paths are relative and identify those local source
artifacts rather than pretending that uncommitted binaries are part of the
repository snapshot.

The profiled run measured CPU-preallocated at about `408.18` environment
transitions/s, `81.19` optimizer updates/s, and `2,598` training samples/s;
GPU-resident replay measured about `379.95`, `75.57`, and `2,418` respectively.
The CPU path spent about `0.26 s` in NumPy-to-pinned copies and `0.68 s` in
H2D copies over the run. The GPU path spent about `2.31 s` in replay insertion,
including about `2.12 s` of GPU copy time, while GPU gather/cast took about
`0.76 s`. These are stage timings from a profiled run, not numbers to add
directly into wall-clock and not a replacement for the unprofiled A/B
throughput comparison.

Recreate the profiled A/B (the `profile_stages` flag adds instrumentation
overhead, so use it for attribution rather than as the production speed
number) with:

```powershell
conda run --name breakout-rl-engineering python profile_batch_size_experiment.py --experiment-id day14-gpu-replay-ab-profiled-reproduction --experiments-root experiments --runs-root assets/day14/gpu-replay-ab-profiled-reproduction-runs --samples-root assets/day14/gpu-replay-ab-profiled-reproduction-profiling --sample-interval 1 --gpu-index 0 --require-cuda configs/performance/throughput-profiled-preallocated-batch-32.json configs/performance/throughput-profiled-gpu-replay-batch-32.json
```

The full trainer A/B keeps batch size and the DQN schedule fixed. In the
measured batch-32 run, CPU replay with preallocated transfer reached about
`361.61` transitions/s while GPU-resident replay reached about `338.13`
transitions/s. The GPU replay storage used about `564.6 MB` of VRAM. The
optimizer-side GPU replay microbenchmark remains a separate upper-bound
measurement and must not be reported as full-trainer SPS.

The stage-level microbenchmarks remain separate evidence:

- `replay-transfer-report.json` measures direct NumPy-to-CUDA versus pinned staging;
- `preallocated-training-512.json` separates replay sampling, NumPy-to-pinned, pinned-to-GPU, and DQN update;
- `gpu-replay-training-512.json` separates GPU gather/cast from the actual DQN update;
- `copy-stream-prefetch-512.json` is an overlap upper bound and explicitly excludes CPU sampling/host staging.

The main batch used seed `42`, `100,000` environment steps per run, `float32`,
`requested_device=cuda`, and the resolved device `cuda:0` (`NVIDIA GeForce RTX
4060 Laptop GPU`). The manifest and run summaries contain the exact PyTorch /
CUDA versions, SPS, wall-clock duration, 25K/50K/75K/100K checkpoint paths, and
peak memory metadata. The 10K batch is retained as screening evidence only and
must not be used to rank the configs.

The profiling gate is recorded separately from quality comparison:

- `experiments/day14-performance-gate/throughput-comparison.json`;
- `assets/day14/performance-runs/day14-throughput-before/*/`;
- `assets/day14/performance-runs/day14-throughput-after/*/`.

It contains the before/after end-to-end SPS, optimizer updates per second,
wall-clock time, CPU thread setting, sampled GPU utilization, peak VRAM,
logging cadence, and the 10K regression checks.

The batch-size GPU-efficiency stage is recorded separately from both the
learning-rate comparison and the CPU-thread profile:

- `experiments/day14-batch-size-profiling-final/manifest.json`;
- `experiments/day14-batch-size-profiling-final/batch-size-comparison.json`;
- `assets/day14/batch-size-runs/day14-batch-size-profiling-final/*/`;
- `assets/day14/batch-size-profiling/day14-batch-size-profiling-final/*/runtime-samples.csv`;
- `assets/day14/batch-size-profiling/day14-batch-size-profiling-final/*/runtime-samples-summary.json`;
- `assets/day14/batch-size-efficiency.png` and its JSON metadata.

The 10K profiling uses the selected `2e-4` learning rate and compares only
batch sizes 32, 64, and 128. The fixed-interval samples contain GPU
utilization, GPU power, used/total memory, process CPU utilization, and the
sampling method. Batch 64 and 128 increased training samples/s but did not
increase end-to-end environment SPS, so no new batch-size candidate met the
100K validation gate. The existing/final batch-32 100K run is registered as
the batch-validation reference at
`experiments/day14-batch-size-validation/`.

The CPU-thread selection evidence is:

- `experiments/day14-thread-selection/batch-size-comparison.json`;
- `experiments/day14-thread-selection/thread-selection.json`;
- `assets/day14/thread-profiling/day14-thread-selection/*/runtime-samples.csv`.

The 1/2/4 profile selected 2 threads by end-to-end SPS. The frozen handoff is
`configs/final/day14-vanilla-dqn.json`, with its 100K evidence under
`experiments/day14-final-frozen-100k/` and
`assets/day14/final-runs/day14-final-frozen-100k/`.

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
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/day14-cuda-lr-100k-main-final/manifest.json --output assets/day14/experiment-return-comparison.png --metrics return
```

Recreate the diagnostic comparison PNG:

```powershell
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/day14-cuda-lr-100k-main-final/manifest.json --output assets/day14/experiment-diagnostics-comparison.png --metrics loss q target gradient epsilon sps
```

Recreate the GPU-replay 100K comparison figures:

```powershell
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/day14-gpu-replay-lr-100k-v2/manifest.json --output assets/day14/experiment-return-comparison.png --metrics return
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/day14-gpu-replay-lr-100k-v2/manifest.json --output assets/day14/experiment-loss-comparison.png --metrics loss
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/day14-gpu-replay-lr-100k-v2/manifest.json --output assets/day14/experiment-q-target-comparison.png --metrics q target
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/day14-gpu-replay-lr-100k-v2/manifest.json --output assets/day14/experiment-gradient-comparison.png --metrics gradient
```

The real checkpoint gameplay GIFs are generated by
`record_checkpoint_gameplay.py`. Each same-basename JSON records the actual
checkpoint step, checkpoint SHA-256, source commit when available, training
run, evaluation seed, evaluation epsilon, frame count, and reproduction
command. Paths are repository-relative when the checkpoint is inside this
worktree. The current artifacts are:

- `gameplay-step-001k.gif` / `gameplay-step-001k.json`;
- `gameplay-step-010k.gif` / `gameplay-step-010k.json`;
- `gameplay-step-050k.gif` / `gameplay-step-050k.json`;
- `gameplay-step-100k.gif` / `gameplay-step-100k.json`.

They are qualitative evidence from real `rgb_array` frames. A single greedy
evaluation episode is not a replacement for the fixed multi-episode policy
evaluation planned for Day 15.

Recreate the batch-size efficiency figure from its real comparison report:

```powershell
conda run --name breakout-rl-engineering python visualize_batch_size_efficiency.py experiments/day14-batch-size-profiling-final/batch-size-comparison.json --output assets/day14/batch-size-efficiency.png
```

The budget-stage diagram is structural and its editable source is
`budget-stages.mmd`. The rendered PNG was produced with:

```powershell
conda run --name breakout-rl-engineering python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day14/budget-stages.mmd assets/day14/budget-stages.png
```

The runner workflow diagram remains in `experiment-workflow.mmd`; its rendered
PNG was produced with:

```powershell
conda run --name breakout-rl-engineering python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day14/experiment-workflow.mmd assets/day14/experiment-workflow.png
```
