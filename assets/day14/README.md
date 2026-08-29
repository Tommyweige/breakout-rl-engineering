# Day 14 evidence assets

This directory preserves the real evidence behind Day 14. The experiment families are intentionally separated because learning-rate comparison, batch/GPU profiling, replay-system profiling, and gameplay recording answer different questions.

## Learning-rate experiments

10K screening evidence:

- `experiments/day14-cuda-lr-comparison-committed/manifest.json`
- `assets/day14/experiment-runs/day14-cuda-lr-comparison-committed/*/`

100K main evidence:

- `experiments/day14-cuda-lr-100k-main-final/manifest.json`
- `experiments/day14-cuda-lr-100k-main-final/comparison.json`
- `assets/day14/experiment-runs/day14-cuda-lr-100k-main-final/*/config.json`
- `assets/day14/experiment-runs/day14-cuda-lr-100k-main-final/*/metrics.csv`
- `assets/day14/experiment-runs/day14-cuda-lr-100k-main-final/*/summary.json`

The main batch used seed `42`, `100,000` environment steps per run, `float32`, `requested_device=cuda`, and resolved device `cuda:0` (`NVIDIA GeForce RTX 4060 Laptop GPU`). The 10K batch is screening evidence only and must not be used to rank configs.

Reproduce the controlled batches from the repository root:

```powershell
conda run --name breakout-rl-engineering python -m scripts.training.run_experiments --require-cuda --experiment-id day14-cuda-lr-100k-reproduction --runs-root assets/day14/experiment-runs configs/dqn_baseline.json configs/experiments/lr-low.json configs/experiments/lr-high.json
conda run --name breakout-rl-engineering python -m scripts.training.run_experiments --require-cuda --experiment-id day14-cuda-lr-10k-screening-reproduction --runs-root assets/day14/experiment-runs configs/screening/dqn_baseline_10k.json configs/screening/lr-low-10k.json configs/screening/lr-high-10k.json
```

Render the main figures from the manifest rather than entering values manually:

```powershell
conda run --name breakout-rl-engineering python -m scripts.visualization.visualize_experiment_comparison experiments/day14-cuda-lr-100k-main-final/manifest.json --output assets/day14/experiment-return-comparison.png --metrics return
conda run --name breakout-rl-engineering python -m scripts.visualization.visualize_experiment_comparison experiments/day14-cuda-lr-100k-main-final/manifest.json --output assets/day14/experiment-diagnostics-comparison.png --metrics loss q target gradient epsilon sps
```

## GPU-resident Replay evidence

Implementation/config/evidence:

- `breakout_rl/replay_gpu.py`
- `configs/final/day14-gpu-replay-vanilla-dqn.json`
- `experiments/day14-gpu-replay-ab-profiled-v2/`
- `experiments/day14-gpu-replay-lr-100k-v2/`
- `experiments/day14-gpu-replay-ab-profiled-v2/batch-size-comparison.json`
- `assets/day14/gpu-replay-lr-100k-v2-comparison.json`

Latest profiled full-trainer A/B:

- `experiments/day14-gpu-replay-ab-profiled-v5/manifest.json`
- `experiments/day14-gpu-replay-ab-profiled-v5/batch-size-comparison.json`
- `assets/day14/gpu-replay-ab-profiled-v5-profiling/*/runtime-samples-summary.json`

Both paths use batch size `32`, `train_frequency=4`, `learning_starts=2048`, the same model, seed, environment budget, and CUDA device. The checked-in report contains aggregate throughput and stage-timing evidence; raw local run/checkpoint paths remain provenance references rather than pretending those large binaries are committed.

The profiled run measured CPU-preallocated at about `408.18` environment transitions/s, `81.19` optimizer updates/s, and `2,598` training samples/s; GPU-resident replay measured about `379.95`, `75.57`, and `2,418`. CPU NumPy-to-pinned and H2D copy time were about `0.26 s` and `0.68 s`; GPU replay insertion was about `2.31 s` including about `2.12 s` GPU copy time, while GPU gather/cast was about `0.76 s`. These are stage timings, not additive wall-clock components.

Reproduce the profiled A/B:

```powershell
conda run --name breakout-rl-engineering python -m scripts.benchmarks.profile_batch_size_experiment --experiment-id day14-gpu-replay-ab-profiled-reproduction --experiments-root experiments --runs-root assets/day14/gpu-replay-ab-profiled-reproduction-runs --samples-root assets/day14/gpu-replay-ab-profiled-reproduction-profiling --sample-interval 1 --gpu-index 0 --require-cuda configs/performance/throughput-profiled-preallocated-batch-32.json configs/performance/throughput-profiled-gpu-replay-batch-32.json
```

The unprofiled batch-32 full-trainer A/B measured about `361.61` transitions/s for CPU replay with preallocated transfer and about `338.13` for GPU-resident replay. GPU replay storage used about `564.6 MB` VRAM. Optimizer-side replay microbenchmarks are upper-bound measurements and are not full-trainer SPS.

Stage-level microbenchmark artifacts remain separate:

- `replay-transfer-report.json`
- `preallocated-training-512.json`
- `gpu-replay-training-512.json`
- `copy-stream-prefetch-512.json`

## Performance, batch-size, and CPU-thread gates

Performance gate:

- `experiments/day14-performance-gate/throughput-comparison.json`
- `assets/day14/performance-runs/day14-throughput-before/*/`
- `assets/day14/performance-runs/day14-throughput-after/*/`

Batch-size profiling:

- `experiments/day14-batch-size-profiling-final/manifest.json`
- `experiments/day14-batch-size-profiling-final/batch-size-comparison.json`
- `assets/day14/batch-size-runs/day14-batch-size-profiling-final/*/`
- `assets/day14/batch-size-profiling/day14-batch-size-profiling-final/*/runtime-samples.csv`
- `assets/day14/batch-size-profiling/day14-batch-size-profiling-final/*/runtime-samples-summary.json`
- `assets/day14/batch-size-efficiency.png` plus JSON metadata

The 10K batch-size profile fixes learning rate `2e-4` and compares only batch sizes 32, 64, and 128. Larger batches increased training samples/s but did not increase end-to-end environment SPS, so no new batch-size candidate passed the 100K validation gate. The batch-32 reference is registered under `experiments/day14-batch-size-validation/`.

CPU-thread evidence:

- `experiments/day14-thread-selection/batch-size-comparison.json`
- `experiments/day14-thread-selection/thread-selection.json`
- `assets/day14/thread-profiling/day14-thread-selection/*/runtime-samples.csv`

The 1/2/4 thread profile selected 2 threads by end-to-end SPS. The frozen handoff is `configs/final/day14-vanilla-dqn.json`, with 100K evidence under `experiments/day14-final-frozen-100k/` and `assets/day14/final-runs/day14-final-frozen-100k/`.

Recreate the batch-size efficiency figure:

```powershell
conda run --name breakout-rl-engineering python -m scripts.visualization.visualize_batch_size_efficiency experiments/day14-batch-size-profiling-final/batch-size-comparison.json --output assets/day14/batch-size-efficiency.png
```

## GPU-Replay comparison figures

```powershell
conda run --name breakout-rl-engineering python -m scripts.visualization.visualize_experiment_comparison experiments/day14-gpu-replay-lr-100k-v2/manifest.json --output assets/day14/experiment-return-comparison.png --metrics return
conda run --name breakout-rl-engineering python -m scripts.visualization.visualize_experiment_comparison experiments/day14-gpu-replay-lr-100k-v2/manifest.json --output assets/day14/experiment-loss-comparison.png --metrics loss
conda run --name breakout-rl-engineering python -m scripts.visualization.visualize_experiment_comparison experiments/day14-gpu-replay-lr-100k-v2/manifest.json --output assets/day14/experiment-q-target-comparison.png --metrics q target
conda run --name breakout-rl-engineering python -m scripts.visualization.visualize_experiment_comparison experiments/day14-gpu-replay-lr-100k-v2/manifest.json --output assets/day14/experiment-gradient-comparison.png --metrics gradient
```

## Gameplay evidence

Real checkpoint gameplay GIFs are generated by `scripts.visualization.record_checkpoint_gameplay`. Each same-basename JSON stores checkpoint step/SHA-256, source commit when available, training run, evaluation seed/epsilon, frame count, and reproduction command.

Current artifacts:

- `gameplay-step-001k.gif` / `.json`
- `gameplay-step-010k.gif` / `.json`
- `gameplay-step-050k.gif` / `.json`
- `gameplay-step-100k.gif` / `.json`

They are qualitative evidence from real `rgb_array` frames, not a replacement for fixed multi-episode evaluation.

## Structural diagrams

Editable sources are `budget-stages.mmd` and `experiment-workflow.mmd`. Their current PNGs were rendered with the local `technical-blog-writer` Mermaid renderer; diagram rendering is separate from quantitative experiment reproduction.
