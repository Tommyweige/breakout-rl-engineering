# Day 16 evidence

The Day 16 evidence is generated from the real `ALE/Breakout-v5` environment,
PyTorch CUDA training, and the fixed Day 15 Contract v2 evaluation protocol.

## Machine-readable sources

- `vectorized-training.json`: 10K accepted-transition screening for 1, 2, 4,
  and 8 environments, including stage timings and fixed-interval CPU/GPU
  samples.
- `replay-insertion.json`: `add_batch` microbenchmark for batch sizes 1, 2, 4,
  8, and 16. The input observation came from one real Breakout reset/step.
- `evaluation-summary.json`: the envs=1 reference and envs=4 near-top-throughput
  candidate evaluated with the same 15-episode Contract v2 settings.
- `runtime-samples/`: raw fixed-interval sampler CSV files for each screening
  run.

## Reproduce

```powershell
conda run --name breakout-rl-engineering python benchmark_vectorized_training.py --device cuda --replay-backend gpu --environment-counts 1 2 4 8 --total-steps 10000 --run-root runs/day16-benchmark-final-v2 --output assets/day16/vectorized-training.json
conda run --name breakout-rl-engineering python benchmark_replay_insertion.py --device cuda --iterations 500 --output assets/day16/replay-insertion.json
conda run --name breakout-rl-engineering python evaluate_vectorized_dqn.py --checkpoint runs/day16-benchmark-final-v2/envs-1/checkpoints/step-00010000.pt --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cuda --output-dir evaluations/day16-final-v2-envs1-contract-v2 --evaluation-id day16-final-v2-envs1-contract-v2
conda run --name breakout-rl-engineering python evaluate_vectorized_dqn.py --checkpoint runs/day16-benchmark-final-v2/envs-4/checkpoints/step-00010000.pt --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cuda --output-dir evaluations/day16-final-v2-envs4-contract-v2 --evaluation-id day16-final-v2-envs4-contract-v2
conda run --name breakout-rl-engineering python summarize_day16_evaluation.py --reference-results evaluations/day16-final-v2-envs1-contract-v2/results.json --reference-checkpoint runs/day16-benchmark-final-v2/envs-1/checkpoints/step-00010000.pt --candidate-results evaluations/day16-final-v2-envs4-contract-v2/results.json --candidate-checkpoint runs/day16-benchmark-final-v2/envs-4/checkpoints/step-00010000.pt --candidate-environment-count 4 --output assets/day16/evaluation-summary.json
conda run --name breakout-rl-engineering python visualize_vectorized_training.py assets/day16/vectorized-training.json --insertion-report assets/day16/replay-insertion.json --output-dir assets/day16
conda run --name breakout-rl-engineering python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day16/vectorized-pipeline.mmd assets/day16/vectorized-pipeline.png
```

The checkpoint files and run directories are intentionally ignored local
training outputs; the evidence JSON records their paths and SHA-256 values.
