# Day 24 evidence

Day 24 measures the native steady-state decision cost of the Day 21 canonical
Final Model and its Day 22 ONNX artifact after the Day 23 correctness pass.
The primary workload is batch=1; the benchmark also records batch 4/16/32 for
throughput context and CPU thread settings 1/2/4/default.

## Evidence

- `benchmarks/day24-final-native-v2-20260905-rtx4060/summary.json` stores the
  runtime/provider matrix, P50/P95 summaries, metadata, initialization timing,
  timing semantics, lineage hashes, and PyTorch peak VRAM.
- `benchmarks/day24-final-native-v2-20260905-rtx4060/raw-samples.json` stores every
  measured steady-state latency and timing breakdown sample.
- `batch1-latency.png` is generated from the summary and raw-sample pair and
  compares P50/P95 for model-only and end-to-end scope.
- `latency-distribution.png` plots the raw batch=1 end-to-end distributions and
  marks P50/P95.
- `inference-benchmark-flow.mmd` / `.png` explain where model-only and
  end-to-end scopes start.
- `inference-timing-flow.mmd` / `.png` explain output materialization,
  argmax, and CUDA synchronization in the timed path.
- `batch1-latency.json` and `latency-distribution.json` record the source
  hashes and reconstruction commands for the figures.

## Reproduction

Run from the repository root with the pinned Conda environment:

```powershell
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.benchmarks.benchmark_inference --benchmark-id day24-final-native-v2-20260905-rtx4060 --warmup 25 --iterations 100 --batch-sizes 1 4 16 32 --cpu-threads 1,2,4,default
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.visualization.visualize_inference_benchmarks --benchmark-id day24-final-native-v2-20260905-rtx4060
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.analysis.generate_day24_report --benchmark-id day24-final-native-v2-20260905-rtx4060
```

The benchmark requires both `torch.cuda.is_available()` and
`CUDAExecutionProvider`. It stops instead of labelling a CPU fallback as GPU.
