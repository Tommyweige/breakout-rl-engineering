# Day 26 TensorRT evidence

This directory contains the native-NVIDIA-only TensorRT experiment. It does
not change or replace the Browser FP32 ONNX path.

## Final observed result

- preflight: `READY`
- host: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9
- driver: 610.74
- PyTorch: 2.13.0+cu130 with CUDA available
- ONNX Runtime: 1.29.0 with verified `CUDAExecutionProvider`
- TensorRT: 11.2.1.2 through an isolated optional site-packages path
- `trtexec`: not found; the Python API build path was used
- fixed-state/latency comparison: completed; TensorRT FP32 sanity check passed
- multi-episode score evaluation: 30 fixed seeds, four runtimes, zero runtime failures
- decision: TensorRT FP32 is native deployment-worthy; TensorRT FP16 is not worthwhile

The earlier 15-episode action-trace comparison is retained as diagnostic
evidence, not as an adoption gate. The formal score evidence uses the same 30
predeclared seeds for every runtime: TensorRT FP32 and ORT FP32 have identical
score distributions and paired scores, while TensorRT FP32 is about 31% faster
at batch-1 end-to-end P95. TensorRT FP16 remains lower in the 30-episode score
distribution and showed 93.33% fixed-state action agreement.

## Evidence map

| artifact | purpose |
| --- | --- |
| `tensorrt-preflight.json` | GPU, CUDA, ORT provider, TensorRT parser, FP16, and VRAM gate |
| `models/model.v2.fp32.engine` + metadata | strict FP32, strongly typed, TF32-disabled hardware-specific engine and build lineage |
| `models/model.v2.fp16.engine` + metadata | strongly typed FP16, TF32-disabled hardware-specific engine and build lineage |
| `tensorrt-comparison.json` | Q parity, action/margin diagnostics, fixed-seed evaluation, and benchmark references |
| `benchmarks/day26-tensorrt-v2-9838273b8269/` | Day 24-style raw samples and summaries for the corrected formal run |
| `tensorrt-comparison.png` | visualization generated from the comparison JSON |
| `runtime-score-comparison.json` | 30-seed per-runtime scores, aggregates, paired score differences, and bootstrap intervals |
| `runtime-score-distribution.png` | visualization generated directly from the runtime score comparison JSON |
| `tensorrt-experiment-flow.mmd` + `.png` | verified experiment control-flow diagram |

The runtime score comparison JSON is the source for the multi-episode numbers
in the article. The TensorRT comparison JSON remains the source for the
fixed-state and latency evidence. Engine metadata records the exact preflight
hash, source ONNX hash, GPU, TensorRT version, strongly typed network policy,
TF32-disabled policy, optimization profile, and workspace policy so a different
machine cannot silently reuse these binaries.

The earlier `day26-tensorrt-e3903363e652/`, `day26-tensorrt-ac70f0138eb8/`,
`day26-tensorrt-88704c3143c2/`, `day26-tensorrt-ce9063408ace/`, and
`day26-tensorrt-8d09a339e7fd/` directories are retained preliminary/final-v1
runs from before the current v2 precision-semantics correction. They are not
referenced by the final comparison and are kept only as auditable historical
output.
