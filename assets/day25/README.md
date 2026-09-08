# Day 25 evidence

Day 25 evaluates internal FP16 for the Day 21 canonical Final Model under the
same native NVIDIA CUDA workload used by Day 24. The FP32 ONNX artifact remains
untouched and remains the Browser baseline.

## Evidence

- `models/model.fp16.onnx` is a separate ONNX artifact with FP32 input/output
  and an internal FP16 graph.
- `models/model.fp16.onnx.metadata.json` records conversion, graph, source
  lineage, model hash, runtime package, and I/O policy.
- `precision-comparison.json` records the four CUDA paths, numerical error,
  action agreement, fixed-seed evaluation parity, latency references, model
  sizes, and the final recommendation.
- `benchmarks/day25-6a73229d443e/summary.json` and `raw-samples.json` store the
  batch=1/4/16/32 benchmark summary and every raw timing sample.
- `evaluations/*.json` store the same Contract v2 evaluation protocol and the
  requested action traces for each precision target.
- `fp32-vs-fp16.png` is generated from the comparison and benchmark artifacts;
  `fp32-vs-fp16.json` records its source hashes and command.
- `precision-comparison-flow.mmd` is the verified source for the matching
  Mermaid-rendered flow diagram.

## Reproduction

Run from the repository root with the pinned Conda environment:

```powershell
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.deployment.convert_onnx_fp16 --force
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.analysis.compare_precision --force
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.visualization.visualize_precision_comparison
conda run --no-capture-output -n breakout-rl-engineering python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py --theme neutral --background-color white --width 1600 --scale 2 assets/day25/precision-comparison-flow.mmd assets/day25/precision-comparison-flow.png
```

The formal comparison requires both PyTorch CUDA and
`CUDAExecutionProvider`. It fails closed when either path is unavailable and
keeps approximately 1 GiB of free VRAM before the experiment.

The observed result is a completed experiment with FP16 validation failed:
the model file is roughly half the size, but the FP16 action agreement and
fixed-seed evaluation parity do not meet the configured thresholds, and the
batch=1 end-to-end P95 does not improve. Native FP16 is therefore not
recommended for this policy; the Browser baseline remains FP32 ONNX.
