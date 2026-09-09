# Day 26 article workflow

## Learning boundary

- Already known: Day 21 froze the final DQN; Day 22 froze the FP32 ONNX
  boundary; Day 23 verified ORT CUDA parity; Day 24 measured inference; Day 25
  measured the FP16 ONNX trade-off.
- New today: TensorRT preflight, a serialized hardware-specific engine,
  source-typed FP16 in TensorRT 11, and correctness-first native optimization.
- Future boundary: TensorRT is not a Browser serving runtime. The Browser
  demo remains the FP32 ONNX → ORT Web path.

## Figure questions

- `tensorrt-comparison.png`: what trade-off did the real probe parity,
  fixed-seed action agreement, batch-1 latency, and artifact sizes show?
- `runtime-score-distribution.png`: after allowing different trajectories, do
  the four runtimes keep the same multi-episode score distribution?
- `tensorrt-experiment-flow.png`: which gate must pass before engine numbers
  can be interpreted, and where does the Browser path remain separate?

## Reproduction commands

The TensorRT wheels were installed into an isolated optional environment; the
Day 22–25 base environment was not modified. Substitute the isolated
environment's `site-packages` directory for `<isolated-site-packages>`.

```powershell
$env:PYTHONPATH = '<isolated-site-packages>'
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.deployment.check_tensorrt_environment `
  --config configs/inference/tensorrt_experiment.json `
  --tensorrt-site-packages '<isolated-site-packages>' --force

conda run --no-capture-output -n breakout-rl-engineering python -m scripts.deployment.build_tensorrt_engine `
  --config configs/inference/tensorrt_experiment.json `
  --preflight assets/day26/tensorrt-preflight.json `
  --tensorrt-site-packages '<isolated-site-packages>' --precision float32 --force

conda run --no-capture-output -n breakout-rl-engineering python -m scripts.deployment.build_tensorrt_engine `
  --config configs/inference/tensorrt_experiment.json `
  --preflight assets/day26/tensorrt-preflight.json `
  --tensorrt-site-packages '<isolated-site-packages>' --precision float16 --force

conda run --no-capture-output -n breakout-rl-engineering python -m scripts.analysis.compare_tensorrt `
  --config configs/inference/tensorrt_experiment.json `
  --preflight assets/day26/tensorrt-preflight.json `
  --tensorrt-site-packages '<isolated-site-packages>' --device-index 0 --force

conda run --no-capture-output -n breakout-rl-engineering python -m scripts.visualization.visualize_tensorrt_experiment
```

The flow diagram was rendered from its adjacent Mermaid source with the
technical-blog-writer Mermaid renderer:

```powershell
python <technical-blog-writer-skill>/scripts/render_mermaid.py `
  assets/day26/tensorrt-experiment-flow.mmd `
  assets/day26/tensorrt-experiment-flow.png
```

## Final artifact hashes

These hashes correspond to the final generated run captured by the comparison
artifact:

- preflight JSON: `5c6cd8ab9ae62da6ac29312e7badf169343c56a7b3cfa1b522b815ed96ca8938`
- comparison JSON: `89d7c39164d5e91fd9ca4db1324dbd081ba558bbcde771d0cafcfdf91f47bd2e`
- FP32 engine: `8f7ce5120d35d9936a88f764efcb628f5947b7b7b1e69bc8aa7b33e2f7ac67ea`
- FP16 engine: `b359690733859597e1b58eddf4f88edbfda047f08b24b0d112a9a0bb2a2980cf`
- comparison plot: `7598f6e92256c048a4e41c36be25d989fe95f3067176446fa28c046045f7bf2e`
- benchmark id: `day26-tensorrt-v2-9838273b8269`
- runtime score config: `b38ef96728e9e4b1cb2e794984d0ee0bdcbc7dc80f3d4ea279cfd1c478db7a5b`
- runtime score JSON: `0db01b8bceceecef08355fb1ddc7f9ecdb4e189d2b9c20fb5a6cf9a15240073a`
- runtime score plot: `5cda3a1d1962f500feaf65b99bd18d838fef167f49b614c7aaad7f3fc12c8dc3`
