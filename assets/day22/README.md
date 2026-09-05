# Day 22 evidence

Day 22 freezes the model-boundary contract for the final Day 21 policy and
exports that policy as one FP32 ONNX graph. The ONNX graph contains only the
neural-network computation; Contract v2 environment preprocessing remains
outside the graph.

## Evidence

- `configs/inference/inference_spec.json` is the shared input/output and
  action-order contract.
- `inference/probe_states.npz` contains 60 fixed Contract v2 observations.
- `inference/pytorch_reference.npz` contains the CUDA-backed PyTorch inputs,
  Q-values, and greedy actions for those observations.
- `inference/metadata.json` records fixture hashes and runtime provenance.
- `models/final_model/model.onnx` is the exported FP32 deployment candidate.
- `models/final_model/model.onnx.metadata.json` separates CUDA source-model
  validation from CPU host-side ONNX checker results.
- `onnx-graph-summary.png` is generated from the exported graph's actual
  inputs, outputs, operator counts, node/initializer counts, and file sizes.
- `inference-contract.mmd` is the retained source for the rendered data-flow
  diagram `inference-contract.png`.

## Reproduction

Run these commands from the repository root:

```powershell
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.analysis.generate_probe_states --contract configs/eval/breakout_contract_v2.json --output assets/day22/inference/probe_states.npz --states-per-seed 4 --stride 32 --max-steps 256
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.deployment.export_onnx
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.visualization.visualize_onnx_graph_summary
conda run --no-capture-output -n breakout-rl-engineering python -m scripts.analysis.generate_day22_report
```

Use the pinned Mermaid renderer from the `technical-blog-writer` skill with
the following arguments:

```text
--theme neutral --background-color white --width 1400 --scale 2 assets/day22/inference-contract.mmd assets/day22/inference-contract.png
```

The formal export command requires an available NVIDIA CUDA device. It stops
when CUDA is unavailable instead of silently replacing the source reference
with a CPU run. ONNX Runtime execution and browser parity are Day 23+ work.
