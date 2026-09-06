# Day 23 evidence

Day 23 validates the Day 22 ONNX artifact against the saved PyTorch CUDA
reference from the Day 21 canonical final model. The native runtime package is
`onnxruntime-gpu==1.29.0`, built for CUDA 13.x on the validation machine.

## Evidence

- `pytorch-vs-onnx-parity.png` is generated from the saved comparison JSON.
- `pytorch-vs-onnx-parity.json` records the visualization source and command.
  It stores both the raw working-tree hash and LF-normalized hash of the
  comparison JSON.
- `pytorch-onnx-parity-flow.mmd` is the verified Mermaid source for the runtime
  and comparison sequence; the matching PNG is the article-ready rendering.
- `onnx-runtime-parity.json` records CPU and CUDA provider
  metadata, source hashes, N=1/N=4 metrics, top-2 margins, and disagreements.
- `onnx-gameplay-smoke.json` records the formal CUDA ORT gameplay
  smoke under Contract v2.
- `pytorch-gameplay-smoke.json` records the same short Contract v2 control path
  using the PyTorch CUDA policy.
- `article-workflow.md` records the required technical-blog-writer and figure
  review workflow for the reader-facing Day 23 article.

## Reproduction

Run from the repository root:

```powershell
python -m scripts.analysis.compare_pytorch_onnx --provider cuda
python -m scripts.analysis.compare_pytorch_onnx --provider both
python -m scripts.visualization.visualize_pytorch_onnx_parity
python -m scripts.demos.play_with_policy --runtime onnx --provider cuda --steps 256 --seed 101
python -m scripts.demos.play_with_policy --runtime pytorch --device cuda --steps 256 --seed 101 --output assets/day23/pytorch-gameplay-smoke.json
```

The CUDA command fails when `CUDAExecutionProvider` is unavailable; the policy
does not replace it with a CPU session. The comparison also verifies that the
ONNX graph assigns all 16 model nodes to CUDA on the validated ORT build.
