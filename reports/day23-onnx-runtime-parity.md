# Day 23 ONNX Runtime parity report

這份 report 回答一個具體問題：同一批 Day 22 固定 probe states，交給 Day 21 canonical Final Model 的 PyTorch CUDA reference 與 native ONNX Runtime 後，Q-values 和 greedy action 是否仍一致？

- status: `passed`
- selected providers: `cpu, cuda`
- probe count: `60`

## Frozen lineage

- canonical_source_model: `assets/day21/models/final_model/model.pt` / `6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a`
- source_checkpoint: `declared by metadata` / `ab07c0a48202428ddbb377c81f4091b3c434ce95e19d19fb1ec335df79841c48`
- onnx_model: `assets/day22/models/final_model/model.onnx` / `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`
- inference_spec: `configs/inference/inference_spec.json` / `68637a63d3f0242f74089314049251521bec14a6fa3267b57fa46470acfff8ec`
- probe_states: `assets/day22/inference/probe_states.npz` / `eec131af71f3aeb089be7b9d90607d87187a99a94739f135767fe9bf1f19c2f3`
- pytorch_reference: `assets/day22/inference/pytorch_reference.npz` / `36ac1d7dac9c2024d92d3669c4ea3cd9dcb1978c99bffe1d10cdea1618e5fd07`

PyTorch reference 的來源是 Day 21 frozen model；ONNX 是 Day 22 export。這裡沒有重新挑選 Day 20 comparison checkpoint，也沒有改寫 preprocessing contract。

## Provider results

### cpu

- requested: `CPUExecutionProvider`
- available: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- active: `['CPUExecutionProvider']`
- actual primary: `CPUExecutionProvider`
- ONNX Runtime: `1.29.0`
- GPU model: `NVIDIA GeForce RTX 4060 Laptop GPU`
- CUDA / cuDNN: `13.0` / `92000`
- graph assignment: `{'status': 'not_requested'}`

| batch N | max abs. error | mean abs. error | max relative error | action agreement | passed |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 0.000162124634 | 3.70393197e-05 | 6.66105871e-05 | 1.000000 | True |
| 4 | 0.000162124634 | 3.70393197e-05 | 6.66105871e-05 | 1.000000 | True |

disagreement sample ids: `[]`

### cuda

- requested: `CUDAExecutionProvider`
- available: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- active: `['CUDAExecutionProvider', 'CPUExecutionProvider']`
- actual primary: `CUDAExecutionProvider`
- ONNX Runtime: `1.29.0`
- GPU model: `NVIDIA GeForce RTX 4060 Laptop GPU`
- CUDA / cuDNN: `13.0` / `92000`
- graph assignment: `{'status': 'verified', 'node_count': 16, 'nodes_by_provider': {'CUDAExecutionProvider': 16}, 'fallback_node_count': 0}`

| batch N | max abs. error | mean abs. error | max relative error | action agreement | passed |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 0.000161886215 | 3.69499127e-05 | 6.64916021e-05 | 1.000000 | True |
| 4 | 0.000161886215 | 3.69389852e-05 | 6.65116115e-05 | 1.000000 | True |

disagreement sample ids: `[]`

## How to read the two correctness metrics

數值 parity 衡量的是每個 action 的 Q-value 差多少；action agreement 衡量的是把四個 Q-values 做 argmax 後，最後選出的 action index 是否相同。前者能抓出輸出數值被改寫，後者能抓出即使誤差很小、卻剛好跨過兩個 action 的決策邊界。兩者都通過，才表示這批 state 在數值和決策兩層都保持一致。

top-2 Q margin 是最大與次大的 Q-value 差距；margin 越小，越容易被浮點誤差 推過決策邊界。它是解讀 action agreement 的診斷資訊，不是替代 agreement 的通行證。

## Reproduction

```powershell
python -m scripts.analysis.compare_pytorch_onnx --provider both --config configs/inference/parity_validation.json --output assets/day23/onnx-runtime-parity.json --report reports/day23-onnx-runtime-parity.md
```

這個結果只證明固定 probe states 上的 native CPU/CUDA ORT parity；它不等於 瀏覽器 WebAssembly/WebGPU parity，也不等於完整遊戲分數相同。下一步才是把 同一個 input/action contract 帶進瀏覽器 runtime。
