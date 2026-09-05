# Day 22 ONNX export report

這份 report 將兩條驗證路徑分開記錄：正式 deployment candidate 的 PyTorch source-model reference 在 CUDA 上產生；ONNX graph 的 checker 與 shape inspection 在 host CPU 上完成。後者不會取代前者，也尚未執行 ONNX Runtime policy parity。

## Frozen source and contract

- status: `completed`
- source model: `double_dqn / dueling`
- source model SHA256: `6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a`
- source checkpoint SHA256: `ab07c0a48202428ddbb377c81f4091b3c434ce95e19d19fb1ec335df79841c48`
- source checkpoint step: `2500000`
- inference spec: `configs/inference/inference_spec.json`
- probe states: `assets/day22/inference/probe_states.npz` (60 states)

## CUDA source-model validation

- framework: `PyTorch` `2.13.0+cu130`
- device: `cuda:0` / `NVIDIA GeForce RTX 4060 Laptop GPU`
- CUDA: `13.0`, device index `0`, available `True`
- probe batch shape: `[60, 4, 84, 84]`
- Q-value output shape: `[60, 4]`
- source output max absolute difference before vs after export: `0.0`
- greedy actions unchanged: `True`

這裡的 `0.0` 只表示匯出前後仍使用同一份 CUDA 載入的 source model 時，reference 沒有被流程改寫；它不是 ONNX Runtime parity 結果。

## Host-side ONNX graph validation

- checker: `onnx.checker.check_model` → `True`
- input: `observation` / `float32` / `[N, 4, 84, 84]`
- output: `q_values` / `float32` / `[N, 4]`
- opset: `17`; IR version: `8`
- nodes: `16`; initializers: `14`
- operator counts: `{"Add":1,"Conv":3,"Flatten":1,"Gemm":4,"ReduceMean":1,"Relu":5,"Sub":1}`
- exported ONNX: `assets/day22/models/final_model/model.onnx` (13,175,034 bytes (12.56 MiB))

ONNX checker 驗證 graph 的格式與內部結構是否符合 ONNX 規則；它不能證明模型和 PyTorch 每個輸出相同，也不能證明瀏覽器 provider、外部 preprocessing 或遊戲表現正確。

## Dynamic batch declaration

- dimension: `N`
- declared input: `[N, 4, 84, 84]`
- declared output: `[N, 4]`
- source-model shape checks: `[1, 4]`
- ONNX Runtime execution: `deferred to Day 23`

## Artifacts

- ONNX model: `assets/day22/models/final_model/model.onnx`
- ONNX metadata: `assets/day22/models/final_model/model.onnx.metadata.json`
- CUDA PyTorch reference: `assets/day22/inference/pytorch_reference.npz`
- fixed probe states: `assets/day22/inference/probe_states.npz`
- fixture manifest: `assets/day22/inference/metadata.json`
- graph summary figure: `assets/day22/onnx-graph-summary.png`

## Reproduction

```powershell
python -m scripts.analysis.generate_probe_states --contract configs/eval/breakout_contract_v2.json --output assets/day22/inference/probe_states.npz --states-per-seed 4 --stride 32 --max-steps 256
python -m scripts.deployment.export_onnx
python -m scripts.visualization.visualize_onnx_graph_summary
python -m scripts.analysis.generate_day22_report
```

正式 export 要求 NVIDIA CUDA；CUDA unavailable 時 pipeline 應停止，而不是產生 CPU 偽 reference。
