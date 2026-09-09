# Day 26 TensorRT optional experiment

這份報告只評估 native NVIDIA GPU 上的 TensorRT；TensorRT 不屬於 Browser/production web serving path。
Browser 仍使用 Day 22 FP32 ONNX → ORT Web WASM/WebGPU。

- experiment status: `completed`
- fixed-state sanity: `TensorRT FP32 action agreement 100%; TensorRT FP16 action agreement 93.33%`
- multi-episode score evaluation: `completed; 30 episodes per runtime; 0 runtime failures`
- adoption status: `TensorRT FP32 passed the score-based policy`
- preflight status: `READY`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU` (compute capability `8.9`)
- TensorRT: `11.2.1.2`
- benchmark scope: `TensorRT end_to_end includes host/device transfers and output materialization; model_only starts from a prepared CUDA tensor.`

## Preflight gate

| check | observed | status |
| --- | --- | :---: |
| PyTorch CUDA | `passed` | `passed` |
| ORT CUDA provider / graph assignment | `CUDAExecutionProvider` / `verified` | `passed` |
| TensorRT Python parser | `True` / `True` | `READY` |
| Strongly typed network | `strongly_typed` / `True` | `READY` |
| TF32 policy | `False` | `READY` |
| trtexec | `not found` | informational |
| FP16 capability | `True` | `READY` |
| Driver / CUDA compatibility | `passed` / `passed` | `READY` |
| Workspace + headroom | `passed` | `READY` |

## Q-value correctness and action agreement

`action agreement` 是 Q-values 取 argmax 後的 action 一致率；`smallest margin` 是樣本中最佳與次佳 action 的最小 Q 差距，能指出最容易受數值誤差影響的決策邊界。

| candidate | max abs | mean abs | max rel | action agreement | smallest margin | passed |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| onnx_cuda_fp32 | 0.000162124634 | 3.69379918e-05 | 6.65915895e-05 | 1.000000 | 8.01086426e-05 | True |
| tensorrt_cuda_fp32 | 0.000162363052 | 3.70512406e-05 | 6.66915769e-05 | 1.000000 | 8.01086426e-05 | True |
| tensorrt_cuda_fp16 | 0.00369119644 | 0.00137813191 | 0.00140698785 | 0.933333 | 0 | False |

Direct Day 22 golden-fixture checks are stored separately so the live PyTorch CUDA reference and the saved fixture cannot be confused:

| candidate | golden action agreement | golden max abs | passed |
| --- | ---: | ---: | :---: |
| pytorch_cuda_fp32 | 1.000000 | 0 | True |
| onnx_cuda_fp32 | 1.000000 | 0.000162124634 | True |
| tensorrt_cuda_fp32 | 1.000000 | 0.000162363052 | True |
| tensorrt_cuda_fp16 | 0.933333 | 0.00369119644 | False |

## Legacy 15-episode trace diagnostic (not an adoption gate)

| candidate | action agreement | return match | length match | action disagreements | passed |
| --- | ---: | ---: | ---: | ---: | :---: |
| onnx_cuda_fp32 | 0.974377 | 0.866667 | 0.866667 | 490 | False |
| tensorrt_cuda_fp32 | 0.974377 | 0.866667 | 0.866667 | 490 | False |
| tensorrt_cuda_fp16 | 0.252621 | 0.066667 | 0.000000 | 13787 | False |

These action-trace, return-match, and length-match values are retained for
diagnosis only. The updated deployment policy uses multi-episode score
distribution rather than requiring a 100% gameplay trace match.

## Multi-episode runtime score evaluation

The formal score artifact uses 30 predeclared episode seeds shared by all four
runtimes. No runtime failures occurred. Aggregate statistics are rebuilt from
the per-episode rows in `assets/day26/runtime-score-comparison.json`.

| runtime | episodes | mean | median | std | P10 | P90 | min | max | failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch CUDA FP32 | 30 | 42.53 | 37 | 16.12 | 26.0 | 68.2 | 18 | 74 | 0 |
| ORT CUDA FP32 | 30 | 42.03 | 37 | 15.58 | 26.0 | 61.0 | 18 | 74 | 0 |
| TensorRT FP32 | 30 | 42.03 | 37 | 15.58 | 26.0 | 61.0 | 18 | 74 | 0 |
| TensorRT FP16 | 30 | 36.80 | 34 | 13.19 | 18.8 | 54.2 | 16 | 64 | 0 |

### Same-seed paired score differences

The difference is always `left runtime score - right runtime score`. Bootstrap
intervals use the deterministic seed and 10,000 resamples recorded in the JSON.

| comparison | mean | median | std | P10 | P90 | min | max | mean 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ORT FP32 - PyTorch FP32 | -0.50 | 0 | 4.61 | -0.6 | 0 | -17 | 16 | [-2.17, 1.10] |
| TensorRT FP32 - PyTorch FP32 | -0.50 | 0 | 4.61 | -0.6 | 0 | -17 | 16 | [-2.17, 1.13] |
| **TensorRT FP32 - ORT FP32** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **[0, 0]** |
| TensorRT FP16 - PyTorch FP32 | -5.73 | -3 | 18.94 | -30.7 | 14.3 | -48 | 25 | [-12.83, 1.00] |

TensorRT FP32 matches ORT FP32 on all 30 paired scores, while its batch-1
end-to-end P95 is 0.760555 ms versus 1.109590 ms for ORT FP32. The formal
recommendation is therefore **native deployment-worthy** for TensorRT FP32;
TensorRT FP16 remains **native not worthwhile** because its score distribution
and fixed-state sanity check are weaker.

## Batch latency

P50/P95 使用同一批 prepared inputs 與 Day 24 的 `model_only`、`end_to_end` 定義；本次 TensorRT 的兩個 scope 都在 `prepare_model_input` 後使用 `execute_prevalidated`，end-to-end 仍包含 uint8→float32 normalization、CPU→GPU transfer、engine execution、GPU→CPU output materialization 與 argmax。

### Model-only (batch=1)

| target | P50 ms | P95 ms |
| --- | ---: | ---: |
| PyTorch CUDA FP32 | 0.549850 | 0.583255 |
| ORT CUDA FP32 | 0.566050 | 0.678905 |
| TensorRT FP32 | 0.185400 | 0.197995 |
| TensorRT FP16 | 0.280300 | 0.316040 |

### End-to-end (batch=1)

| target | engine/model size | batch=1 P50 ms | batch=1 P95 ms | vs ORT FP32 P95 |
| --- | ---: | ---: | ---: | ---: |
| PyTorch CUDA FP32 | 13,179,473 bytes | 0.910550 | 1.080505 | 2.62% |
| ORT CUDA FP32 | runtime | 0.667050 | 1.109590 | 0.00% |
| TensorRT FP32 | 13,392,500 bytes | 0.633900 | 0.760555 | 31.46% |
| TensorRT FP16 | 6,727,508 bytes | 0.994000 | 1.162205 | -4.74% |

## Decision

- TensorRT FP32: **native deployment-worthy**
- TensorRT FP16: **native not worthwhile**
- reason: TensorRT FP32 passed the fixed-state sanity check, matched ORT FP32's 30-episode score distribution and paired scores, improved batch=1 end-to-end P95 by about 31%, and its additional native maintenance cost is explicitly accepted for this scope. Gameplay trace equality is not an adoption gate.
- Browser baseline remains `float32`; `tensorrt_is_web_serving` is `False`.

### Portability and maintenance limits

Serialized TensorRT engines are hardware/software specific. The engine metadata records the GPU, compute capability, CUDA, TensorRT version, source ONNX hash, optimization profile, workspace budget, and exact preflight hash. A different GPU or software stack must rebuild and revalidate the engine.

TensorRT 11 的強型別 network 讓 FP16 precision 由 source ONNX tensor types 決定；因此 FP16 engine 使用 Day 25 的 derived typed ONNX，而不是假設仍存在舊版 builder FP16 flag。

### Installation complexity

本次 base environment 沒有被修改；TensorRT 需要額外的 isolated optional environment/site-packages，且 `trtexec` 不在 PATH，因此正式 parity 與 Python end-to-end benchmark 依賴 TensorRT Python API。這是額外的安裝與維護成本，不應被模型大小的下降掩蓋。

Official references:
- https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html
- https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html

The decision is deliberately native-only; it does not authorize uploading an engine or adding a TensorRT server to the web demo.
