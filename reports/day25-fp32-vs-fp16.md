# Day 25 FP32 vs FP16 precision report

這份 report 只回答 native NVIDIA CUDA 上的 inference optimization 問題：
把 Day 21 Final Model 的 Day 22 FP32 ONNX baseline 改成內部 FP16 後，
Q-values、greedy actions、固定 Contract v2 evaluation 與 batch=1 latency 是否仍可接受？

- experiment status: `completed`
- precision validation: `failed`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- CUDA device: `0`
- headroom requirement: `1073741824` bytes

## Direct FP16 precision parity

| precision pair | max absolute error | mean absolute error | action agreement | passed |
| --- | ---: | ---: | ---: | :---: |
| pytorch_cuda_fp16_vs_pytorch_cuda_fp32 | 0.00279784203 | 0.000733970602 | 0.933333 | False |
| onnx_cuda_fp16_vs_onnx_cuda_fp32 | 0.00278425217 | 0.000711498658 | 0.933333 | False |

ORT FP32 對 PyTorch CUDA FP32 的 provider baseline 仍保存在 comparison JSON；上表則只回答同一 runtime 內 FP32 到 FP16 的精度變化。

`action agreement` 是四個 Q-values 做 argmax 後選到同一個 action 的比例；它和數值誤差一起看，避免只因誤差很小就忽略決策邊界。

## Fixed-seed evaluation parity

| target | action agreement | return match | length match | passed |
| --- | ---: | ---: | ---: | :---: |
| pytorch_cuda_fp16 | 0.255998 | 0.000000 | 0.000000 | False |
| onnx_cuda_fp16 | 0.258318 | 0.000000 | 0.000000 | False |

這裡的 evaluation parity 是相同固定 seed 下的 requested action trace、episode return 與 episode length 比較；它不是把新的遊戲分數硬套成數值相等的理論保證。

## Batch=1 end-to-end latency

| runtime | FP32 P50 | FP16 P50 | P50 change | FP32 P95 | FP16 P95 | P95 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pytorch | 7.917300 ms | 9.267950 ms | -17.06% | 8.978065 ms | 10.689750 ms | -19.07% |
| onnx | 4.139850 ms | 5.341950 ms | -29.04% | 5.541010 ms | 6.726770 ms | -21.40% |

ONNX model size: FP32 `13175034` bytes; FP16 `6590582` bytes; reduction `49.98%`.

## Decision

- PyTorch CUDA: `not recommended: correctness threshold failed`
- ONNX Runtime CUDA: `not recommended: correctness threshold failed`
- browser baseline precision: `float32`
- reason: FP16 cannot be recommended when numeric, action, or fixed-seed evaluation parity fails.

## Reproduction

```powershell
python -m scripts.analysis.compare_precision --config configs/inference/precision_validation.json --device-index 0 --force
```
The result is native CUDA evidence for this model and machine. It does not establish ORT Web FP16 parity or justify changing the Cloudflare Browser baseline.
