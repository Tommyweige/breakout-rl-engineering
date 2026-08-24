# Environment Setup

本專案使用 Conda 管理 Python 環境。

## Conda environment

```text
Environment name: breakout-rl-engineering
Python: 3.12.13
Platform: Windows win-64
```

## 建立環境

```powershell
conda env create --file environment.yml
conda activate breakout-rl-engineering
```

啟動 Atari Breakout 預覽：

```powershell
python play_breakout.py
```

## 目前使用的直接套件

| 套件 | 版本 | 用途 |
|---|---:|---|
| Python | 3.12.13 | 執行環境 |
| Gymnasium | 1.3.0 | 強化學習環境 API |
| ALE-Py | 0.12.0 | Atari Learning Environment Python binding |
| NumPy | 2.5.2 | observation 與數值運算 |
| opencv-python | 5.0.0.93 | AtariPreprocessing 的 grayscale 與 resize |
| Matplotlib | 3.11.1 | 由真實 Q-Learning trace 產生 learning curve 與 update breakdown 圖 |
| Pillow | 12.3.0 | 由真實 CNN forward 產生 Day 7 PNG evidence figure |
| PyTorch | 2.13.0+cu130 | Atari CNN feature extractor、tensor forward 與 CUDA training |

其他 transitive dependencies 的版本會記錄在 [`environment.lock.yml`](../environment.lock.yml)。

## PyTorch 與 device 驗證

Day 7 開始使用 PyTorch。CPU 仍可用於 unit tests 與 portability sanity check；Day 13 的正式 debug preset 則使用 CUDA build，讓 diagnostic training 能實際驗證 GPU optimizer updates。`environment.yml` 會從 PyTorch CUDA wheel index 安裝 `2.13.0+cu130`；如果另一台機器沒有 NVIDIA GPU，仍可用 `--device cpu` 執行測試與 fixed-batch check，但不能把 CPU run 當成 Day 13 的正式 CUDA 驗收。

```powershell
conda run --name breakout-rl-engineering python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

目前開發環境的驗證結果是：

```text
torch 2.13.0+cu130
torch_cuda 13.0
cuda_available True
device NVIDIA GeForce RTX 4060 Laptop GPU
```

`inspect_cnn_dimensions.py --device auto` 會在 CUDA 可用時選 CUDA，否則選 CPU；明確指定 `--device cuda` 但 CUDA 不可用時則會報錯，不會靜默改用 CPU。

## 新增套件時的規則

新增任何套件後，必須同步更新：

1. `environment.yml`：記錄可重建的直接依賴。
2. `environment.lock.yml`：記錄實際安裝版本。
3. 本文件的「目前使用的直接套件」表格：說明套件用途。

更新 lock snapshot：

```powershell
conda env export --name breakout-rl-engineering --no-builds > environment.lock.yml
```

