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

其他 transitive dependencies 的版本會記錄在 [`environment.lock.yml`](../environment.lock.yml)。

## 新增套件時的規則

新增任何套件後，必須同步更新：

1. `environment.yml`：記錄可重建的直接依賴。
2. `environment.lock.yml`：記錄實際安裝版本。
3. 本文件的「目前使用的直接套件」表格：說明套件用途。

更新 lock snapshot：

```powershell
conda env export --name breakout-rl-engineering --no-builds > environment.lock.yml
```

