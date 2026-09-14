# Breakout RL Engineering

Atari Breakout 的端到端強化學習工程專案：從 DQN 訓練、演算法比較，到 ONNX 與瀏覽器部署。

## Demo

直接玩：<https://breakout.tommypan.dev>

左側由玩家控制，右側由訓練好的 RL Agent 控制；遊戲與模型推論都直接在瀏覽器中執行。

## Main branch 內容

`main` 只保留目前仍值得維護與執行的核心程式：

```text
breakout_rl/   reusable RL / training / inference code
configs/       environment、training、evaluation、deployment configs
scripts/       training、evaluation、analysis、benchmark、deployment CLIs
tests/         correctness / regression tests
web/           browser demo
```

其他 30 天鐵人賽文章、圖表、歷史實驗輸出、evaluation 結果與工程報告已移出主分支，保留在：

- `archive/full-history-before-main-cleanup` — 精簡前的完整 repository snapshot

這樣 `main` 可以維持成真正的程式碼主線，不再同時充當文章倉庫與實驗資料庫。

## Python 環境

```bash
conda env create -f environment.yml
conda activate breakout-rl-engineering
```

鎖定版本保留在 `environment.lock.yml`。

## 常用入口

```bash
python -m scripts.training.train_vectorized_dqn --help
python -m scripts.evaluation.evaluate_dqn --help
```

Day 16 之後的 Breakout task semantics 以這份 contract 為準：

```text
configs/eval/breakout_contract_v2.json
```

## Web

```bash
cd web
npm install
npm run dev
```

瀏覽器端包含 ONNX Runtime Web / WebGPU 與最終 Human vs RL demo。
