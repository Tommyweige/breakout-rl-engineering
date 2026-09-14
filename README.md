# Breakout RL Engineering

Atari Breakout 的端到端強化學習工程專案：DQN 訓練、Double / Dueling DQN、CUDA 訓練、ONNX 推論，以及瀏覽器部署。

## Live demo

<https://breakout.tommypan.dev>

遊戲與訓練完成的 RL Agent 都直接在瀏覽器執行，不需要 Python backend。

## Repository

`main` 只保留目前值得維護的程式碼與最終產品：

```text
breakout_rl/   reusable RL / training / inference implementation
configs/       active training, evaluation, and inference configs
scripts/       maintained training, evaluation, analysis, benchmark CLIs
tests/         core correctness / regression tests
web/           browser demo + runtime model assets
```

30 天鐵人賽文章與對應圖表已移到 `docs-ironman-series` 分支。
精簡前的完整 repository snapshot 保留在 `archive/full-history-before-main-cleanup`。

## Python

```bash
conda env create -f environment.yml
conda activate breakout-rl-engineering

python -m scripts.training.train_vectorized_dqn --help
python -m scripts.evaluation.evaluate_dqn --help
```

Breakout 的 canonical environment / evaluation semantics：

```text
configs/eval/breakout_contract_v2.json
```

## Web

```bash
cd web
npm install
npm run dev
```

Production build：

```bash
npm run build
```
