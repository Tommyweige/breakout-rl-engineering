# Breakout RL Engineering

這是一個 30 天的強化學習工程專案，從 Atari Breakout 的環境理解一路做到 **DQN 訓練、演算法比較、ONNX 推論與瀏覽器部署**。

## 最終成果：Human vs AI Breakout

[![Human and RL Breakout running side by side](assets/day30/final-human-vs-rl.png)](https://breakout.tommypan.dev)

**直接玩：** [breakout.tommypan.dev](https://breakout.tommypan.dev)

左邊是你自己控制，右邊是訓練好的 RL Agent。兩邊的 ALE 遊戲與模型推論都直接跑在瀏覽器裡，不需要 Python 後端，也不需要連 GPU server。

## 這個 repo 在做什麼？

這不只是一個「把 DQN train 出來」的專案，而是完整記錄整條工程流程：

```text
ALE / Gymnasium
      ↓
State、Action、Reward
      ↓
DQN Training Loop
      ↓
Replay Buffer + Target Network
      ↓
Vectorized CUDA Training
      ↓
DQN / Double DQN / Dueling Double DQN
      ↓
Evaluation Contract + 可重現實驗
      ↓
PyTorch → ONNX → ONNX Runtime
      ↓
FP32 / FP16 / TensorRT
      ↓
ONNX Runtime Web / WebGPU
      ↓
可直接玩的 Browser Product
```

原本放在根 README 的 Day 1 故事與動機已完整保留在 [`docs/day01-project-introduction.md`](docs/day01-project-introduction.md)。

## 建議從這裡開始看

- **30 天文章索引：** [`docs/README.md`](docs/README.md)
- **Repo 結構與檔案放置規則：** [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)
- **Coding agent / 技術文章規則：** [`AGENTS.md`](AGENTS.md)
- **可執行工具分類：** [`scripts/README.md`](scripts/README.md)
- **Config 地圖：** [`configs/README.md`](configs/README.md)
- **最終瀏覽器 App：** [`web/`](web/)

## Repo 結構

| 路徑 | 用途 |
| --- | --- |
| [`breakout_rl/`](breakout_rl/) | 可重用的 RL、訓練、評估、推論與部署程式 |
| [`scripts/`](scripts/) | Training / Evaluation / Analysis / Benchmark / Visualization / Deployment CLI |
| [`configs/`](configs/) | 環境、訓練、實驗、推論與部署設定 |
| [`tests/`](tests/) | Regression / correctness tests |
| [`docs/`](docs/) | 30 天讀者向技術文章 |
| [`assets/`](assets/) | 文章真正使用的精選證據與圖表 |
| [`experiments/`](experiments/) | 值得保留的受控實驗紀錄 |
| [`evaluations/`](evaluations/) | 固定協議的 policy 評估結果 |
| [`reports/`](reports/) | 從實驗 / 評估證據衍生出的工程報告 |
| [`web/`](web/) | ONNX Runtime Web / WebGPU 與最終互動 Demo |
| [`design-system/`](design-system/) | Web Demo 的 UI design-system 支援 |

`breakout_env.py` 是目前刻意保留的歷史相容性例外；新的可重用 Python 程式不應再直接丟到 repo 根目錄。

## Python 環境

主要 Conda 環境：

```text
environment.yml
```

另外保留 locked snapshot：

```text
environment.lock.yml
```

一般建立方式：

```bash
conda env create -f environment.yml
conda activate breakout-rl-engineering
```

所有 script 建議從 repo root 用 module syntax 執行：

```bash
python -m scripts.training.train_vectorized_dqn --help
python -m scripts.evaluation.evaluate_dqn --help
python -m scripts.analysis.analyze_q_values --help
```

## Canonical Breakout Contract

Day 16 之後，training、evaluation、gameplay recording 與 deployment parity 都應共同使用：

```text
configs/eval/breakout_contract_v2.json
```

它固定了 frame skip / stack、sticky action、FIRE ownership、life-loss handling、evaluation seeds、reward 與 episode limit 等會直接改變 RL 任務定義的設定。

這樣 DQN、Double DQN、Dueling DQN 之間的比較才是真的公平比較，而不是偷偷換了遊戲規則。

## 30 天路線

| Phase | Day | 內容 |
| --- | ---: | --- |
| RL 基礎 | 1–6 | ALE、Gymnasium、State/Action/Reward、MDP、Q-Learning |
| 組出 DQN | 7–15 | CNN、Replay Buffer、Exploration、Target Network、Training / Debugging |
| 訓練系統與演算法 | 16–21 | Vectorized CUDA、Double DQN、Dueling DQN、長時間訓練 |
| 模型工程 | 22–26 | ONNX、ONNX Runtime、Benchmark、FP16、TensorRT |
| Browser Product | 27–30 | ONNX Runtime Web、WebGPU、互動 Demo、最終驗證 |

完整文章入口請看 [`docs/README.md`](docs/README.md)。

## Artifact 怎麼分？

這個 repo 會保存可重現證據，但不同輸出不再全部混在一起：

- 本機一次性 run → `runs/`（預設 ignore）
- 值得保留的 controlled experiment → `experiments/`
- 固定 evaluation protocol 的 policy score → `evaluations/`
- 文章真正引用的圖表 / trace → `assets/dayXX/`
- 從證據整理出的 summary → `reports/`

更完整的規則在 [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)。

## 這個專案真正想完成的事

目標不是停在：

```text
train()
↓
模型有權重
↓
結束
```

而是完整走過：

> 讓 Agent 學會 Breakout → 確認它是不是真的學會 → 公平比較演算法 → 匯出模型 → 驗證與最佳化推論 → 最後把 policy 真的做成瀏覽器裡可以玩的產品。
