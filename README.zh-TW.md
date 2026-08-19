# Breakout RL Engineering

[Day 1 文章](README.md) | **繁體中文專案總覽**

## 開發環境

請參閱[環境設定](docs/environment.md)，了解 Conda 環境建立方式與可重現的套件版本紀錄。

> 從 DQN 到部署：打造、評估、最佳化並部署一個能夠遊玩 Atari Breakout 的強化學習代理人。

## 專案概述

這是一個以 **Atari Breakout** 為核心的端到端強化學習工程專案。

本專案的目標不只是訓練出一個能遊玩 Breakout 的代理人，更要探索完整的 AI 模型生命週期：

**環境 → 資料收集 → 訓練 → 評估 → 最佳化 → 模型匯出 → 推論效能基準測試 → Web 部署**

本專案將作為 **2026 iThome 鐵人賽** 30 天技術系列的一部分持續開發。

## 專案目標

本專案預計：

1. 使用 PyTorch 從零實作 Deep Q-Network（DQN）代理人。
2. 理解並實作深度強化學習的核心元件。
3. 比較多種 DQN 變體：
   * DQN
   * Double DQN
   * Dueling Double DQN
4. 分析訓練穩定性與 Q-value 行為。
5. 使用多個隨機種子進行可重現的實驗。
6. 將 PyTorch policy 匯出為 ONNX。
7. 比較不同推論執行環境的效能。
8. 評估 FP32 與 FP16 推論。
9. 在適合的情況下探索 TensorRT 最佳化。
10. 使用 ONNX Runtime Web / WebGPU 將訓練好的代理人部署到瀏覽器。
11. 同時評估模型品質與部署效能。

---

## 執行環境

主要使用的環境：

```text
ALE/Breakout-v5
```

透過以下工具提供：

* Gymnasium
* Arcade Learning Environment（ALE）

本專案不會自行建立一個客製化的 Breakout clone，而是使用標準化的 Atari 環境，讓實驗更容易重現，也更方便與既有強化學習研究比較。

### Observation Pipeline

原始 Atari 畫面會經過以下預處理：

```text
ALE/Breakout-v5
210 × 160 × 3 RGB / uint8
      ↓
AtariPreprocessing
  - Frame Skip = 4
  - Max Pooling
  - Grayscale
  - Resize = 84 × 84
      ↓
84 × 84 / uint8
      ↓
FrameStackObservation(stack_size=4)
      ↓
4 × 84 × 84 / uint8
```

底層 ALE environment 使用 `frameskip=1`，由 `AtariPreprocessing` 統一負責 frame skipping，避免重複 skip。Frame stacking 則讓代理人能夠從連續畫面推斷球的移動方向與速度等資訊。

---

## 訓練資料從哪裡來？

與監督式學習不同，本專案不使用固定且已標註的資料集。

訓練樣本是代理人與環境互動時動態產生的。

每次互動會產生一筆 transition：

```text
(state, action, reward, next_state, terminated, truncated)
```

這些 transition 會被存放到 **Experience Replay Buffer**，並在訓練時抽樣使用。

```text
Agent
  ↓
Action
  ↓
ALE/Breakout
  ↓
Reward + Next State + Episode Status
  ↓
Replay Buffer
  ↓
Neural Network Training
```

---

## 基準獎勵策略

初始實驗會使用標準 Atari reward，而不是一開始就加入自訂 reward shaping。

訓練時會評估以下 reward clipping 方式：

```text
正 reward → +1
零 reward →  0
負 reward → -1
```

評估時則使用原始 Atari 遊戲分數。

自訂 reward shaping 只會作為受控實驗加入，確保 baseline 仍然具備可比較性與可重現性。

---

## 模型

### DQN

```text
State
 ↓
CNN
 ↓
Q(NOOP)
Q(FIRE)
Q(RIGHT)
Q(LEFT)
 ↓
argmax
 ↓
Action
```

### Double DQN

Double DQN 會將 action selection 與 target evaluation 分離，用來研究 Q-value overestimation 問題。

### Dueling Double DQN

此網路會進一步將狀態價值與動作優勢分離：

```text
          ┌─ Value V(s)
Features ─┤
          └─ Advantage A(s, a)
                ↓
              Q(s, a)
```

---

## 訓練策略

訓練會分成多個階段，而不是一開始就直接執行非常長的實驗。

### Development

```text
10K–50K steps
```

用於 smoke test、除錯與確認整個 pipeline 可以正常運作。

### Pilot Experiments

```text
100K–1M steps
```

用於確認代理人確實開始學習，以及訓練指標是否合理。

### Model Comparison

目標：

```text
3M steps × multiple seeds
```

用來比較：

* DQN
* Double DQN
* Dueling Double DQN

### Final Training

根據實際測得的訓練速度，最佳設定可能會延長至約：

```text
10M environment steps
```

---

## 訓練加速

訓練時不會顯示人類觀看用的遊戲畫面。

可能採用的最佳化方式包括：

* Headless Atari 執行
* Frame skipping
* Vectorized Atari environments
* Batched GPU inference
* 使用高效率的 `uint8` replay buffer 儲存方式
* CPU environment 與 GPU training overlap
* 依據 profiling 結果進行最佳化

平行環境數量會根據實際測得的 **steps per second（SPS）** 選擇。

候選設定：

```text
1 environment
2 environments
4 environments
8 environments
```

---

## 評估指標

訓練會追蹤以下指標：

```text
Episode Return
Average Return
TD Loss
Q-value Mean
Q-value Maximum
Gradient Norm
Epsilon
Steps Per Second
```

在可行的情況下，模型比較會使用多個隨機種子進行。

---

## 部署流程

最終訓練好的 policy 會經過以下 AI engineering pipeline：

```text
PyTorch
   ↓
ONNX
   ↓
ONNX Runtime
   ├── CPU
   ├── CUDA
   └── Web
```

可選的 NVIDIA 部署實驗：

```text
ONNX
 ↓
TensorRT
 ↓
FP32 / FP16 inference
```

---

## 推論效能基準測試

可能比較的執行環境包括：

```text
PyTorch CPU
PyTorch CUDA
ONNX Runtime CPU
ONNX Runtime CUDA
TensorRT FP32
TensorRT FP16
ONNX Runtime Web WASM
ONNX Runtime Web WebGPU
```

測量項目包括：

* 平均延遲
* P50 latency
* P95 latency
* Throughput
* Model size
* GPU memory usage

---

## 部署一致性

如果最佳化後的模型行為不再正確，效能提升就沒有意義。

因此，本專案會使用以下指標比較不同 runtime 的輸出：

```text
Numerical Error
Action Agreement Rate
Average Episode Return
```

例如：

```text
PyTorch FP32
      vs
ONNX Runtime FP32
      vs
FP16
      vs
TensorRT
      vs
ONNX Runtime Web
```

---

## Web Demo

最終目標是建立一個互動式瀏覽器 demo，讓訓練好的強化學習代理人能直接遊玩 Breakout。

預計架構：

```text
Browser
   │
   ├── Atari / Breakout Environment
   │
   ├── Frame Preprocessing
   │
   ├── ONNX Runtime Web
   │
   ├── WASM / WebGPU
   │
   └── Interactive UI
```

介面可能顯示：

```text
Current Model
Current Action
Q Values
Episode Reward
Inference Latency
Backend
```

未來的 demo 也可能允許使用者在 AI 控制與人類控制之間切換。

---

## 30 天開發路線

### Phase 1 — 環境與強化學習基礎

* Day 1 — [專案動機與開發路線](README.md)
* Day 2 — [Atari Breakout、ALE 與 Gymnasium](docs/day02-breakout-ale-gymnasium.md)
* Day 3 — [State、Action、Reward 與強化學習產生的資料](docs/day03-state-action-reward-data.md)
* Day 4 — [Atari 預處理與 frame stacking](docs/day04-atari-preprocessing-frame-stacking.md)
* Day 5 — [MDP、Return 與 Bellman Equation](docs/day05-mdp-bellman-equation.md)
* Day 6 — [從 Bellman Equation 到 Q-Learning，再理解為什麼 Breakout 需要 Deep Q-Learning](docs/day06-q-learning-to-deep-q-learning.md)

### Phase 2 — 建立 DQN

* Day 7 — [CNN 架構與 tensor 維度](docs/day07-cnn-and-tensor-dimensions.md)
* Day 8 — 實作 DQN network
* Day 9 — Experience Replay
* Day 10 — Exploration vs. Exploitation
* Day 11 — Target Network
* Day 12 — 完整 DQN training loop
* Day 13 — 除錯不穩定的 RL 訓練
* Day 14 — Hyperparameter experiments
* Day 15 — DQN milestone 與 evaluation

### Phase 3 — 改進代理人

* Day 16 — Q-value overestimation
* Day 17 — Double DQN
* Day 18 — DQN vs. Double DQN
* Day 19 — Dueling Network Architecture
* Day 20 — 完整 DQN family comparison

### Phase 4 — AI Engineering

* Day 21 — 設計 inference contract
* Day 22 — PyTorch to ONNX
* Day 23 — ONNX Runtime inference
* Day 24 — 正確的 inference benchmarking
* Day 25 — FP32 vs. FP16

### Phase 5 — 部署

* Day 26 — TensorRT optimization experiment
* Day 27 — ONNX Runtime Web
* Day 28 — WebGPU inference
* Day 29 — 互動式瀏覽器 demo
* Day 30 — 最終 evaluation 與工程回顧

---

## 預計使用的技術棧

### Training

* Python
* PyTorch
* Gymnasium
* Arcade Learning Environment
* NumPy

### Experimentation

* TensorBoard
* Pandas
* Matplotlib

### Model Deployment

* ONNX
* ONNX Runtime
* NVIDIA TensorRT

### Web

* TypeScript / JavaScript
* ONNX Runtime Web
* WebGPU

---

## 專案狀態

🚧 **規劃中／初始開發階段**

目前 repository 正在初始化，之後會在 30 天計畫中逐步加入實作與實驗結果。

---

## 參考資料

本專案主要參考以下強化學習研究：

* *Playing Atari with Deep Reinforcement Learning* — Mnih et al.
* *Human-level control through deep reinforcement learning* — Mnih et al.
* *Deep Reinforcement Learning with Double Q-learning* — van Hasselt et al.
* *Dueling Network Architectures for Deep Reinforcement Learning* — Wang et al.

其他實作與工程相關參考資料，會隨專案進度持續補充。
