我已經確認你目前的 GitHub 帳號下 **`breakout-rl-engineering` 還不存在**。但這次 GitHub App 提供給我的權限介面只有「對既有 Repository 建檔、改檔、Issue、PR」等操作，**沒有建立新 Repository 的 API**；本機也沒有可用的 `gh` CLI，所以我不能直接替你按下「Create repository」。

你只需要先在 GitHub 建立一個：

* Repository name：`breakout-rl-engineering`
* 建議：**Public**
* **不要勾 Add a README** 也可以
* 其他先全部空白

建立完之後直接跟我說 **「建好了」**，我就能直接透過 GitHub App 把下面這份 README 寫進 `main`，你不用自己複製貼上。

# Breakout RL Engineering

> From DQN to Deployment: building, evaluating, optimizing, and deploying a reinforcement learning agent for Atari Breakout.

## Project Overview

This project is an end-to-end reinforcement learning engineering project built around **Atari Breakout**.

The goal is not only to train an agent that can play Breakout, but to explore the full lifecycle of an AI model:

**environment → data collection → training → evaluation → optimization → model export → inference benchmarking → web deployment**

The project will be developed as part of a 30-day technical series for the **2026 iThome Ironman**.

## Objectives

This project aims to:

1. Build a Deep Q-Network (DQN) agent from scratch with PyTorch.
2. Understand and implement the core components of Deep Reinforcement Learning.
3. Compare multiple DQN variants:

   * DQN
   * Double DQN
   * Dueling Double DQN
4. Analyze training stability and Q-value behavior.
5. Perform reproducible experiments using multiple random seeds.
6. Export the trained policy from PyTorch to ONNX.
7. Benchmark different inference runtimes.
8. Evaluate FP32 and FP16 inference.
9. Explore TensorRT optimization where appropriate.
10. Deploy the trained agent to the browser using ONNX Runtime Web / WebGPU.
11. Evaluate both model quality and deployment performance.

---

## Environment

The primary environment will be:

```text
ALE/Breakout-v5
```

provided through:

* Gymnasium
* Arcade Learning Environment (ALE)

Instead of creating a custom Breakout clone, this project uses the standardized Atari environment to make experiments more reproducible and comparable with existing reinforcement learning research.

### Observation Pipeline

The raw Atari frame will be preprocessed approximately as:

```text
210 × 160 RGB
      ↓
Grayscale
      ↓
Resize to 84 × 84
      ↓
Frame Skip = 4
      ↓
Stack 4 Frames
      ↓
4 × 84 × 84
```

Frame stacking allows the agent to infer motion, such as the direction and velocity of the ball.

---

## Where Does the Training Data Come From?

Unlike supervised learning, this project does not use a fixed labeled dataset.

Training samples are generated dynamically through interaction between the agent and the environment.

Each interaction produces a transition:

```text
(state, action, reward, next_state, done)
```

These transitions are stored in an **Experience Replay Buffer** and sampled during training.

```text
Agent
  ↓
Action
  ↓
ALE/Breakout
  ↓
Reward + Next State
  ↓
Replay Buffer
  ↓
Neural Network Training
```

---

## Baseline Reward Strategy

The initial experiments will use the standard Atari reward rather than custom reward shaping.

For training, reward clipping will be evaluated using:

```text
positive reward → +1
zero reward     →  0
negative reward → -1
```

Evaluation will use the original Atari game score.

Custom reward shaping will only be introduced as a controlled experiment so that the baseline remains comparable and reproducible.

---

## Models

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

Double DQN will be used to investigate Q-value overestimation by separating action selection from target evaluation.

### Dueling Double DQN

The network will additionally separate:

```text
          ┌─ Value V(s)
Features ─┤
          └─ Advantage A(s, a)
                ↓
              Q(s, a)
```

---

## Training Strategy

Training will be divided into multiple stages rather than immediately running very long experiments.

### Development

```text
10K–50K steps
```

Used for smoke testing and debugging.

### Pilot Experiments

```text
100K–1M steps
```

Used to verify that the agent can learn and that training metrics behave correctly.

### Model Comparison

Target:

```text
3M steps × multiple seeds
```

for comparing:

* DQN
* Double DQN
* Dueling Double DQN

### Final Training

The best-performing configuration may be extended to approximately:

```text
10M environment steps
```

depending on measured training throughput.

---

## Training Acceleration

Training will run without human rendering.

Potential optimizations include:

* Headless Atari execution
* Frame skipping
* Vectorized Atari environments
* Batched GPU inference
* Efficient `uint8` replay-buffer storage
* CPU environment / GPU training overlap
* Profiling-based optimization

The number of parallel environments will be selected experimentally based on measured **steps per second (SPS)**.

Candidate configurations:

```text
1 environment
2 environments
4 environments
8 environments
```

---

## Metrics

Training will track metrics including:

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

Model comparisons will use multiple random seeds where practical.

---

## Deployment Pipeline

The final trained policy will move through an AI engineering pipeline:

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

Optional NVIDIA deployment experiments:

```text
ONNX
 ↓
TensorRT
 ↓
FP32 / FP16 inference
```

---

## Inference Benchmark

Runtime comparisons may include:

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

Measurements will include:

* Mean latency
* P50 latency
* P95 latency
* Throughput
* Model size
* GPU memory usage

---

## Deployment Fidelity

Optimization is only useful if the resulting policy still behaves correctly.

Therefore the project will compare outputs across runtimes using metrics such as:

```text
Numerical Error
Action Agreement Rate
Average Episode Return
```

For example:

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

The final goal is an interactive browser demo where the trained reinforcement learning agent can play Breakout directly.

Planned architecture:

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

Potential UI information:

```text
Current Model
Current Action
Q Values
Episode Reward
Inference Latency
Backend
```

The eventual demo may also allow switching between AI and human control.

---

## 30-Day Roadmap

### Phase 1 — Environment & RL Foundations

* Day 1 — Project motivation and roadmap
* Day 2 — Atari Breakout, ALE, and Gymnasium
* Day 3 — State, Action, Reward, and RL-generated data
* Day 4 — Atari preprocessing and frame stacking
* Day 5 — MDP and Bellman Equation
* Day 6 — From Q-Learning to Deep Q-Learning

### Phase 2 — Building DQN

* Day 7 — CNN architecture and tensor dimensions
* Day 8 — Implementing the DQN network
* Day 9 — Experience Replay
* Day 10 — Exploration vs. Exploitation
* Day 11 — Target Networks
* Day 12 — Complete DQN training loop
* Day 13 — Debugging unstable RL training
* Day 14 — Hyperparameter experiments
* Day 15 — DQN milestone and evaluation

### Phase 3 — Improving the Agent

* Day 16 — Q-value overestimation
* Day 17 — Double DQN
* Day 18 — DQN vs. Double DQN
* Day 19 — Dueling Network Architecture
* Day 20 — Final DQN-family comparison

### Phase 4 — AI Engineering

* Day 21 — Designing the inference contract
* Day 22 — PyTorch to ONNX
* Day 23 — ONNX Runtime inference
* Day 24 — Proper inference benchmarking
* Day 25 — FP32 vs. FP16

### Phase 5 — Deployment

* Day 26 — TensorRT optimization experiment
* Day 27 — ONNX Runtime Web
* Day 28 — WebGPU inference
* Day 29 — Interactive browser demo
* Day 30 — Final evaluation and engineering retrospective

---

## Planned Tech Stack

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

## Project Status

🚧 **Planning / Initial Development**

The repository is currently being initialized. Implementation will be developed incrementally throughout the 30-day project.

---

## References

The project will primarily build upon the following reinforcement learning research:

* *Playing Atari with Deep Reinforcement Learning* — Mnih et al.
* *Human-level control through deep reinforcement learning* — Mnih et al.
* *Deep Reinforcement Learning with Double Q-learning* — van Hasselt et al.
* *Dueling Network Architectures for Deep Reinforcement Learning* — Wang et al.

Additional implementation and engineering references will be documented as the project progresses.
