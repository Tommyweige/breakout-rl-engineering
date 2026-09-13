# 30-Day Article Index

`docs/dayXX-*.md` is the reader-facing Breakout RL Engineering series. The articles explain the learning and engineering story; implementation details, reproduction commands, raw outputs, and maintenance notes belong in code, `assets/`, `experiments/`, `evaluations/`, or `reports/`.

> Repository layout and artifact boundaries: [`PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md)

## Phase 1 — Environment and RL foundations

- Day 01 — [Project introduction and roadmap](day01-project-introduction.md)
- Day 02 — [Atari Breakout, ALE, and Gymnasium](day02-breakout-ale-gymnasium.md)
- Day 03 — [State, Action, Reward, and transition data](day03-state-action-reward-data.md)
- Day 04 — [Atari preprocessing and frame stacking](day04-atari-preprocessing-frame-stacking.md)
- Day 05 — [MDP, Return, and Bellman Equation](day05-mdp-bellman-equation.md)
- Day 06 — [Q-Learning to Deep Q-Learning](day06-q-learning-to-deep-q-learning.md)

## Phase 2 — Build and debug DQN

- Day 07 — [CNN and tensor dimensions](day07-cnn-and-tensor-dimensions.md)
- Day 08 — [DQN network](day08-dqn-network.md)
- Day 09 — [Experience Replay](day09-experience-replay.md)
- Day 10 — [Exploration vs. exploitation](day10-exploration-vs-exploitation.md)
- Day 11 — [Target Network](day11-target-network.md)
- Day 12 — [Complete DQN training loop](day12-complete-dqn-training-loop.md)
- Day 13 — [Debugging unstable RL training](day13-debugging-unstable-rl-training.md)
- Day 14 — [Hyperparameter experiments](day14-hyperparameter-experiments.md)
- Day 15 — [DQN milestone and evaluation](day15-dqn-milestone-and-evaluation.md)

## Phase 3 — Training systems and DQN variants

- Day 16 — [Vectorized DQN training](day16-vectorized-dqn-training.md)
- Day 17 — [Q-value overestimation and Double DQN](day17-q-overestimation-and-double-dqn.md)
- Day 18 — [DQN vs. Double DQN：從 100K 到 500K 的中程公平比較](day18-dqn-vs-double-dqn.md)
- Day 19 — [Dueling Network Architecture](day19-dueling-network-architecture.md)
- Day 20 — [DQN Family Comparison](day20-dqn-family-comparison.md)
- Day 21 — [Final long training](day21-final-long-training.md)

## Phase 4 — Export and inference optimization

- Day 22 — [PyTorch to ONNX](day22-pytorch-to-onnx.md)
- Day 23 — [ONNX Runtime inference](day23-onnx-runtime-inference.md)
- Day 24 — [Correct inference benchmarking](day24-correct-inference-benchmarking.md)
- Day 25 — [FP32 vs. FP16 precision](day25-fp32-vs-fp16.md)
- Day 26 — [TensorRT optional optimization experiment](day26-tensorrt-optimization-experiment.md)

## Phase 5 — Browser deployment and final product

- Day 27 — [ONNX Runtime Web](day27-onnx-runtime-web.md)
- Day 28 — [WebGPU browser inference](day28-webgpu-browser-inference.md)
- Day 29 — [Human vs RL dual Breakout browser demo](day29-interactive-browser-demo.md)
- Day 30 — [From trained policy to a playable Browser product](day30-final-evaluation-and-engineering-review.md)

## Evidence and implementation map

| Need | Location |
| --- | --- |
| Reusable RL/runtime implementation | [`../breakout_rl/`](../breakout_rl/) |
| Executable training/evaluation/analysis tools | [`../scripts/`](../scripts/) |
| Canonical environment and experiment settings | [`../configs/`](../configs/) |
| Curated article evidence | [`../assets/dayXX/`](../assets/) |
| Controlled experiment records | [`../experiments/`](../experiments/) |
| Fixed-protocol policy evaluations | [`../evaluations/`](../evaluations/) |
| Derived engineering reports | [`../reports/`](../reports/) |
| Browser application | [`../web/`](../web/) |

Article-writing and visualization rules are defined in [`AGENTS.md`](../AGENTS.md).
