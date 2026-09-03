# Reader-facing article index

`docs/dayXX-*.md` contains the reader-facing 30-day Breakout RL Engineering series. These files explain the engineering story; implementation details, reproduction commands, raw evidence, and maintenance notes belong in code, `assets/`, or their local README files.

## Published / implemented days

- Day 02 — [Atari Breakout, ALE, and Gymnasium](day02-breakout-ale-gymnasium.md)
- Day 03 — [State, Action, Reward, and transition data](day03-state-action-reward-data.md)
- Day 04 — [Atari preprocessing and frame stacking](day04-atari-preprocessing-frame-stacking.md)
- Day 05 — [MDP, Return, and Bellman Equation](day05-mdp-bellman-equation.md)
- Day 06 — [Q-Learning to Deep Q-Learning](day06-q-learning-to-deep-q-learning.md)
- Day 07 — [CNN and tensor dimensions](day07-cnn-and-tensor-dimensions.md)
- Day 08 — [DQN network](day08-dqn-network.md)
- Day 09 — [Experience Replay](day09-experience-replay.md)
- Day 10 — [Exploration vs. exploitation](day10-exploration-vs-exploitation.md)
- Day 11 — [Target Network](day11-target-network.md)
- Day 12 — [Complete DQN training loop](day12-complete-dqn-training-loop.md)
- Day 13 — [Debugging unstable RL training](day13-debugging-unstable-rl-training.md)
- Day 14 — [Hyperparameter experiments](day14-hyperparameter-experiments.md)
- Day 15 — [DQN milestone and evaluation](day15-dqn-milestone-and-evaluation.md)
- Day 16 — [Vectorized DQN training](day16-vectorized-dqn-training.md)
- Day 17 — [Q-value overestimation and Double DQN](day17-q-overestimation-and-double-dqn.md)
- Day 18 — [DQN vs. Double DQN：從 100K 到 500K 的中程公平比較](day18-dqn-vs-double-dqn.md)
- Day 19 — [Dueling Network Architecture](day19-dueling-network-architecture.md)
- Day 20 — [DQN Family Comparison](day20-dqn-family-comparison.md)

Day 20 onward should continue the same naming and article rules defined in `AGENTS.md`.

## Related locations

- `../assets/dayXX/` — machine-readable and rendered evidence.
- `../scripts/` — executable training, evaluation, diagnostics, benchmarks, and visualization tools.
- `../breakout_rl/` — reusable implementation.
- `../configs/` — task, backend, and experiment contracts.
