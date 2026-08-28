# Day 16 — Vectorized DQN Training evidence review

## Scope

This report covers the transition-counted vectorized vanilla-DQN backend in
Issue #18. It uses the Day 15 Contract v2 environment construction: real
`ALE/Breakout-v5`, frame stack 4, frame skip 4, sticky-action probability 0.25,
environment-side FIRE, raw evaluation reward, and separate `terminated` /
`truncated` fields.

The implementation keeps DQN hyperparameters fixed within the systems A/B. It
does not introduce Double DQN, Dueling, prioritized replay, mixed precision, or
model-capacity changes.

## Implementation seams

- `breakout_rl/replay.py` and `breakout_rl/replay_gpu.py` provide ordered
  `add_batch` ring-buffer writes with wraparound and separate episode flags.
- `breakout_rl/exploration.py` provides batched Q-value action selection with
  independent per-environment random/greedy decisions.
- `breakout_env.py` provides `make_breakout_vector_env` with disabled autoreset;
  the trainer captures final observations before resetting only done envs.
- `breakout_rl/training/vectorized.py` counts accepted environment transitions,
  enumerates every crossed train/target boundary, writes per-environment
  metrics, and reuses the existing vanilla-DQN update.
- `train_vectorized_dqn.py` is the portable CLI; `benchmark_vectorized_training.py`
  and `benchmark_replay_insertion.py` produce the evidence artifacts.

## 10K systems screening

Source: `assets/day16/vectorized-training.json`.

| N | vector iterations | transitions/s | action calls | replay insert calls | optimizer updates | target syncs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,000 | 248.84 | 10,000 | 10,000 | 2,251 | 21 |
| 2 | 5,000 | 316.93 | 5,000 | 5,000 | 2,251 | 21 |
| 4 | 2,500 | 323.75 | 2,500 | 2,500 | 2,251 | 21 |
| 8 | 1,250 | 411.07 | 1,250 | 1,250 | 2,251 | 21 |

N=8 is approximately 1.65x the N=1 accepted-transition throughput under this
single-seed, single-hardware screening. The result is a systems observation,
not a claim that 8 is universally optimal.

The stage timings show why the result is plausible: the N=8 run has 1,250
batched action calls and 1,250 replay insertion calls instead of 10,000 of each.
The CPU `env_step` stage remains material, so this change does not make ALE
itself GPU-parallel.

## Replay insertion microbenchmark

Source: `assets/day16/replay-insertion.json`.

| batch size | transitions/s | latency/call |
|---:|---:|---:|
| 1 | 5,483.26 | 0.182 ms |
| 2 | 8,949.64 | 0.223 ms |
| 4 | 19,713.95 | 0.203 ms |
| 8 | 30,769.89 | 0.260 ms |
| 16 | 46,454.18 | 0.344 ms |

The source observation was produced by a real Breakout reset/step; repeated
rows were used only to measure storage copy cost. This benchmark does not
measure policy quality.

## Contract v2 guardrail

Source: `assets/day16/evaluation-summary.json`.

| Candidate | mean return | median | std | mean length | terminated | truncated |
|---|---:|---:|---:|---:|---:|---:|
| N=1 | 2.80 | 2.00 | 2.74 | 2,030.67 | 14/15 | 1/15 |
| N=8 | 2.67 | 2.00 | 2.44 | 2,033.33 | 14/15 | 1/15 |

The same termination profile and small mean-return difference do not show an
obvious 10K learning regression. Both candidates still have one TimeLimit
truncation in 15 episodes, so this is a guardrail, not a final model-quality
claim.

## Schedule and reproducibility boundary

All candidates used `total_transitions=10000`, `batch_size=32`,
`learning_starts=1000`, `train_frequency=4`, `target_update_interval=500`,
`learning_rate=0.0001`, `epsilon_decay_steps=10000`, float32, seed 42, and GPU
Replay on the recorded RTX 4060 Laptop GPU. The four runs each produced 2,251
optimizer updates and 21 target synchronizations including the initial sync.

The trainer splits replay insertion into transition chunks when a vector step
crosses a train, target, or checkpoint boundary. Boundary counts and event order
are therefore exact, while the vectorized trace is still not bit-for-bit
identical to a single-environment trace because N actions are selected together.
The report treats this as a declared batching boundary, not hidden equivalence.

## Limitations and next step

This is a 10K single-seed screening, not the later multi-seed model comparison.
It does not establish that N=8 is best on another GPU, CPU thread setting, or
longer training budget. The next experiment can reuse this backend to compare
vanilla DQN and Double DQN while keeping the transition budget and evaluation
contract visible.
