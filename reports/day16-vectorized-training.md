# Day 16 — Finalization evidence review

## Scope

This report covers the transition-counted vectorized vanilla-DQN backend after
the Issue #18 finalization gate. It uses the committed Contract v2 environment:
real `ALE/Breakout-v5`, frame skip 4, frame stack 4, sticky-action probability
0.25, environment-owned serve FIRE, raw evaluation reward, and separate
`terminated` / `truncated` signals.

The systems comparison keeps the DQN update, GPU Replay, batch size, seed,
learning schedule, reward handling, precision, and CPU thread setting fixed.
It does not introduce Double DQN, Dueling, prioritized replay, or mixed
precision.

## Correctness gates

`BreakoutFireResetWrapper` keeps a pending serve state until an observable raw
reward or two consecutive observations with at least 0.0001 changed pixels
follow the wrapper-resolved FIRE. It retries FIRE instead of silently releasing
the state after one call, and raises after a bounded eight attempts if serving
cannot be confirmed. The wrapper records the
policy request, action passed to the lower environment, confirmation signal,
attempt number, life count, and life-loss transition. ALE's hidden sticky-action
draw is not exposed and is not inferred as if it were observable.

The formal evaluator now stores both requested and executed/wrapper-resolved
action distributions. The historical `action_distribution` field is retained
with the explicit meaning `executed/wrapper-resolved action`; named provenance
fields, auto-FIRE counts, and auto-FIRE reason counts are present in JSON and
CSV artifacts.

The action-selection parity rule is:

```text
num_envs <= train_frequency
and train_frequency % num_envs == 0
```

This ensures a batched action selection does not span an optimizer boundary.
The trainer supports screening outside this rule, but emits a warning and
records that later transitions in a crossing vector batch use the pre-update
online-network snapshot. A crafted test changes the preferred action at the
update boundary: N=8 keeps all eight old actions in one crossing batch, while
strict N=4 selects the new action in the following vector batch.

## 10K systems screening

Source: `assets/day16/vectorized-training.json`. All four runs completed
10,000 accepted transitions, 2,251 optimizer updates, and 21 target
synchronizations on the recorded RTX 4060 Laptop GPU.

| N | vector iterations | transitions/s | action calls | replay insert calls | strict parity |
|---:|---:|---:|---:|---:|:---:|
| 1 | 10,000 | 298.20 | 10,000 | 10,000 | yes |
| 2 | 5,000 | 387.89 | 5,000 | 5,000 | yes |
| 4 | 2,500 | 456.63 | 2,500 | 2,500 | yes |
| 8 | 1,250 | 483.30 | 1,250 | 2,500 | no |

N=8 is faster in this short systems run, but it crosses the update boundary
when `train_frequency=4`. N=4 is therefore the strict-parity candidate; the
selection is a semantics decision, not a claim that its 10K checkpoint is the
best model.

## Fresh 100K validation

Source: `assets/day16/vectorized-training-100k.json`. N=1 and N=4 both start
fresh under Contract v2 and strict action-selection parity.

| N | transitions/s | wall-clock s | action calls | replay insert calls | optimizer updates | target syncs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 238.67 | 419.00 | 100,000 | 100,000 | 24,751 | 201 |
| 4 | 368.06 | 271.70 | 25,000 | 25,000 | 24,751 | 201 |

The strict N=4 candidate completes the same transition budget about 1.38× as
fast as N=1. The report also preserves stage timing, GPU power/utilization,
VRAM, CPU utilization, action-inference, replay-insertion, optimizer-update,
and training-sample rates. `SyncVectorEnv` still steps ALE environments in the
same process; the result should be attributed to batched inference and replay
insertion, not to a claim of multi-process CPU ALE parallelism.

## Replay insertion microbenchmark

Source: `assets/day16/replay-insertion.json`. The source observation came from
a real Contract v2 reset/step. The benchmark records the requested `NOOP` and
the wrapper-resolved/executed `FIRE`; repeated rows only measure storage-copy
cost.

| batch size | transitions/s | latency/call |
|---:|---:|---:|
| 1 | 4,962.87 | 0.201 ms |
| 2 | 9,831.91 | 0.203 ms |
| 4 | 17,354.98 | 0.230 ms |
| 8 | 32,488.10 | 0.246 ms |
| 16 | 48,982.96 | 0.327 ms |

The result supports the fixed-cost amortization explanation, but it is not a
policy-quality or end-to-end training measurement.

## FIRE/sticky-action diagnostic

Source: `assets/day16/fire-sticky-diagnostic.json` and its trace. The diagnostic
uses the fresh N=4 100K checkpoint, all 15 Contract v2 concrete seeds, and
CUDA inference. Every episode terminated normally: 15/15 terminated, 0/15
truncated, and 0/15 TimeLimit.

For seed 101, the old implementation reproduced a 26,998-step TimeLimit. The
final wrapper run ends at 198 steps with raw return 2.0. The initial serve has
one attempt with no activity, a second with activity that is not yet a complete
confirmation, and a third that completes the two-observation activity streak.
Each of the later four life-loss serves confirms on its second attempt. Across
all 15 seeds there are 151 environment-side FIRE attempts, 120 after life loss
and 31 at initial serve; 77 attempts are retry transitions under the explicit
two-observation confirmation rule. The trace records the action passed
downward, but does not pretend to know whether ALE's hidden sticky draw
accepted it.

## Contract v2 evaluation guardrail

Source: `assets/day16/evaluation-summary.json`. Each row contains 15 fixed
episodes, epsilon 0, raw reward, and requested/executed action provenance.

| Run | mean return | median | std | mean length | terminated | truncated |
|---|---:|---:|---:|---:|---:|---:|
| Random Contract v2 | 1.73 | 2.00 | 1.12 | 197.40 | 15/15 | 0/15 |
| N=1, 100K | 9.00 | 9.00 | 2.03 | 468.67 | 15/15 | 0/15 |
| N=4, 100K | 2.33 | 2.00 | 1.07 | 201.27 | 15/15 | 0/15 |

The 100K guardrail finds no serve deadlock or TimeLimit regression. N=4's
return is lower than N=1 in this single-seed, 15-episode sample, so the result
does not establish quality equivalence; it only clears the contract-failure
gate and keeps the quality difference visible.

## Q-value evidence boundary

The CPU toy simulation in `assets/day16/overestimation-bias.json` uses 500,000
Monte Carlo trials with four equal true action values. At noise standard
deviation 1.0, the measured vanilla maximum is 2.0294 against true value 1.0,
while the independently evaluated decoupled estimator is 1.0002. This
demonstrates why selecting and evaluating with the same noisy estimate can
create an optimistic maximum; it is not a measurement of Breakout bias.

`assets/day16/q-value-diagnostics.json` is deliberately separate: it contains
80 real Breakout probe states from the N=4 100K checkpoint, with model
inference on NVIDIA CUDA under `torch.no_grad()`, checkpoint SHA-256, and
runtime metadata. Its mean maximum Q-value is 2.0819 and mean top-action gap is
0.0387. Without a ground-truth Q-star oracle, those values are exploratory
model outputs, not proof that the checkpoint is overestimating.

## Final backend decision

The Day 16 candidate backend is N=4 with:

```text
ALE/Breakout-v5 / Contract v2
frame_skip=4, frame_stack=4, sticky_action_probability=0.25
environment-owned FIRE serve reset
GPU Replay, float32, batch_size=32
learning_starts=1000, train_frequency=4
target_update_interval=500, epsilon_decay_steps=10000
training seed=42, CPU threads=2
strict action-selection parity enabled
```

This is a systems choice supported by the 100K run's speed and absence of
contract failure, not a claim that N=4 learns better than N=1. The next entry
can study Double DQN with this backend while keeping action provenance,
Contract v2, and the distinction between toy mechanism and real model
diagnostics explicit.
