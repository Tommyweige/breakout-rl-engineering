# Evaluations

This directory stores **fixed-protocol policy evaluation results**.

Evaluation answers a different question from training or benchmarking: given a frozen policy and a defined environment/evaluation contract, how well does the policy perform?

## Source of truth

For Day 16 onward, Breakout evaluation semantics come from:

```text
configs/eval/breakout_contract_v2.json
```

Do not silently change FIRE/reset semantics, sticky-action probability, frame skip/stack, reward handling, seed lists, or episode limits inside an evaluation script.

## Directory boundary

- training and systems experiments → `experiments/`
- fixed policy-quality evaluation → `evaluations/`
- article-ready evidence → `assets/dayXX/`
- derived summaries/comparisons → `reports/`
- disposable local output → `runs/`

## What a preserved evaluation should make recoverable

A useful committed evaluation should identify, directly or through referenced metadata:

- checkpoint/model identity;
- evaluation contract/version;
- concrete seeds;
- episode count and termination semantics;
- raw scores and aggregate statistics;
- code/config provenance when available.

Historical Day 15 Contract v1 evidence remains valid historical evidence, but it should not be presented as directly comparable with Contract v2 evaluation without an explicit caveat.
