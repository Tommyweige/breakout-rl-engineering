# Experiments

This directory stores **committed, controlled experiment records** that are useful for reproducing engineering decisions or model comparisons.

It is intentionally different from `runs/`:

- `runs/` is disposable local output and is ignored by default;
- `experiments/` contains selected experiment results worth preserving;
- `evaluations/` contains fixed-protocol policy evaluation results;
- `assets/` contains curated evidence that is directly used by the 30-day articles;
- `reports/` contains summaries derived from experiment/evaluation evidence.

## Naming

Prefer names that make the purpose of the run obvious:

```text
day14-batch-size-profiling-final/
day16-vectorized-backend-comparison/
day20-dqn-family/
```

Avoid new folders whose only distinction is an unexplained suffix such as `-v2`, `-v5`, `-new`, or `-final-final`. If a rerun replaces an earlier attempt, record the relationship in metadata; if both runs matter, give them names that explain why.

## What belongs here

A committed experiment should normally preserve enough information to answer:

- what question was tested;
- which config/contract defined the run;
- which code revision produced it;
- which seed(s) and training budget were used;
- which metrics or profiler outputs support the conclusion.

Large disposable checkpoints, temporary videos, caches, and one-off debug output should not be committed here unless they are necessary evidence.

## Promotion to article evidence

When an experiment produces a figure or trace that is needed by a reader-facing Day article, promote the **minimal relevant evidence** into `assets/dayXX/`. Do not duplicate an entire experiment directory just to make an article image available.
