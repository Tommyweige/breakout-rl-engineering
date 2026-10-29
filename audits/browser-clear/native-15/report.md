# Issue 25: predictive vision controller evaluation

- Evaluation: `breakout-vision-controller-v1-frame-skip-1`
- Environment contract: `breakout-vision-controller-v1-frame-skip-1` (task semantics `breakout-evaluation-v3-frame-skip-1`)
- Seeds: `101, 102, 103, 104, 105, 202, 203, 204, 205, 206, 303, 304, 305, 306, 307`
- Episodes: 15
- Observation: raw ALE RGB pixels (210, 160, 3); one ALE frame per controller action
- Training seed: none; all runs use the same deterministic controller configuration.

Canonical clear detections are events from the audited `BreakoutCompletionDetector`. They do not establish repository `verified_clears`, which requires separate contract/source/provenance validation.

`requested_action` is the controller decision; `ale_input_action` is the action sent to ALE after FIRE wrapper overrides. With sticky-action probability 0.25, ALE may repeat the previous action; the physical action resolved by ALE is not observed.

## Results

| Metric | Result |
|---|---:|
| Mean raw score | 721.800 |
| Median raw score | 810.000 |
| Maximum raw score | 864.000 |
| Canonical clear detections | 4 / 15 |
| Canonical clear detection rate | 26.667% |
| Best clear ALE frame | 22692 |
| Median clear ALE frame | 24702 |
| Mean lives remaining at clear | 1.500 |
| Ball detection success | 77.175% |
| Lost-ball observations | 88514 |
| Paddle detection success | 100.000% |
| Intercept predictions | 155898 |
| Requested actions | `{"FIRE": 0, "LEFT": 32472, "NOOP": 323185, "RIGHT": 32133}` |
| ALE input actions (after FIRE wrapper) | `{"FIRE": 173, "LEFT": 32472, "NOOP": 323012, "RIGHT": 32133}` |
| LEFT↔RIGHT reversals / rate | 8855 / 2.283% |
| NOOP rate | 83.340% |
| Controller decision latency, CPU mean / p95 | 2.0303 / 2.1213 ms |
| Evaluation throughput | 436.0 agent steps/s |

## Episode outcomes

| Seed | Score | Clear | ALE frames | Lives at clear | Ball detect | Predictions | Failure reason |
|---:|---:|:---:|---:|---:|---:|---:|---|
| 101 | 859.0 | no | n/a | n/a | 78.6% | 9685 | game_over |
| 102 | 537.0 | no | n/a | n/a | 81.1% | 7455 | game_over |
| 103 | 828.0 | no | n/a | n/a | 75.2% | 9308 | game_over |
| 104 | 565.0 | no | n/a | n/a | 78.9% | 8859 | game_over |
| 105 | 810.0 | no | n/a | n/a | 80.1% | 8089 | game_over |
| 202 | 864.0 | yes | 23173 | 2 | 73.6% | 18957 | canonical_clear_detected |
| 203 | 425.0 | no | n/a | n/a | 80.2% | 5059 | game_over |
| 204 | 849.0 | no | n/a | n/a | 77.4% | 11760 | game_over |
| 205 | 750.0 | no | n/a | n/a | 78.8% | 10952 | game_over |
| 206 | 441.0 | no | n/a | n/a | 79.6% | 5860 | game_over |
| 303 | 519.0 | no | n/a | n/a | 83.5% | 5629 | game_over |
| 304 | 864.0 | yes | 22692 | 1 | 74.3% | 16348 | canonical_clear_detected |
| 305 | 788.0 | no | n/a | n/a | 82.3% | 7397 | game_over |
| 306 | 864.0 | yes | 40703 | 1 | 74.6% | 16166 | canonical_clear_detected |
| 307 | 864.0 | yes | 26231 | 2 | 75.9% | 14374 | canonical_clear_detected |

## RL baseline comparison

The strongest published RL checkpoint in the repository's experiment record is the Dueling Double DQN trained with seed 2022 and selected at 2.5M environment transitions. Its reported model-selection evaluation mean was 51.4 over 15 episodes; the later 15-episode holdout mean was 30.933 (median 32). Source: [Day 21 final long training](https://github.com/Tommyweige/breakout-rl-engineering/blob/docs-ironman-series/articles/day21-final-long-training.md).

| Metric | Predictive controller | Dueling Double DQN | Comparability |
|---|---:|---:|---|
| Training required / compute | no / 0 transitions | yes / 2.5M transitions | comparable |
| Mean raw score | 721.800 (this run) | 51.4 selected eval; 30.933 holdout | not directly comparable: RL results use Contract v2 (frame skip 4), controller uses v3 task semantics (frame skip 1) and different seed roles |
| Maximum score | 864.000 (this run) | not reported in the source summary | unavailable |
| Canonical clear detections / clear rate | 4 / 0.267 | not recorded with the canonical clear detector | unavailable |
| Best clear ALE frame | 22692 | not recorded | unavailable |
| Controller / inference cost | CPU mean 2.0303 ms per screen | not benchmarked here on the same machine and evaluation harness | no matched benchmark |

The score figures are kept as historical context. They do not support a ranking because the older RL evaluation used frame skip 4 and selected/holdout seed sets, while this controller uses one ALE frame per decision and the fixed v3 concrete episode seeds. No DQN checkpoint or compatible v3 RL evaluation artifact is present in this checkout, so the report does not infer an RL clear rate or matched inference cost.

This run recorded 4 / 15 canonical clear detections. The controller collected points reliably enough to show that pixel perception and receding-horizon movement work, but these results do not establish that learning is unnecessary for the full-clear objective. The next controller experiment should target paddle-contact angles and brick-lane creation, following the issue's proposed path.

## Artifacts and provenance

- `results.json`: protocol, per-episode outcomes, aggregate metrics, detector support, and source provenance.
- `episodes.csv`: one row per fixed episode.
- `diagnostics.json`: perception/control metrics and preserved trace file names.
- `controller_trace_seed_<seed>.csv`: first real ALE episode's per-step pixels-derived state, `requested_action` / `ale_input_action`, score, lives, and external clear outcome.
- Canonical completion detector: `ale-breakout-two-wall-score-864-v2`; supported=True; clear threshold=864.
