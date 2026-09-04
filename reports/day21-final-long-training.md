# Day 21 Final Long Training

這份 report 記錄 Day 20 winner 在新的 training seeds 上如何經過 1M、2.5M 與 5M actual environment transitions 的 gate，並在最後 freeze 一個 canonical model。

- manifest status: `completed`
- winner: `dueling_double_dqn` (double_dqn / dueling)
- training seeds: `[1011, 2022, 3033]`
- stage targets: `{'stage_a_1m': 1000000, 'stage_b_2_5m': 2500000, 'stage_c_5m': 5000000}` actual environment transitions
- selection evaluation: `[101, 102, 103, 104, 105, 202, 203, 204, 205, 206, 303, 304, 305, 306, 307]` concrete seeds, `5` episodes per group
- final holdout: `[404, 405, 406, 407, 408, 505, 506, 507, 508, 509, 606, 607, 608, 609, 610]` concrete seeds, opened after freeze: `True`
- evaluation order: `selection → final freeze → final holdout`

## Stage C trigger

- primary trigger: `2.5M evaluation showed substantial improvement, so 5M continuation remained justified.`
- evidence: `{'training_seed': 2022, 'stage_a_1m_mean_return': 34.86666666666667, 'stage_b_2_5m_mean_return': 51.4, 'mean_return_improvement': 16.53333333333333}`
- user_requested_5m (supplemental): `True`

## Stage evidence

| seed | stage | run id | target transitions | status | eligible | mean selection return | std return |
| ---: | --- | --- | ---: | --- | --- | ---: | ---: |
| 1011 | stage_a_1m | day21-final-long-training-seed1011 | 1000000 | completed | True | 34.000 | 12.187 |
| 1011 | stage_b_2_5m | day21-final-long-training-seed1011 | 2500000 | completed | True | 42.600 | 15.191 |
| 1011 | stage_c_5m | day21-final-long-training-seed1011 | 5000000 | not_selected | False | unavailable | unavailable |
| 2022 | stage_a_1m | day21-final-long-training-seed2022 | 1000000 | completed | True | 34.867 | 15.466 |
| 2022 | stage_b_2_5m | day21-final-long-training-seed2022 | 2500000 | completed | True | 51.400 | 14.678 |
| 2022 | stage_c_5m | day21-final-long-training-seed2022 | 5000000 | completed | True | 49.933 | 16.258 |
| 3033 | stage_a_1m | day21-final-long-training-seed3033 | 1000000 | completed | True | 33.133 | 15.200 |
| 3033 | stage_b_2_5m | day21-final-long-training-seed3033 | 2500000 | not_selected | False | unavailable | unavailable |
| 3033 | stage_c_5m | day21-final-long-training-seed3033 | 5000000 | not_selected | False | unavailable | unavailable |

## Runtime and provenance

| run id | seed | stage | resolved device | GPU | transitions | Contract v2 |
| --- | ---: | --- | --- | --- | ---: | --- |
| day21-final-long-training-seed1011 | 1011 | stage_a_1m | cuda:0 | NVIDIA GeForce RTX 4060 Laptop GPU | 1000000 | day15-breakout-evaluation-v2-fire-reset |
| day21-final-long-training-seed1011 | 1011 | stage_b_2_5m | cuda:0 | NVIDIA GeForce RTX 4060 Laptop GPU | 2500000 | day15-breakout-evaluation-v2-fire-reset |
| day21-final-long-training-seed2022 | 2022 | stage_a_1m | cuda:0 | NVIDIA GeForce RTX 4060 Laptop GPU | 1000000 | day15-breakout-evaluation-v2-fire-reset |
| day21-final-long-training-seed2022 | 2022 | stage_b_2_5m | cuda:0 | NVIDIA GeForce RTX 4060 Laptop GPU | 2500000 | day15-breakout-evaluation-v2-fire-reset |
| day21-final-long-training-seed2022 | 2022 | stage_c_5m | cuda:0 | NVIDIA GeForce RTX 4060 Laptop GPU | 5000000 | day15-breakout-evaluation-v2-fire-reset |
| day21-final-long-training-seed3033 | 3033 | stage_a_1m | cuda:0 | NVIDIA GeForce RTX 4060 Laptop GPU | 1000000 | day15-breakout-evaluation-v2-fire-reset |
- Contract v2 artifact: `configs/eval/breakout_contract_v2.json`, SHA256 `e9947fc3a1235100aa92f75a79cf33668b36b4f325f22a0bc8b9546a356b85ba`.

## Selection decisions

```json
{
  "stage_a_1m": {
    "status": "complete",
    "stage": "stage_a_1m",
    "candidate_training_seeds": [
      1011,
      2022,
      3033
    ],
    "selected_training_seeds": [
      2022,
      1011
    ],
    "aggregate_values": [
      {
        "training_seed": 1011,
        "mean_return": 34.0,
        "median_return": 31.0,
        "std_return": 12.187425213445756,
        "count": 15,
        "healthy": true
      },
      {
        "training_seed": 2022,
        "mean_return": 34.86666666666667,
        "median_return": 30.0,
        "std_return": 15.465517198665625,
        "count": 15,
        "healthy": true
      },
      {
        "training_seed": 3033,
        "mean_return": 33.13333333333333,
        "median_return": 29.0,
        "std_return": 15.200292394848491,
        "count": 15,
        "healthy": true
      }
    ],
    "selection_metric": "mean raw Atari return across all complete selection episodes",
    "rule": {
      "primary_metric": "mean raw Atari return across all complete selection episodes",
      "priority": [
        "aggregate fixed-evaluation performance",
        "robustness via episode spread",
        "no correctness or health failure",
        "learning curve growth or plateau evidence",
        "earlier checkpoint when quality is near-equal"
      ],
      "near_equal_absolute_gap": 1.0,
      "forbidden_shortcuts": [
        "single best episode",
        "single lucky training seed",
        "training return peak",
        "GIF appearance",
        "final holdout result"
      ]
    },
    "reason": "select at most two fresh seeds by aggregate selection evaluation mean"
  },
  "stage_b_2_5m": {
    "status": "complete",
    "stage": "stage_b_2_5m",
    "candidate_training_seeds": [
      2022,
      1011
    ],
    "selected_training_seeds": [
      2022
    ],
    "aggregate_values": [
      {
        "training_seed": 2022,
        "mean_return": 51.4,
        "median_return": 51.0,
        "std_return": 14.677874505527019,
        "count": 15,
        "healthy": true
      },
      {
        "training_seed": 1011,
        "mean_return": 42.6,
        "median_return": 43.0,
        "std_return": 15.191225537570475,
        "count": 15,
        "healthy": true
      }
    ],
    "selection_metric": "mean raw Atari return across all complete selection episodes",
    "rule": {
      "primary_metric": "mean raw Atari return across all complete selection episodes",
      "priority": [
        "aggregate fixed-evaluation performance",
        "robustness via episode spread",
        "no correctness or health failure",
        "learning curve growth or plateau evidence",
        "earlier checkpoint when quality is near-equal"
      ],
      "near_equal_absolute_gap": 1.0,
      "forbidden_shortcuts": [
        "single best episode",
        "single lucky training seed",
        "training return peak",
        "GIF appearance",
        "final holdout result"
      ]
    },
    "reason": "select one candidate for the requested 5M continuation by aggregate evaluation"
  },
  "stage_c_5m": {
    "status": "complete",
    "stage": "stage_c_5m",
    "candidate_training_seeds": [
      2022
    ],
    "selected_training_seeds": [
      2022
    ],
    "aggregate_values": [
      {
        "training_seed": 2022,
        "mean_return": 49.93333333333333,
        "median_return": 50.0,
        "std_return": 16.258194515040373,
        "count": 15,
        "healthy": true
      }
    ],
    "selection_metric": "mean raw Atari return across all complete selection episodes",
    "rule": {
      "primary_metric": "mean raw Atari return across all complete selection episodes",
      "priority": [
        "aggregate fixed-evaluation performance",
        "robustness via episode spread",
        "no correctness or health failure",
        "learning curve growth or plateau evidence",
        "earlier checkpoint when quality is near-equal"
      ],
      "near_equal_absolute_gap": 1.0,
      "forbidden_shortcuts": [
        "single best episode",
        "single lucky training seed",
        "training return peak",
        "GIF appearance",
        "final holdout result"
      ]
    },
    "reason": "2.5M evaluation showed substantial improvement, so 5M continuation remained justified.",
    "primary_trigger": "2.5M evaluation showed substantial improvement, so 5M continuation remained justified.",
    "trigger_evidence": {
      "training_seed": 2022,
      "stage_a_1m_mean_return": 34.86666666666667,
      "stage_b_2_5m_mean_return": 51.4,
      "mean_return_improvement": 16.53333333333333
    },
    "user_requested_5m": true,
    "request_is_supplemental_provenance": true
  },
  "final_checkpoint": {
    "status": "frozen",
    "selected": {
      "training_seed": 2022,
      "stage": "stage_b_2_5m",
      "target_transitions": 2500000,
      "mean_return": 51.4
    },
    "rule": {
      "primary_metric": "mean raw Atari return across all complete selection episodes",
      "priority": [
        "aggregate fixed-evaluation performance",
        "robustness via episode spread",
        "no correctness or health failure",
        "learning curve growth or plateau evidence",
        "earlier checkpoint when quality is near-equal"
      ],
      "near_equal_absolute_gap": 1.0,
      "forbidden_shortcuts": [
        "single best episode",
        "single lucky training seed",
        "training return peak",
        "GIF appearance",
        "final holdout result"
      ]
    },
    "holdout_was_locked": true,
    "evaluation_order": "selection → final freeze → final holdout"
  }
}
```

## Canonical final model

- model: `assets/day21/models/final_model/model.pt`
- model SHA256: `6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a`
- source checkpoint: `E:\breakout-rl-engineering-day21-runs\day21-final-long-training\seed2022-retry-01-retry-02-retry-03\checkpoints\step-02500000.pt`
- selected stage/seed: `stage_b_2_5m` / `2022`
- holdout status: `completed`

## Final holdout summary

- episodes: `15` complete; mean raw return `30.933`; std `14.484`
- terminated: `15`; truncated: `0`; time-limit truncated: `0`
- Contract v2 health gate: `True`

## Limitations

- Training quality is interpreted only under Contract v2 and its raw-reward evaluation semantics.
- A continuous in-process run preserves Replay state; a crash resume is explicitly non-exact when Replay was not serialized.
- Stage C was justified by substantial 2.5M selection improvement; user_requested_5m is supplemental run-horizon provenance, while final checkpoint selection remains gate-driven.
