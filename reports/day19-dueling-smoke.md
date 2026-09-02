# Day 19 Dueling Double DQN CUDA smoke

這份 report 的問題是：加入 Dueling heads 後，既有的 CUDA/vectorized training hot path 是否仍能完成，且工程成本是否可量測？它不是 model-quality ranking。

- generated at: `2026-08-30T16:38:55.756786+00:00`
- device: `cuda:0`
- contract: `configs/eval/breakout_contract_v2.json` (`day15-breakout-evaluation-v2-fire-reset`)
- seed: `42`
- transitions per run: `5000`

## Observed runtime

| architecture | parameters | environment SPS | optimizer updates/s | training samples/s | wall-clock (s) | peak allocated VRAM | peak reserved VRAM | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard | 1686180 | 376.2515471533683 | 75.32555974010432 | 2410.4179116833384 | 13.288981900084764 | 639141888 | 658505728 | completed |
| dueling | 3292837 | 360.15301086111737 | 72.10263277439569 | 2307.284248780662 | 13.882988200057298 | 673044992 | 1245708288 | completed |

## Comparison interpretation

- both runs completed: `True`
- checkpoint save/load passed for both runs: `True`
- same Double DQN algorithm: `True`
- same seed and transition budget: `True`
- same training settings except architecture: `True`
- parameter-count delta (dueling − standard): `1606657`
- Dueling/standard environment-SPS ratio: `0.957213368518884`

This is an infrastructure and hot-path regression check. It does not rank policy quality or select a Day 20 winner.

Raw values and complete provenance are in `assets/day19/dueling-smoke-runtime.json`; the run directories are reproducible inputs for later inspection.
