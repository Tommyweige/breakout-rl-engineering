# Day 20 DQN family comparison

這份 report 回答的問題是：在同一個 Contract v2、Day 16 CUDA backend、paired training seeds 與 500K actual environment transitions 下，哪個 DQN family 值得進入 Final Long Training？

- manifest status: `completed`
- formal horizon: `500000` actual environment transitions
- training seeds: `[11, 22, 33]`
- evaluation: `[101, 202, 303]` × `5` episodes, epsilon `0.0`, raw reward
- runtime requirement: requested `cuda`, precision `float32`, sequential `True`

## Evidence reuse

Day 18 DQN/Double evidence decision: `compatible_reused`.
Reuse is accepted only after the machine-readable audit confirms the Contract v2, backend controls, seeds, milestones, evaluation/Q-probe artifacts, and CUDA runtime conditions. A failed audit must leave the old entries out of the formal aggregate rather than treating them as zero-valued runs.

## 500K family evidence

| family | complete seeds | mean evaluation return | seed spread | mean SPS | mean wall-clock (s) | peak VRAM (bytes) | parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN | 3/3 | 15.844 | 3.169 | 350.094 | 714.700 | 638179328.000 | 1686180.000 |
| Double DQN | 3/3 | 18.156 | 2.173 | 345.566 | 723.472 | 638179328.000 | 1686180.000 |
| Dueling Double DQN | 3/3 | 20.178 | 0.719 | 219.527 | 1146.308 | 671733248.000 | 3292837.000 |

每個 family 的 quality 欄位是三個 training seed 的 fixed-evaluation mean 再取平均；seed spread 保留跨訓練隨機性的可見程度。SPS、wall-clock、VRAM 與 parameter count 是工程成本，不取代相同 transition budget 下的 quality 比較。

## Optional 1M extension

- status: `complete`
- triggered by the 500K rule: `True`
- completed entries: `6/6`

The extension is reported separately from the 500K screening decision. It can replace the final family selection only after every selected top-two family has complete 1M CUDA evaluation evidence.

| family | complete seeds | mean evaluation return | seed spread | mean SPS | mean wall-clock (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Double DQN | 3/3 | 19.356 | 7.539 | 315.329 | 1589.665 |
| Dueling Double DQN | 3/3 | 36.111 | 1.874 | 268.799 | 1879.139 |

## Selection

- final-training family: `dueling_double_dqn`
- selection horizon: `1000000` actual environment transitions
- deployment candidate: `dueling_double_dqn`
- winner above Contract v2 Random baseline: `True`
- winner beats runner-up on every paired seed: `True`
- 1M extension applied to final selection: `True`

這個選擇不使用 best single episode、best single seed、training return 峰值、100K 分數或 GIF 外觀。若正式 evidence 尚未完整，selection 保持 `incomplete`；若所有 family 都沒有可靠超過 Random baseline，deployment candidate 會保持空值，而不是製造 `best.pt`。

## Reproducible evidence

- `assets/day20/evidence-reuse-audit.json` — Day 18 reuse checks
- `assets/day20/dqn-family-training.png` — seed-level training curves
- `assets/day20/dqn-family-evaluation.png` — fixed evaluation by milestone
- `assets/day20/dqn-family-seed-spread.png` — 500K seed spread
- `assets/day20/dqn-family-runtime-cost.png` — measured engineering cost
- `assets/day20/family-comparison-flow.png` — staged execution/data-flow diagram
