# Day 15｜DQN milestone evaluation

## 先看結論

在固定的 evaluation protocol 下，本次結果分類為 **B：DQN 中心值較高，但分布仍有重疊**。這個分類只描述本次已收集的完整 episode samples，不是多個 training seeds 的統計顯著性檢定。

Random 的平均 raw Atari return 是 **1.33**，DQN 是 **4.53**，平均差值為 **3.20**；中位數則是 1.00 對 5.00。
這裡的 raw reward 是環境回傳的原始遊戲分數，不是訓練時可能使用的 clipped reward。

## Day 14 的學習訊號，為什麼還不算正式評估

Day 14 的 100K 曲線和單局 GIF 是 development evidence：它們顯示模型值得再驗證。Day 15 則把 final 100K checkpoint 凍結，用獨立 seeds 和多局完整 rollout 檢查這個訊號能否重現。checkpoint selection rule 在 evaluation 前固定為完成 100,000 個環境步數的 final checkpoint，沒有從 50K、75K、100K 中挑最好的一局。

### Day 14 candidate provenance

| 欄位 | 實際值 |
|---|---|
| source manifest | `experiments/day14-final-frozen-100k/manifest.json` |
| source run | `day14-final-vanilla-dqn-seed42` |
| checkpoint | `assets/day14/final-runs/day14-final-frozen-100k/day14-final-vanilla-dqn-seed42/checkpoints/step-00100000.pt` |
| checkpoint SHA-256 | `022bda1ea5bb1ebcb0535ebb522fc954af11fd43607e124d5e5dc7a2aec3b79b` |
| checkpoint step | 100,000 |
| training seed | 42 |
| training budget | 100,000 environment steps |
| learning rate | `0.0002` |
| batch size | 32 |
| train frequency | 4 |
| replay backend | `cpu` |
| selection rule | final checkpoint at 100000 environment steps |
| Day 14 trainer PyTorch / CUDA | `2.13.0+cu130` / `13.0` |
| Day 14 trainer commit | `8696d6eaf8a8f3cab18f75661e83113ad5025102` |
| GPU profiling source | `experiments/day14-batch-size-profiling-final/batch-size-comparison.json` |
| selected batch end-to-end SPS | 235.74 |
| selected GPU utilization mean | 30.13% |
| selection rationale | candidate requires completed 10K, finite metrics, and end-to-end SPS strictly above batch 32；validate every short-stage candidate with 100K learning metrics before freezing；do not select a batch size from GPU utilization alone when return or numerical guardrails regress |

### Day 14 Gate A evidence

正式 milestone 不是因為 checkpoint 檔案存在就自動成立；這裡把 Day 14 的長跑結果與選擇依據明確列出。

| Gate A 條件 | 實際證據 |
|---|---|
| status | `passed` |
| 10K reference recent mean | 1.50 (20 episodes) |
| 100K final recent mean | 6.15 |
| return signal | `True` |
| diagnostics healthy | `True` |
| selection not single best episode | `True` |
| checkpoint provenance complete | `True` |
| metrics diagnostics finite | `True` |

## 固定的評估規則

評估只讓 policy 讀取 observation、選 action，再把 raw reward 累積到該局結束。Random 與 DQN 共用 environment construction、seed handling、episode loop、terminated/truncated 判斷、統計與輸出 schema；差別只有 action 如何產生。

| 規則 | 值 |
|---|---|
| environment | `ALE/Breakout-v5` |
| observation shape | `[4, 84, 84]` |
| action count | 4 |
| evaluation seed groups | `[101, 202, 303]` |
| episodes per seed group | 5 |
| total episodes per policy | 15 |
| DQN epsilon | 0.0 (greedy) |
| score | raw Atari reward; no training reward clipping |
| DQN requested / resolved device | `cuda` / `cuda:0` |
| GPU | `NVIDIA GeForce RTX 4060 Laptop GPU` |
| PyTorch / CUDA | `2.13.0+cu130` / `13.0` |

DQN 的模型推論確實在 NVIDIA CUDA 上執行；Random 沒有 neural-network inference，所以它留在 CPU，不把兩者的 runtime 當成效能比較。本報告只比較遊戲回報。

## 不只看平均：每局結果和 spread

平均值描述整批樣本的中心；中位數比較不容易被極端局拉動；std（標準差）則描述回報的 spread。每個 policy 的 15 局都由環境自然 terminated 或 truncated，沒有把 evaluator cap 混進正式結果。

| Policy | N | complete | mean | median | std | min | max | mean episode length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 15 | 15 | 1.33 | 1.00 | 1.30 | 0.00 | 5.00 | 185.73 |
| DQN | 15 | 15 | 4.53 | 5.00 | 3.07 | 1.00 | 11.00 | 18131.33 |

下圖的每個點都是 raw evaluation artifact 裡的一局；箱型圖顯示中間分布，菱形是 mean，短線是 median。右側則按 evaluation seed group 顯示平均與 spread，避免只看一個總平均。

[![Random 與凍結 DQN 的每局回報分布，以及各 evaluation seed group 的平均與 spread](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/3f599ba/assets/day15/random-vs-dqn-returns.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/3f599ba/assets/day15/random-vs-dqn-returns.png)

每個 seed group 的完整 raw return 如下：

| Evaluation seed group | Episode | Concrete reset seed | Random return | DQN return |
|---:|---:|---:|---:|---:|
| 101 | 1 | 101 | 1.00 | 1.00 |
| 101 | 2 | 102 | 0.00 | 7.00 |
| 101 | 3 | 103 | 2.00 | 1.00 |
| 101 | 4 | 104 | 1.00 | 10.00 |
| 101 | 5 | 105 | 5.00 | 5.00 |
| 202 | 1 | 202 | 1.00 | 1.00 |
| 202 | 2 | 203 | 1.00 | 5.00 |
| 202 | 3 | 204 | 3.00 | 5.00 |
| 202 | 4 | 205 | 1.00 | 4.00 |
| 202 | 5 | 206 | 1.00 | 3.00 |
| 303 | 1 | 303 | 0.00 | 1.00 |
| 303 | 2 | 304 | 0.00 | 2.00 |
| 303 | 3 | 305 | 2.00 | 6.00 |
| 303 | 4 | 306 | 0.00 | 6.00 |
| 303 | 5 | 307 | 2.00 | 11.00 |
## FIRE / TimeLimit root-cause audit

Evaluation Contract v1 的原始 `results.json` 保留不變；下面的 sidecar 與新 diagnostic artifacts 將 `truncated` 細分為可追蹤的 ALE TimeLimit，並比較兩種不改 checkpoint 的 diagnostic。

| Mode | epsilon | FIRE assist | mean return | median | terminated | TimeLimit | truncation rate | mean length | term mean | trunc mean | dominant action fraction | life-loss → FIRE latency | auto-FIRE |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 greedy/no assist | 0.00 | no | 4.53 | 5.00 | 5 | 10 | 0.67 | 18131.33 | 8.00 | 2.80 | — | — | 0 |
| Diagnostic A | 0.0 | yes | 8.80 | 8.00 | 15 | 0 | 0.00 | 429.93 | 8.80 | — | 0.58 | 1.00 | 75 |
| Diagnostic B | 0.05 | no | 9.20 | 8.00 | 15 | 0 | 0.00 | 508.53 | 9.20 | — | 0.55 | 30.45 | 0 |

v1 root-cause trace 的 selected episodes 如下：

- seed `101`：`time_limit`, length `26998`, dominant `RIGHT` (0.96), unchanged observation fraction `0.96`；life loss step 54 → FIRE step 16449 (latency 16395)
- seed `102`：`terminated`, length `405`, dominant `RIGHT` (0.53), unchanged observation fraction `0.00`；life loss step 71 → FIRE step 104 (latency 33)；life loss step 203 → FIRE step 204 (latency 1)；life loss step 279 → FIRE step 280 (latency 1)；life loss step 352 → FIRE step 353 (latency 1)；life loss step 405 → FIRE step None (latency None)
- seed `103`：`time_limit`, length `26995`, dominant `RIGHT` (0.97), unchanged observation fraction `0.96`；life loss step 54 → FIRE step 16446 (latency 16392)

root trace summary：`2` 個 TimeLimit、最大連續不變 observation `16323` steps，dominant action `RIGHT` fraction `0.96`。這支持 timeout 的主要近因是 life loss 後長時間沒有有效重發球，並伴隨 action/observation collapse；它不支持把長 episode 直接解讀成模型能正常存活。

Diagnostic A 在相同 checkpoint、concrete seeds、preprocessing 和 raw reward 下加入環境側 FIRE：mean return 8.80、15/15 terminated、auto-FIRE 75 次，平均 life-loss → FIRE latency 1.00 steps。
Diagnostic B 保持 policy-responsible FIRE，只把 epsilon 設為 0.05：mean return 9.20、15/15 terminated，但平均 latency 30.45 steps，它證明少量探索可以偶爾解鎖 serve，卻不是穩定且可重現的 environment contract。

因此 Contract v2 選 Option B：`fire_reset=true`，由 environment 在 initial serve 和觀察到 life loss 後執行 FIRE；仍保留 `terminal_on_life_loss=false`，TimeLimit 只由 ALE 的 `game_truncated` 標記。這是對 serve deadlock 的修正，不是把 v1 與 v2 的分數當成同一環境規格直接比較。
Contract v2 machine-readable 定義保存於 `configs/eval/breakout_contract_v2.json`，Day 16 可直接載入。
比較圖由真實 v1 sidecar 與兩份 diagnostic JSON 重新產生，保存於 `assets/day15/fire-time-limit-diagnostics.png`。


## 這次結果能說到哪裡

在這組固定條件下，DQN 平均回報高於 Random 3.20 分，中位數也較高。這支持「Day 14 checkpoint 在這批獨立 evaluation episodes 中展現較高回報」；它不支持「所有未來起始狀態都會更好」或「已完成 multi-training-seed robustness」。

目前正式驗證的仍是一個 training seed（42）訓練出的 checkpoint；evaluation seed 101、202、303只改變凍結 policy 面對的環境隨機性。後續若要談訓練穩定性，還需要多個 training seeds。

## Day 16 的品質基準

Day 16 會把 single-environment training 改成多環境、批次 action inference 和批次 GPU Replay insertion。它必須重用本日的 seeds、每組 episode 數、greedy epsilon、raw reward、environment contract、done semantics 和 result schema，才能分辨速度最佳化是否造成 policy quality regression。之後的 Double DQN 與 Dueling Network 也應沿用同一套評估尺。

### 可重建的 artifacts

- Random JSON：`evaluations/day15-random-baseline/results.json`
- DQN JSON：`evaluations/day15-dqn-cuda/results.json`
- Random CSV：`evaluations/day15-random-baseline/episodes.csv`
- DQN CSV：`evaluations/day15-dqn-cuda/episodes.csv`
- 圖表由 `visualize_day15_evaluation.py` 從兩份 JSON 重新產生。
- 結果由 `evaluate_dqn.py` 使用 `configs/eval/breakout_eval.json` 產生；正式 DQN 命令指定 `--device cuda`。

Report generated at `2026-08-28T03:57:59.034188Z`。
