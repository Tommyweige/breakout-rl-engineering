# Day 16｜GPU 不是放著就會變快：一次讓多個 Breakout 一起跑

Day 14 有一個很反直覺的結果。把 Replay Buffer（保存 Agent 與環境互動紀錄的循環資料結構）搬到 GPU 後，單獨量資料抽樣和 Deep Q-Network（DQN）更新的速度確實很快；但整個 Breakout 訓練流程沒有因此同比例變快。

原因不是 GPU 算得不夠快，而是每次只丟給它一點點工作。原本的路徑是：

```text
1 個 Breakout environment
→ 產生 1 個 observation
→ GPU 算 1 組 Q-values
→ CPU 讓遊戲前進
→ 寫入 1 筆 transition
→ 重複
```

這就像有一條很寬的高速公路，卻每次只放一台車上去。Day 16 要回答的是：**如果一次同時跑幾個 Breakout，把多張畫面一起送進模型，再把多筆 transition 一起寫入 GPU Replay，完整 trainer 會不會真的變快？**

一次互動紀錄叫做 transition：它至少保存目前畫面、採取的 action、環境回傳的 reward、下一張畫面，以及 episode 是否結束。這些 transition 就是後面 optimizer（根據誤差更新網路參數的工具）會抽樣的資料。

這次還有一個不能省略的前提：Day 15 已經用 Contract v2 固定了 Breakout 的任務語意。frame skip 代表一次 action 讓 Atari 前進的原始畫面數，這裡是 4；frame stack 代表把最近幾張畫面疊成一個 observation，這裡是 4；sticky action probability 是 0.25，代表環境有 25% 機率延續前一個 action。開局和掉命後的必要 FIRE 由環境處理，評估使用不 clipping 的 raw reward。速度比較不能靠偷偷換規則取得。

## 一次跑多個 environment，改變的是資料形狀

這裡的向量化環境（vectorized environment）不是把八份 Python 迴圈複製貼上，而是讓一個環境介面同時管理 N 個相互獨立的 Breakout。每個子環境仍有自己的遊戲狀態，但 observation 可以排成一個 batch，也就是同一批待處理資料：

```text
Env 0 ─┐
Env 1 ─┤
Env 2 ─┤→ (4, 4, 84, 84) observations
Env 3 ─┘
             ↓
          一次 DQN forward
             ↓
          (4, 4) Q-values
             ↓
           4 個 actions
```

第一個 `4` 是 environment 數量；後面的 `(4, 84, 84)` 是每個 Agent 看到的四張 84×84 灰階畫面。Q-value 是模型對「現在做某個 action，未來大概有多值得」的估計，所以 `(4, 4)` 代表四個 environment 各自得到四個 action 的估計。

真正的改變是神經網路只 forward 一次，而不是在 Python 裡對四張畫面各呼叫一次模型。Replay insertion 也採用同一個方向：一次把多筆 transition 寫入 ring buffer，而不是每筆資料各做一次小型 GPU copy。這裡的 GPU Replay 是把 Replay Buffer 的 observation 與欄位直接放在 GPU memory 中，讓後面的抽樣少一次主機到 GPU 的搬移。

這個資料流和實際實作的順序如下。它是依照程式的 component interaction 畫出的結構圖；特別要注意 done environment 的 final observation 必須先保存，才可以局部 reset。

[![單一環境與向量化 DQN trainer 的資料流，以及 done environment 的局部 reset](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/vectorized-pipeline.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/vectorized-pipeline.png)

## `global_step` 必須數實際 transition

單一環境時，step 一次通常得到一筆 transition；但 N=8 時，一次 vector step 會得到八筆。`global_step` 因此代表已接受的實際 environment transitions，而不是 Python loop 跑了幾次：

```text
num_envs = 8
vector step 一次
→ global_step 增加 8
```

這個定義會影響 learning starts、epsilon 探索率、optimizer update、target network 同步、checkpoint step 和 metrics 的橫軸。若把一次八筆資料誤算成一步，整個 training schedule 都會被拉長八倍。

目前 `train_frequency = 4`。因此 N=8 的一次 vector step 會跨過第 4 筆和第 8 筆兩個 optimizer boundary。程式不能只寫成「每次 vector step update 一次」，否則 update-to-data ratio 已經改變。

實作會在這些 transition boundary 拆開 replay insertion：

```text
前 4 筆 → insert → update
後 4 筆 → insert → update
```

這保證第一次 update 看不到本來屬於後半段的資料。不過還有一個更細的語意差異：八個 actions 仍然在 vector step 開始時一次選完，所以後四筆 action 使用的是 update 前的 online network。這不是 replay corruption，卻是 behavior-policy lag，不能說成和 single-env bit-for-bit 完全相同。

因此 trainer 另外定義 strict action-selection parity rule：

```text
num_envs <= train_frequency
and train_frequency % num_envs == 0
```

符合這個規則時，一個 action batch 不會跨過 optimizer boundary；不符合時仍可做 systems screening，但 trainer 會發出 warning 並把 lag 寫入 metadata。這次的 crafted test 讓模型在第 4 筆 update 後改變偏好的 action：N=8 的同一批八個 action 全部保留舊偏好，strict N=4 則在下一個 vector batch 看見新偏好。

## 多個一局遊戲（episode）的邊界比速度更容易出錯

假設 Env 0 結束、Env 1 到 Env 3 還在玩。正確順序是先把 Env 0 的 terminal observation 存成 transition 的 `next_state`，再只 reset Env 0。若 vector API 自動 reset，而 trainer 直接使用 reset 後的 observation，就會把上一局最後一張畫面接到下一局第一張畫面，Replay 內會出現遊戲中不存在的 transition。

`terminated` 表示遊戲本身真的結束；`truncated` 表示受到時間限制而停止。兩者都要讓該子環境重新開始，但它們仍分開保存。這也是為什麼本實作使用 disabled autoreset，也就是不讓 vector API 自動重設 done environment：trainer 可以先取得 final observation，再用 reset mask 只重設已完成的 environments。Day 15 定下來的 FIRE serve state 也一樣逐 environment 管理，不能讓某個子環境掉命影響其他子環境。

sticky action 是 ALE 以固定機率忽略本次 requested action、延續前一個 action 的機制。這會讓「wrapper 送出了 FIRE」和「遊戲已經開始」變成兩件事。新的 wrapper 不再送一次 FIRE 就清掉 pending serve state，而是持續送 FIRE，直到觀察到 raw reward，或連續兩次看到至少 `0.0001` 比例的 observation activity；八次都沒有確認時直接失敗，避免默默放行一個可能永遠卡住的 episode。

這裡仍然要區分三層資訊：policy requested action、wrapper-resolved action（實際往下傳給 AtariPreprocessing/ALE 的 action），以及 ALE 內部不可直接取得的 sticky-action 隨機結果。診斷不會把第三層假裝成已觀察到的資料。

選出的 N=2 100K checkpoint 的 Contract v2 diagnostic 覆蓋全部 15 個固定 seeds，15/15 正常 terminated、0/15 truncated、0/15 TimeLimit。原本會重現 26,998-step TimeLimit 的 seed 101，現在在第 198 個 agent step 正常結束；它的 initial serve 第一次 FIRE 沒有 activity，第二次雖然看見 activity 但還不足以完成確認，第三次才完成連續兩次 activity 的確認。後面四次 life-loss serve 也各在第二次 FIRE 完成確認。這個 trace 也保存 lives、life-loss、raw reward、requested/resolved action、serve attempt 與 observation change signal。

為了分辨 sticky action 是否真的參與了這個 retry，diagnostic 另外用相同 checkpoint 和五個固定 seeds 做兩個 control：兩組都只要求一次 activity confirmation，唯一改變的是 sticky probability。`0.25` control 出現 1 次沒有 observation activity 的 retry；`0.0` control 則是 0 次。這支持「sticky action 可能造成 FIRE 沒有被觀察到」的解釋，但 ALE API 不會公開那次隱藏抽樣，所以仍不能把它寫成直接的因果證明。

## 10K systems screening：batching 確實有用，但 N 越大不等於越好

接著才看完整 trainer 的 systems screening，也就是先量資料流效能而不把短跑分數當成模型結論。四組都使用重新隨機初始化（fresh initialization）、seed `42`、10,000 個實際 transitions、Vanilla DQN、GPU Replay、batch size 32 和 Contract v2；完整 machine-readable source 是 [`vectorized-training.json`](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/vectorized-training.json)。

| 同時環境數 N | vector iterations | accepted transitions/s | action inference calls | Replay insertion calls | strict parity |
|---:|---:|---:|---:|---:|:---:|
| 1 | 10,000 | 298.20 | 10,000 | 10,000 | yes |
| 2 | 5,000 | 387.89 | 5,000 | 5,000 | yes |
| 4 | 2,500 | 456.63 | 2,500 | 2,500 | yes |
| 8 | 1,250 | 483.30 | 1,250 | 2,500 | no |

四組都完成 `2,251` 次 optimizer update 與 `21` 次 target sync。N=8 的吞吐最高，但它一次 batch 會跨過 update boundary；N=2 和 N=4 都符合 strict parity，其中 N=4 是短跑中最快的 strict-parity 設定。最後選哪一個，要留到 fresh 100K guardrail，而不是從 10K checkpoint 的分數推論。

這張圖的左側是每秒完成的 accepted transitions，右側是在相同 10K budget 下的 wall-clock，也就是真實經過的秒數。它回答的是「完整 training pipeline 能處理多少資料」，不是「模型是否學得更好」。

[![1、2、4、8 個環境在相同 10K transition budget 下的吞吐與 wall-clock](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/vectorized-throughput.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/vectorized-throughput.png)

## 速度主要來自減少零碎呼叫

在這組資料中，N=1 需要 10,000 次 model forward；N=4 降成 2,500 次；N=8 則是 1,250 次。每次 forward 的輸入是 `(N, 4, 84, 84)`，輸出是 `(N, 4)`，四個 action 仍然是 `NOOP`、`FIRE`、`RIGHT`、`LEFT`。

[![不同 environment count 的 batched inference throughput 與單次 forward 成本](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/batched-inference.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/batched-inference.png)

獨立的 Replay insertion 微基準測試（microbenchmark）也測了 batch size 1、2、4、8、16。一次測量使用真實 Breakout reset/step 產生 observation，之後只為量 copy cost 而重複資料；它不是拿重複畫面宣稱學習效果。

| 一次寫入幾筆 | transitions/s | 每次呼叫成本 |
|---:|---:|---:|
| 1 | 4,963 | 0.201 ms |
| 2 | 9,832 | 0.203 ms |
| 4 | 17,355 | 0.230 ms |
| 8 | 32,488 | 0.246 ms |
| 16 | 48,983 | 0.327 ms |

[![batch size 1、2、4、8、16 的真實 replay insertion microbenchmark](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/replay-insertion.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/replay-insertion.png)

圖表的結論很具體：批次越大，固定的函式呼叫與 GPU copy 成本越能被攤薄。但這不代表整個 trainer 會按同樣比例加速。完整 pipeline 還要付出 ALE CPU stepping、optimizer update、episode reset 與 metrics 寫入的成本。

`SyncVectorEnv` 也要誠實標成 limitation：它把多個 environment 統一成 vector API，但沒有宣稱 ALE CPU stepping 已經變成多執行緒或多進程平行。這次主要收益來自 batched model inference 與 batched Replay insertion；AsyncVectorEnv/parallel ALE 留給未來的 systems work。

[![1、2、4、8 個環境的固定間隔 CPU/GPU utilization sampling](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/system-utilization.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/system-utilization.png)

這張圖的縱軸是固定間隔 sampler 量到的平均 CPU/GPU utilization，不是 GPU 理論峰值。它支持「零碎呼叫減少是主要改善來源」這個解釋，不能單獨證明某個 N 在所有硬體上都最好。

## Fresh 100K validation：選出的 N=2 通過 contract gate，但品質不能只看速度

10K 是 systems screening；它不足以決定長一點的訓練是否仍維持環境語意。因此再用相同 seed、訓練參數（hyperparameters）、CPU thread setting、GPU Replay 和 Contract v2，重新從隨機初始化開始（fresh start）跑 N=1 reference 與 strict N=2 candidate，各 100,000 transitions。N=4 的同規格長跑則保留作為 supplemental candidate。選出的 N=2 source 是 [`vectorized-training-100k-n2.json`](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/vectorized-training-100k-n2.json)；N=4 source 是 [`vectorized-training-100k.json`](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/vectorized-training-100k.json)。

| N | accepted transitions/s | wall-clock | action calls | Replay insertion calls | optimizer updates | target syncs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 238.67 | 419.00 s | 100,000 | 100,000 | 24,751 | 201 |
| 2 | 380.74 | 262.65 s | 50,000 | 50,000 | 24,751 | 201 |
| 4 | 368.06 | 271.70 s | 25,000 | 25,000 | 24,751 | 201 |

N=2 在這台 RTX 4060 Laptop GPU 上完成相同 budget 約快 `1.60×`，N=4 約快 `1.54×`，三者的 update/sync 次數相同。下圖是 N=1/N=4 supplemental scaling run；真正的 candidate decision 仍要搭配 fixed-seed evaluation。

[![100K N=1 與 N=4 supplemental validation 的吞吐與 wall-clock](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/vectorized-100k-vectorized-throughput.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/vectorized-100k-vectorized-throughput.png)

## Evaluation 必須同時記錄 requested 與 executed action

正式 evaluation 會把 policy 凍結、固定 15 個 concrete seeds、epsilon 設為 0，並使用 raw Atari reward。這裡的 action provenance，也就是記下 action 是誰要求、最後由哪一層送出的來源資訊，是必要的：policy 可能要求 `RIGHT`，但 serve wrapper 為了環境規則實際送出 `FIRE`。Replay 和 action count 都應看 executed/wrapper-resolved action，不能把 requested action 當成已執行。

這次的 formal artifacts 同時保存 `requested_action_distribution`、`executed_action_distribution`、`auto_fire_count` 和 `auto_fire_reason_counts`；歷史欄位 `action_distribution` 仍保留，但明確定義成 executed/wrapper-resolved action。

100K guardrail 的結果如下。Contract v2 Random baseline 平均 `1.73`；N=1 平均 `9.00`；N=2 平均 `6.07`；N=4 平均 `2.33`。四者都是 15/15 terminated、0/15 truncated、0/15 TimeLimit。

| Run | 平均 raw return | 中位數 | 標準差 | 平均 episode length | terminated | truncated |
|---|---:|---:|---:|---:|---:|---:|
| Random Contract v2 | 1.73 | 2.00 | 1.12 | 197.40 | 15/15 | 0/15 |
| N=1, 100K | 9.00 | 9.00 | 2.03 | 468.67 | 15/15 | 0/15 |
| N=2, 100K | 6.07 | 6.00 | 2.54 | 352.33 | 15/15 | 0/15 |
| N=4, 100K | 2.33 | 2.00 | 1.07 | 201.27 | 15/15 | 0/15 |

這個結果回答了兩件事：第一，新的 FIRE confirmation 沒有在這組 fixed seeds 造成 serve deadlock 或 TimeLimit failure；第二，N=2 與 N=4 的 15 局分數都低於 N=1，所以不能把 throughput speedup 寫成 quality equivalence。N=2 仍高於 Random baseline，因此在這次固定 100K guardrail 中，比 N=4 更適合作為 systems candidate；N=1 仍是 model-quality reference。完整結果與 checkpoint hashes 保存在 [`evaluation-summary.json`](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/de5616c7998f48d341c5767a326969a7d5a42285/assets/day16/evaluation-summary.json)。

## Day 16 的另一條線：`max` 也可能改變 value 的意義

向量化解決的是資料收集和計算粒度，不會自動解決 DQN 的估計偏差。若每個 action 的 Q-value 都有一點誤差，`max` 可能偏向挑中誤差剛好為正的 action。這個機制用獨立的 [Q-value overestimation note](day16-q-value-overestimation.md) 說明：CPU toy simulation 展示可重現的選擇偏差，CUDA real-checkpoint probe 則只保存真實模型輸出，兩者不混為同一種證據。

## 最後選擇與下一個問題

Day 16 的 selected systems backend 是 strict-parity N=2：

```text
ALE/Breakout-v5 / Contract v2
frame_skip=4, frame_stack=4, sticky_action_probability=0.25
environment-owned initial/life-loss FIRE
GPU Replay, float32, batch_size=32
learning_starts=1000, train_frequency=4
target_update_interval=500, epsilon_decay_steps=10000
training seed=42, CPU threads=2
strict action-selection parity enabled
```

這個選擇建立在三層證據上：10K screening 顯示 batching 的 systems 收益，crafted instrumentation 誠實標出 N=8 的 behavior-policy lag，fresh 100K 與 15-episode Contract v2 guardrail 則確認 N=2 沒有 serve deadlock 或 TimeLimit regression，且固定 seed 的 return 高於 N=4。N=2 的 return 仍低於 N=1 reference（`6.07` 對 `9.00`），所以這份 evidence 不宣稱 policy-quality equivalence；N=1 保留為後續品質比較的 reference。這個選擇也不是跨硬體、跨 seed 或跨演算法的普遍最優解。

現在可以帶著一個比較準確的問題進入下一步：當 training pipeline 已經能有效率地批次處理 transition，Vanilla DQN target 中的 `max` 是否仍會放大 Q-value 的樂觀誤差？這正是 Double DQN 要處理的演算法問題。
