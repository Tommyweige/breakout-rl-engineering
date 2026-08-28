# Day 16｜一次讓多個 Breakout 一起前進：Vectorized DQN Training

Day 14 把 Replay Buffer 放進 GPU 後，得到一個有點反直覺的結果：Replay 本身的 microbenchmark 很快，但完整 trainer 沒有因此贏過 CPU 版本。原因不在某一個 CUDA kernel，而在資料流仍然長這樣：CPU 等一個 Breakout environment 前進，GPU 只替一張 observation 做一次 batch=1 的 action inference，接著只寫入一筆 transition。

這條路徑就算每個零件都「在使用 GPU」，也會產生大量零碎工作。Day 15 又固定了 Contract v2：開局與掉命後由環境負責 FIRE、保留 `terminated` 和 `truncated` 的語意、評估使用 raw Atari reward。Day 16 的問題因此不是「換一個 RL 演算法」，而是：

> **能不能讓多個環境一起產生資料，同時保留原本的 DQN 訓練節奏與 Breakout episode 語意？**

## 先把「一次」說清楚

向量化環境（vectorized environment）是把多個彼此獨立的遊戲環境放在同一個介面下。一次 `step` 會讓 N 個環境各自執行一次，因此會產生 N 筆 transition；這裡的 N 是環境數量，不是模型的 action 數量。

模型也不必再逐張畫面呼叫 N 次。把 N 個 observation 疊成一個 batch 後，批次推論（batched inference）會一次得到 N 組 Q-values，再對每個環境各自做 epsilon-greedy 決策：有些環境可能探索，有些環境可能採用目前估計最好的 action。

這個實作的結構性資料流如下。圖中的 single-env reference 是原本的對照路徑；vectorized trainer 則把 forward 和 replay insertion 的細碎呼叫合併起來。

[![單一環境與向量化 DQN trainer 的資料流，以及 done environment 的局部 reset](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/vectorized-pipeline.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/vectorized-pipeline.png)

這不是把一個環境複製 N 次就結束。圖上有三個會影響正確性的順序：先保存 done transition 的 final observation，再只 reset 已結束的環境，最後才把實際新增的 transition 數加入 `global_step`。任何一個順序錯了，速度數字都可能很好看，但 Replay 裡的資料已經不是原本的資料。

## `global_step` 必須數 transition，而不是數迭代

在單一環境中，`global_step += 1` 看起來很自然。可是當 `num_envs = 8` 時，一次 vector iteration 會帶來 8 筆互動資料，因此正確的定義是：

```text
一次 vector iteration → 8 transitions
global_step += 8
```

這個差異會直接影響所有以步數為基準的設定。假設 `train_frequency = 4`，一次 vector step 從 0 走到 8，就跨過了 4 和 8 兩個 optimizer-update boundary。若只在每次 vector iteration 更新一次，實際的 update-to-data ratio 就被偷偷減半。

Day 16 的 scheduler 會列出 `(previous_global_step, current_global_step]` 中所有跨過的 boundary，再逐一處理 optimizer update 和 target-network sync。這也是為什麼 10K screening 的四組結果都保留相同的 `2,251` 次 optimizer updates；`learning_starts = 1,000` 之後，每 4 個 transition 更新一次，最後一個 boundary 是 10,000。

這裡的「optimizer update」是用一個 replay mini-batch 修正 online network 參數的一次操作；target sync 則是把 online network 的權重複製到暫時固定的 target network。兩者的計數都要跟實際 transition boundary 對齊，而不是跟 N 的大小對齊。

## 最容易被破壞的是 episode 邊界

多個環境同時前進後，每個環境仍有自己的 episode return、episode length、frame stack 和 reset 狀態。env 0 結束，不能順手把 env 1 到 env N-1 的統計清掉。

更危險的是 autoreset。某些 vector API 會在回傳 done observation 的同一步，直接把該環境換成新 episode 的初始 observation。如果把這張新畫面當成上一個 action 的 `next_state`，Replay 就會記錄一個不存在的轉移：遊戲突然從「上一局結束」跳到「下一局剛 reset」。

因此這個 trainer 使用 Gymnasium 的手動 reset 模式：先取得每個 done environment 的 final observation，寫入 Replay 後，再用 `reset_mask` 只 reset `terminated` 或 `truncated` 為真的環境。`terminated` 代表遊戲本身結束；`truncated` 代表受時間等外部限制截斷。兩個欄位仍然分開保存，並沒有被模糊合併成一個 `done`。

批次路徑的核心可以濃縮成下面這段代表性程式。它省略了 profiling 和錯誤檢查，但保留了資料順序：`current_observations` 是 action 前的畫面，`final_next_observations` 是 action 後真正應保存的畫面。

```python
current_observations = np.array(observations, copy=True)
actions, sources, epsilons = self._select_actions(current_observations)
next_observations, rewards, terminated, truncated, infos = self.env.step(actions)

final_next_observations = _final_observation_batch(
    next_observations, terminated, truncated, infos, ...
)
self.replay.add_batch(
    current_observations, actions, rewards, final_next_observations,
    terminated, truncated,
)
self.global_step += self.num_envs
```

這段程式真正改變的是資料的「批次大小」，不是 transition 的內容。每一列仍然是原本的 `(state, action, reward, next_state, terminated, truncated)`；不同的是 N 列一起通過 model 和 replay 的 public seam。

## Replay 的 ring buffer 也必須支援批次寫入

GPU Replay 原本的 `add` 每次只處理一筆資料。Day 16 新增 `add_batch`，一次把 states、actions、rewards、next states 和兩個 episode flags 寫入連續的 ring-buffer slots。ring buffer 是固定容量的循環陣列：寫到尾端後回到索引 0，新的 transition 會覆蓋最舊的 transition。

批次寫入因此要同時處理三種情況：尚未填滿的 buffer、跨過容量尾端的 wraparound，以及一次 batch 大於剩餘空間。實作先驗證所有欄位，再用最多兩段連續 copy 完成寫入；這樣不會因為中途欄位錯誤而只寫入半個 batch，也不需要對 N 筆資料逐筆呼叫 GPU copy。

插入資料仍然維持 `uint8` observation storage，只有抽樣送到 model 時才轉成 normalized `float32`。這保留了前幾天建立的記憶體邊界，也讓 Day 14 的 GPU-resident replay 能直接接上新的 batch API。

## 10K transitions 的完整 systems screening

下面的數字來自真的 `ALE/Breakout-v5`、PyTorch CUDA、GPU Replay 和固定 seed `42`。四組設定都使用相同的 10,000 accepted transitions、batch size 32、`learning_starts = 1,000`、`train_frequency = 4`、target interval 500、float32 與 Contract v2 的 FIRE reset。完整的 machine-readable source 是 [`vectorized-training.json`](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/vectorized-training.json)。

| 環境數 N | vector iterations | accepted transitions/s | batched action calls | `add_batch` calls | optimizer updates | target syncs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,000 | 248.84 | 10,000 | 10,000 | 2,251 | 21 |
| 2 | 5,000 | 316.93 | 5,000 | 5,000 | 2,251 | 21 |
| 4 | 2,500 | 323.75 | 2,500 | 2,500 | 2,251 | 21 |
| 8 | 1,250 | 411.07 | 1,250 | 1,250 | 2,251 | 21 |

這張圖的左側是每個設定在相同 transition budget 下的 end-to-end throughput，右側是完成同一個 budget 實際花費的 wall-clock。`accepted transitions/s` 是 `global_step` 的速度，不是把 vector iteration 誤當成 transition 的速度。

[![1、2、4、8 個環境在相同 10K transition budget 下的吞吐與 wall-clock](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/vectorized-throughput.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/vectorized-throughput.png)

在這次 RTX 4060 Laptop GPU 與 2 個 CPU threads 的測試上，N=8 比 N=1 約快 `1.65×`。這個結果支持 batching 確實減少了零碎呼叫，但不代表 N 越大永遠越好；當環境 step、CPU 記憶體或 GPU batch 已經飽和後，繼續增加 N 可能只會讓 reset 和 host-side bookkeeping 變重。

## 推論和 Replay insertion 到底省在哪裡？

完整 trainer 的 batched inference stage 從 N=1 的 10,000 次 model call，降到 N=8 的 1,250 次。每次 call 的輸入是 `(N, 4, 84, 84)`，輸出是 `(N, 4)` Q-values；四個 action 的意義仍然是 `NOOP`、`FIRE`、`RIGHT`、`LEFT`。

[![不同 environment count 的 batched inference throughput 與單次 forward 成本](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/batched-inference.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/batched-inference.png)

Replay insertion 另外做了 batch size 1、2、4、8、16 的獨立 microbenchmark。輸入 observation 來自一次真實 Breakout reset/step，之後只為了量測 copy cost 而重複成指定大小；這個測試不是拿重複畫面宣稱模型學習效果。

| insertion batch | transitions/s | 每次 `add_batch` |
|---:|---:|---:|
| 1 | 5,483.26 | 0.182 ms |
| 2 | 8,949.64 | 0.223 ms |
| 4 | 19,713.95 | 0.203 ms |
| 8 | 30,769.89 | 0.260 ms |
| 16 | 46,454.18 | 0.344 ms |

N=16 時，一次呼叫的成本只從 0.182 ms 增加到 0.344 ms，但同一段時間寫入的 transition 數大幅增加。這正是 batching 的工程價值：不是每一筆資料都變得免費，而是把固定的函式呼叫與小型 GPU copy overhead 攤到更多 transition 上。

[![batch size 1、2、4、8、16 的真實 replay insertion microbenchmark](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/replay-insertion.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/replay-insertion.png)

最後看 utilization。固定間隔 sampler 觀察到的 GPU 平均使用率從 N=1 的 `20.81%` 到 N=8 的 `22.64%`，process CPU 平均值則約從 `5.63%` 上升到 `11.11%`。這再次提醒我們：

```text
GPU utilization 高 ≠ trainer 一定更快
trainer 更快 ≠ policy 一定學得更好
```

這次 throughput 的改善來自整個資料流縮短；GPU utilization 只是其中一個觀察值，不能單獨當成選擇依據。

[![1、2、4、8 個環境的固定間隔 CPU/GPU utilization sampling](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/system-utilization.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/system-utilization.png)

## 速度 candidate 還要經過固定 evaluation

systems benchmark 只回答「資料流跑得多快」，不能回答「模型學得好不好」。因此我把 N=1 reference 和吞吐最高的 N=8 candidate 都送進 Day 15 的 Contract v2 evaluation：相同的 15 個 episode、相同 seed 群、epsilon=0、raw reward、環境負責 FIRE。

| candidate | mean raw return | median | std | mean episode length | terminated | TimeLimit truncated |
|---|---:|---:|---:|---:|---:|---:|
| N=1 | 2.80 | 2.00 | 2.74 | 2,030.67 | 14/15 | 1/15 |
| N=8 | 2.67 | 2.00 | 2.44 | 2,033.33 | 14/15 | 1/15 |

兩者的平均回報差距只有 `0.13`，episode termination pattern 也相同。這個小型 guardrail 沒有顯示 N=8 造成明顯的 policy regression；但兩組都各有一次 TimeLimit truncation，且只有 15 局，所以它不是宣布模型品質穩定的證明。完整結果與 checkpoint SHA-256 在 [`evaluation-summary.json`](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/codex/issue-18-day16/assets/day16/evaluation-summary.json)。

這裡也要保留一個重要的實作邊界：transition boundary 的「計數」與 update 次數是精確對齊的，但 vector iteration 會先收集當次的 N 筆資料，再在 iteration 結束後處理跨過的 boundary。因此它不是 single-env 的 bit-for-bit replay/update trace；它是維持相同 transition-based schedule 的 batched training backend。若後續實驗需要逐 transition 完全重現，就必須再付出拆分 batch 或同步的成本，不能只看目前的 throughput 數字。

## Day 16 真正解決了什麼？

Day 14 找到的是：單純把 Replay 搬到 GPU，無法消除 single-environment、batch=1 inference 和 per-transition insertion 的固定成本。Day 16 則把這個瓶頸轉成可重用的 backend：

- 多個環境各自保留 episode 與 frame-stack 狀態；
- 一次 forward 處理 `(N, 4, 84, 84)`；
- 每個環境獨立決定探索或 greedy action；
- terminal transition 使用真正的 final observation；
- `add_batch` 保留 ring-buffer ordering 與兩個 episode flags；
- update、target sync 與 epsilon 都依實際 transition count 前進。

在本次硬體與 10K screening budget 上，N=8 是 throughput candidate；evaluation 則說明它目前沒有顯著偏離 N=1 reference。這足以讓後面的 DQN family experiments 使用更有效率的資料收集路徑，但還不足以回答 Double DQN 或 Dueling Network 哪個演算法更好。

下一篇會處理另一個問題：即使資料流和訓練系統正確，DQN 用同一個 network 選 action、又估計該 action 的 target 時，Q-value 為什麼可能被系統性高估？這會把焦點帶到 Double DQN，而不是再把 Day 16 的 systems optimization 混成新的演算法比較。
