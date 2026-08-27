# Day 14｜從 GPU Replay 到 100K：讓 DQN 實驗更快，也更可信

[Day 13](day13-debugging-unstable-rl-training.md) 已經證明目前這套 DQN trainer 能正常收集經驗、更新模型，也沒有立刻出現 NaN、無限大或明顯的梯度失控。

但它沒有回答另一個更重要的問題：**Agent 真的開始學會 Breakout 了嗎？**

Day 13 的 10,000 environment steps 只跑出 48 個完整 episode。這裡的 environment step，就是 Agent 做一次 action、環境回傳一次新狀態與 reward 的互動單位；episode 則是一整局遊戲。10K 足以抓出「程式有沒有壞」，卻很容易把幾局遊戲的隨機波動誤認成模型真的變強。

所以 Day 14 同時面對兩個問題：

1. 如果訓練要從 10K 拉到 100K，甚至之後跑到更長，現在這條 RTX 4060 的訓練管線夠有效率嗎？
2. 當訓練真的跑得更久，不同 learning rate 的差異會不會才開始出現？

這一天因此不只是「調參數」。我先把 Replay 與 GPU 資料路徑拆開量測，再凍結 systems 設定，最後才跑 100K 的受控實驗。這樣看到 Return 變化時，才比較知道自己到底改了什麼。

## 10K 是健康檢查，不是成績排名

10K 很適合當 short screening。它可以快速回答：Replay Buffer 有沒有累積資料、optimizer 有沒有真的更新、Target Network 有沒有同步、epsilon 探索比例有沒有下降，以及 loss、Q-value、gradient 是否維持有限值。

但「沒有壞掉」和「這個設定比較好」是兩件不同的事。

Breakout 的 reward 不會每一步都出現，而且每一局長度不同。假設 baseline 在 10K 的短期平均 Return 是 `1.15`，另一組設定是 `1.75`，這個差距可能只是最後幾局剛好打得比較好，而不是 learning rate 已經產生穩定效果。

因此 Day 14 把兩種用途分開：

- **10K screening**：確認設定能正常訓練。
- **100K main comparison**：觀察整條 learning curve 是否開始分化。

100K 也不是「一定能學會 Breakout」的神奇門檻。它只是把 Day 13 的觀察時間拉長十倍，讓短期隨機波動比較不容易支配整個結論。

[![從 Day 13 的 10K diagnostic、10K screening 到 Day 14 100K main comparison 的決策流程](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/25fee37/assets/day14/budget-stages.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/25fee37/assets/day14/budget-stages.png)

圖中的重點不是「100K 一定要挑出 winner」，而是反過來：如果 100K 仍然沒有可靠差異，**「目前無法分辨」本身就是合法的實驗結果。**

## 要跑到 100K，先看看時間都花去哪裡

如果長跑前完全不看系統瓶頸，很可能只是把同一套低效率流程重複十倍。

DQN 的 CNN 雖然在 GPU 上計算，但一個完整 training step 不只有神經網路。環境要先產生下一個 observation，transition 要寫進 Replay Buffer，訓練時還要抽樣 batch、整理資料，再把模型需要的 tensor 送進 GPU。

也就是說：**CUDA 有啟用，不代表 GPU 就會一直忙。**

Day 14 先從 Replay 的資料路徑開始。

## GPU-resident Replay：microbenchmark 很快，完整 trainer 卻沒有贏

原本的 Replay Buffer 把過去的互動存在 CPU 的 NumPy 陣列。模型要更新時，流程大致是：抽樣 transition、複製到 host buffer，再傳到 GPU。

GPU-resident replay 的想法更直接：**讓 Replay 本身常駐 GPU。** 抽樣 index、gather transition、把影像從 `uint8` 轉成模型使用的 `float32`，都在 GPU 上完成，接著直接進 DQN update。

從資料路徑看，它少掉了訓練 batch 的 CPU sampling → host staging → H2D transfer。先前的 optimizer-side microbenchmark 也確實顯示 GPU replay 可以做到約 39K–41K training samples/s，看起來非常漂亮。

但 microbenchmark 不是完整遊戲訓練。

我固定 batch32、`train_frequency=4`、`learning_starts=2048`、同一個模型與 seed，只改 Replay backend，跑了一次真正的 10K trainer A/B：

| Replay path | Environment SPS | Optimizer updates/s | Training samples/s | Wall-clock | GPU utilization mean |
|---|---:|---:|---:|---:|---:|
| CPU + preallocated | **361.61** | **71.92** | **2,302** | **27.65 s** | 23.87% |
| GPU-resident replay | 338.13 | 67.25 | 2,152 | 29.57 s | 24.63% |

SPS（steps per second）代表每秒真正完成多少 environment transitions。這張表得到一個很重要的負面結果：**在目前單一 environment、batch32 的完整 trainer 裡，GPU replay 沒有比較快。**

原因不是 GPU gather 太慢，而是 Replay 換到 GPU 後，每一筆新 transition 也要被寫進 GPU。獨立 profiling 顯示，CPU-preallocated path 的主要等待仍在單狀態 action selection 與 environment interaction；GPU-replay path 則多出明顯的 replay insertion copy 成本。GPU sampling 的優勢，在這個 workload 裡還不足以抵銷逐筆寫入的代價。

這並不推翻前面的 39K–41K samples/s。兩個 benchmark 回答的是不同問題：

- optimizer-side microbenchmark：**如果資料已經在 GPU，更新本身可以多快？**
- full trainer A/B：**真正玩 Breakout、寫 Replay、選 action、更新模型時，整體可以多快？**

不能把前者直接當成後者。

### 為什麼我仍然保留 GPU Replay？

因為這次 A/B 沒有證明 GPU replay 是現在的 throughput winner，但它仍然是一條值得保留的 systems backend。

正式 GPU backend 已經維持 Replay 的核心 contract：只從目前已寫入的有效範圍均勻抽樣，同一 batch 不重複 slot，並保留 ring buffer、warmup、`terminated` / `truncated` 與 transition dtype。CPU 與 GPU 不要求產生 bit-exact 相同亂數序列，但 sampling 規則不能偷偷改掉。

更重要的是，GPU replay 把「訓練 batch 已經在 GPU 上」這件事變成固定前提。當後續進入 batched rollout、vectorized environments 或更高的 training throughput 時，這條架構比每個 batch 都經過 CPU staging 更容易繼續往前優化。

所以目前最精確的結論不是「GPU replay 比 CPU replay 快」，而是：

> **GPU replay 在目前單環境、batch32 下尚未贏得完整 trainer A/B；它的主要待解問題是逐筆 transition insertion 成本，但它仍是後續 GPU-oriented training architecture 的候選 backend。**

## 真正先拿到 1.50× 的，是 hot path 整理

除了 Replay，我也檢查了每一步都會經過的 hot path，也就是「只要多做一點，就會在數萬甚至數十萬 steps 被放大的成本」。

原本 trainer 每一步都做較頻繁的 diagnostics、CSV flush，PyTorch CPU threads 也沿用較大的預設值。這些事情本身不改 DQN 方程，卻會讓 Python、CPU 與 GPU 不斷同步。

固定同一個 learning config、同一個 10K seed，比較調整前後：

| 10K gate | 原始 hot path | 調整後 hot path |
|---|---:|---:|
| End-to-end SPS | 145.94 | **218.99** |
| Optimizer updates/s | 32.85 | **49.29** |
| Wall-clock | 68.52 s | **45.66 s** |
| CPU thread count | 12 | 1 |
| Diagnostics / CSV flush interval | 1 / 1 | 100 / 100 |

兩邊都完成 `2,251` 次 optimizer update，Target Network sync 與有限值檢查也維持正常。端到端 SPS 提升約 **1.50×**。

這次改善很能說明系統最佳化和 RL 調參的差別：我沒有增加 batch size、沒有降低 update 次數，也沒有偷偷少做訓練工作，只是把不需要每一步同步的事情移出 hot path。

## GPU utilization 不是目標，batch size 也不是免費加速

看到 GPU utilization 不高，很自然會想：那就把 batch 放大。

但 batch size 不只是硬體設定。它代表一次 optimizer update 使用多少 Replay transitions，會同時改變 gradient 的估計方式與 learning dynamics。因此它不能只用「GPU 吃得比較滿」來選。

固定 learning rate、`train_frequency=4`、seed、Replay、epsilon、Target update 與 CUDA device，只比較 batch `32 / 64 / 128` 的 10K profiling：

| Batch size | Environment SPS | Training samples/s | GPU utilization mean | Device memory peak |
|---:|---:|---:|---:|---:|
| 32 | **235.74** | 1,698 | 30.13% | 1.76 GiB |
| 64 | 203.32 | 2,929 | 32.22% | 1.89 GiB |
| 128 | 177.36 | **5,110** | **34.88%** | 1.97 GiB |

batch128 的 GPU utilization 和 training samples/s 都最高，但 environment SPS 反而最低。這正是「GPU utilization 越高，不代表整體訓練越快」的實例。

原因也很直觀：目前仍然只有一個 Breakout environment，action inference 大部分時間一次只處理一個 state。把 optimizer batch 放大，只是讓每次 update 做更多工作，並沒有消除單環境 rollout 與 Python 控制流程的等待。

所以 Day 14 沒有因為 batch128 看起來比較會「吃 GPU」就把它升成正式訓練設定。

[![Day 14 batch size 32、64、128 的 throughput、GPU utilization、power、VRAM 與短跑 learning guardrails](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/batch-size-efficiency.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/batch-size-efficiency.png)

最後再固定 batch32 測 `1 / 2 / 4` 個 PyTorch CPU threads，SPS 分別是 `290.08 / 306.79 / 304.15`，因此後續使用 **2 threads**。差距不巨大，但至少這個選擇來自實測，而不是沿用預設值。

到這裡，Day 14 的 systems 目標就先凍結：不要一邊做 learning-rate experiment，一邊又改 Replay、batch、thread count，否則最後無法知道 learning curve 的差異到底從哪裡來。

## 系統設定凍結後，100K 才開始比較 learning rate

接下來回到真正的 RL 問題。

learning rate 是每次模型更新時，權重往 gradient 指示方向移動多大一步。太小可能學得很慢；太大則可能讓 Q-value、Target 或 gradient 變得不穩定。

這次採用 one-factor-at-a-time：一次只改一個因素。

| Run | Learning rate | 其他主要條件 |
|---|---:|---|
| baseline | `1e-4` | GPU replay、batch32、seed42、environment、epsilon、Target update、reward clipping、FP32 固定 |
| learning-rate-low | `5e-5` | 同上 |
| learning-rate-high | `2e-4` | 同上 |

10K 只做 screening；三組都通過後，才進入 100K main comparison。

而且 100K 不是只看最後一局。訓練過程保留 checkpoint 與 metrics，讓 25K、50K、75K、100K 都能回頭檢查。episode Return 只在一局真正結束時記錄，不會把中間缺少的值補成 0。

為了避免看完結果才改統計規則，這次固定使用：

- 最近 20 個 completed episodes 的 mean / median。
- 20-episode rolling mean。
- recent trend：最後 20 局的後半平均減去前半平均。

## 100K 之後，learning rate 的差異終於開始出現

三個 GPU replay run 都完整跑到 100,000 transitions：

| Run | Episodes | 最近 20 局 mean | Median | 最佳 rolling20 mean | Recent trend Δ | SPS | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 364 | 5.15 | 5.50 | 5.15 | -0.30 | 358.47 | 278.96 s |
| learning-rate-low | 389 | 1.80 | 2.00 | 2.90 | -0.20 | 387.92 | 257.78 s |
| learning-rate-high | 308 | **9.15** | **8.00** | **9.15** | -0.10 | 390.60 | 256.02 s |

這次 `2e-4` 的 high run 在後段明顯高於 baseline 與 low。這已經比 10K 的短跑排名更有資訊，因為我們看到的是一段更長的 learning curve，而不是幾局遊戲的 final mean。

但它仍然只能得到這個結論：

> **在 seed 42、100K budget、目前 GPU replay 與固定 schedule 下，`2e-4` 是值得進一步驗證的 development candidate。**

它還不能被叫作「最佳 learning rate」。下一步仍然需要多個 training seeds 與正式 evaluation protocol。

[![100K main comparison 中三個 GPU replay learning-rate run 的 raw return 與 20-episode rolling mean](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-return-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-return-comparison.png)

淡色點是每個實際完成 episode 的 raw Return，粗線是固定 20-episode rolling mean。x 軸使用 environment step，而不是 episode index，因為不同設定可能讓每局遊戲長度不同。

這張圖最重要的不是「high 贏了」，而是 **100K 開始讓三條曲線真正分開**。這正是 Day 13 的 10K 做不到的事情。

## Q-value 與 Target 變大，不等於已經數值爆炸

Return 告訴我們 Agent 得了多少分，但要判斷「學得慢」和「訓練開始不穩定」，還需要看模型內部的數值。

Q-value 是模型估計「在目前 state 做某個 action，長期大概值多少」；Target 是計算 Bellman learning target 時使用的參考值；gradient norm 則反映一次更新想把模型參數推動多大。

100K 期間三組的 Q 與 Target 都往上建立更大的價值尺度：

| Run | Q mean：25K → 100K | Target mean：25K → 100K |
|---|---:|---:|
| baseline | 0.751 → 1.364 | 0.731 → 1.373 |
| learning-rate-low | 0.555 → 1.155 | 0.553 → 1.193 |
| learning-rate-high | 0.513 → **1.722** | 0.499 → **1.755** |

high 的價值尺度成長最快，這和它較大的 learning rate 一致：更新步幅更大，Q 與 Target 也更快離開接近零的初始區間。

但「Q 變大」本身不是錯誤。真正值得警戒的是：Q / Target 持續失控、TD error 越拉越大、loss 基線一路抬升、gradient 也同步加速，最後出現極端值或 non-finite。

這次 100K 沒看到完整的失控鏈。100K 附近的 `td_error_mean_abs` 約為 baseline `0.054`、low `0.033`、high `0.044`；high 全程最大的 `td_error_max_abs` 約 `1.18`。所有主要 diagnostics 都維持 finite。

因此目前最合理的判讀是：**high learning rate 讓價值估計發展得更快，值得監看，但還沒有足夠證據把它判成數值不穩定。**

[![100K main comparison 的 loss、Q、Target、gradient、epsilon 與 throughput diagnostics](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-diagnostics-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-diagnostics-comparison.png)

這張圖的用途不是再製造一個 ranking，而是確認 Return 差異沒有伴隨明顯的 NaN、持續 gradient explosion 或完全不同的執行成本。

另外，這組設定的 epsilon 在前 10K steps 就降到 `0.05`，之後 90K 大多處在低探索機率。這不是這次 learning-rate comparison 的變因；如果後續要研究探索速度，應該另外做只改 epsilon decay 的受控實驗。

## 從 1K 到 100K，畫面終於開始出現行為差異

Learning curve 是數值證據，但它看不到 Agent 到底在畫面上做了什麼。

因此 Day 14 另外從不同訓練階段的真實 checkpoint 載入 online network，以同一個 evaluation seed、相同 preprocessing、`evaluation epsilon=0` 的 greedy policy 實際執行 Breakout，再把畫面錄成 GIF。

這裡刻意把 evaluation 的探索關掉，因為如果 1K 和 100K 錄影使用不同的隨機 action，畫面差異可能只是 exploration，而不是 policy 本身變了。

1K 與 10K 的 greedy episode 都得到 `0`。50K 與 100K checkpoint 在同一套 evaluation contract 下，分別得到 Return `4` 與 `7`。

這四個數字不能取代正式 evaluation：它們都只是單一 seed、單一 episode。但畫面仍然提供有用的 qualitative evidence。

1K 與 10K 還看不到可重複得分的行為；50K 已經可以看到磚塊出現缺口；100K 則清掉更多磚塊，並在 363 個 evaluation steps 後結束。至少在這個固定 evaluation episode 裡，checkpoint 的行為確實隨訓練進度出現了可觀察變化。

### 1K steps

[![1K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-001k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-001k.gif)

### 10K steps

[![10K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-010k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-010k.gif)

### 50K steps

[![50K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-050k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-050k.gif)

### 100K steps

[![100K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-100k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-100k.gif)

這裡要把兩種 evidence 分開：

- **GIF**：回答「這個 checkpoint 實際會做出什麼行為？」
- **Return curve / 正式 evaluation**：回答「這個 policy 是否真的比較強，而且差異是否能重現？」

看起來比較會玩，不等於已經通過評估。

## 低 Return 不等於模型太小

跑到 100K 之後，很容易產生下一個直覺：如果分數還不夠高，是不是 CNN 或 hidden layer 太小？

這個結論不能太早下。

如果 Return 在 50K → 100K 還持續上升，第一個合理動作通常是延長 training horizon，而不是立刻放大模型。只有當較長訓練開始形成 plateau、數值 diagnostics 又維持正常，而且多個合理 learning-rate 設定都卡在相近的低水平時，才比較有理由懷疑 model capacity。

也就是說，未來若要做容量實驗，應該先固定其他條件，只改一個 architecture 參數，例如比較 hidden dimension `256 / 512 / 1024`。如果 larger model 在相同 training budget 下真的得到更低的 TD error、更高且可重現的 evaluation Return，才有證據說原本的 512 可能限制了模型。

Day 14 目前沒有這種證據。現在看到的是：100K 的 high learning-rate run 仍比 10K 顯示出更明顯的學習訊號，所以「模型太小」還不是第一個應該下的診斷。

## Day 14 真正得到的答案

這一天最後沒有得到一句簡單的「GPU replay 最快」或「`2e-4` 就是最佳參數」。反而得到幾個更有用的工程結論。

第一，**10K 適合做健康檢查，100K 才開始有資格談 learning dynamics。** Day 13 的短跑沒有錯，只是它回答的是 correctness，不是 model selection。

第二，**GPU-resident replay 的 microbenchmark 優勢沒有直接轉成單環境、batch32 的完整 trainer 優勢。** 現在的主要問題是逐筆 transition insertion、單狀態 action inference 與 environment 仍然讓整條 pipeline 保持高度串行。GPU replay 因此是後續架構候選，不是這次 A/B 的 throughput winner。

第三，**GPU utilization 不能單獨當效率指標。** batch128 確實讓 GPU 更忙，也提高 training samples/s，但完整 environment throughput 反而更低。

第四，**100K 讓 learning-rate 差異開始變得可觀察。** 在目前 seed 42 下，`2e-4` 的後段 Return 高於 baseline 與 low，因此它值得交給下一階段驗證；但 single-seed development evidence 仍然不能替代正式 evaluation。

第五，**真實 gameplay GIF 補上了曲線看不到的行為證據。** 1K、10K、50K、100K 的固定 evaluation episode 顯示 policy 行為確實開始改變，但真正判斷模型是否比 random policy 更強，仍然需要多 episode、固定 seeds 的評估流程。

所以 Day 14 最後凍結的不是「最佳模型」，而是一個比較乾淨的下一步：保留 CPU replay reference 與 GPU replay candidate，固定已量測過的 systems 設定，再把 100K 得到的 candidate 交給 Day 15。

Day 15 不再看 training curve 猜模型有沒有學會，而是會把 model 凍結，用固定 evaluation seeds、greedy policy 與 raw Atari score，正式比較 **Random Policy vs DQN**。
