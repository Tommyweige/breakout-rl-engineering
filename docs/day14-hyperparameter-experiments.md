# Day 14｜先讓訓練值得跑久，再看 100K 的 DQN 到底學到什麼

[Day 13](day13-debugging-unstable-rl-training.md) 已經回答了一個很重要的問題：目前這套 DQN trainer 能不能正常工作？

答案是可以。Replay Buffer 會累積資料、optimizer 會更新模型、Target Network 會同步，loss、Q-value 與 gradient 也沒有立刻出現 NaN 或明顯失控。

但「程式能訓練」和「Agent 已經開始學會 Breakout」是兩件不同的事。

Day 13 只跑了 10,000 environment steps，一共完成 48 個 episode。這種規模很適合抓 bug，卻很容易把幾局遊戲的隨機波動誤認成 learning signal。於是 Day 14 的問題變成：

> **如果 10K 還不足以判斷模型是否真的在學，那我要把實驗拉到 100K；但在跑十倍更久之前，先確認目前的訓練系統沒有把時間浪費掉。**

所以這一天其實只有一條主線：

```text
10K 只能確認 trainer 沒壞
        ↓
要觀察 learning，需要更長的 100K
        ↓
長跑之前先整理 training pipeline
        ↓
凍結 systems 設定
        ↓
只改 learning rate，跑 100K
        ↓
看 Return + diagnostics + 真實 gameplay
        ↓
把「看起來有進步」交給 Day 15 正式 evaluation
```

## 10K 是健康檢查，100K 才開始看 learning curve

10K 可以快速檢查：Replay 是否有填充、optimizer 是否真的更新、epsilon 是否照 schedule 下降，以及 loss、Q-value、Target、gradient 是否維持有限值。

但它不適合拿來做成績排名。

Breakout 的 reward 並不是每一步都有，每一局的長度也不同。假設某個設定在 10K 的短期平均 Return 是 `1.15`，另一組是 `1.75`，這個差距可能只是最後幾局剛好打得比較好，還不能說 learning rate 已經造成穩定差異。

因此 Day 14 把兩種用途分開：

- **10K screening**：回答「這組設定能不能正常訓練？」
- **100K main comparison**：回答「更長的 learning curve 是否開始出現可解釋的差異？」

100K 也不是「一定能學會 Breakout」的門檻。它只是把觀察時間拉長，讓幾局遊戲的短期波動比較不容易主導結論。

[![從 Day 13 的 10K diagnostic、10K screening 到 Day 14 100K main comparison 的決策流程](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/25fee37/assets/day14/budget-stages.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/25fee37/assets/day14/budget-stages.png)

這張圖最重要的不是「100K 一定要選出 winner」，而是反過來：**如果 100K 仍然沒有可靠差異，「目前無法分辨」本身就是合法的結果。**

## 跑 100K 之前，先處理一個更現實的問題：時間花去哪裡？

DQN 的 CNN 的確在 RTX 4060 上計算，但完整 trainer 不只包含 neural network。

每一輪還要經過：

```text
environment 產生 observation
        ↓
選 action
        ↓
寫入 Replay Buffer
        ↓
抽樣 training batch
        ↓
DQN forward / target / backward / optimizer
```

其中 environment 與大量 Python 控制流程仍然在 CPU。也就是說：**CUDA 有啟用，不代表 GPU 就會一直忙。**

Day 14 因此先從最明顯的資料路徑開始檢查：Replay Buffer。

## GPU-resident Replay 看起來應該更快，完整 trainer 卻沒有贏

原本的 Replay Buffer 把過去的 transitions 存在 CPU 的 NumPy 陣列。模型更新時，資料要先從 CPU replay 抽出，再經過 host buffer 傳進 GPU。

GPU-resident replay 的想法很直接：讓 Replay 本身常駐 GPU。抽樣 index、gather transitions、把影像從 `uint8` 轉成 `float32`，接著直接進入 DQN update。

如果只看 optimizer-side microbenchmark，這條路徑很漂亮：GPU replay 約能做到 **39K–41K training samples/s**。

但那個數字有一個重要前提：資料已經在 GPU，而且 benchmark 沒有包含真正的 `env.step()`、action selection、episode bookkeeping 與每筆 transition 寫入 Replay 的成本。

所以我又做了一次真正的 full trainer A/B。固定 batch32、`train_frequency=4`、`learning_starts=2048`、模型與 seed，只改 Replay backend：

| Replay path | Environment SPS | Optimizer updates/s | Training samples/s | Wall-clock | GPU utilization mean |
|---|---:|---:|---:|---:|---:|
| CPU + preallocated | **361.61** | **71.92** | **2,302** | **27.65 s** | 23.87% |
| GPU-resident replay | 338.13 | 67.25 | 2,152 | 29.57 s | 24.63% |

SPS（steps per second）代表每秒真正完成多少 environment transitions。

結果很明確：**在目前單一 environment、batch32 的完整 trainer 裡，GPU replay 沒有比較快。**

問題不在 GPU gather 本身，而是在資料的來源。Breakout environment 仍然在 CPU 產生 observation，因此 GPU replay 每一個 environment step 都要把新的 transition 寫進 GPU。對這種很小、很頻繁的 copy，呼叫與同步成本會變得明顯。

換句話說，兩個結果並不矛盾：

```text
GPU replay microbenchmark
→ 回答「資料已在 GPU 時，DQN update 可以多快？」

Full trainer A/B
→ 回答「真正玩 Breakout、收資料、寫 Replay、更新模型時可以多快？」
```

前者證明 GPU-side update path 有潛力；後者則證明目前整條 pipeline 還沒有辦法把這個潛力轉成 end-to-end throughput。

這也成為後續 Day 16 的伏筆：如果要讓 GPU replay 的優勢真正出來，下一步不是單純把 replay 再優化一次，而是要處理 **single environment、batch=1 action inference 與逐筆 GPU insertion** 這三個串行瓶頸。

## 真正先省下時間的，是把 hot path 整理乾淨

Replay A/B 沒有直接帶來加速，但 profiling 讓另一件事變得很清楚：trainer 裡有一些工作根本不需要每一步都做。

原本 diagnostics、CSV flush 與 CPU thread 設定會讓 Python、CPU 與 GPU 更頻繁同步。這些事情不會改變 Bellman target，也不會讓 Agent 多學一筆資料，卻會在數萬、數十萬 steps 中不斷累積成本。

在同一個 learning config、同一個 10K seed 下，把這些 hot-path overhead 降低後：

| 10K gate | 原始 hot path | 調整後 hot path |
|---|---:|---:|
| End-to-end SPS | 145.94 | **218.99** |
| Optimizer updates/s | 32.85 | **49.29** |
| Wall-clock | 68.52 s | **45.66 s** |
| CPU thread count | 12 | 1 |
| Diagnostics / CSV flush interval | 1 / 1 | 100 / 100 |

兩邊都完成 `2,251` 次 optimizer update，Target Network sync 與 finite diagnostics 也維持正常。端到端 SPS 提升約 **1.50×**。

這才是我想要的 systems optimization：**同樣的 DQN schedule、同樣的訓練工作，只是少做不必要的同步。**

## 「讓 GPU 更忙」不是同一件事

接著我又試了一個很直覺的做法：既然 GPU utilization 不高，那把 optimizer batch 放大是不是就好了？

固定 learning rate、`train_frequency=4`、seed、Replay、epsilon 與 Target update，只比較 batch `32 / 64 / 128`：

| Batch size | Environment SPS | Training samples/s | GPU utilization mean | Device memory peak |
|---:|---:|---:|---:|---:|
| 32 | **235.74** | 1,698 | 30.13% | 1.76 GiB |
| 64 | 203.32 | 2,929 | 32.22% | 1.89 GiB |
| 128 | 177.36 | **5,110** | **34.88%** | 1.97 GiB |

batch128 的 GPU utilization 和 training samples/s 都最高，但完整 trainer 的 environment SPS 反而最低。

原因是 batch size 只增加「每次 optimizer update 做多少工作」，它沒有消除單一 Breakout environment、batch=1 action inference 與 Python 控制流程的等待。而且 batch size 本身也會改變 gradient 的估計方式，所以它不是免費的 systems switch。

[![Day 14 batch size 32、64、128 的 throughput、GPU utilization、power、VRAM 與短跑 learning guardrails](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/batch-size-efficiency.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/batch-size-efficiency.png)

這張圖帶來的結論不是「batch32 最會學」，而是更窄的一句話：**在目前單環境 trainer 裡，batch32 的 end-to-end environment throughput 最高；更大的 batch 雖然讓 GPU 做更多 training samples，卻沒有讓整個遊戲訓練更快。**

最後固定 batch32 測 `1 / 2 / 4` 個 PyTorch CPU threads，SPS 分別是 `290.08 / 306.79 / 304.15`，因此後續採用 **2 threads**。

到這裡就停止改 systems 變數。原因很簡單：如果接下來一邊改 learning rate、一邊又換 Replay、batch、thread count，最後 learning curve 有差異時，就不知道是誰造成的。

## 系統設定凍結後，才正式跑 100K learning-rate experiment

現在終於回到 Day 14 真正想看的 RL 問題：**當 observation horizon 拉到 100K，不同 learning rate 會不會開始走出不同的 learning curve？**

這次採 one-factor-at-a-time，只改 learning rate：

| Run | Learning rate | 其他主要條件 |
|---|---:|---|
| baseline | `1e-4` | GPU replay、batch32、seed42、environment、epsilon、Target update、reward clipping、FP32 固定 |
| learning-rate-low | `5e-5` | 同上 |
| learning-rate-high | `2e-4` | 同上 |

這裡固定使用 GPU replay，不是因為它已經證明比 CPU replay 更快，而是因為它是這輪選定的 development backend；三個 learning-rate runs 必須使用同一條 data path，才能把 learning rate 當成主要變因。

10K 只負責 screening。三組都通過後才進 100K main comparison，而且過程中保留 checkpoint 與 metrics，讓 25K、50K、75K、100K 都能回頭檢查。

統計規則也先固定，不看完結果才挑最好看的 window：

- 最近 20 個 completed episodes 的 mean / median。
- 20-episode rolling mean。
- recent trend：最後 20 局的後半平均減去前半平均。

## 100K 之後，三條 learning curve 終於開始分開

三個 GPU replay runs 都完成 100,000 transitions：

| Run | Episodes | 最近 20 局 mean | Median | 最佳 rolling20 mean | Recent trend Δ | SPS | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 364 | 5.15 | 5.50 | 5.15 | -0.30 | 358.47 | 278.96 s |
| learning-rate-low | 389 | 1.80 | 2.00 | 2.90 | -0.20 | 387.92 | 257.78 s |
| learning-rate-high | 308 | **9.15** | **8.00** | **9.15** | -0.10 | 390.60 | 256.02 s |

`2e-4` 的 high run 在後段高於 baseline 與 low。和 10K 相比，這次看到的不再只是幾局遊戲的短期平均，而是一段更長的 learning curve 開始分化。

但目前能下的結論仍然很有限：

> **在 seed 42、100K budget、目前固定 training backend 下，`2e-4` 是值得進一步驗證的 development candidate。**

它還不能被叫作「最佳 learning rate」，因為這仍然只有一個 training seed。

[![100K main comparison 中三個 GPU replay learning-rate run 的 raw return 與 20-episode rolling mean](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-return-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-return-comparison.png)

淡色點是每個真正完成 episode 的 raw Return，粗線是固定 20-episode rolling mean。x 軸使用 environment step，而不是 episode index，因為不同設定的一局遊戲長度可能不同。

這張圖最重要的不是「high 贏了」，而是 **100K 終於讓 10K 看不清楚的差異開始浮出來。**

## Return 變高之後，還要確認不是數值一起失控

Return 只能告訴我們 Agent 得了多少分，不能單獨回答模型內部是否健康。

因此同一組 100K runs 還要一起看 Q-value、Target、TD error 與 gradient。

100K 期間三組的 Q 與 Target 都建立了更大的價值尺度：

| Run | Q mean：25K → 100K | Target mean：25K → 100K |
|---|---:|---:|
| baseline | 0.751 → 1.364 | 0.731 → 1.373 |
| learning-rate-low | 0.555 → 1.155 | 0.553 → 1.193 |
| learning-rate-high | 0.513 → **1.722** | 0.499 → **1.755** |

high 的價值尺度成長最快。這和較大的更新步幅方向一致，但 **Q-value 變大本身不是錯誤，也不是 performance 指標。**

真正要警戒的是一條連鎖現象：Q / Target 持續失控、TD error 越拉越大、loss 基線抬升、gradient 同步加速，最後出現極端值或 non-finite。

這次 100K 沒看到完整的失控鏈。100K 附近的 `td_error_mean_abs` 約為 baseline `0.054`、low `0.033`、high `0.044`；high 全程最大的 `td_error_max_abs` 約 `1.18`，主要 diagnostics 仍維持 finite。

因此目前比較合理的說法是：**high learning rate 讓價值估計發展得更快，值得繼續監看，但沒有足夠證據把它判成數值不穩定。**

[![100K main comparison 的 loss、Q、Target、gradient、epsilon 與 throughput diagnostics](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-diagnostics-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-diagnostics-comparison.png)

另外，這組設定的 epsilon 在前 10K steps 就降到 `0.05`，後面 90K 大多處在低探索機率。這不是這次 learning-rate comparison 的變因；如果之後要研究 exploration schedule，應該另外做只改 epsilon decay 的實驗。

## 曲線之外，我還想知道 Agent 實際在畫面上做了什麼

Learning curve 與 diagnostics 都是數字，但它們看不到 paddle 到底有沒有開始對球做出比較合理的反應。

所以 Day 14 從不同訓練階段的 checkpoint 載入 online network，以同一個 evaluation seed、相同 preprocessing、`evaluation epsilon=0` 的 greedy policy 真正執行 Breakout，再把畫面錄成 GIF。

這裡把 exploration 關掉，是為了讓 1K、10K、50K、100K 的畫面差異盡量來自 policy 本身，而不是錄影時剛好抽到不同 random actions。

結果是：

- **1K**：Return `0`
- **10K**：Return `0`
- **50K**：Return `4`
- **100K**：Return `7`

這些都只是同一套 evaluation contract 下的單一 seed、單一 episode，所以不能取代正式 evaluation。但它們可以作為 qualitative evidence：至少能直接看到 checkpoint 的實際行為如何改變。

### 1K steps

[![1K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-001k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-001k.gif)

### 10K steps

[![10K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-010k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-010k.gif)

### 50K steps

[![50K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-050k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-050k.gif)

### 100K steps

[![100K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-100k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-100k.gif)

1K 與 10K 還看不到可重複得分的行為；50K 已經可以看到磚塊出現缺口；100K 則清掉更多磚塊，並在 363 個 evaluation steps 後結束。

但這裡仍然要把兩種 evidence 分清楚：

```text
GIF
→ 回答「這個 checkpoint 實際會做出什麼行為？」

正式 evaluation
→ 回答「這個 policy 是否真的比較強，而且差異是否能重現？」
```

看起來比較會玩，不等於已經通過評估。

## 100K 分數還不高，不代表模型一定太小

看到 100K 的 Return 還不算高，很容易直接猜：是不是 CNN 或 hidden layer 不夠大？

目前還不能這樣判斷。

如果 Return 在 50K → 100K 仍然往上走，第一個合理動作通常是繼續觀察更長的 training horizon，而不是立刻把模型放大。只有當較長訓練形成 plateau、diagnostics 又正常，而且多個合理 hyperparameter 設定都卡在相近低水平時，才比較像 model capacity 的問題。

如果未來真的要測 capacity，也應該一次只改一個 architecture 參數，例如 hidden dimension `256 / 512 / 1024`，而不是同時把 CNN channels 與 FC layer 全部放大。

Day 14 現在看到的是「100K 比 10K 更明顯地出現 learning signal」，不是「模型容量已經撞牆」。

## Day 14 最後其實只留下兩個答案

走到這裡，前面的 systems experiments、100K learning curves、diagnostics 與 GIF 可以收斂成兩個很清楚的答案。

### 第一個答案：DQN 的確開始出現值得驗證的 learning signal

10K 只能證明 trainer 沒壞；拉到 100K 後，learning-rate runs 才開始分化。在目前 seed 42 下，`2e-4` 的後段 Return 較高，Q / Target / TD error / gradient 又沒有形成明顯數值失控鏈。

這足以讓它成為下一階段的 **candidate**，但不夠讓它變成「最佳設定」。

所以 Day 15 的工作會很直接：凍結 model，用固定 evaluation seeds、greedy policy 與 raw Atari score，正式比較 **Random Policy vs DQN**。

### 第二個答案：現在真正限制 GPU 的，不是 Replay sampling 本身

GPU-resident replay 的 optimizer-side microbenchmark 很快，但 full trainer 沒有因此變快；更大的 batch 也只是讓 GPU utilization 上升，沒有提高 environment throughput。

這把 systems bottleneck 指向一個更具體的方向：

```text
single environment
+ batch=1 action inference
+ per-transition GPU insertion
```

所以 GPU replay 不會因為這次 A/B 沒贏就被丟掉。它會保留成後續 GPU-oriented backend，而 Day 16 再正式處理：

```text
Vectorized Environments
        ↓
Batched Action Inference
        ↓
Batched GPU Replay Insertion
```

Day 14 的價值因此不是「找到最快 GPU 設定」或「找到最佳 learning rate」，而是把兩件原本混在一起的問題分開了：

> **系統跑得快不代表模型學得好；模型看起來在學，也不能用單一 seed 就宣稱成功。**

下一篇先把第二件事做嚴格：[Day 15 — DQN Milestone Evaluation](day15-dqn-milestone-and-evaluation.md)。