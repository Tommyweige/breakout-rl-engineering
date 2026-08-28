# Day 16｜GPU 不是放著就會變快：一次讓多個 Breakout 一起跑

Day 14 留下了一個很反直覺的結果。

我們已經把 Replay Buffer 搬到 GPU，也確認單獨做資料抽樣和 DQN 更新時確實很快；但把整個 Breakout 訓練流程跑起來，速度卻沒有跟著大幅提升。

原因不是 RTX 4060 不夠快，而是我們一直在用一種很浪費 GPU 的方式工作：**每次只送一份資料進去。**

原本的流程大概是：

```text
1 個 Breakout
→ 產生 1 個 observation
→ GPU 算 1 組 Q-values
→ CPU 讓遊戲往前一步
→ 寫入 1 筆 transition
→ 再來一次
```

GPU 很擅長一次處理很多資料，但這條流程卻像拿一條八線道高速公路，每次只放一台車上去。

所以 Day 16 的問題其實很單純：

> **如果同時跑多個 Breakout，把多張 observation 合成一個 batch，再一次交給 GPU，完整訓練到底會不會真的變快？**

## 從一局遊戲，變成一批遊戲

這次加入的是向量化環境（vectorized environment）。它不是把同一段程式複製好幾份，而是讓一個介面同時管理多個彼此獨立的 Breakout。

假設同時跑 4 個環境，原本四次分開做的 inference，可以改成一次：

```text
4 個 observations
→ 組成一個 batch
→ DQN forward 一次
→ 得到 4 組 Q-values
→ 各自選出 4 個 actions
```

對模型來說，輸入會從一份 `(4, 84, 84)` 的 observation，變成 `(N, 4, 84, 84)`。第一個 `N` 就是同時跑幾個環境。

Replay Buffer 也用同一個思路：不再每產生一筆 transition 就做一次小型 GPU 寫入，而是一次放進多筆。

整個資料流可以簡化成下面這張圖：

[![Day 16 向量化 DQN Trainer 資料流](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/bfa50d87e4d51c5cf450ec89e023144ebe46ab64/assets/day16/vectorized-pipeline-reader.svg?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/bfa50d87e4d51c5cf450ec89e023144ebe46ab64/assets/day16/vectorized-pipeline-reader.svg)

這張圖真正想表達的只有一件事：**多個環境一起產生資料，讓 GPU 一次吃一批，而不是一直處理 batch=1 的零碎工作。**

## 多開環境後，不能把「一步」算錯

向量化最容易讓人誤會的地方，是 `step` 的意義。

單一環境時，呼叫一次 `env.step()`，通常得到一筆 transition；如果同時跑 4 個環境，一次 vector step 就會得到 4 筆 transition。

所以訓練進度不能寫成：

```text
vector loop 跑一次
→ global_step + 1
```

而要按照真正收到的資料量計算。

這很重要，因為 epsilon decay、多久更新一次網路、target network 何時同步、checkpoint 的 100K 到底代表多少資料，都依賴 transition 數量。

另外，多個環境的 episode 也必須彼此獨立。假設四局裡只有 Env 0 game over，只能重設 Env 0，其他三局還要繼續玩；上一局最後一張畫面也不能被下一局 reset 後的畫面取代。

這些 correctness 細節不是 Day 16 的主角，但它們是前提：**如果多環境讓資料本身變錯，再高的 SPS 都沒有意義。**

## 先跑 10K：環境越多，真的越快嗎？

把資料流改完後，我先不碰新的 RL 演算法，只用同一套 Vanilla DQN、GPU Replay、batch size 32、training seed 42 和 Contract v2，分別測 1、2、4、8 個環境。

每一組都只跑 10,000 個實際 transitions。

| 同時環境數 | 10K throughput |
|---:|---:|
| N=1 | 298.20 transitions/s |
| N=2 | 387.89 transitions/s |
| N=4 | 456.63 transitions/s |
| N=8 | 483.30 transitions/s |

[![1、2、4、8 個環境在相同 10K transition budget 下的吞吐與 wall-clock](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/vectorized-throughput.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/vectorized-throughput.png)

結果很明顯：從 N=1 增加到 N=4，完整 trainer 的 throughput 提升非常明顯；但從 N=4 再翻倍到 N=8，收益已經開始變小。

這代表 batching 確實有用，但它不是無限擴張的免費加速。

原因也很好理解。環境數增加之後，GPU inference 的零碎呼叫變少了，但 ALE 本身仍然主要在 CPU 上執行，DQN optimizer update 也沒有因為環境變多就突然變便宜。原本的瓶頸被削弱後，下一個瓶頸自然就會浮出來。

而且 N=8 還有另一個考量：目前 DQN 每 4 筆 transition 更新一次網路，N=8 一次先替 8 個環境選完 action，會跨過一次網路更新點。這不代表資料錯了，但它和單一環境逐步選 action 的節奏不完全相同。

所以 Day 16 不能只看「483 最大」就宣布 N=8 勝出。

## 真正省下來的是大量小工作

這次加速最直接的來源，是 model forward 次數下降。

同樣收集 10K transitions：

```text
N=1 → 10,000 次 action inference
N=2 →  5,000 次
N=4 →  2,500 次
N=8 →  1,250 次
```

GPU 還是做同一個 DQN，但每次處理的 observation 更多，所以不必一直付出啟動一次小 inference 的固定成本。

Replay insertion 也有相同現象。獨立量測時，一次寫入越多 transition，每筆資料平均分攤到的固定成本越低。

[![batch size 1、2、4、8、16 的 GPU Replay insertion microbenchmark](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/replay-insertion.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/replay-insertion.png)

這裡也再次驗證 Day 14 的教訓：**microbenchmark 很快，不代表完整 trainer 一定同比例變快。**

真正有意義的數字，還是整條 training pipeline 最後每秒能處理多少 transitions。

## 10K 很快還不夠，所以再跑 100K

10K 很適合做 systems screening，但它太短，不能直接拿來決定後面幾十萬、幾百萬 transitions 要使用哪個 backend。

因此最後又重新從隨機初始化開始，讓 N=1、N=2 和 N=4 都跑到 100,000 transitions。

| Backend | 100K throughput | 跑完約需時間 |
|---:|---:|---:|
| N=1 | 238.67 transitions/s | 419.00 s |
| N=2 | **380.74 transitions/s** | **262.65 s** |
| N=4 | 368.06 transitions/s | 271.70 s |

N=2 相對 N=1 約快 **1.60 倍**，而且這次甚至比 N=4 稍快。

[![100K vectorized training throughput 與 wall-clock](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/vectorized-100k-vectorized-throughput.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/vectorized-100k-vectorized-throughput.png)

這個結果很值得注意，因為它提醒我們：**10K 最快的設定，不一定就是 100K 最適合的設定。**

短跑裡 N=8 最大、N=4 很接近；拉長之後，N=2 反而成為這台機器上最合理的折衷。

## 速度變快，不代表模型一定學得一樣

Day 16 最後還做了一個很重要的檢查：把 100K checkpoint 放進同一套 Contract v2 evaluation，用固定的 15 個 seeds、`epsilon=0` 和 raw Atari reward 評估。

結果是：

| Run | 平均 raw return | TimeLimit |
|---|---:|---:|
| Random baseline | 1.73 | 0/15 |
| N=1 | **9.00** | 0/15 |
| N=2 | 6.07 | 0/15 |
| N=4 | 2.33 | 0/15 |

這裡最重要的不是誰分數最高，而是我們不能把「trainer 變快」和「policy 品質完全等價」混成同一件事。

N=2 的 100K throughput 最好，而且固定 evaluation 中沒有出現 environment deadlock 或 TimeLimit failure；但它的平均 return 仍低於 N=1。

所以 Day 16 的結論不是：

> N=2 訓練出來的模型一定比 N=1 好。

而是：

> **N=2 是目前選出的 systems backend；N=1 則保留成 model-quality reference。**

後面如果比較 DQN、Double DQN、Dueling Double DQN，就要讓所有候選使用同一個 backend。這樣才能把變因留給演算法，而不是一邊換模型、一邊又換訓練系統。

## Day 16 最後得到的是什麼？

Day 16 沒有換新的 RL 演算法，也沒有讓 RTX 4060 的使用率突然衝到 100%。

真正改變的是餵資料的方式：

```text
以前：
1 個 observation
→ 1 次 inference
→ 1 筆 Replay 寫入

現在：
多個 observations
→ 1 次 batched inference
→ 一批 transitions
→ batched Replay insertion
```

在這台 RTX 4060 Laptop GPU 上，最後選出的 N=2 backend 在 100K 實驗中從 N=1 的 238.67 transitions/s 提升到 380.74 transitions/s，約 **1.60×**。

這一天最值得記住的其實不是 N=2 這個數字，而是另一件事：

> **程式碼裡出現 `cuda`，不代表 GPU 就會自動把整個系統加速。GPU 真正擅長的是一次處理足夠多的工作，而 batching 就是在重新整理資料流，讓硬體有機會發揮。**

現在 training backend 已經有了比較明確的選擇，下一步才開始回到演算法本身：Vanilla DQN 的 `max` 為什麼可能偏向被高估的 Q-value？Double DQN 又是怎麼修改 target 計算來處理這個問題？
