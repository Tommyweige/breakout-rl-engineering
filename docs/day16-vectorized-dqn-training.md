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

observation 是 Agent 目前看到的遊戲畫面；Q-value 則是 DQN 對「現在選某個 action，未來可能有多值得」的估計。GPU 很擅長一次處理很多資料，但這條流程卻像拿一條八線道高速公路，每次只放一台車上去。

Day 16 要回答的問題因此很直接：

> **如果同時跑多個 Breakout，把多張 observation 合成一批再交給 GPU，完整訓練到底會不會真的變快？**

## 從一局遊戲，變成一批遊戲

這次加入的是向量化環境（vectorized environment）。它不是把程式複製四份，而是讓同一個介面同時管理多個彼此獨立的 Breakout。

假設同時跑 4 個環境：

```text
Env 0 ─┐
Env 1 ─┤
Env 2 ─┤→ 4 個 observations
Env 3 ─┘
        ↓
   一次送進 DQN
        ↓
   4 組 Q-values
        ↓
   4 個 actions
```

每個 Breakout 的 observation 仍然是 4 張最近的 84×84 灰階畫面，所以單一輸入原本是 `(4, 84, 84)`。四個環境一起跑後，模型看到的是 `(4, 4, 84, 84)`：最前面的 `4` 才是這次同時處理的環境數量。

這就是批次推論（batched inference）。原本四個環境需要分別呼叫模型四次，現在可以一次完成。

Replay Buffer 也採用同樣的想法。transition 是 Agent 一次互動留下的紀錄，包含目前 state、action、reward、next state 和 episode 是否結束。以前每拿到一筆 transition 就做一次小型 GPU 寫入，現在可以把同一輪產生的多筆資料一起寫進 Replay。

整條資料流變成：

[![單一環境與向量化 DQN trainer 的資料流](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/vectorized-pipeline.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/vectorized-pipeline.png)

Day 14 做的是「把資料放到 GPU」；Day 16 再往前一步，開始處理另一個更重要的問題：**怎麼讓 GPU 每次真的拿到足夠多的工作。**

## 多開環境之後，訓練步數不能算錯

Vectorization 最容易讓人誤會的一點，是「跑一次迴圈」不再等於「得到一筆資料」。

如果同時跑 4 個環境，一次 environment step 會產生 4 筆 transition；同時跑 8 個，就會產生 8 筆。因此 Day 16 的 `global_step` 仍然代表真正收集了多少筆 environment transitions，而不是 Python 迴圈跑了幾次。

這件事很重要，因為 DQN 的 epsilon 探索率、何時開始學習、多久更新一次網路、多久同步一次 target network，全部都和 transition 數量有關。如果把 8 筆資料誤算成 1 步，表面上只是計數方式不同，實際上連演算法的訓練節奏都會一起改掉。

多個環境也各自有自己的 episode。一局結束時，只能重設那一局；Replay 裡保存的 `next_state` 也必須是上一局真正最後看到的畫面，而不是 reset 之後新遊戲的第一張畫面。

這些細節不需要改變我們對 vectorization 的直覺：**可以把資料批次化，但不能因為追求速度，把原本 DQN 正在學的 transition 關係一起改掉。**

Day 15 固定下來的 Contract v2 也繼續沿用，所以 frame skip、frame stack、sticky action 和開局／掉命後的 FIRE 規則都沒有因為 Day 16 而換掉。

## 先跑 10K：環境愈多，真的愈快嗎？

接著才真正量速度。

我固定 Vanilla DQN、GPU Replay、batch size 32、training seed 42 和同一套 Contract v2，只改同時跑幾個 Breakout，讓每組都收集 10,000 筆實際 transitions。

| 同時環境數 | transitions/s | 相對 N=1 |
|---:|---:|---:|
| 1 | 298.20 | 1.00× |
| 2 | 387.89 | 1.30× |
| 4 | 456.63 | 1.53× |
| 8 | 483.30 | 1.62× |

[![1、2、4、8 個環境在相同 10K transition budget 下的吞吐與 wall-clock](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/vectorized-throughput.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/vectorized-throughput.png)

第一眼看起來答案很簡單：N=8 最快，從 298.20 提高到 483.30 transitions/s，約是 **1.62 倍**。

但這裡有一個很重要的轉折。

目前 DQN 每收集 4 筆 transition 就會做一次更新。N=1、2、4 都能讓一批 action 在下一次網路更新之前完整走完；N=8 則會一次先替 8 個環境選好 action，中間第 4 筆資料完成後網路已經更新，但後面幾個 action 仍然來自更新前的模型。

它不是資料壞掉，也不代表 N=8 不能訓練；只是它已經不再是最乾淨的「只改 systems、其他節奏盡量不變」比較。因此 N=8 很適合告訴我們吞吐上限在哪裡，卻不適合只因為 SPS 最大就直接成為後續正式 backend。

這也是 RL 工程和一般 inference benchmark 很不一樣的地方：**最快的設定，不一定是最適合拿來做演算法實驗的設定。**

## 真正省下來的是大量小工作

為什麼多環境會有效？最直接的證據就在 model forward 次數。

同樣收集 10,000 筆 transition：

```text
N=1 → 10,000 次 action inference
N=4 →  2,500 次
N=8 →  1,250 次
```

GPU 並沒有少算那些遊戲畫面，而是把原本大量「叫一次模型、只算一張」的零碎工作合併成較大的 batch。

[![不同 environment count 的 batched inference throughput 與單次 forward 成本](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/batched-inference.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/batched-inference.png)

Replay 寫入也看到相同現象。單獨量一次寫入 1、2、4、8、16 筆 transition 時，吞吐從約 5,000 transitions/s 一路提高到接近 49,000 transitions/s。

[![batch size 1、2、4、8、16 的 Replay insertion microbenchmark](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/replay-insertion.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/0e345d1d053297fd77865fdc5ef8a9f850fe5b98/assets/day16/replay-insertion.png)

不過這張圖不能解讀成「完整 trainer 也快十倍」。Day 14 已經示範過一次：某個局部元件的 microbenchmark 很漂亮，不代表整條 pipeline 會同比例提升。

Breakout 的 ALE environment 仍然主要在 CPU 上往前跑，optimizer update 也還有自己的成本。這次使用的 `SyncVectorEnv` 是把多個環境整理成同一個批次介面，並沒有把 ALE 本身變成真正的多進程平行模擬。

所以 Day 16 真正優化的是：**減少大量 batch=1 的模型呼叫和零碎 GPU 資料搬移。**

## 10K 最快，到了 100K 不一定還是同一個答案

10K 很適合做 systems screening，但還太短，不適合只看這個結果就決定後面幾十萬、幾百萬 transitions 全部用哪個設定。

因此接著重新從隨機初始化開始，讓 N=1、N=2 和 N=4 都跑到 100,000 transitions。

| 環境數 | transitions/s | 跑完 100K | 相對 N=1 |
|---:|---:|---:|---:|
| 1 | 238.67 | 419.00 s | 1.00× |
| 2 | **380.74** | **262.65 s** | **1.60×** |
| 4 | 368.06 | 271.70 s | 1.54× |

結果出現一個比「N=8 10K 最快」更有價值的現象：**跑長一點後，N=2 反而超過 N=4。**

也就是說，短時間 benchmark 看到的排序不一定會完整延續到比較長的訓練。當 workload 變長，episode reset、CPU stepping、GPU update 和其他固定成本所佔的比例都可能改變。

這也是為什麼正式做 training systems selection 時，不能只抓一個最好看的 10K SPS 數字。

## 快之外，還要確認模型真的有在學

速度只是 Day 16 的一半。

100K checkpoint 接著使用同一套 Contract v2、固定的 15 個 evaluation seeds、`epsilon = 0` 和 raw Atari reward 評估。這裡不是要用 15 局決定哪個演算法比較強，而是確認 systems optimization 沒有讓訓練明顯失效。

| Run | 平均 raw return | 中位數 | TimeLimit |
|---|---:|---:|---:|
| Random baseline | 1.73 | 2 | 0/15 |
| N=1 | **9.00** | 9 | 0/15 |
| N=2 | **6.07** | 6 | 0/15 |
| N=4 | 2.33 | 2 | 0/15 |

這個結果不能解讀成「N=1 一定比 N=2 好」，因為這裡只有一個 training seed，目的也不是做 model-family ranking。

但它足以提醒我們另一件很重要的事：**training throughput 和 policy quality 是兩個不同指標。**

N=4 在 10K systems screening 很漂亮，可是 100K 的固定評估只拿到 2.33；N=2 不只在 100K 更快，平均 return 也有 6.07，明顯高於 Random baseline 的 1.73。三個訓練版本的 15 局也都正常結束，沒有出現 TimeLimit。

因此 Day 16 最後沒有選 10K 最快的 N=8，也沒有選 10K strict 設定中最快的 N=4，而是選擇 **N=2** 作為後續的 vectorized training backend。

這不是在宣稱 N=2 是所有硬體、所有 seed、所有演算法下的最佳設定；它只是目前這台 RTX 4060 Laptop GPU、這套 DQN 訓練節奏與 Contract v2 下，速度、訓練語意和 100K guardrail 之間最合理的折衷。

N=1 仍然保留作為 model-quality reference。後面真正比較 DQN、Double DQN 或 Dueling 時，仍然需要用多個 training seeds 和更長的 transition budget 才能談模型優劣。

## Day 16 真正學到的不是「多開幾個遊戲」

如果只看表面，Day 16 好像只是把一個 Breakout 改成同時跑兩個、四個、八個。

但真正重要的是一個更通用的 GPU 工程觀念：

> **程式碼裡出現 `cuda`，不代表 GPU 就會自動被有效利用。**

Day 14 已經把資料搬到 GPU，Day 16 則讓我們看到另一半問題：如果每次只做一點點工作，GPU 仍然會被大量小型呼叫拖住。

把多個 environment 的 observations 合成 batch 後，同樣數量的 transitions 可以用更少的 model calls 和更大的 Replay writes 完成。在這台機器上，最後選出的 N=2 在 100K 訓練中把 throughput 從 238.67 提高到 380.74 transitions/s，約 **1.60×**。

而且這一天也留下另一個之後會反覆用到的原則：

> **系統變快，不等於模型一定學得更好；局部 benchmark 變快，也不等於完整 trainer 會同比例變快。**

Day 16 到這裡把後續訓練需要的 systems backend 固定下來。下一步開始，我們可以暫時不再動資料收集方式，回到 DQN 本身的演算法問題：`max` 為什麼可能偏向被高估的 Q-value，以及 Double DQN 為什麼要把 action selection 和 value evaluation 拆開。