# Day 16｜GPU 不是放著就會變快：一次讓多個 Breakout 一起跑

Day 14 有一個很反直覺的結果。

我們把 Replay Buffer 搬到 GPU 後，單獨量資料抽樣和 DQN update 的速度，確實很快；但把整個 Breakout 訓練流程跑起來，GPU 版本卻沒有因此贏過原本的 CPU Replay。

原因後來愈來愈清楚：**不是 GPU 算得不夠快，而是我們每次只丟給它一點點工作。**

原本的訓練流程大概是：

```text
1 個 Breakout environment
↓
產生 1 個 observation
↓
GPU 幫這 1 個 observation 算 Q-values
↓
CPU 讓遊戲往前一步
↓
寫入 1 筆 transition
↓
再重複一次
```

這就像有一條很寬的高速公路，但每次只放一台車上去。GPU 明明可以同時處理很多資料，卻一直在等 CPU 的遊戲環境慢慢產生下一張畫面。

Day 16 要解的，就是這個問題。

> **如果一次同時跑 2、4、8 個 Breakout，讓它們一起產生 observation，再一次送進 GPU，完整訓練到底會不會真的變快？**

而且這次不能只追求速度。Day 15 已經把 Breakout 規則固定成 Contract v2：frame skip 是 4、frame stack 是 4、sticky action probability 是 0.25，開局與掉命後的必要 FIRE 由環境統一處理。Day 16 要比較的是 training system，不是偷偷換一套遊戲規則。

## 一次跑多個環境，不是把迴圈複製八份

假設同時跑 4 個 Breakout：

```text
Env 0 ─┐
Env 1 ─┤
Env 2 ─┤→ 4 個 observations
Env 3 ─┘
        ↓
   (4, 4, 84, 84)
        ↓
      DQN
        ↓
   4 組 Q-values
        ↓
   4 個 actions
```

第一個 `4` 是 environment 數量；後面的 `(4, 84, 84)` 則是每個 Agent 看到的四張 84×84 灰階畫面。

真正的重點是：**神經網路只 forward 一次。**

不是在 Python 裡對四個 environment 各呼叫一次模型，而是把四份 observation 疊成一個 batch，一次交給 CUDA。這就是 batched inference，可以理解成「一次處理一批資料」。

Replay Buffer 也用同樣的思路。以前每產生一筆 transition，就做一次小型 GPU 寫入；現在可以一次寫入一批。

完整資料流如下：

[![單一環境與向量化 DQN trainer 的資料流](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/vectorized-pipeline.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/vectorized-pipeline.png)

看起來只是「多開幾個遊戲」，但真正麻煩的地方反而不是 GPU，而是：**怎麼確保多開之後，DQN 還是在按照原本的訓練節奏學習。**

## `global_step` 不能再理解成「跑了幾次迴圈」

單一環境時：

```text
step 一次
→ 1 筆 transition
```

所以 `global_step += 1` 很自然。

但如果同時跑 8 個環境：

```text
vector step 一次
→ 8 筆 transitions
```

這時真正的訓練進度應該增加 8，而不是 1。

這個差異非常重要，因為 DQN 裡很多事情都依賴步數：什麼時候開始學習、每隔多少筆資料更新一次網路、epsilon 探索率下降到哪裡、target network 何時同步，以及 checkpoint 的 10K、100K 到底代表多少資料。

例如目前 `train_frequency = 4`。如果一次 vector step 產生 8 筆 transition，它其實跨過了第 4 筆和第 8 筆兩個更新點。若程式只是寫成「每個 vector step 更新一次」，更新次數就會直接少一半。

所以 Day 16 不把 vector iteration 當成訓練步數，而是仍然用**實際收到的 environment transitions**計數。

這也是這次 1、2、4、8 個環境的 10K 實驗，最後全部都得到相同 `2,251` 次 optimizer update 和 `21` 次 target sync 的原因。

速度變了，但更新頻率沒有因為多開環境而偷偷縮水。

## 多個遊戲一起跑，最怕把不同 episode 接錯

另一個比速度更危險的問題，是 episode 邊界。

假設 Env 0 已經 game over，但 Env 1、2、3 還在玩。這時只能重設 Env 0，其他環境不能一起 reset。

更麻煩的是，有些 vector environment 會在一局結束後自動重設。如果程式沒注意，Replay 可能會記成：

```text
上一局最後一張畫面
→ action
→ 下一局剛 reset 的第一張畫面
```

但這個 transition 在真實遊戲裡根本不存在。

因此目前的做法是先保存真正的 final observation，把這筆 transition 寫進 Replay，再只 reset 已經 `terminated` 或 `truncated` 的 environment。

Day 15 定下來的 FIRE 規則也要各自獨立。某個 environment 剛掉命，需要由環境處理必要的 FIRE；另外幾個 environment 可能還在正常打球，不能一起被影響。

而且如果 policy 原本要求 `RIGHT`，但環境的 serve wrapper 這一步改成 `FIRE`，Replay 裡必須保存 wrapper 最後送進環境的 `FIRE`，否則模型之後會拿錯誤的 `(state, action, reward)` 關係來學習。

這些細節看起來不像 GPU 加速，但它們才是 RL systems 最容易出錯的地方：**速度可以量，錯掉的 transition 卻不一定會立刻報錯。**

## 真正跑起來後，4 個環境已經吃到大部分收益

接著才是 Day 16 最想知道的數字。

在同一張 RTX 4060 Laptop GPU 上，固定 Vanilla DQN、GPU Replay、batch size 32、seed 42 和 Contract v2，分別跑：

```text
N = 1
N = 2
N = 4
N = 8
```

每組都使用 10,000 個實際 environment transitions，先比較完整 trainer 的短跑速度。

| 同時環境數 | transitions/s | 跑完 10K 約需 | action inference 次數 | Replay 寫入次數 | 平均 GPU util. |
|---:|---:|---:|---:|---:|---:|
| 1 | 214.99 | 48.10 s | 10,000 | 10,000 | 40.6% |
| 2 | 232.98 | 43.13 s | 5,000 | 5,000 | 41.3% |
| 4 | 310.26 | 32.46 s | 2,500 | 2,500 | 41.4% |
| 8 | 318.57 | 31.59 s | 1,250 | 2,500 | 45.2% |

[![1、2、4、8 個環境在相同 10K transition budget 下的吞吐與 wall-clock](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/vectorized-throughput.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/vectorized-throughput.png)

從 N=1 到 N=8，完整 trainer 的 throughput 從 214.99 提高到 318.57 transitions/s，大約是 **1.48 倍**。

但更有意思的是：

```text
N=4：310.26
N=8：318.57
```

把環境數再翻倍，速度只多了約 2.7%。

所以這次不是得到「環境愈多愈好」的結論，而是看到一個很典型的系統現象：**batching 一開始可以快速攤薄固定成本，但到某個點之後，新的瓶頸就會浮出來。**

N=4 已經取得大部分收益，因此後續真正要選正式 backend 時，不能只看誰的 SPS 最大，還要一起考慮訓練節奏與實作複雜度。

## 速度主要不是來自「GPU 使用率暴增」

如果只看 GPU utilization，這次其實沒有從 40% 突然跳到 90%。

[![不同 environment count 的 GPU、CPU 與功耗量測](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/system-utilization.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/system-utilization.png)

真正明顯下降的是**零碎呼叫的次數**。

光是 action inference 這一段：

```text
N=1：10,000 次 forward，累積約 16.8 秒
N=4： 2,500 次 forward，累積約  5.6 秒
N=8： 1,250 次 forward，累積約  3.2 秒
```

[![batched inference 的 throughput 與單次 forward 成本](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/batched-inference.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/batched-inference.png)

Replay insertion 也有同樣效果。獨立量 `add_batch` 時：

| 一次寫入幾筆 | transitions/s |
|---:|---:|
| 1 | 4,168 |
| 2 | 6,275 |
| 4 | 12,930 |
| 8 | 22,240 |
| 16 | 34,123 |

[![GPU Replay 批次寫入的實測結果](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/replay-insertion.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6244f3858c1ac1d0c383c1e424f74cd4dcc831c2/assets/day16/replay-insertion.png)

這個 microbenchmark 可以證明「一次搬多筆」比「一筆一筆搬」有效率，但它仍然不是完整 trainer 的速度。

Day 14 已經吃過一次這個虧：局部 benchmark 很漂亮，不代表整個訓練流程就會同比例變快。

這次完整 profiling 反而把差別說得更清楚。從 N=1 到 N=8：

```text
action inference：約 16.8 s → 3.2 s
Replay insertion：約 7.5 s → 2.7 s
ALE env step：     約 7.1 s → 8.1 s
DQN update：       約 10.5 s → 13.3 s
```

也就是說，vectorization 並沒有讓 ALE 本身變成 GPU 程式，也沒有讓每一次 optimizer update 神奇地更快。

它真正做的是：**把原本每個 transition 都要付一次的模型呼叫與資料搬移成本攤薄。**

## 為什麼 N=8 的 Replay 寫入次數沒有降到 1,250？

N=8 一次會產生 8 筆 transition；照理說 10K transitions 只需要 1,250 次 `add_batch`。

但目前 `train_frequency = 4`。

一次 8 筆資料會同時跨過：

```text
第 4 筆 → optimizer update
第 8 筆 → optimizer update
```

如果整批 8 筆先全部塞進 Replay，第一次 update 就會提前看到本來應該屬於第 5～8 筆的資料，訓練資料順序就被改了。

所以寫入會按照 transition boundary 分成：

```text
4 筆 → insert → update
4 筆 → insert → update
```

結果 N=8 的 Replay insertion 仍然是 2,500 次，和 N=4 一樣。

這也解釋了為什麼 N=4 已經非常接近 N=8：N=8 還能繼續減少 action-inference calls，但 Replay insertion 已經受到 update boundary 限制，ALE 的 CPU 工作也沒有因為多開環境而消失。

這比「N=8 比 N=4 快 2.7%」本身更有意思，因為它讓我們看到下一個效能上限在哪裡。

## Day 16 真正得到的答案

Day 16 沒有讓 DQN 變成新的演算法，也沒有讓 GPU 使用率衝到 100%。

它做的是把原本這條很碎的流程：

```text
1 個 observation
→ 1 次 inference
→ 1 筆 Replay 寫入
→ 再等 CPU
```

改成：

```text
多個 observations
→ 1 次 batched inference
→ 一批 transitions
→ batched Replay insertion
```

在這台 RTX 4060 Laptop GPU 上，10K 短跑的完整 trainer 從約 215 transitions/s 提高到約 310～319 transitions/s。N=4 已經取得大部分收益，N=8 只再增加一點點。

更重要的是，我們現在知道這個 1.48× 是從哪裡來的：不是「CUDA 自動加速」，而是**減少大量 batch=1 的模型呼叫與零碎 GPU 寫入**。

10K 在這一天主要回答的是 systems 問題，不拿來判斷哪個 checkpoint 學得比較好。真正的模型品質比較，仍然要在相同環境規則與更長訓練預算下進行。

下一步，焦點會從「怎麼更有效率地收集與處理 transition」轉回演算法本身：Vanilla DQN 的 `max` 為什麼可能偏向被高估的 Q-value？Double DQN 又是怎麼把 action selection 和 value evaluation 拆開的？
