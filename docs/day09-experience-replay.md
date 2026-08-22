# Day 9｜Experience Replay：把連續遊戲經驗變成可重複抽樣的訓練資料

Day 8 已經讓 DQN 能把一個 Breakout state 轉成四個 action 的 Q-values。但 network 會算 Q-values，不代表它已經有適合的資料可以學。

Agent 和 environment 互動時，每一步都會產生一筆 transition：

```text
(state, action, reward, next_state, terminated, truncated)
```

如果最新一筆 transition 一出現，就立刻只拿這一筆更新 network，訓練資料會一直沿著遊戲時間順序進來。Atari 的相鄰畫面通常只差一點點：球移動幾個 pixel、paddle 稍微換位置，其他內容幾乎沒有變。

DQN 因此多了一層 **Experience Replay**：先把 interaction data 保存起來，training 時再從過去經驗中隨機抽出一個 mini-batch。

```text
Environment interaction
        ↓
transition
        ↓
Replay Buffer
        ↓
uniform random sample
        ↓
mini-batch
        ↓
DQN training
```

Day 9 要完成的不是 training loop，而是把這個資料層建立起來。

## 連續 Interaction 不適合直接當 Training Batch

假設球正往右移，連續幾個 state 可能只差一小段距離：

```text
state t     → 球在 x = 50
state t + 1 → 球在 x = 52
state t + 2 → 球在 x = 54
state t + 3 → 球在 x = 56
```

這幾筆資料不是四個互相獨立的例子，而是同一段遊戲過程的連續切片。

如果 network 每次都只看最新資料，連續幾次更新看到的內容會非常接近；稍早發生的 transition 也很快失去再次被利用的機會。

Replay Buffer 把兩件事情拆開：

```text
現在發生什麼          這次 training 學什麼
      ↓                        ↓
收集 transition      從過去資料隨機 sample
      └──────── Replay Buffer ────────┘
```

這不會讓 RL 資料突然變成完全獨立，但能降低一個 mini-batch 被相鄰 frames 主導的程度。

## Replay Buffer：保存與抽樣分開處理

這次實作的 buffer 會保存六個欄位：

```text
states       : (capacity, 4, 84, 84) uint8
actions      : (capacity,)           int64
rewards      : (capacity,)           float32
next_states  : (capacity, 4, 84, 84) uint8
terminated   : (capacity,)           bool
truncated    : (capacity,)           bool
```

實際用 Breakout environment 收集 40 筆 transition，再從 capacity 128 的 buffer sample 32 筆，可以看到：

```text
ReplayBuffer
  capacity          : 128
  current size      : 40
  observation shape : (4, 84, 84)
  observation dtype : uint8
  allocated memory  : 6.892 MiB

Sampled batch
  states      : (32, 4, 84, 84) uint8
  actions     : (32,)           int64
  rewards     : (32,)           float32
  next_states : (32, 4, 84, 84) uint8
  terminated  : (32,)           bool
  truncated   : (32,)           bool
```

這裡最容易混淆的是 **capacity** 和 **batch size**。

`capacity` 決定最多保存多少筆歷史 transition；`batch size` 則決定一次 training 要抽多少筆。

例如：

```text
Replay capacity : 100,000 transitions
Training batch  : 32 transitions
```

這兩個數字負責的是完全不同的事情。

## Ring Buffer：容量滿了就覆蓋最舊經驗

Replay Buffer 不可能無限制長大，所以這次使用固定容量的 **ring buffer**。

假設 capacity 是 5，前五筆 transition 會先填滿五個 slot。第六筆進來時，不會再新增第六格，而是回到 slot 0 覆蓋最舊資料。

Day 9 的圖直接用 seed 42 的 Breakout environment 收集 8 筆真實 transition，寫進 capacity 5 的 ReplayBuffer，再從同一個 buffer 做 sampling 與記憶體估算。

[![Day 9 Replay Buffer wraparound, sampled Breakout observations, and memory estimates](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9c58589e056f4fabe6438ee6c5f17a06b37fd41d/assets/day09/replay-buffer.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9c58589e056f4fabe6438ee6c5f17a06b37fd41d/assets/day09/replay-buffer.png)

寫入 8 次之後，buffer 仍然只有 5 筆資料：

```text
capacity    = 5
writes      = 8
size        = 5
write_index = 3

oldest → newest slots
[3, 4, 0, 1, 2]
```

第 6、7、8 筆資料已經分別回頭覆蓋 slot 0、1、2，所以「最舊到最新」的順序不再等於陣列索引的 `0 → 1 → 2 → 3 → 4`。

圖下方顯示的三張小圖，則是同一個 buffer 實際 sample 出來的 Breakout observations；seed 42 下抽到的 physical slots 是：

```text
[4, 0, 3]
```

這張圖證明的是 **wraparound 與 random sampling 確實發生在真實 ReplayBuffer 上**。它不代表這三筆資料比較重要，也不代表 Agent 已經學會玩 Breakout。

產圖資料同時保存在：

```text
/assets/day09/replay-buffer.json
```

可用下面的命令重新生成：

```powershell
conda run --name breakout-rl-engineering python visualize_replay_buffer.py --seed 42
```

## Uniform Sampling：降低時間順序的影響

baseline Experience Replay 採用 uniform random sampling。

意思是：只要一筆 transition 還留在 buffer 裡，它就和其他有效 transition 一樣，有機會被抽進下一個 mini-batch。

例如 buffer 目前保存 10,000 筆資料，每次抽 32 筆：

```text
10,000 stored transitions
          ↓
uniform random sampling
          ↓
32-transition mini-batch
```

同一個 mini-batch 裡不重複抽相同 slot，但一筆 transition 在不同 training steps 之間仍然可能再次出現。

這就是「Replay」的意思：一段 experience 不需要只使用一次。

Uniform replay 也有明確限制。它不會判斷某一筆資料是不是特別重要，也不會保證抽出的 samples 完全沒有相關性；它只是先建立一個簡單、可重現的 baseline。Prioritized Experience Replay 不屬於 Day 9 的範圍。

## 舊 Transition 仍然可以拿來學

Replay Buffer 裡的 transition 可能是幾千甚至幾萬 steps 以前收集的。那為什麼 policy 已經改變了，舊資料還能拿來更新 Q-network？

關鍵在於 transition 記錄的是 environment 真正發生過的一次互動：

```text
在 state s
做了 action a
得到 reward r
並到達 next_state s'
```

這段 environment experience 不會因為目前 policy 改變就失效。

Q-learning 本身也是 **off-policy** 方法：用來學習 Q-value 的資料，不要求一定要由「現在這一刻的 greedy policy」產生。因此早期探索時收集到的 transition，之後仍然可以被 replay。

不過這不代表越舊的資料永遠越好。policy 持續改變後，buffer 中的資料分布也可能和目前 Agent 的行為越來越不同；capacity 因此同時控制了「記得多少歷史」以及「資料有多舊」。

## `uint8` 留在 Storage，`float32` 留到 Model Boundary

一個 Breakout state 有：

```text
4 × 84 × 84 = 28,224 pixels
```

環境原本就是 `uint8 / 0..255`。如果 Replay Buffer 一開始就把所有 observation 轉成 `float32`，pixel storage 大約會放大四倍。

所以資料 contract 保持成：

```text
Replay storage
uint8 NumPy / 0..255
        ↓ sample
TransitionBatch
        ↓ model boundary
float32 torch / 255
        ↓
DQN
```

這次 baseline 同時保存完整的 `state` 和 `next_state`，因此記憶體成本其實不低。實際 estimator 得到：

| Capacity | Baseline replay memory |
| ---: | ---: |
| 10,000 | 0.526 GiB |
| 100,000 | 5.258 GiB |
| 1,000,000 | 52.584 GiB |

這些數字來自 `estimate_replay_memory_bytes()`，和實際 NumPy arrays 使用相同 dtype 與 observation shape。

目前設計的優點是簡單、清楚、容易測試；缺點是相鄰 stacked observations 之間其實共享很多 frame，而 `state + next_state` 會重複保存其中不少 pixel。

更省 RAM 的 Atari replay 可以只保存 frame-level data，sample 時再重建 frame stack。不過 Day 9 先把資料 semantics 做正確，之後真的由 profiling 證明 replay memory 是瓶頸，再做 compact replay 會更容易比較優化前後的差異。

## Sample 後才轉成 Model Tensor

從 Replay Buffer 抽出的資料仍然是 NumPy arrays。真正進 DQN 前才統一轉換：

```text
states       → torch.float32 / 255
next_states  → torch.float32 / 255
actions      → torch.long
rewards      → torch.float32
terminated   → torch.bool
truncated    → torch.bool
```

這個邊界讓 Replay Buffer 專心處理「怎麼保存、怎麼抽樣」，而不是同時負責 CNN preprocessing 或 GPU inference。

Day 12 寫 training loop 時，也不需要把 dtype conversion 散落在每個 training step 裡。

## `terminated` 與 `truncated` 必須保留

兩個 flag 都可能讓 environment 在下一輪 reset，但原因不同：

```text
terminated → environment 真正進入終止狀態
truncated  → 因時間限制等外部條件被截斷
```

對 interaction loop 來說，`terminated or truncated` 都可能代表需要 reset；但對之後的 Bellman target 來說，兩者的語意不應該在資料保存階段就被壓成一個 `done`。

因此 Replay Buffer 從一開始就分開保存這兩個欄位，讓 Day 11、Day 12 還能根據真正的 episode semantics 決定 bootstrap mask。

## Replay Buffer 不是固定 Dataset

它看起來很像 supervised learning 的 dataset：裡面有 samples，也會抽 mini-batch。

差別在於 Replay Buffer 的內容一直在變。

```text
Agent 與 environment 互動
        ↓
新 transition 持續加入
        ↓
buffer 滿了
        ↓
最舊 transition 被覆蓋
```

而且 Agent 的 policy 也會隨訓練改變，所以之後收集到的資料分布，可能和訓練初期完全不同。

Experience Replay 比較像一個會持續更新的 **experience pool**，而不是訓練開始前就固定好的 dataset。

## Day 9 完成的是 DQN 的資料層

到這裡，資料路徑已經接起來：

```text
environment interaction
        ↓
transition
        ↓
fixed-capacity uint8 Replay Buffer
        ↓
uniform random TransitionBatch
        ↓
model-boundary tensor conversion
```

Replay Buffer 解決了「過去經驗如何保存、重用與抽樣」，但還沒有決定 Agent **怎麼收集新的 experience**。

如果每次都選目前 Q-value 最大的 action，Agent 很可能只重複自己已經知道的行為；如果完全亂選，又無法利用目前已經學到的資訊。

這就是 Day 10 要處理的 **Exploration vs. Exploitation**。
