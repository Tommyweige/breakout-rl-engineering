# Day 9｜為什麼 DQN 不能只用最新一筆 transition？

Day 8 的 DQN network 已經能把一個 Breakout state 轉成四個 Q-values，但「能算」和「能學」中間還差一個資料問題：每次和 environment 互動得到的 transition，要怎麼留到之後再次使用？

如果 Agent 每收到一筆新資料，就立刻只用那一筆更新模型，資料會有兩個問題。第一，Atari 的連續畫面高度相似，前後幾筆 transition 並不是彼此獨立的例子；第二，新資料一來，舊的經驗就沒有機會再被使用。這篇要建立的 mental model 是：先把互動結果放進一個固定容量的記憶體，再從目前保存的經驗中隨機抽出一個 mini-batch。

## 先看一批實際保存的資料

Day 3 已經看過一筆 transition 的時間邊界：

```text
(state, action, reward, next_state, terminated, truncated)
```

Day 4 又把 observation 固定成四張 `84 × 84` 的灰階畫面。因此 Day 9 的資料結構不是一串抽象的數字，而是許多筆形狀為 `(4, 84, 84)` 的 state，以及每筆 action、reward 和 episode status。

先用專案的預處理 Breakout environment 收集 40 筆真實 transition，再放進 capacity 為 128 的 buffer：

```powershell
conda run --name breakout-rl-engineering python inspect_replay_buffer.py --capacity 128 --batch-size 32 --steps 40 --seed 42
```

這次執行的摘要如下：

```text
ReplayBuffer
  capacity          : 128
  current size      : 40
  observation shape : (4, 84, 84)
  observation dtype : uint8
  allocated memory  : 6.892 MiB

Sampled NumPy batch
  states      : shape=(32, 4, 84, 84), dtype=uint8
  actions     : shape=(32,), dtype=int64
  rewards      : shape=(32,), dtype=float32
  next_states : shape=(32, 4, 84, 84), dtype=uint8
  terminated  : shape=(32,), dtype=bool
  truncated   : shape=(32,), dtype=bool
```

這裡有一個很容易混淆的差異：`capacity=128` 代表最多可以保存 128 筆 transition，`batch-size=32` 則只代表這一次抽出 32 筆。這次只收集了 40 筆，所以 `current size` 是 40；預配置的陣列仍然按照 capacity 配置空間。

## Replay Buffer 解決的是「保存與重用」

把互動資料保存起來的容器叫做 Experience Replay Buffer，通常簡稱 Replay Buffer。它的工作不是替模型計算 loss，也不是決定 action，而是先守住一筆 transition 的完整邊界，讓之後的訓練可以從過去經驗取樣。

在這個專案裡，sample 回來的資料被包成具名的 `TransitionBatch`：

```text
states       : (B, 4, 84, 84) uint8
actions      : (B,)           int64
rewards      : (B,)           float32
next_states  : (B, 4, 84, 84) uint8
terminated   : (B,)           bool
truncated    : (B,)           bool
```

用具名欄位而不是一個沒有語意的六元素 tuple，會讓 Day 12 的 training loop 不必靠位置猜測「第四個欄位到底是 next state 還是 reward」。更重要的是，Replay Buffer 只保存資料，不在這一層把 pixel 轉成模型 tensor；這兩個責任要分開，才能知道記憶體裡究竟存了什麼。

## capacity 滿了之後，最舊的經驗會去哪裡？

固定容量的 Replay Buffer 不能讓 Python list 無限制成長。這裡使用 ring buffer：寫入位置到達陣列尾端後，回到 slot 0，覆蓋最舊的 transition。

下面的圖不是手寫的示意數字。腳本用 seed `42` 的真實 Breakout environment 產生 8 筆 transition，讓 capacity 為 5 的 ReplayBuffer 實際寫入，再用 buffer 回報的 physical slot、sample 結果與 memory estimator 產圖。它回答的問題是：**寫入超過容量時，資料在固定陣列中如何移動，以及 capacity 如何影響配置記憶體？**

[![Day 9 Replay Buffer wraparound, real sampled observations, and memory estimates](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/main/assets/day09/replay-buffer.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/main/assets/day09/replay-buffer.png)

上方的表格橫軸是寫入順序，縱軸是陣列的 physical slot，格子裡的數字是實際第幾次寫入。第 1 到第 5 筆先填滿 slot 0 到 4；第 6 筆回到 slot 0，第 7 筆覆蓋 slot 1，第 8 筆覆蓋 slot 2。因此最後從 oldest 到 newest 讀取時，slot 順序是 `[3, 4, 0, 1, 2]`，而不是陣列索引的 `[0, 1, 2, 3, 4]`。

這個順序差異就是 ring buffer 最重要的邊界：`write_index` 指向下一個要覆蓋的位置，`size` 則最多增加到 capacity。圖下方的三張小圖是同一個 buffer 隨機抽出的三筆 state 的最新 frame；它們只是說明 sample 確實拿到 `(4, 84, 84)` 的實際 observation，不代表模型已經學會打磚塊。

圖中的 JSON metadata 保存在 `/assets/day09/replay-buffer.json`，可以用同一個腳本重新生成：

```powershell
conda run --name breakout-rl-engineering python visualize_replay_buffer.py --seed 42
```

這個小 capacity 是為了讓 wraparound 在一張圖中看得見；它不是建議的訓練設定。實際 capacity 應該根據可用 RAM、訓練步數和資料的新鮮度取捨。

## 為什麼要隨機抽樣，而不是照時間順序拿資料？

Breakout 的 state 是連續畫面堆疊。相鄰的 state 往往只差一個 frame，球和 paddle 也只移動了一小段距離。如果每次都拿最新 transition 更新，模型看到的資料會集中在同一個很短的時間片段；一個罕見但重要的事件，則可能很快被後續畫面淹沒。

Replay Buffer 讓這些 transition 暫時脫離原本的時間順序，從目前仍然有效的資料中做 uniform random sampling。uniform 的意思是每個尚未被覆蓋的 slot 有相同的抽樣機會；baseline 不會因為 reward 大、看起來驚訝，或某個 transition 比較新，就偷偷提高它的權重。

實作的核心選擇只有一行：

```python
indices = rng.choice(self.size, size=batch_size, replace=False)
```

`replace=False` 表示同一個 mini-batch 裡不重複抽同一個 slot；它不表示一筆 transition 從此只能用一次。下一個 batch 仍然可能再次抽到它，只要它還沒有被新的資料覆蓋。顯式傳入 NumPy generator，則讓固定 seed 時的抽樣順序可以重現。

因此，Replay Buffer 不是把資料打亂一次後就結束的固定資料集。它會持續收到新 transition、淘汰最舊內容，並且由 Agent 當前的行為逐步改變資料分布；抽樣只是讓同一段經驗能在有效期限內被重複利用，也降低連續 observation 之間的相關性。

## `batch size` 不等於 `capacity`

`capacity` 控制的是歷史窗口：最多保留多少筆 transition，以及 ring buffer 何時開始覆蓋舊資料。`batch size` 控制的是一次更新要拿多少筆資料。前者影響記憶體和資料新鮮度，後者影響一次計算的輸入量。

在上面的實際執行中，capacity 是 128、目前保存 40 筆、每次 sample 32 筆。當 buffer 還沒收集到 32 筆時，這個 baseline 會清楚拒絕 sample，而不是默默回傳一個較小的 batch；當 buffer 滿了之後，`len(buffer)` 仍然最多是 128。

## `uint8` 應該留在 storage，`float32` 再交給模型

一個 `(4, 84, 84)` observation 有：

```text
4 × 84 × 84 = 28,224 pixels
```

`uint8` 每個 pixel 只需要 1 byte，且正好表達環境的 `0..255` contract。若長期用 `float32` 保存同一份 pixel，單是 observation array 就會變成約四倍大小；但 CNN 真正運算時仍需要 `float32`，而且要把 pixel 除以 255 變成 `0..1`。

所以轉換時機放在 sample 之後、送進模型之前：

```text
Replay storage       → model boundary
uint8 NumPy           → float32 torch / 255
int64 actions         → torch.long
float32 rewards       → torch.float32
bool episode flags    → torch.bool
```

這次 baseline 會同時保存 `states` 和 `next_states`。以 `(4, 84, 84)`、capacity `10000` 為例，記憶體估算器實際算出 `564,620,000` bytes，也就是圖中的約 `0.526 GiB`；capacity `100000` 和 `1000000` 則分別約為 `5.258 GiB` 和 `52.584 GiB`。這些數字不是從圖上目測，也不是文章手算後填入，而是由 `estimate_replay_memory_bytes()` 使用同一組 dtype 和 shape 產生。

每筆 transition 的 observation 部分本身就是 `2 × 28,224 = 56,448` bytes；再加上 int64 action 的 8 bytes、float32 reward 的 4 bytes，以及兩個 bool flag，baseline 的配置量是每筆 `56,462` bytes。這是 NumPy arrays 的配置估算，不包含 Python 物件、allocator 或其他 runtime overhead。

這個設計用較簡單的資料 layout 換取可讀性和可測試性，但也誠實留下了浪費：連續 stacked observations 之間其實共享許多 frame，`states` 和 `next_states` 可能保存重複 pixel。若 profiling 證明 RAM 成為瓶頸，之後可以研究 frame-level compact replay；現在先不讓記憶體最佳化遮住資料邊界本身。

## 為什麼要把 `terminated` 和 `truncated` 分開？

兩個 flag 都可能讓 environment 在下一輪 reset，但原因不同。`terminated` 表示遊戲本身達到終止條件；`truncated` 表示被時間上限等外部限制截斷。收集 transition 時可以用 `terminated or truncated` 決定是否 reset，卻不代表兩者在未來計算 target 時意思相同。

因此 buffer 使用兩個獨立的 bool array，而不是立即壓成一個 `done`。Day 11 和 Day 12 會再決定 target 的 bootstrap mask 如何使用 `terminated`；Day 9 只負責把 episode semantics 原樣保存下來。

## 這個 baseline 還沒有回答哪些問題？

Experience Replay 解決的是「過去資料如何保存、抽樣與重用」，不是完整 DQN training loop。這篇沒有加入 Prioritized Replay、epsilon-greedy exploration、Target Network、TD loss 或 optimizer；它們分別回答抽樣權重、資料收集策略、目標穩定性和參數更新等後續問題。

今天真正接起來的資料流是：

```text
environment interaction
        ↓
(state, action, reward, next_state, terminated, truncated)
        ↓
固定容量 uint8 ring buffer
        ↓
uniform random TransitionBatch
        ↓
sample 後轉成 model tensors
```

這條路徑讓同一筆經驗有機會被重複使用，又不會把模型計算需要的 normalization 混進 storage。它也讓一個重要的限制變得可見：capacity 越大，能保留的歷史越長，但 RAM 成本會線性增加，而且舊資料可能和目前 policy 越來越不相似。

下一個問題因此很自然：Agent 在繼續收集 transition 時，應該一直選目前 Q-value 最高的 action，還是要刻意探索還沒試過的 action？這會帶我們進入 Day 10 的 Exploration vs. Exploitation。


