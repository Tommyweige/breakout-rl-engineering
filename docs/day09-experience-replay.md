# Day 9｜Experience Replay：把遊戲經驗存起來，再隨機拿回來學

Day 8 已經讓 DQN 能把一個 Breakout 畫面狀態轉成四個 action 的 Q-values。

但「模型會算 Q-value」和「模型真的有辦法學」是兩回事。下一個問題是：**Agent 和遊戲互動時產生的資料，要怎麼保存，又要怎麼拿來訓練？**

每和 environment 互動一步，就會得到一筆 **transition**。Transition 可以先理解成「一次完整互動的紀錄」：

```text
(state, action, reward, next_state, terminated, truncated)
```

它記錄了 action 發生前看到的 state、Agent 做了哪個 action、得到多少 reward、來到哪個 next state，以及這一局是否真正結束或被外部條件提前截斷。

如果每拿到最新一筆 transition，就立刻只用這一筆更新 network，會遇到兩個問題：相鄰的 Atari 畫面非常相似，而且稍早發生過的經驗很快就失去再次被使用的機會。

DQN 因此加入 **Experience Replay（經驗回放）**：先把過去發生過的互動存起來，訓練時再從這些經驗中隨機抽一批回來學。

先把整體位置放對：

[![Experience Replay 從 transition、Replay Buffer、隨機抽樣到模型輸入的資料流](https://github.com/Tommyweige/breakout-rl-engineering/blob/298fcd98efba982799098c276fa296e385140269/assets/day09/replay-data-flow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/298fcd98efba982799098c276fa296e385140269/assets/day09/replay-data-flow.png)

這張圖只需要先記住一件事：**Agent 產生 transition，Replay Buffer 負責保存；真正訓練時，再從裡面抽出一批資料交給模型。**

這個用來保存過去 transition 的記憶體，就是 **Replay Buffer**。

## 連續畫面太相似，不適合一路照順序學

假設球正在往右移，連續幾個 state 可能只有一點點差別。

在一段很短的觀察裡，球的位置可能依序是 `x = 50`、`x = 52`、`x = 54`、`x = 56`。這些 state 不是四個完全不同的例子，而是同一段遊戲過程切成的四小段。

如果 network 每次都只拿最新一筆來更新，連續幾次訓練看到的東西會非常接近。模型才剛看過「球在右邊」，下一次又看到幾乎一樣的畫面。

Replay Buffer 把兩件事分開：

- Agent 現在產生什麼資料；
- 這次模型要拿哪些資料來學。

新的 transition 先保存起來；真正要更新模型時，再從目前累積的經驗裡抽一批。這樣資料收集的時間順序，就不會直接決定下一次更新一定看到哪些畫面。

這不代表抽出來的資料就完全彼此獨立，但至少不會每一次都只看時間上緊鄰的幾張畫面。

## Replay Buffer 裡保存的資料

一筆 transition 有六個部分，Replay Buffer 會把它們分開保存：

```text
states
actions
rewards
next_states
terminated
truncated
```

其中 `state` 和 `next_state` 都是 Day 4 建立好的 Breakout observation。這裡的 observation 是「Agent 真正拿來做決策的遊戲狀態」。它把最近四張灰階畫面疊在一起，所以一筆 state 的形狀是：

```text
(4, 84, 84)
```

三個數字分別代表：

- `4`：最近四張畫面；
- `84`：每張畫面的高度；
- `84`：每張畫面的寬度。

在 NumPy 裡，資料除了形狀之外還有 **資料型別（dtype）**。dtype 決定每個數字用什麼格式保存，也會直接影響記憶體用量。

這次 Replay Buffer 的主要保存格式是：

| 欄位 | 保存格式 | 作用 |
| --- | --- | --- |
| state / next_state | `uint8` | 保存 `0..255` 的灰階像素 |
| action | `int64` | 保存離散 action 編號 |
| reward | `float32` | 保存環境回傳的即時回饋 |
| terminated / truncated | `bool` | 分別保存兩種結束狀態 |

`uint8` 是 8-bit unsigned integer，可以表示 `0 ~ 255` 的整數。Atari 灰階 pixel 原本就在這個範圍，因此很適合直接保存成 `uint8`。

實際用 Breakout environment 收集 40 筆 transition，放進最多可保存 128 筆資料的 Replay Buffer，再隨機抽 32 筆，可以看到：

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

`allocated memory` 是 Replay Buffer 預先保留的 RAM。`MiB` 是記憶體單位，1 MiB 等於 1,048,576 bytes。

## Capacity 和 Batch Size 是兩件不同的事

**Capacity** 是 Replay Buffer 最多能保存多少筆 transition。

例如：

```text
capacity = 100,000
```

代表最多保留最近 100,000 筆經驗。

**Batch size** 則是一次訓練從 buffer 裡抽多少筆。

例如：

```text
batch size = 32
```

代表這次只拿 32 筆資料交給 network 訓練。

因此完全可能出現這種情況：Replay Buffer 裡有 100,000 筆歷史資料，但某一次訓練只抽 32 筆。

可以把兩者記成：

- capacity 決定「最多記住多少經驗」；
- batch size 決定「一次拿多少經驗來學」。

## Ring Buffer：滿了之後覆蓋最舊資料

Replay Buffer 不可能一直無限制成長，不然遊戲跑得越久，RAM 就會一直增加。

這次使用 **Ring Buffer（環形緩衝區）**。它可以想成一排固定數量的格子。

假設只有五格：

```text
slot 0  slot 1  slot 2  slot 3  slot 4
```

前五筆資料依序放滿後，第六筆不會新增第六格，而是繞回最前面，把目前最舊的資料覆蓋掉。

所以 Ring Buffer 的重點不是「一直增加空間」，而是：**容量固定，新的經驗持續進來，太舊的經驗逐漸被替換。**

Day 9 的視覺化使用 seed 42 的 Breakout environment 收集 8 筆真實 transition，再寫進 capacity 5 的 Replay Buffer：

[![Replay Buffer wraparound、真實 Breakout 抽樣畫面與記憶體估算](https://github.com/Tommyweige/breakout-rl-engineering/blob/9c58589e056f4fabe6438ee6c5f17a06b37fd41d/assets/day09/replay-buffer.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/9c58589e056f4fabe6438ee6c5f17a06b37fd41d/assets/day09/replay-buffer.png)

寫入 8 次後，buffer 還是只有 5 筆：

```text
capacity    = 5
writes      = 8
size        = 5
write_index = 3
```

`write_index` 可以理解成「下一筆資料準備寫到哪一格」。現在是 3，所以第 9 筆資料會覆蓋 slot 3。

此時真正由最舊排到最新的格子順序是：

```text
[3, 4, 0, 1, 2]
```

因為 slot 0、1、2 已經被後來的新資料覆蓋過，所以陣列索引 `0,1,2,3,4` 不再等於資料的時間順序。

圖下方的三張小圖則是 Replay Buffer 實際隨機抽出的 Breakout observations。這次 seed 42 抽到的 slot 是：

```text
[4, 0, 3]
```

這些數字是資料存在 buffer 裡的位置，不是 action，也不是 reward。

這張圖提供的是實際執行證據：**固定容量覆蓋與隨機抽樣真的有發生，而且抽出的資料確實來自 Breakout。**

## Uniform Sampling：每筆有效經驗都有相同抽樣機會

Day 9 使用最基本的 **Uniform Random Sampling（均勻隨機抽樣）**。

「Uniform」在這裡的意思是：只要一筆 transition 還留在 Replay Buffer 裡，它和其他有效資料一樣，都有相同機會被抽到。

例如 buffer 裡有 10,000 筆資料，一次抽 32 筆，這一批會從目前 10,000 筆保存資料中選出 32 筆，而且同一批不重複抽同一格。

下一次訓練時，同一筆舊資料仍然可能再次被抽中。

所以 Experience **Replay** 的「Replay」很重要：過去發生過的經驗不是用一次就丟掉，而是可以被重新利用。

目前不會因為某一筆 reward 比較大，就提高它被抽中的機率。那種讓部分經驗有較高抽樣機率的方法叫 Prioritized Experience Replay，Day 9 先不做。

## 舊 Transition 仍然可以重複使用

Replay Buffer 裡可能有幾千 steps 以前的資料，而 Agent 現在的行為已經和當時不同。

這些舊資料仍然有價值，是因為 transition 記錄的是環境真的發生過的一次互動：

```text
在 state s
做 action a
得到 reward r
來到 next_state s'
```

只要這筆互動當時真的發生過，它不會因為 Agent 後來變聰明就變成假的。

Q-learning 還有一個重要特性叫 **off-policy**。白話來說，它允許我們用「以前用不同方式收集到的經驗」，去學現在想要的 Q-value。

因此，早期 Agent 還很常亂試 action 時留下的經驗，之後仍然可以拿回來訓練。

但也不是越舊越好。Replay Buffer 如果大到保存非常久以前的資料，裡面的經驗可能和現在 Agent 常遇到的狀態差很多。因此 capacity 也在控制一個取捨：**要記得更多歷史，還是讓資料保持比較新。**

## 畫面用 `uint8` 保存更省記憶體

一筆 Breakout state 有：

```text
4 × 84 × 84 = 28,224 pixels
```

每個 pixel 都是一個灰階亮度值。

如果用 `uint8` 保存，一個 pixel 只需要 1 byte；如果改成神經網路計算常用的 `float32`，一個數字需要 4 bytes。

也就是說，同樣的畫面資料，如果長期全部用 `float32` 保存，大約需要四倍的空間。

所以這個專案把「保存」和「計算」分開：

- 長期放在 Replay Buffer 裡時，保留 `uint8 / 0..255`；
- 真正抽出來送進模型時，才轉成 `float32 / 0..1`。

把 `0..255` 轉成 `0..1` 的做法，就是除以 255：

| 原始 `uint8` pixel | 模型輸入 `float32` |
| ---: | ---: |
| 0 | 0.0 |
| 128 | 約 0.502 |
| 255 | 1.0 |

這樣既保留 Replay Buffer 的記憶體效率，又能在真正計算時提供神經網路需要的格式。

## Replay Buffer 其實非常吃 RAM

即使用 `uint8`，Atari Replay Buffer 還是很吃記憶體。

目前最簡單的設計會完整保存 `state` 和 `next_state`，而每一個 state 本身又包含最近四張畫面。

依照目前實作實際使用的資料格式估算：

| Capacity | 約需要的 Replay Buffer RAM |
| ---: | ---: |
| 10,000 | 0.526 GiB |
| 100,000 | 5.258 GiB |
| 1,000,000 | 52.584 GiB |

`GiB` 也是記憶體單位，1 GiB 等於 1024 MiB。

這些數字不是憑感覺寫出來的，而是按照目前 Replay Buffer 真正保存的欄位、資料型別和 observation 大小計算。

其中最大的浪費來自畫面重複。

Day 4 的一個 state 是最近四張畫面：

```text
state t
[frame 1, frame 2, frame 3, frame 4]

state t+1
[frame 2, frame 3, frame 4, frame 5]
```

可以看到，前後兩個 state 有三張畫面重複。

目前的簡單版 Replay Buffer 仍然把兩份完整 state 都存下來。好處是結構直接、容易驗證；代價就是 RAM 使用量比較高。

更省記憶體的做法，是每張 frame 只保存一次，需要某筆 state 時再把相鄰的幾張 frame 組回 `(4,84,84)`。這樣可以省掉大量重複畫面，但也要額外處理「episode 已經結束，不能把上一局和下一局的 frame 拼在一起」之類的邊界情況。

所以 Day 9 先選擇簡單、正確、容易驗證的版本。等之後真的量到 Replay Buffer 成為 RAM 瓶頸，再改成更省空間的儲存方式，才比較容易判斷優化到底帶來多少效果。

## Sample 之後才轉成 PyTorch Tensor

Replay Buffer 內保存的是 NumPy array。NumPy 是 Python 常用的數值陣列工具，適合保存與整理資料。

DQN 則使用 PyTorch，所以真正送進 neural network 前，還要把抽出的資料轉成 PyTorch 的 **Tensor**。Tensor 可以先理解成 PyTorch 用來進行神經網路計算的多維陣列。

轉換後會變成：

| Replay Buffer 欄位 | 模型邊界格式 |
| --- | --- |
| states / next_states | `float32 Tensor`，並除以 255 |
| actions | 整數 Tensor |
| rewards | `float32 Tensor` |
| terminated / truncated | `True / False Tensor` |

這樣 Replay Buffer 只負責「資料怎麼保存、怎麼抽」，PyTorch 需要的格式則在真正要訓練時才準備。

把這兩個責任分開，後面的 training loop 會比較容易理解，也比較不容易在資料格式上出錯。

## `terminated` 和 `truncated` 不是同一種結束

這兩個欄位看起來都像「這一局結束了」，但原因不同。

`terminated` 表示 environment 本身真的進入終止狀態。

`truncated` 則表示遊戲不是因為真正的終止條件結束，而是被外部限制提前截斷，例如時間步數達到上限。

因此在和 environment 互動時，兩者通常都代表接下來要 reset；但 Replay Buffer 不能現在就把它們合併成單一 `done`。

原因是後面計算 DQN 的學習目標時，需要知道「這個 state 真的是終點」，還是「只是因為外部限制而停止」。

Day 9 先完整保存兩個資訊，Day 11、Day 12 再真正用到它們。

## Replay Buffer 是持續更新的經驗池

Replay Buffer 看起來有點像一般機器學習的 dataset：裡面有很多資料，也會一次抽一批來訓練。

但兩者最重要的差別是：一般 dataset 通常在訓練開始前就已經準備完成；Replay Buffer 則會隨著 Agent 繼續玩遊戲，不斷改變內容。

這個生命週期可以拆成一個很簡單的流程：

[![Replay Buffer 持續加入新 transition，未滿時增加 size，已滿時覆蓋最舊資料](https://github.com/Tommyweige/breakout-rl-engineering/blob/ca3c2e4f06e30eb2157b8399294b2f32fda03ea6/assets/day09/replay-lifecycle.svg?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/ca3c2e4f06e30eb2157b8399294b2f32fda03ea6/assets/day09/replay-lifecycle.svg)

Buffer 還沒滿時，新 transition 會讓目前保存的資料數量增加；一旦達到 capacity，空間就不再增加，後來的新 transition 會逐步把最舊的資料換掉。

更重要的是，Agent 本身也在學習。訓練後期的 Agent 和一開始亂玩的 Agent，產生的遊戲經驗可能很不一樣。因此 Replay Buffer 並不是一份永遠不變的資料，而是一個**會跟著訓練持續更新的經驗池**。

這也解釋了為什麼 capacity 不只是「記憶體設定」：它同時決定了 Agent 能保留多長一段歷史。

## Day 9 把 DQN 的資料層接起來了

走到這裡，DQN 的資料路徑已經接起來：Agent 和 Breakout 互動後產生 transition，Replay Buffer 保存過去經驗；訓練時再隨機抽出一批，轉成 PyTorch Tensor，交給後續的 DQN training loop。

Experience Replay 解決的核心問題是：**過去經驗怎麼保存、怎麼重複利用，以及怎麼避免模型每次只盯著最新幾張高度相似的畫面。**

但還有另一個問題沒有處理：Agent 在收集新資料時，到底應該一直選目前 Q-value 最高的 action，還是故意去試一些自己還不確定的 action？

這就是 Day 10 的 **Exploration vs. Exploitation**：探索新行為，和利用目前已經知道的最好行為之間，要怎麼取得平衡。
