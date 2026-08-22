# Day 9｜Experience Replay：把遊戲經驗存起來，再隨機拿回來學

Day 8 已經讓 DQN 能把一個 Breakout 畫面狀態轉成四個 action 的 Q-values。

但「模型會算 Q-value」和「模型真的有辦法學」是兩回事。下一個問題是：**Agent 和遊戲互動時產生的資料，要怎麼保存，又要怎麼拿來訓練？**

每和 environment 互動一步，就會得到一筆 transition，也就是一次完整的「狀態變化紀錄」：

```text
(state, action, reward, next_state, terminated, truncated)
```

可以先把它理解成：

先記下 action 發生前看到的 state，再記下 Agent 做了哪個 action；環境執行後回傳 reward、next_state，以及這一局是否真的結束或被外部條件中止。這六個欄位必須屬於同一次互動，不能把 reset 後的新 state 接到上一局的 transition 裡。

如果每得到最新一筆 transition，就立刻只拿這一筆更新 network，會有兩個問題：相鄰的 Atari 畫面太相似，而且稍早的經驗很快就沒有再次被使用的機會。

DQN 因此加入 **Experience Replay**。它的概念很直接：

> 先把過去發生過的互動存起來，訓練時再從這些經驗中隨機抽一批回來學。

下面這張 structural diagram 把這個資料流和 capacity 分支放在一起。節點對應本專案真正的 `ReplayBuffer.add()`、`sample()` 與 `replay_batch_to_tensors()`；它是依 implementation 整理出的結構圖，不是偽造的 runtime trace。

[![Day 9 replay data flow from Breakout interaction through ring-buffer sampling and tensor conversion](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/298fcd98efba982799098c276fa296e385140269/assets/day09/replay-data-flow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/298fcd98efba982799098c276fa296e385140269/assets/day09/replay-data-flow.png)

圖中最值得注意的是中間的分支：buffer 尚未滿時增加 `size`；已滿時則覆蓋 `write_index`。兩條路最後都會回到 `sample()`，再把抽出的 NumPy batch 交給模型邊界轉換。

可編輯的 Mermaid source 保存在 `/assets/day09/replay-data-flow.mmd`，這次使用 repository root 執行以下命令產生 PNG：

```powershell
python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day09/replay-data-flow.mmd assets/day09/replay-data-flow.png --theme neutral --background-color white --width 1200 --scale 2
```

這個用來保存過去 transition 的記憶體，就是 **Replay Buffer**。

## 連續畫面太相似，不適合一路照順序學

假設球正在往右移，連續幾個 state 可能只有一點點差別：

在一段很短的觀察裡，球的位置可能依序是 `x = 50`、`x = 52`、`x = 54`、`x = 56`。這些 state 不是四個完全不同的例子，而是同一段遊戲過程切成的四小段。

如果 network 每次都只拿最新一筆來更新，連續幾次訓練看到的東西會非常接近。模型可能才剛看過「球在右邊」，下一次又看到幾乎一樣的畫面。

Replay Buffer 把「現在產生什麼資料」和「這次訓練要拿什麼資料」分開：Agent 產生的新 transition 先交給 `add()` 保存，真正要更新模型時，呼叫 `sample()` 從目前仍在 buffer 裡的經驗取一批。這個分離讓資料收集的時間順序，不再直接決定下一次更新看到的資料。

這不代表抽出來的資料就完全彼此獨立，但至少不會每次都只看到時間上緊鄰的畫面。

## Replay Buffer 裡到底存什麼？

這次實作會把 transition 的六個部分分開保存：

```text
states
 actions
 rewards
 next_states
 terminated
 truncated
```

其中 `state` 和 `next_state` 都是 Day 4 建立好的 Breakout observation：四張最近的灰階畫面疊在一起，所以一筆 state 的形狀是：

```text
(4, 84, 84)
```

這裡的三個維度分別代表四張最近畫面、每張畫面的高度，以及每張畫面的寬度。把維度含義先釐清，後面才不會把 frame 數量誤認成 batch size。

在 NumPy 裡，資料除了「形狀」以外，還有一個 **資料型別（dtype）**。dtype 決定每個數字用什麼格式、占多少記憶體。

這次 Replay Buffer 的主要資料型別是：

| 欄位 | 保存格式 | 作用 |
| --- | --- | --- |
| state / next_state | `uint8` | 保存 `0..255` 的 stacked pixels |
| action | `int64` | 保存離散 action 編號 |
| reward | `float32` | 保存環境回傳的即時回饋 |
| 結束狀態 | `bool` | 分開保存 `terminated` 與 `truncated` |

`uint8` 是 8-bit unsigned integer，也就是只能表示 `0 ~ 255` 的整數。這正好符合 Atari 灰階 pixel 原本的範圍，所以很適合拿來保存畫面。

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

這裡的 `allocated memory` 是 Replay Buffer 預先保留的 RAM。`MiB` 是記憶體常用的單位，1 MiB 等於 1,048,576 bytes。

## Capacity 和 Batch Size 是兩件不同的事

這兩個數字很容易混在一起。

**Capacity** 是 Replay Buffer 的總容量，也就是最多能記住多少筆 transition。

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

所以完全可能出現：

Replay Buffer 裡可以有 100,000 筆歷史資料，但這一次的 mini-batch 仍然只抽 32 筆。capacity 決定「能記多久」，batch size 決定「一次學多少」。

## Ring Buffer：滿了之後直接覆蓋最舊資料

Replay Buffer 不可能一直無限制成長，不然跑得越久，RAM 就會一直增加。

這次採用 **Ring Buffer（環形緩衝區）**。

它可以想成一排固定數量的格子。假設只有五格：

```text
slot 0  slot 1  slot 2  slot 3  slot 4
```

前五筆資料依序放滿後，第六筆不會新增第六格，而是回到最前面，把最舊的資料覆蓋掉：

第 1 到第 5 筆會依序填入 slot 0 到 slot 4；第 6 筆則回到 slot 0，覆蓋最舊資料。

這就是「環形」的意思：寫到最後一格後，再繞回第一格繼續寫。

Day 9 的視覺化直接使用 seed 42 的 Breakout environment 收集 8 筆真實 transition，再寫進 capacity 5 的 Replay Buffer。

[![Day 9 Replay Buffer wraparound, sampled Breakout observations, and memory estimates](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9c58589e056f4fabe6438ee6c5f17a06b37fd41d/assets/day09/replay-buffer.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9c58589e056f4fabe6438ee6c5f17a06b37fd41d/assets/day09/replay-buffer.png)

寫入 8 次後，buffer 還是只有 5 筆：

```text
capacity    = 5
writes      = 8
size        = 5
write_index = 3
```

`write_index` 的意思是「下一筆資料要寫到哪一格」。現在是 3，所以第 9 筆資料會覆蓋 slot 3。

此時真正由最舊排到最新的格子順序是：

```text
[3, 4, 0, 1, 2]
```

因為 slot 0、1、2 已經被第 6、7、8 筆新資料覆蓋過，所以不能再把陣列索引 `0,1,2,3,4` 直接當成時間順序。

圖下方的三張小圖則是 Replay Buffer 實際隨機抽出的 Breakout observations。這次 seed 42 抽到：

```text
[4, 0, 3]
```

這裡的數字是資料存在 buffer 裡的格子位置，不是 action，也不是 reward。

這張圖要證明的是：**固定容量覆蓋與隨機抽樣真的有發生，而且抽到的確實是真實 Breakout 畫面。**

## Uniform Sampling：每筆有效經驗都有相同抽樣機會

Day 9 使用的是最基本的 **Uniform Random Sampling（均勻隨機抽樣）**。

「Uniform」在這裡的意思是：只要一筆 transition 還留在 Replay Buffer 裡，它就和其他資料一樣，有相同機會被抽到。

例如 buffer 裡有 10,000 筆資料，一次要抽 32 筆：

這次抽樣會從 10,000 筆保存中的 transition 選出 32 筆；同一批不重複抽同一格，但下一次訓練仍然可能再次抽到同一筆舊資料。

同一批 32 筆裡不會重複抽同一格，但下一次訓練時，同一筆舊資料仍然可能再次被抽中。

所以 Experience **Replay** 的重點就在這裡：過去發生過的經驗，不是用一次就丟掉。

目前不會因為某一筆 reward 比較大，就讓它比較容易被抽到。那種「讓重要經驗有更高抽樣機率」的做法叫 Prioritized Experience Replay，Day 9 先不做。

## 舊 Transition 為什麼還能繼續使用？

Replay Buffer 裡可能有幾千 steps 以前的資料，而 Agent 現在的行為已經和當時不同。

這些舊資料仍然有價值，是因為 transition 記錄的是環境真的發生過的一次互動：

```text
在 state s
做 action a
得到 reward r
來到 next_state s'
```

只要這筆資料當時是真實發生的，它不會因為 Agent 後來變聰明就變成假的。

Q-learning 還有一個重要特性叫 **off-policy**。

白話來說，off-policy 表示：

> 我可以用「以前用別種行動方式收集到的經驗」，來學現在想要的 Q-value。

因此，早期 Agent 還很常亂試 action 時留下的經驗，之後仍然可以拿回來訓練。

但也不是越舊越好。Replay Buffer 如果大到保存非常久以前的資料，裡面的經驗可能和現在 Agent 常去的狀態差很多。所以 capacity 其實也在控制一個取捨：**要記得更多歷史，還是讓資料保持比較新。**

## 為什麼畫面要用 `uint8` 保存？

一筆 Breakout state 有：

```text
4 × 84 × 84 = 28,224 pixels
```

每個 pixel 都是一個灰階亮度值。

如果用 `uint8` 保存，一個 pixel 只需要 1 byte；如果改成神經網路計算常用的 `float32`，一個數字需要 4 bytes。

也就是說，同樣一張畫面，用 `float32` 長期存放，大約會需要四倍空間。

所以這個專案把兩件事分開：

長期保存時保留 `uint8 / 0..255`；真正要丟進模型時，才轉成 `float32 / 0..1`。這樣 storage 不必為了尚未抽到的資料先支付較高的格式成本。

這裡的「轉成 `0..1`」就是把原本 `0..255` 的 pixel 除以 255。例如：

| 原始 `uint8` pixel | 模型輸入 `float32` |
| ---: | ---: |
| 0 | 0.0 |
| 128 | 約 0.502 |
| 255 | 1.0 |

這樣比較適合神經網路計算，但沒有必要讓 Replay Buffer 裡幾萬、幾十萬筆畫面都提前用 `float32` 保存。

## Replay Buffer 其實非常吃 RAM

目前最簡單的設計會完整保存：

```text
state
next_state
```

而一個 state 本身又包含最近四張畫面。

實際的記憶體估算結果是：

| Capacity | 約需要的 Replay Buffer RAM |
| ---: | ---: |
| 10,000 | 0.526 GiB |
| 100,000 | 5.258 GiB |
| 1,000,000 | 52.584 GiB |

`GiB` 是另一個記憶體單位，1 GiB 約等於 1024 MiB。

這些數字是 `estimate_replay_memory_bytes()` 算出來的。這個函式不是在猜數字，而是按照 Replay Buffer 實際使用的資料格式去計算：每筆 state 多大、next_state 多大、action 和 reward 各占多少空間，再乘上 capacity。

這裡還有一個浪費空間的地方。

Day 4 的一個 state 是「最近四張畫面」：

```text
state t
[frame 1, frame 2, frame 3, frame 4]
```

下一個 state 很可能是：

```text
state t+1
[frame 2, frame 3, frame 4, frame 5]
```

可以看到，兩個 state 有三張畫面其實是重複的。

目前這個簡單版 Replay Buffer 還是會把兩份完整 state 都存下來，所以比較容易理解與實作，但不算最省 RAM。

更進階的 Atari Replay Buffer 可以改成「每張 frame 只保存一次」，等抽到某筆資料時，再把前後幾張 frame 組回 `(4,84,84)` 的 state。這樣能省很多重複資料，但程式會複雜不少，還要額外處理 episode 結束時哪些 frame 不能跨局組在一起。

Day 9 先選擇簡單而正確的版本。等之後實際量測發現 Replay Buffer 真的成為 RAM 瓶頸，再換成更省記憶體的設計，會比較容易知道優化到底帶來多少效果。

## Sample 之後，才轉成 PyTorch Tensor

Replay Buffer 內保存的是 NumPy array。NumPy 是 Python 常用的數值陣列工具，適合保存與整理這些資料。

但 DQN 是用 PyTorch 寫的，所以真正送進 neural network 前，需要再把資料轉成 PyTorch 的 **Tensor**。Tensor 可以先把它理解成 PyTorch 用來進行神經網路計算的多維陣列。

轉換後會變成：

| Replay Buffer 欄位 | 模型邊界格式 |
| --- | --- |
| states / next_states | `float32 Tensor`，並除以 255 |
| actions | 整數 Tensor |
| rewards | `float32 Tensor` |
| terminated / truncated | `True / False Tensor` |

這樣的好處是 Replay Buffer 只負責保存資料，PyTorch 需要的格式則在真正要訓練時才準備。

換句話說：

Replay Buffer 關心的是資料怎麼存、怎麼抽；DQN 關心的是抽到的資料怎麼拿來計算。這兩個責任分開，後面的 training loop 會比較容易理解。

## `terminated` 和 `truncated` 不是同一種結束

這兩個欄位看起來都像「這一局結束了」，但原因不同。

`terminated` 表示 environment 本身真的進入終止狀態。

`truncated` 則表示遊戲不是因為真正的終止條件結束，而是被外部限制提前截斷，例如時間步數達到上限。

所以在和 environment 互動時：

```text
terminated or truncated
```

通常都代表下一步要 reset。

但 Replay Buffer 不能現在就把它們合併成單一 `done`，因為之後計算 DQN 的學習目標時，需要知道「是真的終止」還是「只是被截斷」。

Day 9 先把這兩個資訊完整保存，Day 11、Day 12 再使用它們計算訓練目標。

## Replay Buffer 不是一份固定 Dataset

它看起來很像一般機器學習的 dataset：裡面有很多資料，也會一次抽一批來訓練。

但差別很大。

一般 dataset 通常在訓練開始前就已經準備完成；Replay Buffer 則會一邊訓練、一邊改變內容：

Agent 繼續玩遊戲時，新 transition 不斷交給 `add()`；buffer 滿了之後，`write_index` 會指向下一個要被覆蓋的 slot。前面的 Mermaid 結構圖已經把「加入、判斷是否已滿、保留或覆蓋」這個生命週期畫出來。

而且 Agent 的行為也會隨著訓練改變，因此後期收集到的遊戲經驗，可能和一開始亂玩的資料很不一樣。

所以 Replay Buffer 比較像一個**持續更新的經驗池**，而不是一份永遠不變的訓練資料集。

## Day 9 把 DQN 的資料層接起來了

走到這裡，資料路徑已經完整很多：

Agent 和 Breakout 互動後產生 transition，Replay Buffer 保存過去經驗；之後由 `sample()` 抽出一批，再轉成 PyTorch Tensor，交給後續 DQN training loop。這條完整路徑與前面的 Mermaid 圖相同，差別只在 Day 9 先停在模型邊界，還沒有加入完整的 loss 與 optimizer。

Experience Replay 解決的是「過去經驗怎麼保存、怎麼重複使用、怎麼避免每次只看最新畫面」。

但還有一個問題沒有處理：Agent 在收集新資料時，到底應該一直選目前 Q-value 最高的 action，還是故意去試一些自己還不確定的 action？

這就是 Day 10 的 **Exploration vs. Exploitation**：探索新行為，和利用目前已經知道的最好行為之間，要怎麼取得平衡。
