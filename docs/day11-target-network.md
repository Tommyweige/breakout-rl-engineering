# Day 11｜Target Network：讓 Bellman 目標暫時固定下來

Day 10 結束時，Deep Q-Network（DQN）已經能從 Breakout 的 state（當下觀察）產生每個 action（動作）的 Q-value；Q-value 可以先理解成「在這個狀態採取某個動作後，未來可能得到多少回報」的估計。Agent 也能用 epsilon-greedy 決定要探索還是利用。真正開始更新 network 之後，還會遇到一個更根本的問題：**拿來教模型的答案，竟然也由同一個正在被修改的模型產生。**

先看這次實際執行的最小實驗：兩個獨立的 DQN model 先同步，接著只對 online network 做一次 optimizer step（根據 loss 調整模型參數的一次更新），最後再把 online 的參數整份複製給 target。

```text
before online update: max abs diff = 0.00000000
after online update : max abs diff = 0.00746517
after target sync   : max abs diff = 0.00000000
```

這三個數字先回答了今天的核心問題：**target network 不是永遠不變，而是在兩次硬同步之間保持不變。** 這裡的 Bellman target 是「用 reward 和下一個 state 算出的學習目標」。接下來要理解的是，為什麼需要這個額外的 network，以及它在這個 target 裡究竟負責哪一部分。

## 同一個 network 同時當學生和答案，會發生什麼事？

Q-learning 每次更新都需要比較兩個數字。第一個是目前模型對實際 action 的估計 `Q(s, a)`；第二個是根據 reward（環境回傳的即時回饋）和下一個 state 算出的目標值。`gamma` 是折扣因子，用來控制未來 reward 在目前目標裡所占的比重。

在還沒有神經網路時，Q-value 可以放在表格裡。Day 8 之後，這張表改成由 DQN 近似：輸入 state，輸出所有 action 的 Q-values。因此最基本、不加入 Double DQN 分離策略的 DQN target，通常稱為 vanilla DQN target，可以寫成：

```text
target = reward + gamma × max_a Q(next_state, a)
```

這裡的 `max_a` 表示「在下一個 state 的所有 action 中取最大值」。如果目前只有一個 network，左邊的目前估計和右邊的 target 都來自同一組參數：同一個 network 一方面算目前的 `Q(s, a)`，另一方面又算 target 裡的 `max Q(next_state, a)`。

接著 optimizer 會根據兩者的差距調整參數。問題是，參數一改，下一次算出來的 target 也立刻改了。模型不是在追一個暫時固定的答案，而是在追一個每次更新都跟著自己移動的答案。這個現象叫做 **moving target（移動中的目標）**。

`torch.no_grad()` 可以阻止 target 計算建立 gradient graph（PyTorch 為之後反向傳播保存的運算關係），但它只處理「梯度要不要回傳」；它不會讓同一個 network 的參數在 optimizer step 後保持原值。所以 Day 11 要拆開的是參數更新的時間尺度，而不只是把梯度關掉。

## Online network 和 Target network 的角色分工

解法是建立兩個獨立的 model instance（各自擁有一份參數的模型物件）。負責持續學習、接受 optimizer 更新的叫 **online network**；用來暫時提供下一個 state 的 Q-value、在一段時間內保持固定的副本叫 **target network**。

它們的關係不是「一個物件取兩個名字」，而是：

```python
online = DQNNetwork(num_actions=4)
target = DQNNetwork(num_actions=4)
hard_update(target, online)

# 只有 online 進入 optimizer；target 只在同步時取得參數副本。
optimizer = torch.optim.SGD(online.parameters(), lr=0.01)
```

`target = online` 是錯的，因為那只會建立另一個 reference。兩個變數最後仍然指向同一個 Python 物件，也就共用同一份 parameter storage；online 一改，target 沒有可能保持舊值。`state_dict` 是 PyTorch 用來按名稱保存模型 parameters 與其他狀態值的鍵值集合；`hard_update()` 複製的就是這份集合，並把 target parameters 設成不需要梯度，保留兩個獨立 instance 的角色邊界。

下面的結構圖把這個分工放回一次 DQN target 計算裡。圖是依照目前 helper 的資料流整理出的結構示意，不是某一局 Breakout 的逐步 trace；它要回答的是「哪個 network 產生哪個數字，以及同步箭頭往哪裡走」。

[![Transition 批次分出 next state、reward 與 terminated，target network 計算 Bellman target，online network 產生目前 Q-value](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/d8ddf24/assets/day11/target-network-flow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/d8ddf24/assets/day11/target-network-flow.png)

實線是一次 batch 內的數值流：`next_state` 交給 target network，所有 action 的輸出再取最大值，最後和 reward、terminated mask 組成 target。下方的 online network 產生目前 action 的 Q-value，未來才會和 target 一起形成 training loss。虛線則表示週期性的 hard update：參數由 online 複製到 target，而不是反過來。

## Vanilla DQN target 只用 target network 估計下一步

現在可以把公式寫得更精確。對一個 batch 來說，`r` 是每筆 transition 的 reward，`s'` 是下一個 state，`terminated` 表示 environment 是否真的進入終止狀態，`Q_target` 是 target network 的輸出：

```text
y = r + gamma × (1 - terminated) × max_a Q_target(s', a)
```

`terminated = 1` 時，後面不應該再接續未來價值，所以 target 就是當下的 reward。其他 transition 才會 bootstrap，也就是把下一個 state 的估計價值帶回目前的目標。

程式中真正承擔這個計算的核心很短：

```python
with torch.no_grad():
    next_q_values = target_network(next_states)

next_q_max = next_q_values.max(dim=1).values
bootstrap_mask = (~terminated).to(dtype=next_q_values.dtype)
target = rewards + gamma * bootstrap_mask * next_q_max
```

這裡的 `Tensor` 是 PyTorch 用來做神經網路計算的多維陣列；`dim=1` 代表在 action 那一維取最大值，留下每筆 transition 一個 target，因此輸出 shape 是 `(B,)`。`B` 是 batch size，也就是這次同時處理的 transition 數量。

這個版本是 vanilla DQN：選最大 action 的動作和評估它的 network 都是 target network。Day 17 的 Double DQN 才會把這一段拆成「由 online network 選 action，再由 target network 評估該 action」；Day 11 不提前加入那個變化，否則會無法分辨 Target Network 本身帶來的效果。

### 為什麼 target inference 不需要 gradient？

PyTorch 的 gradient graph 會記住運算之間的關係，讓之後的 backward 能把 loss 的影響傳回參數。target 在這裡扮演的是暫時的參考答案，不是這一步要被 optimizer 直接修改的對象，因此 target inference 放在 `torch.no_grad()` 裡。

這有兩層意義：一是不要為了不會 backward 的分支保存中間計算；二是讓 target network 的參數不會因為 target 計算得到 gradients。真正會接受 optimizer 更新的是 online network。target 只有在 hard update（把 online 參數整份複製到 target 的硬同步）發生時才會換成新的參數副本。

## `terminated` 和 `truncated` 的差別會改變 target

Replay Buffer 仍然分開保存 `terminated` 和 `truncated`。兩者都可能讓 environment 接下來 reset，但它們代表的原因不同：

- `terminated=True`：遊戲本身已經進入真正的終止狀態；
- `truncated=True`：遊戲被外部限制提前截斷，例如達到時間上限。

因此 target helper 只接受 `terminated` mask，不把所有「episode 結束」都自動改成 terminal。這保留了 Day 3、Day 9 的資料語意：如果 transition 是 truncated 但 `terminated=False`，它仍然可以 bootstrap。

用一個簡化數字看差別：若 `reward=1`、`gamma=0.5`、下一個 state 的最大 Q-value 是 `4`，非 terminal target 是 `1 + 0.5 × 4 = 3`；若 `terminated=True`，target 則是 `1`。這不是兩種 loss，而是同一個 Bellman target 對不同 episode 結束原因的不同處理。

## Hard update：固定的是一段期間，不是永久不變

**Hard update** 指的是在某個時間點，把 online network 的參數整份複製到 target network。第一次同步讓兩者從同一個起點開始；之後 online 可以每一步更新，target 則留在上一次同步的版本。

同步時機用 `target_update_interval` 表示。它是「隔多少個 global step（從訓練開始累計的步數）同步一次」的設定，不是由理論唯一決定的正確答案。可重用的判斷可以寫成：

```python
if should_update_target(global_step, target_update_interval):
    hard_update(target, online)
```

目前的規則是 `global_step % target_update_interval == 0`，而且 step `0` 會觸發初始同步；interval 必須是正整數。Day 11 不替所有訓練情況選一個永遠正確的數字：interval 太短，target 很快又會變成 moving target；interval 太長，target 雖然穩定，卻可能落後 online 太久。Day 12 先需要一個 development baseline，Day 14 再用受控實驗比較設定。

這也說明了「固定 target」的精確意思：**在兩次 hard update 之間固定，而不是從此不再更新。**

## 用真實 DQN output 看同步前後的分歧

前面的三行 output 已經顯示差異數字，但把 online 和 target 的 Q-values 放在同一張圖上，更容易看出「誰先改變」。這次圖表要回答的技術問題是：**只做一次 online optimizer step 時，target network 的輸出是否保持原樣；hard sync 後兩者是否重新一致？**

[![實際 DQN forward output 在初始同步、online 更新後與 hard sync 後的變化，以及三個階段的 max absolute difference](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/d8ddf24/assets/day11/target-network-sync.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/d8ddf24/assets/day11/target-network-sync.png)

這張圖不是長時間 Breakout training 的成績圖，而是 seed `42`、CPU、`(2, 4, 84, 84)`、每個值以 `float32`（32 位元浮點資料型別）儲存的 synthetic batch（為了固定實驗輸入而建立的合成批次）上，真實 DQN forward 的結果。上半部只畫 batch 中第 `0` 筆 sample 的四個 Q-values：實線是 online，虛線是 target；顏色區分四個 action。初始 hard sync 時兩組線重疊，online optimizer step 後只有實線移動，最後一次 hard sync 又讓兩組線重疊。

下半部直接畫整個 batch 的最大絕對差距 `max |online - target|`：三個階段依序是 `0.00000000`、`0.00746517`、`0.00000000`。同一次執行還量到 online parameter 的最大變化是 `0.00250000`，target parameter 的變化是 `0.00000000`。因此這個實驗支持的是「兩個 instance 沒有共用會同步變動的參數，且 hard update 確實重新複製了值」；它不支持「DQN 已經學會 Breakout」或「某一個 interval 一定能讓訓練成功」。圖上的 network 仍是隨機初始化，synthetic batch 也只是為了隔離同步機制，不是遊戲表現評估。

## Replay Buffer 和 Target Network 解決的是不同的不穩定來源

兩者常常一起出現在 DQN，但它們不是同一個技巧的兩個名字：

| 機制 | 它改變什麼 | 主要處理的問題 |
| --- | --- | --- |
| Replay Buffer | 保存過去 transition，訓練時隨機重用其中一批 | 避免模型只連續看到高度相關的最新資料，也讓一筆經驗可以被多次利用 |
| Target Network | 暫時保存 online network 的參數副本，週期性同步 | 避免 Bellman target 在每個 optimizer step 後立刻跟著目前估計改變 |

Replay Buffer 穩定的是「拿哪些資料來學」；Target Network 穩定的是「這批資料要拿什麼 target 來比較」。前者不能取代後者，因為即使 sample 已經被打散，同一個 network 仍然可能一邊產生答案、一邊被修改。

## Target Network 不是 RL 成功的保證

Target Network 解決的是 moving target 這個特定的 feedback 問題，並不會自動修正 reward 設計、探索不足、資料分布偏差、學習率不合適或模型估計偏差。它也引入了新的取捨：target 太新時穩定性不足，target 太舊時又可能提供落後的估計。

所以今天的結果應該精確地解讀成：online network 可以接受 optimizer 更新；target network 是不同的 model instance，不進 online optimizer；target 計算不建立 gradient graph；`terminated=True` 時不 bootstrap，truncated 不會被自動當成 terminal；hard update 讓 target 在一段 interval 內固定，之後再重新同步。

這些元件現在仍然是分開驗證的。下一步自然會出現：environment 每走一步、Replay Buffer 何時抽 batch、online loss 何時更新、target 何時同步，究竟要如何放進同一條 training loop？那就是 Day 12 要回答的問題。

上一篇：[Day 10 — Exploration vs. Exploitation](day10-exploration-vs-exploitation.md)

下一篇：Day 12 — 完整 DQN training loop
