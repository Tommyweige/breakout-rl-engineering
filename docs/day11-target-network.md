# Day 11｜Target Network：別讓 DQN 的學習目標每一步都跟著自己跑

Day 10 已經解決了「Agent 要怎麼在探索與利用之間選 action」。接下來真的開始訓練 DQN 時，還有另一個更麻煩的問題：**模型一邊被更新，拿來當作學習目標的數字也可能跟著一起變。**

這正是 Target Network 要處理的事情。

先回到 Q-learning 最核心的更新概念。模型會拿目前的估計 `Q(s, a)`，去接近一個由 reward 和下一個 state 算出的目標值：

```text
target = reward + gamma × max Q(next_state, action)
```

`gamma` 是折扣因子，用來控制「未來可能得到的 reward」在目前目標裡占多少比重。

如果 Q-value 只是放在一張表裡，這件事還比較單純。但 DQN 之後，Q-value 改成由神經網路估計。問題也從這裡開始。

## 同一個 network 同時當學生和答案來源

假設現在只有一個 DQN。

它先算出目前 state 的 `Q(s, a)`，接著又用自己算出下一個 state 的 `Q(next_state, a)`，把其中最大的值拿來組成學習目標。之後 optimizer 會根據兩者的差距調整 network 參數；optimizer 可以先理解成「根據誤差去修改模型參數的更新機制」。

問題是，network 的參數一被修改，下一次拿同一個 network 算出來的目標也會跟著改。

模型等於一邊追答案，一邊又在改答案。

這種情況通常稱為 **moving target（移動中的目標）**。它不代表每一次更新一定失敗，而是讓學習變得更難穩定：模型剛往某個方向修正，下一步參考的目標又因為同一組參數改動而一起移動。

這裡有一個容易混淆的地方：`torch.no_grad()` 只能阻止 PyTorch 為這段計算保存之後反向傳播需要的資訊，**它不能阻止同一個 network 在 optimizer 更新後產生不同的 Q-value**。

所以真正要拆開的不是「要不要算 gradient」，而是**目前正在學習的 network，和提供下一步參考值的 network，不要每一步都一起變。**

## Online Network 負責學，Target Network 暫時當參考

DQN 的做法是準備兩個彼此獨立的 network：

| Network | 角色 |
| --- | --- |
| **Online Network** | 持續接受訓練更新，代表目前正在學習的 DQN |
| **Target Network** | 暫時保留一份較舊的參數，用來計算下一個 state 的參考 Q-value |

一開始，Target Network 會先複製 Online Network 的參數，所以兩者輸出相同。之後訓練只更新 Online Network，Target Network 暫時不動。等經過一段時間，再把 Online Network 的參數整份複製過去。

因此兩者一定要是**兩個獨立的模型物件**。

如果只是寫成 `target = online`，那不是複製模型，而只是讓兩個變數指向同一個 network。Online 一更新，Target 也等於同時更新，Target Network 就失去存在的意義。

下面這張圖把兩個 network 在 DQN 裡的角色放在一起看：

[![Online Network 與 Target Network 在 Vanilla DQN target 計算中的角色分工](https://github.com/Tommyweige/breakout-rl-engineering/blob/d8ddf24e9d5ad5d4dacccc4f85eb659c413b6867/assets/day11/target-network-flow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/d8ddf24e9d5ad5d4dacccc4f85eb659c413b6867/assets/day11/target-network-flow.png)

Online Network 負責產生目前 state 的 Q-value；Target Network 則負責看下一個 state，提供計算學習目標需要的 Q-value。虛線箭頭表示參數同步的方向是 **Online → Target**。

## Vanilla DQN 的學習目標怎麼算

加入 Target Network 後，Day 11 使用的 Vanilla DQN target 可以寫成：

```text
y = reward + gamma × (1 - terminated) × max Q_target(next_state, action)
```

其中 `Q_target` 表示 Q-value 是由 **Target Network** 算出來的。

假設下一個 state 有四個 action，而 Target Network 給出的 Q-values 是：

```text
[1.2, 0.7, 2.5, 1.8]
```

那這裡會取最大的 `2.5`，再和 reward、gamma 組成這筆 transition 的學習目標。

這也是 Vanilla DQN 的重要特徵：**「選哪個下一步 action」和「評估這個 action 值多少」目前都由 Target Network 完成。** Day 17 的 Double DQN 才會把這兩個角色拆開：由 Online Network 選 action，再由 Target Network 評估它。

另外，「Target Network 暫時固定」不是說所有 transition 的 target 數字都完全一樣。不同 transition 有不同的 reward、next state，因此算出的 target 當然會不同。**真正固定的是產生下一步 Q-value 的那組 network 參數，在兩次同步之間不會因每個 optimizer step 立刻改變。**

## `terminated` 決定還要不要把未來價值算進來

公式裡的 `(1 - terminated)` 是為了處理真正的終止狀態。

如果 `terminated=False`，代表這個 transition 之後還有合理的未來狀態，因此可以把下一個 state 的估計價值算進 target。

例如：

```text
reward = 1
gamma = 0.5
下一個 state 最大 Q-value = 4
```

那麼：

```text
target = 1 + 0.5 × 4 = 3
```

但如果 `terminated=True`，代表環境真的進入終止狀態，後面已經沒有應該延續的未來價值，此時 target 就只剩 reward：

```text
target = 1
```

這裡也要延續前面的資料語意：`truncated=True` 不等於 `terminated=True`。`truncated` 通常表示 episode 因外部限制而被截斷，例如時間上限；它不應該自動被當成真正的 terminal state。

## Hard Update：Target 固定一段時間，再追上 Online

**Hard Update（硬同步）** 指的是在某個時間點，把 Online Network 的參數整份複製到 Target Network。

整個節奏可以理解成三個階段：

1. 一開始同步，Online 和 Target 相同；
2. Online 持續學習，兩者逐漸分開；
3. 到了同步時機，再把 Online 的最新參數複製給 Target，兩者重新一致。

兩次同步之間隔多少步，通常用 `target_update_interval` 表示。

如果 interval 太短，Target Network 幾乎一直追著 Online 變，穩定效果會變弱；如果 interval 太長，Target 又可能落後目前的 Online Network 太多。所以這不是一個存在「唯一正確答案」的數字。Day 12 只需要先選一個開發用 baseline，之後 Day 14 再透過實驗比較不同設定。

## 真實 DQN output：Online 先變，Target 在同步後才變

這次 Day 11 也做了一個最小實驗，專門確認兩個 network 是否真的分開運作。

流程是：先讓 Online 和 Target 完全同步，只更新一次 Online Network，最後再做一次 Hard Update。實際量到兩個 network 輸出的最大差距是：

```text
初始同步後        0.00000000
Online 更新後     0.00746517
再次 Hard Update  0.00000000
```

[![Online Network 更新後與 Target Network 分歧，再經 Hard Update 重新一致的真實 DQN 輸出](https://github.com/Tommyweige/breakout-rl-engineering/blob/d8ddf24e9d5ad5d4dacccc4f85eb659c413b6867/assets/day11/target-network-sync.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/d8ddf24e9d5ad5d4dacccc4f85eb659c413b6867/assets/day11/target-network-sync.png)

圖的上半部顯示同一筆輸入下四個 action 的 Q-values。初始同步時 Online 和 Target 重疊；只更新 Online 後，Online 的輸出改變，但 Target 仍停留在原本的位置；再次同步後，Target 才追上 Online。

下半部把這個差距直接濃縮成 `max |online - target|`，因此可以很清楚看到 `0 → 0.00746517 → 0`。

這個實驗使用的是固定 seed `42`、CPU 和人工建立的固定輸入資料。這樣做是為了把「Target Network 是否真的保持不動」單獨拿出來驗證，而不是測遊戲表現。因此這張圖能證明同步機制正常，**不能證明 Agent 已經學會 Breakout，也不能證明目前的 update interval 最好。**

## Target Network 不進 optimizer，也不需要 gradient

Online Network 是正在學習的模型，所以 optimizer 會修改它的參數。

Target Network 的角色不同：它在兩次同步之間只是提供參考值，因此不應該被同一個 optimizer 更新。計算 Target Network 輸出時也不需要保留 gradient，也就是之後用來反向調整參數的變化資訊。

因此目前實作會讓 Target Network 保持不參與梯度更新，並在計算 target 時使用 no-grad。它的參數真正發生變化，只應該是在 Hard Update 把 Online 的參數複製過來時。

## Target Network 只是讓學習條件更穩定，不是成功保證

Target Network 並不會自動解決所有強化學習問題。

如果探索不足、reward 設計有問題、學習率不合適、Replay Buffer 裡的資料分布很差，DQN 一樣可能學不好。Target Network 只是把其中一個不穩定來源——「學習目標隨 Online Network 每一步一起移動」——暫時隔開。

到 Day 11 為止，我們已經有：DQN、Replay Buffer、epsilon-greedy，以及 Target Network。這些元件目前都能各自運作，但還沒有真正被放進同一條訓練流程。

Day 12 就會把它們接起來：環境產生 transition、Replay Buffer 收資料、抽 batch、Online Network 計算誤差並更新、Target Network 按照 interval 同步。到了那一步，才會第一次形成完整的 DQN training loop。
