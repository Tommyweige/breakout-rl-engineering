# Day 12｜完整 DQN Training Loop：一筆遊戲經驗怎麼真的變成模型更新

前面幾天，我們把 DQN 需要的零件一個一個拆開來做：模型可以輸出 Q-values、Replay Buffer 可以保存經驗、epsilon-greedy 可以決定要探索還是利用、Target Network 也能提供比較穩定的下一步估計。

但直到 Day 12，這些東西才第一次真正接成一條會修改模型參數的訓練流程。

今天真正要回答的是：**Agent 在 Breakout 裡做出一個 action 之後，那筆互動資料怎麼一路變成 Replay Buffer 裡的 transition、Bellman target、loss，最後真的改變 Online Network？**

這一天的完成標準不是「1,000 steps 之後就會玩 Breakout」。我們先確認更基本、也更重要的事情：整條 training pipeline 有沒有接對、模型有沒有真的更新、Target Network 有沒有在正確時間同步，以及訓練過程是否留下足夠的資料讓我們之後能判斷哪裡出了問題。

## 所有元件第一次真的接在一起

完整的 DQN training loop 可以先拆成兩件不斷交替的工作：**收集經驗**，以及**利用過去經驗更新模型**。

[![DQN training loop 分成收集經驗與更新模型兩個不同節奏，Replay Buffer 位在兩者之間](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/4c295d296a358fec55010424d0575021953bd6db/assets/day12/dqn-training-loop-overview.svg?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/4c295d296a358fec55010424d0575021953bd6db/assets/day12/dqn-training-loop-overview.svg)

上半部是**收集經驗**。目前的 state 經過 epsilon-greedy 選出 action，Breakout 執行之後回傳新的結果，最後形成一筆 transition 放進 Replay Buffer。這件事幾乎每個 environment step 都會發生。

下半部才是**更新模型**。只有 Replay Buffer 已經累積到足夠資料，而且目前 step 符合訓練頻率時，才會抽一個 mini-batch。Online Network 提供目前 `Q(s, a)`，Target Network 提供下一個 state 的參考 Q-value，兩邊組成 Bellman target 和 Huber loss，最後 optimizer 才真正修改 Online Network。

圖中最重要的其實是 Replay Buffer 的位置：**它把「剛剛發生的互動」和「現在拿來訓練的資料」隔開了。** Agent 每一步都在產生新經驗，但模型不需要每一步都立刻拿最新那筆資料更新。

這也帶出 training loop 裡第一個很重要的觀念：**environment step 和 optimizer step 是兩條不同的時鐘。**

## Environment Step 與 Optimizer Step 是兩條不同的時鐘

Training loop 裡最容易混淆的詞之一就是 `step`。

這個專案的 **environment step** 指的是 Agent 做出一次 action，並呼叫一次預處理後的 `env.step(action)`。Atari 內部的 frame skip 已經由環境 wrapper 處理，所以 `global_step` 記的是 Agent 和環境互動了幾次，不是 emulator 畫了幾張 frame。

**Optimizer step** 則完全不同。它表示模型真的完成一次反向傳播，並修改 Online Network 的參數。

兩者通常不會一比一發生，因為真正開始訓練前還有兩個條件：

- `learning_starts`：Replay Buffer 至少先累積到多少經驗，才開始訓練；
- `train_frequency`：每隔多少個 environment steps 做一次 optimizer update。

這次 smoke run 使用：

```text
learning_starts = 32
train_frequency = 4
batch_size      = 8
```

所以前 31 個 environment steps 只負責收集資料，第 32 步才第一次更新，之後每隔 4 個 environment steps 更新一次。

因此 1,000 個 environment steps 最後會得到：

```text
(1000 - 32) / 4 + 1 = 243 次 optimizer updates
```

實際執行結果也正好是 243 次。

這個數字不是遊戲成績，但它是一個很有用的 correctness check：**trainer 的時序真的按照我們設定的規則發生。**

`learning_starts` 也不只是為了避免「Buffer 太小抽不到 batch」。剛開始的 Replay Buffer 幾乎只有很接近的開場畫面，如果太早開始學，模型會一直從非常窄的資料分布更新。先收集一段經驗，至少能讓第一次更新看到比較多樣的 transition。

## 一筆 Transition 只更新當時真正做過的 Action

DQN 對一個 state 做 forward 時，會一次輸出所有 action 的 Q-values。假設某個 state 得到：

```text
Q(s, ·) = [0.4, 0.1, 0.8, 0.3]
```

但 Replay Buffer 裡的 transition 還記著：當時 Agent 真正採取的是哪個 action。

如果當時做的是 action `2`，那這筆資料真正要更新的是 `0.8`。原因很直接：這筆 reward 和 next state 都是「執行 action 2 之後」得到的結果，其他三個 action 並沒有在這筆 transition 裡真的發生。

PyTorch 的 `gather` 在這裡就是做這件事：**從每一筆 state 的所有 Q-values 中，挑出當時真正採取的 action 對應的 Q-value。**

假設 batch size 是 8，Online Network 原本會輸出 8 組 Q-values；挑完之後就留下 8 個 `Q(s, a)`，每一筆 transition 各有一個 prediction。

## Bellman Target 決定 Q-value 要往哪裡靠

有了目前的 `Q(s, a)`，還需要一個學習目標。

Day 12 延續 Day 11 的 Vanilla DQN target：

```text
target = reward
       + gamma × (1 - terminated)
       × max Q_target(next_state, action)
```

白話來說，一筆 transition 的學習目標由兩部分組成：

1. 這一步已經真的拿到的 reward；
2. 如果遊戲還沒真正結束，下一個 state 未來可能帶來的價值。

第二部分由 Target Network 計算。這樣 Online Network 即使剛做完一次更新，拿來當參考的下一步 Q-value 也不會立刻跟著同一組參數一起變。

`terminated=True` 時，遊戲已經進入真正的終止狀態，後面沒有合理的未來價值可以再接，因此 target 只剩 reward。

`truncated=True` 則不能直接當成同一件事。它可能只是 episode 因外部限制而被截斷，例如時間上限；這不代表遊戲世界本身進入 terminal state，所以不能自動把未來價值切掉。

這裡仍然是 Vanilla DQN：下一個 state 裡「哪個 action 最大」以及「這個最大值是多少」，都由 Target Network 完成。到 Day 17 的 Double DQN 才會把這兩個角色拆開。

## Huber Loss 把差距變成真正的更新訊號

現在手上有兩個數字：

```text
prediction = Online Network 的 Q(s, a)
target     = Bellman target
```

兩者之間的差距，就是這一次模型需要修正的方向。這個 prediction 和 target 的差距，在強化學習裡常叫 **TD error**。

Day 12 使用 **Huber loss**，PyTorch 裡叫 `SmoothL1Loss`。它在誤差比較小時，會細緻地處理 prediction 和 target 的差距；誤差很大時，成長又不像純平方誤差那麼激烈，因此比較不容易讓少數特別大的 TD error 支配整個更新。

這是一個穩健的 baseline，不代表 Huber loss 在所有情況下永遠最好。

Loss 算出來之後，才會進入真正修改模型參數的階段：清掉上一輪 gradient、做 backward 計算這次的參數變化方向、必要時限制過大的 gradient，最後 optimizer 才更新 Online Network。

**Target Network 不在這個 optimizer 裡。** 它在兩次同步之間保持不動，只在指定的間隔把 Online Network 的參數整份複製過去。

## Reward Clipping 改的是訓練訊號，不是遊戲分數

Atari 的 reward 同時扮演兩個不同角色，這兩個角色不能混在一起。

第一個角色是**訓練訊號**。DQN baseline 可以把 reward clipping 成：正數變 `+1`、零維持 `0`、負數變 `-1`。Replay Buffer 保存的是這個真正拿去算 Bellman target 的 training reward。

第二個角色是**遊戲表現**。如果我們要知道 Agent 在 Breakout 裡到底拿了多少分，就必須保留環境原本回傳的 raw reward，並用它累積 raw episode return。

如果把 clipped reward 和 raw reward 混在一起，之後看到 return 上升，也不知道究竟是遊戲真的得分變高，還是只是在看訓練用的符號化訊號。

所以這兩個數字要從 training loop 一開始就分開保存。

## 真實 Smoke Run：先證明 Pipeline 正確

這次 Day 12 實際跑了一個固定 seed `42`、CPU、1,000 environment steps 的 smoke run。

結果是：

| 指標 | 結果 |
| --- | ---: |
| Environment steps | 1,000 |
| 完成 episodes | 4 |
| Optimizer updates | 243 |
| Target sync count | 11 |
| 最後 Replay size | 256 |
| 最後 loss | 0.000419 |

比最後這幾個數字更重要的是，訓練過程中的 return、loss、Q-value 和 epsilon 都被持續記錄下來，而不是只在最後印一句「training finished」。

[![Day 12 真實 CPU smoke run 的 raw episode return、Huber loss、selected Q mean 與 epsilon](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/5d725ae7d752439d390098726f238dbbd5d01a5a/assets/day12/training-overview.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/5d725ae7d752439d390098726f238dbbd5d01a5a/assets/day12/training-overview.png)

這張圖回答的是另一個問題：**這條 pipeline 實際跑起來之後，各個訊號有沒有真的出現？**

可以看到 epsilon 的確照排程下降；learning warm-up 之後 loss 開始出現；Q-value 也隨著 optimizer updates 改變；episode return 則確實來自實際 Breakout 互動。

這些現象足以支持：**training pipeline 真的在執行，而且 Online Network 真的有被更新。**

但它們還不能支持另一句話：**Agent 已經學會 Breakout。**

## Loss 下降不等於 Policy 變好

這是 Day 12 最重要的界線之一。

這次 run 一共有 243 次 optimizer updates，因此我們也能單獨看每次更新的 Huber loss：

[![Day 12 真實 CPU smoke run 的 DQN Huber loss](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/5d725ae7d752439d390098726f238dbbd5d01a5a/assets/day12/training-loss.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/5d725ae7d752439d390098726f238dbbd5d01a5a/assets/day12/training-loss.png)

圖上的尖峰表示某些 mini-batch 中，Online Network 的 prediction 和 Bellman target 差得比較遠；接近零的點則表示那一批資料上的差距比較小。

Loss 是很重要的健康指標。如果它突然變成 NaN、infinity，或長時間快速爆大，訓練流程很可能有問題。

但 loss 下降只代表模型越來越能貼近**目前 Replay Buffer 裡、目前 Target Network 所定義的學習目標**。這不保證這些 Q-values 最後一定會形成更好的遊戲策略。

這次完成的四個 episode，raw return 只有 `2、3、0、0`。樣本太少、訓練也太短，完全不足以判斷 Agent 是否真的出現學習趨勢。

因此要把兩件事分開：

- **程式有跑**：可以用 optimizer updates、finite loss、Q-value 是否改變、Replay 是否增長、Target 是否同步來驗證；
- **Agent 有學**：要用更長時間的 evaluation return、不同 seed，以及和 baseline 的比較來驗證。

Day 12 只完成前者。

## Metrics 讓「沒有學會」變成可診斷的問題

RL 很常出現一種情況：程式完全沒有 crash，GPU 也一直在跑，但 Agent 其實根本沒有變好。

所以 training loop 不能只記 episode score。至少還要知道 epsilon、loss、Q-value、gradient、Replay size、optimizer update 次數，以及 Target Network 的同步狀態。

每一種 metric 都在回答不同問題。

例如 return 不升、但 loss 很正常，可能代表模型確實在擬合目前資料，只是探索、資料分布或 target 有問題；如果 Q-value 突然快速變得非常大，則可能是估計開始發散；如果 Replay size 一直沒有正常增長，問題甚至可能根本還沒進入神經網路。

Metrics 不會直接告訴我們答案，但它能把「Agent 好像沒在學」這種模糊感覺，拆成可以逐步排查的工程問題。

## Checkpoint 恢復的是主要訓練狀態，不是整條時間線

Day 12 也第一次把 checkpoint 納入 training loop。

Checkpoint 保存 Online Network、Target Network、optimizer、目前的 global step、episode/update 計數，以及能保存的亂數狀態。這代表訓練中斷後，不必重新把模型從隨機初始化開始。

但目前 **Replay Buffer 沒有一起存進 checkpoint**。

原因很直接：Atari Replay Buffer 可能非常大，把整份 frame storage 每次都寫進 checkpoint，會帶來明顯的空間與 I/O 成本。

所以 resume 之後，模型和 optimizer 可以恢復，但 Replay Buffer 必須重新累積到 warm-up 條件後才能再次開始更新。

這代表目前的 resume 是「恢復主要訓練狀態」，不是 bit-exact resume，也不是保證從中斷前的下一個 action 開始，之後每一步都會走出完全相同的結果。

把這個邊界說清楚，比單純寫一句「支援 resume」更重要。

## Day 12 的完成標準：能學習、能觀察、能恢復

到這裡，DQN 第一次有了一條完整閉環。

Agent 用目前的 Online Network 和 epsilon-greedy 選 action；Breakout 產生 transition；Replay Buffer 保存經驗；條件成熟後抽 mini-batch；Online Network 提供 prediction；Target Network 提供 Bellman target；loss 再經過 backward 和 optimizer，真正修改 Online Network。

這次 smoke run 證明的是：**這條閉環可以實際執行、模型參數真的會更新、metrics 能留下來，而且主要訓練狀態可以被 checkpoint 保存。**

它沒有證明 Agent 已經會玩 Breakout。

而這正好帶出 Day 13 真正要處理的問題：**如果 loss 有值、Q-value 也在變、程式完全沒有 crash，但 return 就是不改善，我們要怎麼知道到底是哪一個環節出了問題？**
