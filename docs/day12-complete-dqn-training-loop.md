# Day 12｜完整 DQN training loop：從一筆 transition 到一次參數更新

Day 11 已經把「下一個 state 的價值要從哪裡估計」這個參考來源暫時固定下來，避免正在更新的模型一邊學、一邊改自己的答案。但到目前為止，環境互動還沒有真正變成一次參數更新。今天的核心問題是：一筆遊戲經驗，如何一路變成會改變模型參數的梯度？

這裡的 DQN（Deep Q-Network）是把畫面轉成各個動作價值的神經網路；Replay Buffer 則是保存過去互動紀錄、讓訓練可以稍後抽樣重用的經驗池。收集新資料時，epsilon-greedy 會以 epsilon 機率隨機探索，否則選目前估計最好的動作；Target Network 則是暫時保持舊參數、用來提供下一步價值參考的另一份網路。今天先把這四個角色接成一條可觀察的流程。

今天先跑一個很短的 CPU smoke run。它在 1,000 個 environment steps 中完成 243 次 optimizer updates；optimizer 是根據 loss 的梯度修改模型參數的更新機制。這次也完成 11 次 target synchronizations，並把 loss、Q-value、epsilon 與 replay size 寫進 CSV。這代表整條流程確實動起來了，但不代表 Agent 已經學會 Breakout。真正值得理解的問題是：**環境產生的那一筆資料，究竟如何一路變成會改變 Online Network 的梯度？**

## 一千步之後，為什麼還不能說 Agent 學會了？

先看這次真實 run 的結果。圖中的資料全部來自 `day12-smoke-seed42-reproducible` 的 `metrics.csv`，不是為了示意而手動填入的數字。

[![Day 12 真實 CPU smoke run 的 raw episode return、Huber loss、selected Q mean 與 epsilon](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6467c33363afcaa061995cc58e0a5974844bd87e/assets/day12/training-overview.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6467c33363afcaa061995cc58e0a5974844bd87e/assets/day12/training-overview.png)

這張 overview 同時顯示四種不同性質的訊號：左上是每一步累積中的 raw episode return，右上是每次 optimizer update 才會出現的 Huber loss，左下是被選中 action 的 Q-value 平均，右下則是行為策略使用的 epsilon。可以看到 loss 有有限值、Q-value 也隨著更新改變，epsilon 依排程從約 `0.9` 降到 `0.05`；這些都支持「訓練程式正在執行」。

這次完成了四個 episode，raw return 依序是 `2、3、0、0`，第五局在 1,000 steps 時仍未結束。這不足以支持「Agent 已經學會」的結論。要判斷學習是否有效，還需要更長的訓練、固定的 evaluation protocol，以及和 baseline 的比較；這些屬於後面的 Day 13 與 Day 15。

## 一個 environment step，不等於一次 optimizer step

最容易讓 training loop 失去意義的混淆，是把所有東西都叫作 step。

這裡的 **environment step** 是 Agent 對預處理後的 Breakout 呼叫一次 `env.step(action)`。Atari 的 frame skip 已經由前面的 preprocessing wrapper 處理，所以今天的 `global_step` 只代表一次 Agent/environment interaction，不代表單一 emulator frame。

**Optimizer step** 則是神經網路真的根據一批資料反向傳播，並修改 Online Network 參數一次。兩者的關係不是一比一：Replay Buffer 必須先累積到 `learning_starts`，之後也只在符合 `train_frequency` 時才做更新。

這次 smoke preset 使用 `learning_starts = 32`、`batch_size = 8`、`train_frequency = 4`。因此第 1 到第 31 個 environment steps 只收集資料；第 32 步第一次更新，之後每隔四個 environment steps 更新一次。從第 32 步到第 1,000 步一共是：

```text
(1000 - 32) / 4 + 1 = 243 次 optimizer updates
```

這個數字不是模型表現，而是 loop 的時序是否符合設定的直接證據。`learning_starts` 存在的原因也很實際：如果 buffer 裡只有幾筆高度相似的開場畫面，就立刻抽一個 mini-batch，更新看到的資料既少又偏，還可能根本湊不滿 batch size。先收集一段經驗，再開始學，至少讓第一次更新有一個明確的資料邊界。

一次 interaction 到 update 的實際關係如下。這是一張依照目前 trainer 資料流整理的結構圖；圖中的 `raw reward` 和 `training reward` 分開，是因為兩者在不同地方有不同用途。

[![Agent、Breakout、Replay Buffer、Online Network 與 Target Network 在一次 DQN training loop 中的訊息順序](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6467c33363afcaa061995cc58e0a5974844bd87e/assets/day12/dqn-training-loop.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6467c33363afcaa061995cc58e0a5974844bd87e/assets/day12/dqn-training-loop.png)

這張圖要看的不是每一個函式名稱，而是順序：先和環境互動並保存 transition，達到 warm-up 與頻率條件後才抽 batch；Online Network 和 Target Network 在 update 中各自扮演不同角色，最後才寫 metrics，並在指定間隔保存 checkpoint。

## Replay 裡的哪個 Q-value 才是這筆資料要更新的答案？

一個 batch 的 state 送進 DQN 後，network 會一次輸出所有 action 的 Q-values。假設 Breakout 有四個 action，輸出可能是：

```text
Q(s, ·) = [0.4, 0.1, 0.8, 0.3]
```

但 Replay Buffer 裡的這筆 transition 還保存了當時實際採取的 action。例如 action 是 `2`，這次更新真正要比較的是 `0.8`，不是四個值的平均，也不是每個值都對應同一個 reward。

在 PyTorch 裡，`gather` 的作用就是沿著 action 維度取出每筆 transition 對應的值：

```python
all_q = online_network(states)                    # (B, action_count)
q_selected = all_q.gather(1, actions[:, None]).squeeze(1)

with torch.no_grad():
    next_q_max = target_network(next_states).max(dim=1).values
bootstrap_mask = (~terminated).to(dtype=next_q_max.dtype)
target = rewards + gamma * bootstrap_mask * next_q_max

loss = torch.nn.SmoothL1Loss()(q_selected, target)
```

這段程式把兩個 state 的角色分開了。`states` 用來問「當時做的 action 現在估計值多少」；`next_states` 用來估計「從下一個 state 往後還可能得到多少」。`actions[:, None]` 只是把 `(B,)` 的 action 編號變成 `(B, 1)`，讓 `gather` 能對每一列挑一個 Q-value。`terminated` 是布林值，因此要先用 `~terminated` 取反，再轉成和 Q-value 相同的數值型別；直接對布林 Tensor 寫 `1 - terminated` 在 PyTorch 中不是可執行的寫法。

## Bellman target 為什麼只在真正終止時停止 bootstrap？

模型的預測 `q_selected` 還不是答案，它要接近一個由即時 reward 與下一個 state 組成的目標。這個目標叫 **Bellman target**，可以先用白話寫成：

```text
target = 這一步拿到的 reward
       + 折扣後的下一個 state 最大 Q-value
```

`gamma` 是折扣因子，控制未來價值在目前目標中占多少。若這筆 transition 真的進入終止狀態，就沒有合理的下一步價值可以接上，所以要把未來那一項遮掉：

```text
target = reward + gamma × (1 - terminated) × max Q_target(next_state, ·)
```

這裡刻意只使用 `terminated`。`terminated=True` 表示遊戲本身真的結束；`truncated=True` 則可能只是外部時間限制讓 episode 暫停。兩者都會讓環境 reset，但不代表兩者在學習目標中的語意相同。把 `truncated` 自動當成 terminal，會把本來仍可延續的未來價值截斷。

Day 12 使用的是 vanilla DQN：下一個 state 的最大值由 Target Network 同時負責「挑最大 action」和「評估最大值」。這不是 Double DQN；Double DQN 會在 Day 17 再把 action selection 與 target evaluation 拆給不同 network。

## Huber loss 有值，究竟代表什麼？

把 `q_selected` 和 `target` 放在一起比較後，差距會變成 loss。這裡使用 **Huber loss**，在 PyTorch 中叫 `SmoothL1Loss`。它在誤差小時像平方誤差一樣細緻，誤差很大時則轉成比較接近絕對值的成長方式，因此單筆異常大的 TD error 不會像純 MSE 那樣支配整個更新。這是 Day 12 的穩健 baseline，不是宣稱它一定比其他 loss 最好。

這次 run 的 loss 圖來自 243 個真實 optimizer updates：

[![Day 12 真實 CPU smoke run 的 DQN Huber loss](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6467c33363afcaa061995cc58e0a5974844bd87e/assets/day12/training-loss.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/6467c33363afcaa061995cc58e0a5974844bd87e/assets/day12/training-loss.png)

圖上少數較高的尖峰表示某些 mini-batch 的預測與 Bellman target 差距較大；後面大量接近零的點只表示當下抽到的 batch 誤差較小。它可以用來檢查 loss 是否 finite、是否突然爆掉，但不能單獨證明 policy 變好。loss 下降可能只是模型更貼近目前 Replay Buffer 的分布，不一定等於遊戲分數上升。

## optimizer 實際改了誰？

一次 update 的 gradient lifecycle 很短，但每個順序都有意義：

```text
zero_grad
→ loss.backward()
→ gradient clipping（若啟用）
→ optimizer.step()
```

Optimizer 只收到 Online Network 的 parameters。Target Network 在兩次 hard update 之間保持不動，也不參與反向傳播；計算它的 Q-values 時使用 `no_grad`。每經過 `target_update_interval` 個 environment steps，才把 Online Network 的參數複製給 Target Network。

這次 smoke preset 的同步間隔是 100 steps。summary 顯示 11 次 synchronization，包含一開始建立兩個相同 network 時的初始複製，以及第 100、200……1,000 步的十次 hard update。這個計數方式讓「初始條件」和「訓練中同步」都能在 artifact 裡被看見。

如果 gradient norm 或 loss 變成 NaN / infinity，trainer 會立刻停止並寫 diagnostic checkpoint，而不是讓壞掉的參數繼續污染後面的資料。Day 13 會把這種「程式沒 crash 但訓練不可信」的診斷流程再拆開。

## 為什麼 Replay 存 clipped reward，metrics 卻要保留 raw score？

Atari reward 的數值可以直接代表遊戲得分，但訓練 baseline 有時會把它裁成符號：正數變 `+1`、零變 `0`、負數變 `-1`。這種 **reward clipping** 是訓練輸入的選擇，目的是讓不同幅度的 reward 不要讓更新尺度差異太大；它不應該偷偷改寫我們用來評估遊戲表現的分數。

因此每次 transition 同時保留兩個概念：Replay Buffer 存的是送進 Bellman target 的 training reward，metrics 則累加環境原本回傳的 raw reward，形成 raw episode return。這次圖上左上角的回報就是後者；它不是 clipped reward 的總和。若把兩者混在一起，最後即使看到一條上升曲線，也不知道那是在看遊戲實際得分，還是在看訓練用的符號化訊號。

## checkpoint 能恢復什麼，不能恢復什麼？

每個 run 會留下三類結構化資料：

- `config.json` 保存這次 run 的超參數、seed 與 device；
- `metrics.csv` 逐 environment step 保存 episode、epsilon、loss、selected Q、target mean、gradient norm、replay size、optimizer updates 與 target sync；
- `summary.json` 保存最後步數、更新次數、同步次數、SPS 與最後 checkpoint。

checkpoint 本身保存 Online / Target Network、optimizer state、`global_step`、episode count、更新與同步計數，以及 Python、NumPy、PyTorch 和 action sampler 能保存的亂數狀態。Replay Buffer 沒有被一併寫入，因為完整的 Atari frame storage 很大；resume 時模型與 optimizer 可以接著使用，但 Replay 必須重新 warm up。因此這是「可恢復的訓練狀態」，不是跨環境 bit-exact 的完整續跑。

可用同一個 run 的 checkpoint 重新開始，例如：

```powershell
python train_dqn.py --resume runs/day12-smoke-seed42-reproducible/checkpoints/step-00001000.pt --total-steps 2000 --device cpu
```

這個命令的重點不是宣稱結果和原本完全相同，而是把恢復邊界說清楚：模型狀態可以載入，經驗資料需要重新累積。

## 這次 run 真正證明了什麼？

`day12-smoke-seed42-reproducible` 使用 seed `42`、CPU、1,000 environment steps，實際得到 243 次 updates、finite loss、256 筆 replay，以及 11 次 target synchronization。這些結果證明 transition → replay → batch → Bellman target → loss → optimizer 的資料路徑已經接通，並且 checkpoint 與可重建的 metrics artifact 能留下來。

但五個 episode 的低 raw return 也提醒我們，**「程式有跑」和「Agent 有學」是兩個不同的問題。** 前者可以由 metrics、finite checks、update count 和 checkpoint 驗證；後者還要看長期 evaluation return、不同 seed、baseline 和訓練穩定性。今天建立的是可以被觀察、被恢復、也值得除錯的 training loop，不是已經成功的 Breakout policy。

下一步的問題因此很自然：如果 loss 有數字、Q-value 也在變，但 return 沒有改善，要怎麼分辨是探索、資料、target、梯度，還是 evaluation protocol 出了問題？Day 13 會從這個邊界開始建立 DQN training diagnostics。
