# Day 8｜DQN 為什麼不直接告訴 Agent「往左」或「往右」？

Day 7 已經把 Breakout 的四張灰階畫面送進 CNN，最後得到一個長度 3,136 的 feature vector。

現在真正的問題來了：

> **Agent 最後明明只需要按一個按鍵，為什麼 neural network 不直接輸出 `LEFT` 或 `RIGHT`，而是要輸出四個 Q-values？**

這個問題正好是 DQN 和一般影像分類最容易被混在一起的地方。

今天要把 Day 7 留下的最後一段接起來：

```text
Breakout state
(4, 84, 84) uint8
      ↓
float32 / 255
(1, 4, 84, 84)
      ↓
CNN
      ↓
(1, 3136) features
      ↓
Linear 3136 → 512
ReLU
      ↓
Linear 512 → 4
      ↓
Q(NOOP), Q(FIRE), Q(RIGHT), Q(LEFT)
```

把這條路徑畫成一張圖，會更容易看出每一步到底改變了什麼：

![Day 8 DQN forward path](https://github.com/Tommyweige/breakout-rl-engineering-private/raw/73fbe691af4d7d0a7b00893e700d9f1bbb0c36b5/assets/day08/dqn-forward-flow.svg)

這張圖最重要的地方不是最後的 `argmax`，而是 **DQN 先把 state 轉成「每個 action 的價值」，最後才選 action**。

換句話說：

```text
state → Q-values → action
```

而不是：

```text
state → action label
```

這個差別會一路影響後面的 Bellman target、Experience Replay 和 training loop。

## DQN 真正輸出的是什麼？

Day 8 的 `inspect_dqn_network.py` 會建立 `ALE/Breakout-v5` environment，拿到一個真實 `(4, 84, 84)` observation，再送進 DQN：

```powershell
python .\inspect_dqn_network.py --device cpu --seed 42
```

完整資料流是：

```text
Observation        : (4, 84, 84) uint8
Model input        : (1, 4, 84, 84) torch.float32
Feature shape      : (1, 3136)
Output shape       : (1, 4)
Action meanings    : NOOP FIRE RIGHT LEFT
Q-values           : [四個 raw values]
Greedy action      : argmax 對應的 action
Parameter count    : 1,686,180
state_dict diff    : 0.00000000
```

這裡真正重要的不是某一個 Q-value 恰好是多少，而是資料的意義已經一路改變：

```text
28,224 個 pixel values
        ↓
3,136 個 CNN features
        ↓
4 個 action values
```

前半段的 CNN 在整理畫面資訊，最後的 Q head 才開始回答：

> **在這個 state 下，每一個 action 看起來有多值得做？**

## 為什麼輸出不是一個 action？

先想像另一種最直覺的設計：

```text
state
  ↓
neural network
  ↓
RIGHT
```

看起來很合理，因為 Agent 最後確實只會執行一個 action。

但這種輸出只有「答案」，沒有其他 action 的價值資訊。

Q-Learning 真正想學的是：

```text
Q(state, action)
```

也就是同一個 state 下，每一個 action 都有自己的估計值。

對目前 Breakout 的四個 action，可以把 network 想成同時回答四個問題：

```text
如果現在 NOOP，長期累積 reward 看起來是多少？
如果現在 FIRE，長期累積 reward 看起來是多少？
如果現在 RIGHT，長期累積 reward 看起來是多少？
如果現在 LEFT，長期累積 reward 看起來是多少？
```

因此 network 的輸出自然是：

```text
[Q(NOOP), Q(FIRE), Q(RIGHT), Q(LEFT)]
```

而不是一個 action label。

這也把 Day 6 的 Q-table 和現在的 DQN 接了起來。

以前的小型問題可以寫成：

```text
Q-table[state]
    ↓
[每個 action 的 Q-value]
```

現在則是：

```text
neural network(state)
    ↓
[每個 action 的 Q-value]
```

**Q-value 的概念沒有換，換的是取得 Q-value 的方式。**

## 為什麼剛好是四個輸出？

目前 Breakout environment 的 action space 有四個 action：

```text
0 → NOOP
1 → FIRE
2 → RIGHT
3 → LEFT
```

所以：

```text
num_actions = 4
output shape = (B, 4)
```

但 `DQNNetwork` 本身沒有把 `4` 寫死在 CNN 裡，也不需要依賴 ALE object。

建立模型時，action count 從外面傳進去：

```python
num_actions = int(env.action_space.n)
model = DQNNetwork(num_actions=num_actions)
```

所以如果別的 environment 有 6 個 actions，最後一層就可以變成 `(B, 6)`。

這也是為什麼「environment 負責告訴我們有幾個 actions」和「network 負責輸出多少個 Q-values」最好分開。

## 3,136 個 features 怎麼變成四個 Q-values？

Day 7 的 CNN 最後輸出：

```text
(B, 3136)
```

Day 8 在後面接上 fully connected head：

```python
self.q_head = nn.Sequential(
    nn.Linear(self.feature_extractor.feature_dim, 512),
    nn.ReLU(),
    nn.Linear(512, num_actions),
)
```

資料流就是：

```text
3136 features
     ↓
512 hidden activations
     ↓
4 Q-values
```

CNN 前面的工作是利用畫面的空間結構，把局部 pixel pattern 逐層轉成 features。

到了 Flatten 之後，我們現在不再問「某個 feature 在畫面哪個位置」，而是希望把整組 features 綜合起來，估計每個 action 的價值，因此 fully connected layer 很適合放在這裡。

完整 network 目前有：

```text
1,686,180 parameters
```

其中第一個 fully connected layer 就有：

```text
3136 × 512 + 512
= 1,606,144 parameters
```

也就是說，DQN 的大部分 parameters 其實集中在 CNN 後面的這個 head。

## Q-value 是機率嗎？

不是。

這是 Day 8 最容易寫錯的一件事。

影像分類常看到：

```text
logits
  ↓
softmax
  ↓
class probabilities
```

例如：

```text
cat  = 0.70
dog  = 0.20
bird = 0.10
```

它們介於 0 和 1，而且總和等於 1。

但 Q-value 不是分類機率。

Q-value 想表示的是：

> 在 state `s` 採取 action `a` 後，之後按照目前策略繼續走，預期可以得到多少 discounted return。

所以它可以是：

```text
2.7
-0.4
15.2
0.0
```

甚至可以全部是負數。

它們不需要：

```text
全部 >= 0
全部 <= 1
總和 = 1
```

因此 DQN 的最後一層直接回傳 raw values：

```python
q_values = self.q_head(features)
```

**後面沒有 Softmax。**

如果硬套 Softmax，就會把原本的價值尺度壓成一組總和等於 1 的相對比例，這已經不是 Q-Learning 想估計的 quantity。

## `argmax` 到底做了什麼？

當 network 同時輸出四個 Q-values 後，最直接的 greedy action selection 就是找最大的那一個：

```text
Q(NOOP)  = ...
Q(FIRE)  = ...
Q(RIGHT) = ...  ← 最大
Q(LEFT)  = ...
              ↓
           argmax
              ↓
            RIGHT
```

程式裡可以寫成：

```python
greedy_action = q_values.argmax(dim=1)
```

`argmax` 不會改變 Q-values，也不是 neural network 的一層。

它只是回答：

> **這幾個 action value 裡，哪一個最大？**

## 為什麼現在的 argmax 還沒有策略意義？

因為現在 network 的 weights 還只是初始化值。

它從來沒有被告訴過：

- 哪個 action 之後拿到 reward；
- 哪個 state 很危險；
- 球往哪裡飛時應該怎麼移動球拍；
- 哪一個 Q-value 應該往上或往下修。

所以目前其實是：

```text
Breakout state
      ↓
randomly initialized DQN
      ↓
four raw values
      ↓
argmax
      ↓
one random-weight preference
```

這不是 learned policy。

固定 `torch.manual_seed(42)` 只是在固定初始化，方便重現與除錯，不會讓 network 因此變聰明。

## 為什麼還要做 Q-value 視覺化？

Day 8 另外有：

```powershell
python .\visualize_dqn_network.py --device cpu --seed 42
```

它的資料來源是：

```text
make_breakout_env()
      ↓
真實 observation
      ↓
DQNNetwork forward
      ↓
四個 raw Q-values
      ↓
圖 + JSON metadata
```

預設會生成：

```text
/assets/day08/dqn-q-values.png
/assets/day08/dqn-q-values.json
```

這張圖真正要回答的是：

> **四個 network outputs 到底如何一一對應 `NOOP / FIRE / RIGHT / LEFT`？**

而不是比較「哪根柱子最高，所以 Agent 已經學會了什麼」。在模型還沒訓練以前，柱子的高低只反映初始化後的 forward output。

因此 Day 8 對圖片的解讀有一條很重要的界線：

> **可以用圖理解 Q-value 與 action 的對應，但不能把未訓練 Q-values 解讀成遊戲策略。**

## Batch 進來時，為什麼仍然是一個 state 四個輸出？

單一 state：

```text
(1, 4, 84, 84)
      ↓
DQN
      ↓
(1, 4)
```

如果未來 Replay Buffer 一次 sample 32 個 states：

```text
(32, 4, 84, 84)
       ↓
DQN
       ↓
(32, 4)
```

第一個 `32` 是 batch size。

每一列仍然只對應一個 state：

```text
state 0  → 4 Q-values
state 1  → 4 Q-values
...
state 31 → 4 Q-values
```

所以 batch size 增加，不會改變 action count。

這個 `(B, 4)` shape 之後會非常重要，因為 training 時 DQN 會一次處理一整批 transitions。

## `state_dict` 到底保存了什麼？

現在 network 雖然還沒有訓練，但已經可以先驗證一件很實際的事：model parameters 能不能保存，再正確載回來。

PyTorch 常見做法是：

```python
torch.save(model.state_dict(), path)
```

載入時：

```python
model.load_state_dict(torch.load(path, ...))
```

`state_dict` 可以先理解成：

> **這個 model 裡需要保存的 parameter values。**

它包含 CNN 的 weights / bias，也包含 fully connected head 的 weights / bias。

但 `state_dict` 不是整個 Python project，也不是完整 training checkpoint。

載入前仍然需要先建立相同 architecture：

```python
model = DQNNetwork(num_actions=4)
model.load_state_dict(state_dict)
```

Day 8 的 inspection 和 unit test 都會驗證 save/load round-trip。

同一個 input 在保存前後如果得到完全相同的 output：

```text
state_dict diff = 0.00000000
```

代表最基本的 model serialization 沒有問題。

optimizer、training step、replay buffer 等訓練狀態，會等到真正建立 training loop 後再處理。

## Network 寫完了，為什麼還不能開始穩定學習？

現在我們已經有：

```text
state
  ↓
DQNNetwork
  ↓
Q-values
```

但還缺最重要的一件事：

> **這些 Q-values 到底要用哪些 interaction data 來反覆修正？**

如果 Agent 每跟 environment 互動一次，就只拿最新那一筆 transition 訓練，連續 Atari frames 之間會高度相關，而且舊經驗很快就被新的資料淹沒。

所以經典 DQN 不會把 transition 用完就丟掉。

它會先把過去經驗保存起來：

```text
(state, action, reward, next_state, terminated, truncated)
```

之後再從很多過去 transitions 裡隨機抽樣 mini-batch。

這就是下一篇要進入的 **Experience Replay**。

Day 8 解決的是：

> **一個 state 如何經過 CNN 和 Q head，變成所有 actions 的 Q-values？**

Day 9 要接著回答：

> **這個 network 到底要從哪裡取得可以重複學習的 interaction data？**

下一篇：[Day 9 — Experience Replay](/docs/day09-experience-replay.md)
