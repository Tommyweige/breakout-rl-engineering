# Day 8｜從 CNN Features 到四個 Q-values：完成 DQN Network

Day 7 已經把 Breakout 的四張灰階畫面送進 CNN，最後得到長度 3,136 的 feature vector。

Day 8 要補上最後一段：**把這些 features 轉成每個 action 的 Q-value。**

完整路徑會變成：

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
      ↓
argmax
      ↓
greedy action
```

把這條資料流畫出來會更直觀：

![Day 8 DQN forward path](https://raw.githubusercontent.com/Tommyweige/breakout-rl-engineering-private/73fbe691af4d7d0a7b00893e700d9f1bbb0c36b5/assets/day08/dqn-forward-flow.svg)

這張圖最重要的不是最後那個 `argmax`，而是中間的關係：

```text
state → Q-values → action
```

DQN 不會直接把畫面分類成 `LEFT` 或 `RIGHT`。它先估計每個 action 的價值，再從這些價值中選 action。

這個差別會一路影響後面的 Bellman target、Experience Replay 和 training loop。

## 先確認 DQN Network 能正常 forward

在討論 Q-value 前，先確認 Day 8 新增的 network 本身能正常工作。

執行：

```powershell
python -m unittest tests.test_dqn_network -v
```

這次 CPU 執行結果如下：

![Day 8 actual DQN CPU test run](https://raw.githubusercontent.com/Tommyweige/breakout-rl-engineering-private/e45a237ccc5373802e7f4f836e79bcb0fb551cd6/assets/day08/dqn-test-run.svg)

這張圖不是模擬 terminal。內容來自實際測試輸出，原始文字也保存在：

```text
/assets/day08/dqn-test-run.txt
```

這組測試確認了幾件事：batch size 1 和 8 都能得到正確輸出 shape、action count 可以改變、Q head 沒有 Softmax / Sigmoid、gradient 可以從輸出一路回到 CNN，而且 `state_dict` save/load 後同一個 input 仍然得到相同結果。

但它只證明 **network implementation 能正常工作**，不代表 Agent 已經學會玩 Breakout。

## 從 Breakout State 到四個 Action Values

`inspect_dqn_network.py` 會建立 `ALE/Breakout-v5` environment，拿到一個真正的 `(4, 84, 84)` observation，再送進 DQN：

```powershell
python .\inspect_dqn_network.py --device cpu --seed 42
```

資料會依序經過：

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

從 shape 可以看到資料的角色在逐步改變：

```text
28,224 個 pixel values
        ↓
3,136 個 CNN features
        ↓
4 個 action values
```

CNN 前半段主要負責把影像整理成 features；Q head 才開始把這些 features 轉成和 action 直接相關的數值。

## DQN 輸出的不是 Action Label

如果把這個問題當成一般影像分類，很容易設計成：

```text
state
  ↓
neural network
  ↓
RIGHT
```

但 Q-Learning 要學的不是一個 class label，而是：

```text
Q(state, action)
```

也就是同一個 state 下，每個 action 都要有自己的價值估計。

Breakout 現在有四個 actions：

```text
NOOP
FIRE
RIGHT
LEFT
```

所以 network 一次回傳：

```text
[Q(NOOP), Q(FIRE), Q(RIGHT), Q(LEFT)]
```

這其實就是 Day 6 的 Q-table 概念延伸過來。

以前可以想成：

```text
Q-table[state]
    ↓
[每個 action 的 Q-value]
```

現在則變成：

```text
neural network(state)
    ↓
[每個 action 的 Q-value]
```

**Q-value 的意義沒有改變，改變的是取得它的方法。**

## Output Dimension 跟著 Action Space 走

目前 Breakout environment 的 action space 是：

```text
0 → NOOP
1 → FIRE
2 → RIGHT
3 → LEFT
```

因此：

```text
num_actions = 4
output shape = (B, 4)
```

但 `DQNNetwork` 沒有把 `4` 寫死在 model 裡。

建立模型時，action count 從 environment 傳進來：

```python
num_actions = int(env.action_space.n)
model = DQNNetwork(num_actions=num_actions)
```

所以如果另一個 environment 有 6 個 actions，同一個 model class 就可以輸出：

```text
(B, 6)
```

這讓 model architecture 和特定 Atari environment 保持分離，也讓之後做 export 或 inference 時比較乾淨。

## Q Head：3,136 Features → 4 Q-values

Day 7 的 CNN 最後輸出：

```text
(B, 3136)
```

Day 8 在後面加上 fully connected head：

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

CNN 前面的工作是利用影像的空間結構，把 pixel pattern 逐層轉成 features。

到了 Flatten 之後，資料已經是一個 feature vector。這時候我們需要的是把整組 features 綜合起來，估計每個 action 的價值，因此使用 fully connected layers。

完整 network 目前有：

```text
1,686,180 parameters
```

其中第一個 fully connected layer 就有：

```text
3136 × 512 + 512
= 1,606,144 parameters
```

也就是說，大部分 parameters 其實集中在 CNN 後面的第一個 fully connected layer。

## Q-value 不是機率

這是 DQN 很容易和分類模型混淆的地方。

影像分類常見：

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

這些 probability 介於 0 和 1，而且總和等於 1。

Q-value 不需要符合這些條件。

它想表示的是：

> 在 state `s` 採取 action `a` 後，之後按照目前策略繼續走，預期可以得到多少 discounted return。

因此 Q-values 可以長成：

```text
2.7
-0.4
15.2
0.0
```

也可能全部是負數。

所以 DQN 最後一層直接回傳 raw values：

```python
q_values = self.q_head(features)
```

後面**沒有 Softmax**。

如果硬把 Q-values 做 Softmax，就會把原本的價值尺度壓成總和等於 1 的比例，這已經改變了 Q-Learning 想估計的 quantity。

## Argmax：從 Q-values 選出 Greedy Action

network 同時輸出所有 actions 的 Q-values 後，greedy action selection 就是取最大值的位置：

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

`argmax` 不是 neural network layer，也不會修改 Q-values。它只是從輸出裡找出最大值所在的 action index。

## 未訓練模型的 Argmax 沒有策略意義

Day 8 的 network 還沒有訓練。

目前的 weights 只是 PyTorch 初始化出來的值，模型還不知道：

- 哪個 action 之後拿到 reward；
- 哪個 state 對 Agent 有利或不利；
- 球往哪裡飛時應該怎麼移動球拍；
- 哪些 Q-values 應該被提高或降低。

所以現在的流程實際上是：

```text
real Breakout state
        ↓
randomly initialized DQN
        ↓
four raw values
        ↓
argmax
        ↓
one random-weight preference
```

固定 `torch.manual_seed(42)` 只是讓初始化可以重現，方便測試和除錯，不會讓模型因此變成 learned policy。

因此現在可以檢查 Q-value 的 shape、對應關係和數值流向，但不能把 argmax 解讀成「Agent 已經知道應該做這個 action」。

## 用 Q-value 圖確認 Action 對應

Day 8 也準備了視覺化程式：

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

這張圖的用途是把 network outputs 和：

```text
NOOP / FIRE / RIGHT / LEFT
```

一一對齊，並標出 `argmax`。

它可以幫助理解輸出結構，但在模型尚未訓練時，柱子的高低不能被解讀成策略品質。

## Batch 只增加 State 數，不改 Action 維度

單一 state：

```text
(1, 4, 84, 84)
      ↓
DQN
      ↓
(1, 4)
```

如果 Replay Buffer 之後一次 sample 32 個 states：

```text
(32, 4, 84, 84)
       ↓
DQN
       ↓
(32, 4)
```

第一個 `32` 是 batch size。

每一列仍然代表一個 state：

```text
state 0  → 4 Q-values
state 1  → 4 Q-values
...
state 31 → 4 Q-values
```

所以增加 batch size 不會改變每個 state 的 action count。

這個 `(B, num_actions)` shape 之後會直接進入 DQN training。

## 用 `state_dict` 保存 Model Parameters

network 雖然還沒有訓練，但現在已經可以先確認 model parameters 能正確保存和載入。

PyTorch 常見做法：

```python
torch.save(model.state_dict(), path)
```

載入時：

```python
model.load_state_dict(torch.load(path, ...))
```

`state_dict` 保存的是 CNN 和 Q head 裡的 parameter values，例如 weights 和 bias。

它不是整個 Python project，也不是完整 training checkpoint。載入前仍然要先建立相同 architecture：

```python
model = DQNNetwork(num_actions=4)
model.load_state_dict(state_dict)
```

Day 8 的 inspection 和 unit test 都會驗證 save/load round-trip。

如果同一個 input 在保存前後得到完全相同的輸出：

```text
state_dict diff = 0.00000000
```

就代表最基本的 model serialization 沒有問題。

optimizer、training step、replay buffer 等訓練狀態，會等到完整 training loop 再加入 checkpoint。

## 下一步：Experience Replay

現在 DQN 已經可以完成：

```text
state
  ↓
DQNNetwork
  ↓
Q-values
```

但 network 還沒有真正的學習資料來源。

如果 Agent 每跟 environment 互動一次，就只拿最新 transition 訓練，連續 Atari frames 之間高度相關，舊資料也很容易被新的 experience 淹沒。

經典 DQN 的做法是先把 interaction data 保存起來：

```text
(state, action, reward, next_state, terminated, truncated)
```

再從大量過去 experience 中隨機抽出 mini-batch 做訓練。

這個結構就是 **Experience Replay**。

Day 8 已經把：

```text
Breakout state → CNN features → Q-values
```

接完整。

Day 9 要加入的，則是讓這個 network 有一個可以反覆抽樣、反覆學習的 experience memory。

下一篇：[Day 9 — Experience Replay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/main/docs/day09-experience-replay.md)
