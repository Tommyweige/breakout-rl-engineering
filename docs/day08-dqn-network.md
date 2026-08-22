# Day 8｜從 CNN Features 到四個 Q-values：完成 DQN Network

Day 7 已經把 Breakout 的四張灰階畫面送進 CNN，最後得到長度 3,136 的 feature vector。

Day 8 要補上的，是最後一段：**把這些 features 轉成每個 action 的 Q-value。**

整條資料流會變成：

```text
Breakout state
(4, 84, 84)
      ↓
CNN
      ↓
3,136 features
      ↓
Q head
      ↓
Q(NOOP), Q(FIRE), Q(RIGHT), Q(LEFT)
      ↓
argmax
      ↓
greedy action
```

[![Day 8 DQN forward path](https://github.com/Tommyweige/breakout-rl-engineering/blob/73fbe691af4d7d0a7b00893e700d9f1bbb0c36b5/assets/day08/dqn-forward-flow.svg?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/73fbe691af4d7d0a7b00893e700d9f1bbb0c36b5/assets/day08/dqn-forward-flow.svg)

這張圖最重要的是中間的關係：

```text
state → Q-values → action
```

DQN 不會直接把畫面分類成 `LEFT` 或 `RIGHT`。它先替每個 action 估計一個值，再從這些值中選出 action。

## DQN 輸出的不是 Action Label

如果把 Breakout 當成一般影像分類，很容易直覺地想成：

```text
state → neural network → RIGHT
```

但 Q-Learning 真正想學的是：

```text
Q(state, action)
```

也就是同一個 state 下，每個 action 都有自己的價值估計。

目前 Breakout 有四個 actions：

```text
NOOP
FIRE
RIGHT
LEFT
```

所以 DQN 一次輸出：

```text
[Q(NOOP), Q(FIRE), Q(RIGHT), Q(LEFT)]
```

這其實延續了 Day 6 的 Q-table 概念。

以前可以想成：

```text
Q-table[state]
    ↓
[每個 action 的 Q-value]
```

現在只是把查表換成 neural network：

```text
DQN(state)
    ↓
[每個 action 的 Q-value]
```

**Q-value 的意義沒有改變，改變的是取得它的方法。**

## Output Dimension 跟著 Action Space 走

Breakout 的 action space 目前有四個 action，因此單一 state 最後會得到四個 Q-values。

如果一次送進 32 個 states，輸出就會是：

```text
(32, 4)
```

前面的 `32` 是 batch size，後面的 `4` 才是 action 數量。

如果換成一個有 6 個 actions 的 environment，同一個 DQN 結構最後就應該輸出 6 個值，而不是把 Breakout 的 `4` 寫死在整個模型邏輯裡。

## Q Head：3,136 Features → 4 Q-values

Day 7 的 CNN 已經把畫面轉成：

```text
3,136 features
```

Day 8 在後面接上一個 fully connected head：

```text
3,136 features
      ↓
512 hidden units
      ↓
4 Q-values
```

CNN 前面的工作是利用畫面的空間結構，把 pixel pattern 逐層整理成 features。

到了這裡，模型不再需要保留原本 `84 × 84` 的排列，而是要把整組 features 綜合起來，估計每個 action 的價值。

因此 DQN 可以簡化成兩個角色：

```text
CNN     ：看懂目前的 state
Q head  ：把 state features 轉成 action values
```

這也是 Day 7 和 Day 8 最重要的分界。

## Q-value 不是機率

這是 DQN 很容易和分類模型混淆的地方。

分類模型常看到：

```text
logits → softmax → probabilities
```

例如：

```text
cat  = 0.70
dog  = 0.20
bird = 0.10
```

這些值介於 0 和 1，而且總和等於 1。

Q-value 不是這種機率。

它表示的是：在 state `s` 採取 action `a`，之後按照目前策略繼續走時，預期可以得到多少 discounted return。

所以 Q-values 可以是：

```text
 2.7
-0.4
15.2
 0.0
```

也可能全部是負數。

它們不需要介於 0 和 1，也不需要加總等於 1。

因此 DQN 的輸出保持為 **raw Q-values**，最後不接 Softmax。

如果把 Q-values 強行轉成機率，就會破壞原本的價值尺度，和 Q-Learning 想估計的東西不一樣。

## Argmax：從 Q-values 選出 Greedy Action

假設某個 state 得到：

```text
Q(NOOP)  = 0.12
Q(FIRE)  = 0.08
Q(RIGHT) = 0.31
Q(LEFT)  = 0.17
```

最大的值是 `Q(RIGHT)`，所以 greedy selection 會選 `RIGHT`。

```text
four Q-values
      ↓
    argmax
      ↓
    RIGHT
```

`argmax` 本身不是 neural network 的一層，也不會改變 Q-values；它只是找出最大值所在的位置。

真正值得注意的是：**現在的 DQN 還沒有訓練。**

未訓練模型的 Q-values 只是隨機初始化權重 forward 後產生的數字，因此目前的 argmax 只能確認整條資料流能運作，不能解讀成 Agent 已經知道該怎麼玩 Breakout。

## 用 Q-value 圖看四個 Action 的對應

Day 8 的視覺化會把同一個 Breakout state 經過 DQN 後得到的四個 raw Q-values 畫成柱狀圖，並標示它們分別對應：

```text
NOOP / FIRE / RIGHT / LEFT
```

這張圖要幫助理解的是 **network output 和 action 的一一對應關係**。

在模型還沒有訓練以前，柱子的高低不能拿來判斷策略好壞；它只能顯示目前這次 forward 的數值結果。

產圖程式保留在：

```text
/visualize_dqn_network.py
```

輸出位置為：

```text
/assets/day08/dqn-q-values.png
/assets/day08/dqn-q-values.json
```

圖片必須由真實 Breakout observation 和實際 DQN forward 產生，不使用手寫 Q-values。

## Day 8 完成了哪一塊

走到這裡，從 Atari 畫面到 action value 的路徑已經接起來：

```text
Breakout state
      ↓
CNN feature extractor
      ↓
3,136 features
      ↓
Q head
      ↓
4 Q-values
```

但模型現在仍然只有「怎麼算出 Q-values」的能力，還沒有回答另一個更重要的問題：

**這些 Q-values 要用哪些資料反覆修正？**

經典 DQN 會把過去和 environment 互動得到的 transitions 保存起來，再隨機抽成 mini-batch 訓練。

這個結構就是下一篇的 **Experience Replay**。
