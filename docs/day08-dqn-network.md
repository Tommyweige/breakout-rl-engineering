# Day 8｜DQN 為什麼不直接告訴 Agent「往左」或「往右」？

Day 7 已經把 Breakout 的四張灰階畫面一路送進 CNN，最後得到一個長度 3,136 的 feature vector。

現在真正的問題來了：

> **Agent 最後明明只需要按一個按鍵，為什麼 neural network 不直接輸出 `LEFT` 或 `RIGHT`，而是要輸出四個 Q-values？**

這個問題其實就是 DQN 和一般影像分類最容易被混在一起的地方。

今天要把 Day 7 留下的最後一段補起來：

~~~text
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
~~~

走到這裡，我們第一次擁有一個完整的 **DQN network**。

但要先講清楚一件很重要的事：**network 寫完，不代表 Agent 已經學會玩 Breakout。**

今天的模型還沒有看過任何 training target，也沒有做過一次 optimizer update。現在輸出的四個 Q-values，只是隨機初始化權重經過 forward 後得到的數字。

## 先看完整 network 到底輸出什麼

Day 8 新增的 `inspect_dqn_network.py` 會真的建立 `ALE/Breakout-v5` environment，取得一個 `(4, 84, 84)` observation，再把它送進 DQN：

~~~powershell
python .\inspect_dqn_network.py --device cpu --seed 42
~~~

inspection 會把這條路徑完整列出來：

~~~text
Observation        : (4, 84, 84) uint8
Model input        : (1, 4, 84, 84) torch.float32
Feature shape      : (1, 3136)
Output shape       : (1, 4)
Action meanings    : NOOP FIRE RIGHT LEFT
Q-values           : [四個 raw values]
Greedy action      : argmax 對應的 action
Parameter count    : 1,686,180
state_dict diff    : 0.00000000
~~~

這裡最值得看的不是某一個 Q-value 恰好是多少，而是資料的意義真的改變了：

~~~text
28,224 個 pixel values
        ↓
3,136 個 CNN features
        ↓
4 個 action values
~~~

前兩步在處理「畫面裡有什麼」，最後一步才開始回答「在這個 state 下，每個 action 看起來有多值得做」。

而現在這四個值來自**未訓練的隨機權重**。所以就算其中 `RIGHT` 剛好最大，也不能說模型已經知道要往右。

## 為什麼輸出不是一個 action？

先想像另一種設計：

~~~text
state
  ↓
neural network
  ↓
RIGHT
~~~

看起來很合理，因為 Agent 最後確實只會執行一個 action。

問題是，這樣的輸出只告訴我們「目前選了誰」，卻沒有留下其他 action 的價值資訊。

Q-Learning 真正想學的是：

~~~text
Q(state, action)
~~~

也就是同一個 state 下，每一個 action 都有自己的估計值。

對 Breakout 的四個 action 來說，network 更像是在回答：

~~~text
如果現在 NOOP，長期累積 reward 看起來是多少？
如果現在 FIRE，長期累積 reward 看起來是多少？
如果現在 RIGHT，長期累積 reward 看起來是多少？
如果現在 LEFT，長期累積 reward 看起來是多少？
~~~

因此輸出自然是：

~~~text
[Q(NOOP), Q(FIRE), Q(RIGHT), Q(LEFT)]
~~~

不是一個 action label。

這也讓 Day 6 的 Q-table 和現在的 DQN 接了起來。

以前的小型問題可以想成：

~~~text
Q-table[state]
    ↓
[每個 action 的 Q-value]
~~~

現在只是把「查表」換成：

~~~text
neural network(state)
    ↓
[每個 action 的 Q-value]
~~~

**Q-value 的概念沒有換，換的是得到 Q-value 的方式。**

## 為什麼剛好輸出四個數字？

因為目前 Breakout environment 的 action space 有四個 action：

~~~text
0 → NOOP
1 → FIRE
2 → RIGHT
3 → LEFT
~~~

所以：

~~~text
num_actions = 4
output shape = (B, 4)
~~~

但 `DQNNetwork` 本身沒有去碰 ALE，也沒有把「Breakout 一定是 4」寫死在 model 裡。

建立模型時，action count 由外面傳進去：

~~~python
num_actions = int(env.action_space.n)
model = DQNNetwork(num_actions=num_actions)
~~~

因此 network 的責任很單純：

> 你告訴我有幾個 actions，我就輸出幾個 Q-values。

如果某個別的 environment 有 6 個 actions，同一個 model class 也可以建立成：

~~~text
(B, 6)
~~~

這一點後面做 export 或 inference 也很重要，因為模型本身不需要把整個 Atari environment 帶在身上才能被建立。

## 3,136 個 features 怎麼變成四個 Q-values？

Day 7 的 CNN 最後輸出：

~~~text
(B, 3136)
~~~

這 3,136 個數字已經不再保留原本 `84 × 84` 的畫面排列，而是一組由 CNN 計算出的 features。

接下來使用 fully connected layers：

~~~python
self.q_head = nn.Sequential(
    nn.Linear(self.feature_extractor.feature_dim, 512),
    nn.ReLU(),
    nn.Linear(512, num_actions),
)
~~~

資料流是：

~~~text
3136 features
     ↓
512 hidden activations
     ↓
4 Q-values
~~~

為什麼到這裡適合 fully connected layer？

CNN 前面的工作是利用空間結構，把局部 pixel pattern 慢慢轉成 features。到了 Flatten 之後，我們現在要做的已經不是繼續問「某個 feature 出現在畫面哪個位置」，而是把整組 features 綜合起來，估計每一個 action 的價值。

因此最後這個 Q head 可以把所有 features 一起拿來決定四個輸出。

這裡也可以看到一個有趣的參數量差異。

完整 network 一共有：

~~~text
1,686,180 parameters
~~~

其中光第一個 fully connected layer：

~~~text
3136 × 512 + 512
= 1,606,144 parameters
~~~

已經佔了整個 network 的大部分。

所以「CNN 把畫面變成比較小的 feature vector」不只是 shape 好看而已，它也直接影響後面 fully connected head 的規模。

## Q-value 是機率嗎？

不是。

這是 Day 8 最容易寫錯的一件事。

影像分類常會看到：

~~~text
logits
  ↓
softmax
  ↓
每個 class 的 probability
~~~

例如：

~~~text
cat  = 0.70
dog  = 0.20
bird = 0.10
~~~

這些值被設計成介於 0 和 1，而且總和為 1。

但 Q-value 完全不是這種東西。

Q-value 表示的是：

> 在 state `s` 採取 action `a` 後，按照目前策略繼續下去時，預期可以得到多少 discounted return。

因此它可以是：

~~~text
2.7
-0.4
15.2
0.0
~~~

也可以是其他任意實數。

它們不需要：

~~~text
全部 >= 0
全部 <= 1
總和 = 1
~~~

所以 DQN 的最後一層直接回傳：

~~~python
q_values = self.q_head(features)
~~~

**後面沒有 softmax。**

如果硬套 softmax，會把原本的價值尺度壓成一組彼此競爭、總和等於 1 的機率，Q-Learning 要估計的 quantity 就被改掉了。

## 那 `argmax` 在做什麼？

既然 network 一次會輸出所有 actions 的 Q-values，最直接的 greedy action selection 就是找最大的那一個：

~~~text
Q(NOOP)  = ...
Q(FIRE)  = ...
Q(RIGHT) = ...  ← 最大
Q(LEFT)  = ...
              ↓
           argmax
              ↓
            RIGHT
~~~

`argmax` 本身不會改變 Q-values，也不是另一層 neural network。

它只是問：

> 這四個數字裡，哪一個最大？

因此如果：

~~~python
greedy_action = q_values.argmax(dim=1)
~~~

得到 index 2，就代表目前四個輸出中 `Q(RIGHT)` 最大。

但這裡一定要加上「目前」兩個字。

## 為什麼現在的 argmax 完全不能代表 Agent 會玩？

因為目前 network 的 weights 還只是 PyTorch 初始化出來的值。

它從來沒有被告訴過：

- 哪個 action 後來拿到 reward；
- 哪個 state 很危險；
- 球往哪裡飛時應該移動球拍；
- 某一筆 Q-value 應該往上還是往下修。

所以現在的流程其實是：

~~~text
real Breakout state
        ↓
randomly initialized network
        ↓
four numbers
        ↓
argmax
        ↓
one random-weight preference
~~~

這不是 learned policy。

固定 `torch.manual_seed(42)` 只是讓這組初始 weights 可以重現，方便檢查程式。它不會讓隨機初始化突然變聰明。

這也是為什麼 `inspect_dqn_network.py` 最後會直接提醒：

~~~text
untrained random-weight outputs;
the greedy action is not a learned policy
~~~

## 把四個 Q-values 畫出來，會更容易看出哪裡容易誤解

Day 8 另外新增：

~~~powershell
python .\visualize_dqn_network.py --device cpu --seed 42
~~~

它使用的不是手寫假數字，而是：

~~~text
make_breakout_env()
      ↓
真實 observation
      ↓
DQNNetwork forward
      ↓
四個 raw Q-values
      ↓
產生圖與 JSON metadata
~~~

預設輸出：

~~~text
assets/day08/dqn-q-values.png
assets/day08/dqn-q-values.json
~~~

圖裡會同時放進：

- 真正送進這次 state 的其中一張 Breakout frame；
- `(1, 4, 84, 84) → (1, 3136) → (1, 4)` 的 forward path；
- 四個 action 對應的 raw Q-values；
- 哪一個值被 `argmax` 選到；
- 「未訓練模型」的限制提醒。

這張圖真正要回答的不是「哪個 action 比較好」，而是：

> **DQN 的四個輸出到底和四個 actions 怎麼一一對應？**

只要模型還沒訓練，就不能從柱子的高低推出遊戲策略。

## Batch 進來時，為什麼還是每個 state 四個輸出？

單一 state：

~~~text
(1, 4, 84, 84)
      ↓
DQN
      ↓
(1, 4)
~~~

如果未來 Replay Buffer 一次 sample 32 個 states：

~~~text
(32, 4, 84, 84)
       ↓
DQN
       ↓
(32, 4)
~~~

第一個 `32` 一直都是 batch dimension。

每一列都代表一個 state：

~~~text
state 0 → 4 Q-values
state 1 → 4 Q-values
state 2 → 4 Q-values
...
state 31 → 4 Q-values
~~~

所以 action count 不會因 batch size 增加。

這個 shape 在 Day 12 會非常重要，因為 training 時 network 會一次處理一整個 mini-batch。

## `state_dict` 到底保存了什麼？

現在 network 雖然還沒訓練，但 Day 8 先驗證一件很實際的事情：模型的 parameters 可以被保存，再載入到同樣 architecture 的另一個 model。

PyTorch 最常用的是：

~~~python
torch.save(model.state_dict(), path)
~~~

以及：

~~~python
model.load_state_dict(torch.load(path, ...))
~~~

`state_dict` 可以先理解成：

> **這個 model 裡需要學習、需要保存的參數值。**

它包含 convolution weights、bias，以及 fully connected head 的 weights、bias。

但它不是「把整個 Python project 打包成一個檔」。

要載入它，程式仍然需要先知道要建立什麼 architecture：

~~~python
model = DQNNetwork(num_actions=4)
model.load_state_dict(state_dict)
~~~

Day 8 的 inspection 會做一次 temporary save/load round-trip，然後比較同一個 input 的輸出。

如果保存前後完全相同：

~~~text
state_dict diff = 0.00000000
~~~

代表這個最基本的 serialization contract 是成立的。

真正包含 optimizer、training step、replay 狀態等資訊的 checkpoint，會等到完整 training loop 再處理。

## DQN 已經完整了，為什麼還不能開始正式訓練？

現在我們已經有：

~~~text
state
  ↓
DQNNetwork
  ↓
Q-values
~~~

但 network 還缺最重要的一件事：**用什麼資料反覆修正這些 Q-values？**

如果 Agent 和 environment 互動一次，就只拿最新那一筆 transition 訓練一下，會碰到幾個問題：

- 連續 Atari frames 高度相關；
- 剛發生的資料會一直覆蓋舊經驗；
- 同一筆有價值的 experience 很難被重新利用；
- mini-batch training 也沒有一個可以抽樣的資料來源。

所以 DQN 經典做法不是把 transition 用完就丟掉，而是先把過去的 interaction data 存起來。

資料長得像 Day 3 已經看過的：

~~~text
(state, action, reward, next_state, terminated, truncated)
~~~

把很多筆這種 experience 放進一個可以隨機抽樣的記憶體結構，就是下一步要做的 **Experience Replay**。

Day 8 解決的是：

> **network 怎麼從一個 state 產生所有 actions 的 Q-values？**

Day 9 接著要解決：

> **這個 network 到底要從哪裡取得可以反覆學習的 interaction data？**

下一篇：[Day 9 — Experience Replay](day09-experience-replay.md)
