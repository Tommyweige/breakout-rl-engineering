# Breakout RL Engineering — Day 1

[Day 2 — Atari Breakout、ALE 與 Gymnasium](docs/day02-breakout-ale-gymnasium.md) | [環境設定](docs/environment.md)

---

## 為什麼會有這個系列？

會開始這個專案，其實沒有什麼很複雜的理由。

前陣子我看到 YouTube 頻道 **Yosh** 的影片。

他的影片主要在做一件很有趣的事情：

**用 Reinforcement Learning（強化學習）讓 AI 自己學會玩 Trackmania。**

如果有興趣，可以先看看他的頻道：

> https://www.youtube.com/@yoshtm

我第一次看的時候，最吸引我的並不是「AI 最後可以跑多快」，而是 AI 從一開始幾乎什麼都不會，到慢慢開始出現一些看起來有意義的行為。

一開始可能亂開、撞牆，甚至學到一些我們根本沒預期到的策略。

接著再透過訓練、修改方法、重新測試，慢慢把 Agent 調整得更好。

這種：

```text
什麼都不會
    ↓
亂試
    ↓
得到回饋
    ↓
慢慢學會
    ↓
開始出現有意思的行為
```

的過程，讓我覺得非常有趣。

所以我也產生了一個很直接的想法：

> **那我能不能自己做一個？**

這就是這個系列最一開始的原因。

---

## 不過，我不打算挑戰 Trackmania

Trackmania 很有趣，但如果目標是第一次完整實作一個 Reinforcement Learning 專案，我希望先把問題控制在比較容易處理的範圍。

并且我的運算資源也是非常的有限，只有一個游戲本，12500H + 4060Laptop的組合，所以最後能不能做出成果我也不清楚，但是這也是一個經驗。

所以最後選擇的是另一款非常經典的遊戲：

**Atari Breakout，也就是打磚塊。**

遊戲規則應該不需要太多介紹。

下面有一個可以左右移動的板子，一顆球會在畫面裡反彈，而目標就是不要讓球掉下去，並盡可能打掉上方的磚塊。

這個游戲難度，就畫面上的複雜度而言，我想應該是比Trackmania簡單不少（吧

反正對人來說，這件事情很直覺。

看到球往右下方掉，我們會把板子往右移。

但如果今天控制板子的不是人，而是一個剛開始什麼都不知道的 AI 呢？

它必須透過與遊戲反覆互動，慢慢學會：

```text
現在的狀態
   ↓
做一個動作
   ↓
看看發生什麼事
   ↓
得到 Reward
   ↓
調整之後的行為
```

這就是這次要玩的東西。

---

# 什麼是 Reinforcement Learning？

如果以前沒有接觸過強化學習，現在其實不用急著記任何公式。

先抓住一個概念就好：

**Agent 會在一個 Environment 裡採取 Action，Environment 再把結果和 Reward 回傳給它。**

也就是：

```text
Agent
  │
  │ Action
  ▼
Environment
  │
  │ State + Reward
  ▼
Agent
```

例如放到 Breakout：

```text
AI 看到目前遊戲狀態
        ↓
決定板子往左或往右
        ↓
遊戲繼續執行
        ↓
得到新的狀態與分數
        ↓
再決定下一個動作
```

Agent 就是在大量這樣的互動中逐漸學習。

DeepMind 早期的 DQN 工作，就是讓神經網路直接從 Atari 遊戲畫面學習該採取什麼行動，這也是後來 Deep Reinforcement Learning 很經典的一條發展路線。

這也是為什麼我覺得 Atari 很適合拿來當第一次完整實作 RL 的題目。

---

# 為什麼是 Breakout？

除了遊戲本身簡單之外，我選 Breakout 還有另一個原因：

**它雖然容易理解，但要真的讓 AI 學會，並沒有看起來那麼簡單。**

AI 不會直接收到：

```text
球的位置 = (x, y)
板子的位置 = (x, y)
球的方向 = 右下
```

在這個專案裡，我們會讓它從 Atari 的畫面開始處理。

也就是說，之後還會碰到：

* 遊戲畫面怎麼變成模型輸入？
* 單張圖片看得出球往哪裡飛嗎？
* AI 可以做哪些 Action？
* Reward 到底是什麼？
* 訓練資料從哪裡產生？

這些問題都會在後面的文章一個一個拆開。

所以如果現在完全沒有做過 Reinforcement Learning，其實沒關係。

我自己也希望利用這 30 天，把整條流程真正走過一次，而不是只把某個現成範例跑起來。

---

# 環境的來源

這個系列使用的不是我自己做的 Breakout clone。

預計使用的環境是：

```text
ALE/Breakout-v5
```

主要會接觸兩個東西：

**Gymnasium**

以及

**Arcade Learning Environment（ALE）**

ALE 是一個常被用來研究 Atari 強化學習的環境；原始 DQN 研究同樣使用 Atari 2600 遊戲作為測試環境。

使用既有的標準環境，也代表我們不用先花時間自己處理：

```text
遊戲物理
碰撞
球的速度
板子移動
計分規則
```

可以直接把重點放在：

**怎麼讓 Agent 學習。**

至於 Gymnasium、ALE 和 Atari 三者到底是什麼關係，會留到 Day 2 再慢慢拆。

---

# 這個系列不只是「讓 AI 打贏 Breakout」

最一開始讓我想做這件事的原因，其實很單純：

**我就是覺得看 AI 自己慢慢學會玩遊戲很好玩。**

但既然要花 30 天做，我希望不要只做到：

```text
模型訓練完成
      ↓
AI 會打磚塊
      ↓
結束
```

所以後來我決定把這個專案再往前延伸一些。

除了學 Reinforcement Learning，我也想順便完整走一次模型從開發到部署的流程。

最後大概會變成：

```text
Breakout Environment
        ↓
Reinforcement Learning
        ↓
DQN
        ↓
模型比較
        ↓
Evaluation
        ↓
ONNX
        ↓
Inference
        ↓
Optimization
        ↓
Web Deployment
```

也因此這個專案最後取名為：

# Breakout RL Engineering

前半段的重點是：

**怎麼讓 AI 學會玩 Breakout。**

後半段則是：

**模型學會之後，怎麼把它真正拿出來使用。**

---

# 第一個目標：先做出 DQN

這個系列第一個主要模型會是：

**DQN，Deep Q-Network。**

現在不需要先理解它的公式。

暫時可以把它想成：

> 我們希望訓練一個神經網路，讓它根據目前看到的狀態，判斷不同 Action 哪一個比較值得做。

例如：

```text
目前的 Breakout 畫面
        ↓
       DQN
        ↓
左邊值多少？
右邊值多少？
不動值多少？
        ↓
選擇 Action
```

實際上當然沒有這麼簡單。

後面會陸續碰到：

* Q-Learning
* Neural Network
* Experience Replay
* Exploration
* Target Network

但我不打算在 Day 1 把這些東西全部塞進來。

後面碰到的時候，再從「為什麼需要它」開始介紹。

---

# 做完 DQN 之後呢？

當基本的 DQN 可以訓練之後，我還會繼續嘗試：

```text
DQN
 ↓
Double DQN
 ↓
Dueling Double DQN
```

然後實際跑實驗比較結果。

這部分我覺得也會是整個系列比較有趣的地方。

因為 Reinforcement Learning 並不是：

```text
程式寫完
↓
train()
↓
一定成功
```

實際訓練很可能會遇到很多奇怪的結果。

甚至有 Breakout 實作經驗指出，真正麻煩的部分往往不是把網路寫出來，而是後面的訓練與除錯。

所以如果中間 Agent：

* 完全學不起來
* 表現突然變差
* 學到奇怪的行為
* 實驗結果跟預期不同

我也會把這些過程放進文章。

我反而覺得這些東西比單純貼一張「最後成功了」的結果更值得記錄。

---

# 模型學會玩之後，我還想繼續做下去

到了系列後半段，我會開始離開單純的 RL 演算法。

例如把 PyTorch 訓練好的模型轉成：

**ONNX**

再使用：

**ONNX Runtime**

執行模型。

接著嘗試比較不同推論方式，並看看 FP16、TensorRT 等最佳化方法到底能帶來什麼差異。

最後希望使用：

**ONNX Runtime Web / WebGPU**

讓模型直接在瀏覽器裡執行。

因此理想中的最終成果會是一個可以展示的 Breakout AI，而不只是留下一個模型權重檔。

---

# 這 30 天大概會怎麼走？

目前把整個系列分成五個階段。

## Phase 1 — 先搞懂遊戲和強化學習

Day 1～6 會先處理基礎：

* Atari、ALE、Gymnasium
* State、Action、Reward
* 遊戲畫面怎麼處理
* Q-Learning
* Deep Q-Learning

---

## Phase 2 — 把 DQN 組起來

Day 7～15 開始真正寫模型：

* CNN
* DQN Network
* Experience Replay
* Exploration
* Target Network
* Training Loop
* 訓練與除錯

最後希望得到第一個真的開始學習 Breakout 的 Agent。

---

## Phase 3 — 改進 DQN

Day 16～20 會嘗試：

* Double DQN
* Dueling Network
* 不同 DQN 的實驗比較

看看修改演算法之後，實際結果到底差多少。

---

## Phase 4 — 模型工程

Day 21～25 開始處理：

```text
PyTorch
   ↓
ONNX
   ↓
ONNX Runtime
```

以及推論效能測試。

---

## Phase 5 — 部署

最後 Day 26～30：

* FP16 / TensorRT 實驗
* ONNX Runtime Web
* WebGPU
* Browser Demo
* 最終 Evaluation

如果一切順利，第 30 天應該可以回頭看看：

**一個原本只會亂動的 Agent，最後到底能走到哪裡。**

---

# 這個系列適合誰？

這個系列預計會從 Reinforcement Learning 的基本概念開始。

所以不需要已經有 RL 的實作經驗。

如果本身有一些：

* Python
* 基本的 Machine Learning
* Neural Network

概念，後面的程式應該會比較容易閱讀。

但碰到新的 RL 概念時，我也會盡量先說明：

> **我們現在遇到了什麼問題？**

接著才介紹：

> **為了解決這個問題，所以需要什麼方法？**

而不是一開始就假設所有名詞大家都已經知道。

---

# 專案原始碼

這次的程式碼、實驗紀錄以及相關文件，都會持續放在 GitHub：

https://github.com/Tommyweige/breakout-rl-engineering

Repository 目前還在開發初期，也會跟著這 30 天的文章慢慢長出來。

---

# 下一篇

今天主要先介紹：

**我為什麼突然想讓 AI 玩打磚塊，以及這 30 天到底想做什麼。**

下一篇就正式開始準備我們的遊戲環境。

在寫第一行 DQN 之前，要先搞懂三個之後會一直看到的名字：

**Atari、ALE、Gymnasium。**

它們到底是什麼？

又是怎麼讓 Python 裡的 AI 玩到一款幾十年前的 Atari 遊戲？

Day 2 就從這裡開始。
