# Day 17｜DQN 為什麼會把自己看得太樂觀？從 `max` 的陷阱到 Double DQN

Day 16 解決的是「怎麼訓練得更有效率」。環境規則固定下來後，我們也選出了後續實驗要使用的向量化訓練方式。

但訓練流程穩定，不代表演算法本身就沒有問題。

DQN 每次看到一個狀態，都會替每個 action 估計一個 Q-value。這個數字可以先理解成：**如果現在做這個動作，從現在開始大概能累積多少未來回報。**

問題是，Q-value 不是遊戲偷偷告訴模型的正確答案，而是神經網路自己估出來的。

只要是估計，就會有誤差。

而 Vanilla DQN 剛好有一個很容易把誤差放大的動作：

> **從好幾個估計值裡，直接挑最大的那一個。**

這一篇要處理的，就是這個看起來很自然的 `max`。

---

## 當四個答案都有誤差，挑最大的那個會發生什麼？

先不要急著回到 Breakout。

假設現在有四個 action，而且我們知道它們真正的價值其實全部都是：

```text
0, 0, 0, 0
```

如果估計器完全準確，那當然沒問題。

但現實中的神經網路不可能每次都剛好估到 0。可能某次變成：

```text
-0.2, 0.1, -0.1, 0.3
```

下一次又變成：

```text
0.1, -0.4, 0.8, -0.2
```

這些誤差如果平均起來接近 0，看起來好像很公平：有時高估、有時低估，最後應該互相抵銷。

但 DQN 並不是把四個估計值全部平均。

它會做：

```text
max(Q1, Q2, Q3, Q4)
```

也就是每次專門挑最大的那一個。

這就像四個人同時猜一個答案，每個人的猜測都有正有負的誤差，但我們永遠只採用「猜得最高」的人。久了以後，即使每個人本身都沒有刻意往高處猜，最後被選中的答案仍然很容易偏高。

我用一個最小實驗把這件事直接量出來。四個 action 的真實價值都固定為 0，只改估計誤差的大小；每個設定跑 100,000 次。

| noise std | 所有估計平均 | 每次取 `max` 後平均 | 分開選擇與評估後平均 |
|---:|---:|---:|---:|
| 0.1 | 0.000003 | 0.102909 | 0.000550 |
| 0.5 | 0.000015 | 0.514543 | 0.002750 |
| 1.0 | 0.000029 | 1.029086 | 0.005499 |

可以看到，原本所有估計混在一起時幾乎就是 0；但只要每次都挑最大的那個，平均值就開始往上飄，而且誤差越大，偏得越明顯。

[![四個 action 的估計加入不同 noise 後，max selection 的平均值偏離真實值](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/overestimation-bias.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/overestimation-bias.png)

這就是 **overestimation bias（高估偏差）**。

它不是說「DQN 的每一個 Q-value 都一定高估」，而是：

> 當同一組帶有誤差的估計，同時負責「挑出最好的一個」和「告訴我們它有多好」時，最大值會比較容易保留下剛好偏高的誤差。

這個差別非常重要。

---

## `max` 本身不是錯，問題是它相信了同一份估計

假設某個 next state 的四個 Q-value 是：

```text
NOOP   = -0.8
FIRE   =  0.2
RIGHT  = -0.1
LEFT   =  0.4
```

DQN 當然會選 LEFT，因為 0.4 最大。

另一個 state 可能是：

```text
NOOP   = -0.2
FIRE   =  0.1
RIGHT  =  1.1
LEFT   =  0.3
```

這次就會選 RIGHT。

如果 1.1 真的是 RIGHT 的價值，那完全沒問題。

麻煩的是，我們不知道那個 1.1 到底是「真的比較好」，還是神經網路這次剛好把 RIGHT 高估了。

DQN 又沒有另一份 ground truth 可以馬上核對。

於是，一個偶然偏高的估計可能被選成最佳 action，接著再被放進下一次學習的 target。模型就有機會用自己的高估，繼續教下一個自己。

---

## Vanilla DQN 的 target 為什麼會碰到這個問題？

DQN 不只要估計「目前 state 做了某個 action 的價值」，它還要替這次預測建立一個學習目標。

如果這一步得到的 reward 是 `r`，折扣率是 `γ`，下一個 state 是 `s'`，Vanilla DQN 的核心概念可以寫成：

```text
target
= r + γ × max Q_target(s', a)
```

如果 episode 真正結束，就不再加後面的未來價值。

這裡的 target network，是前面已經做過的那份「更新比較慢的網路」。它的作用是讓學習目標不要每一次 optimizer update 都跟著劇烈變動。

可是，即使有 target network，Vanilla DQN 在 `next_state` 上還是做了同一件事：

```text
Q_target(s', NOOP)
Q_target(s', FIRE)
Q_target(s', RIGHT)
Q_target(s', LEFT)
          ↓
         max
```

也就是說，同一組 Q-values 同時決定：

```text
哪個 action 最好？
```

以及：

```text
那個 action 到底值多少？
```

如果最大的那個值只是剛好被高估，兩個步驟就會一起相信它。

這才是 Double DQN 想拆開的地方。

---

## Double DQN 沒有多做一套模型，它只是把兩個工作分開

名字叫 Double DQN，很容易讓人第一眼以為：是不是又多訓練了一個完全獨立的 DQN？

其實不是。

Vanilla DQN 本來就有：

```text
online network
+
target network
```

Double DQN 沒有把 action space 改掉，也沒有改 observation，更沒有增加另一套遊戲環境。

它只是重新分配這兩個網路在 `next_state` 上的工作。

[![DQN 與 Double DQN 的核心差異：DQN 直接對 Target Network 取 max；Double DQN 由 Online Network 選 action、Target Network 評估](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/916f05fc17cb0d80e29af16008829d1d346d92c3/assets/day17/dqn-vs-double-dqn-core.svg?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/916f05fc17cb0d80e29af16008829d1d346d92c3/assets/day17/dqn-vs-double-dqn-core.svg)

這張圖可以直接左右對照著看。

左邊的 DQN 只有一條路：`next state → Target Network → 直接取 max`。也就是 Target Network 自己先決定「哪個 action 最大」，又直接把這個最大值當成未來價值。如果那個最大值只是剛好被高估，學習 target 就可能跟著被拉高。

右邊的 Double DQN 則把這件事拆成兩步：先由 Online Network 做 `argmax`，只決定要選哪個 action；接著 Target Network 不再自己重新挑最大值，而是只讀取剛剛那個 action 的 Q-value。公式雖然看起來比較長，但真正的核心只有一句：

> **Online 負責「選」，Target 負責「評」。**

這也是 Double DQN 的主要優勢：它不是保證所有 Q-value 都不會高估，而是降低「同一份估計又選又評」時，剛好偏高的最大值被一路放大的機會。

---

## 用一組數字看，差異會非常直觀

假設同一個 next state 上，online network 認為：

```text
[1, 5, 2, 0]
```

它會選 index 1，因為 5 最大。

但 target network 對同一個 state 的估計是：

```text
[4, 3, 2, 1]
```

如果是 Vanilla DQN，它完全不管 online network 怎麼想，直接取 target network 自己的最大值：

```text
max([4, 3, 2, 1]) = 4
```

假設：

```text
reward = 1
gamma  = 0.5
```

那 target 就是：

```text
1 + 0.5 × 4 = 3.0
```

Double DQN 則不一樣。

online network 已經選出 index 1，所以 target network 只去看 index 1 的值：

```text
Q_target(index 1) = 3
```

最後變成：

```text
1 + 0.5 × 3 = 2.5
```

[![同一組 next-state Q-values 下 Vanilla 與 Double DQN 的 target 差異](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/dqn-vs-double-targets.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/dqn-vs-double-targets.png)

這個例子不代表 Double DQN 的 target 永遠比較小，也不代表越小越正確。

它只是很清楚地展示：**兩個演算法真正不同的地方，就是下一個 state 的 action 怎麼選、value 又由誰來讀。**

---

## 回到 Breakout：我們看得到模型的估計，但看不到真正答案

前面的 toy experiment 有一個現實裡不可能得到的優勢：我們事先知道四個 action 的真實價值都是 0。

所以可以直接量「高估了多少」。

Breakout 沒有這麼方便。

對一張真實遊戲畫面，我們沒有一個資料表可以告訴我們：

```text
NOOP  真實 Q-value = ?
FIRE  真實 Q-value = ?
RIGHT 真實 Q-value = ?
LEFT  真實 Q-value = ?
```

因此不能看到模型輸出 0.3，就說「它高估了 0.1」。

能做的是固定一批相同的遊戲狀態，之後反覆把不同 checkpoint 餵進去，看 Q-value 分布和 action preference 怎麼變。

Day 17 固定了 60 個 Breakout states。對這次 Double DQN 的短程 checkpoint 做推論後，greedy action 分布是：

```text
RIGHT = 41
FIRE  = 18
LEFT  = 1
NOOP  = 0
```

[![Day 17 smoke checkpoint 在固定 Breakout states 上的 Q-value 分布與 greedy action](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/q-probe-summary.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/q-probe-summary.png)

這張圖能回答的是：

```text
這個 checkpoint 面對同一批 states 時，
各 action 的估計分布長什麼樣？
它比較常把哪個 action 放在最高？
```

但它不能回答：

```text
RIGHT 真的是最佳 action 嗎？
這個 Q-value 距離真實值差多少？
Double DQN 已經比 Vanilla DQN 強了嗎？
```

這個界線很重要。

固定 probe 的價值，是讓之後的模型有一把共同的尺，而不是假裝我們突然知道 Breakout 的真正 Q-function。

---

## Double DQN 的代價：多一次 forward，但幅度不大

Double DQN 在計算 next-state target 時，需要：

```text
online(next_state)
+
target(next_state)
```

Vanilla DQN 則主要只需要 target network 的 next-state forward。

所以 Double DQN 理論上就會多一點計算成本。

在同一台 RTX 4060 Laptop GPU、相同訓練條件下，各跑 10,000 transitions，實際量到：

| | Vanilla DQN | Double DQN |
|---|---:|---:|
| transitions/s | 248.62 | 237.81 |
| optimizer updates | 2,251 | 2,251 |
| next-state target forward GPU time | 2.48 s | 4.25 s |
| peak VRAM | 639 MB | 639 MB |

[![相同訓練條件下 Vanilla DQN 與 Double DQN 的短程執行成本](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/smoke-performance.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/smoke-performance.png)

Double DQN 的整體吞吐大約低 4.4%。這和它多做一次 next-state online forward 的預期一致。

但這個 10K 實驗只有一個目的：確認 Double DQN 的計算路徑真的可以接進現有訓練流程，而且成本沒有突然失控。

它完全不能拿來回答：

> **Double DQN 到底有沒有學得比較好？**

10,000 transitions 對這個問題還太短，而且只有單一 training seed。

---

## 所以 Double DQN 解決了 overestimation 嗎？

比較精確的說法是：

> Double DQN **降低了讓同一份估計同時負責選擇與評估所造成的高估傾向**。

但不能把它講成：

> Double DQN 讓所有 Q-value 從此都變成正確答案。

真實訓練中的 online network 和 target network 並不是兩個完全獨立的估計器，它們來自同一條 training history，而且 target network 還會定期從 online network 同步權重。

所以兩邊的誤差仍然可能相關。

除此之外，Breakout 的學習結果還會受到探索、Replay 裡的資料分布、optimizer、訓練長度和 random seed 影響。

Double DQN 只處理其中一個很具體的問題：

```text
同一個 noisy estimator
同時選 action
又評估這個 action
```

把這兩件事拆開。

這也是我覺得它很適合接在 Vanilla DQN 後面學的原因：改動不大，但它直接指出 DQN target 裡一個很容易忽略的統計問題。

---

## Day 17 到這裡，真正得到的是一個可以被公平比較的假設

現在我們已經知道：

```text
Vanilla DQN
    ↓
同一組 next-state estimates
    ↓
直接 max
    ↓
可能偏愛剛好被高估的 action
```

而 Double DQN 改成：

```text
online network 選 action
        ↓
target network 評估 action
```

toy experiment 告訴我們這個設計在統計上有理由；真實 Breakout smoke 則確認它可以正常跑，而且額外計算成本目前大約只有幾個百分點。

可是「設計有道理」和「在 Breakout 上真的比較好」仍然是兩件事。

因此 Day 18 才會進入真正的比較：在相同的環境規則、相同訓練 backend、相同 budget 和多個 training seeds 下，讓 Vanilla DQN 和 Double DQN 正面跑一次。

到那時候才有資格回答：

> **把選擇與評估拆開之後，這個理論上的修正，在真正的 Breakout 訓練裡到底值不值得。**