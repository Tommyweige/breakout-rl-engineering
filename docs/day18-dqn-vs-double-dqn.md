# Day 18｜DQN vs Double DQN：100K 看得到學習，不代表已經能選模型

Day 17 我們把 DQN 改成 Double DQN。

兩者的 CNN 和模型大小其實一樣，真正不同的是：**計算下一個 state 的 Q-value 時，怎麼決定「哪個 action 最好」以及「它到底值多少」。**

先用一句話回顧：

- **DQN**：直接從下一個 state 的 Q-values 裡取最大值。
- **Double DQN**：先用 online network 選 action，再用 target network 評估那個 action。

Double DQN 想解決的是 DQN 容易把偶然偏高的 Q-value 當真的，久了可能造成 **Q-value 高估（overestimation）**。

但方法聽起來比較合理，不代表實際玩 Breakout 就一定比較強。

所以 Day 18 真正要回答的是：

> **如果 DQN 和 Double DQN 在完全相同的條件下訓練，而且不只跑一次，Double DQN 還會比較好嗎？**

---

## 為什麼不能只跑一次？

強化學習的結果很容易受到隨機性影響。

即使程式、GPU、learning rate 都一樣，只要 random seed 不同：

- 模型一開始的權重會不同；
- 探索時選到的 action 會不同；
- 後面收進 Replay Buffer 的遊戲經驗也會慢慢不同。

所以如果我只跑一次 DQN、一次 Double DQN，最後看到 Double DQN 高 3 分，我還不能確定：

> 是 Double DQN 真的比較好，還是這一次剛好比較幸運？

這也是為什麼 Day 18 不會在 100K 就急著選 winner。

---

## 100K：先確認「有沒有在學」

這裡的 100K，指的是 **100,000 次 actual environment transitions**，也就是 Agent 實際和 Breakout 環境互動了十萬步。

先看訓練曲線：

[![DQN 與 Double DQN 從 100K 訓練到 500K 的每局分數變化](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/dqn-vs-double-training.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/dqn-vs-double-training.png)

### 這張圖怎麼看？

**橫軸**是模型已經和環境互動多少 transitions，越往右代表訓練越久。

**縱軸**是每一局 Breakout 拿到的 return。RL 的單局分數本來就會上下跳很多，所以不要找最高的那個尖峰來判斷誰比較強。

這張圖在 100K 最適合回答的是：

> **DQN 和 Double DQN 有沒有真的開始學？**

答案是有。

但它還不適合回答：

> **DQN 和 Double DQN 到底誰比較強？**

因為訓練時間還短，而且單次 training run 的波動很大。

這是 Day 18 最重要的第一個觀念：

> **看到 learning curve 往上，和已經有足夠證據選模型，是兩回事。**

---

## 我們怎麼把比較做得公平？

如果 DQN 跑 CPU、Double DQN 跑 GPU，或者一邊跑 500K、一邊只跑 300K，那最後的分數就很難比較。

Day 18 因此把主要條件固定：

```text
遊戲環境          ALE/Breakout-v5
遊戲規則          Contract v2
CNN / 模型大小    相同
training backend  相同 CUDA backend
訓練參數          相同
transition budget 相同
正式 seeds        11 / 22 / 33
```

真正刻意改變的主要變因只有：

```text
DQN target rule
vs
Double DQN target rule
```

Contract v2 可以把它想成「統一考試規則」：畫面怎麼處理、frame skip、生命掉了之後怎麼重新 FIRE、一局什麼時候結束、evaluation 怎麼跑，都先固定好。

整個 Day 18 的流程其實不用畫得很複雜：

[![Day 18 簡化流程：固定條件，100K 看是否開始學，250K 拉長觀察，500K 用三組 seed 正式比較，再固定測試 15 局](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/staged-comparison-flow.svg?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/staged-comparison-flow.svg)

### 這張流程圖怎麼看？

只要記住五步：

1. **固定條件**：兩邊玩同一個 Breakout、用同一套 CUDA 設定。
2. **100K**：先確認兩邊都有開始學，不選 winner。
3. **250K**：把訓練拉長，看早期差距是不是很快消失。
4. **500K**：使用 `11 / 22 / 33` 三組 training seeds，正式做 multi-seed comparison。
5. **固定測試 15 局**：最後才用同一套 evaluation 比較模型品質。

所以 Day 18 不是：

```text
100K 誰高 → 誰贏
```

而是：

```text
先確認能學
→ 拉長訓練
→ 換多個 seed 重跑
→ 再用固定測試比較
```

這樣得到的結論才比較可信。

另外，100K → 250K → 500K 之間會從 checkpoint 接著訓練，但 Replay Buffer 不會完整跟著保存，resume 後需要重新累積經驗。DQN 和 Double DQN 都使用相同的 resume 規則，所以相對比較仍然公平；只是不能把它描述成完全不中斷的一條 500K run。

---

## 500K：三組 seed 都是 Double DQN 較高

正式比較時，每一個 500K checkpoint 都用同一套 evaluation 跑 **15 局**，再計算平均分數。

結果如下：

| Training seed | DQN 平均分 | Double DQN 平均分 | 差距 |
|---:|---:|---:|---:|
| 11 | 14.27 | **17.00** | +2.73 |
| 22 | 13.00 | **16.27** | +3.27 |
| 33 | 20.27 | **21.20** | +0.93 |

三組 training seed 都是 Double DQN 較高。

[![500K 時三組 training seed 的 DQN 與 Double DQN 固定測試分數配對比較](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/paired-seed-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/paired-seed-comparison.png)

### 這張圖怎麼看？

每一條線都是同一個 seed 的配對：

```text
seed 11：DQN ↔ Double DQN
seed 22：DQN ↔ Double DQN
seed 33：DQN ↔ Double DQN
```

三條線都往 Double DQN 的方向上升，代表在這三組 seed 裡，Double DQN 的固定 evaluation 平均分都比較高。

這裡的「paired seed」不代表兩個 Agent 會經歷完全一模一樣的遊戲軌跡。模型開始做出不同 action 後，後面的 experience 自然會分岔。

它真正的用途是：**不要拿 DQN 最幸運的一次去對 Double DQN 最倒楣的一次，而是用相同 seed 編號做比較。**

不過 seed 33 的差距只有 `0.93`，也提醒我們不要把結果講成：

> Double DQN 永遠一定比較強。

比較合理的結論是：

> **在目前 Contract v2、CUDA backend 和這三組 500K training seeds 下，Double DQN 的結果方向一致地高於 DQN。**

作為參考，同一套 Contract v2 下完全隨機玩的平均分大約只有 `1.73`。所以這時兩種模型都已經明顯不是 random policy；Day 18 比較的是兩個已經學到東西的 Agent 之間的差異。

---

## Q-value 真的有比較不容易估太高嗎？

Day 17 留下了 60 個固定的 Breakout states。

Day 18 把完全相同的 60 個 states 丟給六個 500K 模型，觀察它們對「最佳 action」的最高 Q-value 估計。

DQN 三組 seed 的平均大約是：

```text
2.69 / 2.75 / 2.68
```

Double DQN 則大約是：

```text
2.01 / 2.52 / 2.52
```

[![同一批固定 Breakout states 下，DQN 與 Double DQN 的 Q-value 估計比較](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/q-probe-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/q-probe-comparison.png)

### 這張圖可以告訴我們什麼？

同一批畫面下，Double DQN 的最高 Q-value 在三組 seed 都比較低。

這和 Double DQN 的設計動機方向一致：它想減少 DQN 因為 `max` 操作而把偶然偏高的估計持續放大的問題。

但一定要注意：

> **Q-value 比較低，不等於一定比較準，也不等於模型一定比較強。**

我們不知道這 60 個 states 的「真正 Q-value」是多少，因此 Q-probe 只能用來幫助理解模型內部估值，不能拿來當 winner metric。

真正判斷遊戲能力，還是以前面的固定 evaluation 為主。

---

## Double DQN 的代價大嗎？

Double DQN 在計算 target 時需要額外用 online network 選下一個 action，因此會多一些計算。

500K 階段的實際平均成本是：

| 方法 | 每秒 environment transitions | 跑完這 250K 新增 transitions | Peak GPU memory |
|---|---:|---:|---:|
| DQN | 350.09 | 714.7 秒 | 約 609 MiB |
| Double DQN | 345.57 | 723.5 秒 | 約 609 MiB |

[![500K 階段 DQN 與 Double DQN 的速度、時間與 GPU 記憶體比較](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/runtime-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/1bca48867e9ef2fb1396e84ff5f3e78439b4806b/assets/day18/runtime-comparison.png)

### 這張圖怎麼看？

- **transitions/s 越高**：同一台機器每秒能處理更多環境互動。
- **wall-clock 越低**：跑完相同工作量花的真實時間越少。
- **GPU memory**：看兩種方法是否需要明顯不同的顯存。

這批實驗裡，Double DQN 大約只慢 `1%` 左右，而且 GPU memory 幾乎一樣。

所以至少在目前這套 RTX 4060 Laptop GPU 訓練環境中，Double DQN 並不是用巨大的工程成本換取較好的 500K evaluation 結果。

但這張圖只是在看**成本**，不是在看模型品質。跑得快不代表玩得好。

---

## Day 18 真正得到的答案

把整天濃縮成最簡單的版本：

```text
100K
→ DQN / Double DQN 都開始學
→ 還不能選模型

250K
→ 把訓練拉長
→ 繼續觀察

500K + seeds 11 / 22 / 33
→ 三組都做固定 15 局 evaluation
→ 三組都是 Double DQN 平均分較高
```

因此 Day 18 可以說：

> **在目前這套 Breakout Contract v2、CUDA 訓練設定，以及三組 500K training seeds 下，Double DQN 比 DQN 得到更一致的較高 evaluation mean。**

同時，固定 Q-probe 也看到 Double DQN 的 max-Q 估計較低，和「減少過度高估」的設計方向一致。

但 Day 18 **不能**說：

> Double DQN 在所有 Atari、所有 seed、所有超參數下都一定比較強。

因為我們目前只有三組 training seeds，而且後面還要加入 Dueling Network 一起比較。

Day 18 真正建立的，不只是一個 Double DQN 結果，而是一個很重要的 RL 實驗習慣：

> **不要看到一條漂亮的 learning curve 就宣布 winner。先讓模型訓練到足夠的 budget，再換幾個 random seeds，最後用相同的 evaluation 規則比較。**

下一步會加入 Dueling Network，接著再把 DQN family 放到同一套條件下做更完整的比較。
