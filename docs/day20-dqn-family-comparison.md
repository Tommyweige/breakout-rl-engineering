# Day 20｜DQN、Double DQN、Dueling Double DQN：到底該選誰繼續訓練？

Day 19 做完 Dueling Network 之後，我們手上已經有三個可以正常訓練的 DQN family：

```text
DQN
Double DQN
Dueling Double DQN
```

Day 20 要回答的問題很直接：

> **如果接下來只能挑一個做更長的訓練，該選誰？**

最後答案是 **Dueling Double DQN**。

但這個答案不是因為它某一局拿到最高分，也不是因為 500K 時平均分最高就直接決定。

500K 的三個 training seed 平均分是：

```text
DQN                  15.844
Double DQN           18.156
Dueling Double DQN   20.178
```

看起來 Dueling 已經贏了。

可是把三個 seed 拆開後，seed 33 卻是：

```text
Double DQN           21.200
Dueling Double DQN   20.000
```

也就是說，500K 時還不是每一次訓練都支持同一個方向。

所以我們沒有急著宣布 winner，而是只把前兩名——Double DQN 和 Dueling Double DQN——繼續訓練到 1M。

到了 1M：

```text
Double DQN           19.356
Dueling Double DQN   36.111
```

而且 seed 11、22、33 三組比較全部都是 Dueling 較高。

這時候，我們才把 **Dueling Double DQN** 選成 Day 21 的 Final-Training family。

這篇文章真正想講的是：

> **強化學習的結果很會波動，所以選模型不能只看最高分；要看不同 seed 下，結果是否真的有一致方向。**

---

## 先看整個 Day 20 到底做了什麼

下面這張圖就是 Day 20 最重要的流程。

[![Day 20 從 500K 比較、發現 seed 方向不一致，到 top-2 延長 1M 並選出 Dueling Double DQN 的簡化流程](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/family-selection-flow-simple.svg?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/family-selection-flow-simple.svg)

### 這張圖怎麼看？

只要從上往下看五件事：

1. 三個 family 用同一套規則、同樣的 training seeds `11 / 22 / 33`，先比較到 500K。
2. 500K 時 Dueling 的平均分最高。
3. 但是 seed 33 反而是 Double DQN 比較高，所以證據還不夠一致。
4. 因此只把前兩名 Double + Dueling 延長到 1M。
5. 1M 時三個相同 seed 全部都是 Dueling > Double，才做最後選擇。

這張圖最重要的是中間那一步：

> **500K 平均第一，不代表已經可以直接宣布 winner。**

---

## 公平比較之前，先把其他條件固定

如果 DQN 跑 500K、Double 跑 300K、Dueling 又換另一組 seed，那最後三個分數根本不能直接比。

所以 Day 20 固定主要條件：

```text
環境              ALE/Breakout-v5
環境規則          Contract v2
training seeds    11 / 22 / 33
主要比較 budget   500K actual environment transitions
GPU               NVIDIA CUDA
precision         float32
```

另外三個 family 也共用同一套 replay、learning rate、batch size、epsilon schedule 與 evaluation 規則。

Day 18 已經跑過 DQN 和 Double DQN，所以 Day 20 沒有為了形式重新浪費 GPU 時間，而是先檢查 Day 18 的 Contract、backend、seed、evaluation 與 CUDA 設定是否和 Day 20 相同。

檢查通過後才重用舊 evidence；Dueling Double DQN 則補上新的正式 runs。

這樣做的核心很簡單：

> **不是舊結果不能用，而是只有「真的在同一套規則下產生的舊結果」才能用。**

---

## 三個 family 到底差在哪？

三個模型最後都會輸出四個動作的 Q-value：

```text
NOOP
FIRE
RIGHT
LEFT
```

差別可以先用下面這張表理解：

| Family | 最簡單的理解 |
|---|---|
| DQN | 基本版本，直接學每個 action 的 Q-value |
| Double DQN | 把「下一步選哪個 action」和「這個 action 值多少」拆開，減少 Q-value 高估 |
| Dueling Double DQN | 保留 Double DQN，再把網路最後拆成「局面基礎分數」和「每個 action 的加減分」 |


---

## 為什麼不能只跑一次？

在強化學習裡，即使程式、GPU、learning rate 全都一樣，只要 random seed 不同，結果就可能差很多。

因為 seed 會影響：

- 模型一開始的權重；
- 探索時做出的動作；
- 後面收集到的遊戲資料；
- Replay Buffer 裡最後有哪些經驗。

例如 DQN 在 500K 的三個 training seed：

```text
seed 11 → 14.267
seed 22 → 13.000
seed 33 → 20.267
```

如果只剛好跑到 seed 33，我們很可能會覺得 DQN 比實際上穩定得多。

所以 Day 20 不問：

> 「哪一次最高？」

而是問：

> **換三個 seed 後，這個 family 的結果是不是仍然站得住腳？**

---

## Training Curve：先看模型有沒有真的在學

[![三個 DQN family 在 actual environment transitions 軸上的 raw episode return 與 20-episode rolling mean；每條線保留 training seed](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/dqn-family-training.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/dqn-family-training.png)

這張圖不是拿來直接選 winner，而是先看：

> **模型是不是還在學？還是早就卡住了？**

### 這張圖怎麼看？

**橫軸**是 actual environment transitions，可以直接理解成 Agent 已經和 Breakout 環境互動了多少次。

**上半部 raw episode return** 是每一局實際拿到的 reward。這種線很抖是正常的，因為 RL 每一局都可能差很多。

**下半部 rolling mean** 是最近 20 局的平均，用來把單局波動稍微壓掉，看長期趨勢。

**同一種顏色的不同線**代表不同 training seed。

讀這張圖時，不要找「最高的那一根尖峰」。真正要看的是：

```text
整體趨勢有沒有往上？
三個 seed 是不是都還有學習跡象？
某個結果是不是只出現在單一 seed？
```

而且圖中 DQN 到 500K 就停了，不是因為它後面變成 0，而是因為它沒有進入 top-2 extension。

---

## 500K：Dueling 第一，但還不能直接宣布勝負

500K 時，每一個 checkpoint 都用相同的固定 evaluation 規則測試 15 局。

可以把這想成：

> **訓練過程是平常練習，fixed evaluation 則是大家一起寫同一份考卷。**

結果如下：

| Family | seed 11 | seed 22 | seed 33 | 三個 seed 平均 |
|---|---:|---:|---:|---:|
| DQN | 14.267 | 13.000 | 20.267 | 15.844 |
| Double DQN | 17.000 | 16.267 | 21.200 | 18.156 |
| Dueling Double DQN | 21.133 | 19.400 | 20.000 | 20.178 |

如果只看最後一欄：

```text
Dueling > Double > DQN
```

但把 Dueling 和 Double 用相同 seed 編號比較：

```text
seed 11: +4.133
seed 22: +3.133
seed 33: -1.200
```

前兩組是 Dueling 較高，但 seed 33 是 Double 較高。

這就是為什麼 Day 20 在 500K 沒有直接結束。

這裡的 paired seed 也不要理解成兩個模型會看到完全相同的遊戲過程。模型開始做出不同 action 後，後面的軌跡一定會分開。

它只是避免我們拿：

```text
某個 family 最好的 seed
```

去比較：

```text
另一個 family 最差的 seed
```

因此 500K 最合理的結論是：

> **Dueling 暫時領先，但不同 seed 還沒有完全支持同一個方向。**

---

## Fixed Evaluation 圖：真正拿來比較 checkpoint 的「考試」

[![固定 Contract v2 evaluation 下三個 family、各 training seed 在 100K、250K、500K 和 top-2 的 1M extension 的回報](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/dqn-family-evaluation.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/dqn-family-evaluation.png)

### 這張圖怎麼看？

**橫軸**是訓練進度：100K、250K、500K，以及 top-2 才有的 1M。

**縱軸**是固定 evaluation 的平均 raw return，越高代表同一套測試規則下平均拿到的 reward 越高。

**每個點**是一個 family、某個 training seed、某個 checkpoint 的 15 局平均。

**誤差棒**代表同一個 checkpoint 在那 15 局裡波動有多大。誤差棒長，表示它有時玩很好、有時又比較差。

要特別注意：

> **誤差棒不是三個 training seed 之間的差異。**

另外，DQN 沒有 1M 的點，不代表它 1M 得到 0 分，只代表它沒有進入 extension。

這張圖最值得看的不是某一個最高點，而是 1M 時：

```text
seed 11 → Dueling > Double
seed 22 → Dueling > Double
seed 33 → Dueling > Double
```

這是 500K 還沒有出現的一致方向。

---

## 1M：延長之後，方向終於一致

Double DQN 和 Dueling Double DQN 都從各自的 500K checkpoint 接著訓練。

這裡要說清楚一個限制：

> **1M 不是從 0 一口氣完全不中斷跑到 1M。**

模型會從 500K checkpoint 接續，但 Replay Buffer 沒有完整保存，所以 resume 後需要重新收集一批 experience，讓 Replay Buffer 重新 warm up。

兩個 top-2 family 都使用同樣的 resume 規則，因此仍然可以比較，但不能把它寫成完全不中斷的 1M run。

1M 的固定 evaluation 結果：

| Family | seed 11 | seed 22 | seed 33 | 三個 seed 平均 |
|---|---:|---:|---:|---:|
| Double DQN | 28.533 | 10.067 | 19.467 | 19.356 |
| Dueling Double DQN | 33.533 | 36.867 | 37.933 | 36.111 |

相同 seed 的差值變成：

```text
seed 11: +5.000
seed 22: +26.800
seed 33: +18.467
```

三個全部是正的。

所以 1M 真正帶來的新資訊不是「Dueling 又出現一個高分」，而是：

> **三次不同 training seed，現在全部支持 Dueling 高於 Double。**

這就是 Day 20 最終選 Dueling Double DQN 的主要原因。

---

## Seed Spread：平均分之外，再看換 seed 會不會差很多

[![500K main 的 seed-level return spread；每個點是 training seed，誤差棒是固定 evaluation 的 episode spread](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/dqn-family-seed-spread.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/dqn-family-seed-spread.png)

這張圖回答的是：

> **同一個 family 換一個 random seed，結果會不會差很多？**

### 這張圖怎麼看？

每一個點代表某個 family 在一個 training seed 下的 fixed evaluation 平均。

看點與點之間的距離：

```text
靠得近 → 三次訓練結果比較接近
散得開 → 對 random seed 比較敏感
```

500K 的 seed spread：

```text
DQN       3.169
Double    2.173
Dueling   0.719
```

所以在這三個 seed 裡，Dueling 的結果最集中。

但這裡不要犯兩個錯：

1. 結果比較集中，不等於分數一定比較高。
2. 圖上的誤差棒是同一 checkpoint 的 15 局波動，不是 training-seed spread。

因此這張圖是輔助我們看穩定性，不是單獨決定 winner。

---

## Runtime Cost：Dueling 比較值得繼續，但它不是免費的

[![500K main 的實測 SPS、wall-clock、peak allocated CUDA memory 和 parameter count](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/dqn-family-runtime-cost.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/dqn-family-runtime-cost.png)

這張圖完全不是在看誰玩得比較好，而是在問：

> **為了得到這個模型，要付出多少計算成本？**

### 這張圖怎麼看？

它有四個 panel，每個 panel 的單位不同，所以不要直接比較四張小圖裡「誰的點比較高」。

**SPS**：每秒處理多少 environment transitions，越高代表訓練越快。

**Wall-clock**：完成 500K 實際花多少時間，越低越快。

**Peak allocated CUDA memory**：訓練時實際配置到的最高 GPU 記憶體。

**Parameter count**：模型裡可學習參數的數量。

500K 的主要成本大約是：

```text
Parameters
Standard family        1.69M
Dueling family         3.29M

Mean SPS
DQN                     350.1
Double DQN              345.6
Dueling Double DQN      219.5

Mean wall-clock
DQN                     714.7 s
Double DQN              723.5 s
Dueling Double DQN     1146.3 s
```

所以 Dueling 並不是「又快又強」。

它比較像：

> **這次 quality evidence 更好，但模型更大、訓練也更慢。**

這是一個 trade-off。

---

## 一個不能忽略的限制：Dueling 本身也比較大

Standard DQN / Double DQN 約有：

```text
1.69M parameters
```

Dueling Double DQN 約有：

```text
3.29M parameters
```

所以即使 Day 20 最後 Dueling 分數較高，也不能直接寫成：

> 「已經嚴格證明 Value / Advantage 拆分本身就是全部提升來源。」

因為這次同時發生兩件事：

```text
網路結構改變
+
模型容量增加
```

Day 20 做的是 **model family selection**，不是把兩個網路做成完全相同參數量的 architecture ablation。

所以目前可以說的是：

> **在這套實際採用的三個 family、相同 Contract v2 與相同 training budget 下，Dueling Double DQN 最值得進入下一階段。**

---

## 平均分也不能盲信

1M 時，Dueling seed 11 的 15 局 evaluation 是一個很好的例子：

```text
mean        = 33.533
median      = 18
episode std = 51.779
```

平均分 33.533 看起來很高，但中位數只有 18，而且 15 局之間的波動很大。

這代表少數很高分的 episode 可能把平均值往上拉。

所以 Day 20 最有說服力的證據並不是某個 seed 的平均特別漂亮，而是：

```text
seed 11 → Dueling > Double
seed 22 → Dueling > Double
seed 33 → Dueling > Double
```

三個 training seed 的方向一致。

---

## Day 20 的 evaluation 是「模擬考」，不是最後一次考試

Day 20 使用固定 evaluation seeds：

```text
101 / 202 / 303
```

每組跑 5 episodes，一個 checkpoint 合計 15 局。

但是這些 evaluation 結果已經被拿來：

```text
比較 family
→ 決定 top-2
→ 選最後的 Dueling
```

因此它們比較像機器學習裡的 validation set：**用來做模型選擇的模擬考**。

等 Day 21 完成 Final Long Training 後，最後評估應該換一批沒有參與 Day 20 選擇的新 holdout seeds。

所以 Day 20 選到的是：

> **Final-Training family**

不是：

> **Final model**

---

## Day 20 最後到底得到什麼？

把整篇文章縮成六行：

```text
500K：Dueling 平均第一
↓
但 seed 33 仍是 Double > Dueling
↓
不急著宣布 winner，top-2 繼續到 1M
↓
1M：三個 seed 全部 Dueling > Double
↓
Final-Training family = Dueling Double DQN
↓
Day 21 才開始真正的 Final Long Training
```

但這個結論有清楚的邊界：

- 只有三個 training seeds；
- 每個 checkpoint 的 fixed evaluation 只有 15 局；
- 1M 是從 500K checkpoint resume，Replay Buffer 需要重新 warm up；
- DQN 沒有做 1M extension，所以不是三個 family 的完整 1M tournament；
- Dueling 的參數量接近 Standard 的兩倍；
- Day 20 evaluation seeds 已經參與 family selection，不能再當 final untouched test。

因此 Day 20 沒有得到一個可以直接部署的 `best.pt`。

它得到的是一個更重要的工程決策：

> **接下來，把長訓練的計算資源集中在 Dueling Double DQN。**

完整 machine-readable evidence 保留在 [Day 20 comparison report](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/comparison-report.json) 與 [Day 18 reuse audit](https://github.com/Tommyweige/breakout-rl-engineering/blob/aa2231a2bd81ec903f0aa55f54003b27c1ac32e5/assets/day20/evidence-reuse-audit.json) 中。
