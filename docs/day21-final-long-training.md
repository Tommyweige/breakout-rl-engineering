# Day 21｜訓練越久越好嗎？跑到 5M 後，我最後反而選了 2.5M 模型

Day 20 我們已經從 DQN、Double DQN、Dueling Double DQN 三種方案裡，選出了 **Dueling Double DQN**。

但「選到比較好的模型架構」和「得到最後真正要用的模型」，其實是兩件不同的事。

Day 21 要回答的是更實際的問題：

> **同一種模型重新訓練幾次之後，到底要訓練多久？最後又該留下哪一個版本？**

這次結果剛好打破一個很直覺的想法：

```text
訓練越久
≠
模型一定越強
```

同一個 `seed 2022`：

```text
1M   → 34.9
2.5M → 51.4
5M   → 49.9
```

模型從 1M 到 2.5M 明顯進步，但繼續跑到 5M，固定測試的平均分沒有再提高。

所以最後我沒有因為「5M 是訓練最久的版本」就直接選它，而是回頭保留 **2.5M checkpoint**。

更有意思的是，這個 2.5M 模型在挑選階段的平均分是 **51.4**，等到模型確定、不再更換之後，再用一組從來沒參與過挑選的測試條件評估，平均分只有 **30.9**。

因此 Day 21 最值得帶走的是兩件事：

1. **訓練更久，不代表一定更好。**
2. **拿來挑模型的成績，不等於真正面對新測試時的成績。**

---

## 先看它真的在玩 Breakout

前面幾天大多在看 training curve、Q-value 和 evaluation score，但到了這裡，我們其實已經有一個真的會在 Atari Breakout 裡做決策的 Agent。

下面這段就是專案實際跑出的 Breakout gameplay：

[![Breakout RL 實際遊戲畫面](https://img.youtube.com/vi/tzESnOS-8qU/hqdefault.jpg)](https://youtu.be/tzESnOS-8qU)

[點這裡觀看實際 Breakout 演示](https://youtu.be/tzESnOS-8qU)

影片的用途很單純：讓我們直接看到 Agent 到底在做什麼。

它是不是會追著球移動？能不能把球接回去？能不能讓一局維持得更久？這些事情從影片裡會比只看一條分數曲線更有感。

不過影片不能取代正式 evaluation。某一局打得特別漂亮，可能只是剛好遇到比較有利的遊戲過程，所以後面的模型比較仍然要回到固定條件下的多局測試。

---

## 為什麼 Day 21 要重新訓練三次？

Day 20 已經告訴我們 Dueling Double DQN 在目前三種方案裡最值得繼續。

但如果 Day 21 只把 Day 20 最好的那次訓練接著跑，我們其實只知道：

> 那一次訓練最後表現不錯。

我們還不知道換一個隨機起點之後，它是不是一樣能學起來。

所以這次用三個新的 training seeds 從頭開始：

```text
1011
2022
3033
```

可以把 training seed 想成「同一套訓練方法的不同開局」。

模型架構、learning rate、遊戲規則都一樣，但初始權重、早期探索路線，以及最先遇到的遊戲狀態可能不同。

如果只跑一次，很容易出現：

```text
剛好這次跑得很好
↓
誤以為整套方法都很穩
```

因此 Day 21 第一個問題不是「哪個 seed 分數最高」，而是：

> **換三個新的開局之後，這個模型還能不能再次學起來？**

---

## 不需要三個模型全部一路跑到 5M

強化學習的長時間訓練很花時間，所以這次採用逐步縮小候選的方式。

這段決策的順序與分支如下。實線代表實際進入下一階段的路徑；虛線代表保留 2.5M checkpoint，最後拿來和 5M 比較。

[![Day 21 從三個 fresh seed 經過 1M、2.5M、5M 比較，到 2.5M freeze 與 final holdout 的訓練決策流程](https://github.com/Tommyweige/breakout-rl-engineering/blob/dd8085e7f3f855cc22acd3cf90516a732dd85e3a/assets/day21/day21-main-flow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/dd8085e7f3f855cc22acd3cf90516a732dd85e3a/assets/day21/day21-main-flow.png)

這張圖回答的是「哪些候選值得繼續投入計算」，不是直接替每個 checkpoint 打分；真正的分數仍要看後面的 fixed evaluation 圖。它也清楚標出最後的 holdout 必須等模型 freeze 後才開啟。

這裡的 `1M / 2.5M / 5M` 都是 environment transitions，也就是 Agent 真正和遊戲環境互動的次數。

這種做法的目的很實際：先用比較便宜的階段確認哪些候選值得繼續，再把昂貴的長跑資源集中到它們身上。

---

## 1M：三次重新訓練，結果其實很接近

三個 seed 到 1M 後，用同一套固定條件各測 15 局：

| Training seed | 15 局平均分 | 中位數 |
|---:|---:|---:|
| 1011 | 34.000 | 31.000 |
| 2022 | **34.867** | 30.000 |
| 3033 | 33.133 | 29.000 |

最值得看的不是 `2022` 暫時第一，而是：

```text
34.0
34.9
33.1
```

三個結果非常接近。

這表示換了三個新的訓練開局之後，Dueling Double DQN 都有重新學起來。至少到 1M 為止，沒有看到「只有某一個 lucky seed 才能成功」的跡象。

接下來留下 `2022` 和 `1011` 繼續訓練到 2.5M。

`3033` 不是訓練失敗，只是在計算資源有限的情況下，沒有必要讓全部候選都繼續往後跑。

---

## 2.5M：兩個候選開始拉開差距

到了 2.5M：

| Training seed | 1M | 2.5M |
|---:|---:|---:|
| 1011 | 34.000 | 42.600 |
| 2022 | 34.867 | **51.400** |

兩個模型都比 1M 時進步，但 `2022` 的提升特別明顯：

```text
34.9
↓
51.4
```

所以這時候有充分理由把 `2022` 再往後訓練，看看它是不是還有成長空間。

這一點很重要：我們不是因為 roadmap 上寫了 5M，就認定一定要把模型硬跑到 5M；而是因為 1M → 2.5M 的確還在改善，所以繼續觀察有意義。

---

## 5M：更多訓練，沒有換來更高分

`seed 2022` 繼續跑到 5M 後：

```text
2.5M → 51.400
5M   → 49.933
```

5M 沒有訓練壞掉，但也沒有超過 2.5M。

這裡不要直接把 `51.4 → 49.9` 解讀成「模型一定退步了」。兩次 evaluation 都有波動，而且差距並不算大。

真正可以說的是：

> **目前沒有證據支持「多訓練到 5M，模型就會更好」。**

這跟一般 supervised learning 很像：epoch 更多、steps 更多，都只是你投入了更多計算，不是品質保證。

[![固定評估規則下，各 milestone checkpoint 的 15 局 evaluation 結果](https://github.com/Tommyweige/breakout-rl-engineering/blob/dd8085e7f3f855cc22acd3cf90516a732dd85e3a/assets/day21/milestone-evaluation.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/dd8085e7f3f855cc22acd3cf90516a732dd85e3a/assets/day21/milestone-evaluation.png)

### 這張圖怎麼看？

**橫軸**是訓練量：1M、2.5M、5M。

**縱軸**是每個 checkpoint 固定測試 15 局之後的平均分。

**每個點**代表一個 training seed 在那個時間點的模型。

**誤差棒**表示同一個模型跑 15 局時，分數本身有多大的波動。

其中最值得追的是 `seed 2022`：

```text
1M      34.9
2.5M    51.4
5M      49.9
```

前半段有明顯成長，後半段則沒有再拉開新的差距。

所以最後不能只問「哪個 checkpoint 比較晚」，而是要問：

> **繼續訓練之後，真的有變得更好嗎？**

---

## Training curve 很熱鬧，但不要拿最高尖峰選模型

強化學習的 training return 通常非常抖。

有時 Agent 某一局剛好打得特別順，分數突然衝高；下一局又可能掉回來。

[![三個 fresh seeds 的真實 training return 與 20-episode rolling mean](https://github.com/Tommyweige/breakout-rl-engineering/blob/dd8085e7f3f855cc22acd3cf90516a732dd85e3a/assets/day21/long-training-return.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/dd8085e7f3f855cc22acd3cf90516a732dd85e3a/assets/day21/long-training-return.png)

### 這張圖怎麼看？

上半部是每一局真正拿到的分數，因此會看到很多尖峰和掉落。

下半部是最近 20 局的平均，可以稍微壓掉短期雜訊，比較容易看出長期方向。

這張圖主要用來回答：

- 模型到底有沒有在學？
- 訓練有沒有突然崩掉？
- 後期還有沒有明顯向上的趨勢？

但它不適合直接拿來挑「最好的 checkpoint」。

某一局突然打出超高分，只代表那一局很好，不代表當下模型平均而言就是最強。

因此最後比較 checkpoint 時，還是回到固定條件下重新跑的多局 evaluation。

---

## 最後為什麼選 2.5M，而不是 5M？

答案其實很單純。

在同一套評估方式下：

| 模型 | 15 局平均分 |
|---|---:|
| seed 2022 / 2.5M | **51.400** |
| seed 2022 / 5M | 49.933 |

5M 沒有提供更好的結果，所以最後保留 2.5M 版本。

這裡真正重要的是一個很實用的模型訓練原則：

> **停止訓練的理由，應該來自模型表現，而不是單純因為預定步數還沒跑完。**

如果 5M 明顯更好，當然應該選 5M；但這次沒有，因此沒有必要因為它比較晚就硬選它。

---

## 51.4 還不能直接叫做「最後成績」

到這裡很容易犯下一個錯誤：

> 「既然 2.5M 的平均是 51.4，那最後模型的成績就是 51.4。」

其實不能這樣說。

因為這組 evaluation 已經被我們拿來：

```text
比較 1M
比較 2.5M
決定誰進 5M
最後挑 checkpoint
```

也就是說，這批測試結果已經參與模型選擇。

它比較像 **模擬考**，而不是最後一次真正沒看過的考試。

所以模型確定、不再更換之後，才另外使用一組沒有參與過挑選的測試條件。

結果是：

| 評估階段 | 平均分 | 中位數 |
|---|---:|---:|
| 挑模型時使用的 evaluation | **51.400** | 51.000 |
| 最後才看的 holdout | **30.933** | 32.000 |

這個落差非常值得注意。

```text
模擬考：51.4
真正沒看過的新題目：30.9
```

但也不要急著說模型 overfit。

目前最後測試只有 15 局，而 Breakout 本身就有相當大的隨機波動，不同 seed 也會讓遊戲過程走向不同方向。

真正可以確定的是：

> **51.4 不能直接被當成模型面對所有新情況時的平均能力。**

[![最後選定模型在未參與挑選的 holdout seeds 上的 raw return](https://github.com/Tommyweige/breakout-rl-engineering/blob/dd8085e7f3f855cc22acd3cf90516a732dd85e3a/assets/day21/final-holdout.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/dd8085e7f3f855cc22acd3cf90516a732dd85e3a/assets/day21/final-holdout.png)

### 這張圖怎麼看？

每個點都是最後選定模型在一局新測試裡真正拿到的 raw return。

不要找最高分，而是看整批點大致落在哪裡。

如果換成沒有參與 model selection 的 seeds 後，整體分數明顯比較低，就表示原本的 51.4 不能單獨代表模型的泛化能力。

---

## Day 21 真正學到的是什麼？

如果只看工作量，Day 21 好像只是「把模型一路訓練到 5M」。

但真正重要的是兩個觀念。

第一個：

```text
2.5M → 51.4
5M   → 49.9
```

**更多訓練，不保證更好的模型。**

第二個：

```text
挑模型時 → 51.4
新測試   → 30.9
```

**模型選擇時看到的好成績，不等於真正面對新資料時的成績。**

而實際 gameplay 則補上了數字看不到的那一面：這些分數背後，真的有一個 Agent 在畫面裡根據球的位置移動球拍、嘗試把球接回去、延長每一局遊戲。

到了這裡，我們已經不只是「成功把 DQN 跑起來」，而是有了一個經過長時間訓練與獨立測試的 Breakout Agent。

下一步才會開始處理另一個問題：**怎麼把這個 PyTorch 模型帶離訓練環境，變成可以在其他 runtime 執行的模型。**
