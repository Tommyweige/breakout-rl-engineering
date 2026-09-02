# Day 19｜Dueling Network：先判斷「局面好不好」，再看「哪個動作比較好」

前面的 DQN 每看到一個 Breakout 畫面，都會直接估計四個動作的 Q-value：

```text
NOOP   → 這時候不動，大概有多好？
FIRE   → 這時候按 FIRE，大概有多好？
RIGHT  → 這時候往右，大概有多好？
LEFT   → 這時候往左，大概有多好？
```

最後挑 Q-value 最大的動作去做。

這個方法沒有錯，但有一個值得思考的地方。

假設球現在還在畫面上方，離球拍很遠。這一瞬間往左、往右，甚至暫時不動，可能都不會立刻造成很大的差別。

這時候，比起急著問：

> 「四個動作分別值多少？」

其實可以先問：

> **「現在這個局面本身好不好？」**

例如球的位置、移動方向、球拍位置、剩下多少磚塊，這些資訊其實是四個動作共同面對的。

這就是 **Dueling Network** 想做的事情：

> **先替整個局面打一個基礎分數，再看每個動作應該加分還是扣分。**

---

## Standard DQN 和 Dueling 到底差在哪？

Standard DQN 的最後一段很直接：

```text
遊戲畫面
   ↓
CNN 抽出特徵
   ↓
直接輸出 4 個 Q-values
```

Dueling 則把最後一段拆成兩條：

```text
遊戲畫面
   ↓
CNN 抽出特徵
   ↓
   ├─ Value：這個局面整體好不好？
   │
   └─ Advantage：每個動作相對來說好多少、差多少？
   ↓
重新合成 4 個 Q-values
```

可以先把它想成：

```text
Q-value
=
局面的基礎分數
+
這個動作的加分 / 扣分
```

這裡的「局面基礎分數」就是 **Value stream**，通常寫成 `V(s)`。

「每個動作的加分或扣分」則來自 **Advantage stream**，通常寫成 `A(s,a)`。

Day 19 實際使用的網路就是這個結構：

[![Dueling Network 從共享 Atari features 分成 Value 與 Advantage，再重建四個 Q-values 的結構圖](https://github.com/Tommyweige/breakout-rl-engineering/blob/5877108fc34b9e69b2ba835e9932886d1314a2a2/assets/day19/dueling-network-architecture.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/5877108fc34b9e69b2ba835e9932886d1314a2a2/assets/day19/dueling-network-architecture.png)

前面的 CNN 沒有換掉，輸入也還是四張 `84 × 84` 畫面堆在一起。

真正改變的只有 CNN 後面的輸出方式。

Standard DQN 是：

```text
CNN → 一條 Q head → 4 個 Q-values
```

Dueling 是：

```text
CNN → Value 分支
    → Advantage 分支
    → 合成 4 個 Q-values
```

所以對外面來說，其實什麼都沒變。

模型最後還是輸出：

```text
Q(NOOP)
Q(FIRE)
Q(RIGHT)
Q(LEFT)
```

Agent 也還是挑最大的那一個。

---

## 為什麼要把「局面」和「動作」拆開？

可以用一個很簡單的例子理解。

假設某個 Breakout 畫面現在整體是個還不錯的局面，我們先給它：

```text
局面基礎分數 = 10
```

接著四個動作可能是：

```text
NOOP   +0
FIRE   +0.2
RIGHT  +1
LEFT   -1
```

最後就得到：

```text
Q(NOOP)  = 10 + 0   = 10
Q(FIRE)  = 10 + 0.2 = 10.2
Q(RIGHT) = 10 + 1   = 11
Q(LEFT)  = 10 - 1   = 9
```

這樣的好處是，模型不需要四次重複學「這個局面整體其實不錯」。

它可以把共同資訊放在 Value 分支，再讓 Advantage 分支專心處理：

> **這四個動作彼此到底差在哪裡？**

這就是 Dueling 最核心的想法。

它不代表一定會比 Standard DQN 強，但對「很多動作在某些 state 下其實差不多」的問題，這種拆法有機會讓模型更有效率地學到共同資訊。

---

## 為什麼公式不是單純 `Q = V + A`？

直覺上，我們可能會直接寫：

```text
Q = V + A
```

但這會出現一個小麻煩。

假設：

```text
V = 10
A = [0, 1, -1, 0]
```

最後得到：

```text
Q = [10, 11, 9, 10]
```

可是下面這組其實也會得到完全一樣的 Q：

```text
V = 100
A = [-90, -89, -91, -90]
```

結果還是：

```text
Q = [10, 11, 9, 10]
```

所以如果只寫 `V + A`，模型其實沒有明確規則知道：

> 「多少應該算在 V？多少應該算在 A？」

解法很簡單：

> **把 Advantage 的平均值扣掉。**

因此 Day 19 使用：

```text
Q(s,a)
=
V(s)
+
A(s,a)
-
所有 action 的 A 平均值
```

程式就是：

```python
q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
```

這樣做之後，四個動作的 Advantage 平均會變成 0。

因此可以更直覺地理解成：

```text
V
= 這四個 Q-values 的共同基準

Advantage
= 每個 action 相對這個基準要往上或往下多少
```

在理論上，前面那個「V 和 A 可以任意搬來搬去」的問題叫做 **可辨識性問題（identifiability problem）**。

名字可以先不記，真正重要的是：

> **扣掉平均值，是為了讓 Value 和 Advantage 的分工更清楚。**

---

## 沒有 Value 的正確答案，模型怎麼知道要學什麼？

這也是很容易產生的疑問。

Replay Buffer 裡面其實只有：

```text
state
action
reward
next_state
```

裡面並沒有一欄叫：

```text
正確的 V(s)
正確的 A(s,a)
```

Dueling 也沒有另外準備兩份標籤。

它的訓練方式其實和原本 DQN 一樣：

```text
模型算出 Q-value
        ↓
和 Bellman target 比較
        ↓
算出 loss
        ↓
反向傳播
        ↓
更新 CNN、Value 分支、Advantage 分支
```

也就是說，最後真正接受訓練的目標仍然是 **Q-value**。

Value 和 Advantage 只是模型內部把 Q-value 拆開來學的方式。

所以 Dueling 沒有改掉 Q-learning 的核心，只是換了一種網路結構。

---

## Dueling 和 Double DQN 不是同一件事

這兩個名字很容易混在一起，但可以用一句話分清楚：

| 方法 | 改的是什麼？ |
|---|---|
| Double DQN | 下一個 state 的 target 怎麼算 |
| Dueling | Q-value 在網路裡怎麼拆開來學 |

Double DQN 解決的是之前講過的 **Q-value 高估問題**。

Dueling 解決的是：

> 能不能把「局面共同資訊」和「動作之間的差異」分開學？

所以兩個方法完全可以一起用。

這就是：

```text
Dueling Double DQN
=
Double DQN 的 target 算法
+
Dueling 的網路結構
```

Day 19 實際做的就是這個組合。

---

## 用真實 Breakout 畫面驗證一次

前面都是概念。接下來直接拿模型真的算出來的結果看一次。

這張圖使用的是一個 **真實的 Breakout state**，並送進完成 5,000 transitions 短跑驗證後的 Dueling Double DQN checkpoint：

[![一個真實 Contract v2 Breakout state 的 Value、raw Advantage、centered Advantage 與 Q-values](https://github.com/Tommyweige/breakout-rl-engineering/blob/5877108fc34b9e69b2ba835e9932886d1314a2a2/assets/day19/dueling-value-advantage-q.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/5877108fc34b9e69b2ba835e9932886d1314a2a2/assets/day19/dueling-value-advantage-q.png)

這張圖分成四塊。不要一次看全部，按照 **左上 → 右上 → 左下 → 右下** 的順序看會最清楚。

### ① 左上：模型現在看到的是什麼？

左上角是 Breakout 的真實遊戲畫面。

但這裡有一個很重要的細節：**圖上只顯示最後一幀，模型實際收到的不是一張圖，而是連續四幀。**

```text
模型真正的輸入 shape
=
(4, 84, 84)
```

為什麼要四幀？

因為只看一張靜態畫面，我們只知道「球在哪裡」，卻很難知道「球正在往哪裡飛」。把連續幾幀疊起來，模型才有機會從位置變化判斷移動方向。

所以左上角的用途只是讓我們知道：

> **下面的 Value、Advantage 和 Q-value，都是針對這一個真實 Breakout state 算出來的。**

它不是隨便捏造一組數字。

### ② 右上：Value 是這個局面的「共同底分」

右上角只有一個數字：

```text
V(s) = 0.105373
```

可以先把它想成：

> **不管最後選哪一個動作，模型先替現在這個局面給了一個約 0.105 的共同底分。**

這就是 Dueling 和 Standard DQN 最大的不同之一。

Standard DQN 直接算四個 Q-values；Dueling 先抽出一個大家共用的基礎，再去處理動作之間的差別。

所以接下來真正會把 `NOOP / FIRE / RIGHT / LEFT` 分開的，是左下角的 Advantage。

### ③ 左下：Advantage 告訴我們「哪個動作要加分、哪個要扣分」

左下角會看到兩組柱子：

```text
raw A(s,a)
A(s,a) - mean(A)
```

第一組 **raw Advantage** 是 Advantage 分支剛算出來的原始數字：

```text
NOOP   = +0.027567
FIRE   = +0.027929
RIGHT  = +0.028719
LEFT   = +0.027928
```

你會發現四個數字全部都是正的。

但這時候先不要解讀成：

> 「四個動作全部都很好。」

因為前面說過，raw Advantage 還有「基準到底放在哪裡」的問題。真正拿去和 Value 合成 Q-value 前，我們還要先把四個 Advantage 的平均值扣掉。

這次平均值是：

```text
mean(A) = 0.028036
```

所以中心化之後變成：

| Action | raw Advantage | 扣掉平均後 | 直覺 |
|---|---:|---:|---|
| NOOP | +0.027567 | -0.000469 | 比平均稍差 |
| FIRE | +0.027929 | -0.000106 | 幾乎接近平均 |
| RIGHT | +0.028719 | +0.000683 | 四個裡最好 |
| LEFT | +0.027928 | -0.000107 | 幾乎接近平均 |

這時候就很好懂了。

`RIGHT` 是唯一明顯高於平均的動作，所以它會把共同底分往上加；其他三個動作則稍微往下扣。

換句話說，這一塊才是在回答：

> **「在同一個局面下，四個動作彼此差在哪裡？」**

還有一點值得注意：這四個 centered Advantage 都非常接近 0。

這代表在這個 **5K checkpoint 的這一個 state** 裡，模型其實還沒有把四個動作拉開很大的差距。這很合理，因為它目前只是一個短跑驗證用 checkpoint，而不是已經充分訓練好的 Agent。

### ④ 右下：Value + Advantage，才得到真正拿來選動作的 Q-value

最後看右下角。

模型會把右上角共同的 Value，分別加上左下角四個 centered Advantage：

```text
Q(s,a)
=
V(s)
+
centered Advantage(s,a)
```

例如 `RIGHT`：

```text
共同底分
0.105373

+

RIGHT 的加分
0.000683

=

Q(RIGHT)
0.106056
```

其他三個也是完全相同的算法：

| Action | Value | centered Advantage | 最終 Q-value |
|---|---:|---:|---:|
| NOOP | 0.105373 | -0.000469 | 0.104904 |
| FIRE | 0.105373 | -0.000106 | 0.105267 |
| RIGHT | 0.105373 | +0.000683 | **0.106056** |
| LEFT | 0.105373 | -0.000107 | 0.105266 |

所以這張圖其實完整呈現了一次 Dueling 的計算：

```text
真實 Breakout state
        ↓
共同底分 V(s) = 0.105373
        ↓
四個動作各自加分 / 扣分
        ↓
得到 4 個 Q-values
        ↓
RIGHT 最大
        ↓
Greedy action = RIGHT
```

### 這張圖最值得看的，其實不是 `RIGHT`

看到 `RIGHT` 最大，很容易把注意力放在「模型決定往右」。

但 Day 19 真正想驗證的不是這個。

最重要的是：**程式算出來的數字真的符合 Dueling 的公式。**

例如：

```text
Q(RIGHT)
=
0.105373 + 0.000683
≈
0.106056
```

實際 network 輸出的 `Q(RIGHT)` 也是 `0.106056`。

把四個 action 全部重新計算一次，最大的誤差只有：

```text
7.45e-09
```

這已經小到可以視為一般浮點數運算造成的數值誤差。

所以這張圖真正能證明的是：

> **我們寫的 Dueling Network 確實有按照「共同 Value + 每個動作的相對加減分」組回四個 Q-values。**

### 讀這張圖時還有兩個不要搞混的地方

第一，**四個區塊的柱狀圖不是共用同一個刻度**。

所以不要拿右上角 Value 柱子的長度，直接和左下 Advantage 或右下 Q-value 的柱高比較大小。每一塊是為了讓自己的數值變化看得清楚，真正比較時要看圖上標示的數字。

第二，這只是：

```text
1 個 state
+
1 個只跑 5,000 transitions 的 checkpoint
```

它可以拿來確認 Dueling 的計算方式正確，但不能拿來證明：

```text
模型已經學會 Breakout
Dueling 一定比 Standard DQN 強
RIGHT 在這種畫面永遠都是正確答案
```

這三件事都需要更長的訓練與更多 evaluation 才能回答。

完整 machine-readable 數值保留在 [Day 19 V/A/Q evidence](https://github.com/Tommyweige/breakout-rl-engineering/blob/5877108fc34b9e69b2ba835e9932886d1314a2a2/assets/day19/dueling-value-advantage-q.json)。

---

## 模型變複雜了，訓練會不會慢很多？

Dueling 把原本一條 Q head 拆成兩條分支，所以模型確實變大了。

Day 19 用相同設定，在 NVIDIA CUDA 上各跑 5,000 transitions：

| Architecture | Parameters | Training speed | Peak reserved VRAM |
|---|---:|---:|---:|
| Standard | 1.69M | 376 transitions/s | 628 MiB |
| Dueling | 3.29M | 360 transitions/s | 1,188 MiB |

這次 Dueling 的速度大約慢了 4%。

也就是說，它不是免費的：

```text
模型更大
VRAM 用得更多
速度稍微下降
```

但至少在目前這套 RTX 4060 Laptop GPU 的 CUDA training pipeline 裡，Dueling 可以正常完成 forward、loss、backward、optimizer update、target network 更新，以及 checkpoint save/load。

所以 Day 19 可以確認：

> **Dueling 已經能正常接進我們原本的訓練系統。**

完整效能與環境資訊放在 [Day 19 runtime evidence](https://github.com/Tommyweige/breakout-rl-engineering/blob/5877108fc34b9e69b2ba835e9932886d1314a2a2/assets/day19/dueling-smoke-runtime.json) 與 [Day 19 smoke report](https://github.com/Tommyweige/breakout-rl-engineering/blob/5877108fc34b9e69b2ba835e9932886d1314a2a2/reports/day19-dueling-smoke.md)。

---

## Day 19 到底完成了什麼？

今天最重要的不是 5,000 transitions 跑了多少分，而是把一個概念真正接進 DQN：

```text
Standard DQN
直接學四個 Q-values

        ↓

Dueling Network
先學「這個局面整體如何」
再學「每個動作相對好多少、差多少」

        ↓

最後仍然得到四個 Q-values
```

而且它可以直接搭配 Double DQN，不需要重新寫一套訓練器。

不過現在還不能回答最重要的問題：

> **這樣拆開之後，真的會讓 Agent 玩得更好嗎？**

因為 Day 19 的 5K run 只是確認程式能正常運作，而且 Dueling 模型本身也比 Standard 大：

```text
Standard ≈ 1.69M parameters
Dueling  ≈ 3.29M parameters
```

所以即使之後 Dueling 分數比較高，也不能只看一次短跑就說一定是架構本身造成的。

Day 20 才會把：

```text
DQN
Double DQN
Dueling Double DQN
```

放在相同的環境規則、training seeds、CUDA backend 與 500K transitions 下比較。

到那時，我們才真正開始回答：

> **Dueling 不只是「想法合理、程式能跑」，而是真的值得拿去做後面的長訓練嗎？**
