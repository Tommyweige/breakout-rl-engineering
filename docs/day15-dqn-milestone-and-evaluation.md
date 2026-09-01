# Day 15｜DQN 真的變強了嗎？一次評估反而抓出了 FIRE deadlock

Day 14 把 DQN 訓練到 100K 個環境步數後，曲線開始出現學習跡象，實際遊玩也不再像完全亂按。

但這時候最危險的事情，就是太快把「看起來有進步」當成「模型真的學會了」。訓練時的分數會受到探索、模型持續更新、當下遇到的遊戲狀態影響；一段 GIF 更只代表某一局發生過什麼。

所以 Day 15 原本只想回答一個很單純的問題：

> **把模型凍結，不再學習，換幾個不同的遊戲起始條件後，它還能不能穩定比 Random Policy 好？**

結果真正有趣的地方，反而不是最後那個平均分數，而是評估過程暴露出了一個之前被訓練流程掩蓋的問題。

## 先把「會不會玩」和「還在學」分開

這次拿來評估的是 Day 14 訓練到 100K 時保存的模型。評估開始後，權重完全不再更新，也不再從 Replay Buffer 取資料學習。

DQN 每一步只做一件事：把目前看到的四張連續畫面送進網路，得到 `NOOP`、`FIRE`、`RIGHT`、`LEFT` 四個動作的 Q-value，再選最大的那一個。

這種做法叫做 **greedy policy**。這裡的 Q-value 可以先理解成模型對「現在做這個動作有多值得」的估計；`epsilon = 0` 則表示評估時不再故意加入隨機探索。

為了知道這個分數到底有沒有意義，我同時跑了一個 Random Policy。它不看畫面，也不使用神經網路，只是從四個合法動作中隨機選一個。

兩邊都跑相同的 15 個固定遊戲起始條件，分數使用 Atari 環境真正回傳的 raw reward，不使用訓練時可能採用的 reward clipping。

第一眼看起來，結果很漂亮：

| Policy | 平均分數 | 中位數 | 最低 | 最高 |
|---|---:|---:|---:|---:|
| Random | 1.33 | 1 | 0 | 5 |
| DQN | 4.53 | 5 | 1 | 11 |

DQN 的平均分數比 Random 高了 3.20 分，中位數也從 1 提高到 5。從回報分布來看，100K checkpoint 顯然已經不只是完全隨機的行為。

[![Random 與 100K DQN 的每局 raw return 分布](https://github.com/Tommyweige/breakout-rl-engineering/blob/3f599ba/assets/day15/random-vs-dqn-returns.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/3f599ba/assets/day15/random-vs-dqn-returns.png)

如果只看到這裡，我很可能會直接寫下「DQN 已經明顯贏過 Random」。

但另外一個數字非常奇怪。

Random 每局平均只走了大約 186 個 agent steps，DQN 卻平均走了 **18,131 steps**。

這不像是普通的「模型比較會活」。差距大到值得先停下來查原因。

## 18,131 steps 不是超強生存能力

Gymnasium 在一局結束時，會區分兩種情況：

- `terminated`：遊戲本身真的結束，例如所有生命都用完。
- `truncated`：不是正常遊戲結束，而是因為外部限制被截斷。

Random 的 15 局全部都是正常 `terminated`。

DQN 卻只有 5 局正常結束，另外 **10 局都撞上 TimeLimit**。這 10 局幾乎全部停在 26,993～27,000 agent steps。

這個數字不是巧合。Breakout 的 Atari 環境每局上限是 108,000 個 emulator frames，而目前 preprocessing 每個 agent action 會向前跑 4 個 frames：

`108,000 ÷ 4 = 27,000`

也就是說，這些局不是「DQN 撐了兩萬多步還沒死」，而是它一直沒有正常把遊戲結束，最後被環境時間上限強制切掉。

真正的問題因此變成：

> **這兩萬多步裡，遊戲到底還有沒有真的在進行？**

## 把其中一局攤開後，問題變得很明顯

我先追了一局會撞到 TimeLimit 的 seed `101`。

在第 54 個 agent step，DQN 掉了一條命。

接下來第一次真正選到 `FIRE`，竟然已經是第 **16,449 step**。

中間隔了：

**16,395 steps。**

而且這整段期間，約 96% 的 observation 幾乎沒有變化，動作也有約 96% 都是 `RIGHT`。最長一段連續不變的畫面超過 16,000 steps。

另一個 TimeLimit seed `103` 幾乎重現同樣的模式：同樣在 step 54 掉命，之後過了 16,392 steps 才再次 FIRE。

反過來看一局正常結束的 seed `102`，第一次掉命後 33 steps 就重新 FIRE，後面的幾次掉命幾乎都只隔 1 step 就重新開始。

這時候問題已經不是單純的「DQN 偏好 RIGHT」。真正的關鍵是 Breakout 有一個特殊規則：**球局開始需要 FIRE；掉一條命後，也需要再次 FIRE 才會重新發球。**

如果模型掉命後沒有 FIRE，畫面就會停在等待發球的狀態。對模型來說，它會持續看到非常相似的畫面；如果此時 Q-value 又一直偏向 RIGHT，就可能進入：

`RIGHT → 幾乎相同的畫面 → RIGHT → 幾乎相同的畫面 → ...`

最後一路卡到 27,000 steps。

這就是這次看到的 **serve deadlock**：不是遊戲玩得特別久，而是掉命後根本沒有正常重新開始。

## 為什麼訓練時沒有這麼明顯？

這裡又出現一個很容易被忽略的差異。

Day 14 訓練後期的 epsilon 不是 0，而是 `0.05`。

也就是即使模型本身想一直按 RIGHT，仍然有 5% 的機率改成隨機動作。Breakout 一共有四個動作，因此探索過程偶爾就可能亂中 FIRE，把原本卡住的球局重新啟動。

正式評估則把 epsilon 降到 0，完全移除這層隨機救援。這反而讓 greedy policy 真正的缺陷第一次完整暴露出來。

光靠這個推論還不夠，所以我又做了兩個對照實驗。

## 兩個對照實驗，把 FIRE 的影響拆開

兩個實驗都使用完全相同的 100K checkpoint 和相同的 15 個遊戲起始條件，不重新訓練模型。

第一組保持 `epsilon = 0`，但改由環境在開局和掉命後立即執行必要的 FIRE。

第二組不提供 FIRE assist，仍然由模型負責 FIRE，只把 epsilon 恢復成訓練後期的 `0.05`。

結果如下：

| 模式 | epsilon | 環境自動 FIRE | 平均分數 | 正常結束 | TimeLimit | 平均局長 |
|---|---:|:---:|---:|---:|---:|---:|
| 原始 greedy 評估 | 0.00 | 否 | 4.53 | 5/15 | 10/15 | 18,131 |
| FIRE assist | 0.00 | 是 | 8.80 | 15/15 | 0/15 | 430 |
| 保留探索 | 0.05 | 否 | 9.20 | 15/15 | 0/15 | 509 |

[![FIRE、TimeLimit 與兩個對照實驗的真實結果](https://github.com/Tommyweige/breakout-rl-engineering/blob/13a99f0/assets/day15/fire-time-limit-diagnostics.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/13a99f0/assets/day15/fire-time-limit-diagnostics.png)

FIRE assist 最直接：15 局全部正常結束，TimeLimit 從 10 局降到 0，而且掉命後大約下一步就能重新發球。

`epsilon = 0.05` 也能消除這批測試裡的 TimeLimit，這正好支持前面的猜想：**訓練時的隨機探索確實可能偶爾幫模型按到 FIRE，把 deadlock 解開。**

不過第二組平均分數 9.20 比 FIRE assist 的 8.80 高，並不代表我們應該選「評估時保留隨機探索」。這裡真正要決定的不是哪一組剛好分數最高，而是未來到底要怎麼定義這個 RL 任務。

如果把重新發球交給隨機探索，模型有時 1 step 就 FIRE，有時可能等幾十甚至幾百步。這會讓「一局 Breakout」混進很多跟打磚塊能力無關的等待時間，也讓不同模型之間的比較受到運氣影響。

因此我最後選擇把 **serve** 當成環境規則，而不是 Agent 必須自己學會的策略。

## 從 Day 16 開始，FIRE 變成環境的一部分

新的 Contract v2 把 Breakout 任務固定成：

| 設定 | 規則 |
|---|---|
| 遊戲 | `ALE/Breakout-v5` |
| Frame skip | 4 |
| Frame stack | 4 |
| Sticky action probability | 0.25 |
| 開局 FIRE | 環境負責 |
| 掉命後 FIRE | 環境負責 |
| 掉一條命是否結束整局 | 否 |
| 評估 epsilon | 0 |
| 分數 | raw Atari reward |
| TimeLimit | 108,000 raw frames，約 27,000 agent steps |

這代表之後 Agent 專心學的是：

**看球在哪裡、移動 paddle、把球打回去、繼續打磚塊。**

而「這一條命結束後要按一下 FIRE 才重新發球」由環境統一處理。

這個選擇本身沒有唯一標準答案。如果研究問題是「Agent 能不能連遊戲啟動動作都自己學會」，那完全可以讓 policy 負責 FIRE。但這個專案接下來要比較的是 DQN、Double DQN、Dueling Network，以及 vectorized training、GPU batching 和部署；把 serve 行為固定下來，能讓後面的比較更接近我們真正想研究的東西。

更重要的是，從現在開始 **training 和 evaluation 都必須使用同一套 FIRE 規則**。不能訓練時讓環境自動 FIRE，評估時又要求模型自己 FIRE；也不能拿舊規則下的分數和新規則下的分數當成完全相同的實驗直接排名。

所以前面得到的 DQN `4.53`、Random `1.33` 仍然保留，它們證明 100K DQN 已經出現學習訊號，也正是這批資料讓我們找到 deadlock。但它們屬於舊的 Contract v1，不會直接拿來當後續模型家族比較的正式基準。

## Day 15 真正驗證到的是什麼？

回到一開始的問題：「100K DQN 到底有沒有學到東西？」

答案是 **有，而且已經能看到比 Random 更有結構的行為；但還遠不到可以宣布模型很強。**

更重要的是，Day 15 讓我重新確認一件在強化學習裡非常容易踩坑的事：

> **分數變高，不一定只代表模型變好了；也可能是環境、探索策略或 episode 規則正在偷偷影響你看到的結果。**

如果我只停在 `4.53 > 1.33`，這篇文章會得到一個看起來漂亮、但其實不完整的結論。

真正有價值的是繼續往下問：為什麼 DQN 一局可以跑到 18,000 steps？為什麼 10 局都剛好停在 27K？為什麼加入 FIRE 後全部恢復正常？

最後得到的不只是一次評估分數，而是一套之後所有實驗都能共用的 Breakout 規則。

Day 16 接下來要把單一環境改成多環境一起跑，再把 action inference 和 Replay insertion 批次送進 GPU。到時候我們不只要看訓練有沒有變快，也要確認 vectorization 沒有再次偷偷改掉今天才固定下來的環境語意。
