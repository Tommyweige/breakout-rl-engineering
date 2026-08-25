# Day 12｜完整 DQN Training Loop：一筆遊戲經驗怎麼真的變成學習

前面幾天，我們已經把 DQN 需要的幾個重要零件分開做完了：模型可以輸出 Q-value、Replay Buffer 可以保存經驗、epsilon-greedy 可以決定要探索還是利用，Target Network 也能提供比較穩定的下一步估計。

但「每個零件都能運作」還不等於「模型真的會訓練」。

Day 12 第一次把它們全部接起來，回答一個很實際的問題：**Agent 在 Breakout 裡做出一個動作之後，這筆遊戲經驗到底怎麼一路變成一次模型更新？**

今天不要求 Agent 已經會玩 Breakout。我們先確認更基本的事情：遊戲經驗有沒有真的被收集、模型有沒有真的被修改，以及整條訓練流程能不能被觀察。

## 一筆遊戲經驗，先從「玩一步」開始

完整的 DQN 訓練其實一直在做兩件事：**玩遊戲收集經驗**，以及**拿過去的經驗更新模型**。

[![DQN training loop 分成收集經驗與更新模型兩個不同節奏，Replay Buffer 位在兩者之間](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/4c295d296a358fec55010424d0575021953bd6db/assets/day12/dqn-training-loop-overview.svg?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/4c295d296a358fec55010424d0575021953bd6db/assets/day12/dqn-training-loop-overview.svg)

先看圖的上半部。

Agent 拿到目前的遊戲畫面後，用 epsilon-greedy 選一個動作。Breakout 執行這個動作，再回傳新的畫面、reward（獎勵），以及這一局是否結束。

這些資料合在一起，就是一筆**互動紀錄（transition）**：

```text
目前狀態 → 做了什麼動作 → 得到多少 reward → 來到什麼新狀態
```

這筆紀錄不會立刻拿來修改模型，而是先放進 Replay Buffer。

圖的下半部才是學習。只有當 Buffer 裡已經有足夠的經驗，而且到了該更新模型的時間，才會從裡面隨機抽出一小批資料來訓練。

所以 Replay Buffer 正好隔開了兩件事：**Agent 幾乎每一步都在產生新經驗，但模型不需要每走一步就立刻學一次。**

這也帶出第一個容易混淆的地方：遊戲走一步，和模型學一次，其實是兩種不同的「step」。

## 走一步遊戲，不代表模型也更新一步

這篇文章裡會出現兩種 step。

**Environment step** 可以直接理解成「Agent 和遊戲互動一步」：選一個 action，呼叫一次 `env.step(action)`，再拿到環境回傳的結果。

**Optimizer step** 則是「模型真的學一次」：拿一批舊經驗算出誤差，最後修改 Online Network 的參數。

這次短跑測試使用：

```text
learning_starts = 32
train_frequency = 4
batch_size      = 8
```

`learning_starts = 32` 表示前 31 步先收集資料，第 32 步之後才允許開始學習。

原因不只是「資料太少會抽不到」。遊戲剛開始時，Replay Buffer 裡幾乎都是很相似的開場畫面；如果立刻開始學，模型反覆看到的資料會非常單一。先累積一段經驗，第一次更新時至少能看到比較多樣的情況。

`train_frequency = 4` 則表示開始訓練後，每 4 個 environment steps 才更新一次模型。

因此跑到 1,000 個 environment steps 時，更新次數應該是：

```text
(1000 - 32) / 4 + 1 = 243
```

實際結果也正好是 243 次。

這個數字不是遊戲分數，它只是先確認：**我們設定的訓練節奏真的有照預期執行。**

知道「什麼時候要學」之後，下一個問題就是：從 Replay Buffer 抽出資料後，DQN 到底要改哪一個 Q-value？

## 一筆經驗，只能告訴模型當時那個動作的結果

DQN 看一個 state 時，會一次輸出所有 action 的 Q-value。假設某個畫面得到：

```text
Q(s, ·) = [0.4, 0.1, 0.8, 0.3]
```

但 Replay Buffer 裡那筆 transition 還記得 Agent 當時真正做了哪個 action。

如果當時做的是 action `2`，那這筆經驗能直接告訴我們的，就是第三個 Q-value `0.8` 應該怎麼修正。

原因很簡單：這筆 reward 和下一個 state 都是**做了 action 2 之後**才出現的。我們沒有真的做另外三個 action，所以不能拿同一個結果去教另外三個 Q-value。

實作裡會用 PyTorch 的 `gather`，從每一筆資料的所有 Q-values 中挑出當時真正採取的 action。名字看起來像 API 細節，但它在這裡做的事情其實就只有「挑出正確的那一格」。

現在我們已經找到模型目前的預測 `Q(s, a)`。但要訓練模型，還缺另一個東西：**這個 Q-value 應該往哪裡靠？**

## Bellman Target 就是這次更新的參考答案

Day 11 已經介紹過 Target Network。到了完整 training loop 裡，它真正的用途就是幫我們估計「下一個 state 還有多少價值」。

Day 12 的學習目標可以寫成：

```text
target = reward
       + gamma × (1 - terminated)
       × max Q_target(next_state, action)
```

先不用被公式嚇到，它其實只是在說：

**這個 action 的價值 = 現在真的拿到的 reward + 下一個 state 未來可能帶來的價值。**

`gamma` 是折扣因子，用來控制我們有多重視未來的 reward；Target Network 則負責估計後面那一段未來價值。

如果 `terminated=True`，代表遊戲真的已經結束，後面沒有下一步可以繼續，所以這時候學習目標只剩目前拿到的 reward。

`truncated=True` 則不一定代表遊戲世界真的結束。它可能只是因為外部限制，例如時間到了，所以不能看到 truncated 就一律把未來價值清掉。

到這裡，一筆資料終於有了兩個可以比較的數字：

```text
模型目前的預測：Q(s, a)
這次的參考答案：Bellman target
```

接下來只剩最後一步：讓兩者的差距真的改變模型。

## 預測和答案的差距，最後才會改變模型

Day 12 使用 **Huber loss** 來衡量目前的 Q-value 和 Bellman target 差多少。

可以先把 loss 理解成一個數字：**差得越多，這次需要修正得越多；差得越少，代表目前預測比較接近目標。**

Huber loss 的特點是，遇到很大的誤差時，不會像單純平方誤差那樣快速放大，因此很適合先拿來當 DQN 的穩健起點。這不代表它永遠最好，只是 Day 12 先選一個合理的做法。

有了 loss 之後，PyTorch 會用反向傳播（backpropagation）算出模型參數該往哪個方向調整，再由 optimizer——也就是「真正修改參數的機制」——更新 **Online Network**。

Target Network 不會跟著這一步一起被修改。它仍然維持 Day 11 的設計：隔一段時間，才把 Online Network 的最新參數整份複製過去。

所以一次真正的學習可以濃縮成：

```text
抽一批舊經驗
→ 找出當時做過的 action 對應 Q-value
→ 算出 Bellman target
→ 比較兩者差距
→ 更新 Online Network
```

模型現在真的會更新了。不過還有一個很容易混淆的地方：訓練時拿來學的 reward，和最後用來判斷 Agent 玩得好不好的遊戲分數，不一定是同一個數字。

## 訓練用的 Reward 和遊戲分數要分開

Atari 的 reward 在這裡有兩種用途。

第一種是拿來訓練模型。這次可以把 reward 簡化成：

```text
正 reward → +1
0          → 0
負 reward → -1
```

這叫做 **reward clipping**。目的不是竄改遊戲分數，而是讓訓練時不同大小的 reward 不要造成太大的尺度差異。

Replay Buffer 存的是這個拿來訓練的 reward。

但如果我們想知道 Agent 在 Breakout 裡到底拿了幾分，就一定要另外保留環境原本給的 reward。每一局把原始 reward 加起來，才是實際的遊戲成績。

所以兩件事要分開：訓練模型時可以使用 clipped reward；評估 Agent 玩得好不好時，仍然要看原始遊戲分數。

這些元件都接起來之後，接下來就不是再看公式，而是直接讓整條流程真的跑一次。

## 第一次把完整 DQN 訓練真的跑起來

我先用固定 seed `42`、CPU，跑了一個 1,000 environment steps 的短跑測試。這種短時間、低成本的測試通常叫 **smoke run**，目的只是先確認整套系統能正常工作，不是拿來證明 Agent 已經學會遊戲。

先直接看實際畫面：

[![固定 seed 42 的 Breakout DQN smoke run：真實遊戲畫面與 step、epsilon、raw score、training phase](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/5884cd070c380c78209e5cd2b53fce21d9cf5e1e/assets/day12/training-smoke.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/5884cd070c380c78209e5cd2b53fce21d9cf5e1e/assets/day12/training-smoke.gif)

這不是示意動畫，而是 `ALE/Breakout-v5` 實際執行時 render 出來的畫面。GIF 使用固定 seed `42`，每 8 個 environment steps 取一張，共 125 張，播放時間約 12.5 秒。

畫面上只保留幾個現在真的有用的資訊：目前跑到第幾步、epsilon、這一局目前的原始分數，以及這一刻是在收集資料、前期累積經驗，還是剛好執行模型更新。

從 GIF 可以直接確認兩件事：**Agent 的確持續在 Breakout 裡做 action；而且累積到足夠經驗後，模型更新也真的開始發生。**

接著再看這次 1,000 steps 最後留下的數字：

| 項目 | 結果 |
| --- | ---: |
| Environment steps | 1,000 |
| 完成 episodes | 4 |
| 模型更新次數 | 243 |
| Target Network 同步次數 | 11 |
| Replay Buffer 最後大小 | 256 |

除了最後結果，我也把訓練過程中的幾個重要訊號持續記錄下來：

[![Day 12 真實 CPU smoke run 的 raw episode return、Huber loss、selected Q mean 與 epsilon](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/5d725ae7d752439d390098726f238dbbd5d01a5a/assets/day12/training-overview.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/5d725ae7d752439d390098726f238dbbd5d01a5a/assets/day12/training-overview.png)

這張圖不用每條線都急著分析。Day 12 先看最基本的現象就夠了：epsilon 確實隨步數下降；開始訓練之後 loss 真的出現；Q-value 也會隨模型更新而改變。

GIF 和這張圖放在一起，能支持的結論其實很明確：**完整 DQN 訓練流程真的已經接通，而且 Online Network 確實有在被修改。**

但它們都不能證明 Agent 已經學會 Breakout。

## 程式有在訓練，不代表 Agent 已經變強

這次只完成了 4 個 episode，原始分數是 `2、3、0、0`。

樣本這麼少，根本不足以判斷 Agent 是否真的有進步。就算 loss 變小，也只能表示模型更接近目前這批資料提供的學習目標，不代表最後選出的動作一定會讓遊戲分數提高。

這也是為什麼訓練時不能只盯著一個數字。

如果之後分數不升，我們還會想知道：epsilon 有沒有正常下降？Replay Buffer 有沒有正常累積？Q-value 有沒有突然變得很大？模型到底有沒有持續更新？Target Network 有沒有照預期同步？

這些紀錄不會直接告訴我們答案，但能把「Agent 好像沒在學」這句模糊的感覺，拆成一個個可以檢查的問題。

這正是 Day 13 要繼續處理的內容。

## Checkpoint 讓長時間訓練不用全部重來

當訓練時間變長之後，另一個很實際的需求就是存檔。

Day 12 的訓練存檔（checkpoint）會保存 Online Network、Target Network、optimizer，以及目前跑到第幾步等主要狀態。這樣程式中斷後，不必把模型重新從隨機初始化開始。

不過目前 **Replay Buffer 沒有一起存進 checkpoint**。Atari 的 Buffer 會保存大量畫面，如果每次都把整份資料一起存下來，檔案會非常大。

因此恢復訓練後，模型本身可以接著原本的參數繼續，但 Replay Buffer 需要重新累積一段經驗，之後才能再次開始更新。

所以目前的 resume 可以理解成「接回主要模型狀態」，而不是保證中斷前後每一步遊戲都會完全一模一樣。

## Day 12：DQN 第一次形成完整的學習閉環

回頭看今天真正完成的事情，其實只有一條主線。

Agent 先和 Breakout 互動，產生一筆經驗；經驗進入 Replay Buffer；累積到足夠資料後抽出一批舊經驗；Online Network 給出目前的 Q-value，Target Network 提供學習目標；兩者的差距變成 loss，最後真的修改 Online Network。

到這裡，DQN 才第一次不只是「有模型、有 Buffer、有探索策略」，而是形成了一條真正能持續學習的閉環。

這次 1,000 steps 的短跑證明這條閉環可以運作，沒有證明 Agent 已經會玩 Breakout。

而 Day 13 要接著回答的，就是更棘手的問題：**如果程式一直在跑、模型也一直在更新，但分數就是沒有變好，我們要怎麼知道問題出在哪裡？**

下一篇：[Day 13｜除錯不穩定的 RL 訓練：從 sanity check 到 training diagnostics](day13-debugging-unstable-rl-training.md)
