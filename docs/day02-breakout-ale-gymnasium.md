# Day 2｜當 Agent 走進 Atari Breakout：ALE 與 Gymnasium

Day 1 我們談了為什麼想做這個系列，也談到強化學習最後必須回到一個真實的環境裡。今天就讓 Agent 走進 Breakout，看看它究竟看見什麼、可以做什麼，以及一個遊戲畫面是怎麼變成程式可以理解的資料。

今天先從遊戲本身開始，讓一個隨機 Agent 和一個人類玩家同時操作兩個 Breakout，從最基本的互動理解強化學習的介面，再為後面的神經網路訓練做好準備。

## 先從 Atari 的故事說起

Atari 在 1972 年成立，早期的 Pong 讓許多人第一次接觸到街機電子遊戲。幾年後，Atari 在 1976 年推出 Breakout：玩家控制畫面下方的球拍，讓球反彈，逐層擊破上方的磚塊。1977 年，Atari 又推出 Atari VCS，後來更常被稱為 Atari 2600，家用遊戲因此開始走進更多人的客廳。

這段歷史可以參考 [Atari 官方簡介](https://atari.com/pages/about)、[The Strong Museum 的 Pong 介紹](https://www.museumofplay.org/games/pong/) 和 [Science Museum 的電子遊戲時間線](https://blog.scienceandmediamuseum.org.uk/60-years-history-of-videogames-timeline-1951-2011/)。Atari 的 Breakout 資料也明確把它標示為 1976 年的 Atari 街機遊戲。[Atari 的 Breakout 介紹](https://atari.com/pages/among-the-top-five-highest-grossing-arcade-video-games-of-1976)

Breakout 的規則很簡單：球拍左右移動，球在畫面裡反彈，磚塊被打掉之後得到分數。對剛開始學習強化學習的我們來說，這種簡單反而很珍貴。畫面、動作和分數之間的關係清楚可見，也很容易觀察 Agent 每一步做了什麼。

## Atari、Breakout 和 ALE 到底是什麼關係？

先把三個名字分開：Atari 是公司與品牌，Breakout 是 Atari 推出的原始街機遊戲，而 `ALE/Breakout-v5` 是一個可以讓程式操作這款遊戲的環境名稱。

ALE 是 Arcade Learning Environment 的縮寫。它負責讓 Atari 遊戲在模擬器裡執行，處理遊戲規則、畫面更新、分數與可用動作。Gymnasium 則提供一套統一的溝通方式，讓 Agent 可以用相同的 `reset()` 和 `step()` 介面操作不同環境。

可以把整個關係想成一條線：

```text
Agent
  ↓ action
Gymnasium API
  ↓
ALE / Breakout
  ↓ observation、reward、遊戲狀態
Agent
```

模擬器負責磚塊繪製與 Atari 的遊戲規則，Agent 專注於接收畫面、選擇動作，再把動作交回環境。

## Agent 看見的第一件事：一張畫面

建立環境並重設之後，Agent 會拿到一張 observation。這個專案目前保留 Breakout 的原始 RGB 畫面，因此 observation space 會顯示：

```text
Box(0, 255, (210, 160, 3), uint8)
```

這串文字可以拆成四個部分來看。`Box` 表示這是一個多維數值陣列，每個數值都有範圍；`0` 和 `255` 是像素值的上下限；`(210, 160, 3)` 代表畫面高 210、寬 160，最後的 3 是 RGB 三個顏色通道；`uint8` 則表示每個像素通道使用 0 到 255 的 8 位元整數。

換句話說，一張畫面裡有 `210 × 160 × 3` 個數值。對人類來說，這是一張 Breakout 畫面；對 Agent 來說，這是一個形狀固定、數值範圍固定的陣列。

## Agent 可以做什麼？

Breakout 的 action space 是：

```text
Discrete(4)
['NOOP', 'FIRE', 'RIGHT', 'LEFT']
```

`Discrete(4)` 代表共有四個離散動作。它們依序是保持不動、發射球、向右和向左。Agent 實際傳入環境的是整數 0 到 3，環境再依照 action meanings 把整數解讀成對應的動作。

人類玩家和 Agent 使用的是同一組動作。差別只在於：random Agent 從四個選項裡隨機抽一個，人類則透過鍵盤決定要送出哪一個。

## 一次互動是怎麼發生的？

遊戲開始時，先呼叫 `reset()`：

```python
observation, info = env.reset(seed=42)
```

這會建立一個新的 episode，並把第一張畫面交給 Agent。接著，Agent 選一個 action，呼叫 `step()` 讓遊戲前進：

```python
observation, reward, terminated, truncated, info = env.step(action)
```

這五個回傳值就像環境給 Agent 的一則回覆：新的 `observation` 是下一張畫面，`reward` 是這一步得到的分數訊號，`terminated` 表示遊戲本身是否結束，`truncated` 表示外部時間限制是否到達，`info` 則放著額外的環境資訊。

當一局結束，下一局會重新從 `reset()` 開始。這就是強化學習最基本的循環：看見狀態、選擇動作、收到回饋，再看下一個狀態。

## 把 AI 和人類放在同一個畫面裡

為了讓這個互動更直觀，這次的畫面同時放入兩個 Breakout。左側由 random Agent 操作，右側交給人類。兩個環境各自擁有自己的遊戲狀態，所以可以清楚比較兩種控制方式。

程式的核心流程可以濃縮成這幾行：

```python
ai_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
human_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

ai_action = ai_env.action_space.sample()
human_action = human_input.next_action()

ai_observation, ai_reward, ai_terminated, ai_truncated, _ = ai_env.step(ai_action)
human_observation, human_reward, human_terminated, human_truncated, _ = human_env.step(human_action)
```

左邊的 `action_space.sample()` 是目前的 random policy。右邊的 `next_action()` 會讀取鍵盤狀態，按住方向鍵時持續送出移動動作，按一下 `Space` 或 `F` 時送出一次 `FIRE`，把球發射出去。

兩張 RGB 畫面最後由 Tkinter 放到同一個視窗裡。操作右側時，按住 `←` 或 `A` 可以向左移動，按住 `→` 或 `D` 可以向右移動；`R` 會重設右側遊戲，`Esc` 或 `Q` 可以關閉視窗。底部的操作提示保持固定，讓畫面閱讀起來更舒服。

## 實際畫面：AI 與人類並排 GIF

下面的 GIF 來自兩個實際運作中的 `ALE/Breakout-v5` environment。左側使用 random policy，右側使用 scripted input 重現人類的發射與移動，畫面尺寸和操作提示都按照實際視窗呈現。

![AI and human playing Atari Breakout side by side](../assets/day02-ai-vs-human.gif)

這段畫面的重點在兩個完整的互動循環：左側 Agent 自己選擇動作，右側則由人類輸入控制。下一階段加入真正的學習 policy 之後，就能用同一個畫面觀察 Agent 的進步。

## 今天到底完成了什麼？

今天我們把一款 1976 年的 Atari 街機遊戲接上了 Gymnasium，也讓 ALE 成為 Agent 和遊戲之間的執行層。Agent 現在知道自己收到的是一張 `(210, 160, 3)` 的 RGB 畫面，也知道 Breakout 提供四個可用動作。

實際執行時，環境介面會印出：

```text
Observation shape: (210, 160, 3)
Observation space: Box(0, 255, (210, 160, 3), uint8)
Action space: Discrete(4)
Action meanings: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
```

這些資訊看起來很像幾行簡單的輸出，卻是後面所有訓練工作的基礎。當我們知道畫面長什麼樣子、動作怎麼表示、每一步會得到什麼回覆，才有辦法把 random policy 換成真正會學習的 Agent。

Day 2 先到這裡。下一篇開始，我們會進一步處理畫面，讓模型逐漸學會從遊戲狀態中做出更好的選擇。
