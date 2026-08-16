# Day 3｜當 Agent 按下一個動作之後，環境到底回傳了什麼？

當 Agent 按下一個 action 之後，環境到底回傳了什麼？

在 Atari Breakout 裡，按鍵只是表面上看得到的那一刻。對強化學習來說，一次互動真正留下的是一段完整的資料交換：

~~~text
state
  ↓ action
environment
  ↓
reward + next observation
~~~

Agent 在某個 observation 上選擇 action，遊戲依照自己的規則往前推進，再把 reward、下一個 observation，以及 episode 是否結束交回來。今天要理解的不是某個 API 的參數列表，而是這筆資料為什麼會長成這樣，以及每個欄位之間的時間關係。

## 先看一次實際互動

先讓 Agent 用隨機策略走 20 步，保留每一步的主要結果。固定隨機種子後，實際輸出的前 3 筆與最後摘要如下，中間步驟省略：

~~~text
Step 0
  state       : shape=(210, 160, 3), dtype=uint8
  action      : 0 (NOOP)
  reward      : 0.0
  next_state  : shape=(210, 160, 3), dtype=uint8
  terminated  : False
  truncated   : False

Step 1
  state       : shape=(210, 160, 3), dtype=uint8
  action      : 3 (LEFT)
  reward      : 0.0
  next_state  : shape=(210, 160, 3), dtype=uint8
  terminated  : False
  truncated   : False

Step 2
  state       : shape=(210, 160, 3), dtype=uint8
  action      : 2 (RIGHT)
  reward      : 0.0
  next_state  : shape=(210, 160, 3), dtype=uint8
  terminated  : False
  truncated   : False

Transitions collected: 20
Episode return: 0.0
Episodes ended: 0
~~~

在這段 20 步的觀察裡，reward 都是 0.0，遊戲仍持續向前推進。球可能移動，球拍也可能換了位置，下一張畫面照樣被環境回傳。0.0 表示這個時間點的分數回饋是零；它和畫面、action、episode status 一樣，都是互動結果的一部分。

把其中一步寫成程式操作，就是：

~~~python
observation, info = env.reset(seed=42)
env.action_space.seed(42)
action = env.action_space.sample()

next_observation, reward, terminated, truncated, info = env.step(action)
~~~

這裡的關鍵是時間順序：先有 action 前的 observation，接著 action 被環境執行，最後才得到 reward 和下一個 observation。

## Observation 是 Agent 實際看見的資料

環境在 reset 時先交給 Agent 一個 observation。以原始的 Breakout 畫面來說，它是一張形狀為 210 × 160 × 3 的 RGB 圖片，每個像素以 uint8 表示。

這張圖是 Agent 可以使用的資訊。它能看到球拍和球的位置，也能看到磚塊排列；一張靜態畫面則可能只呈現球的速度與移動方向的一部分。這個差異來自資訊本身的邊界：如果方向要靠前後畫面的差異判斷，Agent 就需要同時看見一段時間上的 observation。

在整理一筆 transition 時，常會把 action 發生前的 observation 稱為 state：

~~~text
state = action 前的 observation
next_state = action 後的 observation
~~~

這個寫法方便我們描述資料流。單張畫面是 Agent 當下實際拿到的輸入，而環境內部還有一些遊戲運作所需的資訊只在畫面之外存在。

## Action 是環境能理解的指令

Breakout 的 action space 可以表示成：

~~~text
Discrete(4)
0 → NOOP
1 → FIRE
2 → RIGHT
3 → LEFT
~~~

Agent 傳給環境的是一個整數，環境再依照 action space 把它解讀成遊戲指令。RIGHT 的意思不是「移動固定數量的像素」，而是選擇「向右」這個離散動作；球拍實際移動多少，則由遊戲規則和當下狀態共同決定。

這裡形成一個很重要的分工：policy 決定要選哪個 action，environment 決定這個 action 會造成什麼結果。今天用隨機策略，只是讓我們容易觀察資料如何產生；之後換成神經網路，這個互動邊界仍然成立。

## Reward 是這一步得到的回饋

環境執行 action 之後，會回傳一個數值 reward。它描述的是這一步帶來的回饋，不是模型預測出來的答案，也不是畫面本身的標籤。

Breakout 有許多 step 不會立刻打中磚塊，因此 reward 可能是 0.0。這個數值和畫面變化可以同時成立：遊戲狀態向前走了一步，但分數在這一步維持不變。

單一步的 reward 和一整局累積的 return 也要分開看：

~~~python
episode_return += reward
~~~

reward 是這次 action 的即時回饋；episode return 則是從 episode 開始到目前為止累積的回饋。當我們看到 Episode return 是 0.0，表示這段互動累積的分數仍是零，而每個 observation 仍可能不同。

目前先直接使用遊戲回傳的原始 reward，因為這樣可以清楚觀察 environment 真正提供的訊號。之後若加入 reward clipping 或 reward shaping，就等於重新定義 Agent 看見的學習目標，需要把它當成另一個設計問題討論。

## 一筆 transition 的完整邊界

把前面的概念放回同一個時間點，一筆 transition 可以寫成：

~~~text
(state, action, reward, next_state)
~~~

如果也保存 episode 的結束狀態，資料會是：

~~~text
(state, action, reward, next_state, terminated, truncated)
~~~

一個完整的互動迴圈大致如下：

~~~python
observation, info = env.reset(seed=42)
env.action_space.seed(42)

for step in range(steps):
    state = observation
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    # 在這裡觀察或保存這筆 transition
    observation = next_state

    if terminated or truncated:
        observation, info = env.reset()
~~~

這段流程裡，state 是 action 發生前的 observation，next_state 是這次 action 真正推進遊戲後的 observation。資料被觀察或保存之後，下一輪才把 next_state 當成新的 state。

episode 結束時，next_state 仍然屬於剛剛結束的那一局；reset 產生的 observation 則是下一局的起點。兩者如果在資料邊界上被混在一起，前一局最後一步的結果就會被錯接到下一局。

Gymnasium 用 terminated 和 truncated 表達兩種不同的結束原因。terminated 代表遊戲本身達到終止條件，truncated 代表 episode 因為外部限制而被截斷，例如時間上限。收集資料時，任一欄位為 True 都代表下一輪要重新開始；計算後續的 training target 時，兩者的意義則需要分別保留，因為它們對「環境是否真的到達終點」代表不同的資訊。

## 從 transition 看到強化學習資料的形狀

連續幾筆 transition 串起來，就是一段 trajectory：

~~~text
s0 --a0 / r0--> s1
s1 --a1 / r1--> s2
s2 --a2 / r2--> s3
~~~

這個表示法把一次互動的因果關係留下來：Agent 在什麼 observation 上做了什麼選擇，environment 回傳了什麼回饋，下一個 observation 又是什麼。

因此，Breakout 的訓練資料是由互動逐步生成的。Agent 的 action 會影響遊戲接下來的畫面和 reward；當 policy 改變，之後收集到的資料分布也會跟著改變。這是強化學習和固定資料集訓練之間很重要的差異。

當 transition 累積起來，Replay Buffer 就能從過去的互動中抽樣，讓模型利用整段互動資料更新。要讓這個抽樣真的有意義，前提就是今天看到的 transition 邊界必須正確：state、action、reward、next_state，以及 episode 結束資訊要屬於同一個時間接續。

## 今天真正要帶走的事

當 Agent 按下一個動作之後，環境不是只回傳一個分數，而是完成一次有時間順序的資料交換：

~~~text
action 前的 observation
        ↓
environment 執行 action
        ↓
reward + action 後的 observation + episode status
~~~

從這筆資料，我們可以理解：

- observation 是 Agent 實際看見的輸入。
- action 是 Agent 交給 environment 的離散指令。
- reward 是這一步得到的回饋。
- transition 是把前後 observation、action 和結果綁在一起的單位。
- trajectory 是多筆 transition 按時間串起來的序列。

這些名詞不是獨立的定義，而是同一個互動過程的不同切面。下一步自然會遇到一個問題：如果單張 observation 無法直接表達球的移動方向，怎麼把連續畫面整理成 Agent 更容易使用的輸入？這會帶我們進入 grayscale、resize 和 frame stacking。
