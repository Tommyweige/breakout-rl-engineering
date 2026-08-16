# Day 4｜當一張 Atari 畫面還不夠：Preprocessing、Frame Skip 與 Frame Stacking

Day 3 我們把 `env.step(action)` 拆成了一筆 transition，也看見 Breakout 原始 observation 是一張 `(210, 160, 3)` 的 RGB 畫面。

這張畫面可以讓 Agent 看見球、球拍和磚塊，但它也留下了兩個問題：畫面是不是比模型真正需要的資訊大很多？更重要的是，只看一張畫面，Agent 怎麼知道球正在往哪個方向移動？

今天要處理的不是 DQN，也不是訓練 loop，而是把 environment 回傳的原始畫面整理成一個穩定的 observation contract：

~~~text
原始 Atari 畫面
(210, 160, 3) / uint8
        ↓
AtariPreprocessing
        ↓
(84, 84) / uint8
        ↓
FrameStackObservation(4)
        ↓
(4, 84, 84) / uint8
~~~

這個結果會成為後面 CNN 和 DQN 使用的輸入起點。

## 先看原始 observation 的問題

原始 RGB 畫面共有 `210 × 160 × 3` 個像素值。對人類來說，顏色讓畫面更容易閱讀；對 Breakout 的控制而言，球、球拍和磚塊的位置與形狀通常比完整的色彩資訊更重要。

而且單張畫面本身沒有時間方向。假設球在某個時間點位於 `(x, y)`，它可能正往右下方飛，也可能正往左上方飛。兩種情況都可能產生相似的單張畫面；真正能區分它們的，是前後畫面的變化。

所以 Day 4 的工作分成兩件事：先讓每一張畫面更適合儲存和計算，再保留幾張連續畫面，讓 Agent 有機會從像素變化推斷動態。

## Grayscale 與 84 × 84 resize：保留控制需要的資訊

`AtariPreprocessing` 會把 RGB 畫面轉成 grayscale，再縮小到 `84 × 84`。這不是把畫面變得「更像答案」，而是移除目前控制任務不一定需要的負擔：

- grayscale 把三個色彩通道縮成一個通道，減少每張 observation 的資料量；
- resize 把原始畫面縮小，讓後面的 convolution 不必在不必要的解析度上計算；
- 空間關係仍然保留，球、球拍和磚塊仍然是畫面中的位置與形狀。

因此，一張 processed observation 的 shape 是：

~~~text
(84, 84)
~~~

這裡特別設定 `grayscale_newaxis=False`，所以不會得到 `(84, 84, 1)`。多出來的單通道軸在這個 pipeline 中沒有必要，下一步直接堆疊四張畫面就能得到 `(4, 84, 84)`。

這裡使用的是 Gymnasium 官方的 `AtariPreprocessing`，resize 依賴 OpenCV；環境因此明確安裝 `opencv-python`，而不是另外維護一套自製的 grayscale 或 resize 實作。

## Frame skip：降低 action decision 的頻率

Agent 不需要在 Atari emulator 的每一個 frame 都重新選一次 action。這次選擇 `RIGHT` 之後，可以讓同一個 action 持續數個 emulator frames，再回來要求 Agent 做下一次決策。

這就是 `frame_skip=4` 的意義：

~~~text
Agent 選擇 RIGHT
        ↓
RIGHT 在 emulator 中持續 4 個 frames
        ↓
環境回傳下一個 processed observation
~~~

它控制的是：

> Agent 多久做一次新的 action decision。

`AtariPreprocessing` 也會在跳過的 frame 中，對最近的畫面做 max-pooling。Atari 某些物件可能因為畫面更新方式，在單一 frame 中短暫消失；把最近畫面取最大值，可以降低重要 sprite flickering 造成的資訊遺失。這個 max-pooling 已經是官方 wrapper 的一部分，不需要在外面再做一次。

reset 時的 `noop_max=30` 是另一個容易混在一起的設定。它表示每一局開始時，可以隨機執行最多 30 個 `NOOP`，讓每一局不必永遠從完全相同的 emulator 時刻開始。這不是 frame skip，也不是 frame stacking；固定 seed 時，這個 reset 行為仍然可以重現。

還有一個重要的 wrapper 邊界：底層 `ALE/Breakout-v5` 必須使用 `frameskip=1`。如果 base environment 自己已經 skip frames，再加上 `AtariPreprocessing(frame_skip=4)`，同一段時間就會被重複跳過。Gymnasium 也會直接拒絕這種設定。因此由 `AtariPreprocessing` 統一負責 frame skip，才能清楚知道每一個 agent step 的語意。

目前也保留 `repeat_action_probability=0.25`，不改變 `ALE/Breakout-v5` 的 sticky-action baseline。`terminal_on_life_loss=False` 則表示掉一條命不會在 preprocessing 階段被重新定義成完整 episode 結束；這是 episode semantics 的選擇，不應悄悄混進影像處理。

## Frame stacking：把短期時間資訊交給 Agent

Frame skip 解決的是「多久選一次 action」，但它沒有讓 Agent 同時看見過去的畫面。要讓 Agent 推斷球的方向，還需要把最近幾張 processed observation 放在一起：

~~~text
obs(t-3)
obs(t-2)
obs(t-1)
obs(t)
    ↓
stack
    ↓
(4, 84, 84)
~~~

這就是 `FrameStackObservation(stack_size=4)` 的工作。它控制的是：

> Agent 每次做決策時，可以同時看到多少時間歷史。

兩個設定雖然都使用數字 4，意義卻完全不同：

~~~text
frame skip = 4
  同一個 action 持續幾個 emulator frames

frame stack = 4
  一次 observation 保留幾張 processed frames
~~~

Frame stacking 並不是先替 Agent 算出 `(vx, vy)`，再把速度數值塞進 state。它保留的是一段短時間的像素序列，讓之後的 neural network 可以從相鄰畫面的差異學出球的移動方向。

reset 時，stack 裡還沒有四張不同的歷史畫面。`FrameStackObservation` 預設使用 `padding_type="reset"`，因此會用 reset 時的第一張 processed observation 填滿：

~~~text
reset:
[f0, f0, f0, f0]

step 1:
[f0, f0, f0, f1]

step 2:
[f0, f0, f1, f2]
~~~

這個 padding 讓每一次 reset 都有固定的 shape，也不會把上一局最後幾張畫面帶進下一局。

## 為什麼 shape 是 `(4, 84, 84)`？

最後的第一個維度不是 RGB channel，而是四個時間上的 grayscale observations：

~~~text
4 個時間切片 × 84 高 × 84 寬
~~~

`FrameStackObservation` 會把 stack 軸放在第一個位置，因此這裡是 `(4, 84, 84)`，不是 `(84, 84, 4)`。後者是 channel-last 的排列；前者則是四個時間切片放在 leading axis，扮演後續 CNN 輸入中 channel-like 平面的角色，後兩個維度代表高度與寬度。此時還沒有建立 PyTorch tensor；我們先在 environment 邊界固定資料的 shape，之後 model pipeline 再決定何時轉成 tensor。

## 為什麼先保持 `uint8`？

這次設定 `scale_obs=False`，所以 observation 仍然是 `uint8`，數值範圍是 `0..255`，而不是在 wrapper 中直接變成 `float32`。

這是 Replay Buffer 的工程選擇。每個 `uint8` 像素使用 1 byte；如果一開始就存成 `float32`，同樣的像素會使用 4 bytes。當 buffer 儲存大量 `(4, 84, 84)` observations 時，差異會直接反映在記憶體用量上。

等到真正送入模型之前，再把資料轉成 `float32` 並除以 `255.0`，就能把數值正規化與儲存格式分開處理。Environment 的 contract 因此保持輕量，也和之後使用 `uint8` replay buffer 的方向一致。

## 實際看一次三階段輸出

用下面的命令，以固定 seed 執行 8 個 processed steps：

~~~bash
python inspect_preprocessing.py --steps 8 --seed 42
~~~

實際看到的 observation contract 是：

~~~text
Raw Atari observation
  shape : (210, 160, 3)
  dtype : uint8

After AtariPreprocessing
  shape : (84, 84)
  dtype : uint8

After FrameStackObservation(stack_size=4)
  shape : (4, 84, 84)
  dtype : uint8
  min   : 0
  max   : 148
~~~

接著隨機 Agent 仍然使用原本的四個 Breakout actions；processed environment 只改變 observation，不改變 action semantics：

~~~text
Step 0
  action       : 0 (NOOP)
  stacked_obs  : shape=(4, 84, 84), dtype=uint8
  reward       : 0.0
  terminated   : False
  truncated    : False

Step 1
  action       : 3 (LEFT)
  stacked_obs  : shape=(4, 84, 84), dtype=uint8
  reward       : 0.0
  terminated   : False
  truncated    : False

Processed steps: 8
Episode return: 0.0
Episodes ended: 0
~~~

這裡的重點不是這 8 步得到多少分，而是每次 `step()` 都維持相同的 observation contract，同時仍然回傳 reward、`terminated` 和 `truncated`。這使得 Day 3 的 transition 資料邊界可以延伸到 processed observation，而不必重新發明一套互動介面。

## 把 wrapper chain 收斂成一個 contract

整條 pipeline 的核心設定可以濃縮成下面這段：

~~~python
env = gym.make(
    "ALE/Breakout-v5",
    frameskip=1,
    repeat_action_probability=0.25,
)
env = AtariPreprocessing(
    env,
    frame_skip=4,
    screen_size=84,
    grayscale_obs=True,
    grayscale_newaxis=False,
    scale_obs=False,
)
env = FrameStackObservation(env, stack_size=4)
~~~

這段程式碼要讓我們看見的不是 API 參數清單，而是每一層的責任邊界：base environment 提供 Atari interaction；`AtariPreprocessing` 負責 frame skip、max-pooling、grayscale 和 resize；`FrameStackObservation` 負責保留短期歷史。正式使用時，這條 chain 由同一個 environment factory 建立，避免 inspection、測試和未來 training loop 各自長出不同版本。

## 今天真正建立的是什麼？

Day 4 結束後，Agent 每次看到的已經不再是未處理的 RGB 畫面，而是一個明確的輸入邊界：

~~~text
raw observation       = (210, 160, 3) / uint8
processed observation = (84, 84) / uint8
agent observation     = (4, 84, 84) / uint8
~~~

grayscale 和 resize 減少了不必要的影像負擔；frame skip 降低了 action decision 的頻率；max-pooling 處理 Atari 畫面更新可能造成的 flickering；frame stacking 則把短期時間資訊留給 Agent 自己學習。最後維持 `uint8`，讓資料在大量儲存時保持合理的記憶體成本。

現在 Agent 已經拿到一個比單張 RGB frame 更完整的 observation，但還有一個更深的問題：它要怎麼判斷 `LEFT`、`RIGHT`、`NOOP`、`FIRE` 哪個 action 對未來比較好？

下一篇會從 **MDP 與 Bellman Equation** 開始，討論狀態、動作與長期回饋之間的關係。
