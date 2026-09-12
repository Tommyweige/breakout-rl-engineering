# Day 29｜讓 RL Agent 真的在瀏覽器裡玩 Breakout

Day 27、Day 28 已經證明一件事：訓練好的 ONNX 模型可以直接放進瀏覽器，用 WASM 或 WebGPU 執行，而且對固定畫面會做出和原本模型一致的決策。

但這還不等於「Agent 已經能在網頁裡玩遊戲」。

真正的 Browser RL Demo 還需要把整條流程接起來：

```text
Breakout 產生畫面
→ 整理成模型訓練時看過的輸入
→ 模型替四個動作打分
→ 選出動作
→ 把動作送回 Breakout
→ 遊戲繼續往前跑
```

Day 29 做的就是這件事。

現在同一個頁面裡有兩個真正的 Breakout：左邊由玩家控制，右邊由 RL Agent 自己玩。兩邊各自擁有獨立的 ALE 遊戲環境，所以球的位置、生命、分數與遊戲狀態都互不影響。

[![Human 與 RL Agent 在瀏覽器裡各自執行 Breakout](https://github.com/Tommyweige/breakout-rl-engineering/blob/4bed546/assets/day29/browser-ai-demo.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/4bed546/assets/day29/browser-ai-demo.png)


## 把模型放進瀏覽器，真正困難的是「環境也要一樣」

模型權重一樣，只能保證「同一份輸入送進模型時，模型會算出一樣的答案」。

但 RL 跟一般影像分類不太一樣。

Agent 的下一個輸入，是由「上一個動作 + 遊戲環境」共同產生的。所以只要瀏覽器裡的遊戲初始化方式、畫面處理方式或動作編號跟訓練時不一樣，之後看到的畫面就會一路偏掉。

因此 Browser 版 Breakout 不只是要載入同一份 ONNX，還要盡量重現訓練時的 Atari preprocessing。

這次正式對齊的重點包括：

- 每次開始 episode 時的隨機 NOOP；
- 遊戲本身使用的亂數 seed；
- 一次決策推進 4 個 Atari frame；
- 最後兩張 frame 的合併方式；
- 灰階與 84×84 resize；
- 連續 4 張畫面的 frame stack；
- `NOOP / FIRE / RIGHT / LEFT` 和 ALE 真正動作編號的對應；
- 開球與掉命後的 FIRE 行為。

這些看起來像小細節，但對 RL 來說，它們其實就是模型的一部分「輸入規格」。

## Episode 開始前，為什麼還要先 NOOP？

模型訓練時使用 Gymnasium 的 Atari preprocessing。

每次 reset 後，不會永遠從完全相同的第一個畫面開始，而是會先隨機執行 `1～30` 次 NOOP，也就是讓遊戲往前跑一小段時間，但不移動 paddle。

可以把它理解成：

```text
reset
→ 隨機等待 1～30 個 raw steps
→ 建立第一個 observation
→ Agent 開始決策
```

這樣做可以讓不同 episode 的起始狀態稍微不同，而不是每次都從同一個時間點開始。

正式 Browser evaluation 也沿用相同的 NOOP 數量與 ALE seed，確保同一個 evaluation seed 在 Python 與 Browser 中代表的是同一條遊戲隨機路徑。

這裡有一個很容易誤會的地方：**episode seed 一樣，不代表直接把同一個整數丟給 ALE 就會得到一樣的遊戲。**

Gymnasium 會從 episode seed 再產生不同用途的亂數，其中一部分決定 reset 時 NOOP 幾次，另一部分才交給 ALE。因此正式評估要對齊的是實際使用到的亂數，而不是只比最外層那個 seed 數字。

## 一次決策其實看的是 4 張畫面

DQN 不能只看一張 Breakout 截圖。

如果只看一張圖，模型知道球「在哪裡」，卻不知道球正在往左、往右、往上還是往下。

所以每次送進模型的 observation 是連續 4 張 `84×84` 灰階畫面疊在一起。

瀏覽器裡每次 Agent 做決定時，流程是：

1. 同一個動作連續送給 ALE 4 個 raw frames。
2. 取得最後兩張 raw frame。
3. 每張先轉成灰階。
4. 對兩張灰階畫面逐像素取較亮值。
5. 縮成 `84×84`。
6. 把最新畫面推進 4-frame stack。
7. 再交給 ONNX 模型。

其中「先灰階，再合併」這個順序也要和 native preprocessing 一樣。

對人眼來說，順序不同可能幾乎看不出差異；但神經網路實際收到的是像素數值，因此部署時最好直接驗證數值，而不是只看畫面長得像不像。

## 怎麼證明 Browser 看到的畫面真的一樣？

這次沒有只檢查尺寸是不是 `84×84`，而是直接拿相同 Atari raw frames，分別經過 Python 與 Browser preprocessing，再逐像素比較。

測試涵蓋 seeds `101`、`202`、`303` 的 reset 與後續動作。

結果如下：

| 比較內容 | 最大像素差 | 平均像素差 | 不同像素比例 |
| --- | ---: | ---: | ---: |
| Reset 與 step 的灰階畫面 | `0` | `0` | `0` |
| 最後兩張灰階畫面的合併結果 | `0` | `0` | `0` |
| 84×84 處理後畫面 | `0` | `0` | `0` |
| 4-frame stack | `0` | `0` | `0` |

也就是這批實際測試中：

> **Python 與 Browser 最後送進模型的像素完全一致。**

[![Python 與 Browser preprocessing 的逐像素比較](https://github.com/Tommyweige/breakout-rl-engineering/blob/4bed546/assets/day29/preprocessing-parity-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/4bed546/assets/day29/preprocessing-parity-comparison.png)

這比「Browser 看起來也有灰階畫面」更有意義，因為 DQN 真正吃進去的就是這些數字。

## 模型說 RIGHT，ALE 收到的是哪個數字？

模型輸出的四個位置固定代表：

```text
0 → NOOP
1 → FIRE
2 → RIGHT
3 → LEFT
```

但 ALE 自己的動作代碼不是模型的 index。

在 Breakout 的 minimal action set 裡，實際對應是：

```text
NOOP  → 0
FIRE  → 1
RIGHT → 3
LEFT  → 4
```

所以 Browser 不能直接把模型輸出的 `2` 或 `3` 原封不動送給 ALE，而是要先知道模型選的是哪個語意動作，再轉成 ALE 對應的 action code。

這也是部署 RL 模型時很容易忽略的一點：

> **模型輸出的 action index，只在訓練時定義的 action space 裡有意義。真正接到新環境時，一定要再次確認 mapping。**

## FIRE 不是全部交給模型自己處理

Breakout 還有一個比較特殊的地方：球不是每次都會自動開始移動。

遊戲剛開始，以及掉一條命之後，都可能需要按一次 `FIRE` 才會重新開球。

這個專案從前面的 evaluation contract 就已經把這件事交給 environment 處理。因此 Browser 也沿用相同邏輯：只有在需要開球時，環境會暫時替 Agent 執行 FIRE；一般遊戲過程仍然由模型自己決定 `NOOP / FIRE / RIGHT / LEFT`。

這樣 Browser gameplay 才和模型之前的訓練、evaluation 條件一致。

## Human 與 Agent 是兩個真的遊戲

頁面左邊和右邊不是同一局遊戲的兩種視角。

它們是兩個獨立 ALE instance：

- Human 端讀取鍵盤；
- Agent 端讀取自己的 Atari 畫面並跑模型；
- 各自有自己的球、磚塊、paddle、生命與分數；
- `Start Both`、`Pause Both`、`Reset Both` 只是一起控制兩邊的節奏。

因此玩家操作左邊時，不會改變右邊 Agent 正在玩的那一局。

這樣 Human vs RL 才是一個真正的並排展示，而不是人類和 Agent 輪流搶同一個 environment。

## 最後還要看完整 episode，而不是只看單一步

固定畫面 parity 能證明模型本身在 Browser 裡沒有算壞；pixel parity 能證明 Browser preprocessing 跟 native 對得上。

最後還需要的是：**整局遊戲跑完之後，結果是不是也能對上。**

因此先挑出 15 個和之前 native evaluation 重疊的 seeds，直接比較完整 episode 分數：

| Seed | Native | WASM | WebGPU |
| ---: | ---: | ---: | ---: |
| 101 | 73 | 73 | 73 |
| 102 | 45 | 45 | 45 |
| 103 | 58 | 58 | 58 |
| 104 | 47 | 47 | 47 |
| 105 | 74 | 74 | 74 |
| 202 | 58 | 58 | 58 |
| 203 | 26 | 26 | 26 |
| 204 | 37 | 37 | 37 |
| 205 | 45 | 45 | 45 |
| 206 | 70 | 70 | 70 |
| 303 | 35 | 35 | 35 |
| 304 | 60 | 60 | 60 |
| 305 | 48 | 48 | 48 |
| 306 | 34 | 34 | 34 |
| 307 | 38 | 38 | 38 |

15 局全部一致。

這代表對這批固定 evaluation seeds 來說，現在不只是模型輸出接近，而是從：

```text
reset
→ Atari 畫面
→ preprocessing
→ 4-frame observation
→ ONNX inference
→ action mapping
→ ALE gameplay
→ episode score
```

整條路徑都能和 native evaluation 對上。

## WASM 與 WebGPU 各跑 30 局

接著再讓兩個 Browser backend 各跑 30 個固定 episodes。

結果如下：

| Backend | 平均分數 | 中位數 | 標準差 | P10 / P90 | 最低 / 最高 | 完成 / Crash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WASM | `41.8667` | `37.5` | `14.8632` | `25.9 / 61` | `15 / 74` | `30 / 0` |
| WebGPU | `41.8667` | `37.5` | `14.8632` | `25.9 / 61` | `15 / 74` | `30 / 0` |

兩邊的 score distribution 完全一致。

[![WASM 與 WebGPU 的 30 局 Breakout 分數分布](https://github.com/Tommyweige/breakout-rl-engineering/blob/4bed546/assets/day29/browser-policy-score-distribution.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/4bed546/assets/day29/browser-policy-score-distribution.png)

這裡得到的結論很清楚：

> **對這個 Agent 來說，WASM 與 WebGPU 雖然執行方式不同，但沒有改變這 30 局的遊戲行為與最終分數。**

至於誰比較快，則是另一個問題。Day 28 已經看到，小型 DQN 在單次推論時不一定能從 GPU 得到優勢；所以部署時還是要把「結果是否一致」和「速度是否更快」分開測。

## Day 29 做完後，Browser Demo 到了哪裡？

到這一步，這個專案已經不只是把模型檔案塞進網頁。

現在瀏覽器裡真的有：

- 一個可由鍵盤操作的 Atari Breakout；
- 一個獨立由 DQN 控制的 Atari Breakout；
- 與訓練環境一致的主要 preprocessing；
- 正確的 action mapping；
- WASM / WebGPU 兩種模型執行方式；
- 真正的多局 Browser evaluation。

而且所有推論都直接發生在使用者的瀏覽器裡，不需要另外架一個 Python inference server。

Day 29 最重要的技術重點不是「終於看到遊戲畫面動起來」，而是：

> **RL 模型要搬到另一個執行環境，不能只搬模型；產生 observation 的整個環境與前處理也必須一起對齊。**

當模型、畫面、seed、動作與遊戲規則都對上之後，Browser 才真正是在執行原本訓練好的 Agent，而不是一個「看起來有在玩」但其實輸入條件已經不同的版本。
