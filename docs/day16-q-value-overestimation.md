# Day 16｜Q-value 為什麼可能被 `max` 高估？

Day 13 的訓練診斷看見一個值得保留的疑問：Deep Q-Network（DQN）的 Q-values 和 target 都可能逐漸上升，但數字變大不等於模型真的更了解 Breakout。這次先不急著把現象歸咎於 overestimation，而是回答一個更小、也更容易驗證的問題：**如果每個 action 的價值估計都有一點誤差，單純取最大值會不會天生偏樂觀？**

這篇先建立那個機制的直覺，再把它和真實 Breakout checkpoint 的 CUDA 輸出分開。前者是可控制的 toy simulation；後者是實際模型觀察，不能互相冒充。

## 先從「每個答案都有誤差」開始

Q-value 可以理解成：在某個畫面下先做一個 action，從現在到遊戲結束大概能拿到多少折扣後的回報。Deep Q-Network（DQN）用神經網路同時估計四個 action 的 Q-values：`NOOP`、`FIRE`、`RIGHT`、`LEFT`。

假設四個 action 的真實價值其實都一樣，都是 `1.0`。模型不可能每次都剛好輸出 `1.0`，所以可以把一次估計寫成：

```text
估計值 = 真實值 + 估計誤差
```

如果誤差平均為零，單看其中一個 action，模型沒有系統性偏高。但 DQN 不會隨便挑一個答案，它會先在四個 noisy estimates 中選最大的那個。令 `s` 代表目前的 state（這裡就是一組畫面），被選中的 action 可以寫成：

```text
a* = argmax_a Q̂(s, a)
```

這裡的 `a*` 是被選中的 action，`Q̂` 是模型目前的估計；公式只是在寫「先找最大估計」。問題出在：四個數字中，最容易成為最大值的，往往正是誤差剛好偏高的那一個。於是 `max` 不只是在找最好的 action，也把正向誤差一起挑了出來。

這就是 overestimation 的核心機制。它不需要模型故意樂觀，也不需要每個 action 都有偏高的誤差；只要 action 數量夠多，而且選擇與評分使用同一批誤差，就可能發生。

## 用實際 simulation 看偏差怎麼長出來

為了只觀察這個機制，我用 NumPy 做了 500,000 次 Monte Carlo simulation，也就是反覆抽樣、重複計算的隨機實驗。四個 action 的真實值固定為 `1.0`，每次替所有 action 加上獨立、平均為零的常態（Gaussian）noise，再比較兩種讀法：

- `Vanilla max`：用同一批 noisy estimates 找最大值，並把那個最大值當成評分。
- `Decoupled estimator`：用第一批 estimates 選 action，再用另一批獨立 noise 評估被選中的 action。

第二種是用來隔離「選擇」和「評分」的概念模型，和 Double DQN 想處理的方向相同；它不是這一天已經實作完成的 Double DQN trainer。

[![四個等價 action 在不同估計噪聲下的真實值、Vanilla max 與 decoupled estimator](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/40e8ba4f348016bc89f4b5dbce587e4228f8ef57/assets/day16/overestimation-bias.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/40e8ba4f348016bc89f4b5dbce587e4228f8ef57/assets/day16/overestimation-bias.png)

左圖的橫軸是每個 action 估計誤差的標準差，縱軸是平均估計值；藍線是真實最佳值 `1.0`，橘線是同一批 estimates 取出的最大值，綠線是用獨立第二批 noise 評估的結果。右圖把縱軸改成「平均估計值 − 真實值」，這個差值就是平均偏差（bias），所以零線代表沒有平均偏差。

最值得注意的是 noise standard deviation `1.0` 的實測點：`Vanilla max` 平均為 `2.0294`，bias 是 `+1.0294`；`Decoupled estimator` 平均為 `1.0002`，bias 約 `+0.0002`。這個差距不是手工畫出的理想曲線，而是從保存的 simulation output 算出來的。

圖因此支持一個很精確的結論：**同一個 noisy estimate 同時負責選 action 和提供 value，取最大值會產生選擇性偏高。** 但它不能支持「Breakout 的 Q-value 一定高估了多少」，因為這裡沒有真實 Breakout trajectory，也沒有知道每個畫面理論上真正的最佳 Q-value（Q-star）與可查答案的參照（oracle）。

## 真實 Breakout 的 Q-values 要另外看

toy simulation 只告訴我們一個可能的數值機制。為了不把概念實驗冒充成遊戲證據，我另外用 fresh N=4、100K Contract v2 checkpoint，在 NVIDIA CUDA（NVIDIA GPU 的計算執行環境）上對真實 Breakout observations 做了 80 個 probe states，也就是從固定遊戲畫面取出的測試 state，並執行 model forward（把 observation 送進網路取得輸出）。這個 diagnostics 使用 `torch.no_grad()`，並保存 checkpoint SHA-256、GPU 型號、PyTorch/CUDA 版本與 Contract v2 metadata。

實測的每個 action 平均 Q-value 是：

| Action | 平均 Q-value | 標準差 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| `NOOP` | 2.0271 | 0.2137 | 1.7347 | 2.3854 |
| `FIRE` | 2.0769 | 0.2033 | 1.7416 | 2.4222 |
| `RIGHT` | 2.0394 | 0.2162 | 1.7451 | 2.4190 |
| `LEFT` | 2.0160 | 0.1950 | 1.7577 | 2.3373 |

在每個 state 先取最大 action 後，`max Q` 的平均是 `2.0819`，top-two action gap 的平均是 `0.0387`。這些數字告訴我們模型目前在這批真實畫面上輸出了什麼；它們沒有告訴我們正確答案應該是多少。因此目前只能把這份 [CUDA Q-value diagnostics](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/40e8ba4f348016bc89f4b5dbce587e4228f8ef57/assets/day16/q-value-diagnostics.json) 稱為 exploratory evidence，不能把「Q-value 大於零」或「max 比 action mean 高」直接寫成 Breakout overestimation 已被證明。

這個區分很重要：CPU toy simulation 證明的是選擇機制；CUDA probe 顯示的是目前 checkpoint 的真實輸出；要量出 Breakout 的實際 overestimation，還需要可靠的 value reference 或額外的受控比較。

## 這個機制為什麼會連到 Double DQN

Vanilla DQN 的 target 需要下一個 state 的最大 Q-value。若 online network，也就是目前拿來選 action 的網路，同時負責「挑哪個 action」和「這個 action 值多少」，上面的選擇性偏差就可能進入 target，再透過 bootstrapping（用目前估計的下一個 state value 當作學習目標）影響後續更新。

Double DQN 的核心想法是拆開兩個角色：一個網路選 action，另一個網路評估被選中的 action。這不保證每個 target 都更準，也不會自動解決資料、環境或訓練 budget 問題；它只是減少「同一份誤差被拿來選最大、又拿來替最大值背書」的機會。

Day 16 因此留下兩個邊界。第一，vectorized training 已經把資料收集與 GPU batch 做得更有效率，但不應把 systems throughput 當成演算法品質。第二，overestimation 的 toy mechanism 已經可以重現，真實 checkpoint 的 Q-values 也已經能在 CUDA 上保存；至於 Double DQN 是否真的改善學習，應在下一個受控的演算法比較中回答。

現在可以帶著一個更準確的 mental model 進入下一步：**`max` 不只是選出目前最大的數字，它也會選出最可能被正向誤差推高的數字。**
