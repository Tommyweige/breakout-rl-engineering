# Q-value 為什麼會被 `max` 拉高？從估計誤差走到 Double DQN

Day 16 已經把 Breakout 的環境規則和訓練資料流固定下來，也選出 N=2 的向量化 GPU Replay backend（把 replay 資料放在 GPU 上供訓練抽樣）。現在問題換了：即使資料流沒有 bug，DQN 用來表示「現在做某個動作有多值得」的 Q-value，仍然只是模型的估計，不是真正可以直接查表的答案。

這個差別在 target 計算裡會被放大。Vanilla DQN 會在下一個 state 的多個估計中取最大值；只要估計帶有一點誤差，這個選擇就可能偏愛「剛好被高估」的 action。這篇文章要回答的不是「Double DQN 一定比較強嗎」，而是更前面的問題：**為什麼 `max` 會製造這個偏差，以及 Double DQN 到底把哪兩個工作拆開？**

## 一個單獨看起來沒問題的估計，為什麼合在一起會偏高？

先不碰 Breakout，也不假設神經網路的最佳 Q-value 可以取得。我建立一個最小的 toy experiment：四個 action 的真實價值都設成 `0`，每次估計都加上平均為 `0` 的隨機 noise。單獨看所有估計值，它們的平均應該接近 `0`；但如果每次都挑四個估計裡最大的那個，結果會怎樣？

以下是 seed `42`、每個 noise scale 各 `100,000` trials 的實際輸出。`single estimate mean` 是所有 action 估計混在一起的平均，`vanilla max mean` 是每次挑最大值後再平均，`decoupled mean` 則是用估計 A 選 action、再用獨立的估計 B 評估該 action。

| noise std | single estimate mean | vanilla max mean | decoupled mean |
|---:|---:|---:|---:|
| 0.1 | 0.000003 | 0.102909 | 0.000550 |
| 0.5 | 0.000015 | 0.514543 | 0.002750 |
| 1.0 | 0.000029 | 1.029086 | 0.005499 |

真實值在三組實驗中都是 `0`。單一估計的平均確實接近真實值，但 `vanilla max mean` 隨 noise 增加而明顯高於 `0`。這就是這裡所說的 overestimation bias：**不是每個 Q-value 都一定被高估，而是「選最大值」這個規則的平均結果偏向高估。**

[![四個 action 的估計加入不同 noise 後，max selection 的平均值偏離真實值](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/overestimation-bias.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/overestimation-bias.png)

圖的橫軸是 noise 的標準差，縱軸是估計值的平均。藍線的單一估計和黑色虛線真實值幾乎重疊；橘線則是每次先取最大值再平均，所以被推到真實值上方。綠線把「選 action」和「評估 value」分開後，平均值回到接近 `0`。這個實驗證明的是 max 加上估計誤差會產生偏差；它沒有測量 Breakout 神經網路的真實偏差大小。

## max 不是中立的選擇器

關鍵不在於 noise 是否有正的平均。假設某一輪四個估計是：

```text
[-0.8, 0.2, -0.1, 0.4]
```

它們的誤差可能整體偏低，但 `max` 還是會選 `0.4`。另一輪如果四個誤差剛好是：

```text
[-0.2, 0.1, 0.3, 1.1]
```

`max` 就會選到 `1.1`。在大量輪次中，越多 action 可以比較，就越有機會出現一個特別大的正誤差；max 會把這個正誤差保留下來，卻不會對稱地保留「所有值都偏低」的情況。

因此，「估計器平均無偏」和「估計器取最大值後無偏」是兩個不同命題。前者在 toy experiment 裡成立，後者不成立。這也是為什麼不能看到一個很大的 Q-value，就直接說模型知道那個 action 的真實價值很高。

## Vanilla DQN 的 target 把這個選擇寫進去了

在一次 DQN update 裡，模型先用目前的 online network 估計目前 state 的 Q-value，再拿下一個 state 的估計建立學習目標。target network 是一份暫時固定的網路，用來讓這個目標不要在每個 optimizer step 都跟著目前網路一起漂移；Day 11 已經介紹過它的同步角色。

Vanilla DQN 在下一個 state 做的是：把 target network 輸出的所有 action value 放在一起，直接取最大值。若 `reward` 是這一步得到的回饋、`gamma` 是未來回饋的折扣率，而 `terminated` 表示遊戲本身真的結束，target 可以寫成：

```text
target = reward + gamma × (1 - terminated) × max_a Q_target(next_state, a)
```

這裡的 `max` 就是前面 toy experiment 的選擇器。它同時負責兩件事：決定哪個 action 看起來最好，也使用同一個估計器讀出那個 action 的 value。當 Q estimates 有 noise 時，selection 和 evaluation 會共享同一個「幸運的高估」。

把下一個 state 的估計接回目前 target，這個動作叫做 bootstrap；公式中的 `(1 - terminated)` 就是控制這個動作是否發生的 mask。`truncated` 在這裡仍然和 `terminated` 分開。Contract v2 把 `terminated=True` 作為不 bootstrap 的訊號；時間限制造成的 `truncated=True` 不會被自動改寫成 terminated。這個語意和 Day 16 相同，不能因為換 Double DQN 就重新合併成模糊的 `done`。

## Double DQN 改的是角色分工，不是輸出介面

Double Deep Q-Network（Double DQN）沒有新增另一種 observation，也沒有把四個 action 改成別的數量。它重新分配的是兩個既有網路的工作：

- online network 只負責從 `next_state` 選出看起來最好的 action；
- target network 再負責評估這個已經選定的 action。

`argmax` 可以先理解成「回傳最大值所在的 action index」，`gather` 則是「依照這個 index，從另一組 Q-values 取出對應欄位」。實際 target branch 的核心因此是：

```python
with torch.no_grad():
    online_next_q = online_network(next_states)
    next_actions = online_next_q.argmax(dim=1)

    target_next_q = target_network(next_states)
    next_values = target_next_q.gather(
        1, next_actions[:, None]
    ).squeeze(1)

target = rewards + gamma * (~terminated) * next_values
```

`with torch.no_grad()` 的原因是這段 forward 只在建立 target，不是要更新 next-state branch 的參數；真正需要 backward 的仍然是目前 state 上由 online network 產生、再依 replay action 選出的 prediction Q-value。這樣既保留 current-state online gradient，也不讓 target 分支把梯度接回 update。

這裡的「Double」也不是兩個完全獨立、各自學一套世界模型的意思。Vanilla DQN 原本就有 online network 和 target network；Double DQN 改的是 `next_state` 上的使用方式：**一個網路選，一個網路評估。**

下面的流程圖把這個資料關係和 vanilla 分支並列。它是根據實際 target implementation 整理的結構圖，不是某次 Breakout rollout 的數值紀錄。

[![Vanilla DQN 與 Double DQN 的 next-state target 計算流程](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/double-dqn-target-flow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/double-dqn-target-flow.png)

讀圖時可以只追兩條線：Vanilla 把 target network 的最大值直接送進 target；Double DQN 先從 online network 得到 `a*`，再讓 target network 只評估 `Q_target(next_state, a*)`。兩者最後都還要合併 reward 和 bootstrap mask。

## 讓兩種 target 在同一個 next state 上分開現形

為了避免「剛把 online 權重完整複製給 target、兩個 network 輸出一樣」而看不出差異，我使用一個刻意不同的 synthetic fixture（人工構造、只用來測試計算路徑的輸入）。online 輸出是：

```text
[1, 5, 2, 0]
```

target 輸出是：

```text
[4, 3, 2, 1]
```

令 `reward=1`、`gamma=0.5`，Vanilla DQN 會從 target 的最大值 `4` 得到 `1 + 0.5 × 4 = 3.0`。Double DQN 則由 online 選出 action `1`，再從 target 取 action `1` 的 value `3`，所以 target 是 `1 + 0.5 × 3 = 2.5`。

[![同一個 crafted fixture 下 Vanilla 與 Double DQN 的 evaluated value 和最終 target](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/dqn-vs-double-targets.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/dqn-vs-double-targets.png)

左圖比較的是 next-state value：Vanilla 取 target 的 `4.0`，Double DQN 評估 online 選出的 action `1`，因此使用 `3.0`。右圖再把相同的 reward 和 gamma 套回去，得到 `3.0` 與 `2.5`。這張圖的數字全部來自同一個 inspection fixture 的實際 target function；它用來展示演算法差異，不代表 Breakout 的平均回報或最佳策略下的真實 Q-value。

## 回到 Breakout：先固定 probe，再看實際 Q-values

toy experiment 裡的兩個 estimator 是獨立產生的，真實的 online/target network 卻來自同一條 training history，因此不能把 toy 的 bias 數字直接貼到 Breakout 上。要觀察真實模型，我先在 Contract v2 下固定一批 probe states，也就是之後會反覆餵給模型的 observations：使用 15 個 concrete seeds，每個 seed 取 4 張 observation；狀態形狀是 `(4, 84, 84)`，每個像素以 0–255 的整數資料型別 `uint8` 保存，由 seeded random requested actions 產生，環境仍保有 Contract v2 的 mandatory serve FIRE。

這批 states 只做 diagnostics，不參與訓練，也不因為換 DQN family 而換一批。對 Day 17 smoke checkpoint 做 no-grad inference 後，60 個 probe 中有 41 個選到 `RIGHT`、18 個選到 `FIRE`、1 個選到 `LEFT`；Q-value 的整體平均是 `0.193705`，標準差是 `0.104216`，每個 probe 的最大 Q-value 平均是 `0.206222`。

[![Day 17 smoke checkpoint 在固定 Contract v2 probes 上的 Q-value 分布與 greedy action](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/q-probe-summary.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/q-probe-summary.png)

左圖的四組 boxplot 是 `NOOP`、`FIRE`、`RIGHT`、`LEFT` 在 60 個固定 states 上的實際輸出；右圖則統計每個 state 的 argmax action。它能告訴我們這個 checkpoint 在這批輸入上偏好什麼、Q-values 分布多寬，卻不能告訴我們最佳策略下的真實 Q-value 是多少，也不能只靠「大多數選 RIGHT」就宣稱模型已經學會或完全沒學會 Breakout。這正是固定 probe 的用途：提供可重複的觀察面，而不是偽造真實答案（ground truth）。

## 只改 target rule 的 smoke training

最後把 `algorithm` 放進同一個 `DQNConfig`，讓單一 trainer 依照 config 選 Vanilla 或 Double target；沒有複製第二套 environment loop、Replay、evaluation 或 checkpoint infrastructure。Day 17 的 canonical config 沿用 Day 16 selected backend：Contract v2、N=2、GPU Replay、float32（32-bit 浮點數）、batch size 32、learning starts 1,000、train frequency 4、target sync interval 500、CPU threads 2，唯一主要演算法變因是 `dqn` 或 `double_dqn`。

我在同一台 NVIDIA GeForce RTX 4060 Laptop GPU 上各跑 10,000 transitions，seed 都是 `42`。這是 systems smoke 與 performance regression，不是 Day 18 的 model-quality comparison；其中一次 optimizer update 指的是用一批 Replay 資料完成一次 backward 與參數更新：

| target rule | optimizer updates | transitions/s（每秒 transition） | optimizer updates/s | target forward GPU seconds | peak VRAM（最高顯存） |
|---|---:|---:|---:|---:|---:|
| DQN | 2,251 | 248.62 | 55.96 | 2.48 | 639,140,864 bytes |
| Double DQN | 2,251 | 237.81 | 53.53 | 4.25 | 639,140,864 bytes |

[![同一 N=2 GPU smoke config 下 DQN 與 Double DQN 的實際吞吐與 target-forward 成本](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/smoke-performance.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/1464032ac877d0de02481d6d8490be6534ead2ff/assets/day17/smoke-performance.png)

結果符合機制預期：Double DQN 的 next-state target branch 多做一次 online forward，所以 `target_forward` 的 GPU 累計時間較高，整體 transitions/s 約低 4.4%。但 replay sampling、backward、optimizer step、target sync 都確實跑過，兩邊的 checkpoint metadata 也保存了 `algorithm`、`architecture=standard`、`num_envs=2`、replay backend、training steps，以及 GPU/CUDA/runtime 資訊。這表示新的 target rule 接上了既有 pipeline，也沒有因為追速度而偷偷改 batch、environment count 或 precision。

## 這次學到的是機制，不是「Double DQN 已經贏了」

Day 17 的證據鏈可以收斂成四步：單一 Q estimate 可以平均無偏；`max` 會選中正誤差，形成 overestimation bias；Double DQN 把 action selection 與 value evaluation 分開；同一套 N=2 Contract v2 trainer 可以用 config 切換 target rule，並在真實 CUDA smoke 裡完成 replay、backward、optimizer step 和 target sync。

但 Double DQN 不保證完全消除 overestimation。真實 online/target network 並非 toy experiment 中彼此獨立的 estimator，剩餘誤差也會受到資料分布、同步週期與訓練穩定性影響。這次 smoke 的吞吐差異更不能解讀成哪個演算法的遊戲策略比較好。

因此下一個公平問題要留到 Day 18：固定同一套 Contract v2、N=2 backend、training budget、training seeds 和 Day 15 evaluation protocol，才比較 DQN 與 Double DQN 的學習結果。Day 17 先把「為什麼要比較」和「比較時到底改了什麼」弄清楚。
