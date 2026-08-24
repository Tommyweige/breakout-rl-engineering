# Day 13｜除錯不穩定的 RL 訓練：從 sanity check 到 training diagnostics

Day 12 的深度 Q 網路（Deep Q-Network，DQN）已經能和 Breakout 互動、累積 replay buffer（保存過去 transition，也就是 state、action、reward、next state 一次互動紀錄的資料池）、更新 Online Network（目前會被 optimizer 修改的網路），也會保存 checkpoint。可是「程式沒有 crash」和「Agent 正在變強」是兩件完全不同的事。

正式 debug run 在 `cuda:0`（NVIDIA GeForce RTX 4060 Laptop GPU）上用固定 seed `42` 跑了 `10,000` 個 environment steps（Agent 和環境互動 10,000 次）。流程完成了 `43` 個 episodes、`2,251` 次 optimizer updates（optimizer 是依照梯度修改模型參數的機制），以及 `21` 次 Target Network sync；Target Network 是暫時固定、用來產生 Bellman target（這次更新用來比較的參考值）的參考網路。可是 raw episode return（每局把原始 reward 加總）的平均值只有 `1.77`，第一局是 `2`、最後一局是 `0`。這組數字能證明 GPU training loop 有在工作，不能證明 policy 已經學會 Breakout。

真正要解決的問題是：**當強化學習（Reinforcement Learning，RL）訓練「看起來正常」但行為沒有改善時，如何把模糊的懷疑拆成可以逐一驗證的問題？**

## RL 的錯誤，常常不會讓程式停止

監督式學習通常有一份固定資料和明確標籤；只要輸入、標籤或 loss 的形狀錯了，錯誤比較容易在一次計算中暴露。RL 不一樣：資料是 Agent 自己和環境互動產生的，下一批資料又會受到目前 policy 影響；DQN 的 target 也會隨著模型和 Target Network 的同步而改變。

因此下面這些情況都可能「正常跑完」：

- action index 有效，但永遠只選同一個 action；
- replay 裡有資料，卻沒有真的執行 optimizer update；
- prediction 和 target 的 shape 都正確，但 `gather` 選錯了 action；
- loss（prediction 和 target 差距的摘要）是 finite，卻只是把目前收集到的偏斜資料擬合得更好；
- Q-value（模型對 action 價值的估計）逐步變大，最後變成不合理的數字，但還沒有 NaN。

所以除錯的第一個原則不是「先改 learning rate」，而是先確認 correctness：資料、target、梯度和更新順序真的符合我們以為的意義。

## 一條比改超參數更可靠的除錯順序

這裡的 finite 意思是數值既不是 NaN（不是數字的特殊值），也不是正負 infinity。它是最便宜的檢查，因為一旦 loss、Q-value、target 或 gradient 已經不是有限數字，後面的 return 曲線就沒有解釋價值。

接著才做固定 batch 的 sanity check：把資料和 target 都固定，確認模型至少有能力把一個小問題學會。只有這個檢查通過，才值得往 replay、探索比例、training curves 和 hyperparameters 追。

下面這張圖是依照實際 checklist 和 training diagnostics 的判讀順序整理出的概念流程。gradient norm（把各個參數的梯度合成一個大小）和 TD error（prediction 與 target 的差距）都只是診斷訊號；圖不是某一次 run 的時間軸，而是遇到「不學」時應該如何縮小問題範圍。

[![從 finite checks 到固定 batch、探索訊號和最後超參數調整的 RL 除錯流程](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/debugging-workflow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/debugging-workflow.png)

這個順序的重點在分支：finite check 失敗時，先定位 NaN 或 infinity 的來源；固定 batch 無法下降時，先查 forward、action gather、target shape 和 optimizer；兩者都通過後，才有資格把注意力移到 Agent 的行為。

## 固定 batch 能不能 overfit，是最小的學習能力測試

固定 batch overfit 的意思是：從一小批 transition 取出資料，固定它們對應的 target，連續做多次 optimizer update。這裡的 target 不會在每次更新後重算，所以模型面對的是一個不會移動的參考答案。

這個測試使用和 DQN 相同的資料邊界：輸入是 `(B, 4, 84, 84)` 的 normalized tensor（把像素值轉成模型計算用的浮點數），輸出是 `(B, 4)`，每一列代表四個 Breakout actions 的 Q-values。每一筆資料再用 action index 選出真正要比較的那一格：

```text
Q(states)                  → (B, 4)
固定的 action              → (B,)
gather 選出的 Q(s, a)      → (B,)
固定的 target              → (B,)
```

在 CUDA、seed `42`、`200` 次更新的實際結果中，loss 從 `0.2029491961` 降到 `1.456155e-9`，下降比例約 `99.999999%`。執行方式是：

```text
python debug_overfit_batch.py --updates 200 --device cuda --seed 42
```

這個結果支持的結論很具體：CUDA tensor 上的模型 forward、action selection、loss、backward 和 optimizer 至少能在固定 target 上形成學習閉環。如果這個測試失敗，應該先檢查這些 correctness 問題，而不是把責任推給 Atari 的探索難度。

它仍然不是 Breakout 成績保證。固定 batch 是一個刻意縮小的問題；它沒有測試 replay 分布是否健康，也沒有測試 policy 在新畫面上選出的 action 是否有效。

## Random baseline 先回答「完全沒學習時會怎樣」

接下來要判讀 return，就需要一條沒有 optimizer 的參考線。Random baseline 是使用同一套 Breakout preprocessing 和 episode semantics，只用固定 seed 產生隨機 action；它不是正式的模型 evaluation，也不是 Day 15 的比較協議。

seed `42` 的 5 個 random episodes 實際得到 returns `[2, 1, 5, 1, 5]`，平均 `2.8`，episode lengths 則是 `[235, 160, 319, 215, 317]`。這些數字的用途不是宣布誰比較好，而是提醒我們：短短幾局的 return 變化本來就很吵。若沒有固定 episode 數量、seed 和評估規則，直接拿一個 training episode 和另一個 baseline episode 比高低，很容易把隨機波動誤認成學習。

訓練期間還要看 action distribution。這次 10K-step CUDA run 的累積 action counts 是：

| action | 次數 |
| --- | ---: |
| NOOP | 2,846 |
| FIRE | 2,242 |
| RIGHT | 2,870 |
| LEFT | 2,042 |

四個 action 都有出現，代表 action mapping 和探索並非完全失效。random decisions 有 `4,754` 次、greedy decisions 有 `5,246` 次，random ratio 是 `0.4754`；同一份 run 的 epsilon 從 `0.9` 下降到約 `0.0501`。這能支持「探索排程和決策來源有被記錄」的結論，但不能支持「greedy action 已經是好策略」。

## 只看 loss，會漏掉真正的問題

loss 是目前 prediction 和 Bellman target 的差距摘要。它回答的是「模型對這批 replay target 的擬合程度」，不是「Agent 的動作能不能拿到更多分數」。所以 Day 13 將 return、loss、Q-value 和 gradient norm 分開畫出來，讓每一條訊號回到自己的問題。

### Return：沒有上升趨勢，不能宣稱學會

下圖的 x 軸是 environment step，y 軸是每局完成時累積的 raw episode return。這次 run 共完成 43 局，return 介於 `0` 和 `10`，平均 `1.77`；第一局為 `2`，最後一局為 `0`。

[![10K-step CUDA debug run 的 raw episode return 曲線](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/return-curve.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/return-curve.png)

要注意的是，曲線中間偶爾出現 `6` 或 `10`，不等於已經建立穩定策略；最後沒有比第一局更高，也不等於模型一定完全沒學到。這張圖最多只能說明在這個短 run 和這個 seed 下，還看不到可靠的 return 改善證據。

### Loss：finite 是必要條件，不是成功證明

這次有 `2,251` 個 finite loss samples，平均約 `0.00357`，最大值約 `0.0479`。曲線大多落在低值附近，但偶爾會出現尖峰。

[![10K-step CUDA debug run 的 Huber loss 曲線，尖峰代表部分 batch 的誤差較大](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/loss-curve.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/loss-curve.png)

這裡有一個很實用的判讀界線：本次 run 沒有 non-finite loss，所以沒有看到數值直接爆掉的異常；但尖峰仍然值得和 TD error、reward、replay sample 一起追。不能因為曲線最後有小數字，就把它解釋成 policy 已經變好。

### Q-value 與 target：看相對尺度和趨勢，不設武斷門檻

Q-value 是模型對「在這個 state 做某個 action 有多值得」的估計；target 則是這次更新用來比較的參考值。圖中 selected Q 的 mean、max、min，以及 target mean、max 都來自真實 optimizer update。

[![10K-step CUDA debug run 的 selected Q-values 與 Bellman targets](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/q-values.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/q-values.png)

這次 selected Q mean 大致從接近 `0` 逐步升到約 `0.32`，target max 的最高觀測值約 `1.31`，而所有值都保持 finite。這比較像是「需要繼續觀察尺度」的正常診斷訊號，不足以宣稱 Q-value 合理或不合理。真正的 warning 是相對於 reward 和 target 的尺度快速失控，或 non-finite checks 失敗；不是某個跨所有環境都適用的固定數字。

### Gradient norm：看更新是否有訊號，也看是否逐步失控

gradient norm 是所有參數梯度合成的 L2 大小。它不是分數，而是「這一次 optimizer update 想把參數推動多大」的摘要。

[![10K-step CUDA debug run 的 gradient norm 曲線，數值取 clipping 前的總 norm](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/gradient-norm.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/9ebf324a721688b3ef3a627fdfb19dd9b25de173/assets/day13/gradient-norm.png)

本次 gradient norm 最大約 `0.2958`，沒有 NaN 或 infinity；曲線仍有波動，且會在後段出現幾個較高的尖峰。這些尖峰不能單獨判定為 bug，但如果它們持續擴大、同時伴隨 loss 和 Q-value 爆升，就應該先查 reward、target mask、normalization 和 optimizer，而不是直接換演算法。

## 一次 run 要留下足夠線索，才有辦法回頭查

同一個 `metrics.csv` 只記錄數字還不夠。這次 GPU run 另外保存了 `config.json` 和 `summary.json`：前者包含 Python `3.12.13`、PyTorch `2.13.0+cu130`、CUDA `13.0`、Gymnasium、ALE、NumPy、resolved device `cuda:0`、GPU name、CUDA 狀態、seed、git commit、environment id、observation shape、wall-clock `68.61` 秒、SPS（steps per second，每秒 environment steps）`145.75` 和 peak VRAM（GPU 顯示記憶體）`70,963,200` bytes；後者則保存 steps、episodes、optimizer updates、target sync、action distribution 和最後一次 update 的摘要。

因此 run analyzer 可以從同一組 artifacts 重建 step range、return/loss/Q/gradient summary、epsilon range、replay size、SPS、non-finite count，以及 action 和 random/greedy distribution；輸出也會明確標出 `resolved_device: cuda:0`，不把 CPU 和 GPU run 混在一起。這讓「我記得那次好像有問題」變成「在 seed 42、這個 commit、這個 GPU、這個環境版本的第幾步，哪個訊號先變了」。

## 正確性確認後，才輪到超參數

Day 13 的 failure checklist 把順序固定成：environment / observation → action / reward → replay → dtype / normalization → Q gather / target mask → gradients / optimizer → target sync → exploration → metrics / evaluation → hyperparameters。

這個順序不是形式上的清單，而是因果上的縮小問題範圍。固定 batch 不能 overfit 時，learning rate 可能只是把真正的 gather bug 蓋住；action distribution 沒有 FIRE 時，調 gamma 不會修好 action mapping；return 沒上升時，先確認 replay、epsilon 和 target sync，通常比盲目調參更有資訊。

這次真實 CUDA debug run 最後得到的結論很克制：GPU training loop 能產生資料、執行 updates、同步 target，數值也維持 finite；但 10K steps 的 return 沒有提供穩定的學習證據。下一步不是宣布最佳設定，而是到 Day 14 固定 budget、seed 和 evaluation protocol，做一次真正可比較的 hyperparameter experiment。
