# DQN 訓練除錯檢查表

當訓練程式沒有 crash、loss 也有數字，但 Agent 沒有變好時，按照下面的順序檢查。原則是先證明 correctness，再調整 hyperparameter。

## A. Environment / observation

- [ ] 確認 `reset()` 與 `step()` 的回傳順序符合 Gymnasium API。
- [ ] 確認 observation shape 是 `(4, 84, 84)`、dtype 是 `uint8`，像素值範圍合理。
- [ ] 確認 `terminated` 與 `truncated` 的意義沒有混用。
- [ ] 固定 seed 重跑時，環境初始狀態與第一段 action 序列可重現。

## B. Action / reward

- [ ] 確認 action index 沒有超出 environment action space。
- [ ] 從 action distribution 檢查 `NOOP`、`FIRE`、`RIGHT`、`LEFT` 是否有合理變化。
- [ ] 分開保存原始 episode return 與訓練用 reward；確認 reward clipping 沒有誤套到評估分數。
- [ ] 檢查 random / greedy decision ratio 是否跟 epsilon schedule 一致。

## C. Replay data

- [ ] 確認 Replay Buffer 的 state、next state、action、reward、episode flags 形狀一致。
- [ ] 確認 replay occupancy 會從零增加，且達到 warm-up 後才開始更新。
- [ ] 抽樣檢查 transition 沒有把下一局的畫面接到上一局。

## D. Tensor dtype / normalization

- [ ] storage 保持 `uint8`，只在進入模型的邊界轉成浮點 tensor。
- [ ] 確認 state 與 next state 的 device 相同，action 是整數，reward 是浮點數。
- [ ] 確認模型輸入沒有被重複 normalization，也沒有完全忘記 normalization。
- [ ] 執行 finite checks，先排除 NaN 與 infinity。

## E. Q gather / target mask

- [ ] 確認 `gather` 選到的是 transition 實際採取的 action。
- [ ] 確認 terminal transition 不會 bootstrap 下一個 state 的價值。
- [ ] 確認 target shape 是 `(B,)`，並且 prediction 與 target 對應同一批資料。
- [ ] 執行 fixed-batch overfit；若固定 target 都無法下降，先不要調 exploration。

## F. Gradients / optimizer

- [ ] 確認 `zero_grad → backward → optimizer.step` 順序正確。
- [ ] 確認 gradient norm 是 finite，沒有突然爆大或長期為零。
- [ ] 確認 optimizer 只更新 Online Network，沒有更新 Target Network。
- [ ] 確認至少有 optimizer update，不能只看到 environment steps 增加。

## G. Target network sync

- [ ] 確認 Target Network 初始化時與 Online Network 同步。
- [ ] 確認 target sync count 與設定的 interval 一致。
- [ ] 確認 target network 在兩次同步之間保持固定。

## H. Exploration

- [ ] 確認 epsilon 從設定的起點下降到設定的下限。
- [ ] 對照 random / greedy ratio，而不是只看 epsilon 欄位。
- [ ] 若 action 幾乎永遠相同，先確認 action mapping 與 Q-value 輸出，再考慮調整 epsilon。

## I. Metrics / evaluation

- [ ] 同時查看 raw return、loss、Q-value、target、TD error、gradient norm、replay size、SPS。
- [ ] loss 下降只能表示目前 target 比較容易被擬合，不等於 policy 變強。
- [ ] 用 random-policy baseline 提供「完全沒有學習」的參考線。
- [ ] 檢查 run 是否保存 config、runtime metadata、metrics 與 summary。

## J. Hyperparameters

- [ ] 只有在 A–I 的 correctness 已有證據後，才開始調 learning rate、gamma、batch size 或 epsilon schedule。
- [ ] 一次只改一個變因，固定 seed 與 steps，保存每次 run 的 config 與結果。
- [ ] 不把短跑的偶然 return 當成最佳設定；受控比較留到 Day 14。
