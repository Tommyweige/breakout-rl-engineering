# Day 18｜100K 看得到學習，不代表能分辨 DQN 與 Double DQN

Day 17 把問題縮小到一個很具體的差異：深度 Q 網路（Deep Q-Network，DQN）從畫面為每個 action 估計一個未來回報，再用同一組估計同時挑出最大 action、又把這個最大值當成學習 target；Double DQN 則讓 online network 負責選 action，更新較慢的 target network 負責評估它。這裡的 Q-value 可以先理解成模型對某個 action 未來能帶來多少回報的估計。

這個修改在概念上合理，卻還沒有回答一個更實際的問題：**在 Breakout 裡，這個 target rule 的差異是否大到能跨 training seed 重複出現？**

如果只看一個 100K checkpoint，很容易把「模型已經開始學」誤認成「已經足夠判斷哪個演算法比較好」。Day 18 的工作就是把這兩件事分開。

## 100K 的 learning signal 和演算法差異是兩件事

這裡的 transition 是環境接受一次 action、回傳 reward 和下一個 observation 的一筆資料。向量化環境一次前進兩個環境，但 budget 仍然以接受的 transition 數計算；它不是 vector iteration、optimizer update，也不是 Atari emulator 的 raw frame 數。

100K transitions 已經能讓 episode return（把一局中所有 reward 加總後的回報）出現上升的早期訊號，所以它適合回答「訓練流程有沒有完全失效」。但這個訊號不等於「兩個 model family 已經分開」。要回答後者，還需要相同條件下的重複訓練。

[![DQN 與 Double DQN 在 100K、250K、500K actual environment transitions 上的每局回報與 seed-level rolling curves](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/dqn-vs-double-training.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/dqn-vs-double-training.png)

圖的上半部保留每個完成 episode 的 raw Atari return，下半部是同一批資料的 20-episode rolling mean。每條線都代表一個 training seed，而不是把三個 seed 先揉成一條平均線；垂直線標出 100K screening、250K pilot 和 500K main。可以看到 100K 左右已經有可觀察的 learning signal，但不同 seed 的波動仍然很大。這張圖支持的是「值得把 budget 拉長」，不是「100K 已經選出 winner」。

## 公平比較先固定系統，再改 target rule

公平比較不是只把兩個字串換成 `dqn` 和 `double_dqn`。環境契約（Contract）是把遊戲怎麼建立、怎麼 reset、何時結束、怎麼計算 evaluation reward 固定下來的機器可讀規則；這次兩邊都使用 Contract v2：`ALE/Breakout-v5`、frame skip 4（一次 action 推進四個 emulator frames）、四張畫面堆疊、sticky action probability 0.25（環境有機會重複上一個 action）、環境只在 initial serve 或觀察到 life loss 後代替 policy 執行必要的 FIRE、life loss 不直接結束 episode，evaluation 使用固定 seeds、epsilon 0 和 raw Atari reward。

訓練系統也固定沿用 Day 16 選出的向量化路徑：兩個環境、GPU Replay（保存過去 transition 供抽樣的 replay buffer 放在 GPU）、batch size 32、相同 optimizer/update cadence、相同 epsilon schedule、float32 和兩個 CPU threads。正式 run 都明確 request `cuda`，實際 resolve 到同一張 `cuda:0` 的 NVIDIA GeForce RTX 4060 Laptop GPU；每個 run 也保存 PyTorch/CUDA 版本、SPS、wall-clock 與可取得的 peak VRAM（GPU 記憶體）。這樣 quality 的主要變因才是 target rule，而不是 backend 或硬體。

[![Day 18 staged comparison 的實際 gate 與 evidence flow](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/staged-comparison-flow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/staged-comparison-flow.png)

這張結構圖把執行順序和判斷邊界放在一起看：先完成 100K screening，再讓 seed 11 的 DQN/Double DQN 走完 250K pilot；pilot 的 training、evaluation 和 Q probe 都完整後，才展開 seeds 11、22、33 的 500K main。每個 checkpoint 同時留下 training metrics、CUDA runtime metadata、Contract v2 evaluation 和固定 probe diagnostics，最後只聚合實際完成 target transitions 的 runs。

這批正式 run 的 runtime metadata 保留了當時的 base commit：`1e586a43d88f1aeae936cae4135b62cd9439e10f`。修正後的 evidence 與主要 source/config hashes 則記在 manifest 和 comparison report；包含修正後 artifact 的 stable evidence commit 是 `0bae0fc0e0839d4c9b2e9b430554738e8f89d76a`。但歷史 run 當時只記錄了 Git 的 `HEAD`，沒有記錄工作樹（working tree，也就是尚未 commit 的檔案狀態）是否乾淨或 tracked diff 的 hash，因此無法回溯當時是否存在未 commit 的 Day 18 source 差異。這是 provenance 的限制，不是把 final commit 倒填成 training run 的 commit。

## 三段 budget 各自回答不同問題

100K 是 screening gate。它檢查 run 能否完成、loss/Q/target/gradient 是否保持 finite、Replay 和 target sync 是否照預期運作，以及 learning curve 是否完全失效；它不負責最終 model ranking。

250K 是 paired pilot。第一次先用一組 paired training seed 確認 pipeline 在較長 horizon 仍穩定，再把同一 protocol 擴到三組 seeds。從 100K checkpoint resume 時，這個實作會恢復 model、target network、optimizer、counter、RNG 和 provenance，但 Replay 內容不保存，而是重新建立並重新 warm 到 `learning_starts`。這不是 exact replay restore，所以 manifest 明確記錄了這個限制；重要的是 DQN 與 Double DQN 使用同一個 resume policy。

500K 是主要 quality horizon。這裡的 checkpoint 是在某個 transition milestone 保存下來的模型版本；除非 crash 或 correctness failure，run 不會因為暫時分數低就被手動淘汰。若三組結果仍重疊，正確結論會是「500K 仍不足以分辨」，而不是為文章製造一個 winner。

## 用 training seed 看重複性，用 evaluation seed 看表現

training seed 是一條完整訓練的隨機起點；evaluation seed 則是固定 evaluation episode 的環境起點。後者可以讓不同 checkpoint 面對同一組測試條件，但不能拿來假裝成更多 training replicates。

每個 training seed 的 checkpoint 都用同一套 Contract v2 evaluation：三個 evaluation seed groups、每組五局，共 15 局；分數是 raw Atari episode return，也就是不做 training reward clipping、直接加總環境回報。下圖的每個點仍保留 training seed，誤差棒則表示該 seed 的 15 局 episode spread。

[![DQN 與 Double DQN 在 250K pilot 和 500K main 的 per-training-seed evaluation return 與 spread](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/dqn-vs-double-eval.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/dqn-vs-double-eval.png)

500K 的真實結果如下：

| training seed | DQN mean ± episode std | Double DQN mean ± episode std | DQN − Double DQN |
|---:|---:|---:|---:|
| 11 | 14.27 ± 3.11 | 17.00 ± 6.28 | -2.73 |
| 22 | 13.00 ± 4.65 | 16.27 ± 3.55 | -3.27 |
| 33 | 20.27 ± 8.19 | 21.20 ± 7.05 | -0.93 |

三組 250K pilot 的差值依序是 -0.47、-2.07、-8.00；到了 500K，差距變成 -2.73、-3.27、-0.93。這表示 pilot 已經出現一致方向，但幅度仍會隨 seed 改變。

500K 的 paired differences 三組都小於零，平均差是 -2.31，paired difference 的 population std 是 1.00。按照本次 comparison report 使用的判斷規則：A/B 必須三組 paired differences 都有同一個嚴格方向；因此這次結果屬於 **B：在這個 500K、三組 paired CUDA protocol 下，Double DQN 較強**。

這個結論的 evidence strength 是「three paired CUDA seeds at 500K」，不是統計學上的普遍定理。Contract v2 下的 Random baseline mean 是 1.73 ± 1.12，它只提供同 protocol 的參考尺度，也不能取代演算法之間的 paired comparison。

[![500K DQN 與 Double DQN 的 paired seed mean，連線表示同一 training seed 的配對](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/paired-seed-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/paired-seed-comparison.png)

paired 圖的連線很重要：它把 seed 11 的 DQN 和 seed 11 的 Double DQN 連在一起，而不是先各自平均後才比較兩個孤立的數字。三條線都往 Double DQN 一側移動，但 seed 33 的差距比前兩組小，這正是保留 seed-level evidence 的價值。

## Q probe 顯示估計尺度，不是真實答案

Q-value 是模型對「從這個 state 開始採取某個 action，未來大概能得到多少回報」的估計。固定 probe states 可以讓不同 checkpoint 使用同一把尺；這裡保存了 60 個相同的 Breakout observations，並觀察每個 checkpoint 的 max-Q mean 和 spread。

[![固定 Contract v2 probe states 上各 training seed 的 max-Q mean 與 spread](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/q-probe-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/q-probe-comparison.png)

500K 時 Double DQN 的 max-Q scale 在三個 seed 都低於對應的 DQN：DQN 約為 2.69、2.75、2.68；Double DQN 約為 2.01、2.52、2.52。這個現象和 Day 17 對 overestimation 的假設相容，但它不能單獨證明 Double DQN 的 policy 比較好。沒有 Breakout 的 ground-truth Q-function，我們不知道較低的估計是不是較接近真實值；所以最後的 quality 判斷仍回到固定 evaluation return。

## 品質差異和工程成本要分開

同一張 GPU 上，SPS（每秒接受的 environment transitions）和 wall-clock 可以回答「這個 target rule 的工程代價是多少」，但它們不能回答「哪個模型學得比較好」。500K main 的三個 seed 平均如下：

| algorithm | mean SPS | mean wall-clock | peak allocated VRAM |
|---|---:|---:|---:|
| DQN | 350.09 transitions/s | 714.7 s | 608.6 MiB |
| Double DQN | 345.57 transitions/s | 723.5 s | 608.6 MiB |

[![500K main 的 seed-level SPS、wall-clock 與 peak allocated VRAM](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/runtime-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/0bae0fc0e0839d4c9b2e9b430554738e8f89d76a/assets/day18/runtime-comparison.png)

Double DQN 平均約少 1.3% SPS、wall-clock 約多 1.2%，而 peak allocated VRAM 在這批 run 都約 608.6 MiB。這和它在 next state 多做一次 online-network forward 的計算路徑一致，但 seed-level runtime 仍有波動；因此圖中保留每個 seed 的點，而不是把小差異包裝成精確的固定 overhead。

## 500K 讓問題變得更清楚，也留下下一個問題

Day 18 的結果不是「Double DQN 從此一定比較好」，而是更窄、也更有用的結論：在相同 Contract v2、相同 Day 16 CUDA vectorized backend、相同 float32 與 paired training seeds 下，這三組 500K runs 的 evaluation mean 都偏向 Double DQN；同時它付出一點吞吐和 wall-clock 代價。100K 的 early learning signal 足以支持繼續訓練，卻不足以做這個判斷；250K 開始露出方向，500K 才讓三組 paired evidence 都呈現同一方向。

仍然沒有被回答的，是這個方向能不能在更完整的 DQN family comparison 中保留。下一步會把 Dueling Network 等模型家族放到同一個 500K multi-seed protocol，必要時再延長 top candidates；那會是新的比較問題，不應該把 Day 18 的三組結果外推成所有環境和所有模型設定的保證。
