# Day 14｜為什麼 10K 不夠：把超參數比較拉長到 100K

[Day 13](day13-debugging-unstable-rl-training.md) 的 10,000 environment steps 已經回答了一個重要問題：程式有沒有壞。這裡的 environment step 是 agent 執行一次 action、環境回傳一次結果的互動單位。那次 CUDA（讓 PyTorch 使用 NVIDIA GPU 的介面）diagnostic run 完成 48 局遊戲；一局在環境結束時才算完成，這種完整回合稱為 episode，該局累積的分數就是 raw return。

診斷也確認訓練真的有在動：模型從一批經驗計算梯度並更新權重一次，稱為一次 optimizer update；經驗先放在 Replay Buffer，也就是用來重播過往互動的記憶區；Target Network 則是暫時固定的價值估計副本，用來讓學習目標不要每一步都跟著主模型晃動；epsilon schedule 是隨步數降低隨機探索比例的規則。不過，這些檢查只代表程式能正常執行，不代表 10K 已經足以比較超參數。

這個結果很容易被誤讀。10K 足以發現訓練誤差、價值估計或權重調整量變成非有限值、模型根本沒有更新，或 CUDA 設定錯誤；卻不代表 10K 足以判斷三個超參數設定（hyperparameter config）誰比較好。baseline 是未改動的參考設定，variant 是只改一個因素的比較設定；如果 baseline 的短期平均是 `1.15`、另一個 variant 是 `1.75`，差異可能只是遊戲回合剛好落在不同的隨機波動，而不是 learning rate（每次模型更新調整權重的步幅）真的造成了穩定改變。

所以 Day 14 的問題被重新寫得更精確：**先用 10K 做 health screening（只驗證訓練流程能正常運作），再把相同的 learning-rate comparison 拉長到 100K，觀察 learning curve（回報隨訓練步數變化的曲線）和數值診斷是否開始分化。** 100K 仍然不是保證學會 Breakout 的數字；它只是比 Day 13 長十倍的 observation horizon，也就是能觀察到的訓練時間尺度，讓我們有機會看到短跑看不到的變化。

## 10K 能證明什麼，不能證明什麼？

10,000 steps 可以當成一個便宜的 short screening：如果 loss（模型預測與學習目標的差距）、Q-value（模型估計某個 action 長期價值的數字）、gradient（告訴模型權重應往哪裡調整的方向與幅度），或 Replay Buffer occupancy（記憶區目前存了多少互動）立刻異常，就不用浪費更長時間；如果 CUDA request 無法解析，也應該在這一層被擋下來。

但「沒有立即爆掉」和「這個設定比較好」是兩個不同命題。Breakout 的 reward 很稀疏，一局的長度也會變化；10K 只涵蓋有限的遊戲互動，最後幾個 episode 的分數很容易支配平均值。把這樣的 final return 排名，會把 health check 偽裝成 model comparison。

這也是為什麼新的 workflow 把 screening 和 main comparison 分成不同的 stage。screening 只回答「能不能正常跑」；main comparison 才回答「在更長的訓練時間內，曲線是否出現可解釋的差異」。

[![從 Day 13 的 10K diagnostic、10K screening 到 Day 14 100K main comparison 的決策流程](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/25fee37/assets/day14/budget-stages.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/25fee37/assets/day14/budget-stages.png)

圖中的分支是實際實驗規則，而不是某次 run 的裝飾性流程圖：10K 通過 health checks 才能進入 100K；100K 之後若仍沒有可靠 signal，合法的結論是「目前無法分辨」，不是強行挑一個 winner。

## 一次只改一個因素，100K 只延長 observation horizon

這次仍然採用 one-factor-at-a-time。baseline 是未改動的參考設定；variant 是只覆寫一個 hyperparameter 的比較設定。三個 main configs 只改 learning rate，並且全部使用同一個 GPU-resident replay backend：

| run | learning rate | 其他條件 |
|---|---:|---|
| baseline | `1e-4` | 固定 GPU replay、seed、environment、epsilon、target update、reward clipping、device、precision |
| learning-rate-low | `5e-5` | 同上 |
| learning-rate-high | `2e-4` | 同上 |

learning rate 是每次模型更新調整權重的步幅；它太小可能讓學習變慢，太大可能讓 Q-value 或 gradient 變得更躁動。這次不是同時改 epsilon decay 或 Target Network interval，所以即使結果只是一個初步訊號，也還能回答比較單純的問題：在同一個 100K horizon 下，learning rate 的差異是否值得繼續追蹤？

三個 run 固定 seed `42`。seed 是控制初始化與抽樣的整數起點，能讓實驗條件更容易重現；它不是把 CUDA 執行變成跨硬體 bit-exact deterministic 的保證。三個 run 也都使用 `requested_device=cuda`，解析成同一張 `cuda:0` 的 NVIDIA GeForce RTX 4060 Laptop GPU，precision 是 `float32`（32-bit 浮點數格式）。

## GPU replay 從「搬資料」變成另一個資料路徑

原本的 Replay Buffer 把每筆畫面以 `uint8`（每個像素一個 byte）保存在 NumPy 陣列；更新時才抽樣、轉成 `float32`，再送進 GPU。GPU-resident replay 則把相同欄位直接存到 GPU，讓抽樣與 gather（依 index 收集指定 transition）也在 GPU 完成。這是 systems backend 的改變，只有在兩條路徑對同一種 transition 做出相同處理時，才不會同時改到 DQN 演算法。

這次先做固定 batch32 的完整 trainer A/B。兩個 run 都是 10K environment transitions、`train_frequency=4`、`learning_starts=2048`、同一個 model 與 seed；差別只有 CPU preallocated transfer 和 GPU replay。CPU 路徑的 replay storage 約 `564.6 MB` 在 RAM，GPU 路徑則把它放進 `cuda:0`。

| path | environment SPS | optimizer updates/s | training samples/s | wall-clock | GPU util. mean | replay storage |
|---|---:|---:|---:|---:|---:|---|
| CPU + preallocated | 361.61 | 71.92 | 2,302 | 27.65 s | 23.87% | CPU, 564.6 MB |
| GPU-resident replay | 338.13 | 67.25 | 2,152 | 29.57 s | 24.63% | GPU, 564.6 MB |

這個結果沒有支持「GPU replay 一定讓完整 trainer 更快」。在目前的單環境、batch32 workload 中，GPU replay 每次 environment transition 仍要把新 observation 寫入 GPU；GPU 抽樣的好處不足以抵銷這個成本。這與 optimizer-side microbenchmark 不矛盾：microbenchmark 沒有 `env.step`、replay insertion 或 episode bookkeeping，所以它只能回答 GPU update path 的上限，不能代表完整 Breakout trainer。完整 A/B 的 manifest 與 runtime samples 保存在 `experiments/day14-gpu-replay-ab-profiled-v2/` 與 `assets/day14/gpu-replay-ab-profiled-v2-profiling/`。

GPU replay 也沒有直接複製 NumPy 的 bit-exact RNG sequence。CPU replay 使用目前 active `size` 的 uniform sampling without replacement；正式 GPU backend 保留 active range 與 without-replacement contract，但使用 CUDA RNG。checkpoint 仍然不保存 replay，resume 之後重新 warm up，與原本 trainer 的 contract 一致。這些條件讓它成為可比較的 storage/backend 變因，而不是悄悄換掉訓練資料規則。

## 正式 100K 前先量測資料管線

GPU enabled 不等於整條訓練管線都由 GPU 主導。這個 DQN（用神經網路估計 action 價值的深度 Q-learning）每一步仍要由 Python 環境產生下一個 observation，再把 replay batch 送進 GPU；如果每次更新都把完整診斷統計同步回 Python，並立刻把一列 CSV（以逗號分隔的表格記錄）寫到磁碟，GPU 會在等待環境、同步或磁碟輸入輸出（I/O），而不是持續計算。這些每一步都會經過的耗時路徑稱為 hot path。這正是這一輪先做 throughput gate 的原因：先量測「整個訓練流程每秒完成多少 environment steps」，這個速度稱為 SPS（steps per second）；wall-clock 則是從開始到結束實際經過的時間，再開始昂貴的 100K comparison。

| 10K gate | 原始 hot path | 調整後 hot path |
|---|---:|---:|
| end-to-end SPS（每秒 environment steps） | 145.94 | 218.99 |
| optimizer updates/s | 32.85 | 49.29 |
| wall-clock（實際經過時間） | 68.52 s | 45.66 s |
| CPU thread count | 12 | 1 |
| GPU utilization sample | 28% | 35% |
| diagnostics / CSV flush interval（把記錄寫入磁碟的間隔） | 1 / 1 | 100 / 100 |

這些是同一台機器、同一個 learning config、同一個 10K seed 的 before/after 量測；GPU utilization 是 `nvidia-smi` 取到的樣本，不是整段 run 的平均值。調整只減少非必要的每步統計同步、把 CSV flush 改成批次處理，並測試 CPU thread count；沒有改 batch size 或 train frequency。端到端 SPS 提升 `1.50×`，達到這次的工程目標；兩邊都完成 `2,251` 次 optimizer update、target sync 次數相同、有限值診斷存在。這不是 bit-exact 曲線的要求，因為效能調整可能改變非關鍵的執行順序，但它保留了訓練邏輯的回歸檢查。

峰值保留的 GPU 記憶體約 `90 MiB`；MiB 與 GiB 是以 2 的次方計算的記憶體單位，headroom 是距離顯存上限刻意保留的安全餘裕。相對於約 `8 GiB` 顯存仍有充分 headroom，所以沒有為了速度引入 pinned memory（方便 CPU/GPU 傳輸的記憶體配置）或 vectorized environment（一次並行執行多個環境）。這個結果反而說明瓶頸不在顯存容量：CUDA 確實啟用，但單次模型更新太小，環境與 Python/記錄管線的等待時間更值得先處理。

## 100K 不只看最後一個數字

100K 的價值不在於最後一列 summary 比 10K 更大，而在於可以回頭問「變化何時開始」。因此 main config 每 25,000 steps 保存一次 checkpoint（保存當時模型與 optimizer 狀態的快照），CSV 則保留每個 environment step 的 metrics。comparison report 會取 25K、50K、75K、100K 附近的實際 row，並同時保存 loss、Q、Target、gradient、epsilon 和 SPS。

回合分數仍然只在 episode 完成時出現，所以 return 曲線的每個點都是實際完成的 episode，不會把缺少的值補成零。為了避免看到結果後改規則，這次固定使用最後 20 個 completed episodes 的 mean/median，以及所有 20-episode rolling windows 中的最高平均值。recent trend 則把最後 20 局分成前後兩半，報告後半平均減去前半平均的變化。

## 100K GPU replay main comparison 的真實結果

下表來自正式 GPU replay 100K comparison report。三個 run 都完成 100,000 transitions，使用相同 `replay_backend=gpu`、相同 batch32 與 schedule；report 確認 stage 是 `main`、三個 requested/resolved device 相同，符合正式 CUDA main comparison 條件。

| run | episodes | 最近 20 局 mean | median | 最佳 rolling20 mean | recent trend Δ | SPS | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 364 | 5.15 | 5.50 | 5.15 | -0.30 | 358.47 | 278.96 s |
| learning-rate-low | 389 | 1.80 | 2.00 | 2.90 | -0.20 | 387.92 | 257.78 s |
| learning-rate-high | 308 | 9.15 | 8.00 | 9.15 | -0.10 | 390.60 | 256.02 s |

這次 100K 和 10K 的差異不是「終於得到一個永遠正確的排名」，而是曲線開始提供更長時間尺度的 evidence。GPU replay 條件下，high learning rate 的 rolling return 高於 low 與 baseline；這讓 high 成為值得交給 multi-seed evaluation（用多個不同 seed 重複同一設定）的 candidate。這次 high 的 recent trend Δ 為 `+0.90`，但單一 seed 仍不足以宣稱它是最終最佳 learning rate。

這張圖回答的是：**在相同 100K steps 下，三個 learning-rate run 的 raw return 是否開始沿著不同的 learning curve 前進？** 淡色點是每局實際完成時的 raw episode return；粗線是固定 20-episode rolling mean。x 軸仍然是 environment step，而不是 episode index。

[![100K main comparison 中三個 GPU replay learning-rate run 的 raw return 與 20-episode rolling mean](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-return-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-return-comparison.png)

圖中 high 的 rolling curve 在後段維持較高，baseline 次之，low 較低；這支持「100K 比 10K 更能看出候選差異」這個觀察。它不能支持「`2e-4` 在所有 seed 都最佳」，也不能排除更長訓練後排名改變。SPS 是每秒完成多少 environment transitions 的 throughput，wall-clock 是實際經過的時間；它們是執行成本，不是遊戲品質。GPU replay 的 optimizer-side microbenchmark 約 39K–41K samples/s，也不能被寫成這三個完整 trainer 的 SPS。

## Q、Target、gradient 在較長 horizon 下怎麼走？

Q-value 是模型對 action 長期價值的估計；Target mean 是另一個較穩定的參考估計，用來形成模型要追近的學習目標（Bellman target）；gradient norm 則是一次更新中梯度總量的大小。這些診斷不能單獨判斷策略好壞，但能幫助區分「學得慢」和「數值開始不穩定」。

| run | Q mean：25K → 100K | Target mean：25K → 100K | loss max | gradient max |
|---|---:|---:|---:|---:|
| baseline | 0.751 → 1.364 | 0.731 → 1.373 | 0.0172 | 0.8237 |
| learning-rate-low | 0.555 → 1.155 | 0.553 → 1.193 | 0.0192 | 0.6367 |
| learning-rate-high | 0.513 → 1.722 | 0.499 → 1.755 | 0.0325 | 1.0135 |

三個 run 的 Q 與 Target 都隨 horizon 增加，且 report 沒有非有限值；因此不能把「Q 變大」直接稱為爆炸。low 在 50K 的 gradient norm 約為 `0.883`，高於 baseline 的 `0.075` 與 high 的 `0.331`，但後續又回到較低尺度。合理的判讀是：high learning rate 讓價值估計在後段發展得更快，也值得持續監看；這還不是「已經不穩定」或「一定更好」的證明。

下圖把同一批 run 的 loss、Q mean、Target mean、gradient norm、epsilon 和 SPS 放在相同的 environment-step 軸上。它的用途不是製造另一個 winner，而是檢查 return 差異是否伴隨非有限值、持續增大的梯度，或完全不同的執行成本。

[![100K main comparison 的 loss、Q、Target、gradient、epsilon 與 throughput diagnostics](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-diagnostics-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/experiment-diagnostics-comparison.png)

epsilon 在這組 config 中於前 10K steps 下降到 `0.05`，之後 90K 大多在低探索機率下執行。這是現有 epsilon schedule 的設計條件，不是這次 learning-rate comparison 的變因；如果下一輪要研究探索速度，應另開一個只改 epsilon decay 的 batch。

## 從 checkpoint 看見真實遊戲行為

曲線告訴我們數值如何變化，GIF 則讓我們確認 checkpoint 真的能驅動環境。以下四段影片都從實際 checkpoint 載入 online network，以同一個 evaluation seed、`evaluation epsilon=0`、相同 preprocessing 和最多 500 個 evaluation steps 錄製。1K 與 10K 使用 short-screening checkpoints；50K 與 100K 使用同一個 100K main run 的 checkpoints。

1K 和 10K 的 greedy episode 都得到 `0`，這不是失敗的錄影，而是重要限制：training return 仍包含探索行為，不能直接等同 greedy evaluation policy。50K 與 100K checkpoint 在同一個 evaluation contract 下分別得到 return `4` 與 `7`；這是單一 seed、單一 evaluation episode 的 limited evidence，不能取代 Day 15 的多 episode evaluation。

[![1K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-001k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-001k.gif)

[![10K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-010k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-010k.gif)

[![50K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-050k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-050k.gif)

[![100K checkpoint 的真實 Breakout gameplay](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-100k.gif?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/gameplay-step-100k.gif)

每個 GIF 的 checkpoint step、training run、evaluation seed、實際 frame 數與重現命令保存在同名 JSON metadata。影片回答的是「這個 checkpoint 能做出什麼行為」，不回答「這個配置在多個 seed 下是否有較高 final policy quality」。

## Batch size 同時是硬體效率與學習變因

LR comparison 選出的 development candidate 是 `2e-4`，但它還沒有回答另一個問題：GPU 每次收到的工作是否太小。batch size 是一次 optimizer update 使用的 replay transitions 數量；它變大時，每次更新的 GPU 工作與 training samples/s 可能增加，但也會改變梯度估計與 learning dynamics，所以不能把它當成純效能開關。

因此 Day 14C 先固定 learning rate、`train_frequency=4`、環境、seed、replay、epsilon、target update、precision 和 CUDA device，只比較 batch size `32/64/128`。Stage 1 每個設定跑 10K，並以固定 1 秒間隔保存 GPU utilization、power、device memory、process CPU 和 sampling method；Stage 2 只對 Stage 1 中實際提高 environment SPS 且通過數值 guardrails（攔截 replay、epsilon、episode 與有限值異常的安全檢查）的候選跑 100K。training samples/s 代表每秒送進 optimizer update 的 transitions 數量。

| batch size | environment SPS | wall-clock | optimizer updates/s | training samples/s | GPU utilization mean | GPU power mean | device memory peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 235.74 | 42.42 s | 53.06 | 1,698 | 30.13% | 25.59 W | 1.76 GiB |
| 64 | 203.32 | 49.18 s | 45.77 | 2,929 | 32.22% | 27.11 W | 1.89 GiB |
| 128 | 177.36 | 56.38 s | 39.92 | 5,110 | 34.88% | 30.33 W | 1.97 GiB |

這組真實量測展示了「GPU 利用率越高不一定越快」：batch128 的 GPU utilization 和 training samples/s 都最高，但 environment SPS 最低；因為單一 Breakout environment 的 action inference 仍是 batch=1，環境互動、replay 和小型 GPU update 之間仍然串行等待。batch64 也沒有帶來 environment throughput 的收益。三組的 process CPU 約為 `5.55%`、`5.66%`、`5.77%`（以 16 logical CPUs 歸一化），顯示前一輪 CPU path optimization 已把瓶頸推向更細小的 GPU work，而不是 CUDA 沒有啟用。

所以這次沒有把 batch64 或 batch128 硬送進 100K：它們沒有實際的 end-to-end efficiency gain；batch32 則沿用已完成的長程 reference。這是有意義的 negative result——增加 samples/s 並沒有降低 wall-clock，也沒有理由只因 GPU utilization 上升就改變正式 training config。

[![Day 14 batch size 32、64、128 的 throughput、GPU utilization、power、VRAM 與短跑 learning guardrails](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/batch-size-efficiency.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/27b7e07/assets/day14/batch-size-efficiency.png)

## 用 1/2/4 threads 凍結系統設定，再交給 Day 15

batch-size profiling 之後，仍要決定 PyTorch CPU thread count。固定 batch32、同一個 10K config 實測 1、2、4 threads，end-to-end SPS 分別為 `290.08`、`306.79`、`304.15`；因此選 2，而不是沿用預設約 12 threads。這個選擇同樣以實際 SPS 為主，同時保留 GPU power、VRAM、process CPU 和 finite metrics 作 guardrails。

最後保留兩份可追溯的 Vanilla DQN config：`day14-vanilla-dqn.json` 是 CPU replay reference，`day14-gpu-replay-vanilla-dqn.json` 是 GPU replay candidate。GPU replay high run 的 100K evidence 完成 308 個 episodes，最近 20 局 mean `9.15`、最佳 rolling20 mean `9.15`、recent trend Δ `-0.10`，wall-clock `256.02 s`；四個 milestone 都有 finite loss/Q/Target/gradient。這是 single-seed development evidence，不是 final policy selection。下一篇應使用固定 evaluation contract 和多 seed，比較兩份 config 是否真的在相同 wall-clock 下得到相同或更高 Return。

## 10K screening 與 100K main 必須分開保存

舊的 10K learning-rate artifacts 仍然保留，但現在明確標記為 `stage=screening`、`budget_level=short_screening`。它們可以回答三個 config 是否能正常啟動、更新、寫出 metrics 和完成 CUDA metadata；它們不能拿來和新的 100K recent mean 放在同一張 ranking table 裡。

main manifest 則標記為 `stage=main`、`budget_level=main_day14`，並由 report 檢查三個 run 的 `total_steps`、stage、requested/resolved device 是否一致。這個分層解決了一個常見的實驗錯誤：把不同 observation horizon 的 final return 當成同尺度數字比較。

## 100K 之後仍然不能把單一 seed 當成答案

這次的 GPU replay 100K evidence 支持 high learning rate 成為下一輪候選，也支持 low 在這組條件下暫時落後；但這仍然是 single-seed development evidence。不同 seed 可能改變 episode 結束時間、Replay Buffer 內容與每次 update 的抽樣，CUDA 和不同硬體也可能產生細微數值差異。

如果下一輪 multi-seed 仍然看到 high 的 return curve 高於其他設定，而且 Q/Target/gradient 沒有出現不可接受的失控，再把它交給更正式的 milestone evaluation 才合理。如果 100K 後的差異在多 seed 消失，或所有 config 都沒有可靠 trend，也應該保留「目前無法分辨」這個結果；只有在 100K 已經提供清楚但仍不足的 signal 時，才值得把少數候選延長到 250K 或更長。

Day 14 因此沒有把調參變成「跑得久就一定找到答案」。它建立的是一個可分層的判讀方式：10K 負責健康檢查與 systems profiling，100K 負責觀察 learning dynamics，GPU replay A/B 負責回答 backend 的完整 trainer 成本，25K 到 100K 的 diagnostics 與真實 GIF 負責揭露行為邊界，最後仍由多 seed evaluation 決定這個 signal 是否值得相信。GPU utilization、microbenchmark samples/s 和 training return 各自回答不同問題，不能互相代替。

下一篇會把這個 reference/candidate distinction 放進固定的 milestone evaluation，加入 random baseline，檢查 100K 觀察到的差異是否能在更嚴格的 protocol 下重現。
