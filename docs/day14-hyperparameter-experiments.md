# Day 14｜超參數實驗：用受控比較取代「改一個數字試試看」

Day 13 做完除錯後，我們知道 DQN 的更新真的會發生、數值沒有立刻變成 NaN，探索和 Replay Buffer 也都在運作。但 10,000 個 environment steps 的 return 仍然沒有穩定上升。

這時最容易犯的錯，是開始隨意改 learning rate、epsilon decay、target update interval，看到某次曲線比較高，就把那個設定留下來。問題是：如果一次改了三個地方，我們根本不知道曲線的差異來自哪一個選擇；如果每次跑的 step 數和 seed 也不同，連「比較」本身都沒有固定尺度。

Day 14 要建立的是一個更小、但可以追查的問題：**在程式已經通過 Day 13 correctness 檢查後，只改一個因素，其他條件固定，曲線和統計結果是否提供了足以支持下一步的 evidence？**

這篇先用 learning rate 做一組 short development comparison。它不會教大型自動調參，也不會把單一 seed 的結果寫成最佳答案；目標是建立一個能交給 Day 15 延伸的實驗基線。

## 先分清楚：哪些數字是模型學的，哪些是我們選的？

DQN 在訓練時會更新一大組模型參數（model parameters），也就是 CNN 和 Q-value head 裡真正被 optimizer 修改的 weights。它們是模型從 transition 中學出來的數字。

但還有另一組數字不是模型自己學的，而是我們在訓練開始前選好的設定，例如每次更新走多大步、多久同步一次 Target Network、Replay Buffer 要保留多少 transition。這些設定叫做超參數（hyperparameters）。它們決定學習過程怎麼走，卻不會被反向傳播直接更新。

這個差異很重要。把一個超參數改掉，等於改變了實驗條件；不能把它和模型最後學到的 weights 混在一起，也不能只保存「這次把 learning rate 調低」這句話。完整的 run 必須知道其他條件仍然是什麼，否則之後無法重建這次比較。

## 為什麼一次只改一個因素？

假設 baseline（未改動的參考設定）的 learning rate 是 `1e-4`。一個受控 variant（從 baseline 改出來的比較設定）只把它改成 `5e-5`，其他欄位、seed、step budget、device 都保持不變。這樣兩條曲線的差異至少可以合理地歸因到「這個 learning rate 在這組條件下的影響」。

相反地，如果同一個 variant 同時改了 learning rate、epsilon decay 和 reward clipping，即使最後 return 變高，也不知道是哪個改動造成的，甚至可能是某兩個改動互相抵消後才看起來正常。一次只改一個因素（one-factor-at-a-time）不是保證因果關係的數學證明，但它讓下一輪實驗有清楚的假設可以檢驗。

因此 Day 14 的第一組比較只改 `learning_rate`：baseline 是 `1e-4`，low variant 是 `5e-5`，high variant 是 `2e-4`。這三個值都是設計選擇，不是由某個公式推導出的唯一答案。

## 一次實驗要留下什麼，才不會只剩一張曲線？

每一個 config 都先解析成完整的 DQN 設定。variant 可以從 baseline 繼承，再覆寫一個欄位；runner 會把繼承後的完整值寫進 run artifact，並計算它和 baseline 的 changed fields（實際不同的設定欄位）。這避免了「檔案只寫 learning rate，其他設定靠當時記憶」的問題。

接著，manifest 會在任何 run 開始前記錄 experiment id、baseline、variants、seed、step budget 和輸出目錄。runner 預設順序執行，因為多個 GPU-heavy run 同時搶同一張卡，會讓 throughput 和 memory condition 都變得不清楚。每個 run 則各自保存 `config.json`、逐 environment step 的 `metrics.csv`、最後的 `summary.json` 和 checkpoint。

這個資料流是實際 runner 的結構：config 先解析並驗證，manifest 再列出待執行的 variants；每次只啟動一個 `DQNTrainer`，成功時留下 metrics 和 runtime metadata，CUDA 條件不成立時留下 blocked status，而不是偷偷改用 CPU。

[![Day 14 實驗從 config、manifest、順序執行到比較圖的結構流程](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/7310a7e/assets/day14/experiment-workflow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/7310a7e/assets/day14/experiment-workflow.png)

這是一張依照實際程式控制流整理的 structural diagram，不是某一次 run 的時間截圖。讀圖時要注意兩個關係：variant 是逐一處理的；而每個 run 的結果都會回到同一個 manifest lifecycle，最後才進入比較。`CUDA request available?` 的分支代表明確要求 CUDA 卻無法使用時的 blocked path，並不代表可以把 CPU 結果混進正式 comparison。

## 比較曲線時，為什麼 x 軸要用 environment step？

Episode index 看起來直觀，但不同 policy 可能用不同步數才結束一局。若一個 variant 讓 episode 變長，兩個「第 20 局」其實可能發生在很不同的訓練進度。

所以主要 x 軸使用 environment/global step：Agent 執行一次 action、環境回傳一次 transition，就增加一個 step。這讓 return、loss、Q-value、epsilon 和 SPS 可以對齊到同一個訓練進度。Episode return 仍然只在一局完成時寫入，因此曲線上的點會是稀疏的真實 episode 結果，而不是把缺少的值補成零。

Aggregate 也必須在看結果前固定。這次 report 使用「最後 20 個已完成 episode」計算 recent mean 和 median；`best rolling return` 則是所有連續 20 個 episode 視窗平均值中的最大值。這些規則對三個 run 完全相同，沒有看完曲線才替某個 run 挑一個最有利的區間。

## 真實 CUDA comparison 看到了什麼？

這次 batch 使用固定 seed `42`、固定 10,000 environment steps、`float32`，三個 run 都明確要求 `cuda`，最後都解析成同一張 `cuda:0`：NVIDIA GeForce RTX 4060 Laptop GPU。PyTorch 是 `2.13.0+cu130`，CUDA runtime version 是 `13.0`。下表來自 `experiments/day14-cuda-lr-comparison-committed/manifest.json` 和每個 run 的 `metrics.csv`、`summary.json`，不是手工抄錄的示意數字。

| run | 唯一改動 | 完成 steps | episodes | 最近 20 局 mean | 最近 20 局 median | 最佳 20 局 rolling mean | SPS | wall-clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | `learning_rate=1e-4` | 10,000 | 48 | 1.15 | 1.00 | 1.60 | 127.86 | 78.21 s |
| low | `learning_rate=5e-5` | 10,000 | 45 | 1.75 | 2.00 | 2.00 | 135.26 | 73.93 s |
| high | `learning_rate=2e-4` | 10,000 | 46 | 1.60 | 1.50 | 1.85 | 152.82 | 65.43 s |

這張圖要回答的問題是：**同樣走過 10,000 個 environment steps，只改 learning rate 時，raw episode return 的變化是否呈現一致而可解釋的差異？**

[![固定 seed、step budget 與 CUDA device 後，三個 learning-rate run 的 raw episode return 對 environment step 比較](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/7310a7e/assets/day14/experiment-return-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/7310a7e/assets/day14/experiment-return-comparison.png)

圖的每個點都是 `metrics.csv` 裡有真實 `raw_episode_return` 的 completed episode，水平位置是該 episode 完成時的 environment step；三條線的 label 也保留了 run 名稱、seed 和 resolved device。可以看到三條曲線都很鋸齒，沒有一條從頭到尾穩定壓過另外兩條。這次 low run 的最近 20 局 mean 是 `1.75`，高於 high 的 `1.60` 和 baseline 的 `1.15`，但差異只來自一個 seed，且曲線中仍有很多零分與短暫尖峰；不能把這次排序寫成 low learning rate 已經比較好。

這張圖支持的結論很有限但很實用：三個設定都真的完成了相同 budget，runner 確實能把受控變因和訓練曲線對齊，這組 10K run 可以作為下一輪候選篩選的 development signal。它不能證明 low 或 high 一定比 `1e-4` 好，也不能證明任何設定已經學會 Breakout。

另外，SPS 和 wall-clock 是成本訊號，不是遊戲品質。high run 的 SPS 較高、wall-clock 較短，但這只表示這次執行的 throughput/cost，不表示策略品質較好。report 也保存了 peak allocated VRAM：baseline 與 high 約 `70,963,200` bytes，low 約 `69,758,976` bytes；這些數字用來追查資源條件，不用來替 return 排名。

## 其他超參數各自會改變什麼？

這次只驗證 learning rate，但相同的 workflow 可以逐一測試其他軸。它們的差異先要理解清楚，才知道下一個 variant 應該改什麼：

- **epsilon decay speed** 控制 Agent 從隨機探索轉向目前估計的 greedy action 有多快。它使用 environment step 比 episode 更容易公平對齊，因為每局長度不同；衰減太快可能還沒收集到多樣經驗就停止探索，太慢則可能長時間沒有利用已學到的策略。
- **target update interval** 控制 Target Network 多久複製一次 Online Network。同步太頻繁，目標值跟著被更新的網路一起移動；同步太慢，目標較穩但可能過時。這是穩定性與反應速度的 trade-off，不應和 learning rate 同時改動後再猜原因。
- **replay capacity 與 learning starts** 分別影響經驗可以保留多久，以及收集多少 transition 後才開始更新。容量越大會增加記憶體成本，warm-up 太短則可能讓第一批資料過度相似；但把 capacity 調大也不保證策略變好。
- **reward clipping** 把訓練用 reward 限制到正、零、負的符號，能讓不同大小的分數不會直接造成很大的更新，但也會丟失原始 reward 的幅度。這也是為什麼本專案同時保存 raw reward 和 training reward：比較 clipping 時，訓練訊號可以改，遊戲分數仍然要用原始 return 評估。

每一次比較都應該先寫下假設、固定 aggregate 規則，再跑 config。這樣曲線是用來檢驗問題，不是跑完後才挑故事。

## CUDA 的可重現，是完整記錄條件，不是假裝跨硬體完全相同

這次正式 comparison 的三個 config 都是 `requested_device=cuda`，不是 `auto`。`auto` 在沒有 CUDA 時可以選 CPU，適合 portability smoke test；但正式 Day 14 evidence 使用明確 CUDA，若 CUDA 不可用就標記 blocked，禁止 silent fallback。這次的 precision 是 `float32`，表示模型計算使用 32-bit 浮點數。每個 run 還保存 requested/resolved device、GPU 型號、device index、PyTorch/CUDA version、precision、steps per second（SPS，單位時間完成多少 environment steps）、wall-clock（實際經過的訓練時間）和可取得的 peak VRAM（GPU 曾經使用的最高顯示記憶體）。

固定 seed 只表示我們固定了初始化和抽樣的條件之一。CPU、不同 GPU、不同 CUDA/PyTorch build 仍可能出現數值路徑差異，因此 **reproducible configuration 不等於跨硬體 bit-exact deterministic result**。真正可追查的做法是把完整 config、seed、device、software metadata 和 run artifact 一起保存，讓下一次重跑時知道差異來自哪裡。

## Day 15 的 baseline 應該怎麼選？

這一批不能誠實地宣布某個 learning rate 是最佳設定。若需要交給 Day 15 一個 reference config，`dqn_baseline.json` 仍是較合適的起點：它完成了和兩個 variant 相同的 budget，loss/Q-value 都有有限值，learning rate 位在本次受控範圍的中間，而且不依賴單一尖峰或單一 median 來宣稱勝出。low variant 可以列為下一輪的 candidate，但不能取代 reference 的角色。

這個選擇的意思是「先固定一個可追查的比較基準」，不是「baseline 已經是最強策略」。Day 15 應該在固定 evaluation protocol 下加入 random baseline，並用更多 seed 檢查這個 development signal 是否仍然存在。若多 seed 後 low learning rate 仍然穩定改善，才有理由把 baseline 往它移動。

## 重跑與下一個問題

從 repository root 可以用同一組完整 config 重新建立一個新的 batch；runner 會拒絕覆蓋已存在的 experiment id：

```powershell
conda run --name breakout-rl-engineering python run_experiments.py --require-cuda configs/dqn_baseline.json configs/experiments/lr-low.json configs/experiments/lr-high.json
```

比較與畫圖也都直接讀 manifest 和 run artifacts：

```powershell
conda run --name breakout-rl-engineering python compare_runs.py experiments/<experiment-id>/manifest.json
conda run --name breakout-rl-engineering python visualize_experiment_comparison.py experiments/<experiment-id>/manifest.json
```

現在我們可以回答 Day 14 的中心問題：調參不是逐次憑感覺改數字，而是先固定條件、只改一個因素、保存完整 evidence，再用同一個 step budget 和 aggregate window 比較。這組單 seed CUDA 結果只足以選出下一步要追查的候選，還不足以宣稱穩健的演算法結論。

下一篇會把這些 development signal 放進固定的 milestone evaluation，加入 random baseline 和更嚴格的比較 protocol。
