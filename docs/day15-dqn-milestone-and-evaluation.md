# Day 15｜DQN 到底有沒有真的學會？先把評估做成固定實驗

Day 14 的訓練曲線和 100K checkpoint GIF 已經出現值得追蹤的行為，但它們還不能直接回答一個更重要的問題：**這個模型換幾個起始條件後，是否仍然比亂選 action 好？**

訓練曲線是在模型持續更新時量到的；GIF 則只有一個 checkpoint、一個 evaluation seed 和一局遊戲。兩者可以支持「這個 candidate 值得再驗證」，卻不能等同於「DQN 已經穩定通過正式評估」。Day 15 的工作，就是把這個差距變成一個固定、可重跑的實驗。

## 先固定要評估的模型

如果看完評估結果才回頭挑最好看的 checkpoint，評估就不再是對預先選定模型的檢查。因此 Day 15 先遵守 Day 14 final manifest 裡的規則：使用 `day14-final-vanilla-dqn-seed42` 在 `100,000` 個環境步數完成時保存的 final checkpoint。

這裡的 manifest 是記錄實驗配置、完成狀態和來源 artifact 的 JSON；checkpoint 則是保存某個訓練時刻模型權重的檔案。最新的 Day 14 final manifest 是本次 provenance（來源追蹤）的依據。它實際記錄的候選設定是 training seed `42`、learning rate `2e-4`、batch size `32`（每次模型更新一起處理的資料筆數）、train frequency `4` 和 `100K` budget；有效的 replay backend 是 `cpu`。把 Replay 直接放在 GPU 記憶體中的其他實驗仍是 Day 14 的系統效能證據，沒有在 Day 15 被偷偷換成另一個 checkpoint。

training seed 會影響模型初始化、探索和 Replay 抽樣；evaluation seed 則控制凍結模型在遊戲環境中遇到的隨機性。這次刻意把兩者分開：模型來自 training seed `42`，評估使用 `101`、`202`、`303` 三個 evaluation seed group，每組五局，共 15 局。每個 group 的第 1～5 局用 `seed + episode_index` 形成實際 reset seed，例如 `101`～`105`，所以每一局都能在 artifact 中追查。

這個候選也通過了 Day 14 的 Gate A（進入正式評估前的檢查）。用 batch size 32 的 10K profiling（量測訓練效能與品質的實驗）結果作為基準，最近 20 局平均 raw return 是 `1.50`；100K final run 的最後 20 局平均是 `6.15`。final summary 與 metrics 裡的必要 diagnostics 都是有限數值，選擇依據也保留了品質護欄（quality guardrails）；100K run 共記錄 329 局，不是只挑一局最高分。因此 Day 15 有理由驗證這個 candidate，但仍要把「訓練訊號變好」和「獨立 evaluation 一定更強」分開。

這種「先選模型，再選評估條件」的順序，是避免資料洩漏（data leakage）的基本界線：evaluation 結果可以告訴我們模型表現如何，但不能反過來改寫模型選擇規則。

原始的 Random 與 DQN 結果現在明確標記為 Evaluation Contract v1：FIRE 由 policy 負責、`epsilon = 0`，環境不自動代替 policy 發 FIRE。這些 `results.json` 保留作為歷史基準，後續診斷會寫到新的目錄，不覆寫 v1。

## 凍結模型，也凍結評估時的行為

訓練中的 DQN 會從 Replay Buffer（保存過去互動資料、供模型重複抽樣的資料區）取資料，透過 optimizer（根據誤差更新權重的工具）調整網路，並定期同步 target network（暫時固定的目標網路）。那時候的回報描述的是「模型一邊改變自己、一邊玩遊戲」。

Day 15 要看的則是固定權重的行動規則。正式 DQN 使用 greedy policy：模型輸出四個 action 的 Q-value 後，選擇數值最大的 action。Q-value 是模型對「從目前畫面採取某個 action 有多值得」的估計，不是機率；`epsilon = 0` 表示採取隨機 action 的機率為零，評估不再加入探索。

模型仍然需要把環境送出的畫面交給網路。`uint8` 是用 0～255 整數保存像素的資料型別，`float32` 是神經網路計算常用的 32 位元浮點數，而 tensor 可以先理解成適合模型運算的多維陣列。這次只在模型輸入邊界做這個轉換，並且用 `model.eval()` 加上 `torch.no_grad()`：

```python
state = observation_to_tensor(observation, device=device)
with torch.no_grad():
    q_values = model(state)
action = int(torch.argmax(q_values[0]).item())

observation, reward, terminated, truncated, _ = env.step(action)
episode_return += float(reward)
```

`torch.no_grad()` 關閉梯度追蹤，`model.eval()` 讓網路進入推論模式；兩者都不會更新 optimizer、target network 或 Replay Buffer。每局累積的是環境的 raw reward，也就是 Atari 原始遊戲分數，而不是訓練時可能使用的 reward clipping 分數。

正式 DQN evaluation 明確指定 `--device cuda`。這次 requested device 是 `cuda`，resolved device 是 `cuda:0`，GPU 是 NVIDIA GeForce RTX 4060 Laptop GPU；若 CUDA 不可用，程式會清楚失敗，不會靜默降回 CPU。Random 沒有神經網路推論，所以使用 CPU；這裡比較的是遊戲回報，不是兩個 runtime 的速度。

## Random 和 DQN 必須共用同一個 episode loop

Random policy 的作用是提供一個容易解釋的 baseline：如果凍結 DQN 連固定規則下的亂選 action 都贏不了，就不能只靠訓練曲線宣稱模型學會了。

但 baseline 只有在測量方式相同時才有意義。這次兩種 policy 共用 environment construction、影像 preprocessing、reset seed、raw reward 累積、`terminated`／`truncated` 判斷、統計和輸出格式；唯一不同的是 action 如何產生。這個共用的 episode harness，就是負責一局遊戲外層循環的程式，讓 policy 只需要回答「下一步選哪個 action」。

下面的結構圖顯示這個匯合點：DQN 先載入 Day 14 checkpoint 並驗證 CUDA，Random 直接建立 CPU policy；兩者之後走同一條 evaluation path。

[![Day 15 evaluation contract：DQN 與 Random 進入同一個 episode harness](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/3f599ba/assets/day15/evaluation-contract.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/3f599ba/assets/day15/evaluation-contract.png)

圖中的流程是依照實際 CLI 和 `evaluate_policy` 的結構整理出的 structural diagram，不是某一次 rollout 的畫面截圖。它回答的是「哪些條件在兩種 policy 之間保持不變」，而不是預測分數。

## `terminated` 和 `truncated` 都要留下來

一局遊戲結束時，Gymnasium 會回傳兩個不同的訊號。`terminated` 表示環境本身達到了終止條件；`truncated` 表示環境或 wrapper 因外部限制截斷了這局。對統計來說，兩者都代表 episode loop 可以進入下一局，但原因不能被混成一個模糊的 `done`。

Day 15 沒有另外加一個會改變正式分數的評估器步數上限（evaluator cap），而是讓 Arcade Learning Environment（ALE）的 Breakout-v5 使用自己的 episode 邊界。v1 的 DQN 有 5 局 `terminated`，另 10 局是由 ALE 的 `game_truncated` 標記的 TimeLimit；Random 則是 15 局 `terminated`。`complete = terminated or truncated` 只表示 episode loop 已結束，不表示 DQN 在 timeout 前仍然正常控制遊戲。

[![Day 15 episode loop：分開保存 terminated 與 truncated 狀態](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/3f599ba/assets/day15/evaluation-episode-loop.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/3f599ba/assets/day15/evaluation-episode-loop.png)

10 個 TimeLimit episode 的 agent steps 幾乎都在 `26,993`～`27,000`，所以平均長度 `18,131` 不能被當成「存活能力」的證據。這個數字只告訴我們 evaluator 等到了環境上限；回報、結束原因和中間的 action/observation 行為必須一起看。

## 平均值之外，還要看整個分布

這次固定 protocol 的實際結果如下。mean 是所有局回報的平均；median 是排序後位於中間的回報，較不容易被極端局拉動；std 是 standard deviation（標準差），用來描述回報的 spread，也就是每局離平均值有多分散。

| Policy | 局數 | terminated | truncated | TimeLimit | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 15 | 15 | 0 | 0 | 1.33 | 1.00 | 1.30 | 0 | 5 |
| DQN | 15 | 5 | 10 | 10 | 4.53 | 5.00 | 3.07 | 1 | 11 |

如果只畫兩根 mean bar，會看見 DQN `4.53` 高於 Random `1.33`，卻看不見每一局的差異。下面的圖直接讀取兩份 evaluation JSON：每個點是一局，箱型圖保留中間分布，菱形是 mean，短線是 median；右側則按 evaluation seed group 畫出平均和 population std。population std 把這次收集到的 episodes 當成要描述的整體，不把它包裝成多 training seeds 的不確定性估計。

[![Random 與凍結 DQN 的每局 raw return 分布，以及各 evaluation seed group 的平均與 spread](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/3f599ba/assets/day15/random-vs-dqn-returns.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/3f599ba/assets/day15/random-vs-dqn-returns.png)

圖中可以同時看到兩件事：DQN 在三個 evaluation seed group 的平均都高於 Random，median 也從 `1` 提高到 `5`；另一方面，DQN 的 spread 更大，且有 10 局長度碰到 ALE 的 TimeLimit。這份 evidence 支持「Day 14 checkpoint 在這批固定 episodes 中展現較高回報」，但不能單獨證明所有未來起始狀態都會得到同樣結果。

## 27K timeout 其實發生了什麼

要分辨「遊戲真的持續進行」和「畫面卡在某種狀態」，需要把一局中的每一步攤開來看。root-cause trace 讀取同一個 frozen checkpoint 的 Q-values、greedy action、ALE lives、raw reward、life-loss event、FIRE 時間點，以及前後 observation 的變化比例；lives 與 TimeLimit 都直接來自 ALE，沒有從分數猜測。

在 concrete seed `101` 和 `103`，life loss 都發生在 agent step `54`，但第一次 FIRE 分別延遲到 `16,449` 與 `16,446`，latency 是 `16,395` 與 `16,392` steps。這兩局最後都 TimeLimit；trace 中約 `96%` 的 observation 完全不變，greedy/action 約 `96%` 固定為 `RIGHT`，最長連續不變 observation 超過 `16,000` steps。相對地，terminated seed `102` 在 405 steps 結束，第一次 life loss 後 33 steps 就 FIRE，後續重發球多為 1 step 內。這組對照支持的 root cause 是 **life loss 後的 serve deadlock，加上 RIGHT action collapse**，不是「模型能正常存活很久」。

接著用同一個 checkpoint、同一組 15 個 concrete episode seeds、相同 preprocessing 和 raw reward 做兩個只供診斷的 ablation（只改一個行為因素的對照實驗）：

| Mode | epsilon | FIRE assist | mean return | median | terminated | TimeLimit | mean length | dominant action fraction | life-loss → FIRE latency | auto-FIRE |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Contract v1 | 0.00 | no | 4.53 | 5 | 5 | 10 | 18,131.33 | — | — | 0 |
| Diagnostic A | 0.00 | yes | 8.80 | 8 | 15 | 0 | 429.93 | 0.58 | 1.00 | 75 |
| Diagnostic B | 0.05 | no | 9.20 | 8 | 15 | 0 | 508.53 | 0.55 | 30.45 | 0 |

Diagnostic A 把 FIRE 放到 environment-side serve assist：initial serve 和每次觀察到 life loss 後都執行一次 FIRE。15 局全部正常 `terminated`，且 observation unchanged 的最長 streak 降到 3。Diagnostic B 不代替 FIRE，只加入 `epsilon = 0.05` 的隨機 action；它也讓 15 局全部結束，但平均 life-loss → FIRE latency 是 30.45 steps，代表少量探索偶爾能救回 serve，卻不是固定且可重現的語義。

這張圖把三組真實 artifact 放在同一個視野：上方看每局 raw return 的分布，右上分開 terminated 與 TimeLimit，左下比較平均 episode length，右下只比較兩種重發球策略的延遲。它的重點不是替 A 或 B 頒發較高分，而是顯示「環境明確處理 serve」和「靠隨機探索偶爾碰到 FIRE」是兩種不同的機制。

[![Day 15 FIRE、TimeLimit 與兩個 diagnostic 的真實結果比較](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/13a99f0/assets/day15/fire-time-limit-diagnostics.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/13a99f0/assets/day15/fire-time-limit-diagnostics.png)

因此 Contract v2 選 Option B：`fire_reset = true`，由 environment 在 initial serve 和觀察到 life loss 後執行 FIRE；`terminal_on_life_loss = false` 保留完整遊戲 episode，TimeLimit 明確以 `ale.game_truncated` 標記。這不是把 v1 分數改名，而是根據 trace 和 A/B evidence 定義下一個公平的 environment contract。v1 仍是 legacy baseline，Contract v2 保存在 `configs/eval/breakout_contract_v2.json`，Day 16 應直接載入同一份語義。

## 這次的答案是「有訊號，但還不能過度宣稱」

在 15 局固定 evaluation episodes 中，DQN 的 mean 和 median 都高於 Random，因此 Day 14 的 100K checkpoint 確實保留了可觀察的行動訊號。本次最準確的描述是：**DQN 的中心值較高，但 spread 仍有重疊**；不能只因平均值較高，就宣稱它已經在所有完整遊戲中穩定勝過 Random。

限制同樣清楚：目前只有一個 training seed（42），不是使用多個 training seeds 檢查穩健性的研究；DQN 和 Random 的結束方式也不同，DQN 有 10 局由環境時間限制截斷。這些限制不會讓 evaluation infrastructure 失效，但會限制我們能對模型下的結論強度。如果未來 DQN 沒有高於 Random，也不應立刻斷言卷積神經網路容量不足；還需要更長 training horizon、多個 training seeds 或後續模型比較。

## Day 15 留下的是一把可重用的尺

Day 16 會把 single-environment training 改成多環境、批次 action inference（一次替多個 observation 選 action）和批次 GPU Replay insertion。系統最佳化可能提高 throughput，但也可能因 autoreset（一局結束後自動開始下一局）、global step（已走過的環境步數）、done semantics（如何解讀 terminated/truncated 結束訊號）或 Replay ordering（資料寫入 Replay 的先後順序）改變 policy quality。

因此 Day 16、Day 17、Day 18 和 Day 20 都應重用本日的 evaluation contract：相同的 Breakout preprocessing、concrete evaluation seeds、每組 episode 數、greedy epsilon、raw reward、`terminated`／`truncated`／TimeLimit semantics 和 JSON/CSV schema。這樣下一次比較的差異才主要來自 training system 或 DQN variant，而不是評估規則被換掉。

可重建的原始結果保存在 `evaluations/day15-random-baseline/` 和 `evaluations/day15-dqn-cuda/`；圖表由兩份 `results.json` 重新產生，完整 provenance 和 GPU metadata 則保存在 DQN result 與 milestone report 中。這些 artifacts 讓後續比較可以重新使用同一份 evaluation contract。

Day 15 真正完成的不是替模型頒發「已學會」的稱號，而是先把「怎麼知道它真的變好」這件事固定下來。下一個問題才是：向量化訓練能不能提高系統效率，同時保住這份 policy-quality baseline？
