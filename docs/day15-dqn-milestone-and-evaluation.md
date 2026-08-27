# Day 15｜DQN 到底有沒有真的學會？先把評估做成一個可重複的實驗

Day 14 的 100K 訓練曲線出現了值得追蹤的訊號：`2e-4` 的 learning rate（每次根據誤差調整模型權重時所走的步幅）在同一個 training seed 下，後段回報高於另外兩組設定。再加上 100K checkpoint 的 GIF 看起來比早期模型更會打，直覺上很容易得到一句話：DQN 已經學會 Breakout。

但這句話其實把兩種不同的證據混在一起了。訓練曲線是在模型還持續更新時量到的；GIF 則只有一個 checkpoint、一個 evaluation seed 和一局遊戲。它們可以支持「這個模型值得再驗證」，卻不能回答「換幾個起始條件後，它是否仍然比亂玩好」。

Day 15 的問題因此改成：**把 Day 14 的模型凍結後，用一套事先固定的規則，讓 Random policy 和 DQN 走完全相同的評估流程，結果會是什麼？**

## 訓練時的回報，不能直接當成模型成績

訓練中的 DQN 還在做三件評估時不應該做的事：從 Replay Buffer（保存過去互動資料，也就是 transition，供模型重複抽樣的資料區）抽樣、透過 optimizer（根據誤差計算權重更新的工具）更新神經網路權重，以及按 training schedule 改變探索比例。target network（暫時固定的目標網路）也會在訓練期間同步；那時候看到的回報，代表「模型一邊改變自己、一邊和環境互動」的結果。

正式評估則要看另一件事：**同一份權重固定不動時，它會怎麼行動？** 這裡的 manifest 是記錄實驗配置、完成狀態和來源 artifact 的 JSON 檔；checkpoint 則是保存某個訓練時刻模型權重的檔案。因此這次選用 Day 14 final manifest 指向的 `day14-final-vanilla-dqn-seed42`，固定使用完成 `100,000` 個環境步數時的 checkpoint。沒有回頭從 50K、75K、100K 的局中挑分數最高的那一個。

這個選擇規則很重要。若先看完評估結果才挑 checkpoint，評估就不再是對預先選定模型的檢查，而是另一種形式的資料洩漏。

Day 14 的 `42` 是 training seed：它影響初始化、探索以及 Replay 抽樣。Day 15 改用 `101`、`202`、`303` 三個 evaluation seed group，讓凍結的模型面對另一組環境隨機性。每個 group 執行五局；同一 group 的第 1～5 局使用 `seed + episode_index` 形成實際 reset seed，例如 `101`～`105`。這樣五局不會只是同一個環境初始狀態的重複，也能在結果中追查每一局的來源。

## 凍結模型，也凍結評估時的行為

評估時的 DQN 使用 greedy policy，也就是 `epsilon = 0`；epsilon 是採取隨機 action 的機率，因此這裡完全不探索。模型輸出四個 action 的 Q-value 後，直接選最大值。這裡的 Q-value 是模型對「從目前畫面採取某個 action 有多值得」的估計，不是機率，所以不需要把它轉成機率分布。

程式在模型邊界把 `uint8` 畫面轉成神經網路使用的 `float32` tensor；`uint8` 是用 0～255 整數保存像素的資料型別，`float32` 則是神經網路計算常用的 32 位元浮點數，而 tensor 可以先理解成適合模型運算的多維陣列。除此之外，不會更新模型、optimizer、target network 或 Replay Buffer。`model.eval()` 讓模型進入推論模式，`torch.no_grad()` 則關閉梯度追蹤：

```python
state = observation_to_tensor(observation, device=device)
with torch.no_grad():
    q_values = model(state)
action = int(torch.argmax(q_values[0]).item())

observation, reward, terminated, truncated, _ = env.step(action)
episode_return += float(reward)
```

這段程式的重點不是 `argmax` 本身，而是 action 選擇和 `env.step()` 之間沒有任何訓練更新。每局只累積環境回傳的原始 reward；訓練時為了讓數值較容易學習而使用的 reward clipping，不會被帶進正式分數。

正式 DQN evaluation 也明確指定 `--device cuda`。這次 artifact 記錄的 requested device 是 `cuda`，resolved device 是 `cuda:0`，GPU 為 NVIDIA GeForce RTX 4060 Laptop GPU。若 CUDA 不可用，程式會直接失敗，不會偷偷改用 CPU。Random policy 沒有神經網路推論，因此它使用 CPU；這不是兩種 policy 的效能比較。

## Random baseline 必須走同一條路

Random policy 的作用不是證明「隨機也很強」，而是提供一個可解釋的下界：如果凍結 DQN 連固定規則下的亂選 action 都沒有明顯改善，那 Day 14 的訓練訊號還不足以稱為 milestone。

比較成立的前提，是兩者不能各自擁有一套稍微不同的 episode loop。這次 Random 與 DQN 共用 environment factory、preprocessing、reset seed、raw reward 累積、`terminated`／`truncated` 判斷和結果格式；唯一不同的是「如何選 action」。這個共用的 episode harness，可以把它理解成負責一局遊戲外層循環的程式，讓 policy 只需要提供下一個 action。

下圖是實際實作中的結構性流程：DQN 先解析裝置並檢查 CUDA，Random 則可留在 CPU；兩者之後匯入同一個 episode harness。

[![Day 15 evaluation contract：DQN 與 Random 進入同一個 episode harness](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/bcd9c5b/assets/day15/evaluation-contract.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/bcd9c5b/assets/day15/evaluation-contract.png)

圖中真正值得注意的是匯合點：policy 的差異在 action 選擇處結束，之後的環境、seed、回報與輸出規則都相同。這讓最後的差異比較接近「policy 造成的差異」，而不是兩套測量程式的差異。

## 一局什麼時候才算結束？

Gymnasium 的 `terminated` 表示環境本身已經結束這局；`truncated` 表示外部時間限制或環境 wrapper 截斷了這局。兩者都必須被保留，因為它們對後續學習與統計的意義不同。

Day 15 另外設了一個 `10,000` 步的 evaluator safety cap，避免某個 policy 卡在永不結束的狀態時，整個實驗無限等待。這個 cap 不是環境回傳的 `truncated`，所以結果額外寫入 `evaluator_truncated=true` 和 `complete=false`，不把它偽裝成自然終局。

[![Day 15 episode loop：保留環境終止狀態並區分 evaluator cap](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/bcd9c5b/assets/day15/evaluation-episode-loop.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/bcd9c5b/assets/day15/evaluation-episode-loop.png)

這個區分在真實結果裡不是形式上的欄位。DQN 的 15 局中有 5 局由環境自然結束，另外 10 局在 10,000 步 safety cap 停止；Random 則是 15 局都自然結束。若忽略這個差異，DQN 的長 episode 會被錯誤地當成「完整遊戲表現」。

## 平均值之外，還要看中位數和 spread

這次固定 protocol 的實際結果如下。`mean` 是所有局回報的平均；`median` 是排序後位於中間的值，較不容易被少數極端局拉動；`std` 是 standard deviation（標準差），用來描述每局回報距離平均的典型程度，也就是 spread。這裡使用 population standard deviation，描述的是本次完整收集到的 episode set，而不是估計一個更大母體的 sample standard deviation。

| Policy | 局數 | 自然結束 | evaluator cap | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 15 | 15 | 0 | 1.33 | 1.00 | 1.30 | 0 | 5 |
| DQN | 15 | 5 | 10 | 4.53 | 5.00 | 3.07 | 1 | 11 |

DQN 的平均和中位數都高於 Random，代表這批 samples 確實看到了較高的回報中心；但 DQN 的 spread 也更大，而且十局不是自然終局。平均值因此不能單獨支持「DQN 穩定超過 Random」這個更強的結論。

下圖不只畫兩根 mean bar。每個點都是 evaluation JSON 中的一局；箱型圖顯示中間分布，菱形是 mean，短線是 median，叉號則代表 evaluator cap。右側再按 evaluation seed group 顯示平均和 spread，避免某一組 seed 的結果單獨主導故事。

[![Random 與凍結 DQN 的每局 raw return 分布，以及各 evaluation seed group 的平均與 spread](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/bcd9c5b/assets/day15/random-vs-dqn-returns.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/bcd9c5b/assets/day15/random-vs-dqn-returns.png)

從圖中可以同時看到兩件事：DQN 的回報點大多落在較高的位置，且各 seed group 的平均都高於 Random；另一方面，DQN 也有一批長時間維持遊戲未結束的局，這正是 spread 和 episode status 必須一起呈現的原因。這張圖支持的是「本次固定 samples 的差異」；它不能證明所有未來初始狀態都會得到同樣結果，也不能代替多個 training seeds 的 robustness study。

## 這次的 milestone 判斷

因此 Day 15 的答案不是簡單的「贏」或「沒贏」。在固定的 15 局中，DQN 的 mean `4.53` 高於 Random 的 `1.33`，median 也從 `1` 提高到 `5`，說明 Day 14 的 checkpoint 確實保留了可觀察的行動訊號。

但若 milestone 的意思是「在多局完整遊戲中穩定優於 Random」，這次仍應保守記為 **DQN 觀察平均較高，但有 evaluator cap 的局，尚不能當作完整終局的穩定優勢**。這不是把結果判成失敗，而是把已證明的範圍說清楚：目前只有一個 training seed，且 10/15 局沒有自然結束。

如果未來結果低於 Random，也不應立刻跳到「CNN 太小」或「演算法一定錯」；比較合理的做法是保留這份 evaluation evidence，再把訓練延長、增加 training seeds，或在同一 protocol 下比較後續模型。單一 seed 的 negative result 同樣不能直接定位原因。

## Day 15 留給 Day 16 的不是一個分數，而是一份基準

Day 16 會把 single-environment training 改成多環境、批次 action inference（一次替多個 observation 選 action）和批次 GPU Replay insertion。系統改造後可能更快，但速度變快不代表 policy quality 沒有退步；`global step`（已走過的環境步數）、autoreset（一局結束後自動開始下一局）、done semantics（如何解讀 terminated/truncated 結束訊號）和 Replay ordering（資料寫入 Replay 的先後順序）都可能改變學習結果。

所以 Day 16 只要重用這份 contract：相同的 environment preprocessing、`101/202/303` seed groups、每組五局、`epsilon=0`、raw Atari reward、相同的 completed/cap 欄位和 JSON/CSV schema，就能把 systems optimization 前後的 policy quality 放在同一把尺上。之後的 Double DQN、Dueling Network 也應該沿用這套評估規則，讓差異來自模型方法，而不是 evaluation protocol。

可重建的結果保存在 `evaluations/day15-random-baseline/` 與 `evaluations/day15-dqn-cuda/`；圖表由兩份 `results.json` 重新產生。正式 DQN 評估使用：

```powershell
python evaluate_dqn.py --policy random --config configs/eval/breakout_eval.json

python evaluate_dqn.py `
  --checkpoint assets/day14/final-runs/day14-final-frozen-100k/day14-final-vanilla-dqn-seed42/checkpoints/step-00100000.pt `
  --config configs/eval/breakout_eval.json `
  --device cuda

python visualize_day15_evaluation.py `
  evaluations/day15-random-baseline/results.json `
  evaluations/day15-dqn-cuda/results.json `
  --output assets/day15/random-vs-dqn-returns.png
```

這樣 Day 15 真正完成的不是替模型頒發「已學會」的稱號，而是建立一個可重複、可追查、也能誠實暴露限制的評估方法。下一個問題才會是：在不改變這把尺的前提下，向量化訓練能不能改善系統效率，又保住這個 policy-quality baseline？
