# Day 14｜為什麼 10K 不夠：把超參數比較拉長到 100K

[Day 13](day13-debugging-unstable-rl-training.md) 的 10,000 environment steps 已經回答了一個重要問題：程式有沒有壞。那次 CUDA diagnostic run 完成 48 個 episode，training loop 有 optimizer update、Target Network 有同步，Replay Buffer 和 epsilon schedule 也在運作；但平均 raw return 約為 `1.40`，沒有可靠的學習趨勢。

這個結果很容易被誤讀。10K 足以發現 loss、Q-value、gradient 變成非有限值、模型根本沒有更新，或 CUDA 設定錯誤；卻不代表 10K 足以判斷三個 hyperparameter config 誰比較好。如果 baseline 的短期平均是 `1.15`、另一個 variant 是 `1.75`，差異可能只是遊戲回合剛好落在不同的隨機波動，而不是 learning rate 真的造成了穩定改變。

所以 Day 14 的問題被重新寫得更精確：**先用 10K 做 health screening，再把相同的 learning-rate comparison 拉長到 100K，觀察 learning curve 和數值診斷是否開始分化。** 100K 仍然不是保證學會 Breakout 的數字；它只是比 Day 13 長十倍的 observation horizon，讓我們有機會看到短跑看不到的變化。

## 10K 能證明什麼，不能證明什麼？

這裡的 environment step 是 Agent 執行一次 action、環境回傳一次結果的互動單位。10,000 steps 可以當成一個便宜的 short screening：如果 loss、Q-value、gradient 或 Replay Buffer occupancy 立刻異常，就不用浪費更長時間；如果 CUDA request 無法解析，也應該在這一層被擋下來。

但「沒有立即爆掉」和「這個設定比較好」是兩個不同命題。Breakout 的 reward 很稀疏，一局的長度也會變化；10K 只涵蓋有限的遊戲互動，最後幾個 episode 的分數很容易支配平均值。把這樣的 final return 排名，會把 health check 偽裝成 model comparison。

這也是為什麼新的 workflow 把 screening 和 main comparison 分成不同的 stage。screening 只回答「能不能正常跑」；main comparison 才回答「在更長的訓練時間內，曲線是否出現可解釋的差異」。

[![從 Day 13 的 10K diagnostic、10K screening 到 Day 14 100K main comparison 的決策流程](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/f0b077c/assets/day14/budget-stages.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/f0b077c/assets/day14/budget-stages.png)

圖中的分支是實際實驗規則，而不是某次 run 的裝飾性流程圖：10K 通過 health checks 才能進入 100K；100K 之後若仍沒有可靠 signal，合法的結論是「目前無法分辨」，不是強行挑一個 winner。

## 一次只改一個因素，100K 只延長 observation horizon

這次仍然採用 one-factor-at-a-time。baseline 是未改動的參考設定；variant 是只覆寫一個 hyperparameter 的比較設定。三個 main configs 只改 learning rate：

| run | learning rate | 其他條件 |
|---|---:|---|
| baseline | `1e-4` | 固定 seed、environment、replay、epsilon、target update、reward clipping、device、precision |
| learning-rate-low | `5e-5` | 同上 |
| learning-rate-high | `2e-4` | 同上 |

learning rate 是每次模型更新調整權重的步幅；它太小可能讓學習變慢，太大可能讓 Q-value 或 gradient 變得更躁動。這次不是同時改 epsilon decay 或 Target Network interval，所以即使結果只是一個初步訊號，也還能回答比較單純的問題：在同一個 100K horizon 下，learning rate 的差異是否值得繼續追蹤？

三個 run 固定 seed `42`。seed 是控制初始化與抽樣的整數起點，能讓實驗條件更容易重現；它不是把 CUDA 執行變成跨硬體 bit-exact deterministic 的保證。三個 run 也都使用 `requested_device=cuda`，解析成同一張 `cuda:0` 的 NVIDIA GeForce RTX 4060 Laptop GPU，precision 是 `float32`。

## 100K 不只看最後一個數字

100K 的價值不在於最後一列 summary 比 10K 更大，而在於可以回頭問「變化何時開始」。因此 main config 每 25,000 steps 保存一次 checkpoint，CSV 則保留每個 environment step 的 metrics。comparison report 會取 25K、50K、75K、100K 附近的實際 row，並同時保存 loss、Q、Target、gradient、epsilon 和 SPS。

回合分數仍然只在 episode 完成時出現，所以 return 曲線的每個點都是實際完成的 episode，不會把缺少的值補成零。為了避免看到結果後改規則，這次固定使用最後 20 個 completed episodes 的 mean/median，以及所有 20-episode rolling windows 中的最高平均值。recent trend 則把最後 20 局分成前後兩半，報告後半平均減去前半平均的變化。

## 100K main comparison 的真實結果

下表來自 `experiments/day14-cuda-lr-100k-main/comparison.json`。三個 run 都完成 100,000 steps，report 也確認 stage 是 `main`、三個 requested/resolved device 相同，符合正式 CUDA main comparison 條件。

| run | episodes | 最近 20 局 mean | median | 最佳 rolling20 mean | recent trend Δ | SPS | wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 372 | 4.60 | 4.50 | 4.95 | -2.40 | 162.48 | 615.44 s |
| learning-rate-low | 375 | 2.30 | 2.00 | 3.20 | -0.60 | 151.89 | 658.39 s |
| learning-rate-high | 313 | 7.40 | 6.50 | 9.25 | -0.60 | 152.85 | 654.24 s |

這次 100K 和 10K 的差異不是「終於得到一個永遠正確的排名」，而是曲線開始提供更長時間尺度的 evidence。high learning rate 的 rolling return 明顯高於 low，baseline 落在中間；這讓 high 成為值得交給 multi-seed evaluation 的 candidate。但三個 recent trend Δ 都是負值，代表最後 20 局相對於那 20 局的前半段並沒有繼續上升。這提醒我們：即使 100K 的差距比 10K 更有資訊，也不能只看一個區間就宣稱已經穩定學會。

這張圖回答的是：**在相同 100K steps 下，三個 learning-rate run 的 raw return 是否開始沿著不同的 learning curve 前進？** 淡色點是每局實際完成時的 raw episode return；粗線是固定 20-episode rolling mean。x 軸仍然是 environment step，而不是 episode index。

[![100K main comparison 中三個 learning-rate run 的 raw return 與 20-episode rolling mean](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/f0b077c/assets/day14/experiment-return-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/f0b077c/assets/day14/experiment-return-comparison.png)

圖中 high 的 rolling curve 在後段維持較高，baseline 次之，low 較低；這支持「100K 比 10K 更能看出候選差異」這個觀察。它不能支持「`2e-4` 在所有 seed 都最佳」，也不能排除更長訓練後排名改變。SPS 是每秒完成多少 environment steps 的 throughput，wall-clock 是實際經過的時間；它們是執行成本，不是遊戲品質。high 的速度稍快，不代表它因此學得比較好。

## Q、Target、gradient 在較長 horizon 下怎麼走？

Q-value 是模型對 action 長期價值的估計；Target mean 是用來形成 Bellman learning target 的另一組估計；gradient norm 則是一次更新中梯度總量的大小。這些診斷不能單獨判斷策略好壞，但能幫助區分「學得慢」和「數值開始不穩定」。

| run | Q mean：25K → 100K | Target mean：25K → 100K | loss max | gradient max |
|---|---:|---:|---:|---:|
| baseline | 0.626 → 1.518 | 0.617 → 1.546 | 0.0433 | 1.8043 |
| learning-rate-low | 0.505 → 1.009 | 0.513 → 1.024 | 0.0483 | 1.5869 |
| learning-rate-high | 0.600 → 1.729 | 0.601 → 1.737 | 0.0325 | 1.4534 |

三個 run 的 Q 與 Target 都隨 horizon 增加，且 report 沒有非有限值；因此不能把「Q 變大」直接稱為爆炸。high 在 50K 附近的 gradient norm 量測值約為 `0.538`，高於同一 milestone 的 baseline `0.133` 和 low `0.056`，但它後來又回到較低尺度。合理的判讀是：high learning rate 讓價值估計的發展更快、也值得持續監看；這還不是「已經不穩定」或「一定更好」的證明。

epsilon 在這組 config 中於前 10K steps 下降到 `0.05`，之後 90K 大多在低探索機率下執行。這是現有 epsilon schedule 的設計條件，不是這次 learning-rate comparison 的變因；如果下一輪要研究探索速度，應另開一個只改 epsilon decay 的 batch。

## 10K screening 與 100K main 必須分開保存

舊的 10K learning-rate artifacts 仍然保留，但現在明確標記為 `stage=screening`、`budget_level=short_screening`。它們可以回答三個 config 是否能正常啟動、更新、寫出 metrics 和完成 CUDA metadata；它們不能拿來和新的 100K recent mean 放在同一張 ranking table 裡。

main manifest 則標記為 `stage=main`、`budget_level=main_day14`，並由 report 檢查三個 run 的 `total_steps`、stage、requested/resolved device 是否一致。這個分層解決了一個常見的實驗錯誤：把不同 observation horizon 的 final return 當成同尺度數字比較。

## 100K 之後仍然不能把單一 seed 當成答案

這次的 100K evidence 支持 high learning rate 成為下一輪候選，也支持 low 在這組條件下暫時落後；但這仍然是 single-seed development evidence。不同 seed 可能改變 episode 結束時間、Replay Buffer 內容與每次 update 的抽樣，CUDA 和不同硬體也可能產生細微數值差異。

如果下一輪 multi-seed 仍然看到 high 的 return curve 高於其他設定，而且 Q/Target/gradient 沒有出現不可接受的失控，再把它交給更正式的 milestone evaluation 才合理。如果 100K 後的差異在多 seed 消失，或所有 config 都沒有可靠 trend，也應該保留「目前無法分辨」這個結果；只有在 100K 已經提供清楚但仍不足的 signal 時，才值得把少數候選延長到 250K 或更長。

Day 14 因此沒有把調參變成「跑得久就一定找到答案」。它建立的是一個可分層的判讀方式：10K 負責健康檢查，100K 負責觀察 learning dynamics，25K 到 100K 的 diagnostics 負責揭露穩定性，最後仍由多 seed evaluation 決定這個 signal 是否值得相信。

下一篇會把這個 reference/candidate distinction 放進固定的 milestone evaluation，加入 random baseline，檢查 100K 觀察到的差異是否能在更嚴格的 protocol 下重現。
