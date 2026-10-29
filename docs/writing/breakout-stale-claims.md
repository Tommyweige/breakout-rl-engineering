# tommypan.com 打磚塊系列：錯誤與過時敘述清單

核對日期 2026-10-29。範圍是站上 8 個公開頁面（專案頁、總覽、核心系列 5 篇、reward shaping）。「已不成立」= 依現在的程式與資料，該句不能照原樣寫；「過時」= 句子本身當時沒錯，但專案狀態已經往前走，頁面沒有跟上；「已核對」= 這次逐條驗算過、仍然成立，不要順手改掉。

頁面上的數字大多能在鐵人賽版本找到原始表格，因此驗算方式有兩種：站上句子對鐵人賽表格、以及站上句子對 `main` 的程式碼。引用格式為 `檔案:行號`（`main` @ `a8445df`），或註明分支。

---

## A. 已不成立

### A1. 「加入自動發球並保留隨機動作後」把兩個對照組合併成一個

**位置**：`writing/building-debugging-evaluating-dqn`，「為什麼高分和長局數會誤導人？」一節末段。

**原文**

> 加入自動發球並保留隨機動作後，DQN 在相同的 15 個起點都能正常結束，解決了掉命後長時間等待的問題。

**問題**：實驗裡沒有這個組合。Day 15 的對照表有三列，兩組各改一個變因：

| 模式 | epsilon | 環境自動 FIRE | 平均分數 | 正常結束 |
| --- | ---: | :---: | ---: | ---: |
| 原始 greedy 評估 | 0.00 | 否 | 4.53 | 5/15 |
| FIRE assist | 0.00 | 是 | 8.80 | 15/15 |
| 保留探索 | 0.05 | 否 | 9.20 | 15/15 |

「自動發球」那一組 epsilon 仍是 0；「保留隨機動作」那一組沒有自動發球。句子把前者（機制）和後者（第二組）併在一起，讀者會以為只要加自動發球、又保留 5% 隨機就能全部正常結束。

**正確說法**：讓環境自動發球、epsilon 仍為 0 時，15 局全部正常結束；另一組不自動發球、保留 epsilon 0.05 也全部正常結束，但最後採用的是讓環境負責發球，因為把發球交給隨機探索會讓「一局」長度受運氣影響。

**證據**：`docs-ironman-series` 分支 `articles/day15-dqn-milestone-and-evaluation.md:115-131`；站上同頁前面已經正確寫過「掉命後等了 16,395 次操作才重新發球」，所以只需要修這一句。

### A2. 「模型沒有預先寫好的接球路線」已經不適用於整個展示

**位置**：`projects/breakout-rl-engineering`，「讓模型從畫面學習怎麼玩」一節首句。

**原文**

> 模型沒有預先寫好的接球路線。它讀取最近幾張遊戲畫面，選擇發球、往左、往右或不動，再根據遊戲回饋調整之後的選擇。

**問題**：展示頁的 AI 有四個難度，其中 `Unbeatable` 不是 DQN，而是完全手寫的像素控制器：從畫面找球與球拍、估速度、把球的落點算到球拍平面，再決定往哪邊推。它沒有訓練、沒有權重，而且每個畫面決策一次（其餘難度是每 4 個畫面一次）。所以「接球路線是預先寫好的」在 `Unbeatable` 這一格是成立的，句子卻是全稱。

**正確說法**：預設難度跑訓練好的 DQN；`Unbeatable` 改由不經訓練的像素控制器決策。若要保留原本那句，需限定成「DQN 這一邊沒有預先寫好的接球路線」。

**證據**：
- `web/src/app/App.ts:467`（`const vision = ... === 'unbeatable'`）
- `web/src/app/App.ts:530`（`unbeatable` 直接回傳 `this.visionController.select(...)`，不經過模型）
- `web/src/app/App.ts:838`（`unbeatable` 的 action repeat 是 1，其餘是 4）
- `web/src/app/difficultyPolicy.ts:3`（四個難度）、`:25`（預設 `hard`）
- `web/src/inference/PredictiveVisionController.ts`
- PR #26 `50ee352` 引入控制器、PR #64 `b1a5224` 接進展示、PR #68 `a8445df` 修掉球拍抖動

### A3. 「預設由模型選擇動作」少寫了難度注入的失誤

**位置**：`writing/browser-deployment-for-dqn`，「人類玩家和模型是在同一場比賽嗎？」。

**原文**

> Agent 看自己的畫面，由模型選擇動作。

**問題**：DQN 的動作會先經過難度層：`Easy` 30%、`Medium` 15%、`Hard` 5% 的非發球決策會被換成隨機移動（發球動作不注入失誤），只有 `Unbeatable` 是純策略。預設難度是 `Hard`，所以「由模型選擇動作」大約只有 95% 成立，而頁面把展示當成模型能力的證據。

**正確說法**：Agent 由模型選擇動作，難度設定會依比例把部分決策換成隨機移動；`Unbeatable` 則完全交給像素控制器。

**證據**：`web/src/app/difficultyPolicy.ts:13-18`、`:25`；`web/src/app/App.ts:536`（`difficultyPolicy.select(policy)`）。

---

## B. 過時：句子成立，但專案已經往前走了

### B1. 全站沒有控制器這條路線

**位置**：`projects/breakout-rl-engineering` 與 `writing/breakout-rl-engineering-30-days` 的成果段落。

**原文**

> 這個專案從 DQN 訓練開始，一路做到模型比較、推論最佳化與瀏覽器遊戲，也在過程中發現模型掉命後卡在等待畫面的問題。（總覽）

> 專案將遊戲設定、模型訓練、逐局評估與瀏覽器部署接成一條完整流程。（總覽，末節）

**問題**：兩句都把專案描述成單一條學習路線。10 月 1 日之後多了第二條不經訓練的控制器路線，10 月 29 日又多了一條凍結視覺模型的 session；這些都不在頁面上。頁面最後更新日是 2026-10-06，早於控制器修正（10-08）與 Laya session（10-29）。

**建議**：在「高分之外，還要知道遊戲裡發生了什麼」之後補一節，或直接把下面的新文章接上。至少不要讓「一路做到」讀成「只做了這些」。

**證據**：頁面 metadata「更新 2026年10月6日」；`git log` 的 `50ee352`（10-01）、`a8445df`（10-08）。

### B2. 137.47 不是控制器現在的成績

**位置**：站上沒有出現這個數字；但它會從專案的凍結紀錄被引用進來，而凍結紀錄寫的是舊版。

**會引用到的原文**（`docs/final-results/README.md:29-30`，`codex/issue73-closeout` 分支）

> RGB controller mean | 137.47 (median 95, max 341) | v3, frame_skip=1 | merged PR #26 | historical reported

> RGB controller canonical clears | 0 / 15

**問題**：這是 PR #26 `50ee352` 的評估。PR #68 `a8445df` 修了兩個控制器問題：速度原本用相鄰兩張畫面的差分估計，像素捨入會讓方向抖動（改成跨兩個 native frame 取速度錨點，遇到反彈就重設）；球快接到時控制器仍為了零點幾像素的誤差往反向推，球拍在落點附近來回抖（改成進入球拍半寬就以 NOOP 釋放）。用同一個評估器、同一份 config 與 contract、同一組官方 15 個 seeds 重跑，結果是 **mean 721.8、median 810、max 864、canonical clear detection 4/15（seeds 202、304、306、307）**。

**正確說法**：137.47 是 2026-10-01 那一版的數字；10-08 修正後的同條件重跑是 721.8，15 局中有 4 局被偵測為打穿兩面牆。兩者不可混用，也不要把 137.47 當成「規則式方法的上限」——那 15 局最高只有 341 分，連第一面牆的 432 都沒打到，所以 0 次 clear 是控制器還沒打完，不是尺壞了。

**證據**：`git show a8445df -- breakout_rl/vision_controller.py`（`_velocity_anchor` 與 `catch_margin` 兩段）；`closeout-final` 分支 `docs/clear-semantics.md` 的「137.47 的結論」兩張逐局表（`50ee352`: mean 137.4667、median 95、max 341、0/15；`77a0c7b`，產品程式碼與 `a8445df` 相同: mean 721.8、median 810、max 864、4/15）；`closeout-final` 的 `audits/browser-clear/native-15/episodes.csv`。`codex/issue73-closeout` 的凍結 README 仍把 137.47 標成 `historical reported`，main 上沒有這兩份紀錄——這就是為什麼站上若引用專案紀錄，會引用到舊數字。

### B3. 30.9 需要標上契約，否則會和 721.8 相撞

**位置**：`writing/scaling-dqn-training-and-variants`，末節。

**原文**

> 51.4 分是挑選模型時看到的結果。選定模型後，再讓它從一組未參與挑選的 15 個起點開始遊玩，平均分是 30.9，中位數是 32.0

**問題**：這是 contract v2（每 4 個 emulator frame 決策一次）。同一個 DQN 家族放到 contract v3（每個 frame 決策一次、上限 108,000 次）從頭訓練 200 萬步，15 局是 **mean 6.93、0 clear**。不標版本的話，讀者會拿 30.9 去比 721.8，得出「訓練輸了五倍」的結論，而那是跨契約比較。

**正確說法**：30.9 是 v2 的 holdout；v3 下的 DQN baseline 是 6.93（median 7、max 12、0 clear）。兩個數字不能並排排名，原因見新文章的〈721.8 可以直接和 30.9 比嗎？〉一節。

**證據**：`configs/eval/breakout_contract_v2.json:5`（frame_skip 4）、`:23`（agent_step_limit 27000）；`configs/eval/breakout_contract_v3.json:5`（frame_skip 1）、`:22-23`（108000 / 108000）；`configs/training/issue21_contract_v3_baseline.json`（`codex/issue-21-contract-v3-baseline` 分支，Dueling Double DQN、單 seed、2,000,000 steps、每 checkpoint 15 局）。

### B4. 4.53 對 1.33 屬於 contract v1

**位置**：`writing/breakout-rl-engineering-30-days` 與 `writing/building-debugging-evaluating-dqn` 開頭。

**原文**

> 早期測試中，DQN 平均得分 4.53，高於隨機選擇動作的 1.33。（總覽）

> 早期測試裡，一個 DQN 平均拿到 4.53 分，隨機選動作的玩家只有 1.33 分。（building-debugging，同一組數字的另一種寫法）

**問題**：這批數字是舊規則（模型自己負責發球、沒有後來的 sticky action 設定）。同一個系列後面的 9.00 / 6.07 / 2.33 已是 contract v2。數字本身沒變，但頁面沒有標版本，而這一頁正是最容易被拿來跨版本比較的地方。

**正確說法**：加上「在還沒固定發球規則的版本」；或直接寫「舊規則下 4.53 對 1.33，固定規則後是 9.00 對 1.73」。

**證據**：`docs-ironman-series` 的 `articles/day15-dqn-milestone-and-evaluation.md:160`（4.53 / 1.33 屬舊 Contract v1）；`articles/day16-vectorized-dqn-training.md:147-150`（v2 的 1.73 / 9.00 / 6.07 / 2.33）。

### B5. 用語：canonical clear detection 不等於 verified clear

**位置**：全站目前沒有 clear 相關敘述；這是補寫時必須遵守的規則，不是既有錯誤。

**現況**：15 局中有 4 局被評估器偵測為 clear（864 分、兩面磚牆打穿），但那是 canonical clear detection。判定由 `BreakoutCompletionDetector` 讀 ALE RAM 計分板與 raw reward 兩者一致才成立，runtime 或兩者不一致就 fail closed 成 unknown。`verified_clears` 是另一個門檻：artifact gate 另外要求 evaluation artifact 通過 contract、source 與 provenance 驗證，而 `evaluate_predictive_controller` 沒有做這一層。所以 repository 的 `Verified full clears recorded` 仍是 **0**。

兩個詞的差別不是「有沒有人真的破關」，而是「這次評估的紀錄有沒有通過 artifact provenance」。寫文章時可以寫「偵測到 4 次 clear」；不能寫 `verified_clears` 或「已驗證破關」。

順帶一提：`codex/issue73-closeout` 的凍結 README 說偵測器的正向路徑「只用合成 fixture 驗證」，`closeout-final` 的 `docs/clear-semantics.md` 已經改成用兩條真實 ALE 破關軌跡（v2、v3）驗證，並列出六個 detector mutant 全被殺。兩份紀錄不一致，寫作時只要守住「4 次 canonical clear detection、0 verified_clears」就不受影響。

**證據**：`docs/clear-semantics.md`（`closeout-final`，兩個定義與驗證素材）；`docs/final-results/README.md:25`（`Verified full clears recorded | 0`）、`:123`（canonical detections are not `verified_clears`）、`:116`（best episode 341）；`audits/browser-clear/native-15/report.md`（4/15、mean 721.800）；契約 v3 的評估器輸出欄位是 canonical clear detection。

### B6. Live Demo 連結不一致

**位置**：專案頁用 `.com`，總覽頁與瀏覽器篇用 `.dev`。

**原文**

> 你可以直接開啟遊戲，自己玩，也看看模型怎麼玩。（專案頁，連結指向 `https://breakout.tommypan.com/`）

> 打磚塊網頁版在瀏覽器中執行模型與遊戲。（總覽，連結指向 `https://breakout.tommypan.dev/`）

> 你可以直接遊玩打磚塊網頁版，並排觀察人類玩家與模型各自的操作和得分。（瀏覽器篇，同樣連 `.dev`）

**問題**：不是死連結。實測 `https://breakout.tommypan.dev/` 回 301 轉到 `https://breakout.tommypan.com/`，所以打得開；但同一站在不同頁面給不同網域，讀者與搜尋引擎會收到兩個版本，`.dev` 的示範網址在文章中卻帶著「這是正式展示」的語氣。

**建議**：全站統一 `.com`，`.dev` 只留轉址。

**證據**：`curl -I https://breakout.tommypan.dev/` 回 `HTTP/2 301`、`location: https://breakout.tommypan.com/`；`.com` 直接 `200`。

### B7. reward shaping 的區間寫法容易被讀成信賴區間

**位置**：`writing/breakout-reward-shaping`，第二節末。

**原文**

> 但把訓練之間的差異也算進去後，合理的結果範圍從退步 1.74 分到進步 9.48 分，仍包含「沒有改善」的可能。

**問題**：−1.74 到 +9.48 是三個訓練 seed 的點估計範圍，不是區間估計。凍結紀錄另外記了一個 hierarchical interval `[−2.34, 9.62]`，兩者都不是同一個東西。句子已經寫「仍包含沒有改善的可能」，方向是對的，但數字來源要講清楚。

**建議**：寫「三個訓練 seed 的結果範圍是 −1.74 到 +9.48；把 seed 之間的差異一起估進去，區間仍然跨過 0」。

**證據**：`docs/final-results/README.md:120`（`+3.26`、seed 2023 為負、兩個 cross-seed bootstrap 區間都含 0）；`docs/final-results/summary.json:465`（hierarchical `[-2.3402, 9.62]`）。

---

## C. 已核對，仍然成立（不要動）

逐條對回原始表格驗算過：

| 頁面 | 句子 | 驗算 |
| --- | --- | --- |
| 總覽 | 「約快 1.6 倍」 | 238.67 → 380.74 transitions/s = 1.60× |
| scaling | 表格 239 / 381 / 368、7 分鐘 / 4 分 23 秒 / 4 分 32 秒 | 419.00 s、262.65 s、271.70 s，四捨五入一致 |
| scaling | 測試平均分 9.0 / 6.1 / 2.3 | 9.00 / 6.07 / 2.33 |
| scaling | 15.8 / 18.2 / 20.2，36.1 / 19.4 | 15.844 / 18.156 / 20.178；36.111 / 19.356 |
| scaling | 「參數接近其他版本的兩倍」「19 分鐘 / 12 分鐘」 | 3.29M vs 1.69M；1146.3 s vs 723.5 s |
| scaling | 「33.1 到 34.9」「42.6 和 51.4」「49.9」 | 33.133–34.867；42.600、51.400；49.933 |
| onnx | P50 2.533 / 1.526 / 0.856 / 1.342 ms | 與 Day 24 表格逐格相同 |
| onnx | 「最大差距約 0.000162」「4 次選了不同動作」 | Day 23 / 25 / 26 皆為 0.000162；FP16 4/60 |
| onnx | 「1.110 降到 0.761 毫秒，改善約 31%」 | 31.4% |
| onnx | 30 局平均 42.53 / 42.03 / 42.03 / 36.80 | Day 26 同值 |
| building-debugging | 「16,395 次操作」「96%」 | 16,449 − 54 = 16,395；約 96% |
| building-debugging | 「一千次操作的試跑完成了四局」 | Day 12：4 局，分數 2、3、0、0 |
| building-debugging | 「一萬次操作」 | Day 13：10,000 steps、48 局、2,251 次更新 |
| foundations | 0.027720 / 0.271000 / 0.0099、2.43 | 與 Day 6 trace 相同；0.9² × 3 = 2.43 |
| reward shaping | +9.48 / −1.74 / +2.04 / +3.26，第二、三組範圍含零 | 與凍結紀錄一致 |
| 專案頁 / 瀏覽器篇 | 「60 張固定畫面…動作全部相同」 | Day 23 60/60；Day 29 逐像素一致 |

---

## 摘要

| 類別 | 數量 | 項目 |
| --- | ---: | --- |
| 已不成立、需要改寫 | 3 | A1 合併對照組、A2 全稱的「沒有預先寫好的接球路線」、A3 未提難度失誤注入 |
| 過時、需要更新或加註 | 7 | B1 缺控制器路線、B2 137.47 定位、B3 30.9 未標契約、B4 4.53 屬 v1、B5 clear 用語、B6 網域、B7 區間寫法 |
| 已核對、不需改動 | 16 | 見 C 表 |

優先順序：B2 與 B5 先處理，因為它們決定新文章能不能引用舊數字；A1 與 A2 是唯二會讓讀者誤解機制的句子。
