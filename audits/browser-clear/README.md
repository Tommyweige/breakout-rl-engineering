# 瀏覽器版 Unbeatable：畫面打光 vs canonical clear（120 局）

對象：`main` @ `a8445df`（#68）的瀏覽器 agent
量測：`scripts/analysis/browser-clear-audit.mjs`（Node + pinned ALE WASM）
日期：2026-10-29

## 摘要

- **120 局**（前 20 局 + 新 100 局，seed 完全不重複），共 3,628,104 個 ALE frames，逐局 CSV 在 `episodes-120.csv`。
- **畫面層 brick clear**（磚牆在畫面上消失至少 1 frame）：**107/120（89.2%）**。
- **canonical clear**（RAM[76:78] BCD = 864 且等於 cumulative reward）：**43/120（35.8%）**。
- 分數 mean **731.4**、median **850.5**、min 384、max 864；120 局中 116 局以 0 命結束，4 局撞到 108,000 frame 上限。
- **官方 native ALE 15-seed（追加）**：現行 #68 controller **4/15 局**達到 canonical 864，mean **721.8**；同一條 path 的 pre-#68 controller 是 0/15、mean 137.47。
- 137.47 是 **#68 之前的控制律**的歷史數字，可一字不差重現（mean 137.4667 / median 95 / max 341 / 0 clears）；它不是「定義不同」造成的，也不是現行版本的分數。

## 1. 兩個定義

| | 畫面層 brick clear | canonical clear |
|---|---|---|
| 由誰判定 | ROM 畫面（人眼） | `breakout_rl/completion.py` |
| 訊號 | 18×6 磚塊格全空至少 1 frame | ALE RAM[76:78] packed-BCD == 864 且等於 cumulative raw reward |
| 分數 | 432（一面牆） | 864（連續兩面牆） |
| 之後 | 空場地 30–199 frames（中位 57；107 次打光中 106 次可量到重畫，另 1 次在空場地期間輸光 lives）後 ROM 重畫新牆 | 場地空白，分數停在 864，球繼續打到 lives 用完 |
| 瀏覽器有沒有 | 沒有判定，只是畫面 | 沒有判定（UI 只有 score / lives / GAME OVER） |

`web/src/inference/PredictiveVisionController.ts` 只做 bounds → blob → 球 → 落點 → LEFT/RIGHT/NOOP；不讀分數、磚塊、RAM 或 lives。`App.ts` 只渲染 `agent-score`、`agent-lives`，`displayStageState()` 只回 `PLAYING / PAUSED / READY / GAME OVER`。

## 2. 方法

- Controller：`web/src/inference/PredictiveVisionController.ts` 原樣以 esbuild bundle（source sha256 `00db1c78…`），未改一行。
- Runtime：`web/vendor/ale-wasm/node.mjs`，ALE 0.12.0、ROM sha256 `376323f0…`（與 completion audit 相同）、mode 0、difficulty 0、RAM 128 bytes。
- 重現瀏覽器 agent path：每決策 1 個 ALE native frame、`repeat_action_probability=0.25`、`max_num_frames_per_episode=108000`、FIRE 由環境自動處理（reset 與每次 life loss 後）、controller 讀 act 前的 160×210×3 RGB。
- Seeds：`101,102,103,104,105,202,203,204,205,206,303,304,305,306,307,401,402,403,404,405`（前 20 局）＋ `501–600`（新 100 局），共 120 個不重複 seed。
- 每局都檢查 RAM 分數與 cumulative reward 一致；120 局共 0 次不一致。
- 決定性：seed 101 重跑三次（原始批次、committed harness、replay）都是 52,993 steps / 864 分；20 局批次用 committed harness 重跑，逐局 0 diff。

### 2.1 Python 與瀏覽器控制器的對照

| 項目 | Python (`breakout_rl/vision_controller.py`) | 瀏覽器 (`web/src/inference/PredictiveVisionController.ts`) | 相同 |
|---|---|---|---|
| 決策節奏 | 1 決策 = 1 ALE native frame（evaluator 檢查 frame delta == 1） | 1 決策 = 1 ALE native frame（outerActionRepeat=1；rAF 只影響 wall-clock） | ✅ |
| 參數 | config `breakout_vision_controller_v1.json`（deadband 3、hysteresis 1、max_missing 4、max_speed 8） | 直接 import 同一份 JSON | ✅ |
| deadband / hysteresis | Schmitt trigger，release = max(0, deadband−hysteresis) | 同一條公式（本設定下同值） | ✅ |
| brake / 提前放開 | `abs(error) ≤ paddle_width/2 − 2` 或 `error·paddleVx>0 且 abs(error) ≤ 2*abs(paddleVx)+deadband` → NOOP | 同 | ✅ |
| 落點預測 | `vy > 0.15`、`(paddle.top−2−ball.y)/vy`、reflect 於 `bounds±2` | 同 | ✅ |
| paddle 邊緣 | 寬 <16 → `left+7.5` / `right−7.5` | 同 | ✅ |
| pixel 範圍 | raw RGB 210×160×3 uint8 | `getScreenRGB()` 160×210×3 | ✅ |
| 連通元件 | cv2 `connectedComponentsWithStats`，8-connectivity | 手寫 8-neighbour BFS，同一組 shape filter（4–24 px²、2–5×2–7、fill ≥0.45） | 規則相同、實作不同 |
| moving mask | 與前一 frame 任一 RGB channel 不同 | 同 | ✅ |
| FIRE | wrapper：initial serve + 每次 life loss；8 次上限、2 步確認、0.0001 變動、reward 或 activity | 同一個 contract v3 常數與同一條確認邏輯 | ✅ |
| controller 選 FIRE | 不會（FIRE 由環境覆寫） | 不會 | ✅ |
| 診斷 | 完整 latency / diagnostics | 只有 ball / paddleX / interceptX | 不影響動作 |

結論：**同一套控制律**。差異只有 OpenCV vs 手寫 BFS、float vs int bounds、診斷欄位，以及 runtime 的 RNG 流（native vs WASM）；不是規則或參數。

## 3. 結果

### 20 / 100 / 120 局並列

| 指標 | 前 20 局 | 新 100 局 | 合計 120 局 |
|---|---|---|---|
| 局數 | 20 | 100 | 120 |
| 畫面打光牆（局數） | 16 | 91 | 107 |
| 畫面打光牆佔比 | 80.0% | 91.0% | 89.2% |
| canonical 864（局數） | 9 | 34 | 43 |
| canonical 864 佔比 | 45.0% | 34.0% | 35.8% |
| mean | 679.1 | 741.8 | 731.4 |
| median | 827.5 | 852.0 | 850.5 |
| min / max | 407 / 864 | 384 / 864 | 384 / 864 |
| 標準差 | 205.0 | 172.5 | 178.9 |
| 局長 median（steps） | 24,758 | 25,492 | 25,492.5 |
| 局長 min / max | 13,350 / 108,000 | 8,453 / 108,000 | 8,453 / 108,000 |
| 結束時 0 磚 | 8 | 32 | 40 |
| 結束時 0 命 | 19 | 97 | 116 |
| 108,000 frame 上限 | 1 | 3 | 4 |

### 分數分布

| 分數區間 | 前 20 | 新 100 | 合計 120 |
|---|---|---|---|
| <200 | 0 | 0 | 0 |
| 200–399 | 0 | 1 | 1 |
| 400–599 | 9 | 25 | 34 |
| 600–799 | 0 | 9 | 9 |
| 800–863 | 2 | 31 | 33 |
| 864 | 9 | 34 | 43 |

### 抽樣誤差（bootstrap，20,000 次重抽，95% CI）

| 指標 | 合計 120 局 |
|---|---|
| mean | 699.0 – 762.5 |
| median | 832.0 – 858.0 |
| canonical 佔比 | 27.5% – 44.2% |
| 畫面打光牆佔比 | 83.3% – 94.2% |

20 局與 120 局的差異在抽樣誤差內：canonical 45% vs 35.8%（two-proportion z ≈ 0.94，p ≈ 0.35）、打光牆 80% vs 89.2%（z ≈ −1.45，p ≈ 0.15）。20 局的區間本來就寬（canonical 25–65%），所以「20 局看起來比較會破關」不是真的差異；120 局只是把區間縮到 ±8pp。

### 逐局 CSV

`episodes-120.csv` 欄位與 harness 輸出一致：`seed, score, lives_remaining, bricks_remaining, min_bricks, steps, emulator_frames, brick_clear_events, pre864_wall_clear_events, post864_wall_clear_events, level_advance_events, board_empty_at_end, canonical_cleared, terminated, truncated, duration_ms`。`pre864_wall_clear_events` 是「畫面層打光」的依據（>0 即算一次），`canonical_cleared` 是 canonical 判定。

## 4. Native ALE 官方 15-seed 驗證（2026-10-29 追加）

用 repo 的官方入口跑現行 controller（#68），不經 WASM harness：

```bash
python -m scripts.evaluation.evaluate_predictive_controller \
  --output-dir outputs/issue25-vision-controller
```

環境：`ale-py==0.12.0`、`gymnasium==1.3.0`、`ALE/Breakout-v5`、mode 0、difficulty 0、frame_skip 1、sticky 0.25、raw RGB、無 reset noops、`BreakoutFireResetWrapper`。15 個 contract seeds、每局上限 108,000 frames、官方 `BreakoutCompletionDetector`。

| seed | score | canonical | total steps | clear @ frame | lives at clear |
|---|---|---:|---|---:|---:|
| 101 | 859 | no | 23,496 | — | — |
| 102 | 537 | no | 17,979 | — | — |
| 103 | 828 | no | 23,819 | — | — |
| 104 | 565 | no | 21,695 | — | — |
| 105 | 810 | no | 19,547 | — | — |
| 202 | 864 | **yes** | 48,828 | 23,173 | 2 |
| 203 | 425 | no | 12,103 | — | — |
| 204 | 849 | no | 29,152 | — | — |
| 205 | 750 | no | 26,928 | — | — |
| 206 | 441 | no | 14,246 | — | — |
| 303 | 519 | no | 13,169 | — | — |
| 304 | 864 | **yes** | 42,057 | 22,692 | 1 |
| 305 | 788 | no | 17,610 | — | — |
| 306 | 864 | **yes** | 41,022 | 40,703 | 1 |
| 307 | 864 | **yes** | 36,139 | 26,231 | 2 |

合計：mean **721.8**、median **810**、max 864、**4/15 canonical（26.7%）**、387,790 frames、889 秒（436 steps/s，RSS ~110 MB）。bootstrap 95% CI：mean 636.2–800.6、canonical 10.0–50.0%。11 局失敗原因都是 `game_over`，沒有 `time_limit`。

Provenance：這一跑在 audit branch HEAD `9d88918`，產品程式碼與 `a8445df` 完全相同（`git diff a8445df 9d88918 -- breakout_env.py breakout_rl configs scripts/evaluation` 為空）；`source_provenance.working_tree_dirty=true` 是因為當下目錄有 untracked 的 audit 檔案，不是程式碼改動。

### 三種組合的對照

| runtime | controller | mean | median | max | canonical | 打光一面牆 |
|---|---|---:|---:|---:|---|---|
| native ALE | pre-#68（#26 檔案） | 137.47 | 95 | 341 | 0/15 | 0/15（最高 341 < 432） |
| native ALE | #68 | 721.8 | 810 | 864 | 4/15 | 14/15* |
| browser WASM | #68 | 674.5 | 849 | 864 | 7/15 | 11/15 |

\* native 沒有逐 frame 螢幕紀錄，這一欄用 score ≥ 432（清掉第一面牆）推得；browser 是實測畫面。

native 的 4/15 與 browser 的 7/15 在 n=15 下屬抽樣誤差；但逐局贏家不同（203：native 425 / browser 864；304：native 864 / browser 407），因為同一個 seed 在兩個 runtime 的 sticky-action RNG 流不同——只有分布可以比，逐局不能對照。

這條 native 路徑把「#68 之後會破關」從 harness 條件裡解放出來：官方入口、官方 15 seeds、官方 detector、native ALE，4/15 局達到 864。

## 5. 137.47 的重現方法與參數

歷史紀錄（issue #73 body、closeout 的 `docs/final-results/summary.json`）標的是 `historical_reported`、`raw_artifact_status: missing`：

```
mean 137.47, median 95, max 341, 0 canonical clears, 15 seeds
```

重現腳本：`audits/browser-clear/reproduce-137-47.py`（需要 `ale-py==0.12.0`、`gymnasium==1.3.0`、`opencv-python`）：

```bash
python audits/browser-clear/reproduce-137-47.py
```

它做的事：

1. 用 `git show 50ee352:breakout_rl/vision_controller.py` 取出 PR #26 當時的 controller（#67/#68 之前唯一版本；`git log` 顯示這個檔案只有 #26 與 #68 兩個 commit）。
2. 用 `importlib` 載入舊 class，monkeypatch `breakout_rl.vision_evaluation.PredictiveBreakoutController`，其餘（環境、contract v3、15 個 seeds、canonical detector）完全用現行程式。
3. 跑 15 局並印出統計。

結果：

```
mean 137.46666666666667  median 95.0  max 341.0  min 30.0  canonical_clears 0
scores: 337, 136, 59, 31, 55, 201, 42, 185, 95, 112, 341, 85, 41, 312, 30
```

與紀錄的 137.47 / 95 / 341 完全一致。舊版在這個 15 局樣本裡最高 341 分，連一面牆（432 分）都打不完，所以 0 clears 與 clear 定義無關。

#68（`a8445df`）同時改了 Python 與 TS 的控制器：velocity anchor（用兩個 native-frame 區間消除像素進位）＋接近球時提前放開（catch margin）。現行版本在同樣 contract 下是 **721.8**（native 官方 15 局）、**731.4**（WASM 120 局）；舊版是 137.47。

## 6. 對外引用的適用範圍

可以引用：

> **官方評估**（native ALE、15 個 contract seeds、官方 detector）：現行 `a8445df` controller 在 **4/15 局**達到 canonical 864，mean **721.8** / median 810 / max 864；同一條路徑換回 #26 的 controller 是 mean 137.47 / 0/15。
>
> **補充分布**（pinned ALE WASM、120 局）：canonical 43/120（35.8%，bootstrap 95% CI 27.5–44.2%）、畫面層打光牆 107/120（89.2%，CI 83.3–94.2%）、mean 731.4（CI 699.0–762.5）。

還不能宣稱：

- **120 局不是官方評估**：官方 seed 清單只有 15 個、用 native ale-py；120 局是 WASM 上的補充抽樣，501–600 不在 contract 裡。
- **跨 runtime 逐局不可比**：同一 seed 在 WASM 與 native ale-py 的 sticky-action RNG 流不同（seed 203：native 425 vs browser 864；seed 304：native 864 vs browser 407），只能比分布。
- **單一版本**：只量了 #68；沒有其他 commit 的對照（pre-#68 只有 Python controller 的 15 局）。
- **單一 clear 定義**：canonical 是 repo 的 RAM 偵測器；「畫面打光」是本 audit 的像素網格定義。
- **864 是上限而非「結束」**：達到 864 後 ROM 會讓球繼續跑、lives 用完才 game over；4 局（401、501、513、532）之後還會出現整面牆（背景變色、分數停在 864），那些 post-864 事件（共 12 次）已從「打光牆」統計中排除。
- **正式站版本未驗證**：量的是 HEAD 的 controller，不是部署 artifact；行為與 #68 一致，但沒有直接檢查 production bundle。

## 7. 檔案

| 檔案 | 內容 |
|---|---|
| `scripts/analysis/browser-clear-audit.mjs` | harness；`--seeds 501-600`、`--merge`、`--summarize-only`、`--frame-logs` |
| `audits/browser-clear/episodes-120.csv` | 120 局逐局結果 |
| `audits/browser-clear/episodes-20.csv`、`episodes-100.csv` | 兩批各自的逐局結果 |
| `audits/browser-clear/summary-120.json` | 20 / 100 / 120 彙總 + bootstrap CI |
| `audits/browser-clear/summary-20.json`、`summary-100.json` | 兩批各自的完整 run summary（含每局 events） |
| `audits/browser-clear/native-15/` | 官方 native ALE 15-seed 的逐局 CSV、summary、report、diagnostics |
| `audits/browser-clear/compare-native-15.py` | native vs WASM 同 15 seeds 的對照 |
| `audits/browser-clear/reproduce-137-47.py` | 137.47 重現腳本 |
| `audits/browser-clear/figures/` | 打光 / 新牆 / canonical 864 / post-864 截圖 |

重跑：

```bash
(cd web && npm ci)
node scripts/analysis/browser-clear-audit.mjs --seeds 501-600 --frame-logs 1 --out /tmp/audit-100
node scripts/analysis/browser-clear-audit.mjs --summarize-only \
  --merge /tmp/audit-100/summary.json --merge audits/browser-clear/summary-20.json --out /tmp/audit-120
python -m scripts.evaluation.evaluate_predictive_controller \
  --output-dir outputs/issue25-vision-controller
python audits/browser-clear/compare-native-15.py outputs/issue25-vision-controller
```
