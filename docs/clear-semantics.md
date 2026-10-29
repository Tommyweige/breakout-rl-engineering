# Breakout clear 語意：「畫面打光」與「canonical clear」

**狀態：2026-10-29。** 本文件定義專案裡兩種「破關」說法、它們各自能回答的問題、會誤判的情況，以及報告該怎麼引用。所有結論都有可重跑的素材與測試（見[驗證素材](#驗證素材)）。

## 一句話結論

- **畫面打光不等於破關。** Breakout Game 1 有兩面牆。第一面牆打完（432 分）時，畫面會空一小段時間，然後 ROM 畫上新牆，看起來像「進下一關」。v3 fixture 在 native ALE 上重播時是空 34 個 emulator frames（steps 10738–10771）；瀏覽器 WASM audit 量到的是 30–199 frames（中位數 57，見 `audits/browser-clear/README.md`）。這是第一面牆，**不是** clear。
- **Canonical clear 是兩面牆都打完：ROM 計分板等於 864。** `BreakoutCompletionDetector` 在兩條真實 ALE 破關軌跡（Contract v2 與 v3）上，於第二面牆清空的同一個 agent step 判定 clear，在所有反例上都判定 not cleared。這次驗證**沒有找到 detector 誤判**，所以 detector 判定邏輯沒有修改。
- **137.47 可以重現**，但只在 PR #26 的 commit `50ee352`。#68（`a8445df`）改了 controller 之後，同一組 seeds 得到 mean 721.8、4/15 canonical clear detections（`77a0c7b` 與 `a8445df` 的產品程式碼完全相同）。見 [137.47 的結論](#13747-的結論)。

## 兩個定義

### 畫面打光（screen wall cleared）

一面牆的「畫面打光事件」發生在一個 agent step 之後的 raw RGB 畫面 `(210, 160, 3)` 滿足下列任一條件：

1. `count_visible_bricks(screen) == 0`，而上一個 step 的畫面還有磚塊；或
2. 磚塊數比上一個 step 多 50 塊以上（ROM 重畫了一面新牆，代表上一面牆已被清空；當作保險，以防觀測取樣剛好跳過空畫面。實測空畫面維持數十個 frames，所以 frame skip 4 一般也能看到）。

`count_visible_bricks` 在 `breakout_rl/completion.py`：磚牆是 6 列 × 18 欄、每塊 8 × 6 pixels，位於 rows 57–92、columns 8–151；一格至少一半 pixels 亮起才算一塊磚，所以 2 × 4 pixels 的球不會被算成磚。重置後的畫面是 108 塊。

這個定義是**牆層級**的事件。在一局裡它會發生兩次：432 分（第一面牆）與 864 分（第二面牆）。

### Canonical clear（canonical clear detection）

`BreakoutCompletionDetector`（`breakout_rl/completion.py`）在每個 agent step 之後讀取：

- ALE RAM `[76, 77]` 的四位 packed-BCD 計分板；
- 該局累積的 raw reward（未 clip、未 shaping）。

兩者不一致時，該局的狀態變成 `cleared=None`（unknown），之後不再判定。兩者一致且等於 864 時，鎖定（latch）第一次出現的 step 為 clear，並記錄 `clear_agent_step`、`clear_emulator_frame`（ALE native frame 差，非 `agent_step × frame_skip`）與 `lives_remaining_at_clear`。執行環境不是審核過的 ale-py 0.12.0、ROM checksum、mode 0、difficulty 0 時，一律 `cleared=None`。

864 是兩面牆的總分：每面牆 18 欄 × (1 + 1 + 4 + 4 + 7 + 7) = 432。因為 Game 1 不會再產生第三面牆，分數到 864 就代表兩面牆都已清空。素材驗證了這一點：兩條軌跡裡分數到 864 的 step，正好是第二面牆在畫面上清空的 step。

**用語區分：** detector 產生的是 *canonical clear detection*。repository 的 artifact gate（`breakout_rl/evaluation_artifacts.py`）另外要求 evaluation artifact 通過 contract、source 與 provenance 驗證。像 `scripts.evaluation.evaluate_predictive_controller` 這類不做該驗證的 evaluator，只能說「canonical clear detections」。

## 差異對照

| 面向 | 畫面打光 | Canonical clear（Python detector） | 瀏覽器（`web/src`） |
|---|---|---|---|
| 輸入來源 | raw RGB pixels | ALE RAM 計分板 + 累積 raw reward | 沒有任何 clear 判定；只顯示 ALE 分數、命數與畫面 |
| 判定時機 | 每個 agent step 後的畫面 | 每個 agent step 後，episode 結束判斷之前 | 不適用；「打光」是人看畫面的判斷 |
| 容錯 | 半格門檻排除球；抽樣不到空畫面時用「新牆出現」補 | RAM 與 reward 不一致或 runtime 不符時 fail closed（`None`，不算失敗） | 無 |
| 會誤判的情況 | 432 分的第一面牆（偽陽性）；取樣跳過空畫面時只能靠新牆補救 | 見下方 | 見下方 |

`web/src` 搜尋不到任何 brick / wall / 864 / 432 / RAM 讀取邏輯。瀏覽器 ALE 也沒有跑這個 detector。所以「瀏覽器裡 AI 把磚塊打光、進下一關」只是觀察者的描述，它對應的最可能是 432 分時新牆出現的那一刻。

### 什麼情況會誤判

**畫面打光：**

- **第一面牆被當成破關（偽陽性）。** 432 分時畫面空數十個 frames，接著新牆出現。人眼與 pixel 偵測都會看到「打光、進下一關」。
- **取樣跳過空畫面（偽陰性，理論上）。** 實測空畫面維持數十個 frames，Contract v2 每 4 frames 看一次仍會看到；若取樣更稀疏，只數「畫面曾經空」的定義可能漏掉，所以本文件的定義另外用「新牆出現」補救。
- **第二面牆打光之後沒有「下一關」。** 864 分之後畫面保持全空，球繼續彈，不再有任何 reward，直到掉光命或到 time limit。期待「進下一關」的觀察者會以為遊戲卡住。

**Canonical clear：**

- **unknown 被當成 0。** runtime 不符或 RAM/reward 不一致時是 `cleared=None`。報告把它算進「0 次 clear」就是誤報。
- **傳入 shaping 或 clip 過的 reward。** `cumulative_score` 必須是 raw reward，否則與 RAM 不一致，整局變 unknown。
- **只看 episode 結束狀態。** clear 之後 episode 還會繼續，最後常以 `game_over` 結束（現行 controller 的 4 次 clear 都是這樣）。用 `terminated` 或最終命數判斷成敗的程式會把 clear 算成失敗；detector 是 latch，不受影響。
- **沒有每步呼叫 `observe`。** 漏呼叫只會延後 `clear_agent_step`，不會讓 clear 消失（分數停在 864）。但報告的 clear 時間會錯。
- **用 `agent_step × frame_skip` 推算時間。** detector 只接受 ALE native frame；推算值在 v3 會差 4 倍。

**本文件是 detector 正向驗證的權威來源。** 正向路徑現在有兩層證據：(1) 原本的 synthetic RAM/reward fixture（`tests/test_completion.py::BreakoutCompletionDetectorTests::test_ale_scoreboard_ram_fixture_emits_the_documented_maximum_reward`，把 864 寫進 RAM 後驗證下一個真實 ALE transition 的 reward 一致）；(2) 下方兩條真實 ALE 破關軌跡（v2、v3）。`configs/eval/breakout_completion_audit_v1.json` 的 `validation.limitation` 與 `natural_ale_positive_trajectory.status = not_observed` 是這次驗證之前寫的，已被本文件取代；該檔是 evaluator `source_sha256` provenance 的一部分，改動會讓已提交的 `a8445df` native 15 局 provenance 對不上，所以刻意保持原樣。

這次的素材沒有找到 detector 本身的誤判：偽陽性（432、860）、偽陰性（clear 後 game over、clear 與 time limit 同一步、clear 與失去最後一命同一步）與時間都正確。

## 各自能回答什麼問題

| 問題 | 用哪個 |
|---|---|
| 這局有沒有把兩面牆都打完？什麼時候？剩幾條命？ | Canonical clear |
| clear rate / 第一次破關（First Clear）/ Reliable Clear | Canonical clear（只有全部 episode 狀態已知時 clear rate 才有值） |
| 觀察者在畫面上看到的「打光」是哪一面牆？ | 畫面打光事件 + 當下分數（432 = 第一面，864 = 第二面） |
| pixel controller 看到的牆是否與 RAM 一致？ | 兩者比對（素材測試就是這樣做的） |
| 瀏覽器 demo 裡有沒有破關？ | 都不能直接回答：瀏覽器沒有記錄 detector 狀態。需要用 Python evaluator 跑同一個 controller 與 seeds |

## 驗證素材

素材在 `tests/fixtures/breakout_clear/`，可重複生成：

```bash
python -m tests.fixtures.breakout_clear.generate_fixtures           # 重用已提交的 action plan，重算 ground truth
python -m tests.fixtures.breakout_clear.generate_fixtures --replan  # 重新搜尋 action plan（v3 約 1 分鐘、v2 約 4 分鐘）
```

每個 fixture 是一串 **requested actions**（run-length encoded），從 `env.reset(seed=101)` 開始，經過 repository 的 evaluation environment factory（v2：`make_breakout_env(**breakout_environment_kwargs(contract))`；v3：`make_vision_breakout_env(contract)`），在 sticky action 0.25 下重播。sticky-action RNG 由 reset seed 決定，所以重播完全可重複。

產生方式：搜尋器讀 ALE RAM 追球，用 ALE state clone 嘗試瞄準 offset，丟掉會失去命的片段。ALE `restoreState` 沒有完整還原 sticky-action 狀態，因此搜尋環境只負責「提案」。另一個只會往前走的 committed 環境實際執行片段後才接受，所以 fixture 等於從 reset 重播的結果。搜尋器與 ground truth **都不呼叫 detector**；ground truth 來自畫面磚塊數、raw reward 與 `ale.lives()`。

| Fixture / scenario | 內容 | 最終分數 | 畫面打光事件（step: 分數） | 兩面牆 clear（step, frame, 剩餘命） |
|---|---|---:|---|---|
| v2 `full_clear_then_game_over` | 完整破關後避球直到 game over | 864 | 2655: 432；4799: 864 | 4799, 19196, 5 |
| v2 `first_wall_only_then_game_over` | 第一面牆打光後避球 | 435 | 2655: 432 | — |
| v2 `last_brick_left_then_game_over` | 剩最後一塊磚時避球 | 860 | 2655: 432 | — |
| v3 `full_clear_then_game_over` | 完整破關後避球直到 game over | 864 | 10738: 432；18368: 864 | 18368, 18368, 5 |
| v3 `first_wall_only_then_game_over` | 第一面牆打光後避球 | 435 | 10738: 432 | — |
| v3 `last_brick_left_then_game_over` | 剩最後一塊磚時避球 | 860 | 10738: 432 | — |

`tests/test_completion.py::GenuineClearFixtureTests` 用這些素材驗證：

- 每個 scenario 逐步餵 detector（與 evaluator 相同的呼叫方式），`cleared`、`clear_agent_step`、`clear_emulator_frame`、`lives_remaining_at_clear` 都要等於畫面 ground truth；任何一步 RAM/reward 不一致都會讓 `cleared=None` 而失敗。
- 第一次畫面全空發生在 432 分，且當下 detector 判定 not cleared。
- **clear 與 episode 結束同一步：** 把 ALE `max_num_frames_per_episode` 設成 clear frame，真實軌跡在 clear 那一步 truncated，仍要判定 clear。
- **clear 與失去最後一命同一步：** 真實軌跡物理上做不到（球從磚到底要很多 frames），所以用 detector 直接輸入 `lives_remaining=0` 驗證。
- **sticky action 重複輸入：** 同一串 requested actions 在 sticky 0 下重播不會破關，detector 必須跟著 emulator 結果判定 not cleared，而不是跟著輸入紀錄。
- **v2 evaluation pipeline 端到端：** 用 scripted Q-network 驅動 `evaluate_policy`，`EpisodeResult` 的 clear 與 `completion_outcome` 要正確（破關後 game over 仍是 `cleared`）。

測試會在錯誤的 detector 上失敗。以下 mutation 都被新測試抓到：

| 錯誤的 detector | 抓到它的測試 |
|---|---|
| 432 分（第一面牆）就算 clear | 全部 scenario、第一次空畫面、time limit 同步、最後一命同步、v2 pipeline |
| 分數 ≥ 860（只剩一塊磚）就算 clear | 全部 scenario、time limit 同步、最後一命同步、v2 pipeline |
| 失去最後一命時把 clear 作廢 | 全部 scenario（每個都以 game over 結束）、最後一命同步、v2 pipeline |
| clear step 晚記一步 | 全部 scenario、time limit 同步、最後一命同步、v2 pipeline |
| 用 `agent_step × 4` 當 emulator frame | 全部 scenario（v3 每步 1 frame） |
| 不 latch，之後的 864 觀測覆寫 clear step | 全部 scenario、v2 pipeline |

（Mutation 以 scratch script 暫時替換 `BreakoutCompletionDetector.observe` 後執行 `GenuineClearFixtureTests`；未提交。）

## 報告裡該怎麼引用

- 寫「canonical clear detections: k / n（`ale-breakout-two-wall-score-864-v2`，Contract v3，seeds …，commit …）」。沒有通過 artifact provenance 驗證的 evaluation，不能寫成比 canonical clear detection 更強的說法。
- clear rate 只有在每局狀態都已知時才寫數字；有任何 `cleared=None` 就寫「unknown」，不要寫 0。
- clear 時間用 `clear_emulator_frame`（ALE native frames）與 `clear_agent_step`，並註明 contract；不換算成秒。
- 描述畫面時寫「第一面牆打光（432 分）」或「兩面牆打光（864 分）」，不要只寫「打光」或「進下一關」。
- 瀏覽器觀察只能當 qualitative 描述，必須標註「未經 detector 驗證」。
- 分數一定要附 commit。同一個 evaluator 在不同 commit 的 controller 會得到完全不同的結果（見下節）。

## 137.47 的結論

**可以重現：** 在 PR #26 merge 後的 commit `50ee352`，用 `python -m scripts.evaluation.evaluate_predictive_controller`（config `configs/eval/breakout_vision_controller_v1.json`，Contract v3，seeds 101–105、202–206、303–307）跑兩次，逐局分數完全相同：

- mean **137.4667**（報告寫 137.47）、median 95、max 341、min 30；
- canonical clears **0 / 15**，15 局都是 `game_over`；
- source commit `50ee352`、working tree clean、source sha256 `4a0f4fd177b6…`、config sha256 `25ddff675674…`、contract sha256 `d97b7fcb5758…`。

逐局紀錄（`50ee352`）：

| Seed | Raw score | Agent steps | ALE frames | 結果 |
|---:|---:|---:|---:|---|
| 101 | 337 | 8657 | 8657 | game_over |
| 102 | 136 | 7341 | 7341 | game_over |
| 103 | 59 | 3934 | 3934 | game_over |
| 104 | 31 | 3675 | 3675 | game_over |
| 105 | 55 | 4196 | 4196 | game_over |
| 202 | 201 | 7164 | 7164 | game_over |
| 203 | 42 | 5212 | 5212 | game_over |
| 204 | 185 | 6347 | 6347 | game_over |
| 205 | 95 | 4464 | 4464 | game_over |
| 206 | 112 | 5109 | 5109 | game_over |
| 303 | 341 | 7491 | 7491 | game_over |
| 304 | 85 | 5584 | 5584 | game_over |
| 305 | 41 | 4474 | 4474 | game_over |
| 306 | 312 | 7613 | 7613 | game_over |
| 307 | 30 | 3205 | 3205 | game_over |

最高分 341 低於第一面牆的 432。所以這 15 局的 0 次 clear 不是尺壞了：controller 連第一面牆都沒有打完。

**但它不代表現在的 controller：** #68（`a8445df`）改了 `breakout_rl/vision_controller.py`，加入 velocity anchor 與接球前的 paddle 釋放規則（瀏覽器的 Unbeatable 也用同一套）。evaluator、config、contract 與 detector 都沒變（config sha256 `25ddff675674…`、contract sha256 `d97b7fcb5758…` 相同）。在 `77a0c7b`（clean worktree，source sha256 `5989e6177452…`）用同一個指令與 seeds 跑兩次（本分支與 clean worktree 各一次），逐局完全相同：

- mean **721.8**、median 810、max 864；
- canonical clears **4 / 15**（seeds 202、304、306、307）；
- 15 局全部以 `terminated=True`（game over）結束，包括 4 局 clear：clear 之後球繼續彈，最後掉光命。

逐局紀錄（`77a0c7b`）：

| Seed | Raw score | Agent steps | ALE frames | 結果 | Clear step / frame | 剩餘命 |
|---:|---:|---:|---:|---|---:|---:|
| 101 | 859 | 23496 | 23496 | game_over（差 1 塊） | — | — |
| 102 | 537 | 17979 | 17979 | game_over | — | — |
| 103 | 828 | 23819 | 23819 | game_over | — | — |
| 104 | 565 | 21695 | 21695 | game_over | — | — |
| 105 | 810 | 19547 | 19547 | game_over | — | — |
| 202 | 864 | 48828 | 48828 | **clear**，之後 game_over | 23173 | 2 |
| 203 | 425 | 12103 | 12103 | game_over | — | — |
| 204 | 849 | 29152 | 29152 | game_over | — | — |
| 205 | 750 | 26928 | 26928 | game_over | — | — |
| 206 | 441 | 14246 | 14246 | game_over | — | — |
| 303 | 519 | 13169 | 13169 | game_over | — | — |
| 304 | 864 | 42057 | 42057 | **clear**，之後 game_over | 22692 | 1 |
| 305 | 788 | 17610 | 17610 | game_over | — | — |
| 306 | 864 | 41022 | 41022 | **clear**，之後 game_over | 40703 | 1 |
| 307 | 864 | 36139 | 36139 | **clear**，之後 game_over | 26231 | 2 |

（v3 每個 agent step 是一個 ALE frame，所以 clear step 等於 clear frame。）

**結論：**

1. 137.47 / 0 of 15 是 `50ee352` controller 的正確量測，不是 detector 壞掉。
2. 把它當成現在 controller 的成績是過期的（README、`docs/final-results/` 與 case study 已在 PR #75 更正）。#68 之後（`a8445df`）的 controller 在同一個 evaluator 下有 4/15 canonical clear detections；evaluator 沒做 artifact provenance 驗證，所以只能稱為 canonical clear detections。
3. 瀏覽器 Unbeatable 跑的是 #68 之後的 controller，所以使用者看到的「打光、進下一關」符合現行版本的量測：14 局分數超過 432，會看到第一面牆打光、新牆出現；4 局真的打完第二面牆。

重現指令（需 ale-py 0.12.0、gymnasium 1.3.0）：

```bash
git worktree add --detach /tmp/wt 50ee352   # 或 77a0c7b
cd /tmp/wt && python -m scripts.evaluation.evaluate_predictive_controller --output-dir /tmp/out
```

在 4 核 ARM64 CPU 上，`50ee352` 約 4.5 分鐘，`77a0c7b` 約 30 分鐘（局數長得多）。
