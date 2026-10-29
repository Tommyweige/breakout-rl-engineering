# Final Results — Breakout RL Engineering

**Status: frozen evidence closeout, 2026-10-28; Laya live browser session added 2026-10-29.** This is an audited index of what the project can and cannot support. It is not a new training campaign and it does not claim that Breakout was solved.

- **Interactive case study:** <https://breakout.tommypan.com/case-study.html>
- **Playable demo:** <https://breakout.tommypan.com/>
- **Machine-readable summary:** [`summary.json`](summary.json)
- **Experiment inventory:** [`evidence-inventory.csv`](evidence-inventory.csv)
- **Artifact manifest (expected paths and hashes):** [`artifact-manifest.json`](artifact-manifest.json)
- **Visual evidence:** [before/after screenshots](screenshots/README.md) · [real gameplay recording](../../web/public/case-study/gameplay.mp4)
- **Handoff and deferred work:** [`handoff.md`](handoff.md)

## Research question

Atari Breakout is a low-dimensional control problem: the ball and paddle are visually separable, motion is close to piecewise linear between collisions, and a controller can re-plan after every observation.

> Does Breakout need a trained DQN policy, or can an explainable pixel tracker plus predictive control make useful decisions more cheaply?

The project built both. It did **not** produce a verified full clear, so the honest answer at freeze time is: **the question is still open, and the evidence does not support a fair winner claim.**

## Results at a glance

| Result | Value | Contract | Evidence | Status |
| --- | ---: | --- | --- | --- |
| Verified full clears recorded | 0 | audited 864-point detector | repository audit through 2026-10-28 | **verified absence of record** |
| DQN final holdout mean | 30.933 (median 32.000) | v2, frame_skip=4 | Day 21 archived article | **historical reported** |
| DQN model-selection mean | 51.400 | v2, frame_skip=4 | Day 21 archived article | **historical reported** |
| DQN training cost | 2,500,000 transitions | v2 | frozen ONNX metadata | **verified artifact** |
| RGB controller mean | 137.47 (median 95, max 341) | v3, frame_skip=1 | merged PR #26 | **historical reported** |
| RGB controller canonical clears | 0 / 15 | v3, frame_skip=1 | merged PR #26 | **historical reported** |
| RGB controller latency | about 2.12 ms/frame | v3, frame_skip=1 | merged PR #26 | **historical reported** |
| Reward shaping mean effect | +3.26, 95% interval [-1.74, 9.48] | v2, frame_skip=4 | issue #9 / #10 | **historical reported; not promoted** |
| PER vs Uniform at 500K | 19.98 vs 19.31 (+0.67) at 1.26x time | v2, frame_skip=4 | merged PR #16 | **historical reported; descriptive** |
| Laya complete episodes | 0 | v2 primary | draft PR #74 | **not yet evaluated** |
| Laya browser session | 4 at 22.7 s (lives 0) | v2, frame_skip=4 | [live browser session](laya-vision.md) | **verified session** |
| Laya warm latency p50 | 72.56 ms/decision | v2 primary | issue #70 | **missing raw artifacts** |

**Do not read across the Contract column.** The DQN and controller rows use different control cadences. Their raw scores are descriptive, not a ranking. See [Contracts and fairness](#contracts-and-fairness).

## What is verified

Each item in this repository is labelled with one of four statuses:

- **verified** — checked against an artifact in this repository or a deterministic command that can be rerun here.
- **historical reported** — reported in a merged PR, closed issue, or archived article. The raw run output is not in `main`.
- **missing raw artifacts** — a result is described, but the raw episodes, logs, or weights needed to audit it are not available in this checkout.
- **not yet evaluated** — no completed evaluation exists for the claim.

What is actually verified in this checkout:

- The frozen DQN artifact: 3,292,837 parameters, 4x84x84 input, 4 actions, ONNX SHA-256 `cf90c74b...ef7f12`, source checkpoint SHA-256 `ab07c0a4...79841c48`, source model SHA-256 `6002029d...624389a`.
- ONNX/PyTorch parity on 60 fixed probe states: 60/60 actions match, maximum absolute Q-value difference 0.0.
- Contract v2 and v3 SHA-256 values and their differences: v2 is frame_skip=4 with a 27,000-agent-step limit; v3 is frame_skip=1 with a 108,000-agent-step limit. Both keep the same 108,000 ALE episode-frame limit, frame stack 4, sticky-action probability 0.25, FIRE reset, and 15 evaluation seeds.
- The completion detector's synthetic 864-point RAM/reward fixture. A **naturally completed** ALE trajectory has not been observed.
- The Laya live browser session: score 4, lives 0 after 22.7 s, decision interval p50 67.0 ms, 58.5 presented frames/s, in two runs whose raw records are committed. See [laya-vision.md](laya-vision.md).
- The web runtime model manifest and asset hashes.

What is **not** verified here:

- Any DQN, RGB-controller, reward-shaping, PER, or Laya **raw episode log**. Those runs were produced outside `main`; their generated directories were intentionally ignored.
- Any **verified clear**. The repository's `verified_clears` gate requires contract/source/provenance validation that no recorded run has passed.
- Any stable clearing rate, completion time, or world-record claim.
- Any complete Laya **evaluator** episode or clear. The interactive browser session is verified as a session with committed raw records, not as an evaluated episode.

## Contracts and fairness

Atari results depend on frame skip, frame stack, sticky actions, FIRE handling, life-loss handling, episode limits, and seed roles. The repository keeps two explicit contracts and does not reinterpret one under the other.

| Property | Contract v2 | Contract v3 |
| --- | --- | --- |
| Contract ID | `day15-breakout-evaluation-v2-fire-reset` | `breakout-evaluation-v3-frame-skip-1` |
| File | `configs/eval/breakout_contract_v2.json` | `configs/eval/breakout_contract_v3.json` |
| SHA-256 | `7eca5ae5...dc31a2a` | `d97b7fcb...25c5a786` |
| Decision cadence | one per 4 emulator frames | one per emulator frame |
| Agent step limit | 27,000 | 108,000 |
| ALE episode-frame limit | 108,000 | 108,000 |
| Frame stack | 4 | 4 |
| Sticky actions | 0.25 | 0.25 |
| FIRE reset | yes | yes |
| Terminal on life loss | no | no |
| Evaluation seeds | 101-105, 202-206, 303-307 | same 15 seeds |
| Evaluation epsilon | 0.0 | 0.0 |

The comparison between the DQN holdout and the RGB controller is **not directly comparable**:

- v2 gives the policy 27,000 decisions in an episode; v3 gives it 108,000.
- A v2 agent step is four emulator frames; a v3 agent step is one. Agent-step budgets and "frames per decision" are different units.
- The DQN holdout is 15 evaluation episodes from a frozen model selected over multiple checkpoints. The controller is deterministic, has no training seed, and was run once per evaluation seed.
- The controller's 137.47 mean and the DQN's 30.933 holdout mean therefore answer different questions. The controller's higher raw score does **not** establish that it is a better method, and the DQN's lower holdout does **not** establish that learning is unnecessary.

If a matched comparison is ever run, it must fix one contract, one seed set, one episode budget, and one evaluation command for both methods, and it must be predeclared before looking at results. That comparison was **not** run in this closeout.

## Method evolution

```text
Atari Breakout (ALE/Breakout-v5)
        │
        ├── DQN family (Contract v2)
        │      DQN → Double DQN → Dueling Double DQN
        │      500K family screen → 1M extension → 2.5M final training
        │      frozen ONNX policy in the browser
        │
        ├── Controlled negative/inconclusive experiments (Contract v2)
        │      life-loss reward shaping → not promoted
        │      prioritized replay → descriptive, no stable win
        │      first-clear probes → no verified clear
        │
        └── Pixel-only predictive controller (Contract v3)
               RGB ball/paddle detection → velocity → wall-aware intercept
               → deadband/hysteresis → canonical completion evaluation
               → 0/15 canonical clear detections
```

The DQN family comparison selected Dueling Double DQN at 500K (mean 20.178 versus Double 18.156 and DQN 15.844) and kept it through 1M (36.111). The final 2.5M model was chosen because 2.5M outperformed 5M during selection (51.4 versus 49.9), not because it was the longest run. The later holdout mean fell to 30.933, which is the clearest warning in the project against treating a selection score as a final score.

The controller was built because Breakout's perception problem is explicit: detect the ball and paddle, estimate velocity, predict the paddle-plane intercept with wall reflections, and close the loop on observed paddle position. It is explainable and cheap — about 2.12 ms per controller frame — but it did not clear the game in 15 episodes. Its best episode scored 341, well below 864.

## What the negative results mean

- **Reward shaping (issue #9):** the -1.0 life-loss penalty had a positive mean effect at 1M (+3.26 raw score), but seed 2023 was negative and both cross-training-seed bootstrap intervals included zero. The promotion gate did not pass, and the 2.5M comparison was not run. This is a **positive but inconsistent point estimate**, not a stable improvement.
- **PER (issue #12):** PER was +0.67 raw score over continuous Uniform at 500K, at 1.26x training time, with three training seeds. The result is descriptive and does not establish statistical superiority.
- **First-clear probes (issues #41, #43, #45, #48-#61):** the frozen probes ended without a verified clear. Several were explicitly INCONCLUSIVE or BLOCKED_ENVIRONMENT. They are evidence about specific failure modes, not proof that a method "cannot" clear.
- **RGB controller (issue #25):** 0/15 canonical clear detections. Canonical detections are not `verified_clears`; the detector's positive path is validated with a synthetic fixture, not a natural clear.

## The Laya Vision side experiment

**Motivation.** Test whether a small, frozen, non-generative vision decision model can choose legal Breakout actions from pixels at a practical inference cost.

**Frozen hypothesis.** A 201M-parameter SmolVLM-based checkpoint (`thaitea/laya-vision`, revision `f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`, upstream code `9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c`) can act as a frozen policy without project-specific training.

**Important context.** The model was already trained on some Breakout expert-action data. This is **not** a zero-shot transfer claim, and its upstream normalized score of 0.20 is not a 20% clear rate and may not use this repository's ALE contract.

**What was measured.** Warm decision latency, not gameplay quality:

- 72.56 ms p50 over 100 warmed predictions on one fixed RGB fixture (issue #70).
- Optimized local proxy p50 68.36-69.48 ms across three rounds, with 30/30 action agreement on fixed fixtures (draft PR #74).
- A 3-seed x 80-decision control diagnostic reduced immediate reversal commands from 76 to 10. Both runs totalled 2 reward; lives were [3,3,4] raw and [4,4,3] filtered. That is 320 native frames per seed, not a complete episode.

**Live browser session (verified, 2026-10-29).** The Decision Model controller ran in headed
Chrome on Windows against the local CUDA bridge, on the shipped page and the v2 contract's
seed 101. Two recorded sessions both scored **4** and lost all five lives after **22.7 s** of
play (22.715 s and 22.751 s). The decision interval p50 was 67.0 ms, of which 50-51 ms was
model inference; the AI panel presented 58.5 real frames per second on a 169.5 Hz display;
GPU utilization was 72.5-75.0% p50 at 989 MiB. Controller switching (Decision Model to ONNX
WASM DQN and back) and pause/resume produced no JavaScript errors. Method, per-metric
tables, and the raw records are in [`laya-vision.md`](laya-vision.md).

**What is still not measured.** No complete **evaluator** episode, per-episode artifact, or
clear exists, so the session cannot be compared with evaluated DQN and controller episodes
and does not imply a clear. The model is not zero-shot: the upstream checkpoint was trained
on game frames that include Breakout. Latency still does not imply gameplay quality.

**Setting and resource record (issue #70, draft PR #74).**

| Item | Reported value |
| --- | --- |
| Model | 201M SmolVLM-based; checkpoint revision `f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`; upstream code `9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c` |
| Pretraining domain | Some Breakout expert-action data; **not zero-shot** |
| Contract / cadence | Contract v2 primary; four native ALE frames per decision; contracts unchanged |
| Device / precision | RTX 4060 Laptop, CUDA, FP32, GPU preprocessing |
| Warm latency | 72.56 ms p50 over 100 predictions on one fixed RGB fixture |
| Cold start | Process RSS baseline 517.72 MiB; sampled loading peak 2,176.45 MiB; first inference +664 MiB |
| Warm memory | RSS +0.44 MiB over 100 predictions; optimized 1,000-request run RSS +0.074 MiB |
| Peak VRAM | **Not reported** in the reviewed records |
| Legal actions | `NOOP` / `FIRE` / `RIGHT` / `LEFT` validation; fixed-fixture action agreement 30/30 |
| Complete episode | **None** under the evaluator; two interactive browser sessions in [`laya-vision.md`](laya-vision.md) reached game over |

All values in this table are historical reported or missing raw artifacts, except the
browser-session results in [`laya-vision.md`](laya-vision.md), which are verified against the
records committed with this closeout.

**Decision.** Laya remains a research lead, not a result. Its live browser behaviour is now
recorded, but the line still has no complete evaluator episode; see
[`handoff.md`](handoff.md#laya-line-still-blocked).

## Reproducibility

The commands below are the maintained entry points. They require the project's Python environment (Python 3.12, ALE, Gymnasium, NumPy; CUDA/PyTorch for DQN evaluation) or the web environment. They do not recreate the missing raw runs.

```bash
# Completion / controller / closeout schema checks (CPU-only subset)
python -m unittest \
  tests.test_completion \
  tests.test_vision_controller \
  tests.test_final_results_summary

# RGB predictive controller under Contract v3
python -m scripts.evaluation.evaluate_predictive_controller \
  --config configs/eval/breakout_vision_controller_v1.json \
  --output-dir outputs/issue25-vision-controller

# Reward-shaping screening protocol (not a promoted result)
python -m scripts.training.run_reward_shaping_sweep \
  --output-dir experiments/issue-9-reward-shaping/stage2-250k --parallel
python -m scripts.evaluation.evaluate_reward_shaping_sweep \
  --stage-dir experiments/issue-9-reward-shaping/stage2-250k

# Web build and tests
cd web
npm ci
npm test
npm run build
```

### Re-capturing the real gameplay media

The committed recording and poster are real headless-browser captures from the built app, not mock frames. The rerunnable script records gameplay, the game page before starting, and the case-study page at desktop and mobile widths:

```bash
cd web
npx playwright@1.63.0 install chromium
npm run build
npm run preview -- --port 4173
node scripts/capture-case-study.mjs \
  --url http://localhost:4173 \
  --output ../.amp/in/case-study
```

The script writes `gameplay.webm` plus a poster frame. The committed site uses H.264 MP4 for compatibility; this host's Playwright-bundled ffmpeg has no H.264 encoder, so the conversion below used an ffmpeg build with `libx264`:

```bash
ffmpeg -ss 2.5 -t 14.5 -i gameplay.webm \
  -vf "scale=960:-2:flags=neighbor" \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p \
  -movflags +faststart -an gameplay.mp4
ffmpeg -ss 8 -i gameplay.webm -frames:v 1 \
  -vf "scale=960:-2:flags=neighbor" gameplay-poster.png
```

If the capture script cannot run (for example, no browser can be installed), the blocker must be recorded and the page must keep the real poster/recording explanation rather than substitute synthetic gameplay.

DQN evaluation requires a checkpoint and a compatible CUDA/PyTorch environment:

```bash
python -m scripts.evaluation.evaluate_dqn --help
```

The deployed ONNX policy is trained under Contract v2. Do not evaluate it under v3 and call the result a v2 comparison, or vice versa.

## Artifact provenance and gaps

| Item | Value | Location |
| --- | --- | --- |
| ROM | ALE/Breakout-v5, ROM SHA-256 `376323f0...710c6fd5` | `configs/eval/breakout_completion_audit_v1.json` |
| ALE / Gymnasium | ale-py 0.12.0 / gymnasium 1.3.0 | same audit config |
| Contract v2 | SHA-256 `7eca5ae5...dc31a2a` | `configs/eval/breakout_contract_v2.json` |
| Contract v3 | SHA-256 `d97b7fcb...25c5a786` | `configs/eval/breakout_contract_v3.json` |
| Frozen DQN | 3,292,837 parameters, 2.5M transitions, training seed 2022 | `web/public/models/final_model/model.onnx.metadata.json` |
| ONNX runtime model | 13,175,034 bytes, SHA-256 `cf90c74b...ef7f12` | `web/public/models/final_model/model.onnx` |
| Browser parity | 60/60 actions, max Q diff 0.0 | model metadata |

**Missing raw artifacts.** The DQN holdout episodes, the 15-seed controller episodes, the reward-shaping and PER run directories, and the earlier Laya service runs are not in `main`. The 2026-10-29 browser-session records are committed under `docs/final-results/evidence/laya-vision/`. The repository ignores `outputs/`, `experiments/`, `evaluations/`, and `assets/` by design. No archive or release currently holds them. Until one does, the corresponding numbers remain **historical reported**, not independently reproducible.

This is a limitation of the closeout, not a reason to fabricate logs. The expected paths, producers, and hashes for each missing artifact are listed in [`artifact-manifest.json`](artifact-manifest.json). See [`handoff.md`](handoff.md) for the proposed archive/release step.

## Claims this project does not make

- No verified clear, solved game, reliable clear, or world record.
- No claim that the RGB controller fairly beats DQN. Contract v2 and v3 are not directly comparable.
- No claim that reward shaping or PER produced a stable improvement.
- No claim that Laya Vision is zero-shot, or that its latency predicts gameplay quality.
- No claim that "Unbeatable" is an experimental result. It is a **UI difficulty label**; it means the browser injects no deliberate mistakes, not that the policy cannot lose.
- No claim that a 50-episode browser evaluation or a formal evaluated Laya episode was run in this closeout. Two short interactive sessions are recorded in [laya-vision.md](laya-vision.md); they are not evaluator episodes.

## Related records

- Roadmap: [#14](https://github.com/Tommyweige/breakout-rl-engineering/issues/14) — Phase 1 incomplete; Phase 2/3 deferred.
- Closeout: [#73](https://github.com/Tommyweige/breakout-rl-engineering/issues/73).
- Controller: [#25](https://github.com/Tommyweige/breakout-rl-engineering/issues/25), [PR #26](https://github.com/Tommyweige/breakout-rl-engineering/pull/26).
- Reward shaping: [#9](https://github.com/Tommyweige/breakout-rl-engineering/issues/9), [#10](https://github.com/Tommyweige/breakout-rl-engineering/issues/10).
- PER: [#12](https://github.com/Tommyweige/breakout-rl-engineering/issues/12), [PR #16](https://github.com/Tommyweige/breakout-rl-engineering/pull/16).
- Laya: [#69](https://github.com/Tommyweige/breakout-rl-engineering/issues/69), [#70](https://github.com/Tommyweige/breakout-rl-engineering/issues/70), [#71](https://github.com/Tommyweige/breakout-rl-engineering/issues/71), [#72](https://github.com/Tommyweige/breakout-rl-engineering/issues/72), [draft PR #74](https://github.com/Tommyweige/breakout-rl-engineering/pull/74).
- Development articles: [`docs-ironman-series`](https://github.com/Tommyweige/breakout-rl-engineering/tree/docs-ironman-series/docs).
