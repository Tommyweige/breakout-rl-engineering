# Handoff — Breakout RL Engineering closeout

**Date:** 2026-10-28; integration of the controller, detector, reproduction and i18n branches 2026-10-29
**Scope:** issue #73 closeout only. No new large-scale training, no Laya fixes, no merges, no deployment, no tag, no Release.
**Branch:** `codex/issue73-closeout` → PR to `main`.

## 1. What this closeout changed

- Added the audited results index in [`README.md`](README.md), the machine-readable [`summary.json`](summary.json), the experiment ledger [`evidence-inventory.csv`](evidence-inventory.csv), and this handoff.
- Added an independent, directly linkable web page at `/case-study` in the browser app (`web/case-study.html`), with a prominent entry from the game page.
- Added a rerunnable capture script for real gameplay screenshots/video under `web/scripts/capture-case-study.mjs`.
- Added schema/provenance/link checks in `tests/test_final_results_summary.py` and web render tests.
- Updated `README.md` and `README.zh-TW.md` with a frozen-status summary, key findings, reproducibility commands, and case-study/demo/evidence links.
- Added a narrow `.gitignore` exception so `docs/final-results/` can exist on `main` despite the general `docs/*` rule. No other historical material was restored.
- 2026-10-29: merged the controller re-evaluation (`audits/browser-clear/`: native ALE official 15 seeds at `a8445df`, the exact 137.47 reproduction, and a 120-episode WASM distribution), the clear-detector validation (`docs/clear-semantics.md`, genuine v2/v3 clear fixtures, `count_visible_bricks`), the reproduction path (`docs/repro/`, `scripts/smoke/`, CPU CI), and the bilingual case study. `summary.json` now leads with the controller's 4/15 canonical clear detections.

## 2. Freeze state

| Item | State |
| --- | --- |
| Evidence index | complete and committed |
| Machine-readable summary | complete and committed |
| Case-study page | implemented; browser-rendered verification recorded in the PR |
| DQN results | frozen as historical reported; raw episodes missing |
| RGB controller results | `a8445df`: 4/15 canonical clear detections, mean 721.8 (native ALE, official 15 seeds, per-episode CSV committed); PR #26 law `50ee352`: 137.47, 0/15, reproduced exactly |
| DQN Contract v3 baseline | mean 6.93, 0 clears (seed 11, 2M frames); historical reported, raw episodes missing |
| Clear detector | validated on genuine v2/v3 two-wall clears; 6/6 mutants killed; fail-closed (unknown is never 0) |
| Laya line | frozen as incomplete; draft PR #74 not merged |
| #14 Phase 1 | canonical clear detections exist (4/15); no run has passed artifact provenance validation, so Phase 1 is not declared complete here |
| #14 Phase 2/3 | **deferred, not a blocker** |
| World record / human timed challenge | not attempted |
| Issues closed by this closeout | none |
| Deployment / tag / Release | none |

**Verified vs reported.** The model hashes, ONNX parity, contract hashes, completion-detector configuration and its validation on genuine clears, the `a8445df` controller's native 15-seed run (4/15 canonical clear detections, mean 721.8), the exact reproduction of the PR #26 law's 137.47, the 120-episode WASM audit, and the 2026-10-29 Laya browser session in [`laya-vision.md`](laya-vision.md) are audited in this repository. The DQN 30.933 (v2) holdout, the DQN 6.93 (v3) baseline, the reward-shaping and PER results, and the earlier Laya latency, memory, and control-diagnostic numbers are **historical reported** or **missing raw artifacts**; they are not independently reproducible from `main`.

## 3. Implementation status vs measurement status

### Implemented

- Results index, summary JSON, CSV ledger, handoff.
- Case-study page with real gameplay media and linked evidence.
- Data binding from `summary.json`; the page does not hand-type the headline numbers.
- README/README.zh-TW freeze banner and links.
- Closeout schema/link tests and case-study render tests.

### Measured in this closeout

- `npm test` and `npm run build` in `web/` (actual results in the PR).
- Python completion/controller/schema tests in a local CPU-only venv (actual results in the PR).
- Real headless-browser gameplay capture from the built app (screenshots/video; no mock frames).
- Case-study page at desktop and 390x844 viewport, plus reduced-motion behavior.

### Pending human acceptance

- Live deployment to `breakout.tommypan.com` and visual review on a real phone.
- The 60-second continuous desktop closed-loop session and 390x844 play QA for the Laya line (PR #74). A 2026-10-29 session of about 23 s per run is recorded in [`laya-vision.md`](laya-vision.md); it is not yet the required 60-second run.
- A decision on publishing raw artifacts to an archive/Release.
- A decision on a version tag and Release.

## 4. Deferred work and blockers

<a id="laya-line-still-blocked"></a>

### Laya line (PR #74, issues #69-#72) — do not merge yet

PR #74 is **draft** and must remain draft until the missing acceptance evidence exists. The Laya research question is not answered by the current record.

**Why it is still blocked**

1. **No complete evaluator episode.** The control diagnostic uses 3 seeds x 80 decisions (320 native frames per seed). The 2026-10-29 browser session in [`laya-vision.md`](laya-vision.md) plays to game over and scores 4, but it emits no per-episode artifact, ALE-native frame count, or clear status, so it is not a contract evaluation.
2. **Most raw artifacts are still absent.** The CUDA service benchmarks, fixed fixtures, and 1,000-request memory run were local and ignored. The browser-session records are now committed, but the service-side numbers a reviewer would need for the latency and memory claims are not.
3. **Browser QA is partial.** Two 2026-10-29 sessions record rendered presentation cadence, decision latency, controller switching, and pause/resume without JavaScript errors. The required 60-second continuous desktop run and 390x844 play QA are still outstanding.
4. **The 60 native frames/s budget is not met.** The measured decision interval is 67.0 ms against a 66.7 ms budget for 60 native frames/s at four frames per decision, and the panel presents 58.5 frames/s. That is close, but no 60 FPS claim is made, and the bounded display lag added by presentation pacing is still unmeasured.
5. **Cold start is separate and unfixed.** Issue #70 (closed 2026-10-31 with the project) documents a host-RAM/CPU spike from the upstream loader; no production fix is merged. Issue #70 must not be folded into #71/#72.
6. **Source scope is not settled.** #71/#72 are implementation/performance tickets. Their acceptance criteria include fresh benchmarks, 30+ varied RGB states, 300+ requests, parity evidence, and a 60-second browser session. None of that is complete.

**What would unblock it**

- Publish or attach the exact Laya adapter/benchmark source revision and the raw benchmark output.
- Complete at least one complete seed-101 episode under a declared contract, with per-episode score, ALE-native frames, life losses, action distribution, and clear/termination status.
- Complete the remaining browser QA: a 60-second continuous desktop run and 390x844 play QA with measured human tick lateness and presentation lag. Presentation cadence, decision latency, and switching are now recorded in [`laya-vision.md`](laya-vision.md).
- Record the cold-start measurements for #70 on the same machine and make the readiness gate fail visibly.
- Run the matched DQN comparison under v2 (or explicitly declare not comparable), separate from any v3 controller comparison.

**Do not**: reopen or retag #70/#71/#72 as fixed; do not merge PR #74; do not infer a Laya score from its latency; do not call it zero-shot; do not describe the control diagnostic as a quality improvement.

### #14 roadmap — Phase 2/3 deferred

- Phase 1 (First Clear): the `a8445df` controller has 4/15 **canonical clear detections** under native ALE and the official Contract v3 seeds. The controller evaluator does not perform artifact provenance validation, so this closeout does not declare Phase 1 complete; that decision belongs in #14.
- Phase 2 (reliable clear) and Phase 3 (timed challenge / human record) were **not started**. They are deferred, not blockers for this closeout.
- #14 was closed on 2026-10-31 by the owner: the project is frozen, Phase 1 was reached by the tracker (4 of 15), and Phase 2/3 were not run. Do not reinterpret its criteria.

### World record / human timed challenge

Not attempted. No world-record, "solved", or human-comparable claim is supported.

## 5. Release recommendation (not executed)

**Suggested version:** `v1.0.0-closeout` (or a dated tag such as `v2026.10.28-closeout`). Do not create it until the PR is reviewed and the deployment decision is made.

**Suggested changelog**

- Add the frozen final-results index, machine-readable summary, evidence ledger, and handoff.
- Add an independent case-study page and a prominent entry from the game page.
- Add real gameplay capture and schema/provenance/link checks.
- Update both READMEs with frozen status, key findings, limitations, and reproducibility entry points.
- Add the controller re-evaluation, clear-detector validation, reproduction path with CPU CI, and the bilingual case study.
- No gameplay, contract, or model change; `breakout_rl/completion.py` gains only the screen-level `count_visible_bricks` helper used by the detector tests.

**Deployment checklist**

1. `cd web && npm ci`
2. `npm test`
3. `npm run build`
4. Serve `web/dist` and verify `/` and `/case-study` return 200; `/case-study.html` returns a one-hop 308 to `/case-study`; `/results` returns a one-hop 301 to `/case-study`; `/final-results/summary.json`, the video, and the poster return 200.
5. Verify the case-study page at 1440 px and 390x844, keyboard-only, and `prefers-reduced-motion: reduce`.
6. Verify the game page still starts both games, switches difficulty, pauses, resets, and opens the case study.
7. Deploy through the existing Cloudflare Pages workflow only after human review.

**Rollback**

- Cloudflare Pages: roll back to the previous deployment in the dashboard.
- Git: revert the closeout commit or `git revert <merge-commit>`; no data migration is involved.
- The case-study page and docs are additive; reverting them cannot corrupt model artifacts or contracts.

## 6. `tommypan.com` handoff (not touched)

The personal-site repository was not accessed or modified. If the site owner wants a project card, use this copy and link the primary evidence:

> **Breakout RL Engineering — frozen evidence closeout.** A DQN family, a deterministic RGB predictive controller, and a frozen Laya Vision side experiment, compared only where their contracts allow. With no training, the controller reaches the 864-point two-wall clear in 4 of 15 official Contract v3 episodes by canonical clear detection (native ALE, mean 721.8); the Contract v3 DQN baseline scored 6.93 with 0 clears. The closeout publishes the contracts, metrics, provenance, and limitations instead of claiming a win.
>
> Results: <https://github.com/Tommyweige/breakout-rl-engineering/blob/main/docs/final-results/README.md>
> Case study: <https://breakout.tommypan.com/case-study>
> Play: <https://breakout.tommypan.com/>

Suggested section labels: `Research question`, `What was built`, `What the evidence supports`, `What is still open`. Do not use "solved", "unbeatable", "world record", or "fair winner" in the project card. "Unbeatable" is only the UI difficulty label.

## 7. Evidence archive recommendation (not executed)

The raw logs are the largest remaining gap. A future closeout step should create a GitHub Release (or an external archive) containing:

- DQN holdout episode rows and the exact evaluation command/config hash.
- The `a8445df` controller's full `results.json` and per-seed traces (its `episodes.csv`, summary and diagnostics are already in `audits/browser-clear/native-15/`).
- Reward-shaping and PER comparison reports plus per-seed artifacts.
- Laya benchmark traces and any complete episode.
- SHA-256 for each archive and a `README` mapping each file to the issue/PR.

Do not upload model weights without confirming the license and distribution rights. Laya weights are CC BY-NC-SA 4.0.

## 8. Residual risks

- The historical numbers cannot be independently reproduced from `main` until raw artifacts are archived.
- The DQN 30.933 holdout is a single model and 15 episodes; it is not a population estimate.
- The `a8445df` controller result is one 15-seed run of one deterministic controller (bootstrap 95% CI for the clear rate 10-50%). Its 4/15 are canonical clear detections; the evaluator does not perform artifact provenance validation.
- The committed native run was made from audit commit `9d88918` with untracked audit files present (`working_tree_dirty=true`). Its product code is identical to `a8445df`, and a clean-worktree run at `77a0c7b` (same source sha256 `5989e617…`) gave identical per-episode results (see `docs/clear-semantics.md`), but that clean run's raw output is not committed.
- The 120-episode WASM audit is supplementary: 115 of its seeds are outside the contract and its sticky-action RNG differs from native ALE.
- The deployed production bundle was not measured; the audits ran HEAD `a8445df`.
- The Laya line has no complete evaluator episode; its only gameplay evidence is the two short interactive browser sessions recorded in `docs/final-results/laya-vision.md`.
- A future agent could mistakenly compare v2 and v3 scores. The summary JSON and case-study page label the comparison `not_directly_comparable`; keep that label.
- A future agent could treat "Unbeatable" as a result. It is a UI difficulty label only.

## 9. Issue hygiene

No issue was closed, reopened, relabeled, or modified. In particular:

- #14 is closed; Phase 2/3 were not run.
- #70, #71, #72 were closed with the project on 2026-10-31; the cold-start issue in #70 is not fixed.
- PR #74 remains draft.
- No tag, Release, or deployment was created.

If a later decision changes any of these, record the decision in the relevant issue instead of silently changing this handoff.
