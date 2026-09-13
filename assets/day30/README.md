# Day 30 final evidence

This directory contains the final Browser product evidence for Day 30.

- `final-production-ui.png` — real Pages production screenshot of the player-facing `/` route.
- `final-debug-ui.png` — real `/?debug=1` screenshot with the engineering controls exposed for formal evidence.
- `final-human-vs-rl.png` — real Pages production gameplay screenshot from the final Browser smoke.
- `final-browser-score-comparison.json` — Day 29 WASM 30-episode baseline plus 20 new WebGPU runs for seeds 323–342, merged into 50 unique WebGPU seeds.
- `parallel-increment-webgpu-evaluation.json` — raw 20-episode WebGPU supplement for the new seeds; each episode preserves the Browser action trace and timing arrays.
- `final-browser-score-distribution.png` / `.json` — figure generated from the final comparison JSON.
- `final-validation.json` — Contract v2, backend validation, 50-unique-seed aggregate and Pages production smoke summary.
- `final-artifacts.json` — output of `scripts/analysis/validate_final_artifacts.py`.
- `preview-production-smoke.json` — real `day30-production.breakout-rl-browser.pages.dev` smoke result.
- `production-smoke.json` — real `breakout-rl-browser.pages.dev` production smoke result.
- `fallback-smoke.json` — forced-WASM production smoke result proving the fallback path.
- `preview-qa-final/` — responsive QA screenshots for the deployed Pages preview at 1920, 1366, 768 and 390px.

The supplemental runs use native-derived Browser reset data for the genuinely new
seeds `323–342`. Together with Day 29's 30 unique WebGPU seeds they produce the
final 50-unique-seed coverage. The canonical task definition is
`configs/eval/breakout_contract_v2.json`; overall arbitrary-seed Browser/native
parity remains partial by design.
