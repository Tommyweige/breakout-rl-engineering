# Day 30 Final Engineering Review

## Final product

- Production target: `https://breakout-rl-browser.pages.dev/` (custom domain pending DNS)
- Hosting: Cloudflare Pages project `breakout-rl-browser`
- Runtime: fully client-side Browser; no Python server and no GPU inference API
- Product surface: Human Breakout on the left, RL Agent Breakout on the right
- Human input: Keyboard / Mouse switch without episode reset
- Shared controls: Start / Pause / Restart

## Backend decision

The production gameplay strategy is WebGPU preferred with truthful WASM
fallback. Day 29 fixed-state validation produced 60/60 action agreement for
both providers, and the paired Browser score distributions matched. Day 29
full-loop timing favored WebGPU on the captured Chrome environment, while WASM
remains the compatibility path. Formal validation and evaluation continue to
require an explicit backend and fail closed when the actual provider differs.

Final machine-readable evidence records the selected backend, fallback, model
hash, provider evidence and the browser diagnostics. It does not present the
fallback as WebGPU.

## Contract v2 source of truth

Final evaluation and artifact lineage must point to:

```text
configs/eval/breakout_contract_v2.json
```

The final JSON records the contract id, source path, SHA256, `fire_reset`,
`terminal_on_life_loss`, `frame_skip`, `frame_stack`, sticky-action
probability, `ale.game_truncated` time-limit semantics, raw reward rule and
the concrete final seeds. Contract v1 evidence remains a historical Day 15
milestone and is not treated as a same-protocol final baseline.

## Evaluation reuse boundary

Day 29 already contains a clean 30-episode WASM/WebGPU Browser comparison. Day
30 preserves that artifact and supplements the selected WebGPU backend with
20 new native-derived seeds, producing exactly 50 unique selected-backend seeds without
silently rewriting historical evidence. The final comparison JSON therefore
contains:

- WASM: the reused Day 29 30-episode baseline;
- WebGPU: the reused Day 29 30 episodes plus new seeds `323–342`;
- `uniqueSeedCount=50` for the selected WebGPU result;
- paired differences for the seed overlap that exists in both result sets;
- raw episode traces and aggregate values that can be recomputed from them.

The exact aggregates are intentionally read from
`assets/day30/final-browser-score-comparison.json` after the Browser capture;
this report does not duplicate numbers that could drift from the source JSON.

The captured final values are:

| backend | episodes | mean | median | std | P10 / P90 | min / max | success / crash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WASM (reused Day 29 baseline) | 30 | 41.87 | 37.5 | 14.86 | 25.9 / 61 | 15 / 74 | 30 / 0 |
| WebGPU (Day 30 selected backend) | 50 | 42.12 | 38 | 15.26 | 25.9 / 60.2 | 15 / 94 | 50 / 0 |

The 20 supplemental WebGPU runs use native-derived reset data for the new seeds
`323–342`. They are not repeated Day 29 episodes, so the final selected result
has `episodeCount=50` and `uniqueSeedCount=50`.

## Product QA

The final Browser smoke covers:

- two independent ALE canvases;
- real AI gameplay and Score/Lives updates;
- Keyboard left/right input;
- Mouse left/right input with a dead zone;
- Keyboard/Mouse conflict isolation;
- pause stops further loop decisions;
- resume and repeated restart;
- one side reaching game over without destroying the other side;
- forced-WASM fallback smoke with the same product interactions;
- no page errors, request failures or blocking HTTP responses.

Responsive QA covers 1920px, 1366px, 768px and 390px viewports. The game
canvases remain the visual focus; technical evidence is available only on
`/?debug=1`.

The Pages production deployment smoke passed at:

```text
https://breakout-rl-browser.pages.dev/
```

It recorded `actualGameplayBackend=webgpu`, two ALE canvases,
`crossOriginIsolated=true`, Keyboard `LEFT`, Mouse `LEFT`, pause/resume/restart,
and zero console/page/request/HTTP errors. The Pages custom-domain mapping for
`breakout.tommypan.dev` was created through the Pages API but remains pending
because public DNS is still delegated to Name.com and does not yet publish the
required Pages mapping. No second Pages project or HugoBlog runtime change was
made.

The same production URL was also run with Chrome WebGPU disabled. That fallback
smoke observed `actualBackend=WASM`, preserved the dual-ALE and input checks,
and had zero console/page/request/HTTP errors.

## Limitations

1. Browser fixed-seed preprocessing parity is strong for the recorded fixture,
   but arbitrary interactive seed streams still use the documented fallback;
   overall Browser/native environment parity remains partial.
2. Mouse control uses a canvas center-relative policy because the Browser
   adapter does not expose a reliable paddle coordinate. It emits legal ALE
   actions and does not mutate emulator state directly, but it is not a
   pixel-perfect paddle tracker.
3. The Pages production deployment is smoke-tested, but the custom-domain HTTPS smoke
   cannot pass until the DNS owner publishes the pending mapping for
   `breakout.tommypan.dev`. The public target remains
   `https://breakout.tommypan.dev` once that DNS gate is cleared.

## Artifact lineage

Run the final validator after capture:

```text
python -m scripts.analysis.validate_final_artifacts --production-url https://breakout-rl-browser.pages.dev/
```

The validator checks the Day 21 final model and holdout, Day 22 ONNX export,
Day 23 parity, Day 24 latency, Day 25 precision decision, Day 26 TensorRT
comparison, Day 27 WASM, Day 28 WebGPU, Day 29 dual Browser evidence, Contract
v2 and the Day 30 screenshot, score comparison, distribution figure and final
validation. It writes `assets/day30/final-artifacts.json` and records whether
all checks passed.
