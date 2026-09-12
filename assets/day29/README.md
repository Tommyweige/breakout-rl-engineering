# Day 29 Browser evidence

This directory contains the canonical Day 29 capture from the same Cloudflare
Pages product continued from Day 28:

- preview: `https://day29-human-vs-rl.breakout-rl-browser.pages.dev/`
- browser: Chrome `145.0.7632.117` on Windows
- ORT Web: `1.29.0`
- model SHA256: `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`
- actual gameplay backend: WebGPU
- environment: `ALE/Breakout-v5` through `@farama/ale-wasm@0.12.0`
- Contract v2: `day15-breakout-evaluation-v2-fire-reset`
- environment parity: `partial` overall; fixed 30-seed evaluation scores are native-aligned, with representative raw-frame/preprocessing parity `full`, while arbitrary interactive seed-stream parity remains explicit `partial`
- `crossOriginIsolated`: `true`; dual canvas/ALE instances: `2`
- product controls: HTTPS Arrow keyboard, Pause/Resume, and repeated Reset Both passed
- validation diagnostics: zero console errors, page errors, request failures, and bad responses

## Reproduction

From `web/`:

```bash
npm ci
npm run build
npx wrangler pages deploy dist --project-name breakout-rl-browser --branch day29-human-vs-rl
npm run capture:day29 -- --url https://day29-human-vs-rl.breakout-rl-browser.pages.dev/
```

The capture flow validates WASM and WebGPU fixed states, starts the real Human
and RL ALE instances, captures the live product and preprocessing canvases, then
evaluates 30 fixed seeds on both browser backends. The fixed seeds use a native
Gymnasium-derived `noop_max=30` / ALE seed manifest. `browser-policy-score-comparison.json`
is the source for the score figure; no numbers are entered manually.

For a long formal score run without waiting for the visual smoke episode, use
the parallel two-page evaluator and then build the checked-in artifacts:

```bash
cd web
npm run capture:day29:overlap -- --url https://day29-human-vs-rl.breakout-rl-browser.pages.dev/?v=DEPLOYMENT --count 30
cd ..
python -m scripts.analysis.build_day29_browser_score_artifacts --browser-input assets/day29/browser-overlap-evaluation.json --page-url https://day29-human-vs-rl.breakout-rl-browser.pages.dev/
python -m scripts.visualization.visualize_day29_browser_scores
```

From the repository root, regenerate the score figure:

```bash
python -m scripts.visualization.visualize_day29_browser_scores
```

The timing, parity, screenshot metadata, score comparison, rendered Mermaid
diagram, and source Mermaid file are kept together so the article claims can be
audited against real Browser output.

The preprocessing differential check is reproducible from the repository root:

```bash
python -m scripts.analysis.capture_day29_native_preprocessing_fixture
python -m scripts.analysis.generate_day29_native_noop_manifest
cd web
npm run capture:day29:preprocessing -- --url https://day29-human-vs-rl.breakout-rl-browser.pages.dev/?v=DEPLOYMENT
cd ..
python -m scripts.analysis.compare_day29_preprocessing
```

It compares raw grayscale frames, final-two-frame grayscale max-pools, 84×84
frames, and 4-frame stacks by max absolute pixel difference, mean absolute
pixel difference, and differing-pixel fraction.
