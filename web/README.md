# Day 30 Human vs AI Breakout

This is the production Browser product continued from Day 29. It runs two
independent `@farama/ale-wasm` Breakout environments: the Human side accepts
Keyboard or Mouse input, while the RL side preprocesses real ALE frames and
feeds the Day 21 canonical FP32 ONNX model through ONNX Runtime Web.

The product prefers WebGPU when a real adapter and ORT WebGPU session are
available, and falls back to an explicit WASM session when the browser cannot
run WebGPU. The user-facing page does not expose the backend choice; formal
validation and evaluation remain explicit and fail closed at `/?debug=1`.

## Production

Play the final product at:

```text
https://breakout-rl-browser.pages.dev/
```

The `breakout.tommypan.dev` custom-domain mapping is retained but awaits the
Name.com DNS record before it can be smoke-tested.

Cloudflare Pages is static hosting only. The Browser loads the model, ORT Web,
ALE WASM runtime, ROM and Contract v2 locally; there is no Python or GPU
inference server.

## Local development

Requirements:

- Node.js 20.19+
- npm
- Chrome for WebGPU validation
- Cloudflare authentication only when deploying a Pages preview

From the repository root:

```bash
cd web
npm ci
npm run typecheck
npm test
npm run build
npm run dev
```

The `predev` and `prebuild` hooks copy the pinned ALE WASM/data assets from
`@farama/ale-wasm@0.12.0`, Contract v2 and the native reset manifest into the
same-origin public tree.

## Browser QA and final evidence

Run repeatable responsive QA at 1920, 1366, 768 and 390 pixels:

```bash
npm run qa:day30 -- --url http://127.0.0.1:5173/
```

Capture the real product screenshot and responsive product QA from `/`. Formal
validation and evaluation scripts open `/?debug=1` themselves. The Day 30 path
reuses Day 29's 30-episode Browser baseline and adds 20 new native-derived
WebGPU runs for seeds `323–342`:

```bash
npm run capture:day30:ui -- --url https://breakout-rl-browser.pages.dev/
npm run capture:day30:parallel-increment -- --url https://breakout-rl-browser.pages.dev/ --concurrency 4 --seeds 323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342
python -m scripts.visualization.visualize_day30_browser_scores
python -m scripts.analysis.validate_final_artifacts --production-url https://breakout-rl-browser.pages.dev/
```

The raw JSON preserves every seed, score, episode length, termination reason,
runtime error, executed action trace and timing sample. The validator also
checks that the final artifact manifest points back to Contract v2 and the
Day 21–29 evidence lineage.

## Pages preview and production

Build and deploy to the existing `breakout-rl-browser` project:

```bash
npm run build
npx wrangler pages deploy dist --project-name breakout-rl-browser --branch day30-production
npx wrangler pages deploy dist --project-name breakout-rl-browser --branch main # Pages production branch
npm run smoke:day30 -- https://breakout-rl-browser.pages.dev/ ../assets/day30/production-smoke.json
```
