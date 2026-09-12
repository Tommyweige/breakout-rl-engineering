# Day 29 Human vs RL Dual Breakout

This is the same Browser page continued from Day 28. It now runs two independent
`@farama/ale-wasm` Breakout environments: the Human panel receives keyboard
input, while the RL panel preprocesses real ALE frames and sends a four-frame
observation to the Day 21 canonical FP32 ONNX model through ONNX Runtime Web.

The page keeps the explicit WASM / WebGPU selector. A requested WebGPU session
must expose an actual ORT WebGPU device; initialization errors remain visible
and are never silently relabeled as WASM.

## Local development

Requirements:

- Node.js 20.19+
- npm
- Chrome with WebGPU support for the formal WebGPU smoke
- Cloudflare authentication only when deploying the Pages preview

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
`@farama/ale-wasm@0.12.0` and copy the canonical
`configs/eval/breakout_contract_v2.json` into the same-origin public tree.
The generated browser assets are intentionally not committed; the package
lockfile and preparation script are the reproducible source.

## Browser capture and evaluation

The repeatable capture flow validates both providers, starts the same-page
Human + RL product, waits for a WebGPU episode to finish, captures the real
gameplay and preprocessing canvases, then runs the fixed 30-seed Browser policy
evaluation for WASM and WebGPU:

```bash
npm run capture:day29 -- --url https://<preview>.breakout-rl-browser.pages.dev/
```

The capture requires `actualBackend=webgpu` for the formal smoke and records
the model SHA256, Chrome version, ORT Web version, Contract v2 parity status,
validation artifacts, per-episode raw return/length/game-over/error fields,
and preprocessing/inference/total decision timing.

## Pages preview

Build and deploy to the existing `breakout-rl-browser` project:

```bash
npm run build
npx wrangler pages deploy dist --project-name breakout-rl-browser --branch day29-human-vs-rl
npm run capture:day29 -- --url https://day29-human-vs-rl.breakout-rl-browser.pages.dev/
```

Cloudflare Pages is static hosting only. The Browser loads the model, ORT Web
assets, ALE WASM runtime and preloaded Breakout ROM locally; no Python or GPU
inference server is involved.
