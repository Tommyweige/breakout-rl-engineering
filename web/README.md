# Day 27 Browser Foundation

This directory is the browser-side foundation for Issue #29.

Day 27 deliberately does **not** implement fake Breakout gameplay. The Human and Agent panels are shells until the real ALE browser runtime is connected later.

## Prerequisites

- Node.js 20.19+ (or a newer supported Node release)
- npm
- Python environment that can read the existing Day 22 NumPy fixtures
- Cloudflare authentication only when deploying the preview

## Prepare verified model + fixtures

From the repository root:

```bash
python -m scripts.deployment.prepare_web_model
```

This must create browser assets under `web/public/` from the checked Day 22 FP32 ONNX and fixed fixtures. Do not manually replace the ONNX file with another checkpoint.

## Install and run

```bash
cd web
npm install
npm run typecheck
npm test
npm run build
npm run dev
```

Open the local Vite URL and click **Load model + Validate WASM**.

A PASS is allowed only after the real `onnxruntime-web` WASM session has loaded the packaged FP32 ONNX and executed all fixed fixtures.

For repeatable responsive UI checks, run the Playwright visual QA matrix against localhost (or pass a deployed `--url`):

```bash
npm run qa:visual -- --output-dir ../.qa-redesign
```

The check covers 1440, 1280, 768, 390 and 375 pixel viewports, shared controls, keyboard input, the explicit no-gameplay seam, and the real WASM validation gate. It launches the installed Chrome channel without WebGPU-specific flags.

## Day 27 evidence to generate locally

The implementation agent must generate real evidence rather than committing placeholders:

- `assets/day27/browser-wasm-validation.json`
- `assets/day27/dual-panel-browser-foundation.png`
- `assets/day27/browser-wasm-validation.png`
- `reports/day27-browser-foundation.md`
- `docs/day27-onnx-runtime-web.md`

The browser validation artifact should record at least:

- ORT Web version
- browser name/version
- requested backend = WASM
- actual backend = WASM
- fixed sample count
- max / mean Q-value error
- action agreement
- disagreement indices
- model SHA256 from `web-model-manifest.json`
- preview URL when available

## Preview deployment

After a production build succeeds, deploy `web/dist` to a Cloudflare Pages preview/project and verify the deployed URL can load:

- ONNX model
- ORT Web WASM runtime assets
- fixed fixtures

Localhost-only validation is not the Day 27 Definition of Done.
