# Day 28 WebGPU Inference

This directory is the same browser-side product continued from Day 27 for Issue #30.

Day 28 adds a truthful WASM/WebGPU selector, fixed-state correctness validation, a single in-flight inference scheduler, and a same-page batch=1 benchmark. It still does **not** implement fake Breakout gameplay; the Human and Agent panels remain the seam for the real ALE browser runtime in Day 29.

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

Open the local Vite URL and choose **WASM / CPU baseline** or **WebGPU / GPU** in the Agent Panel, then click **Load + Validate selected backend**.

A PASS is allowed only after the real `onnxruntime-web` session has loaded the packaged FP32 ONNX with the explicitly requested execution provider and executed all fixed fixtures. A WebGPU failure is shown as unavailable/error; it is never relabeled as WASM.

Click **Run WASM / WebGPU benchmark** to execute the full 60-state validation for both providers and measure 100 batch=1 samples per provider after 10 warm-ups. Timing starts from a prepared `Float32Array` and excludes session initialization.

For repeatable responsive UI checks, run the Playwright visual QA matrix against localhost (or pass a deployed `--url`):

```bash
npm run qa:visual -- --output-dir ../.qa-redesign
```

The check covers 1440, 1280, 768, 390 and 375 pixel viewports, shared controls, keyboard input, the backend selector, the explicit no-gameplay seam, and the real WASM regression gate. It launches the installed Chrome channel without WebGPU-specific flags. The formal Day 28 evidence capture additionally requires a real WebGPU adapter and a successful `executionProviders: ['webgpu']` session.

## Day 28 evidence to generate locally

The implementation agent must generate real evidence rather than committing placeholders:

- `assets/day28/webgpu-validation.json`
- `assets/day28/web-benchmark.json`
- `assets/day28/dual-panel-webgpu.png`
- `assets/day28/webgpu-validation.png`
- `assets/day28/wasm-vs-webgpu-latency.png`
- `reports/day28-webgpu-inference.md`
- `docs/day28-webgpu-inference.md`

Run the real Browser capture from `web/` with a local or Pages URL:

```bash
npm run capture:day28 -- --url https://<preview>.breakout-rl-browser.pages.dev/
```

The WebGPU validation artifact should record at least:

- ORT Web version
- browser name/version
- platform and user agent
- requested backend = WebGPU
- actual backend = WebGPU
- WebGPU adapter support and secure-context status
- fixed sample count
- max / mean Q-value error
- action agreement
- Q-margin diagnostics
- disagreement indices
- model SHA256 from `web-model-manifest.json`
- preview URL when available

The benchmark artifact retains raw latency samples, P50/P95/mean/std, warm-up count, timing scope, both actual backends, and the embedded WASM/WebGPU validation summaries.

## Preview deployment

After a production build succeeds, deploy `web/dist` to the existing `breakout-rl-browser` Cloudflare Pages project and verify the deployed URL can load:

- ONNX model
- ORT Web WASM runtime assets
- fixed fixtures
- a real WebGPU session when the formal capture browser supports it

Localhost-only validation is not the Day 28 Definition of Done.
