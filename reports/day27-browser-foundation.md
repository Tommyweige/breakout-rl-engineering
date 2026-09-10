# Day 27 Browser Foundation Report

## Outcome

Day 27 的 Browser foundation 已完成：FP32 ONNX → ORT Web WASM → 60-state Browser validation → Cloudflare Pages PASS。

Day 27 沒有加入假的 Breakout animation，也沒有把 fixed-fixture parity 當成整體 gameplay quality 評估。

## Verified model packaging

從 repository root 執行：

```text
python -m scripts.deployment.prepare_web_model
```

實際 NPZ schema：

- `assets/day22/inference/probe_states.npz`: `observations`, shape `(60, 4, 84, 84)`, dtype `uint8`；另有 metadata scalar。
- `assets/day22/inference/pytorch_reference.npz`: `observations`, `model_inputs`, `q_values` shape `(60, 4)`, `greedy_actions` shape `(60,)`，以及 metadata scalar。

Packaging 產生並驗證了：

- `web/public/models/final_model/model.onnx`
- `web/public/models/final_model/model.onnx.metadata.json`
- `web/public/inference_spec.json`
- `web/public/web-model-manifest.json`
- `web/public/fixtures/day22-probe-observations.u8`
- `web/public/fixtures/day22-reference.json`

Packaged ONNX SHA256：

```text
cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12
```

沒有替換 checkpoint，也沒有手工製造 fixture。

## Browser implementation

實作包含：

- `OrtWebPolicy` 只以 `executionProviders: ['wasm']` 建立正式 correctness session；WASM 初始化失敗時會報錯，不會自動改用其他 provider。
- `onnxruntime-web/wasm` entrypoint，避免 Cloudflare Pages 25 MiB 單檔限制。
- manifest、inference spec、fixture shape、dtype、action mapping 和 row count 的 runtime validation。
- uint8 observation 除以 `255.0`，再以 `[1, 4, 84, 84]` float32 tensor 執行 inference。
- Agent panel 顯示 model/backend、ORT version、代表性 Q-values/reference Q-values、selected action、error、action agreement 和 disagreements。
- shared `Start Both`、`Pause Both`、`Reset Both` 及 `idle/loading/ready/running/paused/error` 狀態。
- `window.__day27Validation` 與可下載 JSON seam；browser automation 直接從真正頁面讀回結果，不人工抄數字。
- Cloudflare `_headers` 預留 `Cross-Origin-Opener-Policy: same-origin` 和 `Cross-Origin-Embedder-Policy: require-corp`。

初始 bundle 使用 JSEP WASM 時超過 Pages 單檔 25 MiB；改成官方 `/wasm` entrypoint 後，正式 WASM asset 為 13,961,845 bytes，production build 可部署。Pages production 的 COOP/COEP 讓 ORT 自動多執行緒初始化在本機環境卡住，因此 Day 27 correctness baseline 明確固定 `numThreads = 1`，使 localhost 和 Pages 使用相同的 WASM 行為。

Repository 仍保留 `runWebGpuSmoke.ts` 作為 preliminary out-of-scope Day 28 groundwork；Day 27 App、正式 artifact、capture gate 和 PASS/FAIL state 都不 import、invoke 或依賴它。

## Runtime evidence

### Localhost

- Browser: Chrome 145.0.7632.117
- ORT Web: 1.29.0
- URL: `http://127.0.0.1:5173/`
- Requested backend: `wasm`
- Actual backend: `wasm`
- Sample count: 60
- Max absolute Q error: `0.00016164779663085938`
- Mean absolute Q error: `0.000037033359209696455`
- Action agreement: `1.0` (`100%`)
- Disagreement indices: `[]`
- Machine-readable result: `assets/day27/browser-wasm-validation-local.json`

### Cloudflare Pages

Cloudflare project：`breakout-rl-browser`。本次 redesign 的 verified preview URL：

```text
https://day27-redesign.breakout-rl-browser.pages.dev/
```

Wrangler 同時回傳 immutable deployment URL `https://95b250be.breakout-rl-browser.pages.dev`。頁面以正常 Chrome 開啟並執行 validation，再由 Playwright capture helper 以相同的、沒有 WebGPU-specific flags 的 Chrome 設定重複驗證。

Verified preview result：

- Browser: Chrome 145.0.7632.117
- ORT Web: 1.29.0
- Requested backend: `wasm`
- Actual backend: `wasm`
- Sample count: 60
- Max absolute Q error: `0.00016164779663085938`
- Mean absolute Q error: `0.000037033359209696455`
- Action agreement: `1.0` (`100%`)
- Disagreement indices: `[]`
- Console errors: none
- Runtime request failures: none
- ONNX, fixture files and WASM response paths returned successfully; WASM response MIME was `application/wasm`.

No custom domain was used. The verified preview loaded the same production `dist` assets as the local build.

正式 artifact：`assets/day27/browser-wasm-validation.json`。

## Tests

Web package versions and commands:

- Node `v24.13.0`
- npm `11.6.2`
- `onnxruntime-web` `1.29.0`
- `npm run typecheck`: pass
- `npm test`: 8 files, 19 tests passed
- `npm run build`: pass
- `package-lock.json`: generated and retained
- Browser capture: `npm run capture:evidence -- --url ...` (headed Chrome, no WebGPU flags)

Python full suite used the repository Conda environment (`Python 3.12.13`, `pytest 8.3.4`) and a temporary directory on the same `E:` drive so Windows path metadata tests do not cross drives:

```text
296 passed, 2 skipped, 5 warnings
```

## Evidence files

- `assets/day27/browser-wasm-validation.json`
- `assets/day27/browser-wasm-validation-local.json`
- `assets/day27/dual-panel-browser-foundation.png`
- `assets/day27/browser-wasm-validation.png`
- `assets/day27/dual-panel-browser-foundation-local.png`
- `assets/day27/browser-wasm-validation-local.png`

The two canonical PNGs are real Chrome screenshots from the production Pages URL: the first shows the Human/RL shell before validation, and the second shows the loaded model, actual WASM backend, representative Q-values/reference Q-values, metrics, and selected action.
