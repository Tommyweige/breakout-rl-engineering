# Day 28 WebGPU inference report

## Outcome

Day 28 continued the Day 27 Browser page and added an explicit WASM/WebGPU
backend selector, truthful actual-backend reporting, fixed-state parity with
Q-margin diagnostics, a single in-flight inference scheduler, and a same-page
batch=1 benchmark. The formal evidence ran on the existing
`breakout-rl-browser` Cloudflare Pages project:

```text
https://day28-webgpu.breakout-rl-browser.pages.dev/
```

The formal Chrome run successfully created an `executionProviders: ['webgpu']`
session. It did not use a WASM fallback.

## Runtime identity

| field | value |
| --- | --- |
| Browser | Chrome 145.0.7632.117 |
| Platform | Windows |
| ORT Web | 1.29.0 |
| Model SHA256 | `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12` |
| Environment Contract v2 | `day15-breakout-evaluation-v2-fire-reset` / `configs/eval/breakout_contract_v2.json` |
| Contract SHA256 | `e9947fc3a1235100aa92f75a79cf33668b36b4f325f22a0bc8b9546a356b85ba` |
| WebGPU support | `navigator.gpu=true`, adapter available, secure context |
| Requested/actual | WASM/WASM and WebGPU/WebGPU |
| Model/runtime page | `https://day28-webgpu.breakout-rl-browser.pages.dev/` |

Chrome emitted one non-fatal warning that `powerPreference` is ignored by
`requestAdapter()` on Windows. No console errors, page errors, runtime request
failures, or HTTP resource failures were observed.

## Fixed-state validation

Both backends consumed the same 60 Day 22 probe observations and Python FP32
reference. The validation compares all four Q-values per sample, greedy action,
and Q-margin (top Q-value minus the runner-up Q-value).

| backend | max abs Q error | mean abs Q error | action agreement | max abs margin error | passed |
| --- | ---: | ---: | ---: | ---: | --- |
| WASM | `0.0001616478` | `0.0000370334` | `60/60` | `0.0000560284` | yes |
| WebGPU | `0.0001618862` | `0.0000368863` | `60/60` | `0.0000560284` | yes |

The WebGPU artifact is `assets/day28/webgpu-validation.json`. The browser
screenshot `assets/day28/webgpu-validation.png` visibly shows
`Requested / actual = WEBGPU / WEBGPU`, WebGPU support, model hash, ORT version,
browser/version, platform, environment parity, active GPU-device evidence,
validation artifact/page identity, Q-value comparison, action agreement, and
Q-margin fields.

The artifact records environment parity as `partial`: the Browser build uses
fixed observations only and does not execute ALE/Breakout-v5 stepping, sticky
actions, serve/life-loss FIRE semantics, TimeLimit handling, episode seeds,
evaluation epsilon, or raw episode reward aggregation. It references the
canonical Contract v2 id/path/SHA so a later gameplay build cannot mistake this
inference result for a full environment validation.

## Benchmark

The benchmark ran in the same Browser page and used the same model and first
fixed observation for both providers. It performed 10 warm-ups and 100 measured
samples per provider. The timed scope is:

```text
prepared Float32Array → tensor/session.run → q_values ready
```

Session initialization and warm-ups are excluded. Raw browser samples are
stored in `assets/day28/web-benchmark.json`; from the repository root, the
figure is generated from those raw samples by:

```text
conda run -n breakout-rl-engineering python -m scripts.visualization.visualize_day28_webgpu --input assets/day28/web-benchmark.json --output assets/day28/wasm-vs-webgpu-latency.png
```

| backend | P50 | P95 | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WASM | `0.9950 ms` | `1.3850 ms` | `1.0430 ms` | `0.1787 ms` | `0.800 ms` | `1.655 ms` |
| WebGPU | `3.9850 ms` | `5.1008 ms` | `4.0106 ms` | `0.6511 ms` | `2.965 ms` | `5.385 ms` |

Observed conclusion: for this model, browser, GPU, and prepared-input batch=1
scope, WebGPU is slower than WASM. This is a deployment measurement, not a
claim about all models or all browser workloads.

## Implementation

- `OrtWebPolicy` now supports `wasm` and `webgpu` through one policy interface.
  WebGPU feature detection and ORT session creation are explicit; failed GPU
  initialization remains unavailable/error instead of falling back.
- `runFixtureValidation` is backend-aware and records Q-margin diagnostics,
  browser/platform identity, and WebGPU support metadata.
- `InferenceScheduler` rejects a second overlapping request and exposes
  `idle`, `running`, `in-flight`, `paused`, and `error` states for Day 29.
- `runBrowserBackendComparison` runs both 60-state validations and the raw
  batch=1 benchmark in one page.
- `capture-day28-evidence.mjs` is the repeatable Pages/Chrome capture flow.
- `visualize_day28_webgpu.py` reconstructs P50/P95 from the raw browser samples
  and writes a provenance sidecar next to the PNG.
- `assets/day28/webgpu-inference-flow.mmd` is the Mermaid source for the
  rendered structural flow diagram.

## Verification

The following checks passed on the Day 28 branch:

```text
npm run typecheck
npm test
npm run build
npm run qa:visual -- --url http://127.0.0.1:5174/
npm run capture:day28 -- --url https://day28-webgpu.breakout-rl-browser.pages.dev/
conda run -n breakout-rl-engineering python -m scripts.visualization.visualize_day28_webgpu --input assets/day28/web-benchmark.json --output assets/day28/wasm-vs-webgpu-latency.png
```

The responsive QA matrix covered 1440, 1280, 768, 390, and 375 pixel
viewports, with no horizontal overflow, no canvas/gameplay regression, intact
keyboard/shared controls, and a passing WASM regression gate. The formal
capture covered the Pages HTTPS path and required actual WebGPU.

## Scope boundary

Day 28 does not claim ALE gameplay, episode score parity, or an adoption
decision independent of the measured environment. Day 29 can reuse the same
page, backend selector, policy contract, and single-flight scheduler when the
real ALE browser runtime is connected.
