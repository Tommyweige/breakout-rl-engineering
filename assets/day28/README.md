# Day 28 WebGPU evidence

這組 evidence 來自同一個 Day 27 延續頁面、同一份 FP32 ONNX、同一個 Chrome
與同一個 Cloudflare Pages preview：

- preview: `https://day28-webgpu.breakout-rl-browser.pages.dev/`
- browser: Chrome `145.0.7632.117` on Windows
- ORT Web: `1.29.0`
- model SHA256: `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12`
- Contract v2: `day15-breakout-evaluation-v2-fire-reset`
- Contract v2 path/SHA256: `configs/eval/breakout_contract_v2.json` /
  `e9947fc3a1235100aa92f75a79cf33668b36b4f325f22a0bc8b9546a356b85ba`
- Browser environment parity: `partial` (fixed inference only; no ALE stepping)
- WebGPU support: `navigator.gpu = true`, adapter available, secure context
- requested/actual: WASM/WASM and WebGPU/WebGPU
- backend evidence: explicit WASM session and ORT `env.webgpu.device`
- article workflow: `technical-blog-writer` was read before drafting and its
  evidence-first, terminology, Mermaid, and final-review gates were applied
- machine-readable workflow record: `article-workflow.json` (includes skill and
  reference SHA256 hashes)

## Reproduction

From `web/`, build and deploy the same Pages project, then run the real Chrome
capture against the resulting `*.pages.dev` URL:

```text
npm ci
npm run build
npx wrangler pages deploy dist --project-name breakout-rl-browser --branch day28-webgpu
npm run capture:day28 -- --url https://day28-webgpu.breakout-rl-browser.pages.dev/
```

The capture writes `webgpu-validation.json`, `web-benchmark.json`, and the two
Browser screenshots. The benchmark uses 10 warm-ups and 100 measured samples
per provider; session initialization is outside the measured scope. The timing
scope is a prepared `Float32Array` through `session.run` until `q_values` are
ready.

From the repository root, the latency figure is generated only from the raw
browser samples:

```text
conda run -n breakout-rl-engineering python -m scripts.visualization.visualize_day28_webgpu --input assets/day28/web-benchmark.json --output assets/day28/wasm-vs-webgpu-latency.png
```

`webgpu-inference-flow.mmd` is the verified structural source for the rendered
`webgpu-inference-flow.png` diagram. Mermaid CLI rendering used the helper from
the `technical-blog-writer` skill; the source and generated image are kept
together for auditability.

## Observed result

The 60 fixed states passed for both providers with 100% action agreement. In
the Pages capture, WASM measured `0.9950 ms` P50 / `1.3850 ms` P95 and WebGPU
measured `3.9850 ms` P50 / `5.10075 ms` P95. This is a result for this browser,
GPU, model, and prepared-input timing scope; it is not a claim that WebGPU is
always slower or that fixed-state parity proves gameplay quality.
