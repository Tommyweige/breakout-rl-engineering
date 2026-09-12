# Day 29 Human vs RL Browser Demo

## Outcome

Day 29 extends the existing Day 28 Cloudflare Pages page instead of creating a
second product. The page now runs two independent `ALE/Breakout-v5` instances:

- Human: keyboard-controlled ALE environment;
- RL Agent: ALE WASM frames → Browser preprocessing → Day 21 canonical FP32
  ONNX → ONNX Runtime Web WASM or WebGPU → action mapping → ALE action.

The formal staging capture ran at:

```text
https://day29-human-vs-rl.breakout-rl-browser.pages.dev/
```

Cloudflare is only static hosting. There is no Python or GPU inference server.

## Runtime identity

| field | value |
| --- | --- |
| Browser | Chrome 145.0.7632.117 / Windows |
| ORT Web | 1.29.0 |
| ALE WASM | `@farama/ale-wasm@0.12.0` |
| model | Day 21 canonical Final Model / FP32 ONNX |
| model SHA256 | `cf90c74b1d09d5d7e2e0c71014f2daef2b5e1833425d300da06e34d1e4ef7f12` |
| requested/actual formal gameplay backend | WebGPU / WebGPU |
| Contract | `day15-breakout-evaluation-v2-fire-reset` |
| environment parity | `partial` overall / fixed evaluation path `full` |
| `crossOriginIsolated` | `true` |
| dual ALE instance count | `2` |
| HTTPS keyboard / Pause / Reset Both | passed |
| console/page/request/HTTP errors | `0 / 0 / 0 / 0` |

## Browser environment contract

The adapter loads `configs/eval/breakout_contract_v2.json` at runtime and
validates its SHA256 against the inference specification. The runtime settings
recorded by the capture are:

| field | Browser value |
| --- | --- |
| ALE raw `frame_skip` | `1` |
| outer action repeat | `4` |
| expected emulator frames / policy decision | `4` |
| max-pool | final two raw grayscale frames |
| processed frame | grayscale `84×84` `uint8` |
| frame stack | `4×84×84` `uint8` |
| sticky action probability | `0.25` |
| Contract v2 FIRE reset | initial serve and after observed life loss |
| action mapping | model `0/1/2/3` → ALE `0/1/3/4` |
| life-loss termination | disabled |
| time-limit source | `ale.game_truncated` |

The native wrapper has additional behavior that is not present in the Contract
v2 JSON: `AtariPreprocessing.reset()` samples `noop_max=30` with an inclusive
`1..30` random NOOP count; Gymnasium derives a separate NumPy seed and ALE seed
from `SeedSequence(episode_seed)`; each raw frame is converted to ALE grayscale
before the final-two-frame max-pool; and `cv2.INTER_AREA` resizes the pooled
grayscale frame. Day 29 records those semantics in
`native-noop-reset-manifest.json` and `environment-parity.json`.

The Browser adapter now applies that path to all fixed Day 29 evaluation
seeds. A representative raw-frame differential fixture covers seeds `101`,
`202`, and `303` across reset NOOP frames, three decisions, grayscale pools,
resized frames, and stacks; every compared group is byte-identical (`max=0`,
`mean=0`, different-pixel fraction `0`). General arbitrary interactive seeds still use a
documented seeded fallback instead of a full NumPy SeedSequence/PCG64
implementation, so the overall environment label remains `partial`.

## Blocker investigation

The low Browser return had two independent causes. First, Browser reset used
the episode seed directly as ALE's `random_seed`, while Gymnasium derives an
`np_seed` for the NOOP draw and a different `ale_seed` for the game/sticky
random stream. Second, the Browser action table swapped the raw ALE codes for
RIGHT and LEFT (`3` and `4`). The first mismatch made later sticky/FIRE
trajectories diverge; the second became visible at the first policy RIGHT/LEFT
decision. The corrected path sets the native-derived ALE seed, performs the
native NOOP count, uses native ALE grayscale before max-pooling, and maps
`RIGHT→3`, `LEFT→4`.

## Fixed-state validation

The same Day 22 fixed observations passed through both explicit providers:

| backend | samples | max absolute Q error | mean absolute Q error | action agreement |
| --- | ---: | ---: | ---: | ---: |
| WASM | 60 | `0.0001616478` | `0.0000370334` | `60 / 60` |
| WebGPU | 60 | `0.0001618862` | `0.0000368863` | `60 / 60` |

This proves model I/O and greedy action agreement on the fixed states. It is
not a gameplay score claim.

## Thirty-episode Browser evaluation

The same 30 fixed seeds ran once with each Browser provider:

| backend | episodes | mean | median | std | P10 / P90 | min / max | success / crash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WASM | 30 | `41.8667` | `37.5` | `14.8632` | `25.9 / 61` | `15 / 74` | `30 / 0` |
| WebGPU | 30 | `41.8667` | `37.5` | `14.8632` | `25.9 / 61` | `15 / 74` | `30 / 0` |

The paired WebGPU minus WASM score difference has mean `0`. The 15 seeds that
overlap Day 26 are exactly equal across native, WASM, and WebGPU:
`101–105 = 73,45,58,47,74`; `202–206 = 58,26,37,45,70`; and
`303–307 = 35,60,48,34,38`. The formal 30-seed Browser set additionally
contains `308–322`, which explains why its aggregate is not the same sample
set as the earlier native 30-episode artifact.

## Full agent-loop timing

The timing artifact includes every decision loop sample, not only the fixed
model fixtures. The final staging capture recorded 31,828 samples per backend:

| backend | samples | preprocessing P50 | inference P50 | total decision P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| WASM | 31,828 | `0.425 ms` | `4.830 ms` | `53.950 / 59.830 ms` |
| WebGPU | 31,828 | `0.325 ms` | `6.280 ms` | `33.080 / 51.730 ms` |

The total includes ALE stepping, preprocessing, inference and action resolution
as measured by the Browser loop. It is therefore different from Day 28's
prepared-input-only benchmark.

## Evidence and reproduction

The real product screenshots are `browser-ai-demo.png`,
`human-vs-rl-dual-breakout.png`, and `browser-preprocessing-debug.png` under
`assets/day29/`. The machine-readable sources are:

- `browser-policy-score-comparison.json` — both 30-episode runs and paired scores;
- each episode's `actionTrace` — requested/executed semantic actions, ALE codes,
  auto-FIRE reason, reward and lives;
- `agent-loop-timing.json` — preprocessing/inference/total decision samples;
- `environment-parity.json` — Contract v2 plus native AtariPreprocessing/reset semantics and fixed-seed parity;
- `preprocessing-parity.json` — raw-frame and model-input pixel comparison evidence;
- `preprocessing-parity-comparison.png` — representative native/Browser frame and absolute-difference image;
- `native-noop-reset-manifest.json` — native-derived `noop_count`, NumPy seed, and ALE seed for fixed evaluation seeds;
- `capture-metadata.json` — browser, backend, model and clean diagnostics;
- `browser-agent-loop.mmd` plus `browser-agent-loop.png` — verified structural flow.

From `web/`, build/deploy/capture with:

```bash
npm ci
npm run build
npx wrangler pages deploy dist --project-name breakout-rl-browser --branch day29-human-vs-rl
npm run capture:day29 -- --url https://day29-human-vs-rl.breakout-rl-browser.pages.dev/
```

Regenerate the score figure from the captured JSON with:

```bash
python -m scripts.visualization.visualize_day29_browser_scores
```

This Day 29 change does not modify `Tommyweige/HugoBlog`.
