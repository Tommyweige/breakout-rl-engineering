# Laya Vision — live browser session (verified)

**Status: verified interactive session, recorded 2026-10-29.** This page records what a
frozen third-party vision model actually did when a person-shaped browser session ran it
as the `Decision Model` controller on the shipped page. It is not a contract evaluation
episode and it is not a method comparison.

Raw records: [`laya-browser-qa-run1.json`](evidence/laya-vision/laya-browser-qa-run1.json),
[`laya-browser-qa-run2.json`](evidence/laya-vision/laya-browser-qa-run2.json).
Producer: [`browser-qa.mjs`](evidence/laya-vision/browser-qa.mjs).

## Claim

With the checkpoint frozen and no project fine-tuning, the browser policy scored **4** and
lost its last life after **22.7 s** of play. Both recorded sessions produced the same
result: score 4, lives 0, 317 and 318 decisions, 22.715 s and 22.751 s.

| Question | Answer | Where |
| --- | --- | --- |
| Can it play? | Yes: it plays in the browser until all five lives are gone | [game outcome](#game-outcome) |
| How well? | Score 4 of 864; game over after 22.7 s | [game outcome](#game-outcome) |
| Is it fast enough? | One decision every 67.0 ms; 50-51 ms of that is model inference | [decision latency](#decision-latency) |
| Does the page render it? | 58.5 presented frames/s on a 169.5 Hz display | [presentation](#presentation) |

**What this is not.** It is not a zero-shot result (the upstream checkpoint was pretrained
on game frames that include Breakout), not a clear, not a score under the formal evaluator,
and not evidence that Laya is better or worse than the DQN or pixel-controller lines. See
[Limitations](#limitations).

## Environment

| Item | Value |
| --- | --- |
| Machine | Windows 11 laptop, WSL2 Ubuntu; RTX 4060 Laptop GPU (8 GB, sm_89), driver 616.92 |
| Display | 169.5 Hz measured requestAnimationFrame cadence |
| Python | 3.12.15, isolated venv outside the repository |
| PyTorch | 2.13.0+cu130, CUDA 13.0, `torch.cuda.is_available()` true |
| Laya package | `laya` 0.2.0.dev0 from commit `9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c` |
| Model service | `python -m scripts.inference.serve_laya_vision`, loopback `127.0.0.1:8766` |
| Web server | Vite 8.2.2 dev server on `0.0.0.0:5180`, proxying `/api/laya` to the service |
| Browser | Chrome 145 on Windows, headed, 1920x960 viewport |
| Page | `http://localhost:5180/?debug=1` |
| Contract | v2, `day15-breakout-evaluation-v2-fire-reset`, frame_skip 4, sticky 0.25, FIRE reset |
| Seed | contract v2 `concrete_episode_seeds[0]` = 101 |

The Python, PyTorch, CUDA, and driver rows are recorded verbatim in
[`service-environment.txt`](evidence/laya-vision/service-environment.txt); the display
cadence and browser identity are recorded in each session JSON.

Service health as returned during the session:

```json
{"ready": true, "device": "cuda", "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
 "dtype": "torch.float32", "variant": "fp32-gpu-preprocess", "inferenceMode": "fixed-eager",
 "preprocessing": {"image_size": 512, "preprocess": "gpu"},
 "modelRevision": "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc",
 "codeRevision": "9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c"}
```

## Model source and revision

| Item | Value |
| --- | --- |
| Weights | `thaitea/laya-vision`, revision `f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc` |
| Main file | `model.safetensors`, 804,696,880 bytes (767 MiB) |
| Other files | `backbone/config.json`, `processor/{chat_template.jinja,processor_config.json,tokenizer.json,tokenizer_config.json}`, `vlm_agent_config.json` |
| License | CC BY-NC-SA 4.0; weights are **not** redistributed in this repository |
| Local cache | `~/.cache/huggingface/hub/models--thaitea--laya-vision` after the pinned download |
| Parameters | about 201M, FP32, GPU image preprocessing at the checkpoint's 512-pixel resolution |

The loader rejects any `laya` install whose `direct_url.json` commit differs from the
pinned code revision, so the runtime identity above is enforced rather than reported.

## Method

The measurement used a separate clean checkout of PR #74 head
`3c79bd4b30817de967aa3fbbf102c4bbf09cd3bd`; this branch does not contain the browser
Decision Model code. The repository under measurement was not modified, and the two
services were not restarted or reconfigured while recording.

1. Start the CUDA service, which warms up twenty predictions before listening.
2. Start the Vite dev server and open the page in headed Chrome on Windows.
3. Select `Decision Model`, press Start, and sample the runtime diagnostics once per second
   until the AI game reports `Game over`.
4. Switch the AI controller to `HARD` (the ONNX WASM DQN), back to `Decision Model`, then
   pause and resume.
5. Sample `nvidia-smi` once per second throughout.

Every number below comes from the two JSON records linked at the top of this page. Each
record stores the full per-second sample list and twenty-one per-decision timing rows, so
any figure can be recomputed from the file rather than trusted from this page.

Reproduce (Windows shell, from the directory holding `browser-qa.mjs`):

```powershell
$env:QA_OUT  = "E:\path\to\output"
$env:QA_URL  = "http://localhost:5180/?debug=1"
node browser-qa.mjs
```

## Game outcome

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Raw score at game over | 4 | 4 |
| Lives at game over | 0 | 0 |
| Play time before game over | 22.715 s | 22.751 s |
| Agent decisions | 317 | 318 |
| Decisions per second | 14.64 | 14.66 |
| Emulator frames per second | 58.57 | 58.65 |
| JavaScript errors | 0 | 0 |

Five lives were lost across the session. Score 4 is raw ALE reward: one point per brick.
Both runs use contract seed 101, so the agreement between them shows determinism for this
seed rather than variance; they differ by one decision and 36 ms.

The 22.7 s figure is the time from the first decision to `Game over`, not the time the page
was open.

## Decision latency

Interval figures come from the loop's rolling window over the last 600 decision intervals;
per-decision figures come from the 21 sampled rows in each record.

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Decision interval p50 | 67.00 ms | 67.00 ms |
| Decision interval p95 | 68.20 ms | 68.20 ms |
| Service inference p50 (p95) | 50.36 (52.53) ms | 51.30 (53.67) ms |
| HTTP request + RGB capture p50 (p95) | 56.10 (59.10) ms | 57.00 (59.40) ms |
| Transport and response p50 | 5.25 ms | 5.34 ms |
| ALE step and frame read p50 | 2.20 ms | 2.30 ms |
| Scheduler wait p50 | 8.40 ms | 8.30 ms |
| Full decision cycle p50 (p95) | 58.30 (61.10) ms | 59.40 (62.40) ms |

Read the interval row as the **delivered** cadence, not as the model's ceiling: the loop
paces Decision Model at 15 decisions per second (60 native frames / 4 frames per decision),
so the interval is 67 ms even though a decision costs about 58 ms. The model itself is the
50-51 ms service row; the remaining time is HTTP transport, one ALE step, frame reads, and
the RGB capture.

## Presentation

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Display cadence (rAF) | 169.5 Hz | 169.5 Hz |
| Render callback interval p50 | 5.90 ms | 5.90 ms |
| Native emulator frames per second | 58.57 | 58.65 |
| Presented frames per second | 58.59 | 58.52 |
| Presented frames counted | 1,267 | 1,268 |
| Maximum buffered presentation frames | 4 | 4 |

The AI panel presents one real emulator frame per 60 Hz presentation slot. Because the
agent produces four native frames per decision at 14.6 decisions per second, it produces
about 58.6 frames per second — under the 60 Hz ceiling, so the ceiling is not what limits
the visible update rate here. The page itself renders at the display's 169.5 Hz; that is
the browser's paint cadence, not the AI panel's update rate, and the two should not be
conflated.

## GPU usage

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| GPU utilization p50 (range) | 72.5% (56-79%) | 75.0% (58-80%) |
| VRAM used | 989 MiB | 989 MiB |
| Board power (range) | 64.4-107.7 W | 53.6-105.1 W |

Utilization is sampled once per second during the session and includes Chrome's
compositing work, so it is an upper bound on the model's share. VRAM was constant at
989 MiB in every sample of both runs.

## Controller switching and pause

| Step | Backend evidence | Decisions/s | Score | Lives |
| --- | --- | ---: | ---: | ---: |
| Decision Model, at game over | Laya CUDA / fixed-eager | 14.64 (run 1) | 4 | 0 |
| Switch to HARD (ONNX WASM DQN) | explicit WASM worker session | 124.0 / 131.9 | 6 / 6 | 4 / 4 |
| Switch back to Decision Model | Laya CUDA / fixed-eager | 14.24 / 13.73 | 1 / 1 | 3 / 3 |
| Pause for 2.5 s | Laya CUDA / fixed-eager | frozen: 114 to 114, 111 to 111 | 1 | 3 |
| Resume | Laya CUDA / fixed-eager | 168 / 167 cumulative | 2 / 2 | 2 / 2 |

Switching in either direction rebuilt the AI episode, so each controller restarted from a
fresh score and lives, swapped the backend label, and produced no JavaScript error in either
run. While paused, the decision count did not move and the presentation queue drained to
zero; after resume, decisions continued from the same episode.

## Limitations

- **Not zero-shot.** The upstream checkpoint was trained on game frames that include
  Breakout expert actions. This session measures a frozen off-the-shelf model, not transfer
  from a model that had never seen the game.
- **No project fine-tuning.** The weights were used exactly as published; the only
  behavioural addition is PR #74's browser-side brake, which replaces a weak LEFT/RIGHT
  reversal with one NOOP when the new direction's advantage is below 0.08. The recorded
  scores include that brake. The Python evaluator remains greedy and was not run here.
- **One model revision, one machine, one display.** Revision `f2fe3c12` on an RTX 4060
  Laptop at 169.5 Hz. Latency and frame pacing on another GPU, display, or CPU would
  differ, and this record says nothing about them.
- **An interactive session is not a contract evaluation.** The page runs the v2 contract's
  frame skip, sticky actions, FIRE ownership, and seed 101, and the episode ends when lives
  reach zero. It emits no per-episode CSV, no `stop_reason`, and no clear-detection record,
  so it cannot be counted as a canonical clear detection or compared with evaluated DQN
  and controller episodes.
- **Two sessions are a repeat, not a distribution.** The environment is seeded, so the
  agreement between runs shows determinism, not variance across seeds or model revisions.
- **Score is raw ALE reward.** 4 points is four bricks. It is not the upstream normalized
  score of 0.20, which uses a different protocol.
- **The frame-rate figures describe the AI panel.** They are the presented-frame cadence of
  that panel, not a rendering benchmark of the whole page and not a claim about input
  latency.

## Evidence index

| File | Contents |
| --- | --- |
| [`laya-browser-qa-run1.json`](evidence/laya-vision/laya-browser-qa-run1.json) | Session 1: environment, health payload, 22 per-second samples, 21 per-decision timing rows, 22 GPU samples, switching and pause snapshots |
| [`laya-browser-qa-run2.json`](evidence/laya-vision/laya-browser-qa-run2.json) | Session 2, identical schema |
| [`browser-qa.mjs`](evidence/laya-vision/browser-qa.mjs) | Producer harness: launches Chrome, drives the page, samples diagnostics and `nvidia-smi`, writes one JSON record |
| [`service-environment.txt`](evidence/laya-vision/service-environment.txt) | Host Python, PyTorch, CUDA, GPU, driver, and CPU-count record |

Machine-readable entries for the figures above are in [`summary.json`](summary.json)
(metric ids prefixed `laya_browser_`) and the expected-path record is in
[`artifact-manifest.json`](artifact-manifest.json).

## Related records

- PR #74, the branch under measurement: <https://github.com/Tommyweige/breakout-rl-engineering/pull/74>
- Issues #69-#72, the Laya research line: <https://github.com/Tommyweige/breakout-rl-engineering/issues/69>
- Closeout #73: <https://github.com/Tommyweige/breakout-rl-engineering/issues/73>
