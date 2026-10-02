# Issue #39 RGB Brick Removal Observability

**Classification: REJECTED** under the frozen Issue #39 criteria.

## Result

The primary metric was one-to-one brick-removal event F1, matched within ±2 native frames and computed within each episode before aggregating counts.

| Seed | Frames | Score-positive labels | Predicted events | TP | FP | FN | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 707 | 5,000 | 36 | 0 | 0 | 0 | 36 | 0.0000 |
| 808 | 5,000 | 33 | 3 | 2 | 1 | 31 | 0.1111 |
| 909 | 5,000 | 39 | 14 | 7 | 7 | 32 | 0.2642 |
| **Total** | **15,000** | **108** | **17** | **9** | **8** | **99** | **0.1440** |

The label sample floor (12 events over at least 2 episodes) was met, and raw-reward transitions agreed with decoded RAM scoreboard deltas for the full label set. The resulting F1 of 0.1440 meets the frozen rejection criterion of F1 ≤ 0.50. All three episodes reached their 5,000-frame caps without a canonical clear.

## Frozen run and budget

- Seeds, in frozen order: 707, 808, 909; one episode per seed.
- Formal budget: 5,000 native frames per episode, 15,000 total.
- Prior detector/controller validation and ALE smoke are conservatively charged as at most 42 native frames, within the 350-frame reserve; at least 308 reserve frames remain. Combined ceiling: 15,350.
- Wall accounting: 20.00 seconds charged to setup/validation/smoke, 63.78 seconds recorded by the runner including finalization reserve, 83.78 seconds total against the 600-second cap.
- Runtime: `ALE/Breakout-v5`, Contract v3, frame skip 1, sticky-action probability 0.25, ROM SHA-256 `376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5`.
- Runtime versions: ale-py 0.12.0, Gymnasium 1.3.0, NumPy 2.3.5, Python 3.12.14.
- Model request: GPT-6 Luna / High. `MODEL_ROUTING_VERIFICATION: UNAVAILABLE`; no explicit routing, quota, or runtime failure occurred.

Exact command:

```text
python -m scripts.evaluation.run_issue39_rgb_brick_observability --config configs/eval/issue39_rgb_brick_observability_v1.json --output-dir outputs/issue-39-rgb-brick-observability --pre-run-wall-seconds 20.00 --wall-budget-seconds 560.00
```

## Provenance and artifacts

- Run source commit: `219e8a448a64ec2381ffce4746a0fa10d109a6cd` on `codex/issue-39-rgb-brick-observability`.
- Probe source SHA-256: `a366649344d14dcd74a2d0555cba79f845f57e8fa75748e7801f89ce8f1adb98`.
- Config SHA-256: `5e1ab1c8a47027f04cdedfa5a6df56128d169bd5e9d6f10e095901b8393db25b`.
- Contract v3 SHA-256: `d97b7fcb5758cba495857b3972a119ed3eb7db74fb0145d151e9d82725c5a786`.
- Controller config SHA-256: `25ddff6756747a6539ddd6a1eb570e7ef4c274f3e700ae011e68c87cae37c724`.
- Lossless RGB crop artifact: `research/issue-39-rgb-brick-observability-artifacts/rgb_brick_crops.npz`, 808,068 bytes, SHA-256 `52b2162544cc8b7b71510b4a001b9fa680c6c9c79646eebe8c5c2bbf3b56bb41`. Verified array shape `(15003, 48, 144, 3)`, dtype `uint8`; this stores the initial post-serve crop and every post-action crop for all three episodes.
- Offline labels, detector outputs, full action/reward traces, and result JSON are in [`results.json`](issue-39-rgb-brick-observability-artifacts/results.json).
- Generated run report: [`report.md`](issue-39-rgb-brick-observability-artifacts/report.md).

The existing canonical completion evaluator read reward/RAM for the frozen clear-stop rule. The detector consumed only captured RGB crops; offline label rows were joined after all action selection. Neither score nor RAM entered controller input, detector input, or action choice.

No clear-probability or controller-benefit claim is made.
