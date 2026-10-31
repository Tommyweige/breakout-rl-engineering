# Reproduce this repository

This directory is the entry point for going from a clean clone to a passing
smoke run. It records what was actually executed on 2026-10-29, including the
parts that are still not fixed.

The verified host for this pass was:

```text
Oracle Linux Server 9.8, aarch64, 4 vCPU, 22 GB RAM
Python 3.12.15
Node.js v22.23.3 / npm 10.9.9
No NVIDIA GPU (torch.cuda.is_available() == False)
```

## Shortest path

```bash
git clone https://github.com/Tommyweige/breakout-rl-engineering.git
cd breakout-rl-engineering

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
python -m pip install \
  "ale-py==0.12.0" "gymnasium==1.3.0" "onnx==1.19.0" "onnxruntime==1.31.1" \
  "numpy==2.5.4" "matplotlib==3.11.2" "pillow==13.0.0" "opencv-python==5.0.0.93"

python -m scripts.smoke.run_smoke --output-dir outputs/smoke
```

The last command takes about two minutes on the verified host. It runs 340
unit tests, trains for 400 agent steps, and evaluates the checkpoint on the 15
concrete seeds pinned by Breakout Contract v2. A passing run ends with:

```text
Unit tests: PASSED (335 passed, 5 skipped, 0 failed in 133.6s)
Training:   PASS in 38.2s (400 agent steps, 1597 emulator frames)
Evaluation: PASS in 44.6s (15 episodes, mean return 0.733, 8421 emulator frames)
Smoke: PASS in 140.7s
```

There are no known failures on `main`: the tests that imported scripts deleted
by the `main` cleanup are gone, the CRLF-derived contract hash is corrected, and
the ONNX policy tests skip when their untracked asset is absent
([`known-test-failures.md`](known-test-failures.md)). The allowlist in
[`../../scripts/smoke/known_failures.json`](../../scripts/smoke/known_failures.json)
is empty, so `--strict-units` and the default mode now agree.

## Documents

| Document | Contents |
| --- | --- |
| [`setup.md`](setup.md) | Clean-clone install, conda and venv paths, system dependencies, ALE ROM source, web build |
| [`smoke.md`](smoke.md) | What the smoke command runs, what it asserts, artifacts, timing |
| [`ci.md`](ci.md) | GitHub Actions workflow, local equivalent, how to trigger it |
| [`versioning.md`](versioning.md) | Tag scheme that maps reported numbers to a commit |
| [`known-test-failures.md`](known-test-failures.md) | The green unit suite, what the `main` cleanup removed, and the evaluation-config mismatch |

## What "pass" means here

The smoke does not hide failures:

- Every discovered unit test is executed. The smoke still prints and counts any
  failure, and matches it against an explicit allowlist, which is now empty: any
  failure at all fails the smoke.
- The end-to-end stage uses the real training and evaluation CLIs against the
  real ALE environment. It validates the checkpoint, the contract runtime
  binding, the episode count, the native emulator-frame count, and the
  evaluation summary.
- The web build runs `tsc --noEmit` and Vite's production build.

It does **not** reproduce the historical scores. The raw 2.5M-transition
checkpoint and the historical run artifacts are not in `main`; see
[`../final-results/README.md`](../final-results/README.md).
