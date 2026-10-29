# GitHub Actions CI

The repository previously had no `.github/` directory. The workflow is
[`../../.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

## Jobs

### `python`: unit tests and smoke

Runs on `ubuntu-latest` with Python 3.12 and the verified CPU dependency set:

```text
torch==2.13.0+cpu          (from https://download.pytorch.org/whl/cpu)
ale-py==0.12.0
gymnasium==1.3.0
onnx==1.19.0
onnxruntime==1.31.1
numpy==2.5.4
matplotlib==3.11.2
pillow==13.0.0
opencv-python==5.0.0.93
```

Steps:

1. install the OpenCV X/GL runtime libraries (`libxcb1`, `libgl1`,
   `libglib2.0-0`, `libxext6`, `libsm6`, `libice6`);
2. verify that `ale_py/roms/breakout.bin` exists and is 2,048 bytes;
3. `python -m scripts.smoke.run_smoke --units-only`;
4. `python -m scripts.smoke.run_smoke --e2e-only --output-dir outputs/smoke`;
5. upload `outputs/smoke` as an artifact (even on failure).

CI deliberately does not use `environment.yml`: that file installs CUDA 13
wheels and produces a 6.3 GB environment, and hosted runners have no GPU. The
CPU wheel set above is the one used for the verified clean-clone run.

### `web`: production build

Runs on `ubuntu-latest` with Node.js 22:

```text
npm ci
npm run build
```

`npm run build` runs `prepare:assets`, `tsc --noEmit`, and `vite build`.

## Triggers

```yaml
on:
  push:
  pull_request:
  workflow_dispatch:
```

Every branch push and pull request runs both jobs. `workflow_dispatch` lets a
maintainer start a run manually from the Actions tab without a new commit.
Tags such as `repro-*` also run because `push` is unfiltered; see
[`versioning.md`](versioning.md).

## Local equivalent

Run exactly what CI runs, in the same order:

```bash
python -m scripts.smoke.run_smoke --units-only
python -m scripts.smoke.run_smoke --e2e-only --output-dir outputs/smoke

cd web
npm ci
npm run build
```

## Validation performed here

- `actionlint 1.7.12` (linux/arm64) reports no errors:

  ```text
  $ actionlint .github/workflows/ci.yml
  (no output; exit 0)
  ```

- The YAML parses to the expected jobs and steps (`python`, `web`).
- The exact `python` job commands were executed in a fresh clone with a fresh
  venv on 2026-10-29; both smoke stages passed (see [`smoke.md`](smoke.md)).
- The exact `web` job commands were executed in the same clone; `npm ci` and
  `npm run build` passed (see [`setup.md`](setup.md)).

## Why this pass did not trigger a real GitHub Actions run

GitHub Actions only executes workflows from a commit that exists on GitHub.
This work was done in a local checkout and no push was authorized, so the
workflow has not run on GitHub yet. Once the branch is pushed (or a pull
request is opened), the workflow runs automatically. If Actions is disabled
for the repository, enable it under **Settings → Actions → General**.

To trigger it directly after pushing:

```bash
git push origin repro-path
```

or use **Actions → CI → Run workflow** (`workflow_dispatch`). The first real
run is the remaining verification step for CI; nothing in the workflow depends
on local state.

## Expected runtime

On the verified 4-vCPU host the smoke stages take about 2 minutes total. On a
2-vCPU GitHub runner, expect roughly 3–5 minutes including dependency
installation. The job timeout is 25 minutes, which is ample. The web job is
under a minute after `npm ci`.
