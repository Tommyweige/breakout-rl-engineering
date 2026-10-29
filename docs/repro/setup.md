# Clean-clone setup

These are the commands that were actually run on 2026-10-29 on Oracle Linux
9.8 (aarch64, no GPU). Outputs are copied from the terminal, not reconstructed.

## Prerequisites

```text
git
Python 3.12 (the venv path below needs the venv module)
Node.js >= 20.19 and npm (for the browser application)
~1.2 GB disk for the CPU venv; ~6.3 GB for the conda environment
```

On this host, `python3.12`, `pip`, and `venv` were already present. On a
minimal Debian/Ubuntu image install the interpreter, the venv module, and the
X/GL runtime libraries that `opencv-python` imports:

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv git \
  libxcb1 libgl1 libglib2.0-0 libxext6 libsm6 libice6
```

No C/C++ compiler, SDL, X11 display server, CUDA toolkit, or display is needed
for the CPU path. ALE runs headless. The X/GL libraries are only runtime
dependencies of the OpenCV wheel; they do not require a running X server. The
conda path installs CUDA runtime wheels, but still needs an NVIDIA driver to
make `torch.cuda.is_available()` true.

### Why the OpenCV libraries are required (verified)

A minimal `ubuntu:24.04` container without those libraries fails during
`make_breakout_env()`:

```text
ImportError: libxcb.so.1: cannot open shared object file: No such file or directory

The above exception was the direct cause of the following exception:
  File ".../gymnasium/wrappers/atari_preprocessing.py", line 96, in __init__
    import cv2  # noqa: F401
```

In that container 54 tests failed and the smoke stopped. After installing
`libxcb1 libgl1 libglib2.0-0 libxext6 libsm6 libice6`, the same container ran
the full smoke successfully:

```text
Ran 315 tests in 15.615s
FAILED (errors=13, skipped=8)
...
Unit tests: PASSED (294 passed, 8 skipped, 13 failed in 15.6s)
Training:   PASS in 19.0s (400 agent steps, 1597 emulator frames)
Evaluation: PASS in 20.8s (15 episodes, mean return 0.733, 8421 emulator frames)
Smoke: PASS in 56.9s
```

The 13 failures are the known ones documented in
[`known-test-failures.md`](known-test-failures.md), not OpenCV failures.

A conda environment may already satisfy some of these through conda-forge
(`tk` pulls X libraries), but do not rely on it on a minimal image. If OpenCV
fails to import, install the same libraries.

## Path A: venv, CPU only (verified)

```bash
git clone https://github.com/Tommyweige/breakout-rl-engineering.git
cd breakout-rl-engineering

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> The verification run on 2026-10-29 cloned `--branch repro-path` because this
> smoke path was still under review. Once it is on `main`, the plain clone
> above is the correct command.

Output:

```text
Successfully installed pip-26.2.2
```

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
```

Output:

```text
Successfully installed MarkupSafe-3.0.3 filelock-4.2.4 fsspec-2026.9.0
jinja2-3.1.6 mpmath-1.3.0 networkx-3.7 setuptools-78.1.0 sympy-1.14.0
torch-2.13.0+cpu typing-extensions-4.16.0
```

```bash
python -m pip install \
  "ale-py==0.12.0" "gymnasium==1.3.0" "onnx==1.19.0" "onnxruntime==1.31.1" \
  "numpy==2.5.4" "matplotlib==3.11.2" "pillow==13.0.0" "opencv-python==5.0.0.93"
```

Output:

```text
Successfully installed ale-py-0.12.0 cloudpickle-3.1.2 contourpy-1.4.0
cycler-0.12.1 farama-notifications-0.0.6 flatbuffers-25.12.19 fonttools-4.68.0
gymnasium-1.3.0 kiwisolver-1.5.1 matplotlib-3.11.2 ml_dtypes-0.6.0
numpy-2.5.4 onnx-1.19.0 onnxruntime-1.31.1 opencv-python-5.0.0.93
packaging-26.3 pillow-13.0.0 protobuf-7.36.2 pyparsing-3.3.3
python-dateutil-3.0.0 six-1.17.0 tzdata-2026.6
```

The complete venv is 1.2 GB. On Windows use `py -3.12 -m venv .venv` and
`.venv\Scripts\activate`; the pinned wheel names are the same.

### Why `--index-url .../whl/cpu`

The default PyTorch wheel carries CUDA libraries and is roughly 10x larger.
On a CPU-only host, `pip install torch==2.13.0` pulls the CUDA build and can
confuse the GPU checks in the tests. The pinned `+cpu` build keeps the smoke
path small and makes `torch.cuda.is_available() == False` explicit.

## Path B: conda (verified with micromamba)

`environment.yml` is a standard conda environment file. The local conda
client is not installed on the verified host, so the file was executed with
micromamba 2.10.0, which uses the same solver metadata:

```bash
export MAMBA_ROOT_PREFIX=/tmp/mamba-root
micromamba create -y -f environment.yml
```

On a normal Miniconda installation the equivalent is:

```bash
conda env create -f environment.yml
conda activate breakout-rl-engineering
```

Actual result:

```text
Successfully installed ... torch-2.13.0+cu130 ... onnxruntime-gpu-1.29.0 ...
ELAPSED 167.33 s
```

The environment is 6.3 GB because PyTorch's `+cu130` wheel installs the CUDA
13 runtime libraries (`nvidia-*`, `triton`, `cuda-toolkit`). It works without a
GPU, but the CUDA runtime is unused:

```text
$ micromamba run -n breakout-rl-engineering python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
2.13.0+cu130 False
```

The full smoke also passed in this conda environment on the same host:

```text
Python 3.12.16, torch 2.13.0+cu130 (CPU execution)
Unit tests: PASSED (294 passed, 8 skipped, 13 failed in 15.6s)
Training:   PASS in 15.5s (400 agent steps, 1597 emulator frames)
Evaluation: PASS in 21.0s (15 episodes, mean return 0.733, 8421 emulator frames)
Smoke: PASS in 58.3s
```

To actually use the GPU, install an NVIDIA driver compatible with CUDA 13 on
the host. The conda file does not install a driver.

### Do not use `environment.lock.yml` on Linux

`environment.lock.yml` is a Windows export. It contains `prefix:
C:\Users\tommy\anaconda3\envs\breakout-rl-engineering` and Windows-only
packages (`vc`, `vc14_runtime`, `ucrt`, `mkl`, `libwinpthread`). Creating it on
Linux fails during solving:

```text
$ micromamba create -y -f environment.lock.yml
error    libmamba Could not solve for environment specs
    The following packages are incompatible
    ├─ intel-openmp =2025.0.0 * does not exist (perhaps a typo or a missing channel);
    ├─ libintl =0.22.5 * does not exist (perhaps a typo or a missing channel);
    ├─ libwinpthread =12.0.0.r4.gg4f2fc60ca * does not exist (perhaps a typo or a missing channel);
    ├─ mkl-service =2.7.2 * does not exist (perhaps a typo or a missing channel);
    ├─ mkl =2025.0.0 * does not exist (perhaps a typo or a missing channel);
    ├─ mkl_fft =2.2.0 * does not exist (perhaps a typo or a missing channel);
    ├─ mkl_random =1.4.1 * does not exist (perhaps a typo or a missing channel);
    ├─ ucrt =10.0.26100.0 * does not exist (perhaps a typo or a missing channel);
    ├─ vc14_runtime =14.51.36247 * does not exist (perhaps a typo or a missing channel);
    ├─ vc =14.5 * does not exist (perhaps a typo or a missing channel);
    ├─ vcomp14 =14.51.36247 * does not exist (perhaps a typo or a missing channel);
    └─ vs2015_runtime =14.51.36247 * does not exist (perhaps a typo or a missing channel).
critical libmamba Could not solve for environment specs
```

Use `environment.yml`, not the lock file, on Linux and macOS. The lock file is
only meaningful on the Windows machine that produced it.

## ALE ROMs

`ale-py==0.12.0` bundles the ROMs in the wheel. No `autorom`, no
`ale-py[accept-rom-license]`, and no manual `roms/` download is needed:

```bash
python - <<'PY'
from pathlib import Path
import ale_py

rom = Path(ale_py.__file__).resolve().parent / "roms" / "breakout.bin"
print(rom)
print("exists:", rom.is_file(), "bytes:", rom.stat().st_size)
PY
```

Actual output:

```text
/.../.venv/lib/python3.12/site-packages/ale_py/roms/breakout.bin
exists: True bytes: 2048
```

The wheel contains 111 ROMs. The browser application does not read this
directory: its Breakout ROM is compiled into `web/vendor/ale-wasm/ale.data`.
`web/vendor/ale-wasm/BUILD.md` pins that ROM's SHA-256
(`376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5`).

## Web build (verified)

```bash
node --version   # v22.23.3
npm --version    # 10.9.9
cd web
npm ci
npm run build
```

Actual output:

```text
added 136 packages, and audited 138 packages in 12s
...
> breakout-rl-browser@0.0.0 prepare:assets
> node scripts/prepare-web-assets.mjs
> breakout-rl-browser@0.0.0 build
> npm run typecheck && vite build
vite v8.2.2 building client environment for production...
✓ 44 modules transformed.
✓ built in 3.19s
```

Use `npm ci`, not `npm install`, when you want the committed lockfile to be
authoritative. `npm run build` runs `prepare:assets` first, which copies:

- `web/node_modules/@farama/ale-wasm/{ale.js,ale.wasm,ale.data}` to `web/public/ale/`
- `configs/eval/breakout_contract_v2.json` to `web/public/configs/eval/`
- `docs/final-results/summary.json` to `web/public/final-results/`

The last three are generated and gitignored; a fresh clone must install
dependencies before building.

Dev server:

```bash
npm run dev
```

Output:

```text
VITE v8.2.2 ready in 254 ms
➜  Local:   http://localhost:5173/
```

Use `http://localhost:5173/`, not `127.0.0.1:5173`. On the verified host Vite
bound only the IPv6 localhost address, so a hard-coded `127.0.0.1` connection
was refused while `localhost` returned HTTP 200.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'scripts'` | running from outside the repository root | `cd` to the clone root; the project is run with `python -m ...`, not installed as a package |
| `ModuleNotFoundError: No module named 'ale_py'` | dependencies installed into a different interpreter | activate the venv/conda env, then `python -m pip install ...` |
| `ImportError: libxcb.so.1: cannot open shared object file` | OpenCV wheel's system libraries are missing | `sudo apt-get install -y libxcb1 libgl1 libglib2.0-0 libxext6 libsm6 libice6` |
| `FileNotFoundError: .../ale_py/roms/breakout.bin` | a non-0.12.0 `ale-py` build without ROMs | install `ale-py==0.12.0` |
| `torch.cuda.is_available()` is `False` with the conda env | no NVIDIA driver, or a driver older than CUDA 13 | install a compatible driver; the CPU smoke does not need it |
| `Could not solve for environment specs ... vc ...` | using `environment.lock.yml` on Linux | use `environment.yml` |
| `Connection refused` on `127.0.0.1:5173` | Vite bound IPv6 localhost | open `http://localhost:5173/` |
| `npm error Unsupported engine` | Node.js older than 20.19 | install Node.js 22 |

Next: [`smoke.md`](smoke.md).
