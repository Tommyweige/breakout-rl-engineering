# Rebuilding the analog ALE WebAssembly package

The browser needs a Human-only entry point that calls ALE's C++
`ALEInterface::act(Action, float)`. The released `@farama/ale-wasm@0.12.0`
package and the corresponding upstream WASM wrapper expose only `act(action)`.
This repo keeps a source patch and the generated runtime package under
`web/vendor/ale-wasm`; `npm ci` installs that local package, and
`prepare-web-assets` copies its WASM and preloaded ROM data into the browser
assets. The local package supplies a browser ESM adapter for the generated UMD
loader and a Node entry for CommonJS interop in tests. No installed `node_modules`
file is edited.

## Pinned inputs

- ALE source: `f96026b362956d89076ac57d73f2ce82a59881ca` (ALE 0.12.0).
- Wrapper and TypeScript change: `web/patches/ale-wasm-paddle-strength.patch`.
- The same patch disables IPO/LTO only for this WASM build; Emscripten's LTO
  output trapped with function-signature mismatches during a real Breakout
  load. The checked-in non-LTO package is covered by an actual ALE load and
  one-frame action smoke test.
- Emscripten: 3.1.68, matching the upstream WASM workflow.
- vcpkg: `3426a5e955f029ab0088ef6656fda06552bd2699`, the baseline in the ALE
  0.12.0 `vcpkg.json` manifest.
- Breakout ROM: 2,048 bytes from the official npm package. The source file at
  `web/vendor/ale-wasm/roms/breakout.bin` has SHA-256
  `376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5`.
  The source package's npm tarball integrity is
  `sha512-8pXhwhGu8ROn/Z2zP6VYaF4nXgmZvmLYMhed8mFOGmgvXx+h5+kWd/M3j0s4+WRK8r5F3eoSsdSx0p/fUZjU7A==`.

## Build

Use Linux, macOS, or WSL2 with Git, CMake, a C++ build toolchain, and the
Emscripten SDK. Install and activate Emscripten 3.1.68, and clone vcpkg at the
pinned baseline:

```bash
git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
~/emsdk/emsdk install 3.1.68
~/emsdk/emsdk activate 3.1.68
export EMSDK=~/emsdk
source "$EMSDK/emsdk_env.sh"

git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
git -C ~/vcpkg checkout --detach 3426a5e955f029ab0088ef6656fda06552bd2699
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_ROOT=~/vcpkg

bash web/scripts/build-ale-wasm.sh
npm ci --prefix web
npm run prepare:assets --prefix web
```

The build script verifies both toolchain revisions and the ROM hash before
building, then replaces `ale.js`, `ale.wasm`, and `ale.data` in the local
package. Commit those generated runtime files with the source patch and package
metadata. `act(action)` remains the upstream discrete API; Human Mouse uses
the added `actWithPaddleStrength(action, strength)` method.
