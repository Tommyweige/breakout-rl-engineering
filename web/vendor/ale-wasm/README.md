# Pinned ALE WebAssembly runtime

This package is built from Farama ALE 0.12.0 commit
`f96026b362956d89076ac57d73f2ce82a59881ca` with the checked-in patch at
`web/patches/ale-wasm-paddle-strength.patch`. The patch keeps `act(action)`
unchanged, adds `actWithPaddleStrength(action, strength)`, and adds
`setBreakoutPaddlePosition(normalizedX)` for Human Mouse's absolute target. It
sets the pinned Breakout ROM's paddle-position state so the paddle reaches the
cursor target on the next raw frame. Human input reapplies the target on each
frame; the game clock and ball continue normally. At screen edges, the visible
paddle center is clamped to its physical travel range. This ROM-specific API is
used only by Human mouse control; the Agent and training/evaluation paths remain
on the discrete `act(action)` API. The source patch also skips IPO/LTO for the
WASM target and removes the WASM link-time LTO flag; the first LTO build
produced function-signature mismatch traps when loading Breakout. Integration
coverage checks both the analog strength API and absolute paddle positioning
after one raw frame.
`index.mjs` loads the generated UMD script in browsers; `node.mjs` uses Node's
CommonJS interop for tests and tooling.

The checked-in build uses Emscripten 3.1.68 and the vcpkg baseline pinned by
ALE's `vcpkg.json` (`3426a5e955f029ab0088ef6656fda06552bd2699`). Rebuild the
runtime from the repository root with `bash web/scripts/build-ale-wasm.sh`;
see `BUILD.md` for toolchain setup and provenance details.

`roms/breakout.bin` is the 2,048-byte Breakout ROM extracted from the official
`@farama/ale-wasm@0.12.0` package. Its SHA-256 is
`376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5`. The
upstream package tarball integrity is recorded in the build documentation.

The ALE license is included as `LICENSE`.
