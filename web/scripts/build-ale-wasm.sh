#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
web_root="$(cd "$script_dir/.." && pwd)"
package_dir="$web_root/vendor/ale-wasm"
upstream_commit="f96026b362956d89076ac57d73f2ce82a59881ca"
vcpkg_baseline="3426a5e955f029ab0088ef6656fda06552bd2699"
rom_sha256="376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5"

if [[ -z "${EMSDK:-}" || ! -f "$EMSDK/emsdk_env.sh" ]]; then
  echo "Set EMSDK to an installed Emscripten 3.1.68 SDK." >&2
  exit 1
fi
if [[ -z "${VCPKG_ROOT:-}" || ! -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]]; then
  echo "Set VCPKG_ROOT to a vcpkg checkout at $vcpkg_baseline." >&2
  exit 1
fi

source "$EMSDK/emsdk_env.sh"
if ! emcc --version | grep -Fq "3.1.68"; then
  echo "Expected Emscripten 3.1.68." >&2
  exit 1
fi
if [[ "$(git -C "$VCPKG_ROOT" rev-parse HEAD)" != "$vcpkg_baseline" ]]; then
  echo "Expected vcpkg baseline $vcpkg_baseline." >&2
  exit 1
fi

build_root="${ALE_WASM_BUILD_ROOT:-$(mktemp -d)}"
source_root="$build_root/Arcade-Learning-Environment"
build_dir="$build_root/build-wasm"
mkdir -p "$build_root"
git clone --filter=blob:none https://github.com/Farama-Foundation/Arcade-Learning-Environment.git "$source_root"
git -C "$source_root" checkout --detach "$upstream_commit"
git -C "$source_root" apply "$web_root/patches/ale-wasm-paddle-strength.patch"

if command -v sha256sum >/dev/null 2>&1; then
  actual_rom_sha256="$(sha256sum "$package_dir/roms/breakout.bin" | cut -d ' ' -f 1)"
elif command -v shasum >/dev/null 2>&1; then
  actual_rom_sha256="$(shasum -a 256 "$package_dir/roms/breakout.bin" | cut -d ' ' -f 1)"
else
  echo "Install sha256sum or shasum to verify the Breakout ROM." >&2
  exit 1
fi
if [[ "$actual_rom_sha256" != "$rom_sha256" ]]; then
  echo "Breakout ROM SHA-256 mismatch: $actual_rom_sha256" >&2
  exit 1
fi
mkdir -p "$source_root/src/ale/python/roms"
cp "$package_dir/roms/breakout.bin" "$source_root/src/ale/python/roms/breakout.bin"

emcmake cmake -S "$source_root" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_WASM_LIB=ON \
  -DWASM_PRELOAD_ROMS=ON \
  -DSDL_SUPPORT=ON \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=wasm32-emscripten \
  -DVCPKG_OVERLAY_TRIPLETS="$source_root/cmake/custom-triplets" \
  -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE="$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"
cmake --build "$build_dir" --target ale-wasm --parallel 2

wasm_output="$build_dir/src/ale/wasm"
for artifact in ale.js ale.wasm ale.data; do
  if [[ ! -f "$wasm_output/$artifact" ]]; then
    echo "Missing expected ALE artifact: $wasm_output/$artifact" >&2
    exit 1
  fi
  cp "$wasm_output/$artifact" "$package_dir/$artifact"
done
cp "$source_root/packages/wasm/ale.d.ts" "$package_dir/ale.d.ts"
cp "$source_root/LICENSE.md" "$package_dir/LICENSE"

echo "Built @farama/ale-wasm 0.12.0-paddle.1 from $upstream_commit"
