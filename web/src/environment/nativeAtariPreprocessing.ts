/**
 * The parts of Gymnasium AtariPreprocessing that are not represented in the
 * repository's Contract v2 JSON. Keeping this adapter explicit prevents the
 * browser environment from silently drifting from the native wrapper.
 */
export const NATIVE_ATARI_PREPROCESSING = {
  noopMax: 30,
  frameSkip: 4,
  screenSize: 84,
  terminalOnLifeLoss: false,
  grayscaleObs: true,
  grayscaleNewaxis: false,
  scaleObs: false,
  maxPoolSpace: 'grayscale',
  resizeInterpolation: 'cv2.INTER_AREA',
} as const;

export interface NativeNoopResetManifest {
  schema_version: 2;
  artifact_type: 'day29_native_noop_reset_manifest';
  source: string;
  noop_max: 30;
  seeds: Record<string, {
    noop_count: number;
    np_seed: number;
    ale_seed: number;
  }>;
}

export function fallbackNoopCount(seed: number, noopMax = NATIVE_ATARI_PREPROCESSING.noopMax): number {
  // The manifest provides exact native draws for the fixed evaluation seeds.
  // This fallback is intentionally deterministic and is not claimed to be a
  // NumPy SeedSequence/PCG64 implementation for arbitrary interactive seeds.
  let state = (seed ^ 0x9e3779b9) >>> 0;
  state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
  state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
  state ^= state >>> 16;
  return 1 + (state >>> 0) % noopMax;
}
