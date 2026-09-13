import { ATARI_SCREEN_HEIGHT, ATARI_SCREEN_WIDTH } from '../environment/atariPreprocessing';

const ROI_TOP_OFFSET = 20;
const ROI_BOTTOM_OFFSET = 3;
const X_MARGIN = 10;
const MIN_PADDLE_WIDTH = 6;
const MIN_COLUMN_HITS = 2;

/**
 * Finds the red paddle in the fixed bottom Atari ROI without touching emulator state.
 * The detector returns null when the frame does not contain enough evidence; callers
 * should retain the last trusted center in that case.
 */
export function detectPaddleCenterNormalized(
  rawRgb: ArrayLike<number>,
  width = ATARI_SCREEN_WIDTH,
  height = ATARI_SCREEN_HEIGHT,
  columnHits = new Uint8Array(width),
): number | null {
  if (rawRgb.length < width * height * 3) return null;
  if (columnHits.length < width) return null;
  const rowStart = Math.max(0, height - ROI_TOP_OFFSET);
  const rowEnd = Math.min(height - 1, height - ROI_BOTTOM_OFFSET);
  columnHits.fill(0, 0, width);
  for (let y = rowStart; y <= rowEnd; y += 1) {
    for (let x = X_MARGIN; x < width - X_MARGIN; x += 1) {
      const offset = (y * width + x) * 3;
      const red = rawRgb[offset] ?? 0;
      const green = rawRgb[offset + 1] ?? 0;
      const blue = rawRgb[offset + 2] ?? 0;
      if (isPaddlePixel(red, green, blue)) columnHits[x] = Math.min(255, (columnHits[x] ?? 0) + 1);
    }
  }

  let bestStart = -1;
  let bestEnd = -1;
  let runStart = -1;
  for (let x = X_MARGIN; x <= width - X_MARGIN; x += 1) {
    const hasEvidence = (columnHits[x] ?? 0) >= MIN_COLUMN_HITS;
    if (hasEvidence && runStart < 0) runStart = x;
    if ((!hasEvidence || x === width - X_MARGIN) && runStart >= 0) {
      const end = hasEvidence && x === width - X_MARGIN ? x : x - 1;
      if (end - runStart + 1 > bestEnd - bestStart + 1) {
        bestStart = runStart;
        bestEnd = end;
      }
      runStart = -1;
    }
  }

  if (bestStart < 0 || bestEnd - bestStart + 1 < MIN_PADDLE_WIDTH) return null;
  return ((bestStart + bestEnd + 1) / 2) / width;
}

function isPaddlePixel(red: number, green: number, blue: number): boolean {
  return red >= 100 && red - green >= 35 && red - blue >= 35;
}
