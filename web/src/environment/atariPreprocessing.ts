export const ATARI_SCREEN_WIDTH = 160;
export const ATARI_SCREEN_HEIGHT = 210;
export const MODEL_SCREEN_SIZE = 84;

const RED_LUMA = 0.299;
const GREEN_LUMA = 0.587;
const BLUE_LUMA = 0.114;

export function rgbToGrayscale(rgb: ArrayLike<number>): Uint8Array {
  if (rgb.length % 3 !== 0) throw new Error(`RGB frame length must be divisible by 3, got ${rgb.length}`);
  const grayscale = new Uint8Array(rgb.length / 3);
  for (let index = 0, pixel = 0; index < rgb.length; index += 3, pixel += 1) {
    grayscale[pixel] = clampByte(
      Math.round(
        RED_LUMA * (rgb[index] ?? 0) +
          GREEN_LUMA * (rgb[index + 1] ?? 0) +
          BLUE_LUMA * (rgb[index + 2] ?? 0),
      ),
    );
  }
  return grayscale;
}

export function maxPoolRgbFrames(first: ArrayLike<number>, second: ArrayLike<number>): Uint8Array {
  if (first.length !== second.length) throw new Error('max-pool frames must have equal lengths');
  const pooled = new Uint8Array(first.length);
  for (let index = 0; index < first.length; index += 1) {
    pooled[index] = Math.max(first[index] ?? 0, second[index] ?? 0);
  }
  return pooled;
}

export function maxPoolGrayscaleFrames(first: ArrayLike<number>, second: ArrayLike<number>): Uint8Array {
  if (first.length !== second.length) throw new Error('max-pool grayscale frames must have equal lengths');
  const pooled = new Uint8Array(first.length);
  for (let index = 0; index < first.length; index += 1) {
    pooled[index] = Math.max(first[index] ?? 0, second[index] ?? 0);
  }
  return pooled;
}

/** Resize a single-channel image with area coverage, matching AtariPreprocessing's intent. */
export function resizeGrayscale(
  source: ArrayLike<number>,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Uint8Array {
  if (!Number.isInteger(sourceWidth) || sourceWidth < 1 || !Number.isInteger(sourceHeight) || sourceHeight < 1) {
    throw new Error('source image dimensions must be positive integers');
  }
  if (!Number.isInteger(targetWidth) || targetWidth < 1 || !Number.isInteger(targetHeight) || targetHeight < 1) {
    throw new Error('target image dimensions must be positive integers');
  }
  if (source.length !== sourceWidth * sourceHeight) {
    throw new Error(`grayscale image length must be ${sourceWidth * sourceHeight}, got ${source.length}`);
  }

  const output = new Uint8Array(targetWidth * targetHeight);
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;

  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    const sourceYStart = targetY * scaleY;
    const sourceYEnd = (targetY + 1) * scaleY;
    const firstSourceY = Math.floor(sourceYStart);
    const lastSourceY = Math.min(sourceHeight - 1, Math.ceil(sourceYEnd) - 1);
    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const sourceXStart = targetX * scaleX;
      const sourceXEnd = (targetX + 1) * scaleX;
      const firstSourceX = Math.floor(sourceXStart);
      const lastSourceX = Math.min(sourceWidth - 1, Math.ceil(sourceXEnd) - 1);
      let weightedSum = 0;
      let weight = 0;
      for (let sourceY = firstSourceY; sourceY <= lastSourceY; sourceY += 1) {
        const yWeight = Math.min(sourceYEnd, sourceY + 1) - Math.max(sourceYStart, sourceY);
        if (yWeight <= 0) continue;
        for (let sourceX = firstSourceX; sourceX <= lastSourceX; sourceX += 1) {
          const xWeight = Math.min(sourceXEnd, sourceX + 1) - Math.max(sourceXStart, sourceX);
          if (xWeight <= 0) continue;
          const cellWeight = xWeight * yWeight;
          weightedSum += (source[sourceY * sourceWidth + sourceX] ?? 0) * cellWeight;
          weight += cellWeight;
        }
      }
      output[targetY * targetWidth + targetX] = clampByte(weight > 0 ? roundToNearestEven(weightedSum / weight) : 0);
    }
  }
  return output;
}

export function preprocessRgbFrame(
  rgb: ArrayLike<number>,
  sourceWidth = ATARI_SCREEN_WIDTH,
  sourceHeight = ATARI_SCREEN_HEIGHT,
  targetSize = MODEL_SCREEN_SIZE,
): Uint8Array {
  if (rgb.length !== sourceWidth * sourceHeight * 3) {
    throw new Error(`RGB frame must be ${sourceWidth}x${sourceHeight}, got ${rgb.length} values`);
  }
  return preprocessGrayscaleFrame(rgbToGrayscale(rgb), sourceWidth, sourceHeight, targetSize);
}

export function preprocessGrayscaleFrame(
  grayscale: ArrayLike<number>,
  sourceWidth = ATARI_SCREEN_WIDTH,
  sourceHeight = ATARI_SCREEN_HEIGHT,
  targetSize = MODEL_SCREEN_SIZE,
): Uint8Array {
  return resizeGrayscale(grayscale, sourceWidth, sourceHeight, targetSize, targetSize);
}

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, value));
}

/** OpenCV's uchar conversion uses the platform's nearest-even tie rule. */
function roundToNearestEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  // The overlap arithmetic can leave an exact .5 tie as 0.5000000000000001.
  // Treat values within a tiny tolerance as the same tie OpenCV sees.
  if (Math.abs(fraction - 0.5) <= 1e-12) return lower % 2 === 0 ? lower : lower + 1;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower + 1;
}
