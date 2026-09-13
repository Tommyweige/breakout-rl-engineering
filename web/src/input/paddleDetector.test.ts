import { describe, expect, it } from 'vitest';

import { detectPaddleCenterNormalized } from './paddleDetector';

function frameWithPaddle(start: number, end: number): Uint8Array {
  const frame = new Uint8Array(160 * 210 * 3);
  for (let y = 196; y <= 203; y += 1) {
    for (let x = start; x <= end; x += 1) {
      const offset = (y * 160 + x) * 3;
      frame[offset] = 204;
      frame[offset + 1] = 76;
      frame[offset + 2] = 76;
    }
  }
  return frame;
}

describe('paddle detector', () => {
  it('detects a horizontal red paddle in the bottom ROI', () => {
    expect(detectPaddleCenterNormalized(frameWithPaddle(60, 79))).toBeCloseTo(0.4375, 4);
  });

  it('ignores small red objects and edge walls', () => {
    const frame = frameWithPaddle(1, 8);
    const ballOffset = (190 * 160 + 80) * 3;
    frame[ballOffset] = 204;
    frame[ballOffset + 1] = 76;
    frame[ballOffset + 2] = 76;
    expect(detectPaddleCenterNormalized(frame)).toBeNull();
  });

  it('fails closed for incomplete frames', () => {
    expect(detectPaddleCenterNormalized(new Uint8Array(12))).toBeNull();
  });
});
