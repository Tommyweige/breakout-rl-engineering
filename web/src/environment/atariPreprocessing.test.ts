import { describe, expect, it } from 'vitest';

import {
  maxPoolRgbFrames,
  preprocessGrayscaleFrame,
  maxPoolGrayscaleFrames,
  preprocessRgbFrame,
  resizeGrayscale,
  rgbToGrayscale,
} from './atariPreprocessing';

describe('browser Atari preprocessing', () => {
  it('converts RGB to uint8 luma and preserves the expected output size', () => {
    const grayscale = rgbToGrayscale(new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]));
    expect([...grayscale]).toEqual([76, 150, 29]);

    const processed = preprocessRgbFrame(new Uint8Array(160 * 210 * 3));
    expect(processed).toBeInstanceOf(Uint8Array);
    expect(processed.length).toBe(84 * 84);
    expect(processed.every((value) => value === 0)).toBe(true);
  });

  it('max-pools the final two raw RGB frames before resizing', () => {
    const first = new Uint8Array([0, 20, 200, 4, 50, 80]);
    const second = new Uint8Array([10, 10, 210, 2, 70, 60]);
    expect([...maxPoolRgbFrames(first, second)]).toEqual([10, 20, 210, 4, 70, 80]);
  });

  it('matches native AtariPreprocessing order: grayscale each sampled frame, then max-pool', () => {
    const red = new Uint8Array([255, 0, 0]);
    const blue = new Uint8Array([0, 0, 255]);
    const grayscaleFrames = [rgbToGrayscale(red), rgbToGrayscale(blue)];
    const pooled = maxPoolGrayscaleFrames(grayscaleFrames[0]!, grayscaleFrames[1]!);
    expect([...preprocessGrayscaleFrame(pooled, 1, 1, 1)]).toEqual([76]);
    expect([...preprocessRgbFrame(maxPoolRgbFrames(red, blue), 1, 1, 1)]).toEqual([105]);
  });

  it('uses area coverage rather than a nearest-neighbour crop', () => {
    const source = new Uint8Array([0, 100, 200, 255]);
    expect([...resizeGrayscale(source, 2, 2, 1, 1)]).toEqual([139]);
  });

  it('uses OpenCV-compatible nearest-even rounding for area ties', () => {
    expect([...resizeGrayscale(new Uint8Array([0, 33]), 2, 1, 1, 1)]).toEqual([16]);
  });
});
