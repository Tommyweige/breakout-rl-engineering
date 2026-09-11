import { describe, expect, it } from 'vitest';

import { summarizeLatency } from './runBrowserBenchmark';

describe('browser benchmark summaries', () => {
  it('derives p50, p95, mean, and standard deviation from raw samples', () => {
    const summary = summarizeLatency([1, 2, 3, 4], 2);
    expect(summary).toMatchObject({
      sampleCount: 4,
      warmupCount: 2,
      p50Ms: 2.5,
      meanMs: 2.5,
      stdMs: Math.sqrt(1.25),
      minMs: 1,
      maxMs: 4,
    });
    expect(summary.p95Ms).toBeCloseTo(3.85, 10);
  });

  it('rejects an empty or invalid raw sample set', () => {
    expect(() => summarizeLatency([], 0)).toThrow(/must not be empty/);
    expect(() => summarizeLatency([1, Number.NaN], 0)).toThrow(/finite/);
  });
});
