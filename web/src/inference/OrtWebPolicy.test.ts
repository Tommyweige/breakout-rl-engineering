import { describe, expect, it } from 'vitest';

import { ACTION_MEANINGS } from './types';
import { normalizeObservation, OrtWebPolicy } from './OrtWebPolicy';

describe('OrtWebPolicy input contract', () => {
  it('normalizes raw uint8 observations exactly once', () => {
    expect([...normalizeObservation(new Uint8Array([0, 128, 255]))]).toEqual([0, 128 / 255, 1].map((value) => expect.closeTo(value, 6)));
  });

  it('copies already-normalized float32 observations without a second divide', () => {
    const input = new Float32Array([0, 0.5, 1]);
    const normalized = normalizeObservation(input);
    expect([...normalized]).toEqual([...input]);
    expect(normalized).not.toBe(input);
  });

  it('declares WASM as the only requested policy backend before a session exists', () => {
    const policy = new OrtWebPolicy();
    expect(policy.requestedBackend).toBe('wasm');
    expect(policy.actualBackend).toBeNull();
    expect(ACTION_MEANINGS).toEqual(['NOOP', 'FIRE', 'RIGHT', 'LEFT']);
  });
});
