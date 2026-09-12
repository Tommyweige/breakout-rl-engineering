import { describe, expect, it } from 'vitest';

import { FrameStack } from './frameStack';

describe('four-frame observation stack', () => {
  it('fills reset history with the first frame and shifts oldest frames out', () => {
    const stack = new FrameStack(2, 2);
    expect([...stack.reset(new Uint8Array([1, 2]))]).toEqual([1, 2, 1, 2]);
    expect([...stack.push(new Uint8Array([3, 4]))]).toEqual([1, 2, 3, 4]);
    expect([...stack.push(new Uint8Array([5, 6]))]).toEqual([3, 4, 5, 6]);
  });

  it('returns copies so policy input cannot mutate stack history', () => {
    const stack = new FrameStack(4, 1);
    const observation = stack.reset(new Uint8Array([9]));
    observation[0] = 0;
    expect(stack.current()[0]).toBe(9);
  });
});
