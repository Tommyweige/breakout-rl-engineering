import { describe, expect, it, vi } from 'vitest';

import { InferenceScheduler } from './InferenceScheduler';

describe('InferenceScheduler', () => {
  it('allows only one in-flight inference and exposes the lifecycle state', async () => {
    let resolveInference!: (value: string) => void;
    const infer = vi.fn(() => new Promise<string>((resolve) => {
      resolveInference = resolve;
    }));
    const scheduler = new InferenceScheduler(infer);
    const states: string[] = [];
    scheduler.subscribe((status) => states.push(status));
    scheduler.start();

    const first = scheduler.run('first');
    expect(scheduler.currentStatus).toBe('in-flight');
    await expect(scheduler.run('second')).rejects.toThrow('already in flight');
    expect(infer).toHaveBeenCalledTimes(1);

    resolveInference('done');
    await expect(first).resolves.toBe('done');
    expect(scheduler.currentStatus).toBe('running');
    expect(states).toEqual(['idle', 'running', 'in-flight', 'running']);
  });

  it('does not run while paused and can be reset after an error', async () => {
    const scheduler = new InferenceScheduler(async () => {
      throw new Error('runtime failed');
    });
    scheduler.pause();
    await expect(scheduler.run('input')).rejects.toThrow('paused');
    expect(scheduler.currentStatus).toBe('paused');

    scheduler.start();
    await expect(scheduler.run('input')).rejects.toThrow('runtime failed');
    expect(scheduler.currentStatus).toBe('error');
    scheduler.reset();
    expect(scheduler.currentStatus).toBe('idle');
  });
});
