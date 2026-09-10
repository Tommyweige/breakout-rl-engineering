import { describe, expect, it } from 'vitest';

import { reduceRuntimeStatus } from './runtimeState';

describe('shared panel state transitions', () => {
  it('covers loading, ready, running, paused, and error states', () => {
    expect(reduceRuntimeStatus('idle', 'load-start')).toBe('loading');
    expect(reduceRuntimeStatus('loading', 'load-success')).toBe('ready');
    expect(reduceRuntimeStatus('ready', 'start')).toBe('running');
    expect(reduceRuntimeStatus('running', 'pause')).toBe('paused');
    expect(reduceRuntimeStatus('paused', 'start')).toBe('running');
    expect(reduceRuntimeStatus('loading', 'load-error')).toBe('error');
    expect(reduceRuntimeStatus('error', 'reset')).toBe('ready');
  });

  it('does not let a shared Start control interrupt an active model load', () => {
    expect(reduceRuntimeStatus('loading', 'start')).toBe('loading');
    expect(reduceRuntimeStatus('ready', 'pause')).toBe('ready');
  });
});
