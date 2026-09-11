import { afterEach, describe, expect, it } from 'vitest';

import { detectWebGpuSupport } from './webgpuSupport';

const originalGpu = (navigator as Navigator & { gpu?: unknown }).gpu;

afterEach(() => {
  Object.defineProperty(navigator, 'gpu', { configurable: true, value: originalGpu });
});

describe('WebGPU feature detection', () => {
  it('reports unavailable without navigator.gpu instead of inventing a backend', async () => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: undefined });

    await expect(detectWebGpuSupport()).resolves.toMatchObject({
      supported: false,
      navigatorGpu: false,
      adapterAvailable: false,
      error: 'navigator.gpu is unavailable in this browser',
    });
  });

  it('reports the adapter and its metadata when the browser exposes WebGPU', async () => {
    Object.defineProperty(navigator, 'gpu', {
      configurable: true,
      value: {
        requestAdapter: async () => ({ info: { vendor: 'test-vendor', device: 'test-device' } }),
      },
    });

    await expect(detectWebGpuSupport()).resolves.toMatchObject({
      supported: true,
      navigatorGpu: true,
      adapterAvailable: true,
      adapterInfo: { vendor: 'test-vendor', device: 'test-device' },
      error: null,
    });
  });
});
