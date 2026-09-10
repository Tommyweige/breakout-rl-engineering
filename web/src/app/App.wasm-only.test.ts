// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

const { runFixtureValidationMock, runWebGpuSmokeMock } = vi.hoisted(() => ({
  runFixtureValidationMock: vi.fn(),
  runWebGpuSmokeMock: vi.fn(),
}));

vi.mock('../validation/runFixtureValidation', () => ({
  runFixtureValidation: runFixtureValidationMock,
}));
vi.mock('../inference/runWebGpuSmoke', () => ({
  runWebGpuSmoke: runWebGpuSmokeMock,
}));

import type { FixtureValidationResult } from '../inference/types';
import { App } from './App';

let app: App | undefined;

const wasmPassResult: FixtureValidationResult = {
  sampleCount: 60,
  maxAbsoluteError: 0.0001,
  meanAbsoluteError: 0.00001,
  actionAgreementRate: 1,
  disagreementIndices: [],
  requestedBackend: 'wasm',
  actualBackend: 'wasm',
  passed: true,
  modelSha256: 'a'.repeat(64),
  ortWebVersion: '1.29.0',
  browser: { name: 'Chrome', version: '145.0.0.0', userAgent: 'test' },
  timestamp: '2026-09-06T00:00:00.000Z',
  pageUrl: 'http://localhost:5173/',
  previewUrl: null,
  representative: {
    sampleIndex: 0,
    qValues: [1, 2, 3, 4],
    referenceQValues: [1, 2, 3, 4],
    selectedAction: 'LEFT',
    referenceAction: 'LEFT',
    actionIndex: 3,
    referenceActionIndex: 3,
  },
  thresholds: {
    maxAbsoluteError: 0.0002,
    meanAbsoluteError: 0.0001,
    actionAgreementRate: 1,
  },
};

afterEach(() => {
  app?.destroy();
  app = undefined;
  document.body.innerHTML = '';
  vi.clearAllMocks();
});

describe('Day 27 WASM-only validation boundary', () => {
  it('stays ready and PASS when navigator.gpu is unavailable', async () => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: undefined });
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    runFixtureValidationMock.mockResolvedValue(wasmPassResult);
    runWebGpuSmokeMock.mockResolvedValue({
      requestedBackend: 'webgpu',
      actualBackend: 'unavailable',
      navigatorGpu: false,
      adapterAvailable: false,
      adapterInfo: {},
      sampleIndex: 0,
      qValues: [],
      selectedAction: null,
      ortWebVersion: '1.29.0',
      timestamp: '2026-09-06T00:00:00.000Z',
      passed: false,
      error: 'navigator.gpu is unavailable in this browser',
    });

    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();

    expect(root.querySelector('[data-action="webgpu"]')).toBeNull();
    root.querySelector<HTMLButtonElement>('[data-action="validate"]')!.click();
    await vi.waitFor(() => expect(runFixtureValidationMock).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(root.querySelector('[data-role="validation-status"]')?.textContent).toBe('PASS'));

    expect(root.querySelector('[data-role="runtime-status"]')?.textContent).toBe('ready');
    expect(runWebGpuSmokeMock).not.toHaveBeenCalled();
  });
});
