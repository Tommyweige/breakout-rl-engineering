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
  browser: { name: 'Chrome', version: '145.0.0.0', userAgent: 'test', platform: 'test-platform' },
  environmentContract: {
    contractId: 'day15-breakout-evaluation-v2-fire-reset',
    sourcePath: 'configs/eval/breakout_contract_v2.json',
    sha256: 'e'.repeat(64),
    status: 'partial',
    unsupportedFields: ['ALE gameplay'],
    note: 'fixed inference only',
  },
  backendEvidence: 'wasm_session_created_with_explicit_provider',
  webgpuSupport: {
    supported: false,
    navigatorGpu: false,
    adapterAvailable: false,
    adapterInfo: {},
    isSecureContext: false,
    error: 'navigator.gpu is unavailable in this browser',
  },
  qMargin: {
    minMargin: 1,
    meanMargin: 1,
    p50Margin: 1,
    maxMargin: 1,
    referenceMinMargin: 1,
    referenceMeanMargin: 1,
    referenceP50Margin: 1,
    referenceMaxMargin: 1,
    maxAbsoluteMarginError: 0,
  },
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

describe('Day 28 backend validation boundary', () => {
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

    expect(root.querySelector<HTMLSelectElement>('[data-action="backend"]')?.value).toBe('wasm');
    root.querySelector<HTMLButtonElement>('[data-action="validate"]')!.click();
    await vi.waitFor(() => expect(runFixtureValidationMock).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(root.querySelector('[data-role="validation-status"]')?.textContent).toBe('PASS'));

    expect(root.querySelector('[data-role="runtime-status"]')?.textContent).toBe('ready');
    expect(runWebGpuSmokeMock).not.toHaveBeenCalled();
  });

  it('uses the selected WebGPU policy and displays the actual backend', async () => {
    Object.defineProperty(navigator, 'gpu', {
      configurable: true,
      value: { requestAdapter: async () => ({ info: { vendor: 'test-vendor' } }) },
    });
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    runFixtureValidationMock.mockResolvedValue({
      ...wasmPassResult,
      requestedBackend: 'webgpu',
      actualBackend: 'webgpu',
      backendEvidence: 'webgpu_session_exposes_env_webgpu_device',
      webgpuSupport: {
        supported: true,
        navigatorGpu: true,
        adapterAvailable: true,
        adapterInfo: { vendor: 'test-vendor' },
        isSecureContext: true,
        error: null,
      },
    });

    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();

    const backend = root.querySelector<HTMLSelectElement>('[data-action="backend"]')!;
    backend.value = 'webgpu';
    backend.dispatchEvent(new Event('change', { bubbles: true }));
    await vi.waitFor(() => expect(backend.disabled).toBe(false));
    root.querySelector<HTMLButtonElement>('[data-action="validate"]')!.click();
    await vi.waitFor(() => expect(runFixtureValidationMock).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(root.querySelector('[data-role="validation-status"]')?.textContent).toBe('PASS'));

    expect(runFixtureValidationMock.mock.calls[0]?.[0].requestedBackend).toBe('webgpu');
    expect(root.querySelector('[data-role="requested-backend"]')?.textContent).toBe('WEBGPU');
    expect(root.querySelector('[data-role="actual-backend"]')?.textContent).toBe('WEBGPU');
    expect(root.querySelector('[data-role="webgpu-support"]')?.textContent).toBe('available');
  });

  it('keeps WebGPU unavailable explicit instead of presenting a fallback as GPU', async () => {
    Object.defineProperty(navigator, 'gpu', { configurable: true, value: undefined });
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    runFixtureValidationMock.mockRejectedValue(new Error('WebGPU is unavailable: navigator.gpu is unavailable in this browser'));

    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();

    const backend = root.querySelector<HTMLSelectElement>('[data-action="backend"]')!;
    backend.value = 'webgpu';
    backend.dispatchEvent(new Event('change', { bubbles: true }));
    await vi.waitFor(() => expect(backend.disabled).toBe(false));
    root.querySelector<HTMLButtonElement>('[data-action="validate"]')!.click();
    await vi.waitFor(() => expect(runFixtureValidationMock).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(root.querySelector('[data-role="validation-status"]')?.textContent).toBe('Unavailable'));

    expect(root.querySelector('[data-role="actual-backend"]')?.textContent).toBe('UNAVAILABLE');
    expect(root.querySelector('[data-role="webgpu-support"]')?.textContent).toBe('unavailable');
    expect(root.querySelector('[data-role="validation-message"]')?.textContent).toContain('WebGPU is unavailable');
  });
});
