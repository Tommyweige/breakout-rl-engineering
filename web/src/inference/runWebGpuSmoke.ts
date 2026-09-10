import { ACTION_MEANINGS, type WebGpuSmokeResult } from './types';
import { normalizeObservation } from './OrtWebPolicy';

interface GpuAdapterInfoLike {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
}

interface GpuAdapterLike {
  info?: GpuAdapterInfoLike;
  requestDevice?: () => Promise<unknown>;
}

interface GpuLike {
  requestAdapter: () => Promise<GpuAdapterLike | null>;
}

function gpuFromNavigator(): GpuLike | undefined {
  return (navigator as Navigator & { gpu?: GpuLike }).gpu;
}

function adapterInfo(adapter: GpuAdapterLike | null): Record<string, string> {
  if (!adapter?.info) return {};
  return Object.fromEntries(
    Object.entries(adapter.info).filter((entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1].length > 0),
  );
}

export async function runWebGpuSmoke(
  modelUrl: string,
  inputName: string,
  outputName: string,
  observation: Uint8Array,
): Promise<WebGpuSmokeResult> {
  const timestamp = new Date().toISOString();
  const gpu = gpuFromNavigator();
  if (!gpu) {
    return {
      requestedBackend: 'webgpu',
      actualBackend: 'unavailable',
      navigatorGpu: false,
      adapterAvailable: false,
      adapterInfo: {},
      sampleIndex: 0,
      qValues: [],
      selectedAction: null,
      ortWebVersion: 'unknown',
      timestamp,
      passed: false,
      error: 'navigator.gpu is unavailable in this browser',
    };
  }

  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    return {
      requestedBackend: 'webgpu',
      actualBackend: 'unavailable',
      navigatorGpu: true,
      adapterAvailable: false,
      adapterInfo: {},
      sampleIndex: 0,
      qValues: [],
      selectedAction: null,
      ortWebVersion: 'unknown',
      timestamp,
      passed: false,
      error: 'navigator.gpu.requestAdapter() returned no adapter',
    };
  }

  try {
    const ortWebGpu = await import('onnxruntime-web/webgpu');
    ortWebGpu.env.wasm.numThreads = 1;
    const session = await ortWebGpu.InferenceSession.create(modelUrl, {
      // This is a smoke test for WebGPU itself. There is intentionally no WASM
      // fallback in this session option.
      executionProviders: ['webgpu'],
    });
    try {
      const normalized = normalizeObservation(observation);
      const input = new ortWebGpu.Tensor('float32', normalized, [1, 4, 84, 84]);
      const outputs = await session.run({ [inputName]: input });
      const output = outputs[outputName];
      if (!output) throw new Error(`WebGPU output ${outputName} was not returned`);
      const qValues = Array.from(output.data as Float32Array, Number);
      if (qValues.length !== ACTION_MEANINGS.length) {
        throw new Error(`WebGPU returned ${qValues.length} Q-values instead of ${ACTION_MEANINGS.length}`);
      }
      let actionIndex = 0;
      for (let index = 1; index < qValues.length; index += 1) {
        if (qValues[index]! > qValues[actionIndex]!) actionIndex = index;
      }
      return {
        requestedBackend: 'webgpu',
        actualBackend: 'webgpu',
        navigatorGpu: true,
        adapterAvailable: true,
        adapterInfo: adapterInfo(adapter),
        sampleIndex: 0,
        qValues,
        selectedAction: ACTION_MEANINGS[actionIndex]!,
        ortWebVersion: ortWebGpu.env.versions.web ?? 'unknown',
        timestamp,
        passed: true,
      };
    } finally {
      await session.release();
    }
  } catch (error) {
    return {
      requestedBackend: 'webgpu',
      actualBackend: 'unavailable',
      navigatorGpu: true,
      adapterAvailable: true,
      adapterInfo: adapterInfo(adapter),
      sampleIndex: 0,
      qValues: [],
      selectedAction: null,
      ortWebVersion: 'unknown',
      timestamp,
      passed: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
