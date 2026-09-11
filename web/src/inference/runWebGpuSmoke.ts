import { ACTION_MEANINGS, type WebGpuSmokeResult } from './types';
import { OrtWebPolicy } from './OrtWebPolicy';
import { detectWebGpuSupport } from './webgpuSupport';

export async function runWebGpuSmoke(
  modelUrl: string,
  inputName: string,
  outputName: string,
  observation: Uint8Array,
): Promise<WebGpuSmokeResult> {
  const timestamp = new Date().toISOString();
  const support = await detectWebGpuSupport();
  if (!support.supported) {
    return {
      requestedBackend: 'webgpu',
      actualBackend: 'unavailable',
      navigatorGpu: support.navigatorGpu,
      adapterAvailable: support.adapterAvailable,
      adapterInfo: support.adapterInfo,
      isSecureContext: support.isSecureContext,
      sampleIndex: 0,
      qValues: [],
      selectedAction: null,
      ortWebVersion: 'unknown',
      timestamp,
      passed: false,
      error: support.error ?? 'WebGPU is unavailable',
    };
  }

  const policy = new OrtWebPolicy({ backend: 'webgpu', modelUrl, inputName, outputName });
  try {
    await policy.load();
    const result = await policy.infer(observation);
    return {
      requestedBackend: 'webgpu',
      actualBackend: result.actualBackend === 'webgpu' ? 'webgpu' : 'unavailable',
      navigatorGpu: support.navigatorGpu,
      adapterAvailable: support.adapterAvailable,
      adapterInfo: support.adapterInfo,
      isSecureContext: support.isSecureContext,
      sampleIndex: 0,
      qValues: [...result.qValues],
      selectedAction: ACTION_MEANINGS[result.actionIndex] ?? null,
      ortWebVersion: policy.ortWebVersion,
      timestamp,
      passed: result.actualBackend === 'webgpu',
    };
  } catch (error) {
    return {
      requestedBackend: 'webgpu',
      actualBackend: 'unavailable',
      navigatorGpu: support.navigatorGpu,
      adapterAvailable: support.adapterAvailable,
      adapterInfo: support.adapterInfo,
      isSecureContext: support.isSecureContext,
      sampleIndex: 0,
      qValues: [],
      selectedAction: null,
      ortWebVersion: policy.ortWebVersion,
      timestamp,
      passed: false,
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    await policy.release();
  }
}
