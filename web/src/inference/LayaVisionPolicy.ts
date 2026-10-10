import { ACTION_MEANINGS, type PolicyResult } from './types';

interface LayaResponse {
  ready?: unknown;
  device?: unknown;
  actionIndex?: unknown;
  action?: unknown;
  probabilities?: unknown;
  modelRevision?: unknown;
  inferenceMode?: unknown;
  inferenceMs?: unknown;
}

/** One RGB image per request; the local service owns the frozen CUDA model. */
export class LayaVisionPolicy {
  modelRevision = 'unknown';
  async load(): Promise<void> {
    const runtime = await this.request('/api/laya/health');
    if (runtime.ready !== true || typeof runtime.device !== 'string' || !runtime.device.startsWith('cuda')) {
      throw new Error('The local Laya GPU service is not ready.');
    }
    if (typeof runtime.modelRevision !== 'string' || !/^[a-f0-9]{40}$/.test(runtime.modelRevision)) {
      throw new Error('The local Laya service did not identify its pinned checkpoint.');
    }
    this.modelRevision = runtime.modelRevision;
  }

  async infer(rgb: Uint8Array): Promise<PolicyResult> {
    if (rgb.length !== 160 * 210 * 3) throw new Error('Laya requires one complete RGB frame.');
    const result = await this.request('/api/laya/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: new Uint8Array(rgb),
    });
    const index = result.actionIndex;
    const probabilities = result.probabilities;
    if (typeof index !== 'number' || !Number.isInteger(index) || index < 0 || index >= ACTION_MEANINGS.length || ACTION_MEANINGS[index] !== result.action
      || (probabilities != null && (!Array.isArray(probabilities) || probabilities.length !== 4
      || probabilities.some((value: unknown) => typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)
      || Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) > 0.001))
      || (result.inferenceMs != null && (typeof result.inferenceMs !== 'number' || !Number.isFinite(result.inferenceMs) || result.inferenceMs < 0))
      || (result.inferenceMode != null && (typeof result.inferenceMode !== 'string' || !['reference', 'fixed-eager'].includes(result.inferenceMode)))
      || typeof result.device !== 'string' || !result.device.startsWith('cuda')) {
      throw new Error('The local Laya service returned an invalid decision.');
    }
    return { qValues: Array.isArray(probabilities) ? probabilities : [], actionIndex: index, action: ACTION_MEANINGS[index]!,
      requestedBackend: 'cuda', actualBackend: 'cuda', difficulty: 'decision-model',
      mistakeRate: 0, greedyActionIndex: index, mistakeInjected: false,
      inferenceMode: typeof result.inferenceMode === 'string' ? result.inferenceMode : undefined,
      serviceInferenceMs: typeof result.inferenceMs === 'number' ? result.inferenceMs : undefined };
  }

  private async request(path: string, options: RequestInit = {}): Promise<LayaResponse> {
    try {
      const response = await fetch(path, { ...options, signal: AbortSignal.timeout(30_000) });
      if (!response.ok) throw new Error(`Laya service HTTP ${response.status}`);
      const payload: LayaResponse = await response.json();
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('Invalid Laya response');
      return payload;
    } catch (error) {
      throw new Error(`Cannot reach the local Laya GPU service. Start the Laya server and retry. (${String(error)})`);
    }
  }
}
