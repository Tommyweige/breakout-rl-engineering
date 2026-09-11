import * as ortWasm from 'onnxruntime-web/wasm';

import {
  ACTION_MEANINGS,
  type BackendEvidence,
  type InferenceBackend,
  type PolicyResult,
  type WebGpuSupport,
} from './types';
import { detectWebGpuAdapter } from './webgpuSupport';

export interface OrtWebPolicyOptions {
  backend?: InferenceBackend;
  modelUrl?: string;
  inputName?: string;
  outputName?: string;
}

type OrtRuntime = typeof ortWasm;

export class OrtWebPolicy {
  readonly requestedBackend: InferenceBackend;
  private readonly modelUrl: string;
  private readonly inputName: string;
  private readonly outputName: string;
  private runtime: OrtRuntime = ortWasm;
  private session: ortWasm.InferenceSession | null = null;
  private activeBackend: InferenceBackend | null = null;
  private evidence: BackendEvidence | null = null;
  private support: WebGpuSupport | null = null;

  constructor(options: OrtWebPolicyOptions = {}) {
    this.requestedBackend = options.backend ?? 'wasm';
    this.modelUrl = options.modelUrl ?? '/models/final_model/model.onnx';
    this.inputName = options.inputName ?? 'observation';
    this.outputName = options.outputName ?? 'q_values';
  }

  get isLoaded(): boolean {
    return this.session !== null;
  }

  get actualBackend(): InferenceBackend | null {
    return this.activeBackend;
  }

  get backendEvidence(): BackendEvidence | null {
    return this.evidence;
  }

  get ortWebVersion(): string {
    return this.runtime.env.versions.web ?? 'unknown';
  }

  get webgpuSupport(): WebGpuSupport | null {
    return this.support ? { ...this.support, adapterInfo: { ...this.support.adapterInfo } } : null;
  }

  get modelPath(): string {
    return this.modelUrl;
  }

  get inputTensorName(): string {
    return this.inputName;
  }

  get outputTensorName(): string {
    return this.outputName;
  }

  async load(): Promise<void> {
    if (this.session) return;

    try {
      if (this.requestedBackend === 'webgpu') {
        const probe = await detectWebGpuAdapter();
        this.support = probe.support;
        if (!this.support.supported) {
          throw new Error(`WebGPU is unavailable: ${this.support.error ?? 'no adapter available'}`);
        }
        if (!probe.adapter) throw new Error('WebGPU is unavailable: adapter was not returned');
        this.runtime = await import('onnxruntime-web/webgpu');
        // Pin the ORT session to the adapter we just observed. ORT exposes the
        // GPUDevice it actually uses through env.webgpu.device after creation.
        if (!this.runtime.env.webgpu.adapter) {
          this.runtime.env.webgpu.adapter = probe.adapter as GPUAdapter;
        }
      } else {
        this.runtime = ortWasm;
      }
      // The Day 27 correctness baseline is deliberately single-threaded. This
      // keeps the same WASM behavior on localhost and on COOP/COEP Pages hosts
      // without relying on a separately served pthread worker asset.
      this.runtime.env.wasm.numThreads = 1;
      // Keep this list deliberately explicit. If the requested provider cannot
      // initialize, the promise rejects; ORT is not allowed to silently select
      // another execution provider for this policy.
      this.session = await this.runtime.InferenceSession.create(this.modelUrl, {
        executionProviders: [this.requestedBackend],
      });
      if (this.requestedBackend === 'webgpu') {
        const activeDevice = await this.runtime.env.webgpu.device;
        if (!activeDevice || typeof activeDevice !== 'object' || !('queue' in activeDevice)) {
          throw new Error('WebGPU session did not expose an active env.webgpu.device');
        }
        this.activeBackend = 'webgpu';
        this.evidence = 'webgpu_session_exposes_env_webgpu_device';
      } else {
        this.activeBackend = 'wasm';
        this.evidence = 'wasm_session_created_with_explicit_provider';
      }
    } catch (error) {
      const session = this.session;
      this.session = null;
      this.activeBackend = null;
      this.evidence = null;
      if (session) await session.release().catch(() => undefined);
      throw error;
    }
  }

  async infer(observation: Uint8Array | Float32Array): Promise<PolicyResult> {
    if (!this.session) {
      throw new Error('OrtWebPolicy.load() must complete before inference');
    }

    const expectedValues = 4 * 84 * 84;
    if (observation.length !== expectedValues) {
      throw new Error(`expected ${expectedValues} observation values, got ${observation.length}`);
    }

    return this.inferPrepared(normalizeObservation(observation));
  }

  async inferPrepared(preparedObservation: Float32Array): Promise<PolicyResult> {
    if (!this.session) {
      throw new Error('OrtWebPolicy.load() must complete before inference');
    }

    const expectedValues = 4 * 84 * 84;
    if (preparedObservation.length !== expectedValues) {
      throw new Error(`expected ${expectedValues} prepared observation values, got ${preparedObservation.length}`);
    }

    const input = new this.runtime.Tensor('float32', preparedObservation, [1, 4, 84, 84]);
    const outputs = await this.session.run({ [this.inputName]: input });
    const output = outputs[this.outputName];
    if (!output) {
      throw new Error(`ORT Web output ${this.outputName} was not returned`);
    }

    const qValues = Array.from(output.data as Float32Array, Number);
    if (qValues.length !== ACTION_MEANINGS.length) {
      throw new Error(`expected ${ACTION_MEANINGS.length} Q-values, got ${qValues.length}`);
    }

    let actionIndex = 0;
    for (let index = 1; index < qValues.length; index += 1) {
      if ((qValues[index] ?? Number.NEGATIVE_INFINITY) > (qValues[actionIndex] ?? Number.NEGATIVE_INFINITY)) {
        actionIndex = index;
      }
    }

    return {
      qValues,
      actionIndex,
      action: ACTION_MEANINGS[actionIndex]!,
      requestedBackend: this.requestedBackend,
      actualBackend: this.actualBackend ?? this.requestedBackend,
    };
  }

  async release(): Promise<void> {
    if (!this.session) return;
    const session = this.session;
    this.session = null;
    this.activeBackend = null;
    this.evidence = null;
    await session.release();
  }
}

export function normalizeObservation(observation: Uint8Array | Float32Array): Float32Array {
  return observation instanceof Uint8Array
    ? Float32Array.from(observation, (value) => value / 255.0)
    : new Float32Array(observation);
}
