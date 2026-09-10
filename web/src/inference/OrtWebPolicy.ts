import * as ort from 'onnxruntime-web/wasm';

import { ACTION_MEANINGS, type PolicyResult } from './types';

export interface OrtWebPolicyOptions {
  modelUrl?: string;
  inputName?: string;
  outputName?: string;
}

export class OrtWebPolicy {
  readonly requestedBackend = 'wasm' as const;
  private readonly modelUrl: string;
  private readonly inputName: string;
  private readonly outputName: string;
  private session: ort.InferenceSession | null = null;

  constructor(options: OrtWebPolicyOptions = {}) {
    this.modelUrl = options.modelUrl ?? '/models/final_model/model.onnx';
    this.inputName = options.inputName ?? 'observation';
    this.outputName = options.outputName ?? 'q_values';
  }

  get isLoaded(): boolean {
    return this.session !== null;
  }

  get actualBackend(): 'wasm' | null {
    return this.session ? 'wasm' : null;
  }

  get ortWebVersion(): string {
    return ort.env.versions.web ?? 'unknown';
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
      // The Day 27 correctness baseline is deliberately single-threaded. This
      // keeps the same WASM behavior on localhost and on COOP/COEP Pages hosts
      // without relying on a separately served pthread worker asset.
      ort.env.wasm.numThreads = 1;
      // Keep this list deliberately explicit. If WASM cannot initialize, the promise
      // rejects; ORT is not allowed to silently select another execution provider.
      this.session = await ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ['wasm'],
      });
    } catch (error) {
      this.session = null;
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

    const normalized = normalizeObservation(observation);

    const input = new ort.Tensor('float32', normalized, [1, 4, 84, 84]);
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
      requestedBackend: 'wasm',
      actualBackend: this.actualBackend ?? 'wasm',
    };
  }
}

export function normalizeObservation(observation: Uint8Array | Float32Array): Float32Array {
  return observation instanceof Uint8Array
    ? Float32Array.from(observation, (value) => value / 255.0)
    : new Float32Array(observation);
}
