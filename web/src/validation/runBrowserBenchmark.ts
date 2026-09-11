import { detectBrowser, currentPageUrl, currentPreviewUrl } from '../inference/browserInfo';
import { OrtWebPolicy, normalizeObservation } from '../inference/OrtWebPolicy';
import type {
  BrowserBenchmarkBackendResult,
  Day28BenchmarkArtifact,
  FixtureValidationResult,
  InferenceBackend,
  LatencySummary,
} from '../inference/types';
import { buildDay28ValidationSummary } from './buildArtifact';
import { loadBrowserFixtures, runFixtureValidation } from './runFixtureValidation';

export interface BrowserBenchmarkOptions {
  sampleCount?: number;
  warmupCount?: number;
}

export function summarizeLatency(
  rawLatencyMs: readonly number[],
  warmupCount: number,
): LatencySummary {
  if (rawLatencyMs.length === 0) throw new Error('latency samples must not be empty');
  if (!rawLatencyMs.every((sample) => Number.isFinite(sample) && sample >= 0)) {
    throw new Error('latency samples must be finite non-negative numbers');
  }
  const sorted = [...rawLatencyMs].sort((left, right) => left - right);
  const percentile = (probability: number): number => {
    const position = (sorted.length - 1) * probability;
    const lowerIndex = Math.floor(position);
    const upperIndex = Math.ceil(position);
    const lower = sorted[lowerIndex]!;
    const upper = sorted[upperIndex]!;
    return lower + (upper - lower) * (position - lowerIndex);
  };
  const meanMs = rawLatencyMs.reduce((sum, sample) => sum + sample, 0) / rawLatencyMs.length;
  const variance = rawLatencyMs.reduce((sum, sample) => sum + (sample - meanMs) ** 2, 0) / rawLatencyMs.length;
  return {
    sampleCount: rawLatencyMs.length,
    warmupCount,
    p50Ms: percentile(0.5),
    p95Ms: percentile(0.95),
    meanMs,
    stdMs: Math.sqrt(variance),
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
  };
}

async function benchmarkPolicy(
  policy: OrtWebPolicy,
  observation: Uint8Array,
  sampleCount: number,
  warmupCount: number,
): Promise<BrowserBenchmarkBackendResult> {
  await policy.load();
  if (policy.actualBackend !== policy.requestedBackend) {
    throw new Error(
      `benchmark requested ${policy.requestedBackend.toUpperCase()} but actual backend is ${policy.actualBackend ?? 'unavailable'}`,
    );
  }
  const preparedObservation = normalizeObservation(observation);
  for (let index = 0; index < warmupCount; index += 1) {
    await policy.inferPrepared(preparedObservation);
  }

  const rawLatencyMs: number[] = [];
  for (let index = 0; index < sampleCount; index += 1) {
    const startedAt = performance.now();
    await policy.inferPrepared(preparedObservation);
    rawLatencyMs.push(performance.now() - startedAt);
  }

  return {
    requestedBackend: policy.requestedBackend,
    actualBackend: policy.actualBackend!,
    ortWebVersion: policy.ortWebVersion,
    backendEvidence: policy.backendEvidence!,
    warmupCount,
    rawLatencyMs,
    summary: summarizeLatency(rawLatencyMs, warmupCount),
  };
}

function validateOptions(options: BrowserBenchmarkOptions): Required<BrowserBenchmarkOptions> {
  const sampleCount = options.sampleCount ?? 100;
  const warmupCount = options.warmupCount ?? 10;
  if (!Number.isInteger(sampleCount) || sampleCount <= 0) throw new Error('benchmark sampleCount must be positive');
  if (!Number.isInteger(warmupCount) || warmupCount < 0) throw new Error('benchmark warmupCount must be non-negative');
  return { sampleCount, warmupCount };
}

function firstObservation(observations: Uint8Array, valuesPerSample: number): Uint8Array {
  return observations.slice(0, valuesPerSample);
}

function validationSummary(result: FixtureValidationResult) {
  return buildDay28ValidationSummary(result);
}

export async function runBrowserBackendComparison(
  options: BrowserBenchmarkOptions = {},
): Promise<Day28BenchmarkArtifact> {
  const { sampleCount, warmupCount } = validateOptions(options);
  const fixtures = await loadBrowserFixtures();
  const valuesPerSample = fixtures.reference.observation_shape.reduce((product, value) => product * value, 1);
  const observation = firstObservation(fixtures.observations, valuesPerSample);
  const policies: Record<InferenceBackend, OrtWebPolicy> = {
    wasm: new OrtWebPolicy({ backend: 'wasm' }),
    webgpu: new OrtWebPolicy({ backend: 'webgpu' }),
  };

  try {
    const wasmValidation = await runFixtureValidation(policies.wasm);
    const webgpuValidation = await runFixtureValidation(policies.webgpu);
    const wasm = await benchmarkPolicy(policies.wasm, observation, sampleCount, warmupCount);
    const webgpu = await benchmarkPolicy(policies.webgpu, observation, sampleCount, warmupCount);
    if (wasm.actualBackend !== 'wasm' || webgpu.actualBackend !== 'webgpu') {
      throw new Error('backend comparison did not retain the requested execution providers');
    }

    return {
      schemaVersion: 1,
      artifactType: 'day28_web_benchmark',
      timestamp: new Date().toISOString(),
      pageUrl: currentPageUrl(),
      previewUrl: currentPreviewUrl(),
      browser: await detectBrowser(),
      modelSha256: fixtures.manifest.model.sha256,
      requestedBackends: ['wasm', 'webgpu'],
      scope: 'prepared_float32_to_q_values_ready',
      sessionInitialization: 'excluded_from_latency_samples',
      sampleCount,
      warmupCount,
      environmentContract: webgpuValidation.environmentContract,
      backendEvidence: {
        wasm: wasm.backendEvidence,
        webgpu: webgpu.backendEvidence,
      },
      webgpuSupport: webgpuValidation.webgpuSupport,
      validations: {
        wasm: validationSummary(wasmValidation),
        webgpu: validationSummary(webgpuValidation),
      },
      results: { wasm, webgpu },
    };
  } finally {
    await Promise.all([policies.wasm.release(), policies.webgpu.release()]);
  }
}
