export const ACTION_MEANINGS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT'] as const;

export type ActionMeaning = (typeof ACTION_MEANINGS)[number];
export type RuntimeStatus = 'idle' | 'loading' | 'ready' | 'running' | 'paused' | 'error';
export type InferenceBackend = 'wasm' | 'webgpu';
export type BackendName = InferenceBackend | 'unavailable';
export type InferenceSchedulerStatus = 'idle' | 'running' | 'in-flight' | 'paused' | 'error';

export interface BrowserInfo {
  name: string;
  version: string;
  userAgent: string;
  platform: string;
}

export interface WebGpuSupport {
  supported: boolean;
  navigatorGpu: boolean;
  adapterAvailable: boolean;
  adapterInfo: Record<string, string>;
  isSecureContext: boolean;
  error: string | null;
}

export interface BrowserEnvironmentParity {
  contractId: string;
  sourcePath: string;
  sha256: string;
  status: 'partial';
  unsupportedFields: string[];
  note: string;
}

export type BackendEvidence =
  | 'wasm_session_created_with_explicit_provider'
  | 'webgpu_session_exposes_env_webgpu_device';

export interface FixtureRepresentative {
  sampleIndex: number;
  qValues: number[];
  referenceQValues: number[];
  selectedAction: ActionMeaning;
  referenceAction: ActionMeaning;
  actionIndex: number;
  referenceActionIndex: number;
}

export interface PolicyResult {
  qValues: readonly number[];
  actionIndex: number;
  action: ActionMeaning;
  requestedBackend: InferenceBackend;
  actualBackend: InferenceBackend;
}

export interface QMarginDiagnostics {
  minMargin: number;
  meanMargin: number;
  p50Margin: number;
  maxMargin: number;
  referenceMinMargin: number;
  referenceMeanMargin: number;
  referenceP50Margin: number;
  referenceMaxMargin: number;
  maxAbsoluteMarginError: number;
}

export interface FixtureValidationResult {
  sampleCount: number;
  maxAbsoluteError: number;
  meanAbsoluteError: number;
  actionAgreementRate: number;
  disagreementIndices: number[];
  requestedBackend: InferenceBackend;
  actualBackend: InferenceBackend;
  passed: boolean;
  modelSha256: string;
  ortWebVersion: string;
  browser: BrowserInfo;
  environmentContract: BrowserEnvironmentParity;
  backendEvidence: BackendEvidence;
  webgpuSupport: WebGpuSupport;
  qMargin: QMarginDiagnostics;
  timestamp: string;
  pageUrl: string;
  previewUrl: string | null;
  representative: FixtureRepresentative;
  thresholds: {
    maxAbsoluteError: number;
    meanAbsoluteError: number;
    actionAgreementRate: number;
  };
}

export interface WebGpuSmokeResult {
  requestedBackend: 'webgpu';
  actualBackend: 'webgpu' | 'unavailable';
  navigatorGpu: boolean;
  adapterAvailable: boolean;
  adapterInfo: Record<string, string>;
  isSecureContext: boolean;
  sampleIndex: number;
  qValues: number[];
  selectedAction: ActionMeaning | null;
  ortWebVersion: string;
  timestamp: string;
  passed: boolean;
  error?: string;
}

export interface BrowserValidationArtifact {
  schemaVersion: 1;
  artifactType: 'day27_browser_wasm_validation';
  sampleCount: number;
  requestedBackend: 'wasm';
  actualBackend: 'wasm';
  maxAbsoluteError: number;
  meanAbsoluteError: number;
  actionAgreementRate: number;
  disagreementIndices: number[];
  browser: BrowserInfo;
  ortWebVersion: string;
  modelSha256: string;
  timestamp: string;
  pageUrl: string;
  previewUrl: string | null;
  passed: boolean;
  representative: FixtureRepresentative;
  thresholds: {
    maxAbsoluteError: number;
    meanAbsoluteError: number;
    actionAgreementRate: number;
  };
}

export interface Day28ValidationArtifact {
  schemaVersion: 1;
  artifactType: 'day28_browser_validation';
  sampleCount: number;
  requestedBackend: InferenceBackend;
  actualBackend: InferenceBackend;
  maxAbsoluteError: number;
  meanAbsoluteError: number;
  actionAgreementRate: number;
  disagreementIndices: number[];
  qMargin: QMarginDiagnostics;
  environmentContract: BrowserEnvironmentParity;
  backendEvidence: BackendEvidence;
  webgpuSupport: WebGpuSupport;
  browser: BrowserInfo;
  ortWebVersion: string;
  modelSha256: string;
  timestamp: string;
  pageUrl: string;
  previewUrl: string | null;
  passed: boolean;
  representative: FixtureRepresentative;
  thresholds: {
    maxAbsoluteError: number;
    meanAbsoluteError: number;
    actionAgreementRate: number;
  };
}

export interface LatencySummary {
  sampleCount: number;
  warmupCount: number;
  p50Ms: number;
  p95Ms: number;
  meanMs: number;
  stdMs: number;
  minMs: number;
  maxMs: number;
}

export interface BrowserBenchmarkBackendResult {
  requestedBackend: InferenceBackend;
  actualBackend: InferenceBackend;
  ortWebVersion: string;
  backendEvidence: BackendEvidence;
  warmupCount: number;
  rawLatencyMs: number[];
  summary: LatencySummary;
}

export interface Day28ValidationSummary {
  sampleCount: number;
  requestedBackend: InferenceBackend;
  actualBackend: InferenceBackend;
  maxAbsoluteError: number;
  meanAbsoluteError: number;
  actionAgreementRate: number;
  disagreementIndices: number[];
  qMargin: QMarginDiagnostics;
  backendEvidence: BackendEvidence;
  passed: boolean;
}

export interface Day28BenchmarkArtifact {
  schemaVersion: 1;
  artifactType: 'day28_web_benchmark';
  timestamp: string;
  pageUrl: string;
  previewUrl: string | null;
  browser: BrowserInfo;
  modelSha256: string;
  requestedBackends: ['wasm', 'webgpu'];
  scope: 'prepared_float32_to_q_values_ready';
  sessionInitialization: 'excluded_from_latency_samples';
  sampleCount: number;
  warmupCount: number;
  environmentContract: BrowserEnvironmentParity;
  backendEvidence: {
    wasm: BackendEvidence;
    webgpu: BackendEvidence;
  };
  webgpuSupport: WebGpuSupport;
  validations: {
    wasm: Day28ValidationSummary;
    webgpu: Day28ValidationSummary;
  };
  results: {
    wasm: BrowserBenchmarkBackendResult;
    webgpu: BrowserBenchmarkBackendResult;
  };
}
