export const ACTION_MEANINGS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT'] as const;

export type ActionMeaning = (typeof ACTION_MEANINGS)[number];
export type RuntimeStatus = 'idle' | 'loading' | 'ready' | 'running' | 'paused' | 'error';
export type BackendName = 'wasm' | 'webgpu' | 'unavailable';

export interface BrowserInfo {
  name: string;
  version: string;
  userAgent: string;
}

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
  requestedBackend: 'wasm';
  actualBackend: 'wasm';
}

export interface FixtureValidationResult {
  sampleCount: number;
  maxAbsoluteError: number;
  meanAbsoluteError: number;
  actionAgreementRate: number;
  disagreementIndices: number[];
  requestedBackend: 'wasm';
  actualBackend: 'wasm';
  passed: boolean;
  modelSha256: string;
  ortWebVersion: string;
  browser: BrowserInfo;
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
