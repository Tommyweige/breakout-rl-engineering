import type {
  BrowserValidationArtifact,
  Day28ValidationArtifact,
  Day28ValidationSummary,
  FixtureValidationResult,
  QMarginDiagnostics,
  WebGpuSupport,
} from '../inference/types';

function copyQMargin(qMargin: QMarginDiagnostics): QMarginDiagnostics {
  return { ...qMargin };
}

export function buildDay28ValidationArtifact(result: FixtureValidationResult): Day28ValidationArtifact {
  return {
    schemaVersion: 1,
    artifactType: 'day28_browser_validation',
    sampleCount: result.sampleCount,
    requestedBackend: result.requestedBackend,
    actualBackend: result.actualBackend,
    maxAbsoluteError: result.maxAbsoluteError,
    meanAbsoluteError: result.meanAbsoluteError,
    actionAgreementRate: result.actionAgreementRate,
    disagreementIndices: [...result.disagreementIndices],
    qMargin: copyQMargin(result.qMargin),
    environmentContract: {
      ...result.environmentContract,
      unsupportedFields: [...result.environmentContract.unsupportedFields],
    },
    backendEvidence: result.backendEvidence,
    webgpuSupport: copyWebGpuSupport(result.webgpuSupport),
    browser: { ...result.browser },
    ortWebVersion: result.ortWebVersion,
    modelSha256: result.modelSha256,
    timestamp: result.timestamp,
    pageUrl: result.pageUrl,
    previewUrl: result.previewUrl,
    passed: result.passed,
    representative: {
      ...result.representative,
      qValues: [...result.representative.qValues],
      referenceQValues: [...result.representative.referenceQValues],
    },
    thresholds: { ...result.thresholds },
  };
}

export function buildDay28ValidationSummary(result: FixtureValidationResult): Day28ValidationSummary {
  return {
    sampleCount: result.sampleCount,
    requestedBackend: result.requestedBackend,
    actualBackend: result.actualBackend,
    maxAbsoluteError: result.maxAbsoluteError,
    meanAbsoluteError: result.meanAbsoluteError,
    actionAgreementRate: result.actionAgreementRate,
    disagreementIndices: [...result.disagreementIndices],
    qMargin: copyQMargin(result.qMargin),
    backendEvidence: result.backendEvidence,
    passed: result.passed,
  };
}

function copyWebGpuSupport(support: WebGpuSupport): WebGpuSupport {
  return { ...support, adapterInfo: { ...support.adapterInfo } };
}

export function buildBrowserValidationArtifact(
  result: FixtureValidationResult,
): BrowserValidationArtifact {
  if (result.requestedBackend !== 'wasm' || result.actualBackend !== 'wasm') {
    throw new Error('Day 27 browser validation artifact only accepts a WASM result');
  }
  return {
    schemaVersion: 1,
    artifactType: 'day27_browser_wasm_validation',
    sampleCount: result.sampleCount,
    requestedBackend: result.requestedBackend,
    actualBackend: result.actualBackend,
    maxAbsoluteError: result.maxAbsoluteError,
    meanAbsoluteError: result.meanAbsoluteError,
    actionAgreementRate: result.actionAgreementRate,
    disagreementIndices: [...result.disagreementIndices],
    browser: { ...result.browser },
    ortWebVersion: result.ortWebVersion,
    modelSha256: result.modelSha256,
    timestamp: result.timestamp,
    pageUrl: result.pageUrl,
    previewUrl: result.previewUrl,
    passed: result.passed,
    representative: {
      ...result.representative,
      qValues: [...result.representative.qValues],
      referenceQValues: [...result.representative.referenceQValues],
    },
    thresholds: { ...result.thresholds },
  };
}
