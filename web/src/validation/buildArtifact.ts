import type { BrowserValidationArtifact, FixtureValidationResult } from '../inference/types';

export function buildBrowserValidationArtifact(
  result: FixtureValidationResult,
): BrowserValidationArtifact {
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
