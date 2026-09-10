import { ACTION_MEANINGS, type FixtureRepresentative, type FixtureValidationResult } from '../inference/types';
import { detectBrowser, currentPageUrl, currentPreviewUrl } from '../inference/browserInfo';
import { validateInferenceSpec, validateWebModelManifest, type InferenceSpec, type WebModelManifest } from '../inference/manifest';
import { OrtWebPolicy } from '../inference/OrtWebPolicy';
import { calculateFixtureValidation } from './calculateFixtureValidation';
import { validateBrowserFixtureReference, validateObservationFixture, type BrowserFixtureReference } from './fixtures';

interface LoadedFixtures {
  manifest: WebModelManifest;
  spec: InferenceSpec;
  reference: BrowserFixtureReference;
  observations: Uint8Array;
}

async function readResponse(response: Response, description: string): Promise<unknown> {
  if (!response.ok) throw new Error(`failed to load ${description}: ${response.status}`);
  return response.json();
}

async function loadFixtures(): Promise<LoadedFixtures> {
  const [manifestResponse, specResponse, referenceResponse, observationResponse] = await Promise.all([
    fetch('/web-model-manifest.json'),
    fetch('/inference_spec.json'),
    fetch('/fixtures/day22-reference.json'),
    fetch('/fixtures/day22-probe-observations.u8'),
  ]);
  const manifest = validateWebModelManifest(await readResponse(manifestResponse, 'web model manifest'));
  const spec = validateInferenceSpec(await readResponse(specResponse, 'inference spec'));
  const reference = validateBrowserFixtureReference(await readResponse(referenceResponse, 'browser reference fixture'));
  const observations = new Uint8Array(await (async () => {
    if (!observationResponse.ok) throw new Error(`failed to load browser observation fixture: ${observationResponse.status}`);
    return observationResponse.arrayBuffer();
  })());
  validateObservationFixture(observations, reference);

  if (manifest.fixtures.sample_count !== reference.sample_count) {
    throw new Error('manifest fixture sample_count does not match reference fixture');
  }
  if (spec.input.name !== 'observation' || spec.output.name !== 'q_values') {
    throw new Error('inference spec tensor names do not match the browser policy');
  }
  if (spec.actions.meanings.join('|') !== ACTION_MEANINGS.join('|')) {
    throw new Error('inference spec action mapping does not match the browser policy');
  }

  return { manifest, spec, reference, observations };
}

function representativeFrom(
  result: Awaited<ReturnType<OrtWebPolicy['infer']>>,
  expectedQValues: number[],
  expectedActionIndex: number,
): FixtureRepresentative {
  return {
    sampleIndex: 0,
    qValues: [...result.qValues],
    referenceQValues: [...expectedQValues],
    selectedAction: result.action,
    referenceAction: ACTION_MEANINGS[expectedActionIndex]!,
    actionIndex: result.actionIndex,
    referenceActionIndex: expectedActionIndex,
  };
}

export async function runFixtureValidation(policy: OrtWebPolicy): Promise<FixtureValidationResult> {
  const { manifest, reference, observations } = await loadFixtures();
  if (policy.modelPath !== manifest.model.url) {
    throw new Error(`policy model URL does not match manifest: ${policy.modelPath}`);
  }

  await policy.load();
  if (policy.actualBackend !== 'wasm') {
    throw new Error(`requested WASM but actual backend is ${policy.actualBackend ?? 'unavailable'}`);
  }

  const valuesPerSample = reference.observation_shape.reduce((product, value) => product * value, 1);
  const predictedQValues: number[][] = [];
  const predictedActions: number[] = [];
  let representative: FixtureRepresentative | null = null;

  for (let index = 0; index < reference.sample_count; index += 1) {
    const start = index * valuesPerSample;
    const observation = observations.slice(start, start + valuesPerSample);
    const result = await policy.infer(observation);
    predictedQValues.push([...result.qValues]);
    predictedActions.push(result.actionIndex);
    if (index === 0) {
      representative = representativeFrom(result, reference.q_values[0]!, reference.greedy_actions[0]!);
    }
  }

  if (!representative) throw new Error('fixture validation did not produce a representative inference');
  const calculation = calculateFixtureValidation(predictedQValues, predictedActions, reference);
  const timestamp = new Date().toISOString();

  return {
    sampleCount: reference.sample_count,
    ...calculation,
    requestedBackend: 'wasm',
    actualBackend: policy.actualBackend!,
    modelSha256: manifest.model.sha256,
    ortWebVersion: policy.ortWebVersion,
    browser: await detectBrowser(),
    timestamp,
    pageUrl: currentPageUrl(),
    previewUrl: currentPreviewUrl(),
    representative,
    thresholds: {
      maxAbsoluteError: reference.thresholds.max_absolute_error,
      meanAbsoluteError: reference.thresholds.mean_absolute_error,
      actionAgreementRate: reference.thresholds.action_agreement_rate,
    },
  };
}
