import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import { validateInferenceSpec, validateWebModelManifest } from '../inference/manifest';

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

describe('published browser asset provenance', () => {
  it('pins the exact contract, inference spec, fixtures, and model bytes from the repository', () => {
    const contract = readFileSync(new URL('../../../configs/eval/breakout_contract_v2.json', import.meta.url));
    const sourceSpec = readFileSync(new URL('../../../configs/inference/inference_spec.json', import.meta.url));
    const publicSpec = readFileSync(new URL('../../public/inference_spec.json', import.meta.url));
    const reference = readFileSync(new URL('../../public/fixtures/day22-reference.json', import.meta.url));
    const observations = readFileSync(new URL('../../public/fixtures/day22-probe-observations.u8', import.meta.url));
    const model = readFileSync(new URL('../../public/models/final_model/model.onnx', import.meta.url));
    const manifest = validateWebModelManifest(JSON.parse(readFileSync(
      new URL('../../public/web-model-manifest.json', import.meta.url),
      'utf8',
    )));
    const spec = validateInferenceSpec(JSON.parse(publicSpec.toString('utf8')));

    expect(publicSpec).toEqual(sourceSpec);
    expect(sha256(contract)).toBe(spec.environment_contract.sha256);
    expect(sha256(publicSpec)).toBe(manifest.inference_spec.sha256);
    expect(sha256(reference)).toBe(manifest.fixtures.reference_sha256);
    expect(sha256(observations)).toBe(manifest.fixtures.observations_sha256);
    expect(sha256(model)).toBe(manifest.model.sha256);
  });
});
