import { describe, expect, it } from 'vitest';

import { validateInferenceSpec, validateWebModelManifest } from './manifest';

const manifest = {
  schema_version: 1,
  artifact_type: 'day27_web_model_manifest',
  model: { url: '/models/final_model/model.onnx', source_path: 'assets/day22/models/final_model/model.onnx', sha256: 'a'.repeat(64), size_bytes: 10 },
  inference_spec: { url: '/inference_spec.json', source_path: 'configs/inference/inference_spec.json', sha256: 'b'.repeat(64) },
  fixtures: { observations_url: '/fixtures/observations.u8', reference_url: '/fixtures/reference.json', sample_count: 60, observations_sha256: 'c'.repeat(64), reference_sha256: 'd'.repeat(64) },
  requested_backend: 'wasm',
  browser_runtime: 'onnxruntime-web',
};

const spec = {
  schema_version: 1,
  input: { name: 'observation', dtype: 'float32', shape: ['N', 4, 84, 84] },
  output: { name: 'q_values', dtype: 'float32', shape: ['N', 4] },
  preprocessing: { source_observation_dtype: 'uint8', source_observation_shape: [4, 84, 84], normalization_divisor: 255 },
  actions: { meanings: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'], greedy_rule: 'argmax', index_base: 0 },
};

describe('web model manifest validation', () => {
  it('accepts the packaged model contract', () => {
    expect(validateWebModelManifest(manifest).model.sha256).toBe('a'.repeat(64));
  });

  it('rejects a manifest that could point to a different backend or model', () => {
    expect(() => validateWebModelManifest({ ...manifest, requested_backend: 'cpu' })).toThrow(/WASM/);
    expect(() => validateWebModelManifest({ ...manifest, model: { ...manifest.model, url: '/models/other.onnx' } })).toThrow(/model URL/);
  });
});

describe('inference action contract', () => {
  it('accepts the Day 22 input, output, normalization, and action mapping', () => {
    expect(validateInferenceSpec(spec).actions.meanings).toEqual(['NOOP', 'FIRE', 'RIGHT', 'LEFT']);
  });

  it('rejects a changed action mapping', () => {
    expect(() => validateInferenceSpec({ ...spec, actions: { ...spec.actions, meanings: ['NOOP', 'FIRE', 'LEFT', 'RIGHT'] } })).toThrow(/action mapping/);
  });
});
