export interface WebModelManifest {
  schema_version: 1;
  artifact_type: 'day27_web_model_manifest';
  model: {
    url: string;
    source_path: string;
    sha256: string;
    size_bytes: number;
  };
  inference_spec: {
    url: string;
    source_path: string;
    sha256: string;
  };
  fixtures: {
    observations_url: string;
    reference_url: string;
    sample_count: number;
    observations_sha256: string;
    reference_sha256: string;
  };
  requested_backend: 'wasm';
  browser_runtime: 'onnxruntime-web';
}

export interface InferenceSpec {
  schema_version: number;
  environment_contract: {
    contract_id: string;
    path: string;
    sha256: string;
  };
  input: {
    name: string;
    dtype: string;
    shape: unknown[];
  };
  output: {
    name: string;
    dtype: string;
    shape: unknown[];
  };
  preprocessing: {
    source_observation_dtype: string;
    source_observation_shape: number[];
    normalization_divisor: number;
  };
  actions: {
    meanings: string[];
    greedy_rule: string;
    index_base: number;
  };
}

const SHA256_PATTERN = /^[a-f0-9]{64}$/i;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function requiredString(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`manifest field ${key} must be a non-empty string`);
  }
  return value;
}

function requiredSha256(record: Record<string, unknown>, key: string): string {
  const value = requiredString(record, key);
  if (!SHA256_PATTERN.test(value)) {
    throw new Error(`manifest field ${key} must be a SHA256 hex digest`);
  }
  return value;
}

function requiredPositiveInteger(record: Record<string, unknown>, key: string): number {
  const value = record[key];
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`manifest field ${key} must be a positive integer`);
  }
  return value as number;
}

function requiredObject(record: Record<string, unknown>, key: string): Record<string, unknown> {
  const value = record[key];
  if (!isRecord(value)) throw new Error(`manifest field ${key} must be an object`);
  return value;
}

export function validateWebModelManifest(value: unknown): WebModelManifest {
  if (!isRecord(value)) throw new Error('web model manifest must be an object');
  if (value.schema_version !== 1) throw new Error('unsupported web model manifest schema');
  if (value.artifact_type !== 'day27_web_model_manifest') {
    throw new Error('unexpected web model manifest artifact type');
  }
  if (value.requested_backend !== 'wasm') {
    throw new Error('web model manifest must request the WASM backend');
  }
  if (value.browser_runtime !== 'onnxruntime-web') {
    throw new Error('web model manifest must name onnxruntime-web');
  }

  const model = requiredObject(value, 'model');
  const inferenceSpec = requiredObject(value, 'inference_spec');
  const fixtures = requiredObject(value, 'fixtures');
  const modelUrl = requiredString(model, 'url');
  if (modelUrl !== '/models/final_model/model.onnx') {
    throw new Error(`unexpected model URL: ${modelUrl}`);
  }
  const inferenceSpecUrl = requiredString(inferenceSpec, 'url');
  if (inferenceSpecUrl !== '/inference_spec.json') {
    throw new Error(`unexpected inference spec URL: ${inferenceSpecUrl}`);
  }
  const observationsUrl = requiredString(fixtures, 'observations_url');
  const referenceUrl = requiredString(fixtures, 'reference_url');
  if (!observationsUrl.startsWith('/fixtures/') || !referenceUrl.startsWith('/fixtures/')) {
    throw new Error('fixture URLs must be same-origin absolute paths');
  }

  return {
    schema_version: 1,
    artifact_type: 'day27_web_model_manifest',
    model: {
      url: modelUrl,
      source_path: requiredString(model, 'source_path'),
      sha256: requiredSha256(model, 'sha256'),
      size_bytes: requiredPositiveInteger(model, 'size_bytes'),
    },
    inference_spec: {
      url: inferenceSpecUrl,
      source_path: requiredString(inferenceSpec, 'source_path'),
      sha256: requiredSha256(inferenceSpec, 'sha256'),
    },
    fixtures: {
      observations_url: observationsUrl,
      reference_url: referenceUrl,
      sample_count: requiredPositiveInteger(fixtures, 'sample_count'),
      observations_sha256: requiredSha256(fixtures, 'observations_sha256'),
      reference_sha256: requiredSha256(fixtures, 'reference_sha256'),
    },
    requested_backend: 'wasm',
    browser_runtime: 'onnxruntime-web',
  };
}

export function validateInferenceSpec(value: unknown): InferenceSpec {
  if (!isRecord(value)) throw new Error('inference spec must be an object');
  const environmentContract = requiredObject(value, 'environment_contract');
  const contractId = requiredString(environmentContract, 'contract_id');
  const contractPath = requiredString(environmentContract, 'path');
  if (contractId !== 'day15-breakout-evaluation-v2-fire-reset') {
    throw new Error(`inference spec must reference Contract v2, got ${contractId}`);
  }
  if (contractPath !== 'configs/eval/breakout_contract_v2.json') {
    throw new Error(`inference spec must reference the canonical Contract v2 path, got ${contractPath}`);
  }
  const input = requiredObject(value, 'input');
  const output = requiredObject(value, 'output');
  const preprocessing = requiredObject(value, 'preprocessing');
  const actions = requiredObject(value, 'actions');
  const meanings = actions.meanings;
  if (!Array.isArray(meanings) || meanings.some((item) => typeof item !== 'string')) {
    throw new Error('inference spec action meanings must be strings');
  }
  if (meanings.join('|') !== 'NOOP|FIRE|RIGHT|LEFT') {
    throw new Error(`unexpected action mapping: ${meanings.join(', ')}`);
  }
  if (input.name !== 'observation' || input.dtype !== 'float32') {
    throw new Error('unexpected inference input contract');
  }
  if (output.name !== 'q_values' || output.dtype !== 'float32') {
    throw new Error('unexpected inference output contract');
  }
  if (preprocessing.source_observation_dtype !== 'uint8' || preprocessing.normalization_divisor !== 255) {
    throw new Error('unexpected browser preprocessing contract');
  }
  if (actions.greedy_rule !== 'argmax' || actions.index_base !== 0) {
    throw new Error('unexpected action selection contract');
  }

  return {
    schema_version: typeof value.schema_version === 'number' ? value.schema_version : 0,
    environment_contract: {
      contract_id: contractId,
      path: contractPath,
      sha256: requiredSha256(environmentContract, 'sha256'),
    },
    input: {
      name: input.name,
      dtype: input.dtype,
      shape: Array.isArray(input.shape) ? input.shape : [],
    },
    output: {
      name: output.name,
      dtype: output.dtype,
      shape: Array.isArray(output.shape) ? output.shape : [],
    },
    preprocessing: {
      source_observation_dtype: preprocessing.source_observation_dtype,
      source_observation_shape: Array.isArray(preprocessing.source_observation_shape)
        ? preprocessing.source_observation_shape.filter((item): item is number => typeof item === 'number')
        : [],
      normalization_divisor: preprocessing.normalization_divisor,
    },
    actions: {
      meanings: [...meanings],
      greedy_rule: actions.greedy_rule,
      index_base: actions.index_base,
    },
  };
}
