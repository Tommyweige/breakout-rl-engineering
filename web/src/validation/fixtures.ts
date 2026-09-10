export interface BrowserFixtureReference {
  schema_version: number;
  artifact_type: string;
  sample_count: number;
  observation_dtype: 'uint8';
  observation_shape: [number, number, number];
  q_values: number[][];
  greedy_actions: number[];
  action_meanings: string[];
  thresholds: {
    max_absolute_error: number;
    mean_absolute_error: number;
    action_agreement_rate: number;
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isFiniteNumberArray(value: unknown, length: number): value is number[] {
  return Array.isArray(value) && value.length === length && value.every((item) => typeof item === 'number' && Number.isFinite(item));
}

export function validateBrowserFixtureReference(value: unknown): BrowserFixtureReference {
  if (!isRecord(value)) throw new Error('browser reference fixture must be an object');
  const sampleCount = value.sample_count;
  const shape = value.observation_shape;
  const qValues = value.q_values;
  const greedyActions = value.greedy_actions;
  const thresholds = value.thresholds;
  if (!Number.isInteger(sampleCount) || (sampleCount as number) <= 0) {
    throw new Error('fixture sample_count must be a positive integer');
  }
  if (!Array.isArray(shape) || shape.length !== 3 || shape.some((item) => item !== 4 && item !== 84)) {
    throw new Error('fixture observation_shape must be [4, 84, 84]');
  }
  if (shape[0] !== 4 || shape[1] !== 84 || shape[2] !== 84) {
    throw new Error('fixture observation_shape must be [4, 84, 84]');
  }
  if (!Array.isArray(qValues) || qValues.length !== sampleCount || qValues.some((row) => !isFiniteNumberArray(row, 4))) {
    throw new Error('fixture q_values must contain four finite values per sample');
  }
  if (!Array.isArray(greedyActions) || greedyActions.length !== sampleCount || greedyActions.some((item) => !Number.isInteger(item) || item < 0 || item > 3)) {
    throw new Error('fixture greedy_actions must contain action indices from 0 to 3');
  }
  if (!isRecord(thresholds)) throw new Error('fixture thresholds must be an object');
  const maxError = thresholds.max_absolute_error;
  const meanError = thresholds.mean_absolute_error;
  const agreement = thresholds.action_agreement_rate;
  if (![maxError, meanError, agreement].every((item) => typeof item === 'number' && Number.isFinite(item))) {
    throw new Error('fixture thresholds must be finite numbers');
  }
  if (value.observation_dtype !== 'uint8') throw new Error('fixture observation_dtype must be uint8');
  if (!Array.isArray(value.action_meanings) || value.action_meanings.join('|') !== 'NOOP|FIRE|RIGHT|LEFT') {
    throw new Error('fixture action mapping is not the Day 22 mapping');
  }

  return {
    schema_version: typeof value.schema_version === 'number' ? value.schema_version : 0,
    artifact_type: typeof value.artifact_type === 'string' ? value.artifact_type : '',
    sample_count: sampleCount as number,
    observation_dtype: 'uint8',
    observation_shape: [4, 84, 84],
    q_values: qValues as number[][],
    greedy_actions: greedyActions as number[],
    action_meanings: [...(value.action_meanings as string[])],
    thresholds: {
      max_absolute_error: maxError as number,
      mean_absolute_error: meanError as number,
      action_agreement_rate: agreement as number,
    },
  };
}

export function validateObservationFixture(bytes: Uint8Array, reference: BrowserFixtureReference): void {
  const valuesPerSample = reference.observation_shape.reduce((product, value) => product * value, 1);
  const expectedBytes = reference.sample_count * valuesPerSample;
  if (bytes.length !== expectedBytes) {
    throw new Error(`fixture byte length mismatch: expected ${expectedBytes}, got ${bytes.length}`);
  }
}
