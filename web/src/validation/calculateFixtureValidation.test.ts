import { describe, expect, it } from 'vitest';

import { calculateFixtureValidation } from './calculateFixtureValidation';
import { validateBrowserFixtureReference } from './fixtures';

const reference = validateBrowserFixtureReference({
  schema_version: 1,
  artifact_type: 'day27_browser_fixture_reference',
  sample_count: 2,
  observation_dtype: 'uint8',
  observation_shape: [4, 84, 84],
  q_values: [[1, 2, 3, 4], [4, 3, 2, 1]],
  greedy_actions: [3, 0],
  action_meanings: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'],
  thresholds: { max_absolute_error: 0.1, mean_absolute_error: 0.1, action_agreement_rate: 1 },
});

describe('fixture validation calculation', () => {
  it('passes exact Q-values and actions', () => {
    expect(calculateFixtureValidation(reference.q_values, reference.greedy_actions, reference)).toMatchObject({
      maxAbsoluteError: 0,
      meanAbsoluteError: 0,
      actionAgreementRate: 1,
      disagreementIndices: [],
      passed: true,
    });
  });

  it('reports numerical error and action disagreements instead of hiding them', () => {
    const result = calculateFixtureValidation([[1.2, 2, 3, 4], [4, 3, 2, 1]], [3, 1], reference);
    expect(result.maxAbsoluteError).toBeCloseTo(0.2);
    expect(result.disagreementIndices).toEqual([1]);
    expect(result.passed).toBe(false);
  });
});
