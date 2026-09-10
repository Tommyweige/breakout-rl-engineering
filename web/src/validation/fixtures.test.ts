import { describe, expect, it } from 'vitest';

import { validateBrowserFixtureReference, validateObservationFixture } from './fixtures';

const reference = {
  schema_version: 1,
  artifact_type: 'day27_browser_fixture_reference',
  sample_count: 2,
  observation_dtype: 'uint8',
  observation_shape: [4, 84, 84],
  q_values: [[1, 2, 3, 4], [4, 3, 2, 1]],
  greedy_actions: [3, 0],
  action_meanings: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'],
  thresholds: { max_absolute_error: 0.1, mean_absolute_error: 0.1, action_agreement_rate: 1 },
};

describe('browser fixture loading and shape checks', () => {
  it('accepts the expected four-frame 84x84 fixture', () => {
    const parsed = validateBrowserFixtureReference(reference);
    expect(parsed.sample_count).toBe(2);
    expect(() => validateObservationFixture(new Uint8Array(2 * 4 * 84 * 84), parsed)).not.toThrow();
  });

  it('rejects truncated observation bytes and malformed rows', () => {
    const parsed = validateBrowserFixtureReference(reference);
    expect(() => validateObservationFixture(new Uint8Array(3), parsed)).toThrow(/byte length/);
    expect(() => validateBrowserFixtureReference({ ...reference, q_values: [[1, 2]] })).toThrow(/q_values/);
    expect(() => validateBrowserFixtureReference({ ...reference, observation_shape: [1, 84, 84] })).toThrow(/observation_shape/);
  });
});
