import { describe, expect, it } from 'vitest';

import {
  ALE_ACTIONS,
  actionMeaningFromModelIndex,
  mapModelActionToAle,
  validateMinimalActionSet,
} from './actionMapping';

describe('Breakout model action mapping', () => {
  it('maps the model order to the ALE minimal action codes', () => {
    expect(ALE_ACTIONS).toEqual({ NOOP: 0, FIRE: 1, RIGHT: 3, LEFT: 4 });
    expect([0, 1, 2, 3].map((index) => mapModelActionToAle(index).aleAction)).toEqual([0, 1, 3, 4]);
    expect(actionMeaningFromModelIndex(2)).toBe('RIGHT');
  });

  it('rejects unknown policy outputs and incomplete ALE action sets', () => {
    expect(() => mapModelActionToAle(-1)).toThrow(/model action index/);
    expect(() => mapModelActionToAle(4)).toThrow(/model action index/);
    expect(() => validateMinimalActionSet([0, 1, 3])).toThrow(/missing/);
    expect(() => validateMinimalActionSet([0, 1, 2, 3])).toThrow(/ALE action/);
  });
});
