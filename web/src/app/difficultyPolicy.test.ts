import { describe, expect, it } from 'vitest';

import { AI_DIFFICULTIES, createSeededRandom, DifficultyPolicy } from './difficultyPolicy';
import type { PolicyResult } from '../inference/types';

const policy: PolicyResult = {
  qValues: [10, 1, 4, 3],
  actionIndex: 0,
  action: 'NOOP',
  requestedBackend: 'webgpu',
  actualBackend: 'webgpu',
};

describe('DifficultyPolicy', () => {
  it('keeps UNBEATABLE exactly greedy and does not consume randomness', () => {
    let calls = 0;
    const difficulty = new DifficultyPolicy(() => { calls += 1; return 0; });
    difficulty.setDifficulty('unbeatable');
    const selected = difficulty.select(policy);
    expect(selected.actionIndex).toBe(policy.actionIndex);
    expect(selected.greedyActionIndex).toBe(policy.actionIndex);
    expect(selected.mistakeInjected).toBe(false);
    expect(calls).toBe(0);
  });

  it('injects only a legal movement mistake when the rate triggers', () => {
    const difficulty = new DifficultyPolicy(() => 0);
    difficulty.setDifficulty('easy');
    const selected = difficulty.select(policy);
    expect([0, 2, 3]).toContain(selected.actionIndex);
    expect(selected.actionIndex).not.toBe(policy.actionIndex);
    expect(selected.action).not.toBe('FIRE');
    expect(selected.mistakeInjected).toBe(true);
  });

  it('does not inject a mistake into FIRE actions', () => {
    const difficulty = new DifficultyPolicy(() => 0);
    difficulty.setDifficulty('easy');
    const selected = difficulty.select({ ...policy, actionIndex: 1, action: 'FIRE' });
    expect(selected.actionIndex).toBe(1);
    expect(selected.action).toBe('FIRE');
    expect(selected.mistakeInjected).toBe(false);
  });

  it('keeps UNBEATABLE equal to fixed greedy fixture actions', () => {
    const fixtures = [
      { ...policy, qValues: [0.1, 0.2, 0.9, 0.3], actionIndex: 2, action: 'RIGHT' as const },
      { ...policy, qValues: [0.8, 0.2, 0.1, 0.7], actionIndex: 0, action: 'NOOP' as const },
      { ...policy, qValues: [0.1, 0.7, 0.2, 0.9], actionIndex: 3, action: 'LEFT' as const },
    ];
    const difficulty = new DifficultyPolicy(createSeededRandom(44));
    difficulty.setDifficulty('unbeatable');

    for (const fixture of fixtures) {
      const selected = difficulty.select(fixture);
      expect(selected.actionIndex).toBe(fixture.actionIndex);
      expect(selected.action).toBe(fixture.action);
      expect(selected.mistakeInjected).toBe(false);
    }
  });

  it('reproduces the expected mistake-rate ordering with a fixed independent RNG seed', () => {
    const sampleCount = 10_000;
    const countMistakes = (difficultyName: (typeof AI_DIFFICULTIES)[number]): number => {
      const sampler = new DifficultyPolicy(createSeededRandom(6801));
      sampler.setDifficulty(difficultyName);
      let mistakes = 0;
      for (let index = 0; index < sampleCount; index += 1) {
        if (sampler.select(policy).mistakeInjected) mistakes += 1;
      }
      return mistakes;
    };

    const counts = AI_DIFFICULTIES.map((difficultyName) => countMistakes(difficultyName));
    expect(counts).toEqual([2994, 1510, 520, 0]);
    expect(counts[0]!).toBeGreaterThan(counts[1]!);
    expect(counts[1]!).toBeGreaterThan(counts[2]!);
    expect(counts[2]!).toBeGreaterThan(counts[3]!);
    expect(countMistakes('easy')).toBe(counts[0]!);
  });
});
