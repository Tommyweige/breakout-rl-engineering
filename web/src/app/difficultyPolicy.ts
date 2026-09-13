import { ACTION_MEANINGS, type PolicyResult } from '../inference/types';

export const AI_DIFFICULTIES = ['easy', 'medium', 'hard', 'unbeatable'] as const;
export type AiDifficulty = (typeof AI_DIFFICULTIES)[number];

export const AI_DIFFICULTY_LABELS: Record<AiDifficulty, string> = {
  easy: 'EASY',
  medium: 'MEDIUM',
  hard: 'HARD',
  unbeatable: 'UNBEATABLE',
};

export const AI_DIFFICULTY_MISTAKE_RATES: Record<AiDifficulty, number> = {
  easy: 0.30,
  medium: 0.15,
  hard: 0.05,
  unbeatable: 0,
};

const MOVEMENT_ACTION_INDICES = [0, 2, 3] as const;
const FIRE_ACTION_INDEX = ACTION_MEANINGS.indexOf('FIRE');

/** Adds player-facing difficulty noise after model inference, without changing the model. */
export class DifficultyPolicy {
  private difficulty: AiDifficulty = 'hard';

  constructor(private readonly random: () => number = Math.random) {}

  get currentDifficulty(): AiDifficulty {
    return this.difficulty;
  }

  get currentMistakeRate(): number {
    return AI_DIFFICULTY_MISTAKE_RATES[this.difficulty];
  }

  setDifficulty(difficulty: AiDifficulty): void {
    this.difficulty = difficulty;
  }

  select(policy: PolicyResult): PolicyResult {
    const greedyActionIndex = policy.actionIndex;
    const mistakeRate = this.currentMistakeRate;
    const canInjectMistake = greedyActionIndex !== FIRE_ACTION_INDEX && mistakeRate > 0;
    const mistakeInjected = canInjectMistake && this.random() < mistakeRate;
    if (!mistakeInjected) {
      return {
        ...policy,
        difficulty: this.difficulty,
        mistakeRate,
        greedyActionIndex,
        mistakeInjected: false,
      };
    }

    const alternatives = MOVEMENT_ACTION_INDICES.filter((index) => index !== greedyActionIndex);
    const alternative = alternatives[Math.floor(this.random() * alternatives.length)] ?? greedyActionIndex;
    return {
      ...policy,
      actionIndex: alternative,
      action: ACTION_MEANINGS[alternative]!,
      difficulty: this.difficulty,
      mistakeRate,
      greedyActionIndex,
      mistakeInjected: alternative !== greedyActionIndex,
    };
  }
}

/** Small standalone RNG used by QA fixtures; it never shares ALE's random state. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

export function isAiDifficulty(value: string): value is AiDifficulty {
  return (AI_DIFFICULTIES as readonly string[]).includes(value);
}
