import { describe, expect, it } from 'vitest';

import { aggregateEpisodeScores, browserEvaluationSeeds, runPolicyEvaluation } from './policyEvaluation';
import type { BreakoutContractV2 } from '../environment/breakoutContract';

const testContract: BreakoutContractV2 = {
  schema_version: 2,
  contract_id: 'day15-breakout-evaluation-v2-fire-reset',
  environment_id: 'ALE/Breakout-v5',
  frame_skip: 4,
  frame_stack: 4,
  sticky_action_probability: 0.25,
  fire_reset: true,
  fire_reset_confirmation: {
    max_fire_attempts: 8,
    confirmation_steps: 2,
    min_observation_change_fraction: 0.0001,
    confirmation_operator: 'any',
    confirmation_signals: ['raw_reward', 'observation_activity_streak'],
  },
  terminal_on_life_loss: false,
  time_limit_semantics: {
    source: 'ale.game_truncated',
    max_num_frames_per_episode: 108000,
    agent_step_limit: 27000,
    truncated_is_finished: true,
    external_time_limit_wrapper: false,
  },
  concrete_episode_seeds: [101],
  evaluation_epsilon: 0,
  raw_reward_rule: 'sum environment rewards without clipping',
  sha256: 'b'.repeat(64),
  source_path: 'configs/eval/breakout_contract_v2.json',
  parity: { status: 'partial', unsupported_fields: [], note: 'test' },
};

describe('browser policy evaluation', () => {
  it('keeps a fixed set of at least 30 seeds', () => {
    const seeds = browserEvaluationSeeds(testContract);
    expect(seeds).toHaveLength(30);
    expect(new Set(seeds).size).toBe(30);
  });

  it('computes robust score aggregates from episode returns', () => {
    expect(aggregateEpisodeScores([1, 2, 3, 4, 5])).toMatchObject({
      count: 5,
      mean: 3,
      median: 3,
      min: 1,
      max: 5,
      p10: 1.4,
      p90: 4.6,
    });
  });

  it('runs sequential episodes and records backend, steps, and errors', async () => {
    const actions: number[] = [];
    const artifact = await runPolicyEvaluation({
      backend: 'wasm',
      seeds: [101, 102],
      modelSha256: 'a'.repeat(64),
      createEnvironment: async ({ seed }) => {
        let finished = false;
        let currentReturn = 0;
        let observation = new Uint8Array(4 * 84 * 84);
        return {
          observation,
          currentSeed: seed,
          currentLives: 5,
          currentAgentStep: 0,
          reset: () => {
            finished = false;
            currentReturn = 0;
          },
          step: (action: number) => {
            actions.push(action);
            currentReturn += action;
            observation = new Uint8Array(observation);
            finished = true;
            return {
              observation,
              processedFrame: new Uint8Array(84 * 84),
              rawRgb: new Uint8Array(160 * 210 * 3),
              requestedModelAction: action,
              requestedAction: 'NOOP' as const,
              requestedAleAction: 0 as const,
              executedModelAction: action,
              executedAction: 'NOOP' as const,
              executedAleAction: 0 as const,
              autoFire: false,
              autoFireReason: null,
              fireConfirmation: null,
              observationChangedFraction: 1,
              reward: action,
              episodeReturn: currentReturn,
              lives: 5,
              frameNumber: 4,
              agentStep: 1,
              actualEmulatorFrames: 4,
              rawFrameSkip: 1,
              outerActionRepeat: 4,
              terminated: true,
              truncated: false,
              gameOverReason: 'terminated' as const,
              timing: { aleStepMs: 1, preprocessingMs: 1, totalMs: 2 },
            };
          },
          get isFinished() {
            return finished;
          },
          get currentReturn() {
            return currentReturn;
          },
          dispose: () => undefined,
        };
      },
      infer: async () => ({
        qValues: [1, 2, 3, 4],
        actionIndex: 3,
        action: 'LEFT' as const,
        requestedBackend: 'wasm' as const,
        actualBackend: 'wasm' as const,
      }),
      policy: {
        requestedBackend: 'wasm' as const,
        actualBackend: 'wasm' as const,
        ortWebVersion: 'test',
        load: async () => undefined,
        infer: async () => ({
          qValues: [1, 2, 3, 4],
          actionIndex: 3,
          action: 'LEFT' as const,
          requestedBackend: 'wasm' as const,
          actualBackend: 'wasm' as const,
        }),
        release: async () => undefined,
      },
      browser: { name: 'Test', version: '1', userAgent: 'test', platform: 'test' },
      contract: testContract,
    });

    expect(actions).toEqual([3, 3]);
    expect(artifact.episodes).toHaveLength(2);
    expect(artifact.aggregate.successCount).toBe(2);
    expect(artifact.aggregate.crashCount).toBe(0);
    expect(artifact.actualBackend).toBe('wasm');
  });
});
