import { describe, expect, it, vi } from 'vitest';

import type { PolicyResult } from '../inference/types';
import type { EnvironmentStep } from '../environment/aleEnvironment';
import { DualGameLoop, type LoopEnvironment } from './dualGameLoop';

function fakeStep(action: number): EnvironmentStep {
  return {
    observation: new Uint8Array(4 * 84 * 84),
    processedFrame: new Uint8Array(84 * 84),
    rawRgb: new Uint8Array(160 * 210 * 3),
    requestedModelAction: action,
    requestedAction: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'][action] as EnvironmentStep['requestedAction'],
    requestedAleAction: action as 0 | 1 | 3 | 4,
    executedModelAction: action,
    executedAction: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'][action] as EnvironmentStep['executedAction'],
    executedAleAction: action as 0 | 1 | 3 | 4,
    autoFire: false,
    autoFireReason: null,
    fireConfirmation: null,
    observationChangedFraction: 1,
    reward: 0,
    episodeReturn: 0,
    lives: 5,
    frameNumber: 4,
    agentStep: 1,
    actualEmulatorFrames: 4,
    rawFrameSkip: 1,
    outerActionRepeat: 4,
    terminated: false,
    truncated: false,
    gameOverReason: null,
    timing: { aleStepMs: 1, preprocessingMs: 1, totalMs: 2 },
  };
}

class FakeEnvironment implements LoopEnvironment {
  observation = new Uint8Array(4 * 84 * 84);
  isFinished = false;
  currentSeed = 1;
  actions: number[] = [];

  constructor(private readonly finishAfter: number | null = null) {}

  reset(seed = this.currentSeed): void {
    this.currentSeed = seed;
    this.isFinished = false;
    this.actions = [];
  }

  step(action: number): EnvironmentStep {
    this.actions.push(action);
    if (this.finishAfter !== null && this.actions.length >= this.finishAfter) this.isFinished = true;
    return fakeStep(action);
  }
}

const policyResult = (actionIndex: number): PolicyResult => ({
  qValues: [0, 0, 0, 0],
  actionIndex,
  action: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'][actionIndex] as PolicyResult['action'],
  requestedBackend: 'wasm',
  actualBackend: 'wasm',
});

describe('dual game loop', () => {
  it('steps independent sides and keeps requested human action separate from agent inference', async () => {
    const human = new FakeEnvironment();
    const agent = new FakeEnvironment();
    const onAgentStep = vi.fn();
    const loop = new DualGameLoop({
      human,
      agent,
      humanAction: () => 3,
      infer: async () => policyResult(2),
      onAgentStep,
    });

    await loop.stepOnce();

    expect(human.actions).toEqual([3]);
    expect(agent.actions).toEqual([2]);
    expect(onAgentStep).toHaveBeenCalledOnce();
  });

  it('does not infer concurrently and lets the other side continue after game over', async () => {
    const human = new FakeEnvironment();
    const agent = new FakeEnvironment(1);
    let release!: () => void;
    const infer = vi.fn(() => new Promise<PolicyResult>((resolve) => {
      release = () => resolve(policyResult(1));
    }));
    const loop = new DualGameLoop({ human, agent, humanAction: () => 0, infer });

    const first = loop.stepOnce();
    const second = loop.stepOnce();
    expect(infer).toHaveBeenCalledOnce();
    release();
    await first;
    await second;
    await loop.stepOnce();

    expect(agent.actions).toEqual([1]);
    expect(human.actions).toEqual([0, 0]);
  });
});
