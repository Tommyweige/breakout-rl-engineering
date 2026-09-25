import { describe, expect, it, vi } from 'vitest';

import type { EnvironmentStep, HumanEnvironmentStep } from '../environment/aleEnvironment';
import type { PolicyResult } from '../inference/types';
import type { HumanPaddleCommand } from '../input/paddleCommand';
import { DualGameLoop, type AgentLoopStep, type HumanLoopEnvironment, type LoopEnvironment } from './dualGameLoop';

function fakeAgentStep(action: number): EnvironmentStep {
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

function fakeHumanStep(action: number, frame: number): HumanEnvironmentStep {
  const meaning = ['NOOP', 'FIRE', 'RIGHT', 'LEFT'][action] as HumanEnvironmentStep['requestedAction'];
  return {
    rawRgb: new Uint8Array(160 * 210 * 3),
    requestedModelAction: action,
    requestedAction: meaning,
    requestedDirection: meaning,
    requestedPaddleStrength: null,
    requestedPaddlePositionX: null,
    requestedAleAction: action as 0 | 1 | 3 | 4,
    executedModelAction: action,
    executedAction: meaning,
    executedDirection: meaning,
    executedPaddleStrength: null,
    appliedPaddleTargetX: null,
    executedAleAction: action as 0 | 1 | 3 | 4,
    autoFire: false,
    autoFireReason: null,
    fireConfirmation: null,
    reward: 0,
    episodeReturn: 0,
    lives: 5,
    frameNumber: frame,
    rawFrameNumber: frame,
    humanStep: frame,
    actualEmulatorFrames: 1,
    rawFrameSkip: 1,
    outerActionRepeat: 1,
    stickyActionProbability: 0,
    terminated: false,
    truncated: false,
    gameOverReason: null,
    timing: { aleStepMs: 1, totalMs: 1 },
  };
}

class FakeHumanEnvironment implements HumanLoopEnvironment {
  isFinished = false;
  currentSeed = 1;
  actions: number[] = [];
  paddleCommands: HumanPaddleCommand[] = [];
  private frame = 0;

  reset(seed = this.currentSeed): void {
    this.currentSeed = seed;
    this.isFinished = false;
    this.actions = [];
    this.paddleCommands = [];
    this.frame = 0;
  }

  step(action: number): HumanEnvironmentStep {
    this.actions.push(action);
    this.frame += 1;
    return fakeHumanStep(action, this.frame);
  }

  stepPaddle(command: HumanPaddleCommand): HumanEnvironmentStep {
    this.paddleCommands.push(command);
    this.frame += 1;
    return fakeHumanStep(command.direction === 'RIGHT' ? 2 : command.direction === 'LEFT' ? 3 : 0, this.frame);
  }
}

class FakeAgentEnvironment implements LoopEnvironment {
  observation = new Uint8Array(4 * 84 * 84);
  isFinished = false;
  currentSeed = 1;
  actions: number[] = [];
  stepAsync?: LoopEnvironment['stepAsync'];

  reset(seed = this.currentSeed): void {
    this.currentSeed = seed;
    this.isFinished = false;
    this.actions = [];
  }

  step(action: number): EnvironmentStep {
    this.actions.push(action);
    return fakeAgentStep(action);
  }
}

const policyResult = (actionIndex: number): PolicyResult => ({
  qValues: [0, 0, 0, 0],
  actionIndex,
  action: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'][actionIndex] as PolicyResult['action'],
  requestedBackend: 'wasm',
  actualBackend: 'wasm',
});

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe('dual game loop', () => {
  it('steps independent sides through the deterministic test seam', async () => {
    const human = new FakeHumanEnvironment();
    const agent = new FakeAgentEnvironment();
    const onAgentStep = vi.fn();
    const loop = new DualGameLoop({
      human,
      agent,
      agentRuntime: { outerActionRepeat: 4, stickyActionProbability: 0.25 },
      humanCommand: () => ({ kind: 'discrete', actionIndex: 3 }),
      infer: async () => policyResult(2),
      onAgentStep,
    });

    await loop.stepOnce();

    expect(human.actions).toEqual([3]);
    expect(agent.actions).toEqual([2]);
    expect(onAgentStep).toHaveBeenCalledOnce();
  });

  it('reports policy inference separately from the Agent environment step', async () => {
    const human = new FakeHumanEnvironment();
    const agent = new FakeAgentEnvironment();
    const steps: AgentLoopStep[] = [];
    const loop = new DualGameLoop({
      human,
      agent,
      agentRuntime: { outerActionRepeat: 4, stickyActionProbability: 0.25 },
      humanCommand: () => ({ kind: 'discrete', actionIndex: 0 }),
      infer: async () => {
        await wait(10);
        return policyResult(2);
      },
      onAgentStep: (step) => steps.push(step),
    });

    await loop.stepOnce();

    const step = steps[0];
    expect(step).toBeDefined();
    expect(step!.inferenceMs).toBeGreaterThanOrEqual(10);
    expect(step!.environmentStepMs).toBeGreaterThanOrEqual(0);
    expect(step!.totalDecisionMs).toBeGreaterThanOrEqual(step!.inferenceMs + step!.environmentStepMs);
  });

  it('routes absolute paddle commands only to Human while Agent keeps discrete actions', async () => {
    const human = new FakeHumanEnvironment();
    const agent = new FakeAgentEnvironment();
    const command: HumanPaddleCommand = {
      direction: 'RIGHT',
      targetX: 0.8,
      paddleCenterX: 0.5,
      positionError: 0.3,
    };
    const loop = new DualGameLoop({
      human,
      agent,
      agentRuntime: { outerActionRepeat: 4, stickyActionProbability: 0.25 },
      humanCommand: () => ({ kind: 'paddle', command }),
      infer: async () => policyResult(2),
    });

    await loop.stepOnce();

    expect(human.actions).toEqual([]);
    expect(human.paddleCommands).toEqual([command]);
    expect(agent.actions).toEqual([2]);
  });

  it('passes the configured raw-frame repeat to the async Agent environment', async () => {
    const human = new FakeHumanEnvironment();
    const agent = new FakeAgentEnvironment();
    const repeats: number[] = [];
    agent.stepAsync = async (actionIndex, rawFrameRepeat = 4) => {
      agent.actions.push(actionIndex);
      repeats.push(rawFrameRepeat);
      return { ...fakeAgentStep(actionIndex), actualEmulatorFrames: rawFrameRepeat, outerActionRepeat: rawFrameRepeat };
    };
    const loop = new DualGameLoop({
      human,
      agent,
      agentRuntime: { outerActionRepeat: 1, stickyActionProbability: 0.25 },
      humanCommand: () => ({ kind: 'discrete', actionIndex: 0 }),
      infer: async () => policyResult(2),
    });

    await loop.stepOnce();

    expect(agent.actions).toEqual([2]);
    expect(repeats).toEqual([1]);
    expect(loop.runtimeDiagnostics.agentRawFrameDelta).toBe(1);
  });

  it('continues Human raw ticks while Agent inference is slow', async () => {
    const human = new FakeHumanEnvironment();
    const agent = new FakeAgentEnvironment();
    let inferenceCount = 0;
    let inFlight = 0;
    let maxInFlight = 0;
    const loop = new DualGameLoop({
      human,
      agent,
      agentRuntime: { outerActionRepeat: 4, stickyActionProbability: 0.25 },
      humanCommand: () => ({ kind: 'discrete', actionIndex: 0 }),
      humanTargetFps: 60,
      agentTargetFps: 15,
      infer: () => new Promise((resolve) => {
        inferenceCount += 1;
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        setTimeout(() => {
          inFlight -= 1;
          resolve(policyResult(1));
        }, 50);
      }),
    });

    loop.start();
    await wait(260);
    loop.pause();

    expect(human.actions.length).toBeGreaterThanOrEqual(6);
    expect(agent.actions.length).toBeGreaterThanOrEqual(2);
    expect(inferenceCount).toBeGreaterThanOrEqual(agent.actions.length);
    expect(inferenceCount - agent.actions.length).toBeLessThanOrEqual(1);
    expect(maxInFlight).toBe(1);
    expect(loop.runtimeDiagnostics.humanRawFrameDelta).toBe(human.actions.length);
  });

  it('does not add a full idle interval after an Agent decision overruns its target', async () => {
    const human = new FakeHumanEnvironment();
    const agent = new FakeAgentEnvironment();
    let inFlight = 0;
    let maxInFlight = 0;
    const loop = new DualGameLoop({
      human,
      agent,
      agentRuntime: { outerActionRepeat: 4, stickyActionProbability: 0.25 },
      humanCommand: () => ({ kind: 'discrete', actionIndex: 0 }),
      agentTargetFps: 15,
      infer: () => new Promise((resolve) => {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        setTimeout(() => {
          inFlight -= 1;
          resolve(policyResult(1));
        }, 80);
      }),
    });

    loop.start();
    await wait(420);
    loop.pause();
    const diagnostics = loop.runtimeDiagnostics;
    await wait(90);
    loop.destroy();

    expect(agent.actions.length).toBeGreaterThanOrEqual(3);
    expect(diagnostics.agentDecisionP50Ms).toBeLessThan(125);
    expect(maxInFlight).toBe(1);
  });

  it('stops both simulation clocks on pause and clears work on destroy', async () => {
    const human = new FakeHumanEnvironment();
    const agent = new FakeAgentEnvironment();
    const loop = new DualGameLoop({
      human,
      agent,
      agentRuntime: { outerActionRepeat: 4, stickyActionProbability: 0.25 },
      humanCommand: () => ({ kind: 'discrete', actionIndex: 0 }),
      infer: async () => policyResult(0),
    });

    loop.start();
    await wait(70);
    loop.pause();
    const humanCount = human.actions.length;
    const agentCount = agent.actions.length;
    await wait(70);
    expect(human.actions.length).toBe(humanCount);
    expect(agent.actions.length).toBe(agentCount);

    loop.start();
    await wait(30);
    loop.destroy();
    const destroyedHumanCount = human.actions.length;
    await wait(70);
    expect(human.actions.length).toBe(destroyedHumanCount);
    expect(loop.currentStatus).toBe('paused');
  });
});
