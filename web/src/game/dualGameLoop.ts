import type { PolicyResult } from '../inference/types';
import type { EnvironmentStep } from '../environment/aleEnvironment';

export type LoopStatus = 'idle' | 'running' | 'paused' | 'error';

export interface LoopEnvironment {
  readonly observation: Uint8Array;
  readonly isFinished: boolean;
  readonly currentSeed: number;
  reset(seed?: number): unknown;
  step(actionIndex: number): EnvironmentStep;
}

export interface AgentLoopStep {
  policy: PolicyResult;
  environment: EnvironmentStep;
  inferenceMs: number;
  totalDecisionMs: number;
}

export interface DualGameLoopOptions {
  human: LoopEnvironment;
  agent: LoopEnvironment;
  humanAction: () => number;
  infer: (observation: Uint8Array) => Promise<PolicyResult>;
  onHumanStep?: (step: EnvironmentStep) => void;
  onAgentStep?: (step: AgentLoopStep) => void;
  onFrame?: () => void;
  onError?: (error: unknown) => void;
  targetFrameMs?: number;
}

export class DualGameLoop {
  private status: LoopStatus = 'idle';
  private scheduledTimer: number | null = null;
  private inFlight = false;
  private pendingStep: Promise<void> | null = null;

  constructor(private readonly options: DualGameLoopOptions) {}

  get currentStatus(): LoopStatus {
    return this.status;
  }

  get isInFlight(): boolean {
    return this.inFlight;
  }

  start(): void {
    if (this.status === 'running') return;
    this.status = 'running';
    this.schedule(0);
  }

  pause(): void {
    if (this.scheduledTimer !== null) {
      window.clearTimeout(this.scheduledTimer);
      this.scheduledTimer = null;
    }
    if (this.status === 'running') this.status = 'paused';
  }

  async reset(): Promise<void> {
    this.pause();
    if (this.pendingStep) await this.pendingStep;
    this.options.human.reset(this.options.human.currentSeed);
    this.options.agent.reset(this.options.agent.currentSeed);
    this.status = 'idle';
    this.options.onFrame?.();
  }

  async stepOnce(): Promise<void> {
    if (this.inFlight) return this.pendingStep ?? Promise.resolve();
    const step = this.processStep();
    this.pendingStep = step;
    await step;
  }

  private async processStep(): Promise<void> {
    if (this.inFlight) return;
    this.inFlight = true;
    const startedAt = now();
    try {
      if (!this.options.human.isFinished) {
        const humanStep = this.options.human.step(this.options.humanAction());
        this.options.onHumanStep?.(humanStep);
      }
      if (!this.options.agent.isFinished) {
        const inferenceStartedAt = now();
        const policy = await this.options.infer(this.options.agent.observation);
        const environment = this.options.agent.step(policy.actionIndex);
        this.options.onAgentStep?.({
          policy,
          environment,
          inferenceMs: now() - inferenceStartedAt,
          totalDecisionMs: now() - startedAt,
        });
      }
      this.options.onFrame?.();
    } catch (error) {
      this.status = 'error';
      this.options.onError?.(error);
      throw error;
    } finally {
      this.inFlight = false;
      this.pendingStep = null;
      if (this.status === 'running') this.schedule(Math.max(0, (this.options.targetFrameMs ?? 1000 / 60) - (now() - startedAt)));
    }
  }

  private schedule(delayMs: number): void {
    if (this.scheduledTimer !== null || this.status !== 'running') return;
    this.scheduledTimer = window.setTimeout(() => {
      this.scheduledTimer = null;
      void this.stepOnce().catch(() => undefined);
    }, delayMs);
  }
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
