import type { HumanEnvironmentStep } from '../environment/aleEnvironment';
import type { PolicyResult } from '../inference/types';
import type { EnvironmentStep } from '../environment/aleEnvironment';
import type { HumanPaddleCommand } from '../input/paddleCommand';

export type LoopStatus = 'idle' | 'running' | 'paused' | 'error';

export interface LoopEnvironment {
  readonly observation: Uint8Array;
  readonly isFinished: boolean;
  readonly currentSeed: number;
  reset(seed?: number): unknown;
  step(actionIndex: number): EnvironmentStep;
  stepAsync?(actionIndex: number): Promise<EnvironmentStep>;
}

export interface HumanLoopEnvironment {
  readonly isFinished: boolean;
  readonly currentSeed: number;
  reset(seed?: number): unknown;
  step(actionIndex: number): HumanEnvironmentStep;
  stepPaddle(command: HumanPaddleCommand): HumanEnvironmentStep;
}

export type HumanLoopCommand =
  | { kind: 'discrete'; actionIndex: number }
  | { kind: 'paddle'; command: HumanPaddleCommand };

export interface AgentLoopStep {
  policy: PolicyResult;
  environment: EnvironmentStep;
  inferenceMs: number;
  totalDecisionMs: number;
}

export interface AgentRuntimeSemantics {
  outerActionRepeat: number;
  stickyActionProbability: number;
}

export interface DualLoopDiagnostics {
  status: LoopStatus;
  elapsedMs: number;
  humanRawFrameRepeat: 1;
  humanStickyActionProbability: 0;
  humanSchedulerTargetFps: number;
  humanRawTickCount: number;
  humanRawFrameDelta: number;
  humanRawFps: number;
  humanTickP50Ms: number;
  humanTickP95Ms: number;
  lateHumanTicks: number;
  droppedHumanTicks: number;
  catchUpBursts: number;
  agentOuterActionRepeat: number;
  agentStickyActionProbability: number;
  agentDecisionCount: number;
  agentRawFrameDelta: number;
  agentDecisionsPerSecond: number;
  agentDecisionP50Ms: number;
  agentDecisionP95Ms: number;
  agentInferenceInFlight: boolean;
}

export interface DualGameLoopOptions {
  human: HumanLoopEnvironment;
  agent: LoopEnvironment;
  agentRuntime: AgentRuntimeSemantics;
  humanCommand: () => HumanLoopCommand;
  infer: (observation: Uint8Array) => Promise<PolicyResult>;
  onHumanStep?: (step: HumanEnvironmentStep) => void;
  onAgentStep?: (step: AgentLoopStep) => void;
  onFrame?: () => void;
  onDiagnostics?: (diagnostics: DualLoopDiagnostics) => void;
  onError?: (error: unknown) => void;
  humanTargetFps?: number;
  agentTargetFps?: number;
}

/**
 * Coordinates two simulation clocks and one presentation clock.
 *
 * Human ticks are synchronous one-raw-frame steps. Agent decisions are
 * asynchronous and may take longer than one display frame. Neither clock is
 * allowed to catch up with a burst after a late callback.
 */
export class DualGameLoop {
  private status: LoopStatus = 'idle';
  private humanTimer: ReturnType<typeof setTimeout> | null = null;
  private agentTimer: ReturnType<typeof setTimeout> | null = null;
  private renderHandle: number | ReturnType<typeof setTimeout> | null = null;
  private humanDeadline: number | null = null;
  private agentDeadline: number | null = null;
  private activeStartedAt: number | null = null;
  private accumulatedActiveMs = 0;
  private humanInFlight = false;
  private agentInFlight = false;
  private pendingAgentDecision: Promise<void> | null = null;
  private pendingStep: Promise<void> | null = null;
  private destroyed = false;
  private humanRawTickCount = 0;
  private humanRawFrameDelta = 0;
  private agentDecisionCount = 0;
  private agentRawFrameDelta = 0;
  private lateHumanTicks = 0;
  private droppedHumanTicks = 0;
  private catchUpBursts = 0;
  private readonly humanTickIntervals: number[] = [];
  private readonly agentDecisionIntervals: number[] = [];
  private lastHumanTickAt: number | null = null;
  private lastAgentDecisionAt: number | null = null;

  constructor(private readonly options: DualGameLoopOptions) {}

  get currentStatus(): LoopStatus {
    return this.status;
  }

  get isInFlight(): boolean {
    return this.humanInFlight || this.agentInFlight;
  }

  get runtimeDiagnostics(): DualLoopDiagnostics {
    const elapsedMs = this.elapsedMs();
    return {
      status: this.status,
      elapsedMs,
      humanRawFrameRepeat: 1,
      humanStickyActionProbability: 0,
      humanSchedulerTargetFps: this.options.humanTargetFps ?? 60,
      humanRawTickCount: this.humanRawTickCount,
      humanRawFrameDelta: this.humanRawFrameDelta,
      humanRawFps: elapsedMs > 0 ? (this.humanRawFrameDelta / elapsedMs) * 1000 : 0,
      humanTickP50Ms: percentile(this.humanTickIntervals, 0.5),
      humanTickP95Ms: percentile(this.humanTickIntervals, 0.95),
      lateHumanTicks: this.lateHumanTicks,
      droppedHumanTicks: this.droppedHumanTicks,
      catchUpBursts: this.catchUpBursts,
      agentOuterActionRepeat: this.options.agentRuntime.outerActionRepeat,
      agentStickyActionProbability: this.options.agentRuntime.stickyActionProbability,
      agentDecisionCount: this.agentDecisionCount,
      agentRawFrameDelta: this.agentRawFrameDelta,
      agentDecisionsPerSecond: elapsedMs > 0 ? (this.agentDecisionCount / elapsedMs) * 1000 : 0,
      agentDecisionP50Ms: percentile(this.agentDecisionIntervals, 0.5),
      agentDecisionP95Ms: percentile(this.agentDecisionIntervals, 0.95),
      agentInferenceInFlight: this.agentInFlight,
    };
  }

  start(): void {
    if (this.destroyed || this.status === 'running') return;
    const current = now();
    this.status = 'running';
    this.activeStartedAt = current;
    this.humanDeadline = current;
    this.agentDeadline = current;
    this.scheduleHuman(0);
    this.scheduleAgent(0);
    if (this.options.onFrame) this.scheduleRender();
    this.emitDiagnostics();
  }

  pause(): void {
    // A decision already inside ALE is allowed to finish atomically; clearing
    // the schedules prevents any new Human/Agent tick, and avoids leaving the
    // formal Agent frame stack half-updated between its four raw frames.
    this.clearScheduledWork();
    if (this.status === 'running') {
      this.accumulatedActiveMs += now() - (this.activeStartedAt ?? now());
      this.activeStartedAt = null;
      this.status = 'paused';
    }
    this.emitDiagnostics();
  }

  async reset(): Promise<void> {
    this.pause();
    if (this.pendingAgentDecision) await this.pendingAgentDecision;
    if (this.pendingStep) await this.pendingStep;
    this.options.human.reset(this.options.human.currentSeed);
    this.options.agent.reset(this.options.agent.currentSeed);
    this.resetMetrics();
    this.status = 'idle';
    this.options.onFrame?.();
    this.emitDiagnostics();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.pause();
    this.destroyed = true;
    this.clearScheduledWork();
    this.emitDiagnostics();
  }

  /** Deterministic one-step seam retained for unit tests and diagnostics. */
  async stepOnce(): Promise<void> {
    if (this.destroyed) return;
    if (this.pendingStep) return this.pendingStep;
    const step = (async () => {
      try {
        this.processHumanTick(now());
        await this.processAgentDecision(true);
        this.options.onFrame?.();
        this.emitDiagnostics();
      } catch (error) {
        this.fail(error);
        throw error;
      }
    })();
    this.pendingStep = step;
    try {
      await step;
    } finally {
      if (this.pendingStep === step) this.pendingStep = null;
    }
  }

  private scheduleHuman(delayMs: number): void {
    if (this.humanTimer !== null || this.status !== 'running' || this.destroyed) return;
    this.humanTimer = setTimeout(() => {
      this.humanTimer = null;
      if (this.status !== 'running' || this.destroyed) return;
      const startedAt = now();
      try {
        this.processHumanTick(startedAt);
      } catch (error) {
        this.fail(error);
        return;
      }
      const interval = this.humanIntervalMs();
      this.humanDeadline = startedAt + interval;
      this.scheduleHuman(Math.max(0, this.humanDeadline - now()));
      this.emitDiagnostics();
    }, Math.max(0, delayMs));
  }

  private scheduleAgent(delayMs: number): void {
    if (this.agentTimer !== null || this.status !== 'running' || this.destroyed) return;
    this.agentTimer = setTimeout(() => {
      this.agentTimer = null;
      if (this.status !== 'running' || this.destroyed) return;
      const startedAt = now();
      this.agentDeadline = startedAt + this.agentIntervalMs();
      if (this.agentInFlight) {
        this.scheduleAgent(this.agentIntervalMs());
        return;
      }
      const promise = this.processAgentDecision(false);
      this.pendingAgentDecision = promise;
      void promise
        .catch((error) => {
          this.fail(error);
        })
        .finally(() => {
          if (this.pendingAgentDecision === promise) this.pendingAgentDecision = null;
          if (this.status === 'running' && !this.destroyed) {
            if ((this.agentDeadline ?? 0) <= now()) this.agentDeadline = now() + this.agentIntervalMs();
            this.scheduleAgent(Math.max(0, (this.agentDeadline ?? now()) - now()));
          }
          this.emitDiagnostics();
        });
    }, Math.max(0, delayMs));
  }

  private processHumanTick(startedAt: number): void {
    if (this.humanInFlight || this.options.human.isFinished) return;
    this.humanInFlight = true;
    try {
      const expected = this.humanDeadline ?? startedAt;
      const lateness = startedAt - expected;
      if (lateness > 2) this.lateHumanTicks += 1;
      if (lateness > this.humanIntervalMs()) this.droppedHumanTicks += 1;
      const previous = this.lastHumanTickAt;
      if (previous !== null) this.humanTickIntervals.push(startedAt - previous);
      this.lastHumanTickAt = startedAt;
      const command = this.options.humanCommand();
      const step = command.kind === 'paddle'
        ? this.options.human.stepPaddle(command.command)
        : this.options.human.step(command.actionIndex);
      this.humanRawTickCount += 1;
      this.humanRawFrameDelta += step.actualEmulatorFrames;
      this.options.onHumanStep?.(step);
    } finally {
      this.humanInFlight = false;
    }
  }

  private async processAgentDecision(allowWhenPaused: boolean): Promise<void> {
    if (this.agentInFlight || (!allowWhenPaused && this.status !== 'running') || this.options.agent.isFinished) return;
    this.agentInFlight = true;
    const startedAt = now();
    try {
      const inferenceStartedAt = now();
      const policy = await this.options.infer(this.options.agent.observation);
      if (!allowWhenPaused && (this.status !== 'running' || this.destroyed)) return;
      const environment = this.options.agent.stepAsync
        ? await this.options.agent.stepAsync(policy.actionIndex)
        : this.options.agent.step(policy.actionIndex);
      const finishedAt = now();
      const previous = this.lastAgentDecisionAt;
      if (previous !== null) this.agentDecisionIntervals.push(startedAt - previous);
      this.lastAgentDecisionAt = startedAt;
      this.agentDecisionCount += 1;
      this.agentRawFrameDelta += environment.actualEmulatorFrames;
      this.options.onAgentStep?.({
        policy,
        environment,
        inferenceMs: finishedAt - inferenceStartedAt,
        totalDecisionMs: finishedAt - startedAt,
      });
    } finally {
      this.agentInFlight = false;
    }
  }

  private scheduleRender(): void {
    if (this.renderHandle !== null || this.status !== 'running' || this.destroyed || !this.options.onFrame) return;
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      this.renderHandle = window.requestAnimationFrame(() => {
        this.renderHandle = null;
        if (this.status !== 'running' || this.destroyed) return;
        try {
          this.options.onFrame?.();
        } catch (error) {
          this.fail(error);
          return;
        }
        this.scheduleRender();
      });
      return;
    }
    this.renderHandle = setTimeout(() => {
      this.renderHandle = null;
      if (this.status !== 'running' || this.destroyed) return;
      try {
        this.options.onFrame?.();
      } catch (error) {
        this.fail(error);
        return;
      }
      this.scheduleRender();
    }, 1000 / 60);
  }

  private clearScheduledWork(): void {
    if (this.humanTimer !== null) clearTimeout(this.humanTimer);
    if (this.agentTimer !== null) clearTimeout(this.agentTimer);
    if (this.renderHandle !== null) {
      if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function' && typeof this.renderHandle === 'number') {
        window.cancelAnimationFrame(this.renderHandle);
      } else {
        clearTimeout(this.renderHandle);
      }
    }
    this.humanTimer = null;
    this.agentTimer = null;
    this.renderHandle = null;
  }

  private fail(error: unknown): void {
    this.clearScheduledWork();
    this.status = 'error';
    this.options.onError?.(error);
    this.emitDiagnostics();
  }

  private resetMetrics(): void {
    this.accumulatedActiveMs = 0;
    this.activeStartedAt = null;
    this.humanDeadline = null;
    this.agentDeadline = null;
    this.humanRawTickCount = 0;
    this.humanRawFrameDelta = 0;
    this.agentDecisionCount = 0;
    this.agentRawFrameDelta = 0;
    this.lateHumanTicks = 0;
    this.droppedHumanTicks = 0;
    this.catchUpBursts = 0;
    this.humanTickIntervals.length = 0;
    this.agentDecisionIntervals.length = 0;
    this.lastHumanTickAt = null;
    this.lastAgentDecisionAt = null;
  }

  private elapsedMs(): number {
    return this.accumulatedActiveMs + (this.activeStartedAt === null ? 0 : now() - this.activeStartedAt);
  }

  private humanIntervalMs(): number {
    return 1000 / (this.options.humanTargetFps ?? 60);
  }

  private agentIntervalMs(): number {
    return 1000 / (this.options.agentTargetFps ?? 15);
  }

  private emitDiagnostics(): void {
    this.options.onDiagnostics?.(this.runtimeDiagnostics);
  }
}

function percentile(values: readonly number[], fraction: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(fraction * sorted.length) - 1));
  return sorted[index] ?? 0;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
