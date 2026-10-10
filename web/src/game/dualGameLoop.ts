import type { HumanEnvironmentStep } from '../environment/aleEnvironment';
import type { PolicyResult } from '../inference/types';
import type { EnvironmentStep } from '../environment/aleEnvironment';
import type { HumanPaddleCommand } from '../input/paddleCommand';

export type LoopStatus = 'idle' | 'running' | 'paused' | 'error';

export interface LoopEnvironment {
  readonly observation: Uint8Array;
  readonly isFinished: boolean;
  readonly currentSeed: number;
  reset(seed?: number, rawFrameRepeat?: number): unknown;
  step(actionIndex: number): EnvironmentStep;
  stepAsync?(actionIndex: number, rawFrameRepeat?: number): Promise<EnvironmentStep>;
  stepInteractiveFrame?(actionIndex: number, rawFrameRepeat: number): EnvironmentStep;
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
  environmentStepMs: number;
  totalDecisionMs: number;
  scheduleWaitMs: number;
}

export interface AgentRuntimeSemantics {
  outerActionRepeat: number;
  stickyActionProbability: number;
  stepMode?: 'interactive-frame' | 'decision';
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
  agentNativeFramesPerSecond: number;
  agentScheduleWaitP50Ms: number;
  agentScheduleWaitP95Ms: number;
  renderCallbackP50Ms: number;
  renderCallbackP95Ms: number;
  agentPresentationFramesConsumed: number;
  agentPresentationBufferedFrames: number;
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
 * asynchronous; expensive decision mode yields through timers independently of
 * presentation. Neither simulation clock catches up with a burst after delay.
 */
export class DualGameLoop {
  private status: LoopStatus = 'idle';
  private humanTimer: ReturnType<typeof setTimeout> | null = null;
  private agentTimer: ReturnType<typeof setTimeout> | null = null;
  private agentFrameHandle: number | null = null;
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
  private lastAgentCompletedAt: number | null = null;
  private agentElapsedBaselineMs = 0;
  private readonly agentScheduleWaits: number[] = [];
  private readonly renderIntervals: number[] = [];
  private lastRenderAt: number | null = null;
  private readonly presentationFrames: Uint8Array[] = [];
  private presentationRgb: Uint8Array | null = null;
  private presentationFramesConsumed = 0;
  private nextPresentationAt = 0;

  get agentPresentationRgb(): Uint8Array | null {
    return this.presentationRgb;
  }

  constructor(private readonly options: DualGameLoopOptions) {}

  get currentStatus(): LoopStatus {
    return this.status;
  }

  get isInFlight(): boolean {
    return this.humanInFlight || this.agentInFlight;
  }

  get runtimeDiagnostics(): DualLoopDiagnostics {
    const elapsedMs = this.elapsedMs();
    const agentElapsedMs = elapsedMs - this.agentElapsedBaselineMs;
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
      agentDecisionsPerSecond: agentElapsedMs > 0 ? (this.agentDecisionCount / agentElapsedMs) * 1000 : 0,
      agentDecisionP50Ms: percentile(this.agentDecisionIntervals, 0.5),
      agentDecisionP95Ms: percentile(this.agentDecisionIntervals, 0.95),
      agentInferenceInFlight: this.agentInFlight,
      agentNativeFramesPerSecond: agentElapsedMs > 0 ? (this.agentRawFrameDelta / agentElapsedMs) * 1000 : 0,
      agentScheduleWaitP50Ms: percentile(this.agentScheduleWaits, 0.5),
      agentScheduleWaitP95Ms: percentile(this.agentScheduleWaits, 0.95),
      renderCallbackP50Ms: percentile(this.renderIntervals, 0.5),
      renderCallbackP95Ms: percentile(this.renderIntervals, 0.95),
      agentPresentationFramesConsumed: this.presentationFramesConsumed,
      agentPresentationBufferedFrames: this.presentationFrames.length,
    };
  }

  start(): void {
    if (this.destroyed || this.status === 'running') return;
    const current = now();
    this.status = 'running';
    this.activeStartedAt = current;
    this.lastRenderAt = null;
    this.lastAgentCompletedAt = null;
    this.lastHumanTickAt = null;
    this.lastAgentDecisionAt = null;
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
    this.presentationFrames.length = 0;
    this.presentationRgb = null;
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

  /** Switch control cadence only after pending work finishes; keep the Human game. */
  async resetAgent(runtime: AgentRuntimeSemantics): Promise<void> {
    this.pause();
    if (this.pendingAgentDecision) await this.pendingAgentDecision;
    if (this.pendingStep) await this.pendingStep;
    this.options.agent.reset(this.options.agent.currentSeed, runtime.outerActionRepeat);
    this.options.agentRuntime = runtime;
    this.agentDecisionCount = 0;
    this.agentRawFrameDelta = 0;
    this.agentDecisionIntervals.length = 0;
    this.lastAgentDecisionAt = null;
    this.lastAgentCompletedAt = null;
    this.agentScheduleWaits.length = 0;
    this.agentElapsedBaselineMs = this.elapsedMs();
    this.presentationFramesConsumed = 0;
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
    if (this.agentTimer !== null || this.agentFrameHandle !== null || this.status !== 'running' || this.destroyed) return;
    if (this.options.agentRuntime.stepMode !== 'decision' && typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      this.agentFrameHandle = window.requestAnimationFrame(() => {
        this.agentFrameHandle = null;
        if (this.status !== 'running' || this.destroyed) return;
        this.startAgentDecision(now());
      });
      return;
    }
    this.agentTimer = setTimeout(() => {
      this.agentTimer = null;
      if (this.status !== 'running' || this.destroyed) return;
      this.startAgentDecision(now());
    }, Math.max(0, delayMs));
  }

  private startAgentDecision(startedAt: number): void {
    if (this.agentInFlight) return;
    if (this.options.agentRuntime.stepMode === 'decision'
      && this.presentationFrames.length > this.options.agentRuntime.outerActionRepeat) {
      this.scheduleAgent(1000 / 60);
      return;
    }
    this.agentDeadline = startedAt + this.agentIntervalMs();
    const promise = this.processAgentDecision(false);
    this.pendingAgentDecision = promise;
    void promise
      .catch((error) => {
        this.fail(error);
      })
      .finally(() => {
        if (this.pendingAgentDecision === promise) this.pendingAgentDecision = null;
        if (this.status === 'running' && !this.destroyed) {
          const completedAt = now();
          // A slow decision never triggers catch-up work, but it also should
          // not incur another full target interval before the next decision.
          if ((this.agentDeadline ?? 0) <= completedAt) this.agentDeadline = completedAt;
          this.scheduleAgent(Math.max(0, (this.agentDeadline ?? completedAt) - completedAt));
        }
        this.emitDiagnostics();
      });
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
      if (previous !== null) recordInterval(this.humanTickIntervals, startedAt - previous);
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
    const scheduleWaitMs = this.lastAgentCompletedAt === null ? 0 : startedAt - this.lastAgentCompletedAt;
    try {
      const inferenceStartedAt = now();
      const policy = await this.options.infer(this.options.agent.observation);
      const inferenceMs = now() - inferenceStartedAt;
      if (!allowWhenPaused && (this.status !== 'running' || this.destroyed)) return;
      const environmentStartedAt = now();
      const environment = this.options.agentRuntime.stepMode !== 'decision' && this.options.agent.stepInteractiveFrame
        ? this.options.agent.stepInteractiveFrame(policy.actionIndex, this.options.agentRuntime.outerActionRepeat)
        : this.options.agent.stepAsync
        ? await this.options.agent.stepAsync(policy.actionIndex, this.options.agentRuntime.outerActionRepeat)
        : this.options.agent.step(policy.actionIndex);
      const environmentStepMs = now() - environmentStartedAt;
      const finishedAt = now();
      const previous = this.lastAgentDecisionAt;
      if (previous !== null) recordInterval(this.agentDecisionIntervals, startedAt - previous);
      recordInterval(this.agentScheduleWaits, scheduleWaitMs);
      this.lastAgentCompletedAt = finishedAt;
      this.lastAgentDecisionAt = startedAt;
      this.agentDecisionCount += 1;
      this.agentRawFrameDelta += environment.actualEmulatorFrames;
      if (!allowWhenPaused && this.status === 'running' && this.options.onFrame
        && this.options.agentRuntime.stepMode === 'decision' && environment.presentationFrames) {
        this.presentationFrames.push(...environment.presentationFrames);
      }
      this.options.onAgentStep?.({
        policy,
        environment,
        inferenceMs,
        environmentStepMs,
        totalDecisionMs: finishedAt - startedAt,
        scheduleWaitMs,
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
          this.renderFrame();
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
        this.renderFrame();
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
    if (this.agentFrameHandle !== null && typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
      window.cancelAnimationFrame(this.agentFrameHandle);
    }
    if (this.renderHandle !== null) {
      if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function' && typeof this.renderHandle === 'number') {
        window.cancelAnimationFrame(this.renderHandle);
      } else {
        clearTimeout(this.renderHandle);
      }
    }
    this.humanTimer = null;
    this.agentTimer = null;
    this.agentFrameHandle = null;
    this.renderHandle = null;
  }

  private renderFrame(): void {
    const current = now();
    if (this.lastRenderAt !== null) recordInterval(this.renderIntervals, current - this.lastRenderAt);
    this.lastRenderAt = current;
    // Consume at most one real emulator frame per 60 Hz presentation interval.
    const expected = this.presentationRgb === null ? current : this.nextPresentationAt;
    if (this.presentationFrames.length && current + 1 >= expected) {
      this.presentationRgb = this.presentationFrames.shift()!;
      this.nextPresentationAt = current - expected > 1000 / 60 ? current + 1000 / 60 : expected + 1000 / 60;
      this.presentationFramesConsumed += 1;
    }
    this.options.onFrame?.();
  }

  private fail(error: unknown): void {
    this.clearScheduledWork();
    this.presentationFrames.length = 0;
    this.presentationRgb = null;
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
    this.lastAgentCompletedAt = null;
    this.agentElapsedBaselineMs = 0;
    this.agentScheduleWaits.length = 0;
    this.renderIntervals.length = 0;
    this.lastRenderAt = null;
    this.presentationFramesConsumed = 0;
  }

  private elapsedMs(): number {
    return this.accumulatedActiveMs + (this.activeStartedAt === null ? 0 : now() - this.activeStartedAt);
  }

  private humanIntervalMs(): number {
    return 1000 / (this.options.humanTargetFps ?? 60);
  }

  private agentIntervalMs(): number {
    const target = this.options.agentTargetFps ?? 15;
    return 1000 / (this.options.agentRuntime.stepMode === 'decision'
      ? Math.min(target, 60 / this.options.agentRuntime.outerActionRepeat) : target);
  }

  private emitDiagnostics(): void {
    this.options.onDiagnostics?.(this.runtimeDiagnostics);
  }
}

// Percentiles describe the last 600 intervals without growing during long play.
function recordInterval(samples: number[], interval: number): void {
  samples.push(interval);
  if (samples.length > 600) samples.shift();
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
