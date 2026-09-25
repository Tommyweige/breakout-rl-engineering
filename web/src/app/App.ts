import {
  ACTION_MEANINGS,
  type BrowserValidationArtifact,
  type Day28BenchmarkArtifact,
  type Day28ValidationArtifact,
  type FixtureValidationResult,
  type InferenceBackend,
  type PolicyResult,
  type RuntimeStatus,
  type WebGpuSupport,
} from '../inference/types';
import { InferenceScheduler } from '../inference/InferenceScheduler';
import { AgentInferenceWorker } from '../inference/AgentInferenceWorker';
import { OrtWebPolicy } from '../inference/OrtWebPolicy';
import { detectBrowser } from '../inference/browserInfo';
import { detectWebGpuSupport } from '../inference/webgpuSupport';
import { KeyboardController } from '../input/KeyboardController';
import { MouseController, type MouseMotionState } from '../input/MouseController';
import type { HumanPaddleCommand } from '../input/paddleCommand';
import { DifficultyPolicy, AI_DIFFICULTY_LABELS, isAiDifficulty, type AiDifficulty } from './difficultyPolicy';
import {
  BrowserBreakoutEnvironment,
  HumanBreakoutEnvironment,
  type EnvironmentStep,
  type HumanEnvironmentStep,
} from '../environment/aleEnvironment';
import { loadBreakoutContract, type BreakoutContractV2 } from '../environment/breakoutContract';
import { DualGameLoop, type AgentLoopStep, type DualLoopDiagnostics, type HumanLoopCommand } from '../game/dualGameLoop';
import { browserEvaluationSeeds, runPolicyEvaluation, type BrowserPolicyEvaluationArtifact } from '../evaluation/policyEvaluation';
import { buildBrowserValidationArtifact, buildDay28ValidationArtifact } from '../validation/buildArtifact';
import { runBrowserBackendComparison } from '../validation/runBrowserBenchmark';
import { runFixtureValidation } from '../validation/runFixtureValidation';
import { reduceRuntimeStatus, type RuntimeEvent } from './runtimeState';
import { renderAppShell, renderPolicyQValuesMarkup, renderQValuesMarkup } from './view';

declare global {
  interface Window {
    __day27Validation?: BrowserValidationArtifact;
    __day28Validation?: Day28ValidationArtifact;
    __day28Benchmark?: Day28BenchmarkArtifact;
    __day29Evaluation?: BrowserPolicyEvaluationArtifact;
    __day29Ready?: boolean;
    __day29Evaluate?: (seeds?: readonly number[]) => Promise<BrowserPolicyEvaluationArtifact | null>;
    __day29CapturePreprocessingTrace?: (seed: number, actions?: readonly number[]) => Promise<unknown>;
    __day29EvaluationSeeds?: readonly number[];
    __day29EnvironmentDiagnostics?: Record<string, unknown>;
    __day30Evaluation?: BrowserPolicyEvaluationArtifact;
    __day30Ready?: boolean;
    __day30Evaluate?: (seeds?: readonly number[]) => Promise<BrowserPolicyEvaluationArtifact | null>;
    __day30EvaluationSeeds?: readonly number[];
    __day30EnvironmentDiagnostics?: Record<string, unknown>;
    __mouseControlV3Diagnostics?: MouseControlV3Diagnostics;
    __humanInteractiveDiagnostics?: HumanInteractiveDiagnostics;
  }
}

type HumanInputMode = 'keyboard' | 'mouse';
type HumanAction = (typeof ACTION_MEANINGS)[number];

interface HumanActionSample {
  decision: number;
  timestampMs: number;
  targetX: number | null;
  targetChangedAtMs: number | null;
  paddleCenterX: number | null;
  positionError: number | null;
  requestedAction: HumanAction;
  executedAction: HumanAction;
  requestedDirection: HumanAction;
  executedDirection: HumanAction;
  requestedPaddlePositionX: number | null;
  appliedPaddleTargetX: number | null;
  action: HumanAction;
  rawFrameNumber: number;
  actualEmulatorFrames: number;
}

interface MouseControlV3Diagnostics {
  cursorTargetX: number | null;
  paddleCenterX: number | null;
  motionState: MouseMotionState;
  positionError: number | null;
  requestedDirection: HumanPaddleCommand['direction'];
  requestedPaddlePositionX: number | null;
  executedDirection: HumanAction;
  appliedPaddleTargetX: number | null;
  rawFrameNumber: number;
  actualEmulatorFrames: number;
  executedHumanAction: HumanAction;
  decisionCount: number;
  actionHistory: HumanActionSample[];
}

interface HumanInteractiveDiagnostics {
  schemaVersion: 1;
  runtimeContract: Record<string, unknown>;
  runtime: DualLoopDiagnostics | null;
  samples: HumanActionSample[];
}

export class App {
  private readonly debug = new URL(window.location.href).searchParams.get('debug') === '1';
  private policy: OrtWebPolicy;
  private readonly scheduler: InferenceScheduler<Uint8Array, PolicyResult>;
  private readonly keyboard = new KeyboardController();
  private readonly mouse = new MouseController();
  private readonly difficultyPolicy = new DifficultyPolicy();
  private status: RuntimeStatus = 'idle';
  private selectedBackend: InferenceBackend = 'wasm';
  private gameplayBackend: InferenceBackend | null = null;
  private gameplayInferenceWorker: AgentInferenceWorker | null = null;
  private inputMode: HumanInputMode = 'keyboard';
  private mounted = false;
  private busy = false;
  private inputTimer: number | null = null;
  private downloadUrl: string | null = null;
  private benchmarkDownloadUrl: string | null = null;
  private evaluationDownloadUrl: string | null = null;
  private unsubscribeScheduler: (() => void) | null = null;
  private webgpuSupport: WebGpuSupport | null = null;
  private validationResult: FixtureValidationResult | null = null;
  private contract: BreakoutContractV2 | null = null;
  private humanEnvironment: HumanBreakoutEnvironment | null = null;
  private agentEnvironment: BrowserBreakoutEnvironment | null = null;
  private gameLoop: DualGameLoop | null = null;
  private gameplayInitPromise: Promise<void> | null = null;
  private latestHumanStep: HumanEnvironmentStep | null = null;
  private latestAgentStep: AgentLoopStep | null = null;
  private agentAutoFireCount = 0;
  private lastHumanAction: HumanAction = 'NOOP';

  constructor(private readonly root: HTMLElement) {
    this.policy = new OrtWebPolicy({ backend: this.selectedBackend });
    this.scheduler = new InferenceScheduler((observation) => this.policy.infer(observation));
  }

  mount(): void {
    if (this.mounted) return;
    this.mounted = true;
    this.root.innerHTML = `<main class="app-shell${this.debug ? ' is-debug' : ''}">${renderAppShell(this.debug)}</main>`;

    this.keyboard.attach();
    this.mouse.attach(this.required<HTMLCanvasElement>('[data-role="human-canvas"]'));
    window.addEventListener('blur', this.onWindowBlur);
    this.setInputMode('keyboard');
    this.inputTimer = window.setInterval(() => this.renderHumanInput(), 80);
    this.unsubscribeScheduler = this.scheduler.subscribe((status) => {
      if (this.mounted && this.debug) this.setText('[data-role="scheduler-status"]', status);
    });

    this.button('start').addEventListener('click', () => {
      this.dispatch('start');
      this.setUserMessage('Starting both games…');
      void this.startGameplay();
    });
    this.button('pause').addEventListener('click', () => {
      this.dispatch('pause');
      this.clearHumanInput();
      this.gameLoop?.pause();
      this.renderCanvases();
      this.setUserMessage('Both games are paused. Press Start to resume.');
    });
    this.button('reset').addEventListener('click', () => {
      this.dispatch('reset');
      this.clearHumanInput();
      this.setUserMessage('Restarting both games…');
      void this.resetGameplay();
    });
    if (this.debug) {
      this.button('validate').addEventListener('click', () => void this.validateSelectedBackend());
      this.button('benchmark').addEventListener('click', () => void this.runBenchmark());
      this.button('evaluate').addEventListener('click', () => void this.runEvaluation(undefined, 50));
      this.select('backend').addEventListener('change', (event) => {
        void this.changeBackend((event.target as HTMLSelectElement).value);
      });
    }
    this.select('input-mode').addEventListener('change', (event) => {
      this.setInputMode((event.target as HTMLSelectElement).value);
    });
    this.select('difficulty').addEventListener('change', (event) => {
      this.setDifficulty((event.target as HTMLSelectElement).value);
    });
    if (this.debug) {
      window.__day29Evaluate = (seeds) => this.runEvaluation(seeds);
      window.__day30Evaluate = (seeds) => this.runEvaluation(seeds, 50);
      window.__day29CapturePreprocessingTrace = (seed, actions) => this.capturePreprocessingTrace(seed, actions);
      window.__day29EvaluationSeeds = undefined;
      window.__day30EvaluationSeeds = undefined;
      this.updateMouseDiagnostics();
      void this.refreshWebGpuSupport();
    }
  }

  destroy(): void {
    if (!this.mounted) return;
    this.gameLoop?.destroy();
    window.removeEventListener('blur', this.onWindowBlur);
    this.humanEnvironment?.dispose();
    this.agentEnvironment?.dispose();
    this.keyboard.detach();
    this.mouse.detach();
    if (this.inputTimer !== null) window.clearInterval(this.inputTimer);
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    if (this.benchmarkDownloadUrl) URL.revokeObjectURL(this.benchmarkDownloadUrl);
    if (this.evaluationDownloadUrl) URL.revokeObjectURL(this.evaluationDownloadUrl);
    this.unsubscribeScheduler?.();
    void this.gameplayInferenceWorker?.release();
    void this.policy.release().catch(() => undefined);
    this.inputTimer = null;
    this.downloadUrl = null;
    this.benchmarkDownloadUrl = null;
    this.evaluationDownloadUrl = null;
    this.unsubscribeScheduler = null;
    window.__day29Ready = false;
    window.__day30Ready = false;
    delete window.__day29Evaluate;
    delete window.__day30Evaluate;
    delete window.__day29CapturePreprocessingTrace;
    delete window.__day29EvaluationSeeds;
    delete window.__day29EnvironmentDiagnostics;
    delete window.__day29Evaluation;
    delete window.__day30Evaluation;
    delete window.__day30EvaluationSeeds;
    delete window.__day30EnvironmentDiagnostics;
    delete window.__mouseControlV3Diagnostics;
    delete window.__humanInteractiveDiagnostics;
    this.mounted = false;
  }

  private async validateSelectedBackend(): Promise<void> {
    if (this.busy) return;
    const message = this.required('[data-role="validation-message"]');
    this.setBusy(true);
    this.scheduler.start();
    this.dispatch('load-start');
    this.setValidationStatus('Loading');
    message.textContent = `正在載入 manifest、Day 21 Final Model FP32 ONNX 並以 ${this.selectedBackend.toUpperCase()} 執行固定 fixtures…`;

    try {
      await this.prepareFormalPolicy();
      this.validationResult = await runFixtureValidation(this.policy, {
        infer: (observation) => this.scheduler.run(observation),
      });
      this.renderValidation(this.validationResult);
      this.publishArtifact();
      this.dispatch(this.validationResult.passed ? 'load-success' : 'load-error');
    } catch (error) {
      this.validationResult = null;
      this.scheduler.fail();
      const failure = error instanceof Error ? error.message : String(error);
      const unavailable = this.selectedBackend === 'webgpu' && !this.policy.actualBackend;
      this.setValidationStatus(unavailable ? 'Unavailable' : 'ERROR');
      this.setText('[data-role="actual-backend"]', unavailable ? 'UNAVAILABLE' : '—');
      message.textContent = failure;
      this.dispatch('load-error');
    } finally {
      if (this.scheduler.currentStatus === 'running') this.scheduler.reset();
      this.setBusy(false);
    }
  }

  private renderValidation(result: FixtureValidationResult): void {
    this.renderPolicyRuntime(result.actualBackend, result.ortWebVersion, result.browser.name, result.browser.version, result.browser.platform);
    this.setText('[data-role="model-loaded"]', 'loaded / Day 21 canonical');
    this.setText('[data-role="requested-backend"]', result.requestedBackend.toUpperCase());
    this.setText('[data-role="actual-backend"]', result.actualBackend.toUpperCase());
    this.renderWebGpuSupport(result.webgpuSupport);
    this.setText('[data-role="environment-parity"]', `${result.environmentContract.status.toUpperCase()} / inference only`);
    this.setText('[data-role="backend-evidence"]', backendEvidenceLabel(result.backendEvidence));
    this.setText('[data-role="action-agreement"]', `${(result.actionAgreementRate * 100).toFixed(2)}%`);
    this.setText('[data-role="max-error"]', result.maxAbsoluteError.toExponential(3));
    this.setText('[data-role="mean-error"]', result.meanAbsoluteError.toExponential(3));
    this.setText('[data-role="q-margin"]', `${result.qMargin.minMargin.toExponential(3)} / ${result.qMargin.meanMargin.toExponential(3)}`);
    this.setText('[data-role="margin-error"]', result.qMargin.maxAbsoluteMarginError.toExponential(3));
    this.setText('[data-role="disagreements"]', result.disagreementIndices.length ? result.disagreementIndices.join(', ') : 'none');
    this.setText('[data-role="sample-count"]', `${result.sampleCount} samples`);
    this.setText('[data-role="model-sha"]', result.modelSha256);
    this.setText('[data-role="evidence-artifact"]', `day28-${result.actualBackend}-validation.json`);
    this.setText('[data-role="evidence-page"]', pageHost(result.pageUrl));
    this.setText('[data-role="gameplay-backend"]', result.actualBackend.toUpperCase());
    this.setValidationStatus(result.passed ? 'PASS' : 'FAIL');
    this.setText('[data-role="current-action"]', result.representative.selectedAction);
    this.renderQValues(result);
    this.setText('[data-role="validation-message"]', result.passed
      ? `${result.actualBackend.toUpperCase()} fixed-state validation passed on ${result.sampleCount} samples. This confirms model I/O, not gameplay quality.`
      : `${result.actualBackend.toUpperCase()} validation failed. disagreement indices: ${result.disagreementIndices.join(', ') || 'none'}`);
  }

  private renderQValues(result: FixtureValidationResult): void {
    const container = this.required('[data-role="gameplay-q-values"]');
    container.innerHTML = renderQValuesMarkup(result);
  }

  private publishArtifact(): void {
    if (!this.validationResult) return;
    const artifact = buildDay28ValidationArtifact(this.validationResult);
    window.__day28Validation = artifact;
    if (this.validationResult.actualBackend === 'wasm') window.__day27Validation = buildBrowserValidationArtifact(this.validationResult);
    else delete window.__day27Validation;
    const link = this.required<HTMLAnchorElement>('[data-role="download-validation"]');
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    this.downloadUrl = URL.createObjectURL(new Blob([`${JSON.stringify(artifact, null, 2)}\n`], { type: 'application/json' }));
    link.href = this.downloadUrl;
    link.download = `day28-${artifact.actualBackend}-validation.json`;
    link.textContent = `Download ${artifact.actualBackend.toUpperCase()} validation JSON`;
    link.hidden = false;
  }

  private async runBenchmark(): Promise<void> {
    if (this.busy) return;
    const message = this.required('[data-role="validation-message"]');
    this.setBusy(true);
    this.setStatus('loading');
    message.textContent = '正在同一個 Browser session 中執行 WASM / WebGPU fixed-state parity 與 batch=1 benchmark…';
    this.setText('[data-role="benchmark-status"]', 'running');
    try {
      const artifact = await runBrowserBackendComparison({ sampleCount: 100, warmupCount: 10 });
      window.__day28Benchmark = artifact;
      this.renderBenchmark(artifact);
      this.setText('[data-role="validation-message"]', `WASM / WebGPU benchmark completed with ${artifact.sampleCount} measured samples per backend. Initialization and warm-ups were excluded.`);
      this.setText('[data-role="evidence-artifact"]', 'day28-web-benchmark.json');
      this.setText('[data-role="evidence-page"]', pageHost(artifact.pageUrl));
      this.setStatus('ready');
    } catch (error) {
      this.setText('[data-role="benchmark-status"]', 'error');
      message.textContent = error instanceof Error ? error.message : String(error);
      this.setStatus('error');
    } finally {
      this.setBusy(false);
    }
  }

  private renderBenchmark(artifact: Day28BenchmarkArtifact): void {
    const container = this.required('[data-role="benchmark-summary"]');
    container.innerHTML = `
      <div class="benchmark-heading"><span>BENCHMARK / BATCH=1</span><strong data-role="benchmark-status">PASS</strong></div>
      <div class="benchmark-grid">
        <div><span>WASM P50 / P95</span><strong>${formatMs(artifact.results.wasm.summary.p50Ms)} / ${formatMs(artifact.results.wasm.summary.p95Ms)}</strong><small>actual: ${artifact.results.wasm.actualBackend.toUpperCase()}</small></div>
        <div><span>WebGPU P50 / P95</span><strong>${formatMs(artifact.results.webgpu.summary.p50Ms)} / ${formatMs(artifact.results.webgpu.summary.p95Ms)}</strong><small>actual: ${artifact.results.webgpu.actualBackend.toUpperCase()}</small></div>
      </div>
      <p>${artifact.sampleCount} measured samples per backend; ${artifact.warmupCount} warm-ups excluded. Session initialization is outside the timed scope.</p>
    `;
    container.hidden = false;
    const link = this.required<HTMLAnchorElement>('[data-role="download-benchmark"]');
    if (this.benchmarkDownloadUrl) URL.revokeObjectURL(this.benchmarkDownloadUrl);
    this.benchmarkDownloadUrl = URL.createObjectURL(new Blob([`${JSON.stringify(artifact, null, 2)}\n`], { type: 'application/json' }));
    link.href = this.benchmarkDownloadUrl;
    link.hidden = false;
  }

  private async runEvaluation(seeds?: readonly number[], episodeTarget = 30): Promise<BrowserPolicyEvaluationArtifact | null> {
    if (this.busy) return null;
    this.setBusy(true);
    this.gameLoop?.pause();
    this.clearHumanInput();
    this.setText('[data-role="evaluation-status"]', 'running');
    this.setText('[data-role="evaluation-result"]', 'running');
    this.setText('[data-role="evaluation-message"]', `正在用固定 ${episodeTarget} seeds 跑 Browser ALE policy evaluation；每局保存 raw return、length、game-over 與 agent-loop timing。`);
    this.required<HTMLElement>('[data-role="evaluation-summary"]').hidden = false;
    try {
      const contract = await this.getContract();
      await this.prepareFormalPolicy();
      const evaluationSeeds = seeds ?? browserEvaluationSeeds(contract, episodeTarget);
      window.__day29EvaluationSeeds = browserEvaluationSeeds(contract, 30);
      window.__day30EvaluationSeeds = browserEvaluationSeeds(contract, 50);
      const artifact = await runPolicyEvaluation({
        backend: this.selectedBackend,
        policy: this.policy,
        contract,
        seeds: evaluationSeeds,
        modelSha256: this.validationResult?.modelSha256,
        artifactType: episodeTarget >= 50 ? 'day30_browser_policy_evaluation' : 'day29_browser_policy_evaluation',
        onEpisode: (_episode, completed, total) => {
          this.setText('[data-role="evaluation-status"]', `${completed}/${total}`);
        },
      });
      window.__day29Evaluation = artifact;
      window.__day30Evaluation = artifact;
      this.renderEvaluation(artifact);
      this.publishEvaluationArtifact(artifact);
      this.setText('[data-role="validation-message"]', `${artifact.actualBackend.toUpperCase()} ${artifact.episodes.length}-episode Browser evaluation completed. Score and latency are reported separately.`);
      return artifact;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.setText('[data-role="evaluation-status"]', 'error');
      this.setText('[data-role="evaluation-result"]', 'ERROR');
      this.setText('[data-role="evaluation-message"]', message);
      this.setText('[data-role="validation-message"]', message);
      return null;
    } finally {
      this.setBusy(false);
      if (this.status === 'running') this.gameLoop?.start();
    }
  }

  private async capturePreprocessingTrace(seed: number, actions: readonly number[] = [1, 1, 2, 2, 3]): Promise<unknown> {
    const contract = await this.getContract();
    const environment = await BrowserBreakoutEnvironment.create({ contract, seed });
    try {
      environment.enablePreprocessingTrace();
      environment.reset(seed);
      for (const action of actions) {
        if (!Number.isInteger(action) || action < 0 || action > 3) throw new Error(`invalid trace action ${action}`);
        environment.step(action);
      }
      return environment.getPreprocessingTrace();
    } finally {
      environment.dispose();
    }
  }

  private renderEvaluation(artifact: BrowserPolicyEvaluationArtifact): void {
    this.required<HTMLElement>('[data-role="evaluation-summary"]').hidden = false;
    this.setText('[data-role="evaluation-status"]', artifact.completed ? 'PASS' : 'PARTIAL');
    this.setText('[data-role="evaluation-result"]', artifact.completed ? 'PASS' : 'PARTIAL');
    this.setText('[data-role="evaluation-mean"]', `${artifact.aggregate.mean.toFixed(2)} / ${artifact.aggregate.median.toFixed(2)}`);
    this.setText('[data-role="evaluation-p10-p90"]', `${artifact.aggregate.p10.toFixed(2)} / ${artifact.aggregate.p90.toFixed(2)}`);
    this.setText('[data-role="evaluation-min-max"]', `${artifact.aggregate.min.toFixed(2)} / ${artifact.aggregate.max.toFixed(2)}`);
    this.setText('[data-role="evaluation-success"]', `${artifact.aggregate.successCount} / ${artifact.aggregate.crashCount}`);
    this.setText('[data-role="evaluation-message"]', `${artifact.episodes.length} episodes / ${artifact.actualBackend.toUpperCase()} / model ${artifact.modelSha256.slice(0, 12)}… / parity ${artifact.environmentContract.status.toUpperCase()}`);
  }

  private publishEvaluationArtifact(artifact: BrowserPolicyEvaluationArtifact): void {
    const link = this.required<HTMLAnchorElement>('[data-role="download-evaluation"]');
    if (this.evaluationDownloadUrl) URL.revokeObjectURL(this.evaluationDownloadUrl);
    this.evaluationDownloadUrl = URL.createObjectURL(new Blob([`${JSON.stringify(artifact, null, 2)}\n`], { type: 'application/json' }));
    link.href = this.evaluationDownloadUrl;
    const day = artifact.artifactType.startsWith('day30') ? 'day30' : 'day29';
    link.download = `${day}-${artifact.actualBackend}-browser-policy-evaluation.json`;
    link.hidden = false;
  }

  private async startGameplay(): Promise<void> {
    if (this.gameplayInitPromise) return this.gameplayInitPromise;
    if (this.busy) return;
    this.gameplayInitPromise = this.initializeGameplay();
    try {
      await this.gameplayInitPromise;
    } finally {
      this.gameplayInitPromise = null;
    }
  }

  private async initializeGameplay(): Promise<void> {
    try {
      const contract = await this.getContract();
      await this.prepareGameplayPolicy();
      const actualBackend = this.policy.actualBackend;
      if (!actualBackend) throw new Error('The browser policy did not expose an active backend.');
      if (this.gameLoop) {
        this.scheduler.start();
        this.gameLoop.start();
        window.__day29Ready = true;
        window.__day30Ready = true;
        this.setUserMessage('Both games are live.');
        return;
      }
      if (!this.humanEnvironment || !this.agentEnvironment) {
        const seed = contract.concrete_episode_seeds[0] ?? 101;
        [this.humanEnvironment, this.agentEnvironment] = await Promise.all([
          HumanBreakoutEnvironment.create({ contract, seed }),
          BrowserBreakoutEnvironment.create({ contract, seed }),
        ]);
      }
      const environmentDiagnostics = {
        ...this.agentEnvironment.runtimeDiagnostics,
        humanRuntime: this.humanEnvironment.runtimeDiagnostics,
        agentRuntime: this.agentEnvironment.runtimeDiagnostics,
        humanInstanceId: this.humanEnvironment.instanceId,
        agentInstanceId: this.agentEnvironment.instanceId,
        crossOriginIsolated: window.crossOriginIsolated,
        dualInstanceCount: new Set([this.humanEnvironment.instanceId, this.agentEnvironment.instanceId]).size,
        preferredBackend: this.webgpuSupport?.supported ? 'webgpu' : 'wasm',
        actualGameplayBackend: this.gameplayBackend,
        gracefulFallback: this.gameplayBackend !== 'webgpu' && this.webgpuSupport?.supported === true,
      };
      if (this.debug) {
        const browser = await detectBrowser();
        this.renderPolicyRuntime(actualBackend, this.policy.ortWebVersion, browser.name, browser.version, browser.platform);
        this.setText('[data-role="environment-parity"]', `${contract.parity.status.toUpperCase()} / Contract v2`);
        this.setText('[data-role="backend-evidence"]', this.policy.backendEvidence === 'webgpu_session_exposes_env_webgpu_device' ? 'runtime GPU device observed' : 'explicit WASM session');
        this.setText('[data-role="model-loaded"]', 'loaded / Day 21 canonical');
        this.setText('[data-role="gameplay-backend"]', actualBackend.toUpperCase());
        this.setText('[data-role="model-sha"]', this.validationResult?.modelSha256 ?? 'manifest hash recorded by evaluation');
        window.__day29EnvironmentDiagnostics = environmentDiagnostics;
        window.__day30EnvironmentDiagnostics = environmentDiagnostics;
      }
      this.agentAutoFireCount = 0;
      this.latestHumanStep = null;
      this.latestAgentStep = null;
      this.initializeHumanDiagnostics();
      this.gameLoop = new DualGameLoop({
        human: this.humanEnvironment,
        agent: this.agentEnvironment,
        agentRuntime: {
          outerActionRepeat: this.agentEnvironment.contract.frame_skip,
          stickyActionProbability: this.agentEnvironment.contract.sticky_action_probability,
        },
        humanCommand: () => this.currentHumanCommand(),
        infer: async (observation) => this.difficultyPolicy.select(
          await (this.gameplayInferenceWorker?.infer(observation) ?? this.scheduler.run(observation)),
        ),
        onHumanStep: (step) => {
          this.latestHumanStep = step;
          this.recordHumanStep(step);
          this.renderHumanStep(step);
        },
        onAgentStep: (step) => {
          this.latestAgentStep = step;
          if (step.environment.autoFire) this.agentAutoFireCount += 1;
          this.renderAgentStep(step);
        },
        onFrame: () => this.renderCanvases(),
        onDiagnostics: (diagnostics) => this.updateHumanRuntimeDiagnostics(diagnostics),
        onError: (error) => this.reportRuntimeError(error),
        // Chrome timer/render overhead is measurable on the production page;
        // this deadline keeps observed Human raw cadence near the 60 Hz Atari
        // target without changing the Human environment's one-frame semantics.
        humanTargetFps: 80,
        agentTargetFps: 15,
      });
      this.updateHumanRuntimeDiagnostics(this.gameLoop.runtimeDiagnostics);
      this.scheduler.start();
      this.gameLoop.start();
      window.__day29Ready = true;
      window.__day30Ready = true;
      this.setUserMessage('Both games are live.');
      this.renderCanvases();
    } catch (error) {
      this.setStatus('error');
      this.reportRuntimeError(error);
    }
  }

  private async resetGameplay(): Promise<void> {
    if (!this.gameLoop) return;
    try {
      await this.gameLoop.reset();
      this.latestHumanStep = null;
      this.latestAgentStep = null;
      this.agentAutoFireCount = 0;
      this.initializeHumanDiagnostics();
      this.renderCanvases();
      this.setStatus('ready');
      this.setUserMessage('Both games restarted. Press Start to play again.');
    } catch (error) {
      this.setStatus('error');
      this.reportRuntimeError(error);
    }
  }

  private renderCanvases(): void {
    if (!this.humanEnvironment || !this.agentEnvironment) return;
    if (!this.latestHumanStep) this.mouse.updatePaddleCenterFromFrame(this.humanEnvironment.rawRgb);
    this.humanEnvironment.render(this.required<HTMLCanvasElement>('[data-role="human-canvas"]'));
    this.agentEnvironment.render(this.required<HTMLCanvasElement>('[data-role="agent-canvas"]'));
    const loopState = this.gameLoop?.currentStatus ?? 'idle';
    this.setText('[data-role="human-stage-state"]', displayStageState(this.humanEnvironment.isFinished, loopState));
    this.setText('[data-role="agent-stage-state"]', displayStageState(this.agentEnvironment.isFinished, loopState));
    this.setText('[data-role="human-state"]', displayCardState(this.humanEnvironment.isFinished, loopState));
    this.setText('[data-role="agent-state"]', displayCardState(this.agentEnvironment.isFinished, loopState));
    this.renderMouseTargetMarker();
  }

  private renderHumanStep(step: HumanEnvironmentStep): void {
    this.mouse.updatePaddleCenterFromFrame(step.rawRgb);
    this.setText('[data-role="human-score"]', step.episodeReturn.toFixed(0));
    this.setText('[data-role="human-lives"]', `${step.lives}`);
    this.setText('[data-role="human-state"]', step.terminated || step.truncated ? 'Game over' : 'Playing');
    this.setText('[data-role="human-stage-state"]', step.terminated || step.truncated ? 'GAME OVER' : 'PLAYING');
    if (this.debug) {
      this.setText('[data-role="human-requested-action"]', step.requestedDirection);
      this.setText('[data-role="human-executed-action"]', step.executedAction);
      this.setText('[data-role="requested-paddle-position"]', formatNormalized(step.requestedPaddlePositionX));
      this.setText('[data-role="applied-paddle-target"]', formatNormalized(step.appliedPaddleTargetX));
      this.setText('[data-role="human-raw-frame-number"]', `${step.rawFrameNumber}`);
      this.setText('[data-role="human-actual-emulator-frames"]', `${step.actualEmulatorFrames}`);
      this.setText('[data-role="human-frame-repeat"]', `${step.actualEmulatorFrames}`);
      this.setText('[data-role="human-sticky"]', `${step.stickyActionProbability}`);
      this.updateHumanRuntimeDiagnostics(this.gameLoop?.runtimeDiagnostics ?? null);
    }
  }

  private renderAgentStep(step: AgentLoopStep): void {
    const environment = step.environment;
    this.setText('[data-role="agent-score"]', environment.episodeReturn.toFixed(0));
    this.setText('[data-role="agent-lives"]', `${environment.lives}`);
    this.setText('[data-role="agent-state"]', environment.terminated || environment.truncated ? 'Game over' : 'Playing');
    this.setText('[data-role="agent-stage-state"]', environment.terminated || environment.truncated ? 'GAME OVER' : 'PLAYING');
    if (this.debug) {
      this.setText('[data-role="current-action"]', environment.autoFire ? `FIRE / auto-${environment.autoFireReason}` : environment.executedAction);
      this.setText('[data-role="debug-difficulty"]', AI_DIFFICULTY_LABELS[this.difficultyPolicy.currentDifficulty]);
      this.setText('[data-role="difficulty-rate"]', `${((step.policy.mistakeRate ?? 0) * 100).toFixed(0)}%`);
      this.setText('[data-role="greedy-action"]', ACTION_MEANINGS[step.policy.greedyActionIndex ?? step.policy.actionIndex] ?? step.policy.action);
      this.setText('[data-role="mistake-injected"]', step.policy.mistakeInjected ? 'yes' : 'no');
      this.setText('[data-role="episode-return"]', environment.episodeReturn.toFixed(0));
      this.setText('[data-role="inference-latency"]', `${step.inferenceMs.toFixed(3)} ms`);
      this.setText('[data-role="agent-frame"]', `${environment.frameNumber} / ${environment.agentStep}`);
      this.setText('[data-role="auto-fire"]', `${this.agentAutoFireCount}`);
      this.required('[data-role="gameplay-q-values"]').innerHTML = renderPolicyQValuesMarkup(step.policy.qValues, step.policy.actionIndex);
      this.renderPreprocessing(environment.observation, environment.processedFrame);
    }
  }

  private renderPreprocessing(observation: Uint8Array, processedFrame: Uint8Array): void {
    drawGrayscale(this.required<HTMLCanvasElement>('[data-role="preprocess-frame"]'), processedFrame, 84);
    for (let index = 0; index < 4; index += 1) {
      drawGrayscale(this.required<HTMLCanvasElement>(`canvas[data-stack-index="${index}"]`), observation.slice(index * 84 * 84, (index + 1) * 84 * 84), 84);
    }
  }

  private async refreshWebGpuSupport(): Promise<void> {
    const support = await detectWebGpuSupport();
    if (!this.mounted) return;
    this.webgpuSupport = support;
    this.renderWebGpuSupport(support);
  }

  private async prepareFormalPolicy(): Promise<void> {
    if (this.policy.isLoaded && this.policy.requestedBackend === this.selectedBackend && this.policy.actualBackend === this.selectedBackend) return;
    if (!this.policy.isLoaded && this.policy.requestedBackend === this.selectedBackend) return;
    await this.policy.release();
    this.policy = new OrtWebPolicy({ backend: this.selectedBackend });
  }

  private async prepareGameplayPolicy(): Promise<void> {
    const support = this.webgpuSupport ?? await detectWebGpuSupport();
    this.webgpuSupport = support;
    const candidates: InferenceBackend[] = support.supported ? ['webgpu', 'wasm'] : ['wasm'];
    let lastError: unknown = null;
    for (const backend of candidates) {
      try {
        if (!this.policy.isLoaded || this.policy.requestedBackend !== backend || this.policy.actualBackend !== backend) {
          await this.gameplayInferenceWorker?.release();
          this.gameplayInferenceWorker = null;
          await this.policy.release();
          this.policy = new OrtWebPolicy({ backend });
          await this.policy.load();
        }
        if (this.gameplayInferenceWorker?.actualBackend !== backend) {
          await this.gameplayInferenceWorker?.release();
          this.gameplayInferenceWorker = new AgentInferenceWorker(backend);
          await this.gameplayInferenceWorker.load();
        }
        if (this.gameplayInferenceWorker.actualBackend !== backend) {
          throw new Error(`gameplay worker requested ${backend.toUpperCase()} but used ${this.gameplayInferenceWorker.actualBackend ?? 'unavailable'}`);
        }
        this.gameplayBackend = this.policy.actualBackend;
        if (backend === 'wasm' && support.supported && this.debug) {
          this.setTechnicalMessage('WebGPU could not start for gameplay; the product is running on its truthful WASM fallback.');
        }
        return;
      } catch (error) {
        lastError = error;
        if (backend === 'webgpu' && this.debug) {
          this.setTechnicalMessage(`Preferred WebGPU gameplay backend unavailable: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }
    throw lastError instanceof Error ? lastError : new Error('The game could not start in this browser.');
  }

  private renderWebGpuSupport(support: WebGpuSupport): void {
    if (!this.debug) return;
    this.setText('[data-role="webgpu-support"]', support.supported ? 'available' : 'unavailable');
    const element = this.required('[data-role="webgpu-support"]');
    element.dataset.status = support.supported ? 'available' : 'unavailable';
  }

  private async changeBackend(value: string): Promise<void> {
    if (value !== 'wasm' && value !== 'webgpu') return;
    if (value === this.selectedBackend || this.busy) return;
    this.busy = true;
    this.setBusy(true);
    this.gameLoop?.pause();
    try {
      await this.gameplayInferenceWorker?.release();
      this.gameplayInferenceWorker = null;
      await this.policy.release();
      this.selectedBackend = value;
      this.policy = new OrtWebPolicy({ backend: value });
      this.scheduler.reset();
      this.validationResult = null;
      delete window.__day27Validation;
      delete window.__day28Validation;
      this.resetValidationView();
    } finally {
      this.busy = false;
      this.setBusy(false);
    }
  }

  private resetValidationView(): void {
    this.setStatus('idle');
    this.setValidationStatus('Not validated');
    this.setText('[data-role="model-loaded"]', 'not loaded');
    this.setText('[data-role="requested-backend"]', this.selectedBackend.toUpperCase());
    this.setText('[data-role="actual-backend"]', '—');
    this.setText('[data-role="ort-version"]', '—');
    this.setText('[data-role="browser-info"]', '—');
    this.setText('[data-role="environment-parity"]', '—');
    this.setText('[data-role="backend-evidence"]', '—');
    this.setText('[data-role="current-action"]', 'NOOP');
    this.setText('[data-role="cursor-target-x"]', '—');
    this.setText('[data-role="paddle-center-x"]', '—');
    this.setText('[data-role="motion-state"]', 'STOPPED');
    this.setText('[data-role="position-error"]', '—');
    this.setText('[data-role="human-requested-action"]', 'NOOP');
    this.setText('[data-role="human-executed-action"]', 'NOOP');
    this.setText('[data-role="requested-paddle-position"]', '—');
    this.setText('[data-role="applied-paddle-target"]', '—');
    this.setText('[data-role="human-raw-frame-number"]', '0');
    this.setText('[data-role="human-actual-emulator-frames"]', '0');
    this.setText('[data-role="human-raw-fps"]', '0');
    this.setText('[data-role="human-tick-count"]', '0');
    this.setText('[data-role="human-late-ticks"]', '0');
    this.setText('[data-role="human-dropped-ticks"]', '0');
    this.setText('[data-role="human-catch-up-bursts"]', '0');
    this.setText('[data-role="human-tick-p50-p95"]', '0 / 0 ms');
    this.setText('[data-role="human-response-latency"]', '—');
    this.setText('[data-role="human-frame-repeat"]', '1');
    this.setText('[data-role="human-sticky"]', '0');
    this.setText('[data-role="agent-decision-rate"]', '0');
    this.setText('[data-role="agent-decision-count"]', '0');
    this.setText('[data-role="agent-decision-p50-p95"]', '0 / 0 ms');
    this.setText('[data-role="agent-raw-frame-delta"]', '0');
    this.setText('[data-role="executed-human-action"]', 'NOOP');
    this.setText('[data-role="debug-difficulty"]', AI_DIFFICULTY_LABELS[this.difficultyPolicy.currentDifficulty]);
    this.setText('[data-role="difficulty-rate"]', `${(this.difficultyPolicy.currentMistakeRate * 100).toFixed(0)}%`);
    this.setText('[data-role="greedy-action"]', 'NOOP');
    this.setText('[data-role="mistake-injected"]', 'no');
    this.setText('[data-role="episode-return"]', '0');
    this.setText('[data-role="inference-latency"]', '—');
    this.setText('[data-role="agent-frame"]', '0 / 0');
    this.setText('[data-role="auto-fire"]', '0');
    this.setText('[data-role="gameplay-backend"]', '—');
    this.setText('[data-role="action-agreement"]', '—');
    this.setText('[data-role="max-error"]', '—');
    this.setText('[data-role="mean-error"]', '—');
    this.setText('[data-role="q-margin"]', '—');
    this.setText('[data-role="margin-error"]', '—');
    this.setText('[data-role="disagreements"]', '—');
    this.setText('[data-role="evidence-artifact"]', '—');
    this.setText('[data-role="evidence-page"]', '—');
    this.setText('[data-role="sample-count"]', '0 samples');
    this.setText('[data-role="model-sha"]', '—');
    this.setText('[data-role="validation-message"]', `尚未執行。結果必須來自 Browser 的真實 ORT Web ${this.selectedBackend.toUpperCase()} inference。`);
    this.required('[data-role="gameplay-q-values"]').innerHTML = '<p class="muted">Q-values appear after the first real agent decision.</p>';
    this.required<HTMLAnchorElement>('[data-role="download-validation"]').hidden = true;
    if (this.webgpuSupport) this.renderWebGpuSupport(this.webgpuSupport);
  }

  private renderHumanInput(): void {
    const mouseCommand = this.mouse.peekCommand();
    const inputs = this.inputMode === 'keyboard'
      ? [...this.keyboard.snapshot()]
      : mouseCommand.direction === 'NOOP' ? [] : [mouseCommand.direction];
    this.setText('[data-role="human-input"]', inputs.length ? inputs.join(' + ') : 'none');
    if (this.inputMode === 'mouse') this.setText('[data-role="input-hint"]', mouseStatusMessage());
    this.renderMouseTargetMarker();
    if (this.debug) {
      this.setText('[data-role="cursor-target-x"]', formatNormalized(this.mouse.targetX));
      this.setText('[data-role="paddle-center-x"]', formatNormalized(this.mouse.paddleCenterX));
      this.setText('[data-role="motion-state"]', this.mouse.motionState);
      this.setText('[data-role="position-error"]', formatNormalized(this.mouse.positionError));
      this.setText('[data-role="executed-human-action"]', this.lastHumanAction);
      this.updateMouseDiagnostics();
    }
  }

  private setInputMode(value: string): void {
    if (value !== 'keyboard' && value !== 'mouse') return;
    this.inputMode = value;
    const keyboardEnabled = value === 'keyboard';
    this.keyboard.setEnabled(keyboardEnabled);
    this.mouse.setEnabled(!keyboardEnabled);
    this.lastHumanAction = 'NOOP';
    const label = value === 'keyboard' ? 'Keyboard' : 'Mouse';
    this.setText('[data-role="human-input-mode"]', label);
    this.setText('[data-role="input-hint"]', keyboardEnabled ? '← / → move · Space serves' : mouseStatusMessage());
    this.required<HTMLCanvasElement>('[data-role="human-canvas"]').dataset.inputMode = value;
    this.renderHumanInput();
  }

  private setDifficulty(value: string): void {
    if (!isAiDifficulty(value)) return;
    this.difficultyPolicy.setDifficulty(value as AiDifficulty);
    const label = AI_DIFFICULTY_LABELS[value as AiDifficulty];
    this.setText('[data-role="ai-difficulty-label"]', label);
    if (this.debug) {
      this.setText('[data-role="debug-difficulty"]', label);
      this.setText('[data-role="difficulty-rate"]', `${(this.difficultyPolicy.currentMistakeRate * 100).toFixed(0)}%`);
    }
    this.setUserMessage(`Difficulty: ${label}`);
  }

  private clearHumanInput(): void {
    this.keyboard.clear();
    this.mouse.clear();
    this.lastHumanAction = 'NOOP';
    this.renderHumanInput();
  }

  private readonly onWindowBlur = (): void => {
    this.clearHumanInput();
  };

  private currentHumanCommand(): HumanLoopCommand {
    if (this.inputMode === 'mouse') {
      const command = this.mouse.currentCommand();
      this.lastHumanAction = command.direction;
      if (this.debug) this.updateMouseDiagnostics(command);
      return { kind: 'paddle', command };
    }
    const action = this.keyboard.currentAction();
    this.lastHumanAction = action;
    if (this.debug) {
      this.setText('[data-role="executed-human-action"]', action);
      this.updateMouseDiagnostics();
    }
    return { kind: 'discrete', actionIndex: ACTION_MEANINGS.indexOf(action) };
  }

  private renderMouseTargetMarker(): void {
    const marker = this.root.querySelector<HTMLElement>('[data-role="mouse-target-marker"]');
    const canvas = this.root.querySelector<HTMLCanvasElement>('[data-role="human-canvas"]');
    const stage = this.root.querySelector<HTMLElement>('.human-game-stage');
    const targetX = this.mouse.targetX;
    if (!marker || !canvas || !stage || this.inputMode !== 'mouse' || targetX === null) {
      if (marker) marker.hidden = true;
      return;
    }
    const canvasBounds = canvas.getBoundingClientRect();
    const stageBounds = stage.getBoundingClientRect();
    marker.hidden = false;
    marker.style.left = `${canvasBounds.left - stageBounds.left + targetX * canvasBounds.width}px`;
  }

  private updateMouseDiagnostics(command = this.mouse.peekCommand()): void {
    if (!this.debug) return;
    const current = {
      cursorTargetX: this.mouse.targetX,
      paddleCenterX: this.mouse.paddleCenterX,
      motionState: this.mouse.motionState,
      positionError: this.mouse.positionError,
      requestedDirection: command.direction,
      requestedPaddlePositionX: command.targetX,
      executedDirection: this.latestHumanStep?.executedDirection ?? 'NOOP',
      appliedPaddleTargetX: this.latestHumanStep?.appliedPaddleTargetX ?? null,
      rawFrameNumber: this.latestHumanStep?.rawFrameNumber ?? 0,
      actualEmulatorFrames: this.latestHumanStep?.actualEmulatorFrames ?? 0,
      executedHumanAction: this.lastHumanAction,
    };
    const diagnostics = window.__mouseControlV3Diagnostics;
    if (diagnostics) {
      Object.assign(diagnostics, current);
      window.__mouseControlV3Diagnostics = diagnostics;
      return;
    }
    window.__mouseControlV3Diagnostics = { ...current, decisionCount: 0, actionHistory: [] };
  }

  private initializeHumanDiagnostics(): void {
    if (!this.debug || !this.humanEnvironment) return;
    window.__humanInteractiveDiagnostics = {
      schemaVersion: 1,
      runtimeContract: this.humanEnvironment.runtimeDiagnostics,
      runtime: null,
      samples: [],
    };
  }

  private recordHumanStep(step: HumanEnvironmentStep): void {
    if (!this.debug) return;
    const diagnostics = window.__humanInteractiveDiagnostics;
    if (!diagnostics) return;
    const sample: HumanActionSample = {
      decision: diagnostics.samples.length + 1,
      timestampMs: performance.now(),
      targetX: this.mouse.targetX,
      targetChangedAtMs: this.mouse.targetChangedAt,
      paddleCenterX: this.mouse.paddleCenterX,
      positionError: this.mouse.positionError,
      requestedAction: step.requestedAction,
      executedAction: step.executedAction,
      requestedDirection: step.requestedDirection,
      executedDirection: step.executedDirection,
      requestedPaddlePositionX: step.requestedPaddlePositionX,
      appliedPaddleTargetX: step.appliedPaddleTargetX,
      action: step.executedAction,
      rawFrameNumber: step.rawFrameNumber,
      actualEmulatorFrames: step.actualEmulatorFrames,
    };
    diagnostics.samples.push(sample);
    if (diagnostics.samples.length > 20_000) diagnostics.samples.splice(0, diagnostics.samples.length - 20_000);
    const legacy = window.__mouseControlV3Diagnostics;
    if (legacy) {
      legacy.decisionCount += 1;
      legacy.actionHistory.push({ ...sample, decision: legacy.decisionCount });
      if (legacy.actionHistory.length > 512) legacy.actionHistory.splice(0, legacy.actionHistory.length - 512);
    }
    this.lastHumanAction = step.executedAction;
    if (this.debug) {
      this.setText('[data-role="requested-paddle-position"]', formatNormalized(step.requestedPaddlePositionX));
      this.setText('[data-role="applied-paddle-target"]', formatNormalized(step.appliedPaddleTargetX));
      this.setText('[data-role="human-raw-frame-number"]', `${step.rawFrameNumber}`);
      this.setText('[data-role="human-actual-emulator-frames"]', `${step.actualEmulatorFrames}`);
    }
    this.updateMouseDiagnostics();
  }

  private updateHumanRuntimeDiagnostics(diagnostics: DualLoopDiagnostics | null): void {
    if (!this.debug) return;
    const humanDiagnostics = window.__humanInteractiveDiagnostics;
    if (humanDiagnostics) humanDiagnostics.runtime = diagnostics;
    if (!diagnostics) return;
    this.setText('[data-role="human-raw-fps"]', diagnostics.humanRawFps.toFixed(2));
    this.setText('[data-role="human-tick-count"]', `${diagnostics.humanRawTickCount}`);
    this.setText('[data-role="human-late-ticks"]', `${diagnostics.lateHumanTicks}`);
    this.setText('[data-role="human-dropped-ticks"]', `${diagnostics.droppedHumanTicks}`);
    this.setText('[data-role="human-catch-up-bursts"]', `${diagnostics.catchUpBursts}`);
    this.setText('[data-role="human-tick-p50-p95"]', `${diagnostics.humanTickP50Ms.toFixed(2)} / ${diagnostics.humanTickP95Ms.toFixed(2)} ms`);
    this.setText('[data-role="agent-decision-rate"]', diagnostics.agentDecisionsPerSecond.toFixed(2));
    this.setText('[data-role="agent-decision-count"]', `${diagnostics.agentDecisionCount}`);
    this.setText('[data-role="agent-decision-p50-p95"]', `${diagnostics.agentDecisionP50Ms.toFixed(2)} / ${diagnostics.agentDecisionP95Ms.toFixed(2)} ms`);
    this.setText('[data-role="agent-raw-frame-delta"]', `${diagnostics.agentRawFrameDelta}`);
    this.setText('[data-role="human-response-latency"]', this.formatInputResponseLatency());
  }

  private formatInputResponseLatency(): string {
    const samples = window.__humanInteractiveDiagnostics?.samples ?? [];
    if (samples.length < 2) return '—';
    const current = samples[samples.length - 1];
    const previous = samples[samples.length - 2];
    if (!current || current.requestedPaddlePositionX === null || current.targetChangedAtMs === null) return '—';
    return `${Math.max(0, current.timestampMs - current.targetChangedAtMs).toFixed(2)} ms`;
  }

  private setUserMessage(message: string): void {
    this.setText('[data-role="user-message"]', message);
  }

  private setTechnicalMessage(message: string): void {
    if (!this.debug) return;
    this.setText('[data-role="validation-message"]', message);
  }

  private reportRuntimeError(error: unknown): void {
    const detail = error instanceof Error ? error.message : String(error);
    this.setTechnicalMessage(detail);
    this.setUserMessage('The game could not start. Try Restart and try again.');
  }

  private dispatch(event: RuntimeEvent): void {
    this.setStatus(reduceRuntimeStatus(this.status, event));
    if (event === 'start' && this.status === 'running') this.scheduler.start();
    if (event === 'pause' && this.status === 'paused') this.scheduler.pause();
    if (event === 'reset' && this.status === 'ready' && !this.scheduler.isInFlight) this.scheduler.reset();
  }

  private setStatus(status: RuntimeStatus): void {
    this.status = status;
    if (!this.debug) return;
    const badge = this.root.querySelector<HTMLElement>('[data-role="runtime-status"]');
    if (!badge) return;
    this.setText('[data-role="runtime-status-label"]', status);
    badge.dataset.status = status;
  }

  private setValidationStatus(status: string): void {
    if (!this.debug) return;
    const badge = this.required('[data-role="validation-status"]');
    badge.textContent = status;
    badge.dataset.status = status.toLowerCase();
  }

  private renderPolicyRuntime(actualBackend: InferenceBackend, ortVersion: string, browserName: string, browserVersion: string, platform: string): void {
    if (!this.debug) return;
    this.setText('[data-role="requested-backend"]', this.selectedBackend.toUpperCase());
    this.setText('[data-role="actual-backend"]', actualBackend.toUpperCase());
    this.setText('[data-role="gameplay-backend"]', actualBackend.toUpperCase());
    this.setText('[data-role="ort-version"]', ortVersion);
    this.setText('[data-role="browser-info"]', `${browserName} ${browserVersion} / ${platform}`);
  }

  private async getContract(): Promise<BreakoutContractV2> {
    if (!this.contract) this.contract = await loadBreakoutContract();
    return this.contract;
  }

  private button(action: string): HTMLButtonElement {
    const element = this.root.querySelector<HTMLButtonElement>(`button[data-action="${action}"]`);
    if (!element) throw new Error(`missing ${action} button`);
    return element;
  }

  private select(action: string): HTMLSelectElement {
    const element = this.root.querySelector<HTMLSelectElement>(`select[data-action="${action}"]`);
    if (!element) throw new Error(`missing ${action} select`);
    return element;
  }

  private setBusy(busy: boolean): void {
    this.busy = busy;
    if (!this.debug) return;
    this.button('validate').disabled = busy;
    this.button('benchmark').disabled = busy;
    this.button('evaluate').disabled = busy;
    this.select('backend').disabled = busy;
  }

  private setText<T extends HTMLElement = HTMLElement>(selector: string, text: string): void {
    this.required<T>(selector).textContent = text;
  }

  private required<T extends HTMLElement = HTMLElement>(selector: string): T {
    const element = this.root.querySelector<T>(selector);
    if (!element) throw new Error(`missing element ${selector}`);
    return element;
  }
}

function drawGrayscale(canvas: HTMLCanvasElement, frame: ArrayLike<number>, size: number): void {
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  if (!context) return;
  const image = context.createImageData(size, size);
  for (let index = 0; index < size * size; index += 1) {
    const value = frame[index] ?? 0;
    const offset = index * 4;
    image.data[offset] = value;
    image.data[offset + 1] = value;
    image.data[offset + 2] = value;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

function formatMs(value: number): string {
  return `${value.toFixed(3)} ms`;
}

function backendEvidenceLabel(evidence: FixtureValidationResult['backendEvidence']): string {
  return evidence === 'webgpu_session_exposes_env_webgpu_device' ? 'runtime GPU device observed' : 'explicit WASM session';
}

function pageHost(pageUrl: string): string {
  try {
    return new URL(pageUrl).hostname;
  } catch {
    return pageUrl;
  }
}

function displayStageState(finished: boolean, status: string): string {
  if (finished) return 'GAME OVER';
  if (status === 'running') return 'PLAYING';
  if (status === 'paused') return 'PAUSED';
  return 'READY';
}

function displayCardState(finished: boolean, status: string): string {
  if (finished) return 'Game over';
  if (status === 'running') return 'Playing';
  if (status === 'paused') return 'Paused';
  return 'Ready';
}

function mouseStatusMessage(): string {
  return 'Move mouse to control paddle';
}

function formatNormalized(value: number | null): string {
  return value === null ? '—' : value.toFixed(3);
}
