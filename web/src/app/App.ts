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
import { OrtWebPolicy } from '../inference/OrtWebPolicy';
import { detectBrowser } from '../inference/browserInfo';
import { detectWebGpuSupport } from '../inference/webgpuSupport';
import { KeyboardController } from '../input/KeyboardController';
import { BrowserBreakoutEnvironment, type EnvironmentStep } from '../environment/aleEnvironment';
import { loadBreakoutContract, type BreakoutContractV2 } from '../environment/breakoutContract';
import { DualGameLoop, type AgentLoopStep } from '../game/dualGameLoop';
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
  }
}

export class App {
  private policy: OrtWebPolicy;
  private readonly scheduler: InferenceScheduler<Uint8Array, PolicyResult>;
  private readonly keyboard = new KeyboardController();
  private status: RuntimeStatus = 'idle';
  private selectedBackend: InferenceBackend = 'wasm';
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
  private humanEnvironment: BrowserBreakoutEnvironment | null = null;
  private agentEnvironment: BrowserBreakoutEnvironment | null = null;
  private gameLoop: DualGameLoop | null = null;
  private gameplayInitPromise: Promise<void> | null = null;
  private latestHumanStep: EnvironmentStep | null = null;
  private latestAgentStep: AgentLoopStep | null = null;
  private agentAutoFireCount = 0;

  constructor(private readonly root: HTMLElement) {
    this.policy = new OrtWebPolicy({ backend: this.selectedBackend });
    this.scheduler = new InferenceScheduler((observation) => this.policy.infer(observation));
  }

  mount(): void {
    if (this.mounted) return;
    this.mounted = true;
    this.root.innerHTML = `<main class="app-shell">${renderAppShell()}</main>`;

    this.keyboard.attach();
    this.inputTimer = window.setInterval(() => this.renderHumanInput(), 80);
    this.unsubscribeScheduler = this.scheduler.subscribe((status) => {
      if (this.mounted) this.setText('[data-role="scheduler-status"]', status);
    });

    this.button('start').addEventListener('click', () => {
      this.dispatch('start');
      void this.startGameplay();
    });
    this.button('pause').addEventListener('click', () => {
      this.dispatch('pause');
      this.gameLoop?.pause();
    });
    this.button('reset').addEventListener('click', () => {
      this.dispatch('reset');
      void this.resetGameplay();
    });
    this.button('validate').addEventListener('click', () => void this.validateSelectedBackend());
    this.button('benchmark').addEventListener('click', () => void this.runBenchmark());
    this.button('evaluate').addEventListener('click', () => void this.runEvaluation());
    this.select('backend').addEventListener('change', (event) => {
      void this.changeBackend((event.target as HTMLSelectElement).value);
    });
    window.__day29Evaluate = (seeds) => this.runEvaluation(seeds);
    window.__day29CapturePreprocessingTrace = (seed, actions) => this.capturePreprocessingTrace(seed, actions);
    window.__day29EvaluationSeeds = undefined;
    void this.refreshWebGpuSupport();
  }

  destroy(): void {
    if (!this.mounted) return;
    this.gameLoop?.pause();
    this.humanEnvironment?.dispose();
    this.agentEnvironment?.dispose();
    this.keyboard.detach();
    if (this.inputTimer !== null) window.clearInterval(this.inputTimer);
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    if (this.benchmarkDownloadUrl) URL.revokeObjectURL(this.benchmarkDownloadUrl);
    if (this.evaluationDownloadUrl) URL.revokeObjectURL(this.evaluationDownloadUrl);
    this.unsubscribeScheduler?.();
    void this.policy.release().catch(() => undefined);
    this.inputTimer = null;
    this.downloadUrl = null;
    this.benchmarkDownloadUrl = null;
    this.evaluationDownloadUrl = null;
    this.unsubscribeScheduler = null;
    window.__day29Ready = false;
    delete window.__day29Evaluate;
    delete window.__day29CapturePreprocessingTrace;
    delete window.__day29EvaluationSeeds;
    delete window.__day29EnvironmentDiagnostics;
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

  private async runEvaluation(seeds?: readonly number[]): Promise<BrowserPolicyEvaluationArtifact | null> {
    if (this.busy) return null;
    this.setBusy(true);
    this.gameLoop?.pause();
    this.setText('[data-role="evaluation-status"]', 'running');
    this.setText('[data-role="evaluation-result"]', 'running');
    this.setText('[data-role="evaluation-message"]', '正在用固定 30 seeds 跑 Browser ALE policy evaluation；每局保存 raw return、length、game-over 與 agent-loop timing。');
    this.required<HTMLElement>('[data-role="evaluation-summary"]').hidden = false;
    try {
      const contract = await this.getContract();
      window.__day29EvaluationSeeds = browserEvaluationSeeds(contract);
      const artifact = await runPolicyEvaluation({
        backend: this.selectedBackend,
        policy: this.policy,
        contract,
        seeds,
        modelSha256: this.validationResult?.modelSha256,
        onEpisode: (_episode, completed, total) => {
          this.setText('[data-role="evaluation-status"]', `${completed}/${total}`);
        },
      });
      window.__day29Evaluation = artifact;
      this.renderEvaluation(artifact);
      this.publishEvaluationArtifact(artifact);
      this.setText('[data-role="validation-message"]', `${artifact.actualBackend.toUpperCase()} 30-episode Browser evaluation completed. Score and latency are reported separately.`);
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
    link.download = `day29-${artifact.actualBackend}-browser-policy-evaluation.json`;
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
      if (!this.policy.isLoaded) await this.policy.load();
      if (this.policy.actualBackend !== this.selectedBackend) {
        throw new Error(`requested ${this.selectedBackend.toUpperCase()} but actual backend is ${this.policy.actualBackend ?? 'unavailable'}`);
      }
      if (this.gameLoop) {
        this.scheduler.start();
        this.gameLoop.start();
        window.__day29Ready = true;
        return;
      }
      if (!this.humanEnvironment || !this.agentEnvironment) {
        const seed = contract.concrete_episode_seeds[0] ?? 101;
        [this.humanEnvironment, this.agentEnvironment] = await Promise.all([
          BrowserBreakoutEnvironment.create({ contract, seed }),
          BrowserBreakoutEnvironment.create({ contract, seed }),
        ]);
      }
      const browser = await detectBrowser();
      this.renderPolicyRuntime(this.policy.actualBackend, this.policy.ortWebVersion, browser.name, browser.version, browser.platform);
      this.setText('[data-role="environment-parity"]', `${contract.parity.status.toUpperCase()} / Contract v2`);
      this.setText('[data-role="backend-evidence"]', this.policy.backendEvidence === 'webgpu_session_exposes_env_webgpu_device' ? 'runtime GPU device observed' : 'explicit WASM session');
      this.setText('[data-role="model-loaded"]', 'loaded / Day 21 canonical');
      this.setText('[data-role="gameplay-backend"]', this.policy.actualBackend.toUpperCase());
      this.setText('[data-role="model-sha"]', this.validationResult?.modelSha256 ?? 'manifest hash recorded by evaluation');
      window.__day29EnvironmentDiagnostics = {
        ...this.agentEnvironment.runtimeDiagnostics,
        humanInstanceId: this.humanEnvironment.instanceId,
        agentInstanceId: this.agentEnvironment.instanceId,
        crossOriginIsolated: window.crossOriginIsolated,
        dualInstanceCount: new Set([this.humanEnvironment.instanceId, this.agentEnvironment.instanceId]).size,
      };
      this.agentAutoFireCount = 0;
      this.latestHumanStep = null;
      this.latestAgentStep = null;
      this.gameLoop = new DualGameLoop({
        human: this.humanEnvironment,
        agent: this.agentEnvironment,
        humanAction: () => ACTION_MEANINGS.indexOf(this.keyboard.currentAction()),
        infer: (observation) => this.scheduler.run(observation),
        onHumanStep: (step) => {
          this.latestHumanStep = step;
          this.renderHumanStep(step);
        },
        onAgentStep: (step) => {
          this.latestAgentStep = step;
          if (step.environment.autoFire) this.agentAutoFireCount += 1;
          this.renderAgentStep(step);
        },
        onFrame: () => this.renderCanvases(),
        onError: (error) => this.setText('[data-role="validation-message"]', error instanceof Error ? error.message : String(error)),
      });
      this.scheduler.start();
      this.gameLoop.start();
      window.__day29Ready = true;
      this.renderCanvases();
    } catch (error) {
      this.setStatus('error');
      this.setText('[data-role="validation-message"]', error instanceof Error ? error.message : String(error));
    }
  }

  private async resetGameplay(): Promise<void> {
    if (!this.gameLoop) return;
    try {
      await this.gameLoop.reset();
      this.latestHumanStep = null;
      this.latestAgentStep = null;
      this.agentAutoFireCount = 0;
      this.renderCanvases();
    } catch (error) {
      this.setStatus('error');
      this.setText('[data-role="validation-message"]', error instanceof Error ? error.message : String(error));
    }
  }

  private renderCanvases(): void {
    if (!this.humanEnvironment || !this.agentEnvironment) return;
    this.humanEnvironment.render(this.required<HTMLCanvasElement>('[data-role="human-canvas"]'));
    this.agentEnvironment.render(this.required<HTMLCanvasElement>('[data-role="agent-canvas"]'));
    if (this.latestHumanStep) this.renderHumanStep(this.latestHumanStep);
    if (this.latestAgentStep) this.renderAgentStep(this.latestAgentStep);
    this.setText('[data-role="human-stage-state"]', this.humanEnvironment.isFinished ? 'GAME OVER' : this.gameLoop?.currentStatus.toUpperCase() ?? 'READY');
    this.setText('[data-role="agent-stage-state"]', this.agentEnvironment.isFinished ? 'GAME OVER' : this.gameLoop?.currentStatus.toUpperCase() ?? 'READY');
  }

  private renderHumanStep(step: EnvironmentStep): void {
    this.setText('[data-role="human-score"]', step.episodeReturn.toFixed(0));
    this.setText('[data-role="human-lives"]', `${step.lives}`);
    this.setText('[data-role="human-frame"]', `${step.frameNumber} / ${step.agentStep}`);
    this.setText('[data-role="human-action"]', step.executedAction);
    this.setText('[data-role="human-state"]', step.terminated || step.truncated ? 'game over' : 'running');
  }

  private renderAgentStep(step: AgentLoopStep): void {
    const environment = step.environment;
    this.setText('[data-role="current-action"]', environment.autoFire ? `FIRE / auto-${environment.autoFireReason}` : environment.executedAction);
    this.setText('[data-role="episode-return"]', environment.episodeReturn.toFixed(0));
    this.setText('[data-role="inference-latency"]', `${step.inferenceMs.toFixed(3)} ms`);
    this.setText('[data-role="agent-frame"]', `${environment.frameNumber} / ${environment.agentStep}`);
    this.setText('[data-role="auto-fire"]', `${this.agentAutoFireCount}`);
    this.setText('[data-role="agent-stage-state"]', environment.terminated || environment.truncated ? 'GAME OVER' : this.gameLoop?.currentStatus.toUpperCase() ?? 'RUNNING');
    this.required('[data-role="gameplay-q-values"]').innerHTML = renderPolicyQValuesMarkup(step.policy.qValues, step.policy.actionIndex);
    this.renderPreprocessing(environment.observation, environment.processedFrame);
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

  private renderWebGpuSupport(support: WebGpuSupport): void {
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
    const inputs = [...this.keyboard.snapshot()];
    this.setText('[data-role="human-input"]', inputs.length ? inputs.join(' + ') : 'none');
    this.setText('[data-role="human-action"]', this.keyboard.currentAction());
  }

  private dispatch(event: RuntimeEvent): void {
    this.setStatus(reduceRuntimeStatus(this.status, event));
    if (event === 'start' && this.status === 'running') this.scheduler.start();
    if (event === 'pause' && this.status === 'paused') this.scheduler.pause();
    if (event === 'reset' && this.status === 'ready' && !this.scheduler.isInFlight) this.scheduler.reset();
  }

  private setStatus(status: RuntimeStatus): void {
    this.status = status;
    const badge = this.required('[data-role="runtime-status"]');
    this.setText('[data-role="runtime-status-label"]', status);
    badge.dataset.status = status;
  }

  private setValidationStatus(status: string): void {
    const badge = this.required('[data-role="validation-status"]');
    badge.textContent = status;
    badge.dataset.status = status.toLowerCase();
  }

  private renderPolicyRuntime(actualBackend: InferenceBackend, ortVersion: string, browserName: string, browserVersion: string, platform: string): void {
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
