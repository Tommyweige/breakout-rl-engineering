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
import { detectWebGpuSupport } from '../inference/webgpuSupport';
import { KeyboardController } from '../input/KeyboardController';
import { buildBrowserValidationArtifact, buildDay28ValidationArtifact } from '../validation/buildArtifact';
import { runBrowserBackendComparison } from '../validation/runBrowserBenchmark';
import { runFixtureValidation } from '../validation/runFixtureValidation';
import { reduceRuntimeStatus, type RuntimeEvent } from './runtimeState';
import { renderAppShell, renderQValuesMarkup } from './view';

declare global {
  interface Window {
    __day27Validation?: BrowserValidationArtifact;
    __day28Validation?: Day28ValidationArtifact;
    __day28Benchmark?: Day28BenchmarkArtifact;
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
  private unsubscribeScheduler: (() => void) | null = null;
  private webgpuSupport: WebGpuSupport | null = null;
  private validationResult: FixtureValidationResult | null = null;

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

    this.button('start').addEventListener('click', () => this.dispatch('start'));
    this.button('pause').addEventListener('click', () => this.dispatch('pause'));
    this.button('reset').addEventListener('click', () => this.dispatch('reset'));
    this.button('validate').addEventListener('click', () => void this.validateSelectedBackend());
    this.button('benchmark').addEventListener('click', () => void this.runBenchmark());
    this.select('backend').addEventListener('change', (event) => {
      void this.changeBackend((event.target as HTMLSelectElement).value);
    });
    void this.refreshWebGpuSupport();
  }

  destroy(): void {
    if (!this.mounted) return;
    this.keyboard.detach();
    if (this.inputTimer !== null) window.clearInterval(this.inputTimer);
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    if (this.benchmarkDownloadUrl) URL.revokeObjectURL(this.benchmarkDownloadUrl);
    this.unsubscribeScheduler?.();
    void this.policy.release().catch(() => undefined);
    this.inputTimer = null;
    this.downloadUrl = null;
    this.benchmarkDownloadUrl = null;
    this.unsubscribeScheduler = null;
    this.mounted = false;
  }

  private async validateSelectedBackend(): Promise<void> {
    if (this.busy) return;
    const message = this.required('[data-role="validation-message"]');
    this.setBusy(true);
    this.scheduler.start();
    this.dispatch('load-start');
    this.setValidationStatus('Loading');
    message.textContent = `正在載入 manifest、FP32 ONNX 並以 ${this.selectedBackend.toUpperCase()} 執行 60 個固定 fixtures…`;

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
    this.setText('[data-role="model-loaded"]', 'loaded');
    this.setText('[data-role="requested-backend"]', result.requestedBackend.toUpperCase());
    this.setText('[data-role="actual-backend"]', result.actualBackend.toUpperCase());
    this.renderWebGpuSupport(result.webgpuSupport);
    this.setText('[data-role="ort-version"]', result.ortWebVersion);
    this.setText('[data-role="browser-info"]', `${result.browser.name} ${result.browser.version} / ${result.browser.platform}`);
    this.setText('[data-role="environment-parity"]', `${result.environmentContract.status} / inference only`);
    this.setText('[data-role="backend-evidence"]', backendEvidenceLabel(result.backendEvidence));
    this.setText('[data-role="action-agreement"]', `${(result.actionAgreementRate * 100).toFixed(2)}%`);
    this.setText('[data-role="max-error"]', result.maxAbsoluteError.toExponential(3));
    this.setText('[data-role="mean-error"]', result.meanAbsoluteError.toExponential(3));
    this.setText(
      '[data-role="q-margin"]',
      `${result.qMargin.minMargin.toExponential(3)} / ${result.qMargin.meanMargin.toExponential(3)}`,
    );
    this.setText('[data-role="margin-error"]', result.qMargin.maxAbsoluteMarginError.toExponential(3));
    this.setText('[data-role="disagreements"]', result.disagreementIndices.length ? result.disagreementIndices.join(', ') : 'none');
    this.setText('[data-role="sample-count"]', `${result.sampleCount} samples`);
    this.setText('[data-role="model-sha"]', result.modelSha256);
    this.setText('[data-role="evidence-artifact"]', `day28-${result.actualBackend}-validation.json`);
    this.setText('[data-role="evidence-page"]', pageHost(result.pageUrl));
    this.setValidationStatus(result.passed ? 'PASS' : 'FAIL');
    this.setText('[data-role="selected-action"]', `${result.representative.selectedAction} (fixture #${result.representative.sampleIndex})`);
    this.renderQValues(result);
    this.setText(
      '[data-role="validation-message"]',
      result.passed
        ? `${result.actualBackend.toUpperCase()} validation passed on ${result.sampleCount} fixed states. This is a correctness sanity check, not a gameplay score evaluation.`
        : `${result.actualBackend.toUpperCase()} validation failed. disagreement indices: ${result.disagreementIndices.join(', ') || 'none'}`,
    );
  }

  private renderQValues(result: FixtureValidationResult): void {
    const container = this.required('[data-role="q-values"]');
    const values = [...result.representative.qValues];
    const referenceValues = [...result.representative.referenceQValues];
    const scale = Math.max(1, ...values.map((value) => Math.abs(value)), ...referenceValues.map((value) => Math.abs(value)));
    container.innerHTML = renderQValuesMarkup(result);
    ACTION_MEANINGS.forEach((_action, index) => {
      const actualValue = values[index]!;
      const referenceValue = referenceValues[index]!;
      this.setText(`[data-q-kind="actual"][data-q-index="${index}"]`, actualValue.toFixed(6));
      this.setText(`[data-q-kind="reference"][data-q-index="${index}"]`, referenceValue.toFixed(6));
      this.required<HTMLElement>(`[data-q-kind="actual-bar"][data-q-index="${index}"]`).style.width = `${Math.min(100, (Math.abs(actualValue) / scale) * 100).toFixed(2)}%`;
      this.required<HTMLElement>(`[data-q-kind="reference-bar"][data-q-index="${index}"]`).style.width = `${Math.min(100, (Math.abs(referenceValue) / scale) * 100).toFixed(2)}%`;
    });
  }

  private publishArtifact(): void {
    if (!this.validationResult) return;
    const artifact = buildDay28ValidationArtifact(this.validationResult);
    window.__day28Validation = artifact;
    if (this.validationResult.actualBackend === 'wasm') {
      window.__day27Validation = buildBrowserValidationArtifact(this.validationResult);
    } else {
      delete window.__day27Validation;
    }
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
    message.textContent = '正在同一個 Browser session 中執行 WASM / WebGPU 60-state parity 與 batch=1 benchmark…';
    this.setText('[data-role="benchmark-status"]', 'running');
    try {
      const artifact = await runBrowserBackendComparison({ sampleCount: 100, warmupCount: 10 });
      window.__day28Benchmark = artifact;
      this.renderBenchmark(artifact);
      this.setText(
        '[data-role="validation-message"]',
        `WASM / WebGPU benchmark completed with ${artifact.sampleCount} measured samples per backend. Initialization and warm-ups were excluded from the timed scope.`,
      );
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
    this.setText('[data-role="selected-action"]', '—');
    this.setText('[data-role="action-agreement"]', '—');
    this.setText('[data-role="max-error"]', '—');
    this.setText('[data-role="mean-error"]', '—');
    this.setText('[data-role="q-margin"]', '—');
    this.setText('[data-role="margin-error"]', '—');
    this.setText('[data-role="disagreements"]', '—');
    this.setText('[data-role="sample-count"]', '0 samples');
    this.setText('[data-role="model-sha"]', '—');
    this.setText('[data-role="evidence-artifact"]', '—');
    this.setText('[data-role="evidence-page"]', '—');
    this.setText('[data-role="validation-message"]', `尚未執行。結果必須來自 Browser 的真實 ORT Web ${this.selectedBackend.toUpperCase()} inference。`);
    this.required('[data-role="q-values"]').innerHTML = '<p class="muted">Q-values appear after a real fixture inference.</p>';
    this.required<HTMLAnchorElement>('[data-role="download-validation"]').hidden = true;
    if (this.webgpuSupport) this.renderWebGpuSupport(this.webgpuSupport);
  }

  private renderHumanInput(): void {
    const inputs = [...this.keyboard.snapshot()];
    this.setText('[data-role="human-input"]', inputs.length ? inputs.join(' + ') : 'none');
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
