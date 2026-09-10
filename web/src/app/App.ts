import { ACTION_MEANINGS, type BrowserValidationArtifact, type FixtureValidationResult, type RuntimeStatus } from '../inference/types';
import { OrtWebPolicy } from '../inference/OrtWebPolicy';
import { KeyboardController } from '../input/KeyboardController';
import { buildBrowserValidationArtifact } from '../validation/buildArtifact';
import { runFixtureValidation } from '../validation/runFixtureValidation';
import { reduceRuntimeStatus, type RuntimeEvent } from './runtimeState';
import { renderAppShell, renderQValuesMarkup } from './view';

declare global {
  interface Window {
    __day27Validation?: BrowserValidationArtifact;
  }
}

export class App {
  private readonly policy = new OrtWebPolicy();
  private readonly keyboard = new KeyboardController();
  private status: RuntimeStatus = 'idle';
  private mounted = false;
  private inputTimer: number | null = null;
  private downloadUrl: string | null = null;
  private validationResult: FixtureValidationResult | null = null;

  constructor(private readonly root: HTMLElement) {}

  mount(): void {
    if (this.mounted) return;
    this.mounted = true;
    this.root.innerHTML = `<main class="app-shell">${renderAppShell()}</main>`;

    this.keyboard.attach();
    this.inputTimer = window.setInterval(() => this.renderHumanInput(), 80);

    this.button('start').addEventListener('click', () => this.dispatch('start'));
    this.button('pause').addEventListener('click', () => this.dispatch('pause'));
    this.button('reset').addEventListener('click', () => this.dispatch('reset'));
    this.button('validate').addEventListener('click', () => void this.validateWasm());
  }

  destroy(): void {
    if (!this.mounted) return;
    this.keyboard.detach();
    if (this.inputTimer !== null) window.clearInterval(this.inputTimer);
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    this.inputTimer = null;
    this.downloadUrl = null;
    this.mounted = false;
  }

  private async validateWasm(): Promise<void> {
    const message = this.required('[data-role="validation-message"]');
    const validateButton = this.button('validate');
    validateButton.disabled = true;
    this.dispatch('load-start');
    this.setValidationStatus('Loading');
    message.textContent = '正在載入 manifest、FP32 ONNX 並執行 60 個固定 fixtures…';

    try {
      this.validationResult = await runFixtureValidation(this.policy);
      this.renderValidation(this.validationResult);
      this.publishArtifact();
      this.dispatch(this.validationResult.passed ? 'load-success' : 'load-error');
    } catch (error) {
      this.validationResult = null;
      this.setValidationStatus('ERROR');
      message.textContent = error instanceof Error ? error.message : String(error);
      this.dispatch('load-error');
    } finally {
      validateButton.disabled = false;
    }
  }

  private renderValidation(result: FixtureValidationResult): void {
    this.setText('[data-role="model-loaded"]', 'loaded');
    this.setText('[data-role="requested-backend"]', result.requestedBackend.toUpperCase());
    this.setText('[data-role="actual-backend"]', result.actualBackend.toUpperCase());
    this.setText('[data-role="ort-version"]', result.ortWebVersion);
    this.setText('[data-role="action-agreement"]', `${(result.actionAgreementRate * 100).toFixed(2)}%`);
    this.setText('[data-role="max-error"]', result.maxAbsoluteError.toExponential(3));
    this.setText('[data-role="mean-error"]', result.meanAbsoluteError.toExponential(3));
    this.setText('[data-role="disagreements"]', result.disagreementIndices.length ? result.disagreementIndices.join(', ') : 'none');
    this.setText('[data-role="sample-count"]', `${result.sampleCount} samples`);
    this.setText('[data-role="model-sha"]', result.modelSha256);
    this.setValidationStatus(result.passed ? 'PASS' : 'FAIL');
    this.setText('[data-role="selected-action"]', `${result.representative.selectedAction} (fixture #${result.representative.sampleIndex})`);
    this.renderQValues(result);
    this.setText(
      '[data-role="validation-message"]',
      result.passed
        ? `WASM validation passed on ${result.sampleCount} fixed states. This is a correctness sanity check, not a gameplay score evaluation.`
        : `WASM validation failed. disagreement indices: ${result.disagreementIndices.join(', ') || 'none'}`,
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
    const artifact = buildBrowserValidationArtifact(this.validationResult);
    window.__day27Validation = artifact;
    const link = this.required<HTMLAnchorElement>('[data-role="download-validation"]');
    if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    this.downloadUrl = URL.createObjectURL(new Blob([`${JSON.stringify(artifact, null, 2)}\n`], { type: 'application/json' }));
    link.href = this.downloadUrl;
    link.hidden = false;
  }

  private renderHumanInput(): void {
    const inputs = [...this.keyboard.snapshot()];
    this.setText('[data-role="human-input"]', inputs.length ? inputs.join(' + ') : 'none');
  }

  private dispatch(event: RuntimeEvent): void {
    this.setStatus(reduceRuntimeStatus(this.status, event));
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

  private setText<T extends HTMLElement = HTMLElement>(selector: string, text: string): void {
    this.required<T>(selector).textContent = text;
  }

  private required<T extends HTMLElement = HTMLElement>(selector: string): T {
    const element = this.root.querySelector<T>(selector);
    if (!element) throw new Error(`missing element ${selector}`);
    return element;
  }
}
