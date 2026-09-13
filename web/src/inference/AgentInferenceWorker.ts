import type { InferenceBackend, PolicyResult } from './types';

interface WorkerReadyMessage {
  type: 'ready';
  actualBackend: InferenceBackend;
  ortWebVersion: string;
}

interface WorkerResultMessage {
  type: 'result';
  requestId: number;
  result: PolicyResult;
}

interface WorkerErrorMessage {
  type: 'error';
  requestId?: number;
  message: string;
}

type AgentWorkerMessage = WorkerReadyMessage | WorkerResultMessage | WorkerErrorMessage;

interface PendingRequest {
  resolve: (result: PolicyResult) => void;
  reject: (error: Error) => void;
}

/** Keeps gameplay inference off the main thread without changing formal evaluation. */
export class AgentInferenceWorker {
  private readonly worker: Worker;
  private readonly pending = new Map<number, PendingRequest>();
  private nextRequestId = 1;
  private readyPromise: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((error: Error) => void) | null = null;
  private loaded = false;
  private disposed = false;
  private activeBackend: InferenceBackend | null = null;
  private activeOrtWebVersion = 'unknown';

  constructor(private readonly backend: InferenceBackend) {
    this.worker = new Worker(new URL('./agentInference.worker.ts', import.meta.url), { type: 'module' });
    this.worker.addEventListener('message', this.onMessage);
    this.worker.addEventListener('error', this.onError);
  }

  get actualBackend(): InferenceBackend | null {
    return this.activeBackend;
  }

  get ortWebVersion(): string {
    return this.activeOrtWebVersion;
  }

  async load(): Promise<void> {
    if (this.loaded) return;
    if (!this.readyPromise) {
      this.readyPromise = new Promise<void>((resolve, reject) => {
        this.readyResolve = resolve;
        this.readyReject = reject;
      });
      this.worker.postMessage({ type: 'load', backend: this.backend });
    }
    await this.readyPromise;
    this.loaded = true;
  }

  async infer(observation: Uint8Array): Promise<PolicyResult> {
    if (this.disposed) throw new Error('agent inference worker has been released');
    await this.load();
    const requestId = this.nextRequestId++;
    const transferableObservation = new Uint8Array(observation);
    return new Promise<PolicyResult>((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      this.worker.postMessage(
        { type: 'infer', requestId, observation: transferableObservation.buffer },
        [transferableObservation.buffer],
      );
    });
  }

  async release(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    const error = new Error('agent inference worker released');
    this.readyReject?.(error);
    this.readyReject = null;
    this.readyResolve = null;
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
    this.worker.removeEventListener('message', this.onMessage);
    this.worker.removeEventListener('error', this.onError);
    this.worker.postMessage({ type: 'release' });
    this.worker.terminate();
  }

  private readonly onMessage = (event: MessageEvent<AgentWorkerMessage>): void => {
    const message = event.data;
    if (message.type === 'ready') {
      this.activeBackend = message.actualBackend;
      this.activeOrtWebVersion = message.ortWebVersion;
      this.readyResolve?.();
      this.readyResolve = null;
      this.readyReject = null;
      return;
    }
    if (message.type === 'result') {
      const request = this.pending.get(message.requestId);
      if (!request) return;
      this.pending.delete(message.requestId);
      request.resolve(message.result);
      return;
    }
    const error = new Error(message.message);
    if (message.requestId !== undefined) {
      const request = this.pending.get(message.requestId);
      if (request) {
        this.pending.delete(message.requestId);
        request.reject(error);
      }
      return;
    }
    this.readyReject?.(error);
    this.readyReject = null;
    this.readyResolve = null;
  };

  private readonly onError = (event: ErrorEvent): void => {
    const error = new Error(event.message || 'agent inference worker failed');
    this.readyReject?.(error);
    this.readyReject = null;
    this.readyResolve = null;
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  };
}
