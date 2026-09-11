import type { InferenceSchedulerStatus } from './types';

export type SchedulerListener = (status: InferenceSchedulerStatus) => void;

export class InferenceScheduler<TInput, TOutput> {
  private status: InferenceSchedulerStatus = 'idle';
  private inFlight = false;
  private readonly listeners = new Set<SchedulerListener>();

  constructor(private readonly infer: (input: TInput) => Promise<TOutput>) {}

  get currentStatus(): InferenceSchedulerStatus {
    return this.status;
  }

  get isInFlight(): boolean {
    return this.inFlight;
  }

  subscribe(listener: SchedulerListener): () => void {
    this.listeners.add(listener);
    listener(this.status);
    return () => this.listeners.delete(listener);
  }

  start(): void {
    if (!this.inFlight) this.setStatus('running');
  }

  pause(): void {
    if (!this.inFlight) this.setStatus('paused');
  }

  reset(): void {
    if (this.inFlight) throw new Error('cannot reset an in-flight inference');
    this.setStatus('idle');
  }

  fail(): void {
    this.inFlight = false;
    this.setStatus('error');
  }

  async run(input: TInput): Promise<TOutput> {
    if (this.inFlight) throw new Error('inference already in flight');
    if (this.status === 'paused') throw new Error('inference scheduler is paused');

    this.inFlight = true;
    this.setStatus('in-flight');
    try {
      return await this.infer(input);
    } catch (error) {
      this.setStatus('error');
      throw error;
    } finally {
      this.inFlight = false;
      if (this.status === 'in-flight') this.setStatus('running');
    }
  }

  private setStatus(status: InferenceSchedulerStatus): void {
    this.status = status;
    this.listeners.forEach((listener) => listener(status));
  }
}
