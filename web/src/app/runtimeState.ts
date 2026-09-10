import type { RuntimeStatus } from '../inference/types';

export type RuntimeEvent = 'load-start' | 'load-success' | 'load-error' | 'start' | 'pause' | 'reset';

export function reduceRuntimeStatus(status: RuntimeStatus, event: RuntimeEvent): RuntimeStatus {
  switch (event) {
    case 'load-start':
      return 'loading';
    case 'load-success':
      return 'ready';
    case 'load-error':
      return 'error';
    case 'start':
      return status === 'loading' ? status : 'running';
    case 'pause':
      return status === 'running' ? 'paused' : status;
    case 'reset':
      return status === 'loading' ? status : 'ready';
  }
}
