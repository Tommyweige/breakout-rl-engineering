import { OrtWebPolicy } from './OrtWebPolicy';
import type { InferenceBackend } from './types';

interface LoadMessage {
  type: 'load';
  backend: InferenceBackend;
}

interface InferMessage {
  type: 'infer';
  requestId: number;
  observation: ArrayBuffer;
}

interface ReleaseMessage {
  type: 'release';
}

type AgentWorkerRequest = LoadMessage | InferMessage | ReleaseMessage;

const workerScope = globalThis as unknown as {
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
  addEventListener: (type: 'message', listener: (event: MessageEvent<AgentWorkerRequest>) => void) => void;
};

let policy: OrtWebPolicy | null = null;

workerScope.addEventListener('message', (event) => {
  void handleMessage(event.data);
});

async function handleMessage(message: AgentWorkerRequest): Promise<void> {
  try {
    if (message.type === 'load') {
      policy = new OrtWebPolicy({ backend: message.backend });
      await policy.load();
      workerScope.postMessage({
        type: 'ready',
        actualBackend: policy.actualBackend,
        ortWebVersion: policy.ortWebVersion,
      });
      return;
    }
    if (message.type === 'infer') {
      if (!policy) throw new Error('agent inference worker is not loaded');
      const result = await policy.infer(new Uint8Array(message.observation));
      workerScope.postMessage({ type: 'result', requestId: message.requestId, result });
      return;
    }
    await policy?.release();
    policy = null;
  } catch (error) {
    workerScope.postMessage({
      type: 'error',
      requestId: message.type === 'infer' ? message.requestId : undefined,
      message: error instanceof Error ? error.message : String(error),
    });
  }
}
