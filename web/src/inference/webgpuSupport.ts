import type { WebGpuSupport } from './types';

interface GpuAdapterInfoLike {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
}

interface GpuAdapterLike {
  info?: GpuAdapterInfoLike;
}

interface GpuLike {
  requestAdapter: () => Promise<GpuAdapterLike | null>;
}

export interface WebGpuAdapterProbe {
  support: WebGpuSupport;
  adapter: unknown | null;
}

function gpuFromNavigator(): GpuLike | undefined {
  if (typeof navigator === 'undefined') return undefined;
  return (navigator as Navigator & { gpu?: GpuLike }).gpu;
}

function readAdapterInfo(adapter: GpuAdapterLike | null): Record<string, string> {
  if (!adapter?.info) return {};
  return Object.fromEntries(
    Object.entries(adapter.info).filter(
      (entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1].length > 0,
    ),
  );
}

function secureContext(): boolean {
  return typeof window !== 'undefined' && window.isSecureContext;
}

function unavailable(
  error: string,
  navigatorGpu: boolean,
  adapterAvailable: boolean,
  adapterInfo: Record<string, string> = {},
): WebGpuSupport {
  return {
    supported: false,
    navigatorGpu,
    adapterAvailable,
    adapterInfo,
    isSecureContext: secureContext(),
    error,
  };
}

async function inspectWebGpuSupport(): Promise<WebGpuAdapterProbe> {
  const gpu = gpuFromNavigator();
  if (!gpu) {
    return { support: unavailable('navigator.gpu is unavailable in this browser', false, false), adapter: null };
  }

  let adapter: GpuAdapterLike | null;
  try {
    adapter = await gpu.requestAdapter();
  } catch (error) {
    return {
      support: unavailable(
        `navigator.gpu.requestAdapter() failed: ${error instanceof Error ? error.message : String(error)}`,
        true,
        false,
      ),
      adapter: null,
    };
  }
  if (!adapter) {
    return {
      support: unavailable('navigator.gpu.requestAdapter() returned no adapter', true, false),
      adapter: null,
    };
  }

  return {
    support: {
      supported: true,
      navigatorGpu: true,
      adapterAvailable: true,
      adapterInfo: readAdapterInfo(adapter),
      isSecureContext: secureContext(),
      error: null,
    },
    adapter,
  };
}

export async function detectWebGpuSupport(): Promise<WebGpuSupport> {
  const result = await inspectWebGpuSupport();
  return result.support;
}

export async function detectWebGpuAdapter(): Promise<WebGpuAdapterProbe> {
  return inspectWebGpuSupport();
}
