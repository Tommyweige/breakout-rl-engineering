// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ mainLoad: vi.fn(), workers: [] as { load: ReturnType<typeof vi.fn>; release: ReturnType<typeof vi.fn>; actualBackend: string }[] }));
vi.mock('../inference/OrtWebPolicy', () => ({ OrtWebPolicy: class {
  load = mocks.mainLoad;
  release = vi.fn().mockResolvedValue(undefined);
  actualBackend = null;
} }));
vi.mock('../inference/AgentInferenceWorker', () => ({ AgentInferenceWorker: class {
  load = vi.fn().mockResolvedValue(undefined);
  release = vi.fn().mockResolvedValue(undefined);
  constructor(public actualBackend: string) { mocks.workers.push(this); }
} }));
vi.mock('../inference/webgpuSupport', () => ({ detectWebGpuSupport: vi.fn().mockResolvedValue({ supported: false }) }));
import { App } from './App';
let app: App | undefined;
afterEach(() => { app?.destroy(); app = undefined; mocks.workers.length = 0; vi.clearAllMocks(); vi.unstubAllGlobals(); });

describe('gameplay main-thread budget', () => {
  it('does not rewrite unchanged score text on every game tick', () => {
    const root = document.createElement('div');
    app = new App(root);
    app.mount();
    const observer = new MutationObserver(() => {});
    observer.observe(root.querySelector('[data-role="human-score"]')!, { childList: true });
    Reflect.get(app, 'setText').call(app, '[data-role="human-score"]', '0');
    expect(observer.takeRecords()).toHaveLength(0);
    Reflect.get(app, 'setText').call(app, '[data-role="human-score"]', '1');
    expect(observer.takeRecords()).toHaveLength(1);
    observer.disconnect();
  });
  it('loads WASM inference only in the existing worker and reuses its model', async () => {
    app = new App(document.createElement('div'));
    await Reflect.get(app, 'prepareGameplayPolicy').call(app);
    await Reflect.get(app, 'prepareGameplayPolicy').call(app);
    expect(mocks.mainLoad).not.toHaveBeenCalled();
    expect(mocks.workers).toHaveLength(1);
    expect(mocks.workers[0]!.load).toHaveBeenCalledTimes(1);
    expect(Reflect.get(app, 'gameplayBackend')).toBe('wasm');
  });
  it('does not paint the hidden AI canvas on mobile but still paints it when selected', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: true }));
    const root = document.createElement('div');
    app = new App(root);
    app.mount();
    const human = { rawRgb: new Uint8Array(160 * 210 * 3), isFinished: false, render: vi.fn(), dispose: vi.fn() };
    const agent = { isFinished: false, render: vi.fn(), dispose: vi.fn() };
    Reflect.set(app, 'humanEnvironment', human);
    Reflect.set(app, 'agentEnvironment', agent);
    Reflect.get(app, 'renderCanvases').call(app);
    expect(human.render).toHaveBeenCalledTimes(1);
    expect(agent.render).not.toHaveBeenCalled();
    root.querySelector<HTMLElement>('.panel-grid')!.dataset.view = 'agent';
    Reflect.get(app, 'renderCanvases').call(app);
    expect(human.render).toHaveBeenCalledTimes(1);
    expect(agent.render).toHaveBeenCalledTimes(1);
  });
});
