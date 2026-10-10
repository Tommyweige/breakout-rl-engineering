import { afterEach, describe, expect, it, vi } from 'vitest';
import { LayaVisionPolicy } from './LayaVisionPolicy';

afterEach(() => vi.unstubAllGlobals());

describe('local Laya decision policy', () => {
  function responses(rows: { actionIndex: number; probabilities: number[] | null }[]) {
    const fetch = vi.fn();
    for (const row of rows) fetch.mockResolvedValueOnce(new Response(JSON.stringify({
      ...row, action: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'][row.actionIndex], device: 'cuda:0',
    })));
    vi.stubGlobal('fetch', fetch);
  }

  it('brakes a weak direction reversal for one decision and preserves the raw model choice', async () => {
    responses([
      { actionIndex: 3, probabilities: [0.2502, 0.2206, 0.2405, 0.2887] },
      { actionIndex: 2, probabilities: [0.2255, 0.2446, 0.2997, 0.2302] },
      { actionIndex: 2, probabilities: [0.2255, 0.2446, 0.2997, 0.2302] },
    ]);
    const policy = new LayaVisionPolicy();
    const frame = new Uint8Array(160 * 210 * 3);
    expect((await policy.infer(frame)).action).toBe('LEFT');
    expect(await policy.infer(frame)).toMatchObject({ action: 'NOOP', actionIndex: 0, greedyActionIndex: 2,
      qValues: [0.2255, 0.2446, 0.2997, 0.2302], mistakeInjected: false });
    expect((await policy.infer(frame)).action).toBe('RIGHT');
  });

  it('allows a strong reversal immediately and never delays FIRE', async () => {
    responses([
      { actionIndex: 2, probabilities: [0.15, 0.15, 0.4, 0.3] },
      { actionIndex: 3, probabilities: [0.15, 0.15, 0.25, 0.45] },
      { actionIndex: 1, probabilities: [0.1, 0.4, 0.2, 0.3] },
    ]);
    const policy = new LayaVisionPolicy();
    const frame = new Uint8Array(160 * 210 * 3);
    expect((await policy.infer(frame)).action).toBe('RIGHT');
    expect((await policy.infer(frame)).action).toBe('LEFT');
    expect((await policy.infer(frame)).action).toBe('FIRE');
  });

  it('clears directional memory when probabilities are absent or the episode resets', async () => {
    responses([
      { actionIndex: 2, probabilities: [0.2, 0.2, 0.31, 0.29] },
      { actionIndex: 3, probabilities: null },
      { actionIndex: 2, probabilities: [0.2, 0.2, 0.31, 0.29] },
      { actionIndex: 3, probabilities: [0.2, 0.2, 0.29, 0.31] },
    ]);
    const policy = new LayaVisionPolicy();
    const frame = new Uint8Array(160 * 210 * 3);
    expect((await policy.infer(frame)).action).toBe('RIGHT');
    expect((await policy.infer(frame)).action).toBe('LEFT');
    expect((await policy.infer(frame)).action).toBe('RIGHT');
    policy.reset();
    expect((await policy.infer(frame)).action).toBe('LEFT');
  });

  it('reports service mode and timing without confusing them with browser roundtrip', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      actionIndex: 3, action: 'LEFT', probabilities: null, device: 'cuda:0',
      inferenceMode: 'fixed-eager', inferenceMs: 65,
    }))));
    expect(await new LayaVisionPolicy().infer(new Uint8Array(160 * 210 * 3))).toMatchObject({
      inferenceMode: 'fixed-eager', serviceInferenceMs: 65,
    });
  });

  it.each([{ inferenceMs: -1 }, { inferenceMode: 'fixed-cuda-graph' }])('rejects invalid telemetry %j', async (telemetry) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      actionIndex: 3, action: 'LEFT', probabilities: null, device: 'cuda:0', ...telemetry,
    }))));
    await expect(new LayaVisionPolicy().infer(new Uint8Array(160 * 210 * 3))).rejects.toThrow('invalid');
  });
  it('sends one RGB frame and preserves the model choice without difficulty noise', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      actionIndex: 3, action: 'LEFT', probabilities: [0.2, 0.1, 0.3, 0.4], device: 'cuda:0',
    })));
    vi.stubGlobal('fetch', fetch);
    const frame = new Uint8Array(160 * 210 * 3);
    const result = await new LayaVisionPolicy().infer(frame);
    expect(fetch.mock.calls[0]?.[1].body).toEqual(frame);
    expect(result).toMatchObject({ actionIndex: 3, action: 'LEFT', actualBackend: 'cuda', mistakeInjected: false });
  });

  it('fails visibly when the local service is unavailable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await expect(new LayaVisionPolicy().load()).rejects.toThrow('local Laya');
  });

  it('keeps valid model choices when optional probabilities are absent', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      actionIndex: 2, action: 'RIGHT', probabilities: null, device: 'cuda:0',
    }))));
    expect(await new LayaVisionPolicy().infer(new Uint8Array(160 * 210 * 3))).toMatchObject({
      actionIndex: 2, action: 'RIGHT', qValues: [], actualBackend: 'cuda',
    });
  });

  it('rejects malformed frames and inconsistent action responses', async () => {
    const policy = new LayaVisionPolicy();
    await expect(policy.infer(new Uint8Array(4))).rejects.toThrow('RGB');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      actionIndex: 2, action: 'LEFT', probabilities: [0.25, 0.25, 0.25, 0.25], device: 'cuda:0',
    }))));
    await expect(policy.infer(new Uint8Array(160 * 210 * 3))).rejects.toThrow('invalid');
  });
});
