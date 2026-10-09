import { afterEach, describe, expect, it, vi } from 'vitest';
import { LayaVisionPolicy } from './LayaVisionPolicy';

afterEach(() => vi.unstubAllGlobals());

describe('local Laya decision policy', () => {
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
