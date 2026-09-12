// @vitest-environment jsdom

import { afterEach, describe, expect, it } from 'vitest';

import { App } from './App';

let app: App | undefined;

afterEach(() => {
  app?.destroy();
  app = undefined;
  document.body.innerHTML = '';
});

describe('Day 29 dual ALE browser product', () => {
  it('exposes both real-game canvases and the shared controls', () => {
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();

    expect(root.textContent).toContain('Human Panel');
    expect(root.textContent).toContain('Agent Panel');
    expect(root.textContent).toContain('ALE / HUMAN_01');
    expect(root.textContent).toContain('ALE / RL_AGENT_01');
    expect(root.querySelectorAll('canvas')).toHaveLength(7);
    expect(root.querySelector('[data-action="start"]')).not.toBeNull();
    expect(root.querySelector('[data-action="pause"]')).not.toBeNull();
    expect(root.querySelector('[data-action="reset"]')).not.toBeNull();
  });

  it('drives Start, Pause, and Reset through one shared runtime status', () => {
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();
    const status = root.querySelector<HTMLElement>('[data-role="runtime-status"]')!;

    root.querySelector<HTMLButtonElement>('[data-action="start"]')!.click();
    expect(status.textContent).toBe('running');
    root.querySelector<HTMLButtonElement>('[data-action="pause"]')!.click();
    expect(status.textContent).toBe('paused');
    root.querySelector<HTMLButtonElement>('[data-action="reset"]')!.click();
    expect(status.textContent).toBe('ready');
  });
});
