// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { App } from './App';

let app: App | undefined;

beforeEach(() => {
  window.history.replaceState({}, '', '/');
});

afterEach(() => {
  app?.destroy();
  app = undefined;
  document.body.innerHTML = '';
  window.history.replaceState({}, '', '/');
});

describe('Day 30 Human vs AI browser product', () => {
  it('exposes both real-game canvases, user controls, and input mode selection', () => {
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();

    expect(root.textContent).toContain('YOU');
    expect(root.textContent).toContain('AI');
    expect(root.textContent).toContain('PLAYER 01');
    expect(root.textContent).not.toContain('RL AGENT 01');
    expect(root.textContent).not.toContain('Technical details');
    expect(root.textContent).not.toContain('Q-values');
    expect(root.querySelectorAll('canvas')).toHaveLength(2);
    expect(root.querySelector('[data-action="start"]')).not.toBeNull();
    expect(root.querySelector('[data-action="pause"]')).not.toBeNull();
    expect(root.querySelector('[data-action="reset"]')).not.toBeNull();
    expect(root.querySelector('[data-action="input-mode"]')).not.toBeNull();
    expect(root.querySelector('[data-action="difficulty"]')).not.toBeNull();
    expect((root.querySelector('[data-action="difficulty"]') as HTMLSelectElement).value).toBe('hard');
    expect(root.querySelector('[data-role="technical-details"]')).toBeNull();
  });

  it('switches AI difficulty live without resetting the player surface', () => {
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();

    const difficulty = root.querySelector<HTMLSelectElement>('[data-action="difficulty"]')!;
    difficulty.value = 'medium';
    difficulty.dispatchEvent(new Event('change', { bubbles: true }));

    expect(root.querySelector('[data-role="ai-difficulty-label"]')?.textContent).toBe('MEDIUM');
    expect(root.querySelector('[data-role="user-message"]')?.textContent).toContain('MEDIUM');
    expect(root.querySelectorAll('canvas')).toHaveLength(2);
  });

  it('switches Mouse back to Keyboard without replacing or resetting the game surface', () => {
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();
    const humanCanvas = root.querySelector<HTMLCanvasElement>('[data-role="human-canvas"]');
    const inputMode = root.querySelector<HTMLSelectElement>('[data-action="input-mode"]')!;

    inputMode.value = 'mouse';
    inputMode.dispatchEvent(new Event('change', { bubbles: true }));
    expect(root.querySelector('[data-role="human-input-mode"]')?.textContent).toBe('Mouse');
    expect(root.querySelector('[data-role="input-hint"]')?.textContent).toContain('set a target');

    inputMode.value = 'keyboard';
    inputMode.dispatchEvent(new Event('change', { bubbles: true }));
    expect(root.querySelector('[data-role="human-input-mode"]')?.textContent).toBe('Keyboard');
    expect(root.querySelector<HTMLCanvasElement>('[data-role="human-canvas"]')).toBe(humanCanvas);
    expect(root.querySelector('[data-role="human-score"]')?.textContent).toBe('0');
  });

  it('exposes Mouse v3 diagnostics only on the debug route', () => {
    window.history.replaceState({}, '', '/?debug=1');
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();

    expect(root.querySelector('[data-role="cursor-target-x"]')).not.toBeNull();
    expect(root.querySelector('[data-role="paddle-center-x"]')).not.toBeNull();
    expect(root.querySelector('[data-role="motion-state"]')).not.toBeNull();
    expect(root.querySelector('[data-role="position-error"]')).not.toBeNull();
    expect(window.__mouseControlV3Diagnostics?.startThreshold).toBeCloseTo(6 / 160, 5);
  });

  it('drives Start, Pause, and Restart through one shared player message', () => {
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();
    root.querySelector<HTMLButtonElement>('[data-action="start"]')!.click();
    expect(root.querySelector('[data-role="user-message"]')?.textContent).toContain('Starting');
    root.querySelector<HTMLButtonElement>('[data-action="pause"]')!.click();
    expect(root.querySelector('[data-role="user-message"]')?.textContent).toContain('paused');
    root.querySelector<HTMLButtonElement>('[data-action="reset"]')!.click();
    expect(root.querySelector('[data-role="user-message"]')?.textContent).toContain('Restarting');
  });
});
