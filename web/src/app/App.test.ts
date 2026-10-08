// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';

let app: App | undefined;

beforeEach(() => {
  window.history.replaceState({}, '', '/');
});

afterEach(() => {
  app?.destroy();
  app = undefined;
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
  window.history.replaceState({}, '', '/');
});

describe('Day 30 Human vs AI browser product', () => {
  it('keeps the AI score beside the player in sync with the AI panel', () => {
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();
    Reflect.get(app, 'setText').call(app, '[data-role="agent-score"]', '42');
    Reflect.get(app, 'setText').call(app, '[data-role="agent-lives"]', '3');
    expect([...root.querySelectorAll('[data-role="agent-score"]')].map((el) => el.textContent)).toEqual(['42', '42']);
    expect([...root.querySelectorAll('[data-role="agent-lives"]')].map((el) => el.textContent)).toEqual(['3', '3']);
    expect(root.querySelector('[data-role="touch-pad"]')).toBeNull();
  });
  it('enables dragging when a desktop browser switches to the mobile layout', () => {
    let resize = () => {};
    const mobile = { matches: false, addEventListener: (_: string, listener: () => void) => { resize = listener; }, removeEventListener: vi.fn() };
    vi.stubGlobal('matchMedia', (query: string) => query === '(pointer: coarse)' ? { matches: false } : mobile);
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();
    const input = root.querySelector<HTMLSelectElement>('[data-action="input-mode"]')!;
    input.value = 'keyboard';
    input.dispatchEvent(new Event('change', { bubbles: true }));
    mobile.matches = true;
    resize();
    expect(input.value).toBe('mouse');
    expect(root.querySelector<HTMLCanvasElement>('[data-role="human-canvas"]')!.dataset.inputMode).toBe('mouse');
  });
  it('accepts arrow keys while pointer controls are selected', () => {
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();
    const input = root.querySelector<HTMLSelectElement>('[data-action="input-mode"]')!;
    input.value = 'mouse';
    input.dispatchEvent(new Event('change', { bubbles: true }));
    window.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowRight' }));
    const command = Reflect.get(app, 'currentHumanCommand').call(app);
    expect(command).toEqual({ kind: 'discrete', actionIndex: 2 });
    window.dispatchEvent(new KeyboardEvent('keyup', { code: 'ArrowRight' }));
    expect(Reflect.get(app, 'currentHumanCommand').call(app).kind).toBe('paddle');
  });
  it('defaults touch devices to pointer input and switches views without replacing either game', () => {
    vi.stubGlobal('matchMedia', () => ({ matches: true }));
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();
    const canvases = [...root.querySelectorAll('canvas')];
    expect(root.querySelector<HTMLSelectElement>('[data-action="input-mode"]')!.value).toBe('mouse');
    expect(root.querySelector('[data-role="human-input-mode"]')!.textContent).toBe('Touch');
    expect(root.querySelector<HTMLCanvasElement>('[data-role="human-canvas"]')!.dataset.inputMode).toBe('mouse');
    root.querySelector<HTMLButtonElement>('button[data-view="agent"]')!.click();
    expect(root.querySelector('.panel-grid')!.getAttribute('data-view')).toBe('agent');
    expect(root.querySelector('button[data-view="agent"]')!.getAttribute('aria-pressed')).toBe('true');
    root.querySelector<HTMLButtonElement>('button[data-view="human"]')!.click();
    expect(root.querySelector('.panel-grid')!.getAttribute('data-view')).toBe('human');
    expect([...root.querySelectorAll('canvas')]).toEqual(canvases);
  });
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
    expect(root.querySelector('[data-role="input-hint"]')?.textContent).toBe('Move mouse or use ← / → to move paddle');

    inputMode.value = 'keyboard';
    inputMode.dispatchEvent(new Event('change', { bubbles: true }));
    expect(root.querySelector('[data-role="human-input-mode"]')?.textContent).toBe('Keyboard');
    expect(root.querySelector<HTMLCanvasElement>('[data-role="human-canvas"]')).toBe(humanCanvas);
    expect(root.querySelector('[data-role="human-score"]')?.textContent).toBe('0');
  });

  it('clears Mouse input on pointer leave, mode switch, blur, pause, and reset', async () => {
    window.history.replaceState({}, '', '/?debug=1');
    const root = document.createElement('div');
    document.body.append(root);
    app = new App(root);
    app.mount();

    const humanCanvas = root.querySelector<HTMLCanvasElement>('[data-role="human-canvas"]')!;
    Object.defineProperty(humanCanvas, 'getBoundingClientRect', {
      value: () => ({ left: 100, width: 200, top: 0, height: 210, right: 300, bottom: 210, x: 100, y: 0, toJSON: () => ({}) }),
    });
    const inputMode = root.querySelector<HTMLSelectElement>('[data-action="input-mode"]')!;
    const targetAt = (clientX: number) => humanCanvas.dispatchEvent(new MouseEvent('pointermove', { clientX }));
    const waitForInputRender = () => new Promise((resolve) => window.setTimeout(resolve, 90));

    inputMode.value = 'mouse';
    inputMode.dispatchEvent(new Event('change', { bubbles: true }));
    targetAt(250);
    await waitForInputRender();
    expect(window.__mouseControlV3Diagnostics?.cursorTargetX).toBe(0.75);

    humanCanvas.dispatchEvent(new Event('pointerleave'));
    await waitForInputRender();
    expect(window.__mouseControlV3Diagnostics?.cursorTargetX).toBeNull();
    expect(window.__mouseControlV3Diagnostics?.requestedPaddlePositionX).toBeNull();

    targetAt(250);
    inputMode.value = 'keyboard';
    inputMode.dispatchEvent(new Event('change', { bubbles: true }));
    expect(window.__mouseControlV3Diagnostics?.cursorTargetX).toBeNull();

    inputMode.value = 'mouse';
    inputMode.dispatchEvent(new Event('change', { bubbles: true }));
    targetAt(250);
    await waitForInputRender();
    window.dispatchEvent(new Event('blur'));
    expect(window.__mouseControlV3Diagnostics?.cursorTargetX).toBeNull();

    targetAt(250);
    await waitForInputRender();
    root.querySelector<HTMLButtonElement>('[data-action="pause"]')!.click();
    expect(window.__mouseControlV3Diagnostics?.cursorTargetX).toBeNull();
    targetAt(250);
    await waitForInputRender();
    root.querySelector<HTMLButtonElement>('[data-action="reset"]')!.click();
    expect(window.__mouseControlV3Diagnostics?.cursorTargetX).toBeNull();
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
    expect(root.querySelector('[data-role="requested-paddle-position"]')).not.toBeNull();
    expect(root.querySelector('[data-role="applied-paddle-target"]')).not.toBeNull();
    expect(window.__mouseControlV3Diagnostics?.requestedPaddlePositionX).toBeNull();
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
