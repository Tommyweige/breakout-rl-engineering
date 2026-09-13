import { describe, expect, it } from 'vitest';

import { MouseController } from './MouseController';

function pointerEvent(clientX: number): PointerEvent {
  return { clientX } as unknown as PointerEvent;
}

function canvasFixture() {
  const listeners = new Map<string, EventListener>();
  const canvas = {
    getBoundingClientRect: () => ({ left: 100, width: 200 }),
    addEventListener: (type: string, listener: EventListener) => listeners.set(type, listener),
    removeEventListener: (type: string) => listeners.delete(type),
  } as unknown as HTMLCanvasElement;
  return { canvas, listeners };
}

function moveTo(listeners: Map<string, EventListener>, clientX: number): void {
  listeners.get('pointermove')!(pointerEvent(clientX));
}

describe('MouseController', () => {
  it('uses a small Atari-pixel dead zone', () => {
    const fixture = canvasFixture();
    const controller = new MouseController(3);
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    controller.updatePaddleCenter(0.5);

    moveTo(fixture.listeners, 206); // target 0.53, 4.8 Atari pixels away
    expect(controller.currentAction()).toBe('RIGHT');
    controller.updatePaddleCenter(0.525); // 0.8 Atari pixels from target
    expect(controller.currentAction()).toBe('NOOP');
    expect(controller.motionState).toBe('STOPPED');
  });

  it('allows bidirectional correction on successive raw frames', () => {
    const fixture = canvasFixture();
    const controller = new MouseController(3);
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    controller.updatePaddleCenter(0.5);

    moveTo(fixture.listeners, 260);
    expect(controller.currentAction()).toBe('RIGHT');
    controller.updatePaddleCenter(0.85);
    expect(controller.currentAction()).toBe('LEFT');
    expect(controller.motionState).toBe('LEFT');
  });

  it('uses the latest trusted paddle center when detector input is missing', () => {
    const controller = new MouseController();
    controller.updatePaddleCenter(0.4);
    controller.updatePaddleCenter(null);

    expect(controller.paddleCenterX).toBe(0.4);
  });

  it('returns NOOP when the pointer leaves the canvas', () => {
    const fixture = canvasFixture();
    const controller = new MouseController();
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    controller.updatePaddleCenter(0.5);
    moveTo(fixture.listeners, 260);
    expect(controller.currentAction()).toBe('RIGHT');

    fixture.listeners.get('pointerleave')!(new Event('pointerleave'));
    expect(controller.targetX).toBeNull();
    expect(controller.currentAction()).toBe('NOOP');
  });

  it('clears target and motion state when switching input modes', () => {
    const fixture = canvasFixture();
    const controller = new MouseController();
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    controller.updatePaddleCenter(0.5);
    moveTo(fixture.listeners, 260);
    expect(controller.currentAction()).toBe('RIGHT');

    controller.setEnabled(false);
    expect(controller.targetX).toBeNull();
    expect(controller.motionState).toBe('STOPPED');
    expect(controller.currentAction()).toBe('NOOP');
  });

  it('does not expose or call Pointer Lock APIs', () => {
    const fixture = canvasFixture();
    const controller = new MouseController();
    controller.attach(fixture.canvas);
    controller.setEnabled(true);

    expect('requestPointerLock' in fixture.canvas).toBe(false);
    expect('pointerLockElement' in fixture.canvas).toBe(false);
  });
});
