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

  it('returns proportional paddle strength in Atari-relative units', () => {
    const fixture = canvasFixture();
    const controller = new MouseController(3);
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    controller.updatePaddleCenter(0.5);

    moveTo(fixture.listeners, 211); // 8.8 Atari pixels right of paddle
    const small = controller.currentCommand();
    moveTo(fixture.listeners, 225); // 20 Atari pixels right of paddle
    const medium = controller.currentCommand();
    moveTo(fixture.listeners, 300); // far enough to saturate
    const full = controller.currentCommand();

    expect(small.direction).toBe('RIGHT');
    expect(small.strength).toBeGreaterThan(0);
    expect(small.strength).toBeLessThan(medium.strength);
    expect(medium.strength).toBeLessThan(1);
    expect(full).toMatchObject({ direction: 'RIGHT', strength: 1, targetX: 1, paddleCenterX: 0.5 });
    expect(full.positionError).toBe(0.5);
  });

  it('uses the same proportional strength for left corrections and remains neutral inside the dead zone', () => {
    const fixture = canvasFixture();
    const controller = new MouseController(3);
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    controller.updatePaddleCenter(0.5);

    moveTo(fixture.listeners, 203); // target 0.515, within the 3-pixel dead zone
    expect(controller.currentCommand()).toMatchObject({ direction: 'NOOP', strength: 0 });

    moveTo(fixture.listeners, 175);
    const left = controller.currentCommand();
    moveTo(fixture.listeners, 150);
    const fartherLeft = controller.currentCommand();

    expect(left.direction).toBe('LEFT');
    expect(left.strength).toBeGreaterThan(0);
    expect(fartherLeft.direction).toBe('LEFT');
    expect(fartherLeft.strength).toBeGreaterThan(left.strength);
    expect(fartherLeft.strength).toBe(1);
  });

  it('returns safe neutral input when the pointer or first paddle detection is unavailable', () => {
    const fixture = canvasFixture();
    const controller = new MouseController();
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    moveTo(fixture.listeners, 300);
    expect(controller.currentCommand()).toMatchObject({ direction: 'NOOP', strength: 0 });

    controller.updatePaddleCenter(0.5);
    moveTo(fixture.listeners, 211);
    const lastTrustedCommand = controller.currentCommand();
    controller.updatePaddleCenter(Number.NaN);
    controller.updatePaddleCenter(null);
    expect(controller.currentCommand()).toEqual(lastTrustedCommand);

    fixture.listeners.get('pointerleave')!(new Event('pointerleave'));
    expect(controller.currentCommand()).toMatchObject({ direction: 'NOOP', strength: 0, targetX: null });
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
