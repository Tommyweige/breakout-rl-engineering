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
  it('maps responsive canvas coordinates to an absolute paddle target', () => {
    const fixture = canvasFixture();
    const controller = new MouseController();
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    controller.updatePaddleCenter(0.5);

    moveTo(fixture.listeners, 250);

    expect(controller.currentCommand()).toEqual({
      direction: 'RIGHT',
      targetX: 0.75,
      paddleCenterX: 0.5,
      positionError: 0.25,
    });
  });

  it('keeps the absolute target even when paddle detection is unavailable', () => {
    const fixture = canvasFixture();
    const controller = new MouseController();
    controller.attach(fixture.canvas);
    controller.setEnabled(true);

    moveTo(fixture.listeners, 260);

    expect(controller.currentCommand()).toEqual({
      direction: 'NOOP',
      targetX: 0.8,
      paddleCenterX: null,
      positionError: null,
    });
  });

  it('reports direction from observed position without scaling or dead-zone filtering', () => {
    const fixture = canvasFixture();
    const controller = new MouseController();
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    controller.updatePaddleCenter(0.5);

    moveTo(fixture.listeners, 201);
    expect(controller.currentCommand().direction).toBe('RIGHT');
    expect(controller.currentCommand().targetX).toBeCloseTo(0.505);

    controller.updatePaddleCenter(0.51);
    expect(controller.currentCommand().direction).toBe('LEFT');
  });

  it('uses the last valid detected paddle center for diagnostics', () => {
    const controller = new MouseController();
    controller.updatePaddleCenter(0.4);
    controller.updatePaddleCenter(Number.NaN);
    controller.updatePaddleCenter(null);

    expect(controller.paddleCenterX).toBe(0.4);
  });

  it('clears the target on pointer leave and input mode changes', () => {
    const fixture = canvasFixture();
    const controller = new MouseController();
    controller.attach(fixture.canvas);
    controller.setEnabled(true);
    controller.updatePaddleCenter(0.5);
    moveTo(fixture.listeners, 260);
    expect(controller.currentCommand().targetX).toBe(0.8);

    fixture.listeners.get('pointerleave')!(new Event('pointerleave'));
    expect(controller.currentCommand()).toMatchObject({ direction: 'NOOP', targetX: null });

    moveTo(fixture.listeners, 250);
    controller.setEnabled(false);
    expect(controller.targetX).toBeNull();
    expect(controller.motionState).toBe('STOPPED');
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
