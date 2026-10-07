import { describe, expect, it } from 'vitest';
import { choosePaddleAction, PredictiveVisionController, reflectX } from './PredictiveVisionController';

// Same synthetic RGB court as tests/test_vision_controller.py::screen_fixture.
function screen(ball: readonly [number, number] | null, paddleLeft = 96): Uint8Array {
  const rgb = new Uint8Array(160 * 210 * 3);
  function rect(x: number, y: number, width: number, height: number, color: readonly number[]) {
    for (let py = y; py < y + height; py++) for (let px = x; px < Math.min(160, x + width); px++) rgb.set(color, (py * 160 + px) * 3);
  }
  rect(0, 17, 8, 176, [142, 142, 142]);
  rect(152, 17, 8, 176, [142, 142, 142]);
  rect(0, 17, 160, 15, [142, 142, 142]);
  rect(8, 57, 144, 18, [200, 72, 72]);
  rect(paddleLeft, 189, 16, 4, [200, 72, 72]);
  if (ball) rect(ball[0], ball[1], 2, 4, [200, 72, 72]);
  return rgb;
}

describe('Predictive Vision Controller v1 browser port', () => {
  it('matches Python pixel tracking, reflected interception, loss and reacquisition', () => {
    const controller = new PredictiveVisionController();
    const positions = [[80, 120], [79, 121], null, [77, 123]] as const;
    expect(positions.map(ball => controller.select(screen(ball)).action)).toEqual(['NOOP', 'LEFT', 'LEFT', 'LEFT']);
    expect(controller.diagnostics).toMatchObject({ ball: { x: 77.5, y: 124.5, vx: -1, vy: 1 }, paddleX: 103.5, interceptX: 15 });
    for (let i = 0; i < 5; i++) controller.select(screen(null));
    expect(controller.diagnostics.ball).toBeNull();
    expect(controller.select(screen([20, 130])).action).toBe('NOOP');
    expect(controller.diagnostics.ball).toMatchObject({ vx: null, vy: null });
  });

  it('handles clipped paddles, bounce velocity and reset without stale tracking', () => {
    const controller = new PredictiveVisionController();
    expect(controller.select(screen(null, 144))).toMatchObject({ action: 'NOOP', actualBackend: 'vision', qValues: [] });
    expect(controller.diagnostics.paddleX).toBe(143.5);
    controller.select(screen([50, 100]));
    controller.select(screen([52, 102]));
    controller.select(screen([50, 104]));
    expect(controller.diagnostics.ball).toMatchObject({ vx: -2, vy: 2 });
    controller.reset();
    expect(controller.select(screen([60, 140])).action).toBe('NOOP');
    expect(controller.diagnostics.ball).toMatchObject({ vx: null, vy: null });
    expect(() => controller.select(new Uint8Array(4))).toThrow(/RGB frame/);
  });

  it('folds repeated wall bounces and preserves the original Schmitt deadband', () => {
    expect(reflectX(170, 10, 150)).toBe(130);
    expect(reflectX(-30, 10, 150)).toBe(50);
    expect(reflectX(730, 10, 150)).toBe(130);
    expect(choosePaddleAction(4, 'NOOP')).toBe('NOOP');
    expect(choosePaddleAction(4.1, 'NOOP')).toBe('RIGHT');
    expect(choosePaddleAction(2.1, 'RIGHT')).toBe('RIGHT');
    expect(choosePaddleAction(2, 'RIGHT')).toBe('NOOP');
    expect(choosePaddleAction(-4, 'RIGHT')).toBe('LEFT');
  });
});
