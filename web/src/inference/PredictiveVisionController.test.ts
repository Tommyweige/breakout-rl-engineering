import { describe, expect, it } from 'vitest';
import { choosePaddleAction, PredictiveVisionController, reflectX } from './PredictiveVisionController';

// Same synthetic RGB court as tests/test_vision_controller.py::screen_fixture.
function screen(ball: readonly [number, number] | null, paddleLeft = 96, paddleWidth = 16): Uint8Array {
  const rgb = new Uint8Array(160 * 210 * 3);
  function rect(x: number, y: number, width: number, height: number, color: readonly number[]) {
    for (let py = y; py < y + height; py++) for (let px = x; px < Math.min(160, x + width); px++) rgb.set(color, (py * 160 + px) * 3);
  }
  rect(0, 17, 8, 176, [142, 142, 142]);
  rect(152, 17, 8, 176, [142, 142, 142]);
  rect(0, 17, 160, 15, [142, 142, 142]);
  rect(8, 57, 144, 18, [200, 72, 72]);
  rect(paddleLeft, 189, paddleWidth, 4, [200, 72, 72]);
  if (ball) rect(ball[0], ball[1], 2, 4, [200, 72, 72]);
  return rgb;
}

describe('Predictive Vision Controller v1 browser port', () => {
  it('settles inside the catch region but still aligns a narrower paddle', () => {
    const wide = new PredictiveVisionController();
    wide.select(screen([80, 140], 114));
    expect(wide.select(screen([82, 142], 114)).action).toBe('NOOP');
    const narrow = new PredictiveVisionController();
    narrow.select(screen([80, 140], 118, 8));
    expect(narrow.select(screen([82, 142], 118, 8)).action).toBe('RIGHT');
  });

  it('discards paddle motion history when the paddle is not observed', () => {
    const controller = new PredictiveVisionController();
    controller.select(screen([80, 140], 104));
    controller.select(screen([81, 141], 104, 0));
    expect(controller.select(screen([82, 142], 110)).action).toBe('RIGHT');
  });
  it('releases direction early so a moving paddle settles at a stable landing point', () => {
    const controller = new PredictiveVisionController();
    controller.select(screen([80, 140], 104));
    const action = controller.select(screen([82, 142], 110)).action;
    expect(controller.diagnostics.interceptX).toBe(126);
    expect(action).toBe('NOOP');
  });
  it.each([[1.5, 1], [1, 1.5], [0.5, 2]])('steers steadily on a quantized straight descent (vx=%s, vy=%s)', (vx, vy) => {
    const controller = new PredictiveVisionController();
    const actions: string[] = [];
    const intercepts: number[] = [];
    for (let frame = 0; frame < 16; frame++) {
      // Fractional motion is rounded by the screen; the paddle is stationary.
      const result = controller.select(screen([100 + Math.floor(frame * vx), 112 + Math.floor(frame * vy)], 64));
      if (frame >= 2) {
        actions.push(result.action);
        intercepts.push(controller.diagnostics.interceptX!);
      }
    }
    const movement = actions.filter(action => action !== 'NOOP');
    const reversals = movement.filter((action, i) => i > 0 && action !== movement[i - 1]).length;
    expect(reversals).toBe(0);
    expect(Math.max(...intercepts) - Math.min(...intercepts)).toBeLessThanOrEqual(1);
  });
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
    controller.select(screen([48, 102]));
    expect(controller.diagnostics.ball).toMatchObject({ vx: -2, vy: -2 });
    controller.reset();
    expect(controller.select(screen([60, 140])).action).toBe('NOOP');
    expect(controller.diagnostics.ball).toMatchObject({ vx: null, vy: null });
    expect(() => controller.select(new Uint8Array(4))).toThrow(/RGB frame/);
  });

  it('starts a fresh velocity segment after an impossible jump or a missing-frame gap', () => {
    const controller = new PredictiveVisionController();
    controller.select(screen([20, 100]));
    controller.select(screen([22, 102]));
    controller.select(screen(null));
    controller.select(screen([26, 106]));
    expect(controller.diagnostics.ball).toMatchObject({ vx: 2, vy: 2 });
    controller.select(screen([27, 107]));
    expect(controller.diagnostics.ball).toMatchObject({ vx: 1, vy: 1 });
    for (let i = 0; i < 5; i++) controller.select(screen(null));
    controller.select(screen([100, 120]));
    controller.select(screen([110, 122]));
    // A directly reacquired candidate can jump beyond the velocity trust limit.
    controller.select(screen([111, 123]));
    expect(controller.diagnostics.ball).toMatchObject({ vx: 1, vy: 1 });
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
