import { describe, expect, it } from 'vitest';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import createALEModule from '@farama/ale-wasm';
import { detectPaddleCenterNormalized } from '../input/paddleDetector';

const alePackageDirectory = fileURLToPath(new URL('../../node_modules/@farama/ale-wasm/', import.meta.url));

describe('ALE analog paddle WASM integration', () => {
  it('keeps the discrete API and moves the paddle by different amounts at different strengths', async () => {
    const module = await createALEModule({
      locateFile: (filename: string) => resolve(alePackageDirectory, filename),
      print: () => undefined,
      printErr: () => undefined,
    });

    const slower = captureBreakoutFrame(module, 0.1);
    const faster = captureBreakoutFrame(module, 1);
    let differingPaddlePixels = 0;
    const paddleStart = 160 * 190 * 3;

    for (let offset = paddleStart; offset < slower.length; offset += 1) {
      if (slower[offset] !== faster[offset]) differingPaddlePixels += 1;
    }

    expect(differingPaddlePixels).toBeGreaterThan(0);
  });

  it('puts the Breakout paddle at the cursor target in the next raw frame', async () => {
    const module = await createALEModule({
      locateFile: (filename: string) => resolve(alePackageDirectory, filename),
      print: () => undefined,
      printErr: () => undefined,
    });
    const ale = new module.ALEInterface();
    ale.setInt('random_seed', 11);
    ale.setInt('frame_skip', 1);
    ale.setFloat('repeat_action_probability', 0);
    ale.loadROM('/roms/breakout.bin');
    ale.resetGame();
    ale.act(1);
    ale.act(1);

    try {
      const targets = [0.25, 0.5, 0.75];
      const positions = targets.map((targetX) => {
        ale.resetGame();
        ale.act(1);
        ale.act(1);
        ale.setBreakoutPaddlePosition(targetX);
        ale.act(0);
        return detectPaddleCenterNormalized(ale.getScreenRGB());
      });

      expect(positions.every((center) => center !== null)).toBe(true);
      for (const [index, targetX] of targets.entries()) {
        expect(Math.abs(positions[index]! - targetX)).toBeLessThanOrEqual(0.02);
      }
    } finally {
      (ale as unknown as { delete?: () => void }).delete?.();
    }
  });
});

function captureBreakoutFrame(module: Awaited<ReturnType<typeof createALEModule>>, strength: number): Uint8Array {
  const ale = new module.ALEInterface();
  ale.setInt('random_seed', 11);
  ale.setInt('frame_skip', 1);
  ale.setFloat('repeat_action_probability', 0);
  ale.loadROM('/roms/breakout.bin');
  ale.resetGame();
  ale.act(1);
  ale.act(1);
  for (let frame = 0; frame < 30; frame += 1) ale.actWithPaddleStrength(3, strength);
  const screen = new Uint8Array(ale.getScreenRGB());
  (ale as unknown as { delete?: () => void }).delete?.();
  return screen;
}
