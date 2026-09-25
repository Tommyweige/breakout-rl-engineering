import { describe, expect, it } from 'vitest';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import createALEModule from '@farama/ale-wasm';

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
