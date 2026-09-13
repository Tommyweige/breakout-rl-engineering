import { describe, expect, it } from 'vitest';

import {
  createBrowserBreakoutEnvironmentForTest,
  createHumanBreakoutEnvironmentForTest,
  type AleLike,
} from './aleEnvironment';
import type { BreakoutContractV2 } from './breakoutContract';

const contract: BreakoutContractV2 = {
  schema_version: 2,
  contract_id: 'day15-breakout-evaluation-v2-fire-reset',
  environment_id: 'ALE/Breakout-v5',
  frame_skip: 4,
  frame_stack: 4,
  sticky_action_probability: 0.25,
  fire_reset: true,
  fire_reset_confirmation: {
    max_fire_attempts: 8,
    confirmation_steps: 2,
    min_observation_change_fraction: 0.0001,
    confirmation_operator: 'any',
    confirmation_signals: ['raw_reward', 'observation_activity_streak'],
  },
  terminal_on_life_loss: false,
  time_limit_semantics: {
    source: 'ale.game_truncated',
    max_num_frames_per_episode: 108000,
    agent_step_limit: 27000,
    truncated_is_finished: true,
    external_time_limit_wrapper: false,
  },
  concrete_episode_seeds: [101],
  evaluation_epsilon: 0,
  raw_reward_rule: 'sum environment rewards without clipping',
  sha256: 'a'.repeat(64),
  source_path: 'configs/eval/breakout_contract_v2.json',
  parity: { status: 'partial', unsupported_fields: [], note: 'test' },
};

class FakeAle {
  frame = 0;
  actions: number[] = [];
  settings = new Map<string, number | boolean>();
  loadPath = '';

  setBool(key: string, value: boolean): void { this.settings.set(key, value); }
  getBool(key: string): boolean { return Boolean(this.settings.get(key)); }
  setInt(key: string, value: number): void { this.settings.set(key, value); }
  getInt(key: string): number { return Number(this.settings.get(key) ?? 0); }
  setFloat(key: string, value: number): void { this.settings.set(key, value); }
  getFloat(key: string): number { return Number(this.settings.get(key) ?? 0); }
  setString(): void {}
  getString(): string { return ''; }
  loadROM(path: string): void { this.loadPath = path; }
  act(action: number): number { this.actions.push(action); this.frame += 1; return 0; }
  resetGame(): void { this.frame = 0; this.actions = []; }
  gameOver(): boolean { return false; }
  gameTruncated(): boolean { return false; }
  lives(): number { return 5; }
  getFrameNumber(): number { return this.frame; }
  getEpisodeFrameNumber(): number { return this.frame; }
  getScreenRGB(): Uint8ClampedArray { return new Uint8ClampedArray(160 * 210 * 3).fill(this.frame % 255); }
  getScreenGrayscale(): Uint8ClampedArray { return new Uint8ClampedArray(160 * 210).fill(this.frame % 255); }
  getScreenImageData(): ImageData { throw new Error('not used'); }
  renderToCanvas(): void {}
  getScreenWidth(): number { return 160; }
  getScreenHeight(): number { return 210; }
  getRAM(): Uint8Array { return new Uint8Array(128); }
  setRAM(): void {}
  getLegalActionSet(): number[] { return [0, 1, 3, 4]; }
  getMinimalActionSet(): number[] { return [0, 1, 3, 4]; }
  getAvailableModes(): number[] { return [0]; }
  setMode(): void {}
  getMode(): number { return 0; }
  getAvailableDifficulties(): number[] { return [0]; }
  setDifficulty(): void {}
  getDifficulty(): number { return 0; }
  saveState(): Uint8Array { return new Uint8Array(); }
  loadState(): void {}
}

describe('ALE Browser environment contract', () => {
  it('owns four raw frames per decision and auto-FIREs only the serve phase', () => {
    const ale = new FakeAle();
    const environment = createBrowserBreakoutEnvironmentForTest(ale as unknown as AleLike, contract, 101);

    const first = environment.step(2);
    const second = environment.step(2);
    const third = environment.step(2);

    expect(ale.loadPath).toBe('/roms/breakout.bin');
    expect(ale.getInt('frame_skip')).toBe(1);
    expect(ale.getFloat('repeat_action_probability')).toBe(0.25);
    expect(first.actualEmulatorFrames).toBe(4);
    expect(first.outerActionRepeat).toBe(4);
    expect(first.autoFire).toBe(true);
    expect(second.autoFire).toBe(true);
    expect(third.autoFire).toBe(false);
    expect(first.executedAleAction).toBe(1);
    expect(third.requestedAleAction).toBe(3);
    expect(third.executedAleAction).toBe(3);
    expect(ale.actions).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3]);
  });

  it('keeps the gameplay async Agent step at four raw frames while yielding between frames', async () => {
    const ale = new FakeAle();
    const environment = createBrowserBreakoutEnvironmentForTest(ale as unknown as AleLike, contract, 101);

    const result = await environment.stepAsync(2);

    expect(result.actualEmulatorFrames).toBe(4);
    expect(result.outerActionRepeat).toBe(4);
    expect(ale.actions).toEqual([1, 1, 1, 1]);
  });

  it('keeps Human runtime at one raw frame with sticky actions disabled and exposes the latest RGB frame', () => {
    const ale = new FakeAle();
    const environment = createHumanBreakoutEnvironmentForTest(ale as unknown as AleLike, contract, 101);

    const first = environment.step(2);
    const second = environment.step(2);
    const third = environment.step(2);
    const fourth = environment.step(0);
    const fifth = environment.step(3);
    const sixth = environment.step(0);

    expect(ale.getFloat('repeat_action_probability')).toBe(0);
    expect(first.actualEmulatorFrames).toBe(1);
    expect(second.actualEmulatorFrames).toBe(1);
    expect(first.outerActionRepeat).toBe(1);
    expect(first.rawFrameNumber).toBe(1);
    expect(first.rawRgb[0]).toBe(1);
    expect(first.autoFire).toBe(true);
    expect(second.autoFire).toBe(true);
    expect(third.autoFire).toBe(false);
    expect(third.requestedAction).toBe('RIGHT');
    expect(third.executedAction).toBe('RIGHT');
    expect(fourth.executedAction).toBe('NOOP');
    expect(fifth.executedAction).toBe('LEFT');
    expect(sixth.executedAction).toBe('NOOP');
    expect(ale.actions).toEqual([1, 1, 3, 0, 4, 0]);
    expect(environment.runtimeDiagnostics).toMatchObject({
      runtimeMode: 'interactive-human',
      rawFrameRepeat: 1,
      stickyActionProbability: 0,
      usesModelPreprocessing: false,
    });
  });
});
