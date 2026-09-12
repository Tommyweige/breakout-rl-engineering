import type { ALEInterface, ALEModule } from '@farama/ale-wasm';

import type { ActionMeaning } from '../inference/types';
import { mapModelActionToAle, validateMinimalActionSet, type AleActionCode, type MappedAction } from './actionMapping';
import {
  ATARI_SCREEN_HEIGHT,
  ATARI_SCREEN_WIDTH,
  maxPoolGrayscaleFrames,
  preprocessGrayscaleFrame,
} from './atariPreprocessing';
import { loadBreakoutContract, type BreakoutContractV2 } from './breakoutContract';
import { FrameStack } from './frameStack';
import {
  fallbackNoopCount,
  NATIVE_ATARI_PREPROCESSING,
  type NativeNoopResetManifest,
} from './nativeAtariPreprocessing';

export interface AleLike extends ALEInterface {}

export interface EnvironmentStep {
  observation: Uint8Array;
  processedFrame: Uint8Array;
  rawRgb: Uint8Array;
  requestedModelAction: number;
  requestedAction: ActionMeaning;
  requestedAleAction: AleActionCode;
  executedModelAction: number;
  executedAction: ActionMeaning;
  executedAleAction: AleActionCode;
  autoFire: boolean;
  autoFireReason: 'initial_serve' | 'after_life_loss' | null;
  fireConfirmation: 'reward' | 'observation_activity_streak' | null;
  observationChangedFraction: number;
  reward: number;
  episodeReturn: number;
  lives: number;
  frameNumber: number;
  agentStep: number;
  actualEmulatorFrames: number;
  rawFrameSkip: number;
  outerActionRepeat: number;
  terminated: boolean;
  truncated: boolean;
  gameOverReason: 'terminated' | 'time_limit' | null;
  timing: {
    aleStepMs: number;
    preprocessingMs: number;
    totalMs: number;
  };
}

export interface EnvironmentSnapshot {
  observation: Uint8Array;
  processedFrame: Uint8Array;
  rawRgb: Uint8Array;
  seed: number;
  episodeReturn: number;
  lives: number;
  frameNumber: number;
  agentStep: number;
  terminated: boolean;
  truncated: boolean;
  needsFire: boolean;
}

export interface BrowserBreakoutEnvironmentOptions {
  contract: BreakoutContractV2;
  seed: number;
}

export interface PreprocessingTraceStep {
  rawGrayscaleFrames: number[][];
  pooledGrayscale: number[];
  processedFrame: number[];
  observation: number[];
  requestedModelAction: number;
  executedModelAction: number;
  autoFire: boolean;
  autoFireReason: EnvironmentStep['autoFireReason'];
}

export interface PreprocessingTrace {
  seed: number;
  nativeNpSeed: number;
  nativeAleSeed: number;
  noopMax: number;
  noopCount: number;
  noopSource: 'native_manifest' | 'seeded_fallback' | 'test_override';
  resetNoopGrayscaleFrames: number[][];
  resetGrayscaleFrame: number[];
  resetProcessedFrame: number[];
  resetObservation: number[];
  steps: PreprocessingTraceStep[];
}

let aleModulePromise: Promise<ALEModule> | null = null;
interface NativeSeedConfig {
  noopCount: number;
  npSeed: number;
  aleSeed: number;
}

let nativeNoopManifestPromise: Promise<ReadonlyMap<number, NativeSeedConfig>> | null = null;

export class BrowserBreakoutEnvironment {
  private static nextInstanceId = 1;
  readonly instanceId = BrowserBreakoutEnvironment.nextInstanceId++;
  private readonly frameStack: FrameStack;
  private seed: number;
  private lastRawRgb: Uint8Array;
  private lastProcessedFrame: Uint8Array;
  private lastObservation: Uint8Array;
  private lastLives = 0;
  private needsFire = false;
  private pendingFireReason: EnvironmentStep['autoFireReason'] = null;
  private fireAttempts = 0;
  private fireActivityStreak = 0;
  private episodeReturn = 0;
  private agentStep = 0;
  private terminated = false;
  private truncated = false;
  private disposed = false;
  private lastResetNoopCount = 0;
  private lastResetNoopSource: PreprocessingTrace['noopSource'] = 'seeded_fallback';
  private preprocessingTraceEnabled = false;
  private traceResetNoopGrayscaleFrames: number[][] = [];
  private traceReset: {
    resetGrayscaleFrame: number[];
    resetProcessedFrame: number[];
    resetObservation: number[];
  } | null = null;
  private traceSteps: PreprocessingTraceStep[] = [];

  constructor(
    private readonly ale: AleLike,
    readonly contract: BreakoutContractV2,
    readonly aleVersion: string,
    seed: number,
    private readonly nativeSeedConfigs: ReadonlyMap<number, NativeSeedConfig> = new Map(),
  ) {
    this.frameStack = new FrameStack(contract.frame_stack, 84 * 84);
    this.seed = seed;
    this.lastRawRgb = new Uint8Array(ATARI_SCREEN_WIDTH * ATARI_SCREEN_HEIGHT * 3);
    this.lastProcessedFrame = new Uint8Array(84 * 84);
    this.lastObservation = new Uint8Array(contract.frame_stack * 84 * 84);
  }

  static async create(options: BrowserBreakoutEnvironmentOptions): Promise<BrowserBreakoutEnvironment> {
    const module = await loadAleModule();
    const nativeSeedConfigs = await loadNativeSeedConfigs();
    const ale = new module.ALEInterface();
    configureAle(ale, options.contract, options.seed, nativeSeedConfigs.get(options.seed)?.aleSeed);
    ale.loadROM('/roms/breakout.bin');
    validateMinimalActionSet(ale.getMinimalActionSet());
    const environment = new BrowserBreakoutEnvironment(
      ale,
      options.contract,
      module.ALEInterface.getVersion(),
      options.seed,
      nativeSeedConfigs,
    );
    environment.reset(options.seed);
    return environment;
  }

  static async createFromCanonicalContract(seed: number): Promise<BrowserBreakoutEnvironment> {
    return BrowserBreakoutEnvironment.create({ contract: await loadBreakoutContract(), seed });
  }

  get observation(): Uint8Array {
    return new Uint8Array(this.lastObservation);
  }

  get processedFrame(): Uint8Array {
    return new Uint8Array(this.lastProcessedFrame);
  }

  get rawRgb(): Uint8Array {
    return new Uint8Array(this.lastRawRgb);
  }

  get currentSeed(): number {
    return this.seed;
  }

  get isFinished(): boolean {
    return this.terminated || this.truncated;
  }

  get currentReturn(): number {
    return this.episodeReturn;
  }

  get currentLives(): number {
    return this.lastLives;
  }

  get currentAgentStep(): number {
    return this.agentStep;
  }

  get runtimeDiagnostics(): Record<string, unknown> {
    return {
      instanceId: this.instanceId,
      requestedEnvironment: this.contract.environment_id,
      aleVersion: this.aleVersion,
      rawAleFrameSkip: this.ale.getInt('frame_skip'),
      outerActionRepeat: this.contract.frame_skip,
      expectedEmulatorFramesPerDecision: this.ale.getInt('frame_skip') * this.contract.frame_skip,
      stickyActionProbability: this.ale.getFloat('repeat_action_probability'),
      noopMax: NATIVE_ATARI_PREPROCESSING.noopMax,
      resetNoopCount: this.lastResetNoopCount,
      resetNoopSource: this.lastResetNoopSource,
      nativeNpSeed: this.nativeSeedConfigs.get(this.seed)?.npSeed ?? this.seed,
      nativeAleSeed: this.nativeSeedConfigs.get(this.seed)?.aleSeed ?? this.seed,
      nativeAleSeedSource: this.nativeSeedConfigs.has(this.seed) ? 'native_manifest' : 'seeded_fallback',
      grayscaleBeforeMaxPool: true,
      resizeInterpolation: NATIVE_ATARI_PREPROCESSING.resizeInterpolation,
      fireReset: this.contract.fire_reset,
      terminalOnLifeLoss: this.contract.terminal_on_life_loss,
      timeLimitSource: this.contract.time_limit_semantics.source,
      maxRawFramesPerEpisode: this.contract.time_limit_semantics.max_num_frames_per_episode,
      agentStepLimit: this.contract.time_limit_semantics.agent_step_limit,
    };
  }

  reset(seed?: number): EnvironmentSnapshot {
    this.assertActive();
    const targetSeed = seed ?? this.seed;
    if (!Number.isInteger(targetSeed)) throw new Error(`environment seed must be an integer, got ${targetSeed}`);
    this.seed = targetSeed;
    const nativeSeedConfig = this.nativeSeedConfigs.get(targetSeed);
    if (seed !== undefined) {
      // Gymnasium AtariEnv.reset(seed=...) calls seed_game(), load_game(),
      // then reset_game(). Reloading the ROM here mirrors that seed-bearing
      // path; a reset without a seed only resets the current game state.
      this.ale.setInt('random_seed', nativeSeedConfig?.aleSeed ?? targetSeed);
      this.ale.loadROM('/roms/breakout.bin');
    }
    this.ale.resetGame();
    this.traceResetNoopGrayscaleFrames = [];
    this.traceSteps = [];
    const noopCount = nativeSeedConfig?.noopCount ?? fallbackNoopCount(targetSeed);
    this.lastResetNoopCount = noopCount;
    this.lastResetNoopSource = nativeSeedConfig === undefined
      ? (this.nativeSeedConfigs.size === 0 ? 'test_override' : 'seeded_fallback')
      : 'native_manifest';
    let completedNoops = 0;
    while (completedNoops < noopCount) {
      this.ale.act(0);
      completedNoops += 1;
      if (this.preprocessingTraceEnabled) {
        this.traceResetNoopGrayscaleFrames.push(Array.from(this.ale.getScreenGrayscale(), (value) => value));
      }
      if (this.ale.gameOver() || this.ale.gameTruncated()) {
        this.ale.setInt('random_seed', nativeSeedConfig?.aleSeed ?? targetSeed);
        this.ale.loadROM('/roms/breakout.bin');
        this.ale.resetGame();
      }
    }
    this.lastRawRgb = copyBytes(this.ale.getScreenRGB());
    const resetGrayscale = copyBytes(this.ale.getScreenGrayscale());
    this.lastProcessedFrame = preprocessGrayscaleFrame(resetGrayscale);
    this.lastObservation = this.frameStack.reset(this.lastProcessedFrame);
    this.lastLives = this.ale.lives();
    this.needsFire = true;
    this.pendingFireReason = 'initial_serve';
    this.fireAttempts = 0;
    this.fireActivityStreak = 0;
    this.episodeReturn = 0;
    this.agentStep = 0;
    this.terminated = false;
    this.truncated = false;
    if (this.preprocessingTraceEnabled) {
      this.traceReset = {
        resetGrayscaleFrame: Array.from(resetGrayscale),
        resetProcessedFrame: Array.from(this.lastProcessedFrame),
        resetObservation: Array.from(this.lastObservation),
      };
    }
    return this.snapshot();
  }

  step(modelActionIndex: number): EnvironmentStep {
    this.assertActive();
    if (this.isFinished) throw new Error('cannot step a finished Breakout episode; reset first');
    const startedAt = now();
    const requested = mapModelActionToAle(modelActionIndex);
    const autoFire = this.needsFire;
    const autoFireReason = autoFire ? this.pendingFireReason : null;
    const executed = autoFire ? mapModelActionToAle(1) : requested;
    const beforeObservation = this.lastObservation;
    const beforeFrameNumber = this.ale.getFrameNumber();
    const aleStartedAt = now();
    let reward = 0;
    let secondLastFrame: Uint8Array | null = null;
    let lastFrame: Uint8Array | null = null;
    let secondLastGrayscale: Uint8Array | null = null;
    let lastGrayscale: Uint8Array | null = null;
    const sampledGrayscaleFrames: Uint8Array[] = [];
    let actualRawSteps = 0;

    for (let repeat = 0; repeat < this.contract.frame_skip; repeat += 1) {
      reward += this.ale.act(executed.aleAction);
      const frame = copyBytes(this.ale.getScreenRGB());
      const grayscale = copyBytes(this.ale.getScreenGrayscale());
      actualRawSteps += 1;
      secondLastFrame = lastFrame;
      lastFrame = frame;
      secondLastGrayscale = lastGrayscale;
      lastGrayscale = grayscale;
      sampledGrayscaleFrames.push(grayscale);
      if (this.ale.gameOver() || this.ale.gameTruncated()) break;
    }
    const aleStepMs = now() - aleStartedAt;
    const rawFrame = secondLastFrame && lastFrame ? maxPoolRgbFramesForRender(secondLastFrame, lastFrame) : lastFrame;
    if (!rawFrame) throw new Error('ALE returned no frame after act()');
    const pooledGrayscale = secondLastGrayscale && lastGrayscale
      ? maxPoolGrayscaleFrames(secondLastGrayscale, lastGrayscale)
      : lastGrayscale;
    if (!pooledGrayscale) throw new Error('ALE returned no grayscale frame after act()');
    const preprocessingStartedAt = now();
    const processedFrame = preprocessGrayscaleFrame(pooledGrayscale);
    const observation = this.frameStack.push(processedFrame);
    const preprocessingMs = now() - preprocessingStartedAt;
    const observationChangedFraction = changedFraction(beforeObservation, observation);

    let fireConfirmation: EnvironmentStep['fireConfirmation'] = null;
    if (autoFire) {
      this.fireAttempts += 1;
      if (reward !== 0) fireConfirmation = 'reward';
      if (observationChangedFraction >= this.contract.fire_reset_confirmation.min_observation_change_fraction) {
        this.fireActivityStreak += 1;
      } else {
        this.fireActivityStreak = 0;
      }
      if (!fireConfirmation && this.fireActivityStreak >= this.contract.fire_reset_confirmation.confirmation_steps) {
        fireConfirmation = 'observation_activity_streak';
      }
      if (fireConfirmation || this.ale.gameOver() || this.ale.gameTruncated()) {
        this.needsFire = false;
        this.pendingFireReason = null;
        this.fireAttempts = 0;
        this.fireActivityStreak = 0;
      } else if (this.fireAttempts >= this.contract.fire_reset_confirmation.max_fire_attempts) {
        throw new Error(`FIRE serve was not confirmed after ${this.fireAttempts} attempts for ${autoFireReason}`);
      }
    } else {
      this.fireActivityStreak = 0;
    }

    const lives = this.ale.lives();
    const lifeLoss = lives < this.lastLives;
    if (lifeLoss) {
      this.needsFire = true;
      this.pendingFireReason = 'after_life_loss';
      this.fireAttempts = 0;
      this.fireActivityStreak = 0;
    }
    this.lastLives = lives;
    this.lastRawRgb = new Uint8Array(rawFrame);
    this.lastProcessedFrame = new Uint8Array(processedFrame);
    this.lastObservation = new Uint8Array(observation);
    this.episodeReturn += reward;
    this.agentStep += 1;
    this.terminated = this.ale.gameOver();
    this.truncated = this.ale.gameTruncated();
    const totalMs = now() - startedAt;

    if (this.preprocessingTraceEnabled) {
      this.traceSteps.push({
        rawGrayscaleFrames: sampledGrayscaleFrames.map((frame) => Array.from(frame)),
        pooledGrayscale: Array.from(pooledGrayscale),
        processedFrame: Array.from(processedFrame),
        observation: Array.from(observation),
        requestedModelAction: requested.modelIndex,
        executedModelAction: executed.modelIndex,
        autoFire,
        autoFireReason,
      });
    }

    return {
      observation: new Uint8Array(observation),
      processedFrame: new Uint8Array(processedFrame),
      rawRgb: new Uint8Array(rawFrame),
      requestedModelAction: requested.modelIndex,
      requestedAction: requested.meaning,
      requestedAleAction: requested.aleAction,
      executedModelAction: executed.modelIndex,
      executedAction: executed.meaning,
      executedAleAction: executed.aleAction,
      autoFire,
      autoFireReason,
      fireConfirmation,
      observationChangedFraction,
      reward,
      episodeReturn: this.episodeReturn,
      lives,
      frameNumber: this.ale.getFrameNumber(),
      agentStep: this.agentStep,
      actualEmulatorFrames: this.ale.getFrameNumber() - beforeFrameNumber || actualRawSteps,
      rawFrameSkip: this.ale.getInt('frame_skip'),
      outerActionRepeat: this.contract.frame_skip,
      terminated: this.terminated,
      truncated: this.truncated,
      gameOverReason: this.truncated ? 'time_limit' : this.terminated ? 'terminated' : null,
      timing: { aleStepMs: aleStepMs, preprocessingMs, totalMs },
    };
  }

  enablePreprocessingTrace(): void {
    this.assertActive();
    this.preprocessingTraceEnabled = true;
    this.traceResetNoopGrayscaleFrames = [];
    this.traceReset = null;
    this.traceSteps = [];
  }

  getPreprocessingTrace(): PreprocessingTrace {
    this.assertActive();
    if (!this.preprocessingTraceEnabled || !this.traceReset) {
      throw new Error('preprocessing trace is not enabled or no reset has been captured');
    }
    return {
      seed: this.seed,
      nativeNpSeed: this.nativeSeedConfigs.get(this.seed)?.npSeed ?? this.seed,
      nativeAleSeed: this.nativeSeedConfigs.get(this.seed)?.aleSeed ?? this.seed,
      noopMax: NATIVE_ATARI_PREPROCESSING.noopMax,
      noopCount: this.lastResetNoopCount,
      noopSource: this.lastResetNoopSource,
      resetNoopGrayscaleFrames: this.traceResetNoopGrayscaleFrames.map((frame) => [...frame]),
      resetGrayscaleFrame: [...this.traceReset.resetGrayscaleFrame],
      resetProcessedFrame: [...this.traceReset.resetProcessedFrame],
      resetObservation: [...this.traceReset.resetObservation],
      steps: this.traceSteps.map((step) => ({
        ...step,
        rawGrayscaleFrames: step.rawGrayscaleFrames.map((frame) => [...frame]),
        pooledGrayscale: [...step.pooledGrayscale],
        processedFrame: [...step.processedFrame],
        observation: [...step.observation],
      })),
    };
  }

  render(canvas: HTMLCanvasElement): void {
    this.assertActive();
    canvas.width = ATARI_SCREEN_WIDTH;
    canvas.height = ATARI_SCREEN_HEIGHT;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('2D canvas context is unavailable');
    const image = context.createImageData(ATARI_SCREEN_WIDTH, ATARI_SCREEN_HEIGHT);
    for (let index = 0, pixel = 0; index < this.lastRawRgb.length; index += 3, pixel += 4) {
      image.data[pixel] = this.lastRawRgb[index] ?? 0;
      image.data[pixel + 1] = this.lastRawRgb[index + 1] ?? 0;
      image.data[pixel + 2] = this.lastRawRgb[index + 2] ?? 0;
      image.data[pixel + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }

  private snapshot(): EnvironmentSnapshot {
    return {
      observation: new Uint8Array(this.lastObservation),
      processedFrame: new Uint8Array(this.lastProcessedFrame),
      rawRgb: new Uint8Array(this.lastRawRgb),
      seed: this.seed,
      episodeReturn: this.episodeReturn,
      lives: this.lastLives,
      frameNumber: this.ale.getFrameNumber(),
      agentStep: this.agentStep,
      terminated: this.terminated,
      truncated: this.truncated,
      needsFire: this.needsFire,
    };
  }

  dispose(): void {
    if (this.disposed) return;
    const destroy = (this.ale as unknown as { delete?: () => void }).delete;
    destroy?.call(this.ale);
    this.disposed = true;
  }

  private assertActive(): void {
    if (this.disposed) throw new Error('ALE environment has been disposed');
  }
}

/** Public seam for deterministic contract tests without downloading the WASM module. */
export function createBrowserBreakoutEnvironmentForTest(
  ale: AleLike,
  contract: BreakoutContractV2,
  seed: number,
  aleVersion = 'test',
  nativeSeedConfigs: ReadonlyMap<number, NativeSeedConfig> = new Map([[seed, { noopCount: 0, npSeed: seed, aleSeed: seed }]]),
): BrowserBreakoutEnvironment {
  configureAle(ale, contract, seed);
  ale.loadROM('/roms/breakout.bin');
  validateMinimalActionSet(ale.getMinimalActionSet());
  const environment = new BrowserBreakoutEnvironment(ale, contract, aleVersion, seed, nativeSeedConfigs);
  environment.reset(seed);
  return environment;
}

function maxPoolRgbFramesForRender(first: ArrayLike<number>, second: ArrayLike<number>): Uint8Array {
  if (first.length !== second.length) throw new Error('max-pool frames must have equal lengths');
  const pooled = new Uint8Array(first.length);
  for (let index = 0; index < first.length; index += 1) {
    pooled[index] = Math.max(first[index] ?? 0, second[index] ?? 0);
  }
  return pooled;
}

export function changedFraction(previous: ArrayLike<number>, current: ArrayLike<number>): number {
  if (previous.length !== current.length || previous.length === 0) return 0;
  let changed = 0;
  for (let index = 0; index < previous.length; index += 1) {
    if ((previous[index] ?? 0) !== (current[index] ?? 0)) changed += 1;
  }
  return changed / previous.length;
}

async function loadAleModule(): Promise<ALEModule> {
  if (!aleModulePromise) {
    aleModulePromise = import('@farama/ale-wasm').then(async ({ default: createALEModule }) => {
      const module = await createALEModule({
        locateFile: (filename: string) => new URL(`/ale/${filename}`, window.location.href).toString(),
        print: () => undefined,
        printErr: () => undefined,
      });
      return module;
    });
  }
  return aleModulePromise;
}

async function loadNativeSeedConfigs(): Promise<ReadonlyMap<number, NativeSeedConfig>> {
  if (!nativeNoopManifestPromise) {
    nativeNoopManifestPromise = fetch('/fixtures/day29-native-noop-reset.json', { cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok) return new Map<number, NativeSeedConfig>();
        const manifest = (await response.json()) as NativeNoopResetManifest;
        const values = new Map<number, NativeSeedConfig>();
        for (const [seed, config] of Object.entries(manifest.seeds ?? {})) {
          if (
            Number.isInteger(Number(seed)) &&
            Number.isInteger(config?.noop_count) &&
            config.noop_count >= 1 &&
            config.noop_count <= manifest.noop_max &&
            Number.isInteger(config?.np_seed) &&
            Number.isInteger(config?.ale_seed)
          ) {
            values.set(Number(seed), {
              noopCount: config.noop_count,
              npSeed: config.np_seed,
              aleSeed: config.ale_seed,
            });
          }
        }
        return values;
      })
      .catch(() => new Map<number, NativeSeedConfig>());
  }
  return nativeNoopManifestPromise;
}

function configureAle(ale: AleLike, contract: BreakoutContractV2, seed: number, aleSeed = seed): void {
  ale.setBool('display_screen', false);
  ale.setInt('random_seed', aleSeed);
  // ALE itself advances one raw frame per act(). The outer controller owns the
  // contract's four-frame action repeat and the final-two-frame max-pool.
  ale.setInt('frame_skip', 1);
  ale.setFloat('repeat_action_probability', contract.sticky_action_probability);
  ale.setInt('max_num_frames_per_episode', contract.time_limit_semantics.max_num_frames_per_episode);
}

function copyBytes(values: ArrayLike<number>): Uint8Array {
  return Uint8Array.from(values, (value) => value);
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
