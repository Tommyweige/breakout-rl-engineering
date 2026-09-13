import { ATARI_SCREEN_HEIGHT, ATARI_SCREEN_WIDTH } from '../environment/atariPreprocessing';
import type { ActionMeaning } from '../inference/types';
import { detectPaddleCenterNormalized } from './paddleDetector';

/** The dead zone is intentionally expressed in Atari pixels, not screen CSS pixels. */
export const DEFAULT_MOUSE_DEADZONE_RAW_PX = 6;
export const DEFAULT_MOUSE_DEADZONE = DEFAULT_MOUSE_DEADZONE_RAW_PX / ATARI_SCREEN_WIDTH;

export type MouseMotionState = 'LEFT' | 'RIGHT' | 'STOPPED';

/**
 * Maps a visible absolute pointer target to one legal ALE action per raw frame.
 * The Human environment owns the 60 Hz cadence; this controller only closes
 * the position loop around the latest single raw RGB frame.
 */
export class MouseController {
  private canvas: HTMLCanvasElement | null = null;
  private enabled = false;
  private target: number | null = null;
  private targetChangedAtMs: number | null = null;
  private paddle: number | null = null;
  private state: MouseMotionState = 'STOPPED';
  private readonly detectorColumnHits = new Uint8Array(ATARI_SCREEN_WIDTH);

  constructor(private readonly deadzoneRawPx = DEFAULT_MOUSE_DEADZONE_RAW_PX) {
    if (!Number.isFinite(deadzoneRawPx) || deadzoneRawPx < 0) {
      throw new Error('Mouse dead zone must be a non-negative finite Atari-pixel value');
    }
  }

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (!this.enabled || !this.canvas) return;
    const bounds = this.canvas.getBoundingClientRect();
    if (bounds.width <= 0) return;
    const nextTarget = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    if (this.target === null || Math.abs(nextTarget - this.target) > Number.EPSILON) this.targetChangedAtMs = now();
    this.target = nextTarget;
  };

  private readonly onPointerLeave = (): void => {
    this.target = null;
    this.targetChangedAtMs = null;
    this.state = 'STOPPED';
  };

  attach(canvas: HTMLCanvasElement): void {
    if (this.canvas === canvas) return;
    this.detach();
    this.canvas = canvas;
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerleave', this.onPointerLeave);
  }

  detach(): void {
    if (!this.canvas) return;
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerleave', this.onPointerLeave);
    this.canvas = null;
    this.clear();
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) this.clear();
  }

  clear(): void {
    this.target = null;
    this.targetChangedAtMs = null;
    this.state = 'STOPPED';
  }

  updatePaddleCenter(normalizedX: number | null): void {
    if (normalizedX === null || !Number.isFinite(normalizedX)) return;
    this.paddle = clamp(normalizedX, 0, 1);
  }

  updatePaddleCenterFromFrame(rawRgb: ArrayLike<number>): void {
    this.updatePaddleCenter(detectPaddleCenterNormalized(rawRgb, ATARI_SCREEN_WIDTH, ATARI_SCREEN_HEIGHT, this.detectorColumnHits));
  }

  get targetX(): number | null {
    return this.target;
  }

  get paddleCenterX(): number | null {
    return this.paddle;
  }

  get targetChangedAt(): number | null {
    return this.targetChangedAtMs;
  }

  get motionState(): MouseMotionState {
    return this.state;
  }

  get positionError(): number | null {
    if (this.target === null || this.paddle === null) return null;
    return this.target - this.paddle;
  }

  get deadzoneRawPixels(): number {
    return this.deadzoneRawPx;
  }

  get deadzoneNormalized(): number {
    return this.deadzoneRawPx / ATARI_SCREEN_WIDTH;
  }

  get hasTarget(): boolean {
    return this.target !== null;
  }

  currentAction(): ActionMeaning {
    const action = this.resolveAction();
    this.state = action === 'LEFT' || action === 'RIGHT' ? action : 'STOPPED';
    return action;
  }

  peekAction(): ActionMeaning {
    return this.resolveAction();
  }

  private resolveAction(): ActionMeaning {
    if (!this.enabled || this.target === null || this.paddle === null) return 'NOOP';
    const error = (this.target - this.paddle) * ATARI_SCREEN_WIDTH;
    if (error < -this.deadzoneRawPx) return 'LEFT';
    if (error > this.deadzoneRawPx) return 'RIGHT';
    return 'NOOP';
  }
}

function clamp(value: number, lower: number, upper: number): number {
  return Math.min(upper, Math.max(lower, value));
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
