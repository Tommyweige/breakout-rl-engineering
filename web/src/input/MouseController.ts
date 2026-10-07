import { ATARI_SCREEN_HEIGHT, ATARI_SCREEN_WIDTH } from '../environment/atariPreprocessing';
import type { ActionMeaning } from '../inference/types';
import { detectPaddleCenterNormalized } from './paddleDetector';
import type { HumanPaddleCommand } from './paddleCommand';

export type MouseMotionState = 'LEFT' | 'RIGHT' | 'STOPPED';

/**
 * Maps the pointer directly to the Human paddle controller's absolute X.
 * Paddle detection is telemetry only; movement no longer waits for the paddle
 * detector or a proportional velocity controller.
 */
export class MouseController {
  private canvas: HTMLCanvasElement | null = null;
  private surfaces: HTMLElement[] = [];
  private activePointer: number | null = null;
  private enabled = false;
  private target: number | null = null;
  private targetChangedAtMs: number | null = null;
  private paddle: number | null = null;
  private state: MouseMotionState = 'STOPPED';
  private readonly detectorColumnHits = new Uint8Array(ATARI_SCREEN_WIDTH);

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (!this.enabled || !this.canvas) return;
    if (this.activePointer !== null && event.pointerId !== this.activePointer) return;
    const bounds = ((event.currentTarget as HTMLElement | null) ?? this.canvas).getBoundingClientRect();
    if (bounds.width <= 0) return;
    const nextTarget = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    if (this.target === null || Math.abs(nextTarget - this.target) > Number.EPSILON) this.targetChangedAtMs = now();
    this.target = nextTarget;
  };

  private readonly onPointerDown = (event: PointerEvent): void => {
    if (!this.enabled || this.activePointer !== null) return;
    this.activePointer = event.pointerId;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    this.onPointerMove(event);
  };

  private readonly onPointerEnd = (event: PointerEvent): void => {
    if (event.pointerId !== this.activePointer) return;
    this.clear();
  };

  private readonly onPointerLeave = (): void => {
    if (this.activePointer !== null) return;
    this.target = null;
    this.targetChangedAtMs = null;
    this.state = 'STOPPED';
  };

  attach(canvas: HTMLCanvasElement, touchPad?: HTMLElement): void {
    if (this.canvas === canvas) return;
    this.detach();
    this.canvas = canvas;
    this.surfaces = touchPad ? [canvas, touchPad] : [canvas];
    for (const surface of this.surfaces) {
      surface.addEventListener('pointerdown', this.onPointerDown);
      surface.addEventListener('pointermove', this.onPointerMove);
      surface.addEventListener('pointerleave', this.onPointerLeave);
      surface.addEventListener('pointerup', this.onPointerEnd);
      surface.addEventListener('pointercancel', this.onPointerEnd);
      surface.addEventListener('lostpointercapture', this.onPointerEnd);
    }
  }

  detach(): void {
    if (!this.canvas) return;
    this.clear();
    for (const surface of this.surfaces) {
      surface.removeEventListener('pointerdown', this.onPointerDown);
      surface.removeEventListener('pointermove', this.onPointerMove);
      surface.removeEventListener('pointerleave', this.onPointerLeave);
      surface.removeEventListener('pointerup', this.onPointerEnd);
      surface.removeEventListener('pointercancel', this.onPointerEnd);
      surface.removeEventListener('lostpointercapture', this.onPointerEnd);
    }
    this.surfaces = [];
    this.canvas = null;
    this.clear();
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) this.clear();
  }

  clear(): void {
    const pointer = this.activePointer;
    this.activePointer = null;
    if (pointer !== null) {
      for (const surface of this.surfaces) {
        if (surface.hasPointerCapture(pointer)) surface.releasePointerCapture(pointer);
      }
    }
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

  get hasTarget(): boolean {
    return this.target !== null;
  }

  currentAction(): ActionMeaning {
    return this.currentCommand().direction;
  }

  peekAction(): ActionMeaning {
    return this.peekCommand().direction;
  }

  currentCommand(): HumanPaddleCommand {
    const command = this.resolveCommand();
    this.state = command.direction === 'NOOP' ? 'STOPPED' : command.direction;
    return command;
  }

  peekCommand(): HumanPaddleCommand {
    return this.resolveCommand();
  }

  private resolveCommand(): HumanPaddleCommand {
    const positionError = this.positionError;
    const errorRawPx = positionError === null ? null : positionError * ATARI_SCREEN_WIDTH;
    if (!this.enabled || errorRawPx === null || errorRawPx === 0) {
      return this.command('NOOP', positionError);
    }

    const direction = errorRawPx < 0 ? 'LEFT' : 'RIGHT';
    return this.command(direction, positionError);
  }

  private command(direction: HumanPaddleCommand['direction'], positionError: number | null): HumanPaddleCommand {
    return {
      direction,
      targetX: this.target,
      paddleCenterX: this.paddle,
      positionError,
    };
  }
}

function clamp(value: number, lower: number, upper: number): number {
  return Math.min(upper, Math.max(lower, value));
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
