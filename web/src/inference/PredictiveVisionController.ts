import config from '../../../configs/eval/breakout_vision_controller_v1.json';
import { ACTION_MEANINGS, type ActionMeaning, type PolicyResult } from './types';

const WIDTH = 160;
const HEIGHT = 210;
interface Bounds { left: number; right: number; top: number; bottom: number }
interface Blob { x: number; y: number; left: number; top: number; width: number; height: number; area: number; confidence: number }
interface Ball { x: number; y: number; vx: number | null; vy: number | null; seen: number; confidence: number }

export function reflectX(x: number, left: number, right: number): number {
  const span = right - left;
  const offset = ((x - left) % (2 * span) + 2 * span) % (2 * span);
  return left + (offset <= span ? offset : 2 * span - offset);
}

export function choosePaddleAction(error: number, previous: ActionMeaning): ActionMeaning {
  const { deadband_pixels: deadband, hysteresis_pixels: hysteresis } = config.controller;
  const engage = deadband + hysteresis;
  const release = deadband - hysteresis;
  if (previous === 'RIGHT') return error <= -engage ? 'LEFT' : error > release ? 'RIGHT' : 'NOOP';
  if (previous === 'LEFT') return error >= engage ? 'RIGHT' : error < -release ? 'LEFT' : 'NOOP';
  return error > engage ? 'RIGHT' : error < -engage ? 'LEFT' : 'NOOP';
}

function detectBounds(rgb: Uint8Array): Bounds {
  const lit = (x: number, y: number) => {
    const i = (y * WIDTH + x) * 3;
    return Math.max(rgb[i]!, rgb[i + 1]!, rgb[i + 2]!) > 24 ? 1 : 0;
  };
  const y0 = Math.floor(HEIGHT * 0.12), y1 = Math.floor(HEIGHT * 0.94);
  const density = (x: number) => {
    let sum = 0;
    for (let y = y0; y < y1; y++) sum += lit(x, y);
    return sum / (y1 - y0);
  };
  let left = 0, right = WIDTH - 1;
  while (left < WIDTH / 4 && density(left) >= 0.78) left++;
  while (right >= Math.floor(WIDTH * 3 / 4) && density(right) >= 0.78) right--;
  if (!(left > 0 && right < WIDTH - 1 && right > left + 20)) {
    left = Math.round(WIDTH * 0.05);
    right = Math.round(WIDTH * 0.95) - 1;
  }
  let top = Math.round(HEIGHT * 0.15);
  for (let y = Math.floor(HEIGHT * 0.10); y < Math.floor(HEIGHT * 0.50); y++) {
    let sum = 0;
    for (let x = left + 2; x <= right - 2; x++) sum += lit(x, y);
    if (sum / (right - left - 3) < 0.12) { top = y; break; }
  }
  let bottom = Math.round(HEIGHT * 0.92);
  for (let y = Math.floor(HEIGHT * 0.55); y < HEIGHT; y++) {
    let sum = 0;
    for (let x = 0; x < left; x++) sum += lit(x, y);
    for (let x = right + 1; x < WIDTH; x++) sum += lit(x, y);
    if (sum / (left + WIDTH - right - 1) >= 0.78) bottom = y;
  }
  return { left, right, top, bottom: Math.min(Math.max(bottom, top + 20), HEIGHT - 1) };
}

// Eight-connected components replace OpenCV's connectedComponentsWithStats;
// centroid, bounding box, palette and acceptance thresholds match the Python controller.
function components(mask: Uint8Array, bounds: Bounds): Blob[] {
  const visited = new Uint8Array(mask.length);
  const queue = new Int32Array(mask.length);
  const blobs: Blob[] = [];
  for (let y = bounds.top; y <= bounds.bottom; y++) for (let x = bounds.left; x <= bounds.right; x++) {
    const index = y * WIDTH + x;
    if (!mask[index] || visited[index]) continue;
    let head = 0, tail = 1, sumX = 0, sumY = 0;
    let left = x, right = x, top = y, bottom = y;
    queue[0] = index;
    visited[index] = 1;
    while (head < tail) {
      const pixel = queue[head++]!;
      const px = pixel % WIDTH, py = Math.floor(pixel / WIDTH);
      sumX += px; sumY += py;
      left = Math.min(left, px); right = Math.max(right, px);
      top = Math.min(top, py); bottom = Math.max(bottom, py);
      for (let ny = Math.max(bounds.top, py - 1); ny <= Math.min(bounds.bottom, py + 1); ny++) {
        for (let nx = Math.max(bounds.left, px - 1); nx <= Math.min(bounds.right, px + 1); nx++) {
          const next = ny * WIDTH + nx;
          if (mask[next] && !visited[next]) { visited[next] = 1; queue[tail++] = next; }
        }
      }
    }
    const width = right - left + 1, height = bottom - top + 1;
    blobs.push({ x: sumX / tail, y: sumY / tail, left, top, width, height, area: tail,
      confidence: 1 - Math.min(Math.abs(width - 2) * 0.08 + Math.abs(height - 4) * 0.05, 0.3) });
  }
  return blobs;
}

function isBall(blob: Blob): boolean {
  return blob.area >= 4 && blob.area <= 24 && blob.width >= 2 && blob.width <= 5
    && blob.height >= 2 && blob.height <= 7 && blob.area / (blob.width * blob.height) >= 0.45;
}

/** Browser port of breakout_rl/vision_controller.py, using only raw RGB pixels. */
export class PredictiveVisionController {
  private frame = -1;
  private previousFrame: Uint8Array | null = null;
  private ball: Ball | null = null;
  private velocityAnchor: Ball | null = null;
  private previousPaddleX: number | null = null;
  private trackLive = false;
  private previousDirection: ActionMeaning = 'NOOP';
  diagnostics: { ball: { x: number; y: number; vx: number | null; vy: number | null } | null; paddleX: number | null; interceptX: number | null } = {
    ball: null, paddleX: null, interceptX: null,
  };

  reset(): void {
    this.frame = -1;
    this.previousFrame = null;
    this.ball = null;
    this.velocityAnchor = null;
    this.previousPaddleX = null;
    this.trackLive = false;
    this.previousDirection = 'NOOP';
    this.diagnostics = { ball: null, paddleX: null, interceptX: null };
  }

  select(rgb: Uint8Array): PolicyResult {
    if (rgb.length !== WIDTH * HEIGHT * 3) throw new Error('Vision controller requires a 160×210 RGB frame');
    this.frame++;
    const bounds = detectBounds(rgb);
    const mask = new Uint8Array(WIDTH * HEIGHT);
    const movingMask = new Uint8Array(mask.length);
    for (let p = 0; p < mask.length; p++) {
      const i = p * 3, r = rgb[i]!, g = rgb[i + 1]!, b = rgb[i + 2]!;
      mask[p] = (r >= 160 && g <= 130 && b <= 130 && r - g >= 55)
        || (r >= 190 && g >= 190 && b >= 190) ? 1 : 0;
      if (this.previousFrame && (r !== this.previousFrame[i] || g !== this.previousFrame[i + 1] || b !== this.previousFrame[i + 2])) movingMask[p] = mask[p]!;
    }
    const blobs = components(mask, bounds);
    const paddle = blobs.filter(b => b.width >= 4 && b.width <= 28 && b.height >= 2 && b.height <= 8
      && b.area >= 12 && b.top >= bounds.bottom - 18)
      .sort((a, b) => Math.abs(a.top + a.height - 1 - bounds.bottom) - Math.abs(b.top + b.height - 1 - bounds.bottom))[0];
    let paddleX = paddle?.x ?? null;
    if (paddle && paddle.width < 16) {
      if (paddle.left === bounds.left) paddleX = bounds.left + 7.5;
      else if (paddle.left + paddle.width - 1 === bounds.right) paddleX = bounds.right - 7.5;
    }
    const paddleDelta = paddleX !== null && this.previousPaddleX !== null ? paddleX - this.previousPaddleX : 0;
    const paddleVx = Math.abs(paddleDelta) <= 8 ? paddleDelta : 0;
    this.previousPaddleX = paddleX;
    let candidates = blobs.filter(isBall);
    if (this.previousFrame) for (const blob of components(movingMask, bounds).filter(isBall)) {
      if (!candidates.some(b => Math.hypot(b.x - blob.x, b.y - blob.y) < 2)) candidates.push({ ...blob, confidence: blob.confidence * 0.82 });
    }
    if (paddle) candidates = candidates.filter(b => b.y < paddle.top - 2);
    const { max_missing_frames: maxMissing, max_ball_speed_pixels_per_frame: maxSpeed } = config.controller;
    const last = this.ball;
    const gap = last ? this.frame - last.seen : 0;
    let selected: Blob | undefined;
    let allowReacquire = true;
    if (this.trackLive && last && gap <= maxMissing + 1) {
      const projectedX = reflectX(last.x + (last.vx ?? 0) * gap, bounds.left + 2, bounds.right - 2);
      const projectedY = last.y + (last.vy ?? 0) * gap;
      selected = candidates.filter(b => Math.hypot(b.x - last.x, b.y - last.y) <= maxSpeed * gap + 4)
        .sort((a, b) => Math.hypot(a.x - projectedX, a.y - projectedY) - Math.hypot(b.x - projectedX, b.y - projectedY)
          || b.confidence - a.confidence)[0];
      allowReacquire = gap > maxMissing;
    }
    if (!selected && allowReacquire) selected = [...candidates].sort((a, b) => b.confidence - a.confidence)[0];
    let observedBall: Ball | null = null;
    if (selected) {
      let vx: number | null = null, vy: number | null = null;
      if (last && this.trackLive) {
        const dx = (selected.x - last.x) / Math.max(gap, 1), dy = (selected.y - last.y) / Math.max(gap, 1);
        if (Math.abs(dx) <= maxSpeed && Math.abs(dy) <= maxSpeed) {
          const anchor = this.velocityAnchor;
          const turned = dx * (last.vx ?? 0) < 0 || dy * (last.vy ?? 0) < 0;
          // Two native-frame intervals remove alternating pixel rounding; a bounce starts a new segment.
          if (anchor && gap === 1 && last.seen - anchor.seen === 1 && !turned) {
            vx = (selected.x - anchor.x) / 2;
            vy = (selected.y - anchor.y) / 2;
          } else {
            vx = dx;
            vy = dy;
          }
          this.velocityAnchor = last;
        } else {
          this.velocityAnchor = null;
        }
      } else {
        this.velocityAnchor = null;
      }
      this.ball = { x: selected.x, y: selected.y, vx, vy, seen: this.frame, confidence: selected.confidence };
      this.trackLive = true;
      observedBall = this.ball;
    } else if (last && this.trackLive && gap <= maxMissing) {
      observedBall = { ...last, x: reflectX(last.x + (last.vx ?? 0) * gap, bounds.left + 2, bounds.right - 2),
        y: last.y + (last.vy ?? 0) * gap, confidence: last.confidence * 0.65 ** gap };
    } else if (last && gap > maxMissing) {
      this.trackLive = false;
      this.velocityAnchor = null;
      last.vx = null; last.vy = null;
    }
    this.previousFrame = new Uint8Array(rgb);
    let action: ActionMeaning = 'NOOP', interceptX: number | null = null;
    if (paddle && paddleX !== null && observedBall?.vx !== null && observedBall?.vy !== null && observedBall) {
      const frames = (paddle.top - 2 - observedBall.y) / observedBall.vy;
      if (observedBall.vy > 0.15 && frames > 0) {
        interceptX = reflectX(observedBall.x + observedBall.vx * frames, bounds.left + 2, bounds.right - 2);
        const error = interceptX - paddleX;
        action = choosePaddleAction(error, this.previousDirection);
        // Release early while approaching; opposite steering would create another limit cycle.
        const catchMargin = Math.max(0, paddle.width / 2 - 2);
        if (Math.abs(error) <= catchMargin || (error * paddleVx > 0 && Math.abs(error) <= 2 * Math.abs(paddleVx) + config.controller.deadband_pixels)) action = 'NOOP';
      }
    }
    this.previousDirection = action;
    this.diagnostics = { ball: observedBall, paddleX, interceptX };
    const actionIndex = ACTION_MEANINGS.indexOf(action);
    return { action, actionIndex, qValues: [], requestedBackend: 'vision', actualBackend: 'vision',
      difficulty: 'unbeatable', mistakeRate: 0, greedyActionIndex: actionIndex, mistakeInjected: false };
  }
}
