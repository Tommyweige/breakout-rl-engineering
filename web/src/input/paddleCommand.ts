export type PaddleDirection = 'LEFT' | 'RIGHT' | 'NOOP';

export interface HumanPaddleCommand {
  direction: PaddleDirection;
  targetX: number | null;
  paddleCenterX: number | null;
  positionError: number | null;
}
