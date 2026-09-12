export type HumanInput = 'LEFT' | 'RIGHT' | 'FIRE';
import type { ActionMeaning } from '../inference/types';

export class KeyboardController {
  private readonly pressed = new Set<HumanInput>();
  private attached = false;

  private readonly onKeyDown = (event: KeyboardEvent): void => {
    const input = this.toInput(event.code);
    if (!input) return;
    event.preventDefault();
    this.pressed.add(input);
  };

  private readonly onKeyUp = (event: KeyboardEvent): void => {
    const input = this.toInput(event.code);
    if (!input) return;
    event.preventDefault();
    this.pressed.delete(input);
  };

  private readonly onBlur = (): void => {
    this.pressed.clear();
  };

  attach(target: Window = window): void {
    if (this.attached) return;
    target.addEventListener('keydown', this.onKeyDown);
    target.addEventListener('keyup', this.onKeyUp);
    target.addEventListener('blur', this.onBlur);
    this.attached = true;
  }

  detach(target: Window = window): void {
    if (!this.attached) return;
    target.removeEventListener('keydown', this.onKeyDown);
    target.removeEventListener('keyup', this.onKeyUp);
    target.removeEventListener('blur', this.onBlur);
    this.pressed.clear();
    this.attached = false;
  }

  snapshot(): ReadonlySet<HumanInput> {
    return new Set(this.pressed);
  }

  /** Resolve held keys once per environment decision; both arrows deliberately cancel to NOOP. */
  currentAction(): ActionMeaning {
    if (this.pressed.has('FIRE')) return 'FIRE';
    const left = this.pressed.has('LEFT');
    const right = this.pressed.has('RIGHT');
    if (left && right) return 'NOOP';
    if (left) return 'LEFT';
    if (right) return 'RIGHT';
    return 'NOOP';
  }

  private toInput(code: string): HumanInput | null {
    switch (code) {
      case 'ArrowLeft':
        return 'LEFT';
      case 'ArrowRight':
        return 'RIGHT';
      case 'Space':
        return 'FIRE';
      default:
        return null;
    }
  }
}
