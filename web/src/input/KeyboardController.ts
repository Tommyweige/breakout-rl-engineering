export type HumanInput = 'LEFT' | 'RIGHT' | 'FIRE';

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

  attach(target: Window = window): void {
    if (this.attached) return;
    target.addEventListener('keydown', this.onKeyDown);
    target.addEventListener('keyup', this.onKeyUp);
    this.attached = true;
  }

  detach(target: Window = window): void {
    if (!this.attached) return;
    target.removeEventListener('keydown', this.onKeyDown);
    target.removeEventListener('keyup', this.onKeyUp);
    this.pressed.clear();
    this.attached = false;
  }

  snapshot(): ReadonlySet<HumanInput> {
    return new Set(this.pressed);
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
