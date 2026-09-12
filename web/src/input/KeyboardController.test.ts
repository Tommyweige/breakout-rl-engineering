import { describe, expect, it } from 'vitest';

import { KeyboardController } from './KeyboardController';

describe('KeyboardController', () => {
  it('starts with no pressed inputs', () => {
    const controller = new KeyboardController();
    expect([...controller.snapshot()]).toEqual([]);
  });

  it('returns defensive snapshots', () => {
    const controller = new KeyboardController();
    const snapshot = controller.snapshot() as Set<string>;
    snapshot.add('LEFT');
    expect([...controller.snapshot()]).toEqual([]);
  });

  it('maps the three supported keyboard controls and ignores unrelated keys', () => {
    const listeners = new Map<string, EventListener>();
    const target = {
      addEventListener: (type: string, listener: EventListener) => listeners.set(type, listener),
      removeEventListener: (type: string) => listeners.delete(type),
    } as unknown as Window;
    const eventFor = (code: string) => ({
      code,
      preventDefault: () => undefined,
    }) as unknown as KeyboardEvent;
    const controller = new KeyboardController();
    controller.attach(target);
    listeners.get('keydown')!(eventFor('ArrowLeft'));
    listeners.get('keydown')!(eventFor('ArrowRight'));
    listeners.get('keydown')!(eventFor('Space'));
    listeners.get('keydown')!(eventFor('KeyA'));
    expect([...controller.snapshot()]).toEqual(['LEFT', 'RIGHT', 'FIRE']);
    listeners.get('keyup')!(eventFor('ArrowRight'));
    expect([...controller.snapshot()]).toEqual(['LEFT', 'FIRE']);
    controller.detach(target);
  });

  it('uses an explicit action policy for held keys and clears on window blur', () => {
    const listeners = new Map<string, EventListener>();
    const target = {
      addEventListener: (type: string, listener: EventListener) => listeners.set(type, listener),
      removeEventListener: (type: string) => listeners.delete(type),
    } as unknown as Window;
    const eventFor = (code: string) => ({ code, preventDefault: () => undefined }) as unknown as KeyboardEvent;
    const controller = new KeyboardController();
    controller.attach(target);
    listeners.get('keydown')!(eventFor('ArrowLeft'));
    expect(controller.currentAction()).toBe('LEFT');
    listeners.get('keydown')!(eventFor('ArrowRight'));
    expect(controller.currentAction()).toBe('NOOP');
    listeners.get('keydown')!(eventFor('Space'));
    expect(controller.currentAction()).toBe('FIRE');
    listeners.get('blur')!(new Event('blur'));
    expect(controller.currentAction()).toBe('NOOP');
    controller.detach(target);
  });
});
