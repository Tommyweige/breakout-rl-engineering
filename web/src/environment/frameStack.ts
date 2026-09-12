export class FrameStack {
  private frames: Uint8Array[];

  constructor(private readonly stackSize = 4, private readonly frameSize = 84 * 84) {
    if (!Number.isInteger(stackSize) || stackSize < 1) throw new Error('stackSize must be a positive integer');
    if (!Number.isInteger(frameSize) || frameSize < 1) throw new Error('frameSize must be a positive integer');
    this.frames = [];
  }

  reset(frame: ArrayLike<number>): Uint8Array {
    this.validateFrame(frame);
    const first = Uint8Array.from(frame);
    this.frames = Array.from({ length: this.stackSize }, () => new Uint8Array(first));
    return this.current();
  }

  push(frame: ArrayLike<number>): Uint8Array {
    this.validateFrame(frame);
    if (this.frames.length === 0) this.reset(frame);
    else {
      this.frames.shift();
      this.frames.push(Uint8Array.from(frame));
    }
    return this.current();
  }

  current(): Uint8Array {
    if (this.frames.length !== this.stackSize) throw new Error('frame stack has not been reset');
    const observation = new Uint8Array(this.stackSize * this.frameSize);
    this.frames.forEach((frame, index) => observation.set(frame, index * this.frameSize));
    return observation;
  }

  get size(): number {
    return this.stackSize;
  }

  private validateFrame(frame: ArrayLike<number>): void {
    if (frame.length !== this.frameSize) {
      throw new Error(`frame must contain ${this.frameSize} values, got ${frame.length}`);
    }
  }
}
