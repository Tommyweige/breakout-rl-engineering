/**
 * Runtime semantics for the player-facing loop.
 *
 * This is intentionally separate from `breakout_contract_v2.json`: the
 * contract describes the reproducible RL task, while this config describes a
 * responsive human input device.
 */
export interface HumanInteractiveRuntimeConfig {
  readonly mode: 'interactive-human';
  readonly rawFrameRepeat: 1;
  readonly stickyActionProbability: 0;
  readonly targetRawFps: 60;
  readonly usesModelPreprocessing: false;
}

export const HUMAN_INTERACTIVE_RUNTIME: HumanInteractiveRuntimeConfig = Object.freeze({
  mode: 'interactive-human',
  rawFrameRepeat: 1,
  stickyActionProbability: 0,
  targetRawFps: 60,
  usesModelPreprocessing: false,
});
