export interface BreakoutContractV2 {
  schema_version: 2;
  contract_id: 'day15-breakout-evaluation-v2-fire-reset';
  environment_id: 'ALE/Breakout-v5';
  frame_skip: 4;
  frame_stack: 4;
  sticky_action_probability: 0.25;
  fire_reset: true;
  fire_reset_confirmation: {
    max_fire_attempts: 8;
    confirmation_steps: 2;
    min_observation_change_fraction: 0.0001;
    confirmation_operator: 'any';
    confirmation_signals: ['raw_reward', 'observation_activity_streak'];
  };
  terminal_on_life_loss: false;
  time_limit_semantics: {
    source: 'ale.game_truncated';
    max_num_frames_per_episode: 108000;
    agent_step_limit: 27000;
    truncated_is_finished: true;
    external_time_limit_wrapper: false;
  };
  concrete_episode_seeds: number[];
  evaluation_epsilon: 0;
  raw_reward_rule: 'sum environment rewards without clipping';
  sha256: string;
  source_path: 'configs/eval/breakout_contract_v2.json';
  parity: {
    status: 'partial';
    unsupported_fields: string[];
    note: string;
  };
}

export async function loadBreakoutContract(fetcher: typeof fetch = fetch): Promise<BreakoutContractV2> {
  const [contractResponse, inferenceResponse] = await Promise.all([
    fetcher('/configs/eval/breakout_contract_v2.json', { cache: 'no-store' }),
    fetcher('/inference_spec.json', { cache: 'no-store' }),
  ]);
  if (!contractResponse.ok) throw new Error(`failed to load Contract v2: ${contractResponse.status}`);
  if (!inferenceResponse.ok) throw new Error(`failed to load inference spec: ${inferenceResponse.status}`);
  const contractBytes = new Uint8Array(await contractResponse.arrayBuffer());
  const rawContract: unknown = JSON.parse(new TextDecoder().decode(contractBytes));
  const inferenceSpec: unknown = await inferenceResponse.json();
  const expectedSha256 = readExpectedContractSha256(inferenceSpec);
  const sha256 = await sha256Hex(contractBytes);
  if (sha256 !== expectedSha256) {
    throw new Error(`Contract v2 SHA256 mismatch: expected ${expectedSha256}, got ${sha256}`);
  }
  return validateBreakoutContract(rawContract, sha256);
}

export function validateBreakoutContract(value: unknown, sha256: string): BreakoutContractV2 {
  if (!isRecord(value)) throw new Error('Contract v2 must be a JSON object');
  if (value.schema_version !== 2 || value.contract_id !== 'day15-breakout-evaluation-v2-fire-reset') {
    throw new Error('unexpected Breakout Contract v2 identity');
  }
  if (value.environment_id !== 'ALE/Breakout-v5') throw new Error('Contract v2 environment must be ALE/Breakout-v5');
  if (value.frame_skip !== 4 || value.frame_stack !== 4) throw new Error('Contract v2 requires frame_skip=4 and frame_stack=4');
  if (value.sticky_action_probability !== 0.25 || value.fire_reset !== true) {
    throw new Error('Contract v2 requires sticky_action_probability=0.25 and fire_reset=true');
  }
  if (value.terminal_on_life_loss !== false) throw new Error('Contract v2 must not terminate on life loss');
  const confirmation = record(value.fire_reset_confirmation, 'fire_reset_confirmation');
  const timeLimit = record(value.time_limit_semantics, 'time_limit_semantics');
  if (
    confirmation.max_fire_attempts !== 8 ||
    confirmation.confirmation_steps !== 2 ||
    confirmation.min_observation_change_fraction !== 0.0001 ||
    confirmation.confirmation_operator !== 'any' ||
    !sameStrings(confirmation.confirmation_signals, ['raw_reward', 'observation_activity_streak'])
  ) {
    throw new Error('Contract v2 FIRE confirmation settings do not match the canonical contract');
  }
  if (
    timeLimit.source !== 'ale.game_truncated' ||
    timeLimit.max_num_frames_per_episode !== 108000 ||
    timeLimit.agent_step_limit !== 27000 ||
    timeLimit.truncated_is_finished !== true ||
    timeLimit.external_time_limit_wrapper !== false
  ) {
    throw new Error('Contract v2 time-limit settings do not match the canonical contract');
  }
  if (!Array.isArray(value.concrete_episode_seeds) || value.concrete_episode_seeds.some((seed) => !Number.isInteger(seed))) {
    throw new Error('Contract v2 concrete_episode_seeds must be integer seeds');
  }
  if (value.evaluation_epsilon !== 0 || value.raw_reward_rule !== 'sum environment rewards without clipping') {
    throw new Error('Contract v2 evaluation epsilon/reward rule does not match the canonical contract');
  }
  const concreteSeeds = value.concrete_episode_seeds as number[];
  return {
    schema_version: value.schema_version as 2,
    contract_id: value.contract_id as 'day15-breakout-evaluation-v2-fire-reset',
    environment_id: value.environment_id as 'ALE/Breakout-v5',
    frame_skip: value.frame_skip as 4,
    frame_stack: value.frame_stack as 4,
    sticky_action_probability: value.sticky_action_probability as 0.25,
    fire_reset: value.fire_reset as true,
    fire_reset_confirmation: {
      max_fire_attempts: confirmation.max_fire_attempts as 8,
      confirmation_steps: confirmation.confirmation_steps as 2,
      min_observation_change_fraction: confirmation.min_observation_change_fraction as 0.0001,
      confirmation_operator: confirmation.confirmation_operator as 'any',
      confirmation_signals: [...(confirmation.confirmation_signals as string[])] as ['raw_reward', 'observation_activity_streak'],
    },
    terminal_on_life_loss: value.terminal_on_life_loss as false,
    time_limit_semantics: {
      source: timeLimit.source as 'ale.game_truncated',
      max_num_frames_per_episode: timeLimit.max_num_frames_per_episode as 108000,
      agent_step_limit: timeLimit.agent_step_limit as 27000,
      truncated_is_finished: timeLimit.truncated_is_finished as true,
      external_time_limit_wrapper: timeLimit.external_time_limit_wrapper as false,
    },
    concrete_episode_seeds: [...concreteSeeds],
    evaluation_epsilon: value.evaluation_epsilon as 0,
    raw_reward_rule: value.raw_reward_rule as 'sum environment rewards without clipping',
    sha256,
    source_path: 'configs/eval/breakout_contract_v2.json',
    parity: {
      status: 'partial',
      unsupported_fields: [
        'Python cv2 INTER_AREA byte-for-byte parity is not independently established in the browser',
        'ALE/browser seed stream equivalence is not independently established against native Gymnasium',
      ],
      note: 'Browser ALE and Contract v2 settings are active; raw-score comparison to native Python remains partial until cross-runtime seed and resize parity are audited.',
    },
  };
}

function readExpectedContractSha256(value: unknown): string {
  if (!isRecord(value) || !isRecord(value.environment_contract) || typeof value.environment_contract.sha256 !== 'string') {
    throw new Error('inference spec does not expose the expected Contract v2 SHA256');
  }
  return value.environment_contract.sha256;
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('Web Crypto SHA256 is unavailable');
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes as BufferSource);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`Contract v2 field ${name} must be an object`);
  return value;
}

function sameStrings(value: unknown, expected: readonly string[]): boolean {
  return Array.isArray(value) && value.length === expected.length && value.every((item, index) => item === expected[index]);
}
