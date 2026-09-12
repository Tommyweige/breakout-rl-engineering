import type { BrowserInfo, InferenceBackend, PolicyResult } from '../inference/types';
import { detectBrowser, currentPageUrl, currentPreviewUrl } from '../inference/browserInfo';
import type { EnvironmentStep } from '../environment/aleEnvironment';
import { BrowserBreakoutEnvironment } from '../environment/aleEnvironment';
import { loadBreakoutContract, type BreakoutContractV2 } from '../environment/breakoutContract';
import { OrtWebPolicy } from '../inference/OrtWebPolicy';
import { validateWebModelManifest, type WebModelManifest } from '../inference/manifest';

export function browserEvaluationSeeds(contract: Pick<BreakoutContractV2, 'concrete_episode_seeds'>): number[] {
  const seeds = [...contract.concrete_episode_seeds];
  const used = new Set(seeds);
  let candidate = (seeds.at(-1) ?? 0) + 1;
  while (seeds.length < 30) {
    if (!used.has(candidate)) {
      seeds.push(candidate);
      used.add(candidate);
    }
    candidate += 1;
  }
  return seeds;
}

export interface EvaluationEnvironment {
  readonly observation: Uint8Array;
  readonly currentSeed: number;
  readonly currentReturn: number;
  readonly currentLives: number;
  readonly currentAgentStep: number;
  readonly isFinished: boolean;
  reset(seed?: number): unknown;
  step(actionIndex: number): EnvironmentStep;
  dispose(): void;
}

export interface EvaluationEpisode {
  seed: number;
  backend: InferenceBackend;
  episodeReturn: number;
  episodeLength: number;
  rawFrameCount: number;
  livesRemaining: number;
  gameOverReason: 'terminated' | 'time_limit' | 'runtime_error' | null;
  runtimeError: string | null;
  autoFireCount: number;
  actionCounts: Record<string, number>;
  actionTrace: Array<{
    agentStep: number;
    requestedAction: string;
    executedAction: string;
    requestedAleAction: number;
    executedAleAction: number;
    autoFire: boolean;
    autoFireReason: string | null;
    reward: number;
    lives: number;
  }>;
  timing: {
    aleStepMs: number[];
    preprocessingMs: number[];
    inferenceMs: number[];
    totalDecisionMs: number[];
  };
}

export interface BrowserPolicyEvaluationArtifact {
  schemaVersion: 1;
  artifactType: 'day29_browser_policy_evaluation';
  timestamp: string;
  pageUrl: string;
  previewUrl: string | null;
  browser: BrowserInfo;
  requestedBackend: InferenceBackend;
  actualBackend: InferenceBackend;
  ortWebVersion: string;
  modelSha256: string;
  environmentContract: {
    contractId: string;
    sourcePath: string;
    sha256: string;
    status: 'partial';
    unsupportedFields: string[];
    note: string;
  };
  seeds: number[];
  maxAgentSteps: number;
  episodes: EvaluationEpisode[];
  aggregate: {
    count: number;
    mean: number;
    median: number;
    std: number;
    p10: number;
    p90: number;
    min: number;
    max: number;
    successCount: number;
    crashCount: number;
    pairedScoreDifference: null;
  };
  timing: {
    sampleCount: number;
    preprocessingMs: number[];
    inferenceMs: number[];
    totalDecisionMs: number[];
    summary: {
      preprocessingP50Ms: number;
      inferenceP50Ms: number;
      totalDecisionP50Ms: number;
      totalDecisionP95Ms: number;
    };
  };
  completed: boolean;
}

export interface PolicyEvaluationOptions {
  backend: InferenceBackend;
  seeds?: readonly number[];
  maxAgentSteps?: number;
  modelSha256?: string;
  policy?: EvaluationPolicy;
  contract?: BreakoutContractV2;
  browser?: BrowserInfo;
  createEnvironment?: (options: { contract: BreakoutContractV2; seed: number }) => Promise<EvaluationEnvironment>;
  infer?: (policy: EvaluationPolicy, observation: Uint8Array) => Promise<PolicyResult>;
  onEpisode?: (episode: EvaluationEpisode, completed: number, total: number) => void;
}

interface EvaluationPolicy {
  readonly requestedBackend: InferenceBackend;
  readonly actualBackend: InferenceBackend | null;
  readonly ortWebVersion: string;
  load(): Promise<void>;
  infer(observation: Uint8Array): Promise<PolicyResult>;
  release(): Promise<void>;
}

export async function runPolicyEvaluation(options: PolicyEvaluationOptions): Promise<BrowserPolicyEvaluationArtifact> {
  const contract = options.contract ?? (await loadBreakoutContract());
  const seeds = [...(options.seeds ?? browserEvaluationSeeds(contract))];
  if (seeds.length < 1) throw new Error('browser evaluation requires at least one seed');
  const maxAgentSteps = options.maxAgentSteps ?? contract.time_limit_semantics.agent_step_limit;
  if (!Number.isInteger(maxAgentSteps) || maxAgentSteps < 1) throw new Error('maxAgentSteps must be a positive integer');
  const policy = options.policy ?? new OrtWebPolicy({ backend: options.backend });
  const ownsPolicy = !options.policy;
  const createEnvironment = options.createEnvironment ?? ((environmentOptions) => BrowserBreakoutEnvironment.create(environmentOptions));
  const infer = options.infer ?? ((activePolicy, observation) => activePolicy.infer(observation));
  const modelSha256 = options.modelSha256 ?? (await loadModelManifest()).model.sha256;
  await policy.load();
  if (policy.actualBackend !== options.backend) {
    if (ownsPolicy) await policy.release();
    throw new Error(`evaluation requested ${options.backend.toUpperCase()} but actual backend is ${policy.actualBackend ?? 'unavailable'}`);
  }

  const episodes: EvaluationEpisode[] = [];
  const timing = { preprocessingMs: [] as number[], inferenceMs: [] as number[], totalDecisionMs: [] as number[] };
  let environment: EvaluationEnvironment | null = null;
  try {
    for (let index = 0; index < seeds.length; index += 1) {
      const seed = seeds[index]!;
      if (!environment) environment = await createEnvironment({ contract, seed });
      else environment.reset(seed);
      const episode = await runEpisode(environment, policy, infer, options.backend, seed, maxAgentSteps, timing);
      episodes.push(episode);
      options.onEpisode?.(episode, index + 1, seeds.length);
    }
  } finally {
    environment?.dispose();
    if (ownsPolicy) await policy.release();
  }

  const aggregate = aggregateEpisodeScores(episodes.map((episode) => episode.episodeReturn));
  const completed = episodes.length === seeds.length && episodes.every((episode) => episode.runtimeError === null);
  const browser = options.browser ?? (await detectBrowser());
  return {
    schemaVersion: 1,
    artifactType: 'day29_browser_policy_evaluation',
    timestamp: new Date().toISOString(),
    pageUrl: currentPageUrl(),
    previewUrl: currentPreviewUrl(),
    browser,
    requestedBackend: options.backend,
    actualBackend: policy.actualBackend!,
    ortWebVersion: policy.ortWebVersion,
    modelSha256,
    environmentContract: {
      contractId: contract.contract_id,
      sourcePath: contract.source_path,
      sha256: contract.sha256,
      status: contract.parity.status,
      unsupportedFields: [...contract.parity.unsupported_fields],
      note: contract.parity.note,
    },
    seeds,
    maxAgentSteps,
    episodes,
    aggregate: {
      ...aggregate,
      successCount: episodes.filter((episode) => episode.runtimeError === null).length,
      crashCount: episodes.filter((episode) => episode.runtimeError !== null).length,
      pairedScoreDifference: null,
    },
    timing: {
      sampleCount: timing.totalDecisionMs.length,
      ...timing,
      summary: {
        preprocessingP50Ms: percentile(timing.preprocessingMs, 0.5),
        inferenceP50Ms: percentile(timing.inferenceMs, 0.5),
        totalDecisionP50Ms: percentile(timing.totalDecisionMs, 0.5),
        totalDecisionP95Ms: percentile(timing.totalDecisionMs, 0.95),
      },
    },
    completed,
  };
}

export function aggregateEpisodeScores(scores: readonly number[]) {
  if (scores.length === 0) {
    return { count: 0, mean: 0, median: 0, std: 0, p10: 0, p90: 0, min: 0, max: 0 };
  }
  const sorted = [...scores].sort((left, right) => left - right);
  const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  const variance = scores.reduce((sum, score) => sum + (score - mean) ** 2, 0) / scores.length;
  return {
    count: scores.length,
    mean,
    median: percentile(sorted, 0.5),
    std: Math.sqrt(variance),
    p10: percentile(sorted, 0.1),
    p90: percentile(sorted, 0.9),
    min: sorted[0]!,
    max: sorted[sorted.length - 1]!,
  };
}

async function runEpisode(
  environment: EvaluationEnvironment,
  policy: EvaluationPolicy,
  infer: (policy: EvaluationPolicy, observation: Uint8Array) => Promise<PolicyResult>,
  backend: InferenceBackend,
  seed: number,
  maxAgentSteps: number,
  globalTiming: { preprocessingMs: number[]; inferenceMs: number[]; totalDecisionMs: number[] },
): Promise<EvaluationEpisode> {
  const timing = { aleStepMs: [] as number[], preprocessingMs: [] as number[], inferenceMs: [] as number[], totalDecisionMs: [] as number[] };
  const actionCounts: Record<string, number> = { NOOP: 0, FIRE: 0, RIGHT: 0, LEFT: 0 };
  const actionTrace: EvaluationEpisode['actionTrace'] = [];
  let autoFireCount = 0;
  let runtimeError: string | null = null;
  let gameOverReason: EvaluationEpisode['gameOverReason'] = null;
  while (!environment.isFinished && environment.currentAgentStep < maxAgentSteps) {
    try {
      const inferenceStartedAt = now();
      const result = await infer(policy, environment.observation);
      const inferenceMs = now() - inferenceStartedAt;
      const step = environment.step(result.actionIndex);
      timing.aleStepMs.push(step.timing.aleStepMs);
      timing.preprocessingMs.push(step.timing.preprocessingMs);
      timing.inferenceMs.push(inferenceMs);
      timing.totalDecisionMs.push(step.timing.totalMs + inferenceMs);
      globalTiming.preprocessingMs.push(step.timing.preprocessingMs);
      globalTiming.inferenceMs.push(inferenceMs);
      globalTiming.totalDecisionMs.push(step.timing.totalMs + inferenceMs);
      actionCounts[step.executedAction] = (actionCounts[step.executedAction] ?? 0) + 1;
      actionTrace.push({
        agentStep: step.agentStep,
        requestedAction: step.requestedAction,
        executedAction: step.executedAction,
        requestedAleAction: step.requestedAleAction,
        executedAleAction: step.executedAleAction,
        autoFire: step.autoFire,
        autoFireReason: step.autoFireReason,
        reward: step.reward,
        lives: step.lives,
      });
      if (step.autoFire) autoFireCount += 1;
      if (step.truncated) gameOverReason = 'time_limit';
      else if (step.terminated) gameOverReason = 'terminated';
    } catch (error) {
      runtimeError = error instanceof Error ? error.message : String(error);
      gameOverReason = 'runtime_error';
      break;
    }
  }
  if (!runtimeError && !gameOverReason && environment.currentAgentStep >= maxAgentSteps) gameOverReason = 'time_limit';
  return {
    seed,
    backend,
    episodeReturn: environment.currentReturn,
    episodeLength: environment.currentAgentStep,
    rawFrameCount: environment.currentAgentStep * 4,
    livesRemaining: environment.currentLives,
    gameOverReason,
    runtimeError,
    autoFireCount,
    actionCounts,
    actionTrace,
    timing,
  };
}

async function loadModelManifest(): Promise<WebModelManifest> {
  const response = await fetch('/web-model-manifest.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`failed to load web model manifest: ${response.status}`);
  return validateWebModelManifest(await response.json());
}

function percentile(values: readonly number[], fraction: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower]!;
  const weight = position - lower;
  return sorted[lower]! + (sorted[upper]! - sorted[lower]!) * weight;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
