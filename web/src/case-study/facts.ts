/**
 * Story numbers that are not in the final-results summary, shared by both
 * locales so the two pages cannot drift apart. Each entry names its source.
 * Values are written exactly as the page shows them.
 */
export interface Fact {
  value: string;
  source: string;
}

const DAY15 = 'docs-ironman-series: articles/day15-dqn-milestone-and-evaluation.md';
const DAY16 = 'docs-ironman-series: articles/day16-vectorized-dqn-training.md';
const DAY20 = 'docs-ironman-series: articles/day20-dqn-family-comparison.md';
const DAY21 = 'docs-ironman-series: articles/day21-final-long-training.md';
const DAY27 = 'docs-ironman-series: articles/day27-onnx-runtime-web.md';
const DAY28 = 'docs-ironman-series: articles/day28-webgpu-inference.md';
const SUMMARY_NOTES = 'docs/final-results/summary.json: dqn_v3_2m_mean_score notes';

export const FACTS: Record<string, Fact> = {
  day15_random_mean: { value: '1.33', source: DAY15 },
  day15_dqn_mean: { value: '4.53', source: DAY15 },
  day15_random_steps: { value: '186', source: DAY15 },
  day15_dqn_steps: { value: '18,131', source: DAY15 },
  day15_timelimit_games: { value: '10', source: DAY15 },
  day15_step_limit: { value: '27,000', source: DAY15 },
  day15_life_lost_step: { value: '54', source: DAY15 },
  day15_first_fire_step: { value: '16,449', source: DAY15 },
  day15_gap_steps: { value: '16,395', source: DAY15 },
  day15_assist_mean: { value: '8.80', source: DAY15 },
  day15_assist_steps: { value: '430', source: DAY15 },
  day15_epsilon_mean: { value: '9.20', source: DAY15 },
  day15_epsilon_steps: { value: '509', source: DAY15 },
  vec_n1_throughput: { value: '238.67', source: DAY16 },
  vec_n2_throughput: { value: '380.74', source: DAY16 },
  family_dqn_500k: { value: '15.844', source: DAY20 },
  family_double_500k: { value: '18.156', source: DAY20 },
  family_dueling_500k: { value: '20.178', source: DAY20 },
  dueling_1m: { value: '36.111', source: DAY20 },
  dueling_5m: { value: '49.933', source: DAY21 },
  dqn_v3_400k: { value: '2.13', source: SUMMARY_NOTES },
  dqn_v3_1m: { value: '4.60', source: SUMMARY_NOTES },
  browser_max_q_diff: { value: '0.00016', source: DAY27 },
  web_wasm_p50_ms: { value: '0.995', source: DAY28 },
  web_webgpu_p50_ms: { value: '3.985', source: DAY28 },
  first_wall_score: { value: '432', source: 'docs/clear-semantics.md' },
  best_unclear_score: { value: '859', source: 'audits/browser-clear/native-15/episodes.csv' },
  detector_mutants: { value: '6', source: 'docs/clear-semantics.md (mutation table)' },
};

/** The tracker's 15 official games, in order (audits/browser-clear/native-15/episodes.csv). */
export const TRACKER_EPISODES: ReadonlyArray<{ score: number; cleared: boolean }> = [
  { score: 859, cleared: false },
  { score: 537, cleared: false },
  { score: 828, cleared: false },
  { score: 565, cleared: false },
  { score: 810, cleared: false },
  { score: 864, cleared: true },
  { score: 425, cleared: false },
  { score: 849, cleared: false },
  { score: 750, cleared: false },
  { score: 441, cleared: false },
  { score: 519, cleared: false },
  { score: 864, cleared: true },
  { score: 788, cleared: false },
  { score: 864, cleared: true },
  { score: 864, cleared: true },
];

export const TRACKER_EPISODE_MAX = 864;
export const TRACKER_EPISODE_FIRST_WALL = 432;
