#!/usr/bin/env node
// Export the case-study numbers for the tommypan.com project profile.
//
// The Hugo page reads only the JSON this script writes, so every number it shows
// traces back to a file in this repository (or the archived article branch).
// Evidence links point at `main` rather than a pinned commit, so they keep
// resolving as the branch moves. Rerun after any of the inputs change:
//
//   node scripts/export-case-study-data.mjs --out ../HugoBlog/myblog/data/case_study_breakout.json

import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REPOSITORY = 'https://github.com/Tommyweige/breakout-rl-engineering';
const INPUTS = {
  summary: 'docs/final-results/summary.json',
  episodes: 'audits/browser-clear/native-15/episodes.csv',
  episodeSummary: 'audits/browser-clear/native-15/summary.json',
  facts: 'web/src/case-study/facts.ts',
};
const ARTICLE_REFS = ['docs-ironman-series', 'origin/docs-ironman-series'];
const PARITY_ARTICLE = 'articles/day29-interactive-browser-demo.md';
const ALE_FRAMES_PER_SECOND = 60;

function fail(message) {
  console.error(`export-case-study-data: ${message}`);
  process.exit(1);
}

function git(...args) {
  return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
}

function tryGit(...args) {
  try {
    return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return null;
  }
}

function read(path) {
  return readFileSync(resolve(ROOT, path), 'utf8');
}

// Same rules as formatMetricValue() on the Vite case study (zh-TW copy).
function display(value) {
  if (value === null) return '未執行';
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString('en-US') : String(value);
  return String(value);
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ',') {
      row.push(field);
      field = '';
    } else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field);
      if (row.some((value) => value !== '')) rows.push(row);
      row = [];
      field = '';
    } else {
      field += char;
    }
  }
  row.push(field);
  if (row.some((value) => value !== '')) rows.push(row);
  const [header, ...body] = rows;
  return body.map((values) => Object.fromEntries(header.map((name, index) => [name, values[index] ?? ''])));
}

// facts.ts holds story numbers that are not in summary.json, each with a source.
function parseFacts(source) {
  const constants = Object.fromEntries(
    [...source.matchAll(/^const (\w+) = '([^']*)';$/gm)].map(([, name, value]) => [name, value]),
  );
  const facts = {};
  for (const [, id, value, sourceRef] of source.matchAll(
    /^\s*(\w+): \{ value: '([^']*)', source: ('[^']*'|\w+) \},$/gm,
  )) {
    const resolved = sourceRef.startsWith("'") ? sourceRef.slice(1, -1) : constants[sourceRef];
    if (resolved === undefined) fail(`facts.ts: unknown source constant ${sourceRef} for ${id}`);
    facts[id] = { value, number: Number(value.replaceAll(',', '')), source: resolved };
  }
  if (Object.keys(facts).length === 0) fail('facts.ts: no facts found; did the file format change?');
  return facts;
}

function parseParityTable(markdown) {
  return [...markdown.matchAll(/^\| (\d{3}) \| (\d+) \| (\d+) \| (\d+) \|$/gm)].map(([, seed, native, wasm, webgpu]) => ({
    seed: Number(seed),
    native: Number(native),
    wasm: Number(wasm),
    webgpu: Number(webgpu),
  }));
}

function check(condition, message) {
  if (!condition) fail(`consistency check failed: ${message}`);
}

const outIndex = process.argv.indexOf('--out');
const outPath = outIndex > 0 ? process.argv[outIndex + 1] : undefined;
if (!outPath) fail('usage: node scripts/export-case-study-data.mjs --out <path/to/case_study_breakout.json>');

const summary = JSON.parse(read(INPUTS.summary));
const episodeSummary = JSON.parse(read(INPUTS.episodeSummary));
const facts = parseFacts(read(INPUTS.facts));

const metrics = {};
for (const metric of summary.metrics) {
  metrics[metric.id] = {
    value: metric.value,
    display: display(metric.value),
    seed_count: metric.seed_count,
    label: metric.label,
    unit: metric.unit,
    source: metric.source,
    verification_status: metric.verification_status,
  };
}

const episodes = parseCsv(read(INPUTS.episodes)).map((row, index) => ({
  game: index + 1,
  evaluation_seed: Number(row.evaluation_seed),
  score: Number(row.raw_score),
  cleared: row.cleared === 'True',
  clear_emulator_frame: row.clear_emulator_frame ? Number(row.clear_emulator_frame) : null,
  lives_remaining_at_clear: row.lives_remaining_at_clear ? Number(row.lives_remaining_at_clear) : null,
  failure_reason: row.failure_reason,
}));
const cleared = episodes.filter((episode) => episode.cleared);
const unclearedScores = episodes.filter((episode) => !episode.cleared).map((episode) => episode.score);
const bestClearFrame = Math.min(...cleared.map((episode) => episode.clear_emulator_frame));
const livesAtClear = cleared.map((episode) => episode.lives_remaining_at_clear);
const meanScore = episodes.reduce((total, episode) => total + episode.score, 0) / episodes.length;

// The episode log, its own summary, the frozen results summary and facts.ts must agree.
check(episodes.length === episodeSummary.evaluation_protocol.episode_count, 'episode count vs native-15 summary');
check(cleared.length === episodeSummary.summary.clear_count, 'clear count vs native-15 summary');
check(Math.abs(meanScore - episodeSummary.summary.mean_score) < 1e-9, 'mean score vs native-15 summary');
check(cleared.length === metrics.rgb_controller_a8445df_canonical_clear_detections?.value, 'clear count vs summary.json');
check(episodes.length === metrics.rgb_controller_a8445df_canonical_clear_detections?.seed_count, 'episode count vs summary.json');
check(Math.abs(meanScore - metrics.rgb_controller_a8445df_mean_score?.value) < 0.05, 'mean score vs summary.json');
check(bestClearFrame === metrics.rgb_controller_a8445df_best_clear_ale_frame?.value, 'best clear frame vs summary.json');
check(Math.max(...unclearedScores) === facts.best_unclear_score?.number, 'best uncleared score vs facts.ts');
check(metrics.clear_score_threshold?.value === Math.max(...episodes.map((episode) => episode.score)), 'clear threshold vs max score');

const bestClearSeconds = Math.round(bestClearFrame / ALE_FRAMES_PER_SECOND);
const derived = {
  vec_speedup: {
    value: (facts.vec_n2_throughput.number / facts.vec_n1_throughput.number).toFixed(1),
    source: 'facts.ts: vec_n2_throughput / vec_n1_throughput',
  },
  cleared_games: {
    value: cleared.map((episode) => episode.game).join('、'),
    source: `${INPUTS.episodes}: games with cleared=True`,
  },
  uncleared_count: {
    value: String(episodes.length - cleared.length),
    source: `${INPUTS.episodes}: games with cleared=False`,
  },
  lives_at_clear_min: { value: String(Math.min(...livesAtClear)), source: `${INPUTS.episodes}: lives_remaining_at_clear` },
  lives_at_clear_max: { value: String(Math.max(...livesAtClear)), source: `${INPUTS.episodes}: lives_remaining_at_clear` },
  best_clear_minutes: {
    value: String(Math.floor(bestClearSeconds / 60)),
    source: `best clear emulator frame / ${ALE_FRAMES_PER_SECOND} ALE frames per second`,
  },
  best_clear_seconds: {
    value: String(bestClearSeconds % 60),
    source: `best clear emulator frame / ${ALE_FRAMES_PER_SECOND} ALE frames per second`,
  },
};
for (const [id, fact] of Object.entries(derived)) {
  if (facts[id]) fail(`derived fact ${id} collides with facts.ts`);
  facts[id] = { ...fact, number: Number(fact.value), derived: true };
}

let parity = null;
for (const ref of ARTICLE_REFS) {
  const article = tryGit('show', `${ref}:${PARITY_ARTICLE}`);
  if (article === null) continue;
  const rows = parseParityTable(article);
  if (rows.length === 0) fail(`${ref}:${PARITY_ARTICLE}: parity table not found`);
  parity = { source: `${ref}:${PARITY_ARTICLE}`, source_commit: git('rev-parse', ref), rows };
  break;
}
if (!parity) fail(`could not read ${PARITY_ARTICLE} from ${ARTICLE_REFS.join(' or ')}; fetch the article branch first`);

// Evidence links point at `main`, so they resolve without pinning a commit. evidence_commit still
// records the commit the numbers were exported from.
const sourceCommit = git('rev-parse', 'HEAD');
const inputPaths = Object.values(INPUTS);
const dirty = git('status', '--porcelain', '--', ...inputPaths, 'scripts/export-case-study-data.mjs') !== '';

const output = {
  schema_version: 1,
  generated_at: new Date().toISOString(),
  generator: 'scripts/export-case-study-data.mjs',
  source: {
    repository: REPOSITORY,
    commit: sourceCommit,
    branch: tryGit('rev-parse', '--abbrev-ref', 'HEAD'),
    working_tree_dirty: dirty,
    evidence_commit: sourceCommit,
    evidence_base_url: `${REPOSITORY}/blob/main/`,
    inputs: inputPaths,
  },
  frozen_at: summary.frozen_at,
  links: summary.links,
  metrics,
  facts,
  episodes: {
    source: INPUTS.episodes,
    contract: episodeSummary.environment.base_environment_contract_id,
    controller_source_commit: episodeSummary.source_provenance.source_commit,
    first_wall_score: facts.first_wall_score.number,
    both_walls_score: metrics.clear_score_threshold.value,
    clear_count: cleared.length,
    count: episodes.length,
    items: episodes,
  },
  parity,
};

const target = resolve(process.cwd(), outPath);
mkdirSync(dirname(target), { recursive: true });
writeFileSync(target, `${JSON.stringify(output, null, 2)}\n`);
console.log(
  `wrote ${target}: ${Object.keys(metrics).length} metrics, ${Object.keys(facts).length} facts, ` +
    `${episodes.length} episodes, ${parity.rows.length} parity rows (commit ${sourceCommit.slice(0, 7)}${dirty ? ', dirty' : ''})`,
);
