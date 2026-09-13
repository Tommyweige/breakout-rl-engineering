import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import process from 'node:process';

import { chromium } from 'playwright-core';

const args = new Map();
for (let index = 2; index < process.argv.length; index += 1) {
  const value = process.argv[index];
  if (value?.startsWith('--')) args.set(value.slice(2), process.argv[index + 1] ?? true);
}

const productUrl = String(args.get('url') ?? 'http://127.0.0.1:5173/');
const debugUrl = withDebug(productUrl);
const outputDirectory = resolve(String(args.get('output-dir') ?? '../assets/day30'));
const concurrency = Number(args.get('concurrency') ?? 4);
const defaultSeeds = Array.from({ length: 20 }, (_, index) => 323 + index).join(',');
const seedText = String(args.get('seeds') ?? defaultSeeds);
const seeds = seedText.split(',').map((value) => Number(value.trim())).filter((value) => Number.isInteger(value));
if (seeds.length < 20) throw new Error('--seeds must contain at least 20 integer seeds');
if (new Set(seeds.slice(0, 20)).size !== 20) throw new Error('the Day 30 supplement must contain 20 unique seeds');
if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 8) throw new Error('--concurrency must be between 1 and 8');

function withDebug(value) {
  const parsed = new URL(value);
  parsed.searchParams.set('debug', '1');
  return parsed.toString();
}

const base = JSON.parse(await readFile(resolve(outputDirectory, '../day29/browser-policy-score-comparison.json'), 'utf8'));
const browser = await chromium.launch({ channel: 'chrome', headless: args.get('headed') !== true });
const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });

function percentile(values, fraction) {
  if (!values.length) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  const position = (ordered.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.min(ordered.length - 1, lower + 1);
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower);
}

function aggregate(episodes) {
  const values = episodes.map((episode) => episode.episodeReturn);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return { count: values.length, uniqueSeedCount: new Set(episodes.map((episode) => episode.seed)).size, mean, median: percentile(values, 0.5), std: Math.sqrt(variance), p10: percentile(values, 0.1), p90: percentile(values, 0.9), min: Math.min(...values), max: Math.max(...values), successCount: episodes.filter((episode) => episode.runtimeError === null).length, crashCount: episodes.filter((episode) => episode.runtimeError !== null).length, pairedScoreDifference: null };
}

function mergeEvaluation(baseEvaluation, supplementEpisodes, browserInfo, environmentContract) {
  const episodes = [...baseEvaluation.episodes, ...supplementEpisodes];
  const timing = {
    sampleCount: baseEvaluation.timing.sampleCount + supplementEpisodes.reduce((sum, episode) => sum + episode.timing.totalDecisionMs.length, 0),
    preprocessingMs: [...baseEvaluation.timing.preprocessingMs, ...supplementEpisodes.flatMap((episode) => episode.timing.preprocessingMs)],
    inferenceMs: [...baseEvaluation.timing.inferenceMs, ...supplementEpisodes.flatMap((episode) => episode.timing.inferenceMs)],
    totalDecisionMs: [...baseEvaluation.timing.totalDecisionMs, ...supplementEpisodes.flatMap((episode) => episode.timing.totalDecisionMs)],
  };
  timing.summary = { preprocessingP50Ms: percentile(timing.preprocessingMs, 0.5), inferenceP50Ms: percentile(timing.inferenceMs, 0.5), totalDecisionP50Ms: percentile(timing.totalDecisionMs, 0.5), totalDecisionP95Ms: percentile(timing.totalDecisionMs, 0.95) };
  return { ...baseEvaluation, artifactType: 'day30_browser_policy_evaluation', browser: browserInfo, environmentContract, seeds: episodes.map((episode) => episode.seed), episodes, aggregate: aggregate(episodes), timing, completed: episodes.every((episode) => episode.runtimeError === null) };
}

async function evaluateSeed(seed, index) {
  const page = await context.newPage();
  const errors = [];
  page.on('console', (message) => { if (message.type() === 'error') errors.push(`console: ${message.text()}`); });
  page.on('pageerror', (error) => errors.push(`page: ${String(error)}`));
  try {
    await page.goto(debugUrl, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await page.locator('details[data-role="technical-details"]').evaluate((details) => { details.open = true; });
    await page.locator('select[data-action="backend"]').selectOption('webgpu');
    await page.locator('button[data-action="validate"]').click();
    await page.waitForFunction(() => Boolean(window.__day28Validation), undefined, { timeout: 180_000 });
    const validation = await page.evaluate(() => window.__day28Validation);
    if (!validation?.passed || validation.actualBackend !== 'webgpu') throw new Error(`validation failed for seed ${seed}`);
    const evaluation = await page.evaluate((requestedSeed) => window.__day30Evaluate?.([requestedSeed]) ?? null, seed);
    if (!evaluation?.completed || evaluation.actualBackend !== 'webgpu' || evaluation.episodes.length !== 1) throw new Error(`evaluation failed for seed ${seed}`);
    console.log(`supplement ${index + 1}/${seeds.length} complete: seed=${seed}, return=${evaluation.episodes[0].episodeReturn}, steps=${evaluation.episodes[0].episodeLength}`);
    return { seed, validation, evaluation: evaluation.episodes[0], browser: evaluation.browser, environmentContract: evaluation.environmentContract, modelSha256: evaluation.modelSha256, errors };
  } finally {
    await page.close().catch(() => undefined);
  }
}

async function runPool(items, limit) {
  const results = [];
  let cursor = 0;
  async function worker() {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      results[index] = await evaluateSeed(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => worker()));
  return results;
}

async function captureSmoke() {
  const page = await context.newPage();
  try {
    await page.goto(productUrl, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await page.locator('button[data-action="start"]').click();
    await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PLAYING', undefined, { timeout: 180_000 });
    await page.waitForTimeout(3000);
    await page.evaluate(() => document.activeElement?.blur());
    await page.screenshot({ path: resolve(outputDirectory, 'final-human-vs-rl.png'), fullPage: true });
    const debugPage = await context.newPage();
    await debugPage.goto(debugUrl, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await debugPage.locator('details[data-role="technical-details"]').evaluate((details) => { details.open = true; });
    await debugPage.locator('button[data-action="start"]').click();
    await debugPage.waitForFunction(() => Boolean(window.__day30Ready), undefined, { timeout: 180_000 });
    await debugPage.waitForFunction(() => document.querySelector('[data-role="runtime-status-label"]')?.textContent === 'running', undefined, { timeout: 30_000 });
    const runtimeDiagnostics = await debugPage.evaluate(() => window.__day30EnvironmentDiagnostics ?? null);
    const backend = await debugPage.locator('[data-role="gameplay-backend"]').textContent();
    await debugPage.close();
    return {
      backend,
      state: await page.evaluate(() => ({ humanStage: document.querySelector('[data-role="human-stage-state"]')?.textContent, agentStage: document.querySelector('[data-role="agent-stage-state"]')?.textContent, agentScore: document.querySelector('[data-role="agent-score"]')?.textContent })),
      runtimeDiagnostics,
      productValidation: { dualCanvasCount: await page.locator('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').count() },
    };
  } finally {
    await page.close().catch(() => undefined);
  }
}

try {
  const supplements = await runPool(seeds.slice(0, 20), concurrency);
  const first = supplements[0];
  const supplementEpisodes = supplements.map((result) => result.evaluation);
  const mergedWebGpu = mergeEvaluation(base.results.webgpu, supplementEpisodes, first.browser, first.environmentContract);
  const wasm = { ...base.results.wasm, aggregate: aggregate(base.results.wasm.episodes) };
  const wasmBySeed = new Map(wasm.episodes.map((episode) => [episode.seed, episode.episodeReturn]));
  const pairedSeeds = new Set();
  const paired = mergedWebGpu.episodes.flatMap((episode) => {
    if (!wasmBySeed.has(episode.seed) || pairedSeeds.has(episode.seed)) return [];
    pairedSeeds.add(episode.seed);
    return [{ seed: episode.seed, webgpu: episode.episodeReturn, wasm: wasmBySeed.get(episode.seed), difference: episode.episodeReturn - wasmBySeed.get(episode.seed) }];
  });
  const contractSpec = JSON.parse(await readFile(resolve(outputDirectory, '../../web/public/configs/eval/breakout_contract_v2.json'), 'utf8'));
  const contract = { contractId: contractSpec.contract_id, contractPath: 'configs/eval/breakout_contract_v2.json', contractSha256: first.environmentContract.sha256, environmentId: contractSpec.environment_id, fireReset: contractSpec.fire_reset, terminalOnLifeLoss: contractSpec.terminal_on_life_loss, frameSkip: contractSpec.frame_skip, frameStack: contractSpec.frame_stack, stickyActionProbability: contractSpec.sticky_action_probability, timeLimitSemantics: contractSpec.time_limit_semantics, rawRewardRule: contractSpec.raw_reward_rule, evaluationEpsilon: contractSpec.evaluation_epsilon, finalHoldoutConcreteSeeds: contractSpec.concrete_episode_seeds };
  const gameplaySmoke = await captureSmoke();
  const comparison = { schemaVersion: 1, artifactType: 'day30_browser_policy_score_comparison', timestamp: new Date().toISOString(), pageUrl: debugUrl, productUrl, browser: first.browser, modelSha256: first.modelSha256, environmentContract: first.environmentContract, contract, uniqueSeedCount: mergedWebGpu.aggregate.uniqueSeedCount, finalBackendStrategy: { preferred: 'webgpu', fallback: 'wasm', selectedForFormalEvaluation: 'webgpu', reason: 'Day 29 established equal WASM/WebGPU policy scores; full-loop timing favored WebGPU, while WASM remains the truthful product fallback.', evaluationReuse: 'WASM retains the Day 29 30-episode artifact; WebGPU adds the 20 new native-derived seeds 323-342 to reach 50 unique seeds.' }, validation: { wasm: base.validation.wasm, webgpu: first.validation }, results: { wasm, webgpu: mergedWebGpu }, pairedScoreDifference: { count: paired.length, mean: paired.length ? paired.reduce((sum, item) => sum + item.difference, 0) / paired.length : null, bySeed: paired }, gameplaySmoke, diagnostics: { supplementErrors: supplements.flatMap((result) => result.errors) } };
  await writeFile(resolve(outputDirectory, 'parallel-increment-webgpu-evaluation.json'), `${JSON.stringify({ schemaVersion: 1, artifactType: 'day30_parallel_webgpu_supplement', seeds: seeds.slice(0, 20), uniqueSeedCount: new Set(seeds.slice(0, 20)).size, episodes: supplementEpisodes, aggregate: aggregate(supplementEpisodes) }, null, 2)}\n`, 'utf8');
  await writeFile(resolve(outputDirectory, 'final-browser-score-comparison.json'), `${JSON.stringify(comparison, null, 2)}\n`, 'utf8');
  await writeFile(resolve(outputDirectory, 'final-validation.json'), `${JSON.stringify({ schemaVersion: 1, artifactType: 'day30_final_validation', timestamp: new Date().toISOString(), productionUrl: productUrl, debugUrl, contract, uniqueSeedCount: mergedWebGpu.aggregate.uniqueSeedCount, finalBackendStrategy: comparison.finalBackendStrategy, validation: { wasm: { passed: base.validation.wasm.passed, requestedBackend: base.validation.wasm.requestedBackend, actualBackend: base.validation.wasm.actualBackend, sampleCount: base.validation.wasm.sampleCount }, webgpu: { passed: first.validation.passed, requestedBackend: first.validation.requestedBackend, actualBackend: first.validation.actualBackend, sampleCount: first.validation.sampleCount } }, evaluation: { wasm: wasm.aggregate, webgpu: mergedWebGpu.aggregate, episodeCount: { wasm: wasm.episodes.length, webgpu: mergedWebGpu.episodes.length } }, gameplaySmoke }, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({ supplement: aggregate(supplementEpisodes), finalWebGpu: mergedWebGpu.aggregate, gameplaySmoke }, null, 2));
} finally {
  await context.close().catch(() => undefined);
  await browser.close();
}
