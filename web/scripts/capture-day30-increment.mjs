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
const url = withDebug(productUrl);
const outputDirectory = resolve(String(args.get('output-dir') ?? '../assets/day30'));
const incrementCount = Number(args.get('episodes') ?? 20);
const incrementStart = Number(args.get('start-seed') ?? 323);
if (!Number.isInteger(incrementCount) || incrementCount < 20) throw new Error('--episodes must be at least 20 for the Day 30 supplement');

function withDebug(value) {
  const parsed = new URL(value);
  parsed.searchParams.set('debug', '1');
  return parsed.toString();
}

const basePath = resolve(outputDirectory, '../day29/browser-policy-score-comparison.json');
const base = JSON.parse(await readFile(basePath, 'utf8'));
const requestedSeeds = String(args.get('seeds') ?? '').split(',').map((value) => Number(value.trim())).filter((value) => Number.isInteger(value));
const seeds = requestedSeeds.length >= incrementCount
  ? requestedSeeds.slice(0, incrementCount)
  : Array.from({ length: incrementCount }, (_, index) => incrementStart + index);
if (new Set(seeds).size !== seeds.length) throw new Error('Day 30 supplement seeds must be unique');
const browser = await chromium.launch({ channel: 'chrome', headless: args.get('headed') !== true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
const page = await context.newPage();
const consoleErrors = [];
const pageErrors = [];
const requestFailures = [];
const badResponses = [];
page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
page.on('pageerror', (error) => pageErrors.push(String(error)));
page.on('requestfailed', (request) => requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));
page.on('response', (response) => { if (response.status() >= 400 && !response.url().includes('/favicon')) badResponses.push(`${response.status()} ${response.url()}`); });

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
  return {
    count: values.length,
    uniqueSeedCount: new Set(episodes.map((episode) => episode.seed)).size,
    mean,
    median: percentile(values, 0.5),
    std: Math.sqrt(variance),
    p10: percentile(values, 0.1),
    p90: percentile(values, 0.9),
    min: Math.min(...values),
    max: Math.max(...values),
    successCount: episodes.filter((episode) => episode.runtimeError === null).length,
    crashCount: episodes.filter((episode) => episode.runtimeError !== null).length,
    pairedScoreDifference: null,
  };
}

function mergeEvaluation(baseEvaluation, incrementEvaluation) {
  const episodes = [...baseEvaluation.episodes, ...incrementEvaluation.episodes];
  const timing = {
    sampleCount: baseEvaluation.timing.sampleCount + incrementEvaluation.timing.sampleCount,
    preprocessingMs: [...baseEvaluation.timing.preprocessingMs, ...incrementEvaluation.timing.preprocessingMs],
    inferenceMs: [...baseEvaluation.timing.inferenceMs, ...incrementEvaluation.timing.inferenceMs],
    totalDecisionMs: [...baseEvaluation.timing.totalDecisionMs, ...incrementEvaluation.timing.totalDecisionMs],
  };
  timing.summary = {
    preprocessingP50Ms: percentile(timing.preprocessingMs, 0.5),
    inferenceP50Ms: percentile(timing.inferenceMs, 0.5),
    totalDecisionP50Ms: percentile(timing.totalDecisionMs, 0.5),
    totalDecisionP95Ms: percentile(timing.totalDecisionMs, 0.95),
  };
  return {
    ...incrementEvaluation,
    artifactType: 'day30_browser_policy_evaluation',
    seeds: episodes.map((episode) => episode.seed),
    episodes,
    aggregate: aggregate(episodes),
    timing,
    completed: episodes.every((episode) => episode.runtimeError === null),
  };
}

async function openTechnicalDetails() {
  await page.locator('details[data-role="technical-details"]').evaluate((details) => { details.open = true; });
}

async function validateWebGpu() {
  await openTechnicalDetails();
  await page.locator('select[data-action="backend"]').selectOption('webgpu');
  await page.locator('button[data-action="validate"]').click();
  await page.waitForFunction(() => Boolean(window.__day28Validation), undefined, { timeout: 180_000 });
  const validation = await page.evaluate(() => window.__day28Validation);
  if (!validation?.passed || validation.actualBackend !== 'webgpu') throw new Error(`WebGPU validation failed: ${JSON.stringify(validation)}`);
  return validation;
}

async function captureSmoke() {
  await page.locator('details[data-role="technical-details"]').evaluate((details) => { details.open = false; });
  await page.locator('button[data-action="start"]').click();
  await page.waitForFunction(() => Boolean(window.__day30Ready), undefined, { timeout: 180_000 });
  await page.waitForTimeout(4000);
  await page.evaluate(() => document.activeElement?.blur());
  await page.screenshot({ path: resolve(outputDirectory, 'final-human-vs-rl.png'), fullPage: true });
  const runtimeDiagnostics = await page.evaluate(() => window.__day30EnvironmentDiagnostics ?? null);
  const state = await page.evaluate(() => ({
    humanStage: document.querySelector('[data-role="human-stage-state"]')?.textContent,
    agentStage: document.querySelector('[data-role="agent-stage-state"]')?.textContent,
    score: document.querySelector('[data-role="agent-score"]')?.textContent,
    lives: document.querySelector('[data-role="agent-lives"]')?.textContent,
  }));
  if (!runtimeDiagnostics || runtimeDiagnostics.dualInstanceCount !== 2) throw new Error(`dual ALE smoke failed: ${JSON.stringify(runtimeDiagnostics)}`);
  return { backend: 'webgpu', state, runtimeDiagnostics, productValidation: { dualCanvasCount: await page.locator('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').count(), crossOriginIsolated: runtimeDiagnostics.crossOriginIsolated } };
}

try {
  console.log(`Opening ${url} and supplementing ${incrementCount} WebGPU episodes from seed ${incrementStart}.`);
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.locator('button[data-action="start"]').waitFor({ state: 'visible', timeout: 30_000 });
  const contractSpec = await page.evaluate(async () => {
    const response = await fetch('/configs/eval/breakout_contract_v2.json', { cache: 'no-store' });
    return response.json();
  });
  const validation = await validateWebGpu();
  const gameplaySmoke = await captureSmoke();
  console.log(`Running ${incrementCount}-episode WebGPU supplement...`);
  const increment = await page.evaluate((requestedSeeds) => window.__day30Evaluate?.(requestedSeeds) ?? null, seeds);
  if (!increment?.completed || increment.actualBackend !== 'webgpu' || increment.episodes.length !== incrementCount) throw new Error(`WebGPU supplement failed: ${JSON.stringify(increment?.aggregate)}`);
  const mergedWebGpu = mergeEvaluation(base.results.webgpu, increment);
  const wasm = { ...base.results.wasm, aggregate: aggregate(base.results.wasm.episodes) };
  const wasmBySeed = new Map(wasm.episodes.map((episode) => [episode.seed, episode.episodeReturn]));
  const paired = mergedWebGpu.episodes.flatMap((episode) => wasmBySeed.has(episode.seed) ? [{ seed: episode.seed, webgpu: episode.episodeReturn, wasm: wasmBySeed.get(episode.seed), difference: episode.episodeReturn - wasmBySeed.get(episode.seed) }] : []);
  const contract = {
    contractId: contractSpec.contract_id,
    contractPath: 'configs/eval/breakout_contract_v2.json',
    contractSha256: increment.environmentContract.sha256,
    environmentId: contractSpec.environment_id,
    fireReset: contractSpec.fire_reset,
    terminalOnLifeLoss: contractSpec.terminal_on_life_loss,
    frameSkip: contractSpec.frame_skip,
    frameStack: contractSpec.frame_stack,
    stickyActionProbability: contractSpec.sticky_action_probability,
    timeLimitSemantics: contractSpec.time_limit_semantics,
    rawRewardRule: contractSpec.raw_reward_rule,
    evaluationEpsilon: contractSpec.evaluation_epsilon,
    finalHoldoutConcreteSeeds: contractSpec.concrete_episode_seeds,
  };
  const comparison = {
    schemaVersion: 1,
    artifactType: 'day30_browser_policy_score_comparison',
    timestamp: new Date().toISOString(),
    pageUrl: url,
    productUrl,
    browser: increment.browser,
    modelSha256: increment.modelSha256,
    environmentContract: increment.environmentContract,
    contract,
    uniqueSeedCount: mergedWebGpu.aggregate.uniqueSeedCount,
    finalBackendStrategy: {
      preferred: 'webgpu',
      fallback: 'wasm',
      selectedForFormalEvaluation: 'webgpu',
      reason: 'Day 29 established equal WASM/WebGPU policy scores; full-loop timing favored WebGPU, while WASM remains the truthful product fallback.',
      evaluationReuse: 'WASM retains the Day 29 30-episode artifact; WebGPU adds the 20 new native-derived seeds 323-342 to reach 50 unique seeds.',
    },
    validation: { wasm: base.validation.wasm, webgpu: validation },
    results: { wasm, webgpu: mergedWebGpu },
    pairedScoreDifference: {
      count: paired.length,
      mean: paired.length ? paired.reduce((sum, item) => sum + item.difference, 0) / paired.length : null,
      bySeed: paired,
    },
    gameplaySmoke,
    diagnostics: { consoleErrors, pageErrors, requestFailures, badResponses },
  };
  await writeFile(resolve(outputDirectory, 'increment-webgpu-evaluation.json'), `${JSON.stringify(increment, null, 2)}\n`, 'utf8');
  await writeFile(resolve(outputDirectory, 'final-browser-score-comparison.json'), `${JSON.stringify(comparison, null, 2)}\n`, 'utf8');
  await writeFile(resolve(outputDirectory, 'final-validation.json'), `${JSON.stringify({
    schemaVersion: 1,
    artifactType: 'day30_final_validation',
    timestamp: new Date().toISOString(),
    productionUrl: productUrl,
    debugUrl: url,
    uniqueSeedCount: mergedWebGpu.aggregate.uniqueSeedCount,
    contract,
    finalBackendStrategy: comparison.finalBackendStrategy,
    validation: { wasm: { passed: base.validation.wasm.passed, requestedBackend: base.validation.wasm.requestedBackend, actualBackend: base.validation.wasm.actualBackend, sampleCount: base.validation.wasm.sampleCount }, webgpu: { passed: validation.passed, requestedBackend: validation.requestedBackend, actualBackend: validation.actualBackend, sampleCount: validation.sampleCount } },
    evaluation: { wasm: wasm.aggregate, webgpu: mergedWebGpu.aggregate, episodeCount: { wasm: wasm.episodes.length, webgpu: mergedWebGpu.episodes.length } },
    gameplaySmoke,
    diagnostics: { consoleErrors, pageErrors, requestFailures, badResponses },
  }, null, 2)}\n`, 'utf8');
  if (pageErrors.length || requestFailures.length || badResponses.length || consoleErrors.length) throw new Error(`browser diagnostics were not clean: ${JSON.stringify({ consoleErrors, pageErrors, requestFailures, badResponses })}`);
  console.log(JSON.stringify({ url, increment: increment.aggregate, finalWebGpu: mergedWebGpu.aggregate, gameplaySmoke, outputDirectory }, null, 2));
} finally {
  await context.close().catch(() => undefined);
  await browser.close();
}
