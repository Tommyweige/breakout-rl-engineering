import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import process from 'node:process';

import { chromium } from 'playwright-core';

function parseArgs(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (!value?.startsWith('--')) continue;
    const key = value.slice(2);
    const next = argv[index + 1];
    if (next && !next.startsWith('--')) {
      values.set(key, next);
      index += 1;
    } else values.set(key, true);
  }
  return values;
}

const args = parseArgs(process.argv.slice(2));
const url = String(args.get('url') ?? 'http://127.0.0.1:5173/');
const outputDirectory = resolve(String(args.get('output-dir') ?? '../assets/day29'));
const headless = args.get('headless') === true;
const evaluationCount = args.has('episodes') ? Number(args.get('episodes')) : 30;
const gameplayTimeoutMs = Number(args.get('gameplay-timeout-ms') ?? 360_000);
if (!Number.isInteger(evaluationCount) || evaluationCount < 1 || evaluationCount > 30) throw new Error('--episodes must be an integer between 1 and 30');

const paths = {
  aiScreenshot: resolve(outputDirectory, 'browser-ai-demo.png'),
  dualScreenshot: resolve(outputDirectory, 'human-vs-rl-dual-breakout.png'),
  preprocessingScreenshot: resolve(outputDirectory, 'browser-preprocessing-debug.png'),
  comparison: resolve(outputDirectory, 'browser-policy-score-comparison.json'),
  timing: resolve(outputDirectory, 'agent-loop-timing.json'),
  preprocessing: resolve(outputDirectory, 'preprocessing-parity.json'),
  environment: resolve(outputDirectory, 'environment-parity.json'),
  metadata: resolve(outputDirectory, 'capture-metadata.json'),
};
await mkdir(outputDirectory, { recursive: true });

async function readJsonOrNull(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch {
    return null;
  }
}

const consoleErrors = [];
const pageErrors = [];
const requestFailures = [];
const badResponses = [];
const browser = await chromium.launch({ channel: 'chrome', headless });
const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
const page = await context.newPage();
page.on('console', (message) => {
  if (message.type() === 'error') consoleErrors.push(message.text());
  if (message.type() === 'warning') console.log(`[browser:warning] ${message.text()}`);
});
page.on('pageerror', (error) => pageErrors.push(String(error)));
page.on('requestfailed', (request) => requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));
page.on('response', (response) => {
  if (response.status() >= 400 && !response.url().includes('/favicon')) badResponses.push(`${response.status()} ${response.url()}`);
});

async function validateBackend(backend) {
  await page.locator('select[data-action="backend"]').selectOption(backend);
  await page.locator('button[data-action="validate"]').click();
  const artifactName = backend === 'wasm' ? '__day27Validation' : '__day28Validation';
  await page.waitForFunction((name) => Boolean(window[name]), artifactName, { timeout: 180_000 });
  await page.waitForFunction(() => !(document.querySelector('select[data-action="backend"]') instanceof HTMLSelectElement && document.querySelector('select[data-action="backend"]').disabled), undefined, { timeout: 30_000 });
  const artifact = await page.evaluate((name) => window[name], artifactName);
  if (!artifact?.passed || artifact.requestedBackend !== backend || artifact.actualBackend !== backend) {
    throw new Error(`${backend} validation did not pass truthfully: ${JSON.stringify(artifact)}`);
  }
  return artifact;
}

async function captureGameplay() {
  await page.locator('button[data-action="start"]').click();
  await page.waitForFunction(() => Boolean(window.__day29Ready), undefined, { timeout: 180_000 });
  await page.waitForFunction(() => {
    const action = document.querySelector('[data-role="current-action"]')?.textContent;
    const latency = document.querySelector('[data-role="inference-latency"]')?.textContent;
    return Boolean(action && action !== 'NOOP' && latency && latency !== '—');
  }, undefined, { timeout: 180_000 });
  await page.screenshot({ path: paths.aiScreenshot, fullPage: true });
  await page.screenshot({ path: paths.dualScreenshot, fullPage: true });
  const debug = page.locator('[data-role="preprocess-debug"]');
  await debug.screenshot({ path: paths.preprocessingScreenshot });

  await page.keyboard.down('ArrowLeft');
  await page.waitForTimeout(100);
  const keyboardAction = await page.locator('[data-role="human-input"]').textContent();
  await page.keyboard.up('ArrowLeft');
  if (!keyboardAction?.includes('LEFT')) throw new Error(`HTTPS keyboard input did not reach the Human panel: ${keyboardAction}`);
  await page.keyboard.down('Space');
  await page.waitForTimeout(100);
  const spaceAction = await page.locator('[data-role="human-input"]').textContent();
  await page.keyboard.up('Space');
  if (!spaceAction?.includes('FIRE')) throw new Error(`HTTPS Space input did not reach the Human panel: ${spaceAction}`);

  await page.locator('button[data-action="pause"]').click();
  await page.waitForFunction(() => document.querySelector('[data-role="runtime-status-label"]')?.textContent === 'paused', undefined, { timeout: 30_000 });
  const pausedFrame = await page.locator('[data-role="agent-frame"]').textContent();
  await page.waitForTimeout(250);
  const pausedFrameAfterWait = await page.locator('[data-role="agent-frame"]').textContent();
  await page.locator('button[data-action="start"]').click();
  await page.waitForFunction(() => document.querySelector('[data-role="runtime-status-label"]')?.textContent === 'running', undefined, { timeout: 30_000 });
  await page.locator('button[data-action="reset"]').click();
  await page.waitForTimeout(250);
  const firstResetStatus = await page.locator('[data-role="runtime-status-label"]').textContent();
  await page.locator('button[data-action="reset"]').click();
  await page.waitForTimeout(250);
  const secondResetStatus = await page.locator('[data-role="runtime-status-label"]').textContent();
  await page.locator('button[data-action="start"]').click();
  await page.waitForFunction(() => document.querySelector('[data-role="runtime-status-label"]')?.textContent === 'running', undefined, { timeout: 30_000 });
  await page.waitForFunction(() => {
    const action = document.querySelector('[data-role="current-action"]')?.textContent;
    const latency = document.querySelector('[data-role="inference-latency"]')?.textContent;
    return Boolean(action && action !== 'NOOP' && latency && latency !== '—');
  }, undefined, { timeout: 180_000 });

  const runtimeDiagnostics = await page.evaluate(() => window.__day29EnvironmentDiagnostics ?? null);
  if (!runtimeDiagnostics || runtimeDiagnostics.dualInstanceCount !== 2 || runtimeDiagnostics.humanInstanceId === runtimeDiagnostics.agentInstanceId) {
    const diagnostics = await page.evaluate(() => ({
      ready: window.__day29Ready,
      runtimeStatus: document.querySelector('[data-role="runtime-status-label"]')?.textContent,
      agentStage: document.querySelector('[data-role="agent-stage-state"]')?.textContent,
      backend: document.querySelector('[data-role="gameplay-backend"]')?.textContent,
      message: document.querySelector('[data-role="validation-message"]')?.textContent,
    }));
    throw new Error(`dual ALE runtime diagnostics were not published: ${JSON.stringify({ runtimeDiagnostics, diagnostics })}`);
  }

  let completed = true;
  try {
    await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'GAME OVER', undefined, { timeout: gameplayTimeoutMs });
  } catch {
    completed = false;
  }
  const state = await page.evaluate(() => ({
      humanStage: document.querySelector('[data-role="human-stage-state"]')?.textContent,
      agentStage: document.querySelector('[data-role="agent-stage-state"]')?.textContent,
      action: document.querySelector('[data-role="current-action"]')?.textContent,
      episodeReturn: document.querySelector('[data-role="episode-return"]')?.textContent,
      latency: document.querySelector('[data-role="inference-latency"]')?.textContent,
      backend: document.querySelector('[data-role="gameplay-backend"]')?.textContent,
      parity: document.querySelector('[data-role="environment-parity"]')?.textContent,
    }));
  if (completed && state.humanStage !== 'GAME OVER') throw new Error(`Human episode did not complete with the RL episode: ${JSON.stringify(state)}`);
  return {
    completed,
    state,
    runtimeDiagnostics,
    productValidation: {
      dualCanvasCount: await page.locator('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').count(),
      keyboardAction,
      spaceAction,
      pauseObserved: pausedFrame !== null && pausedFrameAfterWait === pausedFrame,
      pauseStatus: 'paused',
      resumeObserved: true,
      resetBothObserved: firstResetStatus === 'ready' && secondResetStatus === 'ready',
      crossOriginIsolated: runtimeDiagnostics.crossOriginIsolated,
    },
  };
}

async function evaluate(backend, seeds) {
  await page.locator('select[data-action="backend"]').selectOption(backend);
  if (backend === 'wasm') await validateBackend('wasm');
  else await validateBackend('webgpu');
  await page.waitForFunction(() => typeof window.__day29Evaluate === 'function', undefined, { timeout: 30_000 });
  const artifact = await page.evaluate((seedList) => window.__day29Evaluate?.(seedList) ?? null, seeds);
  if (!artifact?.completed || artifact.actualBackend !== backend) {
    const diagnostics = await page.evaluate(() => ({
      status: document.querySelector('[data-role="evaluation-status"]')?.textContent,
      result: document.querySelector('[data-role="evaluation-result"]')?.textContent,
      message: document.querySelector('[data-role="evaluation-message"]')?.textContent,
      validation: document.querySelector('[data-role="validation-message"]')?.textContent,
    }));
    throw new Error(`${backend} multi-episode evaluation did not complete: ${JSON.stringify({ artifact, diagnostics })}`);
  }
  return artifact;
}

try {
  console.log(`Opening ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.locator('button[data-action="validate"]').waitFor({ state: 'visible', timeout: 30_000 });

  const wasmValidation = await validateBackend('wasm');
  const webgpuValidation = await validateBackend('webgpu');
  const gameplay = await captureGameplay();
  if (!gameplay.completed) throw new Error(`WebGPU gameplay did not reach game over within ${gameplayTimeoutMs}ms: ${JSON.stringify(gameplay.state)}`);

  const overlapSeeds = [...Array.from({ length: 5 }, (_, index) => 101 + index), ...Array.from({ length: 5 }, (_, index) => 202 + index), ...Array.from({ length: 5 }, (_, index) => 303 + index)];
  const fixedSeeds = evaluationCount === 30 ? undefined : overlapSeeds.slice(0, evaluationCount);
  const webgpuEvaluation = await evaluate('webgpu', fixedSeeds);
  const wasmEvaluation = await evaluate('wasm', fixedSeeds);
  const wasmBySeed = new Map(wasmEvaluation.episodes.map((episode) => [episode.seed, episode.episodeReturn]));
  const pairedDifferences = webgpuEvaluation.episodes.map((episode) => ({ seed: episode.seed, webgpu: episode.episodeReturn, wasm: wasmBySeed.get(episode.seed) ?? null, difference: wasmBySeed.has(episode.seed) ? episode.episodeReturn - wasmBySeed.get(episode.seed) : null }));
  const comparableDifferences = pairedDifferences.flatMap((item) => item.difference === null ? [] : [item.difference]);
  const comparison = {
    schemaVersion: 1,
    artifactType: 'day29_browser_policy_score_comparison',
    timestamp: new Date().toISOString(),
    pageUrl: url,
    browser: webgpuEvaluation.browser,
    modelSha256: webgpuEvaluation.modelSha256,
    environmentContract: webgpuEvaluation.environmentContract,
    validation: { wasm: wasmValidation, webgpu: webgpuValidation },
    results: { wasm: wasmEvaluation, webgpu: webgpuEvaluation },
    pairedScoreDifference: {
      count: comparableDifferences.length,
      mean: comparableDifferences.length ? comparableDifferences.reduce((sum, value) => sum + value, 0) / comparableDifferences.length : null,
      bySeed: pairedDifferences,
    },
    gameplaySmoke: { backend: 'webgpu', completedEpisode: gameplay.completed, state: gameplay.state, productValidation: gameplay.productValidation, runtimeDiagnostics: gameplay.runtimeDiagnostics },
  };
  await writeFile(paths.comparison, `${JSON.stringify(comparison, null, 2)}\n`, 'utf8');
  await writeFile(paths.timing, `${JSON.stringify({ schemaVersion: 1, artifactType: 'day29_agent_loop_timing', modelSha256: comparison.modelSha256, browser: comparison.browser, backends: { wasm: wasmEvaluation.timing, webgpu: webgpuEvaluation.timing } }, null, 2)}\n`, 'utf8');
  const preprocessing = await readJsonOrNull(paths.preprocessing) ?? {
    schemaVersion: 1,
    artifactType: 'day29_preprocessing_parity',
    source: 'live ALE WASM gameplay + Browser debug canvases',
    actualBackends: ['wasm', 'webgpu'],
    processedFrameShape: [84, 84],
    frameStackShape: [4, 84, 84],
    observationDtype: 'uint8',
    normalizedInput: 'Float32Array / 255.0 in OrtWebPolicy',
  };
  preprocessing.modelSha256 = comparison.modelSha256;
  preprocessing.browser = comparison.browser;
  preprocessing.debugScreenshot = 'assets/day29/browser-preprocessing-debug.png';
  await writeFile(paths.preprocessing, `${JSON.stringify(preprocessing, null, 2)}\n`, 'utf8');
  const environment = await readJsonOrNull(paths.environment) ?? {
    schemaVersion: 1,
    artifactType: 'day29_environment_parity',
    actualBackends: ['wasm', 'webgpu'],
  };
  environment.contract = comparison.environmentContract;
  environment.runtime = comparison.gameplaySmoke.runtimeDiagnostics;
  environment.gameplaySmoke = comparison.gameplaySmoke;
  await writeFile(paths.environment, `${JSON.stringify(environment, null, 2)}\n`, 'utf8');
  await writeFile(paths.metadata, `${JSON.stringify({ schemaVersion: 1, artifactType: 'day29_browser_capture', url, browser: comparison.browser, playwrightBrowserVersion: browser.version(), modelSha256: comparison.modelSha256, actualBackend: 'webgpu', environmentParity: comparison.environmentContract, runtimeDiagnostics: comparison.gameplaySmoke.runtimeDiagnostics, productValidation: comparison.gameplaySmoke.productValidation, validationArtifacts: { wasm: 'assets/day29/browser-policy-score-comparison.json#/validation/wasm', webgpu: 'assets/day29/browser-policy-score-comparison.json#/validation/webgpu' }, screenshots: { aiGameplay: 'assets/day29/browser-ai-demo.png', humanVsRl: 'assets/day29/human-vs-rl-dual-breakout.png', preprocessing: 'assets/day29/browser-preprocessing-debug.png' }, consoleErrors, pageErrors, requestFailures, badResponses }, null, 2)}\n`, 'utf8');

  if (pageErrors.length || requestFailures.length || badResponses.length || consoleErrors.length) {
    throw new Error(`browser diagnostics were not clean: ${JSON.stringify({ consoleErrors, pageErrors, requestFailures, badResponses })}`);
  }
  console.log(JSON.stringify({ url, browser: comparison.browser, modelSha256: comparison.modelSha256, webgpuGameplay: gameplay, wasm: wasmEvaluation.aggregate, webgpu: webgpuEvaluation.aggregate, outputs: paths }, null, 2));
} finally {
  await context.close().catch(() => undefined);
  await browser.close();
}
