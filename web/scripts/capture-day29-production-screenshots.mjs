import { mkdir, writeFile } from 'node:fs/promises';
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
const url = String(args.get('url') ?? 'https://day29-human-vs-rl.breakout-rl-engineering.pages.dev/');
const outputDirectory = resolve(String(args.get('output-dir') ?? '../assets/day29'));
await mkdir(outputDirectory, { recursive: true });
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

async function validateWebGpu() {
  await page.locator('select[data-action="backend"]').selectOption('webgpu');
  await page.locator('button[data-action="validate"]').click();
  await page.waitForFunction(() => Boolean(window.__day28Validation), undefined, { timeout: 180_000 });
  const artifact = await page.evaluate(() => window.__day28Validation);
  if (!artifact?.passed || artifact.actualBackend !== 'webgpu') throw new Error(`WebGPU validation failed: ${JSON.stringify(artifact)}`);
  return artifact;
}

try {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.locator('button[data-action="validate"]').waitFor({ state: 'visible', timeout: 30_000 });
  const validation = await validateWebGpu();
  await page.locator('button[data-action="start"]').click();
  await page.waitForFunction(() => Boolean(window.__day29Ready), undefined, { timeout: 180_000 });
  await page.waitForFunction(() => {
    const action = document.querySelector('[data-role="current-action"]')?.textContent;
    const latency = document.querySelector('[data-role="inference-latency"]')?.textContent;
    return Boolean(action && action !== 'NOOP' && latency && latency !== '—');
  }, undefined, { timeout: 180_000 });
  await page.screenshot({ path: resolve(outputDirectory, 'browser-ai-demo.png'), fullPage: true });
  await page.screenshot({ path: resolve(outputDirectory, 'human-vs-rl-dual-breakout.png'), fullPage: true });
  await page.locator('[data-role="preprocess-debug"]').screenshot({ path: resolve(outputDirectory, 'browser-preprocessing-debug.png') });

  await page.keyboard.down('ArrowLeft');
  await page.waitForTimeout(100);
  const keyboardAction = await page.locator('[data-role="human-input"]').textContent();
  await page.keyboard.up('ArrowLeft');
  await page.keyboard.down('Space');
  await page.waitForTimeout(100);
  const spaceAction = await page.locator('[data-role="human-input"]').textContent();
  await page.keyboard.up('Space');
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
  const runtimeDiagnostics = await page.evaluate(() => window.__day29EnvironmentDiagnostics ?? null);
  const smoke = {
    backend: 'webgpu',
    completedEpisode: false,
    state: await page.evaluate(() => ({
      humanStage: document.querySelector('[data-role="human-stage-state"]')?.textContent,
      agentStage: document.querySelector('[data-role="agent-stage-state"]')?.textContent,
      action: document.querySelector('[data-role="current-action"]')?.textContent,
      episodeReturn: document.querySelector('[data-role="episode-return"]')?.textContent,
      latency: document.querySelector('[data-role="inference-latency"]')?.textContent,
      backend: document.querySelector('[data-role="gameplay-backend"]')?.textContent,
      parity: document.querySelector('[data-role="environment-parity"]')?.textContent,
    })),
    productValidation: {
      dualCanvasCount: await page.locator('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').count(),
      keyboardAction,
      spaceAction,
      pauseObserved: pausedFrame !== null && pausedFrameAfterWait === pausedFrame,
      pauseStatus: 'paused',
      resumeObserved: true,
      resetBothObserved: firstResetStatus === 'ready' && secondResetStatus === 'ready',
      crossOriginIsolated: runtimeDiagnostics?.crossOriginIsolated ?? false,
    },
    runtimeDiagnostics,
    validation: { actualBackend: validation.actualBackend, passed: validation.passed },
  };
  await writeFile(resolve(outputDirectory, 'production-smoke.json'), `${JSON.stringify(smoke, null, 2)}\n`, 'utf8');
  await writeFile(resolve(outputDirectory, 'capture-metadata.json'), `${JSON.stringify({
    schemaVersion: 2,
    artifactType: 'day29_browser_capture',
    url,
    browser: validation.browser,
    modelSha256: validation.modelSha256,
    actualBackend: 'webgpu',
    environmentParity: validation.environmentContract,
    runtimeDiagnostics,
    productValidation: smoke.productValidation,
    screenshots: {
      aiGameplay: 'assets/day29/browser-ai-demo.png',
      humanVsRl: 'assets/day29/human-vs-rl-dual-breakout.png',
      preprocessing: 'assets/day29/browser-preprocessing-debug.png',
    },
    consoleErrors,
    pageErrors,
    requestFailures,
    badResponses,
  }, null, 2)}\n`, 'utf8');
  if (pageErrors.length || requestFailures.length || badResponses.length || consoleErrors.length) {
    throw new Error(`production browser diagnostics were not clean: ${JSON.stringify({ consoleErrors, pageErrors, requestFailures, badResponses })}`);
  }
  console.log(JSON.stringify({ url, smoke, screenshots: outputDirectory }, null, 2));
} finally {
  await context.close().catch(() => undefined);
  await browser.close();
}
