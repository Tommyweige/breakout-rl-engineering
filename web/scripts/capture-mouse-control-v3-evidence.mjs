import { mkdir, writeFile } from 'node:fs/promises';
import { join, relative, resolve } from 'node:path';
import process from 'node:process';

import { chromium } from 'playwright-core';

function parseArgs(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (!value?.startsWith('--')) continue;
    const next = argv[index + 1];
    if (next && !next.startsWith('--')) {
      values.set(value.slice(2), next);
      index += 1;
    } else {
      values.set(value.slice(2), true);
    }
  }
  return values;
}

const args = parseArgs(process.argv.slice(2));
const url = String(args.get('url') ?? 'http://127.0.0.1:5173/');
const outputDirectory = resolve(String(args.get('output-dir') ?? '../assets/mouse-control-v3'));
const repoRoot = resolve(process.cwd(), '..');
const stationaryDurationTargetMs = 5_000;
const targets = [0.20, 0.80, 0.35];

await mkdir(outputDirectory, { recursive: true });

function withDebug(value) {
  const parsed = new URL(value);
  parsed.searchParams.set('debug', '1');
  return parsed.toString();
}

function outputPath(fileName) {
  return join(outputDirectory, fileName);
}

function repoRelative(fileName) {
  return relative(repoRoot, outputPath(fileName)).replaceAll('\\', '/');
}

function attachDiagnostics(page) {
  const diagnostics = { consoleErrors: [], pageErrors: [], requestFailures: [], badResponses: [] };
  page.on('console', (message) => {
    if (message.type() === 'error') diagnostics.consoleErrors.push(message.text());
  });
  page.on('pageerror', (error) => diagnostics.pageErrors.push(String(error)));
  page.on('requestfailed', (request) => diagnostics.requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));
  page.on('response', (response) => {
    if (response.status() >= 400 && !response.url().includes('/favicon')) diagnostics.badResponses.push(`${response.status()} ${response.url()}`);
  });
  return diagnostics;
}

function assertDiagnostics(diagnostics, scope) {
  const failures = Object.entries(diagnostics).filter(([, values]) => values.length > 0);
  if (failures.length > 0) throw new Error(`${scope} diagnostics failed: ${JSON.stringify(Object.fromEntries(failures))}`);
}

async function gotoApp(page, targetUrl) {
  await page.goto(targetUrl, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.locator('button[data-action="start"]').waitFor({ state: 'visible', timeout: 30_000 });
}

async function waitForPlaying(page) {
  await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PLAYING', undefined, { timeout: 180_000 });
}

async function moveTarget(page, canvas, normalizedX) {
  const box = await canvas.boundingBox();
  if (!box) throw new Error('human canvas has no bounding box');
  await canvas.hover({ position: { x: box.width * normalizedX, y: box.height * 0.70 } });
  if (await page.locator('[data-role="cursor-target-x"]').count() === 0) return;
  await page.waitForFunction(
    (expected) => Math.abs((window.__mouseControlV3Diagnostics?.cursorTargetX ?? -1) - expected) < 0.01,
    normalizedX,
    { timeout: 10_000 },
  );
}

async function readDiagnostics(page) {
  return page.evaluate(() => window.__mouseControlV3Diagnostics ?? null);
}

async function waitForActionAfter(page, action, afterDecision) {
  await page.waitForFunction(
    ({ targetAction, decision }) => window.__mouseControlV3Diagnostics?.actionHistory.some((sample) => sample.decision > decision && sample.action === targetAction),
    { targetAction: action, decision: afterDecision },
    { timeout: 10_000 },
  );
  return page.evaluate(
    ({ targetAction, decision }) => window.__mouseControlV3Diagnostics?.actionHistory.find((sample) => sample.decision > decision && sample.action === targetAction) ?? null,
    { targetAction: action, decision: afterDecision },
  );
}

function settledMetrics(samples) {
  const directional = samples.filter((sample) => sample.action === 'LEFT' || sample.action === 'RIGHT');
  let leftRightReversalCount = 0;
  for (let index = 1; index < directional.length; index += 1) {
    if (directional[index]?.action !== directional[index - 1]?.action) leftRightReversalCount += 1;
  }
  const errors = samples.map((sample) => sample.positionError).filter((value) => typeof value === 'number').map((value) => Math.abs(value));
  return {
    leftRightReversalCount,
    noopRatioAfterSettled: samples.length ? samples.filter((sample) => sample.action === 'NOOP').length / samples.length : 1,
    maxSettledError: errors.length ? Math.max(...errors) : null,
    directionalActions: directional.length,
    oscillationObserved: leftRightReversalCount > 1,
  };
}

async function runScenario(mode) {
  const browser = await chromium.launch({ channel: 'chrome', headless: true, args: mode.args });
  const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });
  const productPage = await context.newPage();
  const productDiagnostics = attachDiagnostics(productPage);
  let debugPage;
  let debugDiagnostics;
  try {
    await gotoApp(productPage, url);
    const layout = await productPage.evaluate(() => ({
      viewportHeight: innerHeight,
      documentHeight: document.documentElement.scrollHeight,
      viewportWidth: document.documentElement.clientWidth,
      documentWidth: document.documentElement.scrollWidth,
      canvasCount: document.querySelectorAll('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').length,
      technicalDetailsPresent: Boolean(document.querySelector('[data-role="technical-details"]')),
    }));
    if (layout.documentHeight > layout.viewportHeight + 1 || layout.documentWidth > layout.viewportWidth + 1) throw new Error(`production layout regressed: ${JSON.stringify(layout)}`);
    if (layout.canvasCount !== 2 || layout.technicalDetailsPresent) throw new Error(`production surface is incomplete: ${JSON.stringify(layout)}`);

    await productPage.evaluate(() => {
      const original = HTMLCanvasElement.prototype.requestPointerLock;
      window.__mouseV3PointerLockCalls = 0;
      if (typeof original === 'function') {
        HTMLCanvasElement.prototype.requestPointerLock = function (...requestArgs) {
          window.__mouseV3PointerLockCalls += 1;
          return original.apply(this, requestArgs);
        };
      }
    });
    await productPage.locator('button[data-action="start"]').click();
    await waitForPlaying(productPage);
    await productPage.locator('select[data-action="input-mode"]').selectOption('mouse');
    const productCanvas = productPage.locator('canvas[data-role="human-canvas"]');
    await moveTarget(productPage, productCanvas, targets[1]);
    await productPage.waitForFunction(() => document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === false, undefined, { timeout: 10_000 });
    const productPointerState = await productPage.evaluate(() => ({
      pointerLockUsed: window.__mouseV3PointerLockCalls > 0,
      pointerLockElement: document.pointerLockElement,
      systemCursor: getComputedStyle(document.querySelector('canvas[data-role="human-canvas"]')).cursor,
      targetMarkerVisible: document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === false,
    }));
    const productScreenshot = mode.name === 'webgpu-preferred' ? 'production-mouse-v3.png' : 'production-mouse-v3-wasm-fallback.png';
    await productPage.screenshot({ path: outputPath(productScreenshot), fullPage: true });
    await productPage.close();

    debugPage = await context.newPage();
    debugDiagnostics = attachDiagnostics(debugPage);
    await gotoApp(debugPage, withDebug(url));
    await debugPage.locator('button[data-action="start"]').click();
    await waitForPlaying(debugPage);
    await debugPage.locator('select[data-action="input-mode"]').selectOption('mouse');
    const debugCanvas = debugPage.locator('canvas[data-role="human-canvas"]');
    await moveTarget(debugPage, debugCanvas, targets[1]);
    const fixedTargetStart = await readDiagnostics(debugPage);
    if (!fixedTargetStart) throw new Error('missing Mouse v3 diagnostics after setting a fixed target');
    const stationaryStartDecision = fixedTargetStart.decisionCount;
    const stationaryStartedAt = Date.now();
    await debugPage.waitForTimeout(stationaryDurationTargetMs + 150);
    const stationary = await debugPage.evaluate(({ afterDecision, startedAt }) => {
      const diagnostics = window.__mouseControlV3Diagnostics;
      return {
        durationMs: Date.now() - startedAt,
        samples: diagnostics?.actionHistory.filter((sample) => sample.decision > afterDecision) ?? [],
        finalAction: diagnostics?.executedHumanAction ?? null,
      };
    }, { afterDecision: stationaryStartDecision, startedAt: stationaryStartedAt });
    const stationaryMetrics = settledMetrics(stationary.samples);
    if (stationary.durationMs < stationaryDurationTargetMs || stationary.finalAction !== 'NOOP' || stationaryMetrics.oscillationObserved) {
      throw new Error(`static-target QA failed: ${JSON.stringify({ stationary, stationaryMetrics })}`);
    }

    // The static target window is intentionally long enough to expose a real
    // game-loop regression. Start a fresh episode before direction-response
    // checks so a natural game-over cannot hide a pointer response.
    await debugPage.locator('button[data-action="reset"]').click();
    await debugPage.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'READY', undefined, { timeout: 30_000 });
    await debugPage.locator('button[data-action="start"]').click();
    await waitForPlaying(debugPage);
    await debugPage.locator('select[data-action="input-mode"]').selectOption('mouse');

    const targetResponseSamples = [];
    const targetScenarioStart = await readDiagnostics(debugPage);
    if (!targetScenarioStart) throw new Error('missing Mouse v3 diagnostics after target-response restart');
    let afterDecision = targetScenarioStart.decisionCount;
    await moveTarget(debugPage, debugCanvas, targets[0]);
    const leftResponse = await waitForActionAfter(debugPage, 'LEFT', afterDecision);
    if (!leftResponse) throw new Error('target 20% did not trigger LEFT');
    targetResponseSamples.push({ target: targets[0], action: leftResponse.action, decision: leftResponse.decision });
    afterDecision = leftResponse.decision;
    await moveTarget(debugPage, debugCanvas, targets[1]);
    const rightResponse = await waitForActionAfter(debugPage, 'RIGHT', afterDecision);
    if (!rightResponse) throw new Error('target 80% did not trigger RIGHT');
    targetResponseSamples.push({ target: targets[1], action: rightResponse.action, decision: rightResponse.decision });
    afterDecision = rightResponse.decision;
    await moveTarget(debugPage, debugCanvas, targets[2]);
    const finalLeftResponse = await waitForActionAfter(debugPage, 'LEFT', afterDecision);
    if (!finalLeftResponse) throw new Error('target 35% did not trigger LEFT');
    targetResponseSamples.push({ target: targets[2], action: finalLeftResponse.action, decision: finalLeftResponse.decision });

    await debugPage.locator('select[data-action="input-mode"]').selectOption('keyboard');
    await debugPage.waitForFunction(() => window.__mouseControlV3Diagnostics?.cursorTargetX === null && window.__mouseControlV3Diagnostics.motionState === 'STOPPED' && document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === true, undefined, { timeout: 10_000 });
    const modeSwitchPassed = true;

    await debugPage.locator('select[data-action="input-mode"]').selectOption('mouse');
    await moveTarget(debugPage, debugCanvas, targets[1]);
    await debugPage.locator('button[data-action="pause"]').click();
    await debugPage.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PAUSED' && window.__mouseControlV3Diagnostics?.cursorTargetX === null && window.__mouseControlV3Diagnostics.motionState === 'STOPPED', undefined, { timeout: 10_000 });
    const pauseClearPassed = true;

    await debugPage.locator('button[data-action="start"]').click();
    await waitForPlaying(debugPage);
    await debugPage.locator('select[data-action="input-mode"]').selectOption('mouse');
    await moveTarget(debugPage, debugCanvas, targets[1]);
    await debugPage.locator('button[data-action="reset"]').click();
    await debugPage.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'READY' && window.__mouseControlV3Diagnostics?.cursorTargetX === null && window.__mouseControlV3Diagnostics.motionState === 'STOPPED', undefined, { timeout: 30_000 });
    const restartClearPassed = true;

    await debugPage.locator('select[data-action="input-mode"]').selectOption('keyboard');
    const actualBackend = await debugPage.locator('[data-role="gameplay-backend"]').textContent();
    const environmentDiagnostics = await debugPage.evaluate(() => window.__day30EnvironmentDiagnostics ?? null);
    assertDiagnostics(productDiagnostics, `${mode.name} production`);
    assertDiagnostics(debugDiagnostics, `${mode.name} debug`);

    const mouseEvidence = {
      schemaVersion: 1,
      artifactType: 'issue70_mouse_absolute_hysteresis_qa',
      timestamp: new Date().toISOString(),
      url,
      pointerLockUsed: productPointerState.pointerLockUsed,
      systemCursorVisible: productPointerState.pointerLockElement === null && productPointerState.systemCursor !== 'none',
      startThreshold: fixedTargetStart.startThreshold,
      stopThreshold: fixedTargetStart.stopThreshold,
      stationaryTargetDurationMs: stationary.durationMs,
      stationarySampleCount: stationary.samples.length,
      stationaryDirectionalActionCount: stationaryMetrics.directionalActions,
      leftRightReversalCount: stationaryMetrics.leftRightReversalCount,
      noopRatioAfterSettled: stationaryMetrics.noopRatioAfterSettled,
      maxSettledError: stationaryMetrics.maxSettledError,
      oscillationObserved: stationaryMetrics.oscillationObserved,
      targetChangeResponsePassed: true,
      targetResponseSamples,
      modeSwitchPassed,
      pauseClearPassed,
      restartClearPassed,
      targetMarkerVisible: productPointerState.targetMarkerVisible,
      consoleErrors: [...productDiagnostics.consoleErrors, ...debugDiagnostics.consoleErrors],
      pageErrors: [...productDiagnostics.pageErrors, ...debugDiagnostics.pageErrors],
      requestFailures: [...productDiagnostics.requestFailures, ...debugDiagnostics.requestFailures],
    };
    const smoke = {
      mode: mode.name,
      url,
      layout,
      actualBackend,
      crossOriginIsolated: environmentDiagnostics?.crossOriginIsolated ?? null,
      dualInstanceCount: environmentDiagnostics?.dualInstanceCount ?? null,
      pointerLockUsed: mouseEvidence.pointerLockUsed,
      systemCursorVisible: mouseEvidence.systemCursorVisible,
      stationarySampleCount: stationary.samples.length,
      stationaryDirectionalActionCount: stationaryMetrics.directionalActions,
      leftRightReversalCount: stationaryMetrics.leftRightReversalCount,
      stationaryNoop: stationary.finalAction === 'NOOP',
      oscillationObserved: stationaryMetrics.oscillationObserved,
      targetChangeResponsePassed: true,
      modeSwitchPassed,
      pauseClearPassed,
      restartClearPassed,
      screenshots: { production: repoRelative(productScreenshot) },
      diagnostics: { production: productDiagnostics, debug: debugDiagnostics },
    };
    return { mouseEvidence, smoke };
  } finally {
    await context.close().catch(() => undefined);
    await browser.close();
  }
}

const runs = [];
for (const mode of [
  { name: 'webgpu-preferred', args: [] },
  { name: 'wasm-fallback', args: ['--disable-features=WebGPU', '--disable-gpu'] },
]) {
  runs.push(await runScenario(mode));
}

const primary = runs[0];
const mouseEvidence = primary.mouseEvidence;
const productionSmoke = {
  schemaVersion: 1,
  artifactType: 'issue70_production_smoke',
  timestamp: new Date().toISOString(),
  url,
  customDomain: new URL(url).hostname === 'breakout.tommypan.dev',
  runs: runs.map(({ smoke }) => smoke),
  screenshots: {
    production: repoRelative('production-mouse-v3.png'),
    wasmFallback: repoRelative('production-mouse-v3-wasm-fallback.png'),
  },
};
await writeFile(outputPath('mouse-absolute-hysteresis-qa.json'), `${JSON.stringify(mouseEvidence, null, 2)}\n`, 'utf8');
await writeFile(outputPath('production-smoke.json'), `${JSON.stringify(productionSmoke, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ mouseEvidence, productionSmoke }, null, 2));
