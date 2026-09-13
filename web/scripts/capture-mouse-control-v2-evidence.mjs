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
const outputDirectory = resolve(String(args.get('output-dir') ?? '../assets/mouse-control-v2'));
const repoRoot = resolve(process.cwd(), '..');
const debugUrl = withDebug(url);
const stationaryDurationTargetMs = 3_000;

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

async function gotoApp(page, targetUrl = url) {
  await page.goto(targetUrl, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.locator('button[data-action="start"]').waitFor({ state: 'visible', timeout: 30_000 });
}

async function waitForPlaying(page) {
  await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PLAYING', undefined, { timeout: 180_000 });
}

async function acquirePointerLock(page) {
  const canvas = page.locator('canvas[data-role="human-canvas"]');
  const box = await canvas.boundingBox();
  if (!box) throw new Error('human canvas has no bounding box');
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    await canvas.click({ position: { x: box.width * 0.5, y: box.height * 0.72 } });
    const acquired = await page.waitForFunction(
      () => document.pointerLockElement === document.querySelector('canvas[data-role="human-canvas"]'),
      undefined,
      { timeout: 3_000 },
    ).then(() => true).catch(() => false);
    if (acquired) return canvas;
    if (attempt < 3) await page.waitForTimeout(600);
  }
  throw new Error('Pointer Lock was not acquired after three user-gesture clicks');
}

async function releasePointerLockWithEscape(page) {
  const session = await page.context().newCDPSession(page);
  try {
    await session.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27 });
    await session.send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27 });
  } finally {
    await session.detach();
  }
}

async function diagnosticsSnapshot(page) {
  return page.evaluate(() => window.__mouseControlV2Diagnostics ?? null);
}

async function waitForActionAfter(page, action, afterDecision) {
  await page.waitForFunction(
    ({ action: targetAction, afterDecision: decision }) => window.__mouseControlV2Diagnostics?.actionHistory.some((sample) => sample.decision > decision && sample.action === targetAction),
    { action, afterDecision },
    { timeout: 10_000 },
  );
  return page.evaluate(
    ({ action: targetAction, afterDecision: decision }) => window.__mouseControlV2Diagnostics?.actionHistory.find((sample) => sample.decision > decision && sample.action === targetAction) ?? null,
    { action, afterDecision },
  );
}

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });
const productPage = await context.newPage();
const productDiagnostics = attachDiagnostics(productPage);
let debugPage;
let debugDiagnostics;

try {
  await gotoApp(productPage);
  const layout = await productPage.evaluate(() => ({
    viewportHeight: innerHeight,
    documentHeight: document.documentElement.scrollHeight,
    viewportWidth: document.documentElement.clientWidth,
    documentWidth: document.documentElement.scrollWidth,
    canvasCount: document.querySelectorAll('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').length,
    controlCount: document.querySelectorAll('button[data-action="start"], button[data-action="pause"], button[data-action="reset"]').length,
    technicalDetailsPresent: Boolean(document.querySelector('[data-role="technical-details"]')),
  }));
  if (layout.documentHeight > layout.viewportHeight + 1 || layout.documentWidth > layout.viewportWidth + 1) throw new Error(`production layout regressed: ${JSON.stringify(layout)}`);
  if (layout.canvasCount !== 2 || layout.controlCount !== 3 || layout.technicalDetailsPresent) throw new Error(`production surface is incomplete: ${JSON.stringify(layout)}`);

  await productPage.locator('button[data-action="start"]').click();
  await waitForPlaying(productPage);
  await productPage.locator('select[data-action="input-mode"]').selectOption('mouse');
  await acquirePointerLock(productPage);
  await productPage.waitForFunction(() => document.querySelector('[data-role="input-hint"]')?.textContent?.includes('Mouse captured'), undefined, { timeout: 10_000 });
  await productPage.waitForTimeout(500);
  await productPage.screenshot({ path: outputPath('production-mouse-relative.png'), fullPage: true });
  const productionPointerLockActive = await productPage.evaluate(() => document.pointerLockElement === document.querySelector('canvas[data-role="human-canvas"]'));
  await releasePointerLockWithEscape(productPage);
  await productPage.waitForFunction(() => document.pointerLockElement === null, undefined, { timeout: 10_000 });
  await productPage.close();

  debugPage = await context.newPage();
  debugDiagnostics = attachDiagnostics(debugPage);
  await gotoApp(debugPage, debugUrl);
  await debugPage.locator('button[data-action="start"]').click();
  await waitForPlaying(debugPage);
  const initialDifficulty = await debugPage.locator('select[data-action="difficulty"]').inputValue();
  await debugPage.locator('select[data-action="input-mode"]').selectOption('mouse');
  await acquirePointerLock(debugPage);
  await debugPage.waitForFunction(() => window.__mouseControlV2Diagnostics?.pointerLockActive === true, undefined, { timeout: 10_000 });
  const lockedSnapshot = await diagnosticsSnapshot(debugPage);
  if (!lockedSnapshot?.pointerLockSupported || !lockedSnapshot.pointerLockActive) throw new Error(`Pointer Lock was not active: ${JSON.stringify(lockedSnapshot)}`);

  let afterDecision = lockedSnapshot.decisionCount;
  await debugPage.mouse.move(1_050, 400);
  const rightSample = await waitForActionAfter(debugPage, 'RIGHT', afterDecision);
  if (!rightSample) throw new Error('RIGHT movement did not reach the Human action loop');
  const firstNoop = await waitForActionAfter(debugPage, 'NOOP', rightSample.decision);
  if (!firstNoop) throw new Error('stationary mouse did not return to NOOP');
  const noopTransitionMs = firstNoop.timestampMs - rightSample.timestampMs;

  afterDecision = firstNoop.decision;
  await debugPage.mouse.move(250, 400);
  const leftSample = await waitForActionAfter(debugPage, 'LEFT', afterDecision);
  if (!leftSample) throw new Error('LEFT movement did not reach the Human action loop');

  afterDecision = leftSample.decision;
  await debugPage.mouse.move(1_100, 400);
  const rapidRight = await waitForActionAfter(debugPage, 'RIGHT', afterDecision);
  if (!rapidRight) throw new Error('rapid reversal setup did not produce RIGHT');
  await debugPage.mouse.move(200, 400);
  const rapidLeft = await waitForActionAfter(debugPage, 'LEFT', rapidRight.decision);
  const rapidReversalPassed = Boolean(rapidLeft && rapidLeft.decision > rapidRight.decision);
  if (!rapidReversalPassed) throw new Error('rapid RIGHT to LEFT reversal failed');

  const stationaryStart = Date.now();
  const stationaryAfterDecision = rapidLeft.decision;
  await debugPage.waitForTimeout(stationaryDurationTargetMs + 150);
  const stationaryResult = await debugPage.evaluate(
    ({ afterDecision: decision, startedAt }) => {
      const diagnostics = window.__mouseControlV2Diagnostics;
      const samples = diagnostics?.actionHistory.filter((sample) => sample.decision > decision) ?? [];
      return {
        durationMs: Date.now() - startedAt,
        samples,
        latestMovementX: diagnostics?.latestMovementX ?? null,
        finalAction: diagnostics?.lastHumanAction ?? null,
      };
    },
    { afterDecision: stationaryAfterDecision, startedAt: stationaryStart },
  );
  const directionalStationaryActions = stationaryResult.samples.filter((sample) => sample.action === 'LEFT' || sample.action === 'RIGHT');
  const oscillationObserved = directionalStationaryActions.length > 0;
  if (stationaryResult.durationMs < stationaryDurationTargetMs || stationaryResult.finalAction !== 'NOOP' || oscillationObserved) {
    throw new Error(`stationary mouse QA failed: ${JSON.stringify(stationaryResult)}`);
  }

  await releasePointerLockWithEscape(debugPage);
  await debugPage.waitForFunction(() => window.__mouseControlV2Diagnostics?.pointerLockActive === false && window.__mouseControlV2Diagnostics.pointerLockStatus === 'inactive', undefined, { timeout: 10_000 });
  const escapeSnapshot = await diagnosticsSnapshot(debugPage);
  const escapeReleasePassed = escapeSnapshot?.pointerLockActive === false && escapeSnapshot.pointerLockStatus === 'inactive';
  if (!escapeReleasePassed) throw new Error(`Escape did not release Pointer Lock: ${JSON.stringify(escapeSnapshot)}`);

  await acquirePointerLock(debugPage);
  await debugPage.evaluate(() => {
    document.dispatchEvent(new PointerEvent('pointermove', { movementX: 8, bubbles: true }));
    document.querySelector('button[data-action="pause"]')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
  await debugPage.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PAUSED' && window.__mouseControlV2Diagnostics?.pointerLockActive === false, undefined, { timeout: 10_000 });
  const pauseSnapshot = await diagnosticsSnapshot(debugPage);
  const pauseClearPassed = pauseSnapshot?.mouseIntent === 'NOOP' && pauseSnapshot.lastHumanAction === 'NOOP';

  await debugPage.locator('button[data-action="start"]').click();
  await waitForPlaying(debugPage);
  await acquirePointerLock(debugPage);
  await debugPage.evaluate(() => {
    document.dispatchEvent(new PointerEvent('pointermove', { movementX: -8, bubbles: true }));
    document.querySelector('button[data-action="reset"]')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
  await debugPage.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'READY' && window.__mouseControlV2Diagnostics?.pointerLockActive === false, undefined, { timeout: 30_000 });
  const restartSnapshot = await diagnosticsSnapshot(debugPage);
  const restartClearPassed = restartSnapshot?.mouseIntent === 'NOOP' && restartSnapshot.lastHumanAction === 'NOOP';

  await debugPage.locator('button[data-action="start"]').click();
  await waitForPlaying(debugPage);
  await acquirePointerLock(debugPage);
  await debugPage.evaluate(() => {
    document.dispatchEvent(new PointerEvent('pointermove', { movementX: 8, bubbles: true }));
    const selector = document.querySelector('select[data-action="input-mode"]');
    if (selector instanceof HTMLSelectElement) {
      selector.value = 'keyboard';
      selector.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });
  await debugPage.waitForFunction(() => window.__mouseControlV2Diagnostics?.pointerLockActive === false && document.querySelector('select[data-action="input-mode"]')?.value === 'keyboard', undefined, { timeout: 10_000 });
  const modeSwitchSnapshot = await diagnosticsSnapshot(debugPage);
  const modeSwitchPassed = modeSwitchSnapshot?.mouseIntent === 'NOOP' && modeSwitchSnapshot.lastHumanAction === 'NOOP';
  const finalDifficulty = await debugPage.locator('select[data-action="difficulty"]').inputValue();
  const actualBackend = await debugPage.locator('[data-role="gameplay-backend"]').textContent();
  const environmentDiagnostics = await debugPage.evaluate(() => window.__day30EnvironmentDiagnostics ?? null);

  assertDiagnostics(productDiagnostics, 'production page');
  assertDiagnostics(debugDiagnostics, 'debug page');
  const mouseEvidence = {
    schemaVersion: 1,
    artifactType: 'issue69_mouse_relative_qa',
    timestamp: new Date().toISOString(),
    url,
    pointerLockSupported: lockedSnapshot.pointerLockSupported,
    pointerLockActive: productionPointerLockActive,
    pointerLockReleasedByEscape: escapeReleasePassed,
    threshold: lockedSnapshot.threshold,
    stationaryDurationMs: stationaryResult.durationMs,
    stationaryDecisionCount: stationaryResult.samples.length,
    stationaryMovementX: stationaryResult.latestMovementX,
    stationaryAction: stationaryResult.finalAction,
    noopTransitionMs,
    oscillationObserved,
    rapidReversalPassed,
    modeSwitchPassed,
    pauseClearPassed,
    restartClearPassed,
    rightSample,
    leftSample,
    rapidRight,
    rapidLeft,
    consoleErrors: [...productDiagnostics.consoleErrors, ...debugDiagnostics.consoleErrors],
    pageErrors: [...productDiagnostics.pageErrors, ...debugDiagnostics.pageErrors],
    requestFailures: [...productDiagnostics.requestFailures, ...debugDiagnostics.requestFailures],
    badResponses: [...productDiagnostics.badResponses, ...debugDiagnostics.badResponses],
  };
  const productionSmoke = {
    schemaVersion: 1,
    artifactType: 'issue69_production_smoke',
    timestamp: new Date().toISOString(),
    url,
    debugUrl,
    customDomain: new URL(url).hostname === 'breakout.tommypan.dev',
    layout,
    controls: {
      pointerLockSupported: mouseEvidence.pointerLockSupported,
      pointerLockActiveObserved: mouseEvidence.pointerLockActive,
      stationaryNoop: mouseEvidence.stationaryAction === 'NOOP',
      oscillationObserved,
      rapidReversalPassed,
      escapeReleasePassed,
      modeSwitchPassed,
      pauseClearPassed,
      restartClearPassed,
    },
    preservedRuntime: {
      difficultyBefore: initialDifficulty,
      difficultyAfter: finalDifficulty,
      actualBackend,
      dualInstanceCount: environmentDiagnostics?.dualInstanceCount ?? null,
      crossOriginIsolated: environmentDiagnostics?.crossOriginIsolated ?? null,
    },
    screenshot: repoRelative('production-mouse-relative.png'),
    diagnostics: { production: productDiagnostics, debug: debugDiagnostics },
  };
  await writeFile(outputPath('mouse-relative-qa.json'), `${JSON.stringify(mouseEvidence, null, 2)}\n`, 'utf8');
  await writeFile(outputPath('production-smoke.json'), `${JSON.stringify(productionSmoke, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({ mouseEvidence, productionSmoke }, null, 2));
} finally {
  await context.close().catch(() => undefined);
  await browser.close();
}
