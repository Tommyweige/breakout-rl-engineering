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
const url = String(args.get('url') ?? 'https://breakout.tommypan.dev/');
const outputDirectory = resolve(String(args.get('output-dir') ?? '../assets/human-interactive-runtime'));
const repoRoot = resolve(process.cwd(), '..');
const staticDurationMs = Number(args.get('static-duration-ms') ?? 3_000);
const timingDurationMs = Number(args.get('timing-duration-ms') ?? 30_000);
const targets = [0.20, 0.50, 0.80];
const targetSequence = [0.20, 0.80, 0.35, 0.65, 0.10];

await mkdir(outputDirectory, { recursive: true });

function outputPath(fileName) {
  return join(outputDirectory, fileName);
}

function repoRelative(fileName) {
  return relative(repoRoot, outputPath(fileName)).replaceAll('\\', '/');
}

function withDebug(value) {
  const parsed = new URL(value);
  parsed.searchParams.set('debug', '1');
  return parsed.toString();
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
  const failures = Object.fromEntries(Object.entries(diagnostics).filter(([, values]) => values.length > 0));
  if (Object.keys(failures).length > 0) throw new Error(`${scope} diagnostics failed: ${JSON.stringify(failures)}`);
}

async function waitForPlaying(page) {
  await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PLAYING', undefined, { timeout: 180_000 });
}

async function waitForReady(page) {
  await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'READY', undefined, { timeout: 30_000 });
}

async function moveTarget(page, normalizedX) {
  const canvas = page.locator('canvas[data-role="human-canvas"]');
  const box = await canvas.boundingBox();
  if (!box) throw new Error('human canvas has no bounding box');
  await canvas.hover({ position: { x: box.width * normalizedX, y: box.height * 0.70 } });
  await page.waitForFunction(
    (expected) => Math.abs((window.__humanInteractiveDiagnostics?.samples.at(-1)?.targetX ?? -1) - expected) < 0.02 || Math.abs((window.__mouseControlV3Diagnostics?.cursorTargetX ?? -1) - expected) < 0.02,
    normalizedX,
    { timeout: 10_000 },
  );
}

function samplesInWindow(diagnostics, startedAt) {
  return (diagnostics?.samples ?? []).filter((sample) => sample.timestampMs >= startedAt);
}

function directionFor(target, paddle) {
  if (target === null || paddle === null) return 'NOOP';
  const error = (target - paddle) * 160;
  if (error > 6) return 'RIGHT';
  if (error < -6) return 'LEFT';
  return 'NOOP';
}

function reversalCount(samples) {
  const directional = samples.filter((sample) => sample.executedAction === 'LEFT' || sample.executedAction === 'RIGHT');
  let reversals = 0;
  for (let index = 1; index < directional.length; index += 1) {
    if (directional[index]?.executedAction !== directional[index - 1]?.executedAction) reversals += 1;
  }
  return reversals;
}

function settledWindow(samples, settledAt = null) {
  if (samples.length === 0) return [];
  if (settledAt !== null) return samples.filter((sample) => sample.timestampMs >= settledAt);
  const end = samples[samples.length - 1]?.timestampMs ?? 0;
  return samples.filter((sample) => sample.timestampMs >= end - 1_000);
}

function trackingMetrics(samples, startedAt, settledAt = null) {
  const errors = samples
    .map((sample) => sample.positionError)
    .filter((value) => typeof value === 'number')
    .map((value) => Math.abs(value) * 160);
  const settled = settledWindow(samples, settledAt);
  const settledErrors = settled
    .map((sample) => sample.positionError)
    .filter((value) => typeof value === 'number')
    .map((value) => Math.abs(value) * 160);
  const settledPaddles = settled
    .map((sample) => sample.paddleCenterX)
    .filter((value) => typeof value === 'number')
    .map((value) => value * 160);
  const firstSettled = samples.findIndex((sample) => Math.abs(sample.positionError ?? Infinity) * 160 <= 6);
  const timeToSettleMs = settledAt !== null
    ? settledAt - startedAt
    : firstSettled < 0 ? null : (samples[firstSettled]?.timestampMs ?? startedAt) - startedAt;
  const range = settledPaddles.length ? Math.max(...settledPaddles) - Math.min(...settledPaddles) : null;
  const settledDurationMs = settled.length > 1 ? (settled.at(-1)?.timestampMs ?? 0) - (settled[0]?.timestampMs ?? 0) : 0;
  const reversals = reversalCount(settled);
  return {
    sampleCount: samples.length,
    timeToSettleMs,
    finalAbsoluteErrorRawPx: errors.at(-1) ?? null,
    meanSettledErrorRawPx: settledErrors.length ? settledErrors.reduce((sum, value) => sum + value, 0) / settledErrors.length : null,
    p95SettledErrorRawPx: percentile(settledErrors, 0.95),
    settledPaddleRangeRawPx: range,
    settledObservationDurationMs: settledDurationMs,
    actualDirectionReversalCountAfterSettled: reversals,
    oscillationFrequencyHz: settledDurationMs > 0 ? reversals / (settledDurationMs / 1_000) : 0,
  };
}

function percentile(values, fraction) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)] ?? null;
}

function weightedAverage(values, metricKey) {
  const weighted = values.filter((value) => typeof value?.elapsedMs === 'number' && value.elapsedMs > 0 && typeof value?.[metricKey] === 'number');
  const totalWeight = weighted.reduce((sum, value) => sum + value.elapsedMs, 0);
  return totalWeight ? weighted.reduce((sum, value) => sum + value.elapsedMs * value[metricKey], 0) / totalWeight : 0;
}

function aggregateTimingSegments(segments) {
  const elapsedMs = segments.reduce((sum, segment) => sum + (segment.elapsedMs ?? 0), 0);
  const humanRawFrameDelta = segments.reduce((sum, segment) => sum + (segment.humanRawFrameDelta ?? 0), 0);
  const agentDecisionCount = segments.reduce((sum, segment) => sum + (segment.agentDecisionCount ?? 0), 0);
  const agentRawFrameDelta = segments.reduce((sum, segment) => sum + (segment.agentRawFrameDelta ?? 0), 0);
  return {
    status: 'running',
    elapsedMs,
    humanRawFrameRepeat: segments[0]?.humanRawFrameRepeat,
    humanStickyActionProbability: segments[0]?.humanStickyActionProbability,
    humanSchedulerTargetFps: segments[0]?.humanSchedulerTargetFps,
    humanRawTickCount: segments.reduce((sum, segment) => sum + (segment.humanRawTickCount ?? 0), 0),
    humanRawFrameDelta,
    humanRawFps: elapsedMs ? (humanRawFrameDelta / elapsedMs) * 1_000 : 0,
    humanTickP50Ms: weightedAverage(segments, 'humanTickP50Ms'),
    humanTickP95Ms: Math.max(...segments.map((segment) => segment.humanTickP95Ms ?? 0), 0),
    lateHumanTicks: segments.reduce((sum, segment) => sum + (segment.lateHumanTicks ?? 0), 0),
    droppedHumanTicks: segments.reduce((sum, segment) => sum + (segment.droppedHumanTicks ?? 0), 0),
    catchUpBursts: segments.reduce((sum, segment) => sum + (segment.catchUpBursts ?? 0), 0),
    agentOuterActionRepeat: segments[0]?.agentOuterActionRepeat,
    agentStickyActionProbability: segments[0]?.agentStickyActionProbability,
    agentDecisionCount,
    agentRawFrameDelta,
    agentDecisionsPerSecond: elapsedMs ? (agentDecisionCount / elapsedMs) * 1_000 : 0,
    agentDecisionP50Ms: weightedAverage(segments, 'agentDecisionP50Ms'),
    agentDecisionP95Ms: Math.max(...segments.map((segment) => segment.agentDecisionP95Ms ?? 0), 0),
    agentInferenceInFlight: false,
    segments,
  };
}

async function startMouseEpisode(page) {
  await page.locator('button[data-action="reset"]').click();
  await waitForReady(page);
  await page.locator('button[data-action="start"]').click();
  await waitForPlaying(page);
  await page.locator('select[data-action="input-mode"]').selectOption('mouse');
}

async function waitForSettledTarget(page, target) {
  await page.waitForFunction((expectedTarget) => {
    const samples = (window.__humanInteractiveDiagnostics?.samples ?? [])
      .filter((sample) => Math.abs((sample.targetX ?? -1) - expectedTarget) < 0.01)
      .slice(-12);
    if (samples.length < 12) return false;
    const errors = samples.map((sample) => Math.abs(sample.positionError ?? Infinity) * 160);
    const paddles = samples.map((sample) => (sample.paddleCenterX ?? 0) * 160);
    return errors.every((error) => error <= 6) && Math.max(...paddles) - Math.min(...paddles) <= 6;
  }, target, { timeout: 10_000 });
  return page.evaluate(() => window.__humanInteractiveDiagnostics?.samples.at(-1)?.timestampMs ?? performance.now());
}

async function runTimingWindow(page, durationMs) {
  // Reset/restart is part of the wall-clock window, but its short setup time
  // is not active simulation time. Add a small guard so saved active segments
  // still cover the requested duration.
  const endAt = Date.now() + durationMs + 1_000;
  const segments = [];
  while (Date.now() < endAt) {
    await page.waitForTimeout(Math.min(250, Math.max(25, endAt - Date.now())));
    const state = await page.evaluate(() => ({
      human: document.querySelector('[data-role="human-stage-state"]')?.textContent,
      agent: document.querySelector('[data-role="agent-stage-state"]')?.textContent,
      runtime: window.__humanInteractiveDiagnostics?.runtime ?? null,
    }));
    if (state.human === 'GAME OVER' || state.agent === 'GAME OVER') {
      if (state.runtime) segments.push(state.runtime);
      await page.locator('button[data-action="reset"]').click();
      await waitForReady(page);
      await page.locator('button[data-action="start"]').click();
      await waitForPlaying(page);
      await page.locator('select[data-action="input-mode"]').selectOption('keyboard');
    }
  }
  const finalRuntime = await page.evaluate(() => window.__humanInteractiveDiagnostics?.runtime ?? null);
  if (finalRuntime) segments.push(finalRuntime);
  return aggregateTimingSegments(segments);
}

async function runStaticTracking(page) {
  const results = [];
  for (const target of targets) {
    for (let run = 1; run <= 2; run += 1) {
      await startMouseEpisode(page);
      await moveTarget(page, target);
      const before = await page.evaluate(() => window.__humanInteractiveDiagnostics?.samples.at(-1)?.timestampMs ?? performance.now());
      const settledAt = await waitForSettledTarget(page, target);
      await page.waitForTimeout(staticDurationMs);
      const diagnostics = await page.evaluate(() => window.__humanInteractiveDiagnostics ?? null);
      const samples = samplesInWindow(diagnostics, before);
      results.push({ target, run, settledAt, ...trackingMetrics(samples, before, settledAt), samples });
    }
  }
  return {
    schemaVersion: 1,
    artifactType: 'issue71_mouse_tracking_static',
    timestamp: new Date().toISOString(),
    url,
    targets,
    runsPerTarget: 2,
    deadzoneRawPx: 6,
    results,
    aggregate: {
      meanSettledErrorRawPx: mean(results.map((result) => result.meanSettledErrorRawPx)),
      p95SettledErrorRawPx: percentile(results.map((result) => result.p95SettledErrorRawPx).filter((value) => value !== null), 0.95),
      maxSettledPaddleRangeRawPx: Math.max(...results.map((result) => result.settledPaddleRangeRawPx ?? 0)),
      maxDirectionReversals: Math.max(...results.map((result) => result.actualDirectionReversalCountAfterSettled)),
    },
    measurementNotes: 'The ALE paddle moves in quantized raw-pixel increments. A 3px dead zone produced repeated corrections; the 6px dead zone is the smallest stable setting observed here, with no 10px+ settled drift.',
  };
}

async function runDynamicTracking(page) {
  await startMouseEpisode(page);
  const transitions = [];
  for (const target of targetSequence) {
    const before = await page.evaluate(() => window.__humanInteractiveDiagnostics?.samples.at(-1) ?? null);
    await moveTarget(page, target);
    const changedAt = await page.evaluate(() => performance.now());
    const expectedDirection = directionFor(target, before?.paddleCenterX ?? null);
    await page.waitForTimeout(750);
    const diagnostics = await page.evaluate(() => window.__humanInteractiveDiagnostics ?? null);
    const samples = diagnostics?.samples ?? [];
    const after = samples.filter((sample) => (!before || sample.decision > before.decision) && sample.timestampMs >= changedAt);
    const firstCorrect = after.find((sample) => sample.executedAction === expectedDirection && expectedDirection !== 'NOOP');
    const firstMotion = after.find((sample) => before?.paddleCenterX !== null && sample.paddleCenterX !== null && Math.abs(sample.paddleCenterX - (before?.paddleCenterX ?? sample.paddleCenterX)) * 160 >= 1);
    const settle = trackingMetrics(after, changedAt);
    transitions.push({
      target,
      targetChangeTimestamp: changedAt,
      expectedDirection,
      firstCorrectDirectionTimestamp: firstCorrect?.timestampMs ?? null,
      firstPaddleMotionTimestamp: firstMotion?.timestampMs ?? null,
      inputToCorrectActionMs: firstCorrect ? firstCorrect.timestampMs - changedAt : null,
      inputToObservedPaddleMotionMs: firstMotion ? firstMotion.timestampMs - changedAt : null,
      timeToSettleMs: settle.timeToSettleMs,
    });
  }
  return {
    schemaVersion: 1,
    artifactType: 'issue71_mouse_tracking_dynamic',
    timestamp: new Date().toISOString(),
    url,
    targetSequence,
    transitions,
    summary: {
      inputToCorrectActionP50Ms: percentile(transitions.map((item) => item.inputToCorrectActionMs).filter((value) => value !== null), 0.5),
      inputToCorrectActionP95Ms: percentile(transitions.map((item) => item.inputToCorrectActionMs).filter((value) => value !== null), 0.95),
      inputToObservedPaddleMotionP50Ms: percentile(transitions.map((item) => item.inputToObservedPaddleMotionMs).filter((value) => value !== null), 0.5),
      inputToObservedPaddleMotionP95Ms: percentile(transitions.map((item) => item.inputToObservedPaddleMotionMs).filter((value) => value !== null), 0.95),
      timeToSettleP95Ms: percentile(transitions.map((item) => item.timeToSettleMs).filter((value) => value !== null), 0.95),
    },
  };
}

async function captureResponsiveLayouts() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const viewports = [
    { width: 1920, height: 1080, desktop: true },
    { width: 1536, height: 864, desktop: true },
    { width: 1366, height: 768, desktop: true },
    { width: 1280, height: 720, desktop: true },
    { width: 768, height: 900, desktop: false },
    { width: 390, height: 844, desktop: false },
  ];
  const results = [];
  try {
    for (const viewport of viewports) {
      const context = await browser.newContext({ viewport, deviceScaleFactor: 1 });
      const page = await context.newPage();
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
      await page.locator('canvas[data-role="human-canvas"]').waitFor({ state: 'attached', timeout: 30_000 });
      const layout = await page.evaluate(() => ({
        documentWidth: document.documentElement.scrollWidth,
        documentHeight: document.documentElement.scrollHeight,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
        canvasCount: document.querySelectorAll('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').length,
        technicalDetailsPresent: Boolean(document.querySelector('[data-role="technical-details"]')),
      }));
      if (layout.documentWidth > layout.viewportWidth + 1) throw new Error(`responsive horizontal overflow at ${viewport.width}px: ${JSON.stringify(layout)}`);
      if (viewport.desktop && layout.documentHeight > layout.viewportHeight + 1) throw new Error(`desktop vertical overflow at ${viewport.width}px: ${JSON.stringify(layout)}`);
      if (layout.canvasCount !== 2 || layout.technicalDetailsPresent) throw new Error(`responsive production surface failed at ${viewport.width}px: ${JSON.stringify(layout)}`);
      results.push({ viewport, layout });
      await context.close();
    }
  } finally {
    await browser.close();
  }
  return results;
}

async function captureScenario(mode, detailed = false) {
  const browser = await chromium.launch({ channel: 'chrome', headless: true, args: mode.args });
  const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const diagnostics = attachDiagnostics(page);
  const debugPage = await context.newPage();
  const debugDiagnostics = attachDiagnostics(debugPage);
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    const layout = await page.evaluate(() => ({
      viewport: { width: innerWidth, height: innerHeight },
      document: { width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight },
      canvasCount: document.querySelectorAll('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').length,
      technicalDetailsPresent: Boolean(document.querySelector('[data-role="technical-details"]')),
    }));
    if (layout.document.height > layout.viewport.height + 1 || layout.document.width > layout.viewport.width + 1) throw new Error(`production no-scroll failed: ${JSON.stringify(layout)}`);
    if (layout.canvasCount !== 2 || layout.technicalDetailsPresent) throw new Error(`production UI failed: ${JSON.stringify(layout)}`);
    await page.locator('button[data-action="start"]').click();
    await waitForPlaying(page);
    await page.locator('button[data-action="pause"]').click();
    await page.locator('button[data-action="start"]').click();
    await waitForPlaying(page);
    await page.locator('button[data-action="reset"]').click();
    await waitForReady(page);
    await page.screenshot({ path: outputPath(mode.name === 'webgpu-preferred' ? 'production-human-runtime.png' : 'production-human-runtime-wasm-fallback.png'), fullPage: true });

    await debugPage.goto(withDebug(url), { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await debugPage.locator('button[data-action="start"]').click();
    await waitForPlaying(debugPage);
    await debugPage.locator('select[data-action="input-mode"]').selectOption('mouse');
    await moveTarget(debugPage, 0.5);
    const staticEvidence = detailed ? await runStaticTracking(debugPage) : null;
    const dynamicEvidence = detailed ? await runDynamicTracking(debugPage) : null;

    await debugPage.locator('button[data-action="reset"]').click();
    await waitForReady(debugPage);
    await debugPage.locator('button[data-action="start"]').click();
    await waitForPlaying(debugPage);
    await debugPage.locator('select[data-action="input-mode"]').selectOption('keyboard');
    await debugPage.keyboard.down('ArrowRight');
    await debugPage.waitForTimeout(120);
    await debugPage.keyboard.up('ArrowRight');
    await debugPage.waitForTimeout(50);
    await debugPage.keyboard.down('ArrowLeft');
    await debugPage.waitForTimeout(120);
    await debugPage.keyboard.up('ArrowLeft');
    const keyboardTrace = await debugPage.evaluate(() => (window.__humanInteractiveDiagnostics?.samples ?? []).slice(-20));

    const timing = await runTimingWindow(debugPage, timingDurationMs);
    const humanDiagnostics = await debugPage.evaluate(() => window.__humanInteractiveDiagnostics ?? null);
    const environmentDiagnostics = await debugPage.evaluate(() => window.__day30EnvironmentDiagnostics ?? null);
    if (environmentDiagnostics?.dualInstanceCount !== 2) throw new Error(`expected two independent ALE instances, got ${JSON.stringify(environmentDiagnostics)}`);
    await debugPage.locator('details[data-role="technical-details"]').evaluate((details) => { details.open = true; });
    await debugPage.screenshot({ path: outputPath(mode.name === 'webgpu-preferred' ? 'debug-human-runtime.png' : 'debug-human-runtime-wasm-fallback.png'), fullPage: true });
    const actualBackend = await debugPage.locator('[data-role="gameplay-backend"]').textContent();
    const difficulty = await debugPage.locator('[data-role="ai-difficulty-label"]').textContent();
    if (difficulty !== 'HARD') throw new Error(`unexpected production difficulty: ${difficulty}`);
    if (mode.name === 'wasm-fallback' && actualBackend !== 'WASM') throw new Error(`WASM fallback expected WASM, got ${actualBackend}`);
    assertDiagnostics(diagnostics, `${mode.name} production`);
    assertDiagnostics(debugDiagnostics, `${mode.name} debug`);
    return { mode: mode.name, layout, actualBackend, difficulty, humanDiagnostics, environmentDiagnostics, keyboardTrace, staticEvidence, dynamicEvidence, timing, diagnostics: { production: diagnostics, debug: debugDiagnostics } };
  } finally {
    await context.close().catch(() => undefined);
    await browser.close();
  }
}

function mean(values) {
  const filtered = values.filter((value) => typeof value === 'number');
  return filtered.length ? filtered.reduce((sum, value) => sum + value, 0) / filtered.length : null;
}

function required(value, label) {
  if (value === null || value === undefined) throw new Error(`missing required evidence field: ${label}`);
  return value;
}

const primary = await captureScenario({ name: 'webgpu-preferred', args: [] }, true);
const fallback = await captureScenario({ name: 'wasm-fallback', args: ['--disable-features=WebGPU', '--disable-gpu'] }, false);
const responsive = await captureResponsiveLayouts();
const runtime = required(primary.timing ?? primary.humanDiagnostics?.runtime, 'human timing runtime');
const contract = required(primary.humanDiagnostics?.runtimeContract ?? primary.environmentDiagnostics?.humanRuntime, 'human runtime contract');
const agentRuntime = required(primary.environmentDiagnostics?.agentRuntime, 'agent runtime diagnostics');
if (primary.environmentDiagnostics?.dualInstanceCount !== 2) throw new Error('production smoke did not observe two independent ALE instances');

await writeFile(outputPath('human-runtime-contract.json'), `${JSON.stringify({
  schemaVersion: 1,
  artifactType: 'issue71_human_runtime_contract',
  timestamp: new Date().toISOString(),
  url,
  human: {
    rawFrameRepeat: required(contract.rawFrameRepeat, 'human.rawFrameRepeat'),
    stickyActionProbability: required(contract.stickyActionProbability, 'human.stickyActionProbability'),
    targetRawFps: required(contract.targetRawFps, 'human.targetRawFps'),
    usesModelPreprocessing: required(contract.usesModelPreprocessing, 'human.usesModelPreprocessing'),
  },
  agent: {
    outerActionRepeat: required(agentRuntime.outerActionRepeat, 'agent.outerActionRepeat'),
    stickyActionProbability: required(agentRuntime.stickyActionProbability, 'agent.stickyActionProbability'),
    rawAleFrameSkip: required(agentRuntime.rawAleFrameSkip, 'agent.rawAleFrameSkip'),
    grayscaleBeforeMaxPool: required(agentRuntime.grayscaleBeforeMaxPool, 'agent.grayscaleBeforeMaxPool'),
  },
  humanAndAgentIndependent: (primary.environmentDiagnostics?.dualInstanceCount ?? 0) === 2,
  runtimeDiagnostics: { human: contract, agent: agentRuntime },
}, null, 2)}\n`, 'utf8');

await writeFile(outputPath('human-timing-30s.json'), `${JSON.stringify({
  schemaVersion: 1,
  artifactType: 'issue71_human_timing_30s',
  timestamp: new Date().toISOString(),
  url,
  elapsedMs: runtime.elapsedMs,
  humanRawTickCount: runtime.humanRawTickCount,
  humanRawFrameDelta: runtime.humanRawFrameDelta,
  humanRawFps: runtime.humanRawFps,
  humanSchedulerTargetFps: runtime.humanSchedulerTargetFps,
  humanTickP50Ms: runtime.humanTickP50Ms,
  humanTickP95Ms: runtime.humanTickP95Ms,
  lateHumanTicks: runtime.lateHumanTicks,
  catchUpBursts: runtime.catchUpBursts,
  agentDecisionCount: runtime.agentDecisionCount,
  agentRawFrameDelta: runtime.agentRawFrameDelta,
  agentDecisionsPerSecond: runtime.agentDecisionsPerSecond,
  agentDecisionP50Ms: runtime.agentDecisionP50Ms,
  agentDecisionP95Ms: runtime.agentDecisionP95Ms,
  segments: runtime.segments,
}, null, 2)}\n`, 'utf8');

await writeFile(outputPath('mouse-tracking-static.json'), `${JSON.stringify(primary.staticEvidence, null, 2)}\n`, 'utf8');
await writeFile(outputPath('mouse-tracking-dynamic.json'), `${JSON.stringify(primary.dynamicEvidence, null, 2)}\n`, 'utf8');
await writeFile(outputPath('agent-regression.json'), `${JSON.stringify({
  schemaVersion: 1,
  artifactType: 'issue71_agent_regression',
  timestamp: new Date().toISOString(),
  url,
  actualBackend: primary.actualBackend,
  fallbackBackend: fallback.actualBackend,
  agentRuntime,
  humanRuntime: contract,
  dualInstanceCount: primary.environmentDiagnostics?.dualInstanceCount,
}, null, 2)}\n`, 'utf8');

await writeFile(outputPath('production-smoke.json'), `${JSON.stringify({
  schemaVersion: 1,
  artifactType: 'issue71_production_smoke',
  timestamp: new Date().toISOString(),
  url,
  customDomain: new URL(url).hostname === 'breakout.tommypan.dev',
  runs: [primary, fallback].map((run) => ({
    mode: run.mode,
    actualBackend: run.actualBackend,
    difficulty: run.difficulty,
    dualInstanceCount: run.environmentDiagnostics?.dualInstanceCount ?? null,
    humanRuntime: run.humanDiagnostics?.runtimeContract ?? null,
    consoleErrors: [...run.diagnostics.production.consoleErrors, ...run.diagnostics.debug.consoleErrors],
    pageErrors: [...run.diagnostics.production.pageErrors, ...run.diagnostics.debug.pageErrors],
    requestFailures: [...run.diagnostics.production.requestFailures, ...run.diagnostics.debug.requestFailures],
    badResponses: [...run.diagnostics.production.badResponses, ...run.diagnostics.debug.badResponses],
  })),
  screenshots: {
    production: repoRelative('production-human-runtime.png'),
    debug: repoRelative('debug-human-runtime.png'),
    wasmFallback: repoRelative('production-human-runtime-wasm-fallback.png'),
  },
  responsive,
}, null, 2)}\n`, 'utf8');

console.log(JSON.stringify({
  url,
  runtime,
  static: primary.staticEvidence?.aggregate,
  dynamic: primary.dynamicEvidence?.summary,
  primaryBackend: primary.actualBackend,
  fallbackBackend: fallback.actualBackend,
  responsive,
}, null, 2));
