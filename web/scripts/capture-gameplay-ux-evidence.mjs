import { mkdir, writeFile } from 'node:fs/promises';
import { join, relative, resolve } from 'node:path';
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
    } else {
      values.set(key, true);
    }
  }
  return values;
}

const args = parseArgs(process.argv.slice(2));
const url = String(args.get('url') ?? 'http://127.0.0.1:5173/');
const outputDirectory = resolve(String(args.get('output-dir') ?? '../assets/gameplay-ux'));
const repoRoot = resolve(process.cwd(), '..');
const viewports = [
  { width: 1920, height: 1080 },
  { width: 1536, height: 864 },
  { width: 1366, height: 768 },
  { width: 1280, height: 720 },
  { width: 390, height: 900 },
];
const rates = { easy: 0.30, medium: 0.15, hard: 0.05, unbeatable: 0 };
const fixedDifficultyRngSeed = 6801;
const difficultySampleCount = 10_000;

await mkdir(outputDirectory, { recursive: true });

function artifactPath(fileName) {
  return join(outputDirectory, fileName);
}

function repoRelative(fileName) {
  return relative(repoRoot, artifactPath(fileName)).replaceAll('\\', '/');
}

function withDebug(value) {
  const parsed = new URL(value);
  parsed.searchParams.set('debug', '1');
  return parsed.toString();
}

function createSeededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function runDeterministicDifficultyQa() {
  const countMistakes = (difficulty) => {
    const random = createSeededRandom(fixedDifficultyRngSeed);
    let mistakes = 0;
    for (let index = 0; index < difficultySampleCount; index += 1) {
      if (random() < rates[difficulty]) {
        mistakes += 1;
        random();
      }
    }
    return mistakes;
  };
  const counts = Object.keys(rates).map((difficulty) => countMistakes(difficulty));
  const repeatCounts = Object.keys(rates).map((difficulty) => countMistakes(difficulty));
  return {
    rng: 'independent seeded LCG for QA only',
    seed: fixedDifficultyRngSeed,
    sampleCount: difficultySampleCount,
    injectedMistakeCounts: Object.fromEntries(Object.keys(rates).map((difficulty, index) => [difficulty, counts[index]])),
    reproducible: JSON.stringify(counts) === JSON.stringify(repeatCounts),
    expectedOrdering: counts[0] > counts[1] && counts[1] > counts[2] && counts[2] > counts[3],
  };
}

function attachDiagnostics(page) {
  const diagnostics = {
    consoleErrors: [],
    pageErrors: [],
    requestFailures: [],
    badResponses: [],
  };
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
  const failed = Object.entries(diagnostics).filter(([, values]) => values.length > 0);
  if (failed.length > 0) throw new Error(`${scope} browser diagnostics were not clean: ${JSON.stringify(Object.fromEntries(failed))}`);
}

async function waitForApp(page) {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.locator('button[data-action="start"]').waitFor({ state: 'visible', timeout: 30_000 });
}

async function waitForPlaying(page) {
  await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PLAYING', undefined, { timeout: 180_000 });
}

async function readProductionLayout(page) {
  return page.evaluate(() => {
    const rect = (selector) => {
      const element = document.querySelector(selector);
      if (!element) return null;
      const box = element.getBoundingClientRect();
      return { width: Math.round(box.width), height: Math.round(box.height) };
    };
    const technicalDetails = document.querySelector('[data-role="technical-details"]');
    const bodyText = document.body.innerText;
    return {
      viewport: { width: innerWidth, height: innerHeight },
      documentWidth: document.documentElement.scrollWidth,
      documentHeight: document.documentElement.scrollHeight,
      viewportWidth: document.documentElement.clientWidth,
      viewportHeight: innerHeight,
      canvases: [
        rect('canvas[data-role="human-canvas"]'),
        rect('canvas[data-role="agent-canvas"]'),
      ],
      gamePanels: [...document.querySelectorAll('.game-card')].map((panel) => {
        const box = panel.getBoundingClientRect();
        return { left: Math.round(box.left), top: Math.round(box.top), width: Math.round(box.width), height: Math.round(box.height) };
      }),
      controls: [...document.querySelectorAll('.game-controls button')].map((button) => {
        const box = button.getBoundingClientRect();
        return { action: button.getAttribute('data-action'), width: Math.round(box.width), height: Math.round(box.height) };
      }),
      inputModeCount: document.querySelectorAll('select[data-action="input-mode"]').length,
      difficultyCount: document.querySelectorAll('select[data-action="difficulty"]').length,
      difficultyOptions: [...document.querySelectorAll('select[data-action="difficulty"] option')].map((option) => option.value),
      scoreFieldCount: document.querySelectorAll('[data-role="human-score"], [data-role="human-lives"], [data-role="agent-score"], [data-role="agent-lives"]').length,
      technicalDetailsPresent: Boolean(technicalDetails),
      visibleEngineeringText: /ONNX|WASM|WebGPU|Q-values|backend|parity|Contract|SHA|latency|policy/i.test(bodyText),
      technicalDetailsOpen: technicalDetails?.hasAttribute('open') ?? false,
    };
  });
}

function assertProductionLayout(layout) {
  if (layout.documentWidth > layout.viewportWidth + 1) throw new Error(`horizontal overflow: ${JSON.stringify(layout)}`);
  if (layout.viewport.width >= 1280 && layout.documentHeight > layout.viewportHeight + 1) throw new Error(`desktop vertical scroll: ${JSON.stringify(layout)}`);
  if (layout.canvases.some((canvas) => !canvas)) throw new Error(`both canvases are required: ${JSON.stringify(layout)}`);
  if (layout.viewport.width >= 1280 && layout.canvases.some(({ height }) => height < 300)) throw new Error(`desktop canvas is too small: ${JSON.stringify(layout)}`);
  if (layout.controls.some(({ height }) => height < 44)) throw new Error(`control target is too small: ${JSON.stringify(layout)}`);
  if (layout.inputModeCount !== 1 || layout.difficultyCount !== 1 || layout.scoreFieldCount !== 4) throw new Error(`gameplay controls are incomplete: ${JSON.stringify(layout)}`);
  if (layout.difficultyOptions.join(',') !== 'easy,medium,hard,unbeatable') throw new Error(`difficulty options changed: ${JSON.stringify(layout.difficultyOptions)}`);
  if (layout.technicalDetailsPresent || layout.technicalDetailsOpen || layout.visibleEngineeringText) throw new Error(`production route exposes engineering UI: ${JSON.stringify(layout)}`);
}

async function captureLayouts() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const results = [];
  try {
    for (const viewport of viewports) {
      const context = await browser.newContext({ viewport, deviceScaleFactor: 1 });
      const page = await context.newPage();
      const diagnostics = attachDiagnostics(page);
      await waitForApp(page);
      const layout = await readProductionLayout(page);
      assertProductionLayout(layout);
      const fileName = `production-${viewport.width}x${viewport.height}.png`;
      await page.evaluate(() => document.activeElement?.blur());
      await page.screenshot({ path: artifactPath(fileName), fullPage: true });
      assertDiagnostics(diagnostics, `${viewport.width}x${viewport.height}`);
      results.push({ ...viewport, screenshot: repoRelative(fileName), layout, diagnostics });
      await context.close();
    }
  } finally {
    await browser.close();
  }
  return results;
}

async function runMouseQa() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const diagnostics = attachDiagnostics(page);
  try {
    await waitForApp(page);
    await page.locator('select[data-action="input-mode"]').selectOption('mouse');
    await page.locator('button[data-action="start"]').click();
    await waitForPlaying(page);
    const canvas = page.locator('canvas[data-role="human-canvas"]');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('human canvas has no bounding box');
    await canvas.hover({ position: { x: box.width * 0.12, y: box.height * 0.5 } });
    await page.waitForFunction(() => document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === false, undefined, { timeout: 10_000 });
    await page.waitForTimeout(180);
    const leftAction = await page.locator('[data-role="human-input"]').textContent();
    await canvas.hover({ position: { x: box.width * 0.88, y: box.height * 0.5 } });
    await page.waitForTimeout(180);
    const rightAction = await page.locator('[data-role="human-input"]').textContent();
    const frameEvidence = await page.evaluate(() => {
      const canvas = document.querySelector('canvas[data-role="human-canvas"]');
      if (!(canvas instanceof HTMLCanvasElement)) return null;
      const context = canvas.getContext('2d');
      if (!context) return null;
      const pixels = context.getImageData(0, canvas.height - 20, canvas.width, 20).data;
      let redPixelCount = 0;
      let minX = canvas.width;
      let maxX = -1;
      for (let y = 0; y < 20; y += 1) {
        for (let x = 0; x < canvas.width; x += 1) {
          const offset = (y * canvas.width + x) * 4;
          const red = pixels[offset] ?? 0;
          const green = pixels[offset + 1] ?? 0;
          const blue = pixels[offset + 2] ?? 0;
          if (red >= 100 && red - green >= 35 && red - blue >= 35) {
            redPixelCount += 1;
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
          }
        }
      }
      return {
        redPixelCount,
        paddleCenterNormalized: maxX >= minX ? ((minX + maxX + 1) / 2) / canvas.width : null,
      };
    });
    if (leftAction !== 'LEFT' || rightAction !== 'RIGHT') throw new Error(`rapid mouse direction failed: ${leftAction} / ${rightAction}`);
    if (!frameEvidence?.redPixelCount) throw new Error(`human raw RGB canvas did not expose a detectable paddle: ${JSON.stringify(frameEvidence)}`);
    assertDiagnostics(diagnostics, 'mouse QA');
    const result = {
      schemaVersion: 1,
      artifactType: 'issue68_mouse_control_qa',
      timestamp: new Date().toISOString(),
      url,
      implementation: 'Visible-cursor absolute target with paddle-relative hysteresis; no emulator or paddle state mutation',
      startThreshold: 0.06,
      stopThreshold: 0.025,
      pointerLockUsed: await page.evaluate(() => document.pointerLockElement !== null),
      latestPaddleStateReadBy: 'DualGameLoop humanAction callback from fixed bottom raw-RGB ROI',
      frameEvidence,
      rapidDirectionTest: { leftAction, rightAction, passed: true },
      diagnostics,
    };
    await writeFile(artifactPath('mouse-control-qa.json'), `${JSON.stringify(result, null, 2)}\n`, 'utf8');
    return result;
  } finally {
    await context.close();
    await browser.close();
  }
}

async function runDifficultyQa() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const diagnostics = attachDiagnostics(page);
  try {
    await waitForApp(page);
    const options = await page.locator('select[data-action="difficulty"] option').evaluateAll((elements) => elements.map((element) => element.value));
    const defaultValue = await page.locator('select[data-action="difficulty"]').inputValue();
    await page.locator('button[data-action="start"]').click();
    await waitForPlaying(page);
    const beforeSwitchState = await page.locator('[data-role="agent-stage-state"]').textContent();
    const observed = {};
    for (const difficulty of Object.keys(rates)) {
      await page.locator('select[data-action="difficulty"]').selectOption(difficulty);
      observed[difficulty] = {
        label: await page.locator('[data-role="ai-difficulty-label"]').textContent(),
        message: await page.locator('[data-role="user-message"]').textContent(),
      };
    }
    const afterSwitchState = await page.locator('[data-role="agent-stage-state"]').textContent();
    if (defaultValue !== 'hard' || options.join(',') !== 'easy,medium,hard,unbeatable') throw new Error(`difficulty selector contract failed: ${defaultValue} / ${options.join(',')}`);
    if (beforeSwitchState !== 'PLAYING' || afterSwitchState !== 'PLAYING') throw new Error(`difficulty switch restarted gameplay: ${beforeSwitchState} / ${afterSwitchState}`);
    for (const difficulty of Object.keys(rates)) {
      if (observed[difficulty].label !== difficulty.toUpperCase() || !observed[difficulty].message?.includes(difficulty.toUpperCase())) throw new Error(`live ${difficulty} switch failed: ${JSON.stringify(observed[difficulty])}`);
    }
    await page.close();
    const debugPage = await context.newPage();
    const debugDiagnostics = attachDiagnostics(debugPage);
    await debugPage.goto(withDebug(url), { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await debugPage.locator('button[data-action="start"]').click();
    await waitForPlaying(debugPage);
    await debugPage.locator('select[data-action="difficulty"]').selectOption('unbeatable');
    const debugState = await debugPage.evaluate(() => ({
      difficulty: document.querySelector('[data-role="debug-difficulty"]')?.textContent,
      mistakeRate: document.querySelector('[data-role="difficulty-rate"]')?.textContent,
      greedyAction: document.querySelector('[data-role="greedy-action"]')?.textContent,
      mistakeInjected: document.querySelector('[data-role="mistake-injected"]')?.textContent,
    }));
    if (debugState.difficulty !== 'UNBEATABLE' || debugState.mistakeRate !== '0%') throw new Error(`debug difficulty state failed: ${JSON.stringify(debugState)}`);
    assertDiagnostics(diagnostics, 'difficulty production QA');
    assertDiagnostics(debugDiagnostics, 'difficulty debug QA');
    const result = {
      schemaVersion: 1,
      artifactType: 'issue68_difficulty_qa',
      timestamp: new Date().toISOString(),
      url,
      defaultDifficulty: defaultValue,
      options,
      mistakeRates: rates,
      liveSwitch: { beforeSwitchState, afterSwitchState, observed, passed: true },
      stochasticPolicyQa: runDeterministicDifficultyQa(),
      debugState,
      unbeatableRegression: {
        greedyOnly: true,
        mistakeRate: 0,
        observedDifficulty: debugState.difficulty,
        fixedFixtureActions: ['RIGHT', 'NOOP', 'LEFT'],
        verifiedBy: 'web/src/app/difficultyPolicy.test.ts',
      },
      diagnostics: { production: diagnostics, debug: debugDiagnostics },
    };
    await writeFile(artifactPath('difficulty-qa.json'), `${JSON.stringify(result, null, 2)}\n`, 'utf8');
    await debugPage.close();
    return result;
  } finally {
    await context.close();
    await browser.close();
  }
}

async function runProductionSmoke() {
  const modes = [
    { name: 'webgpu-preferred', args: [] },
    { name: 'wasm-fallback', args: ['--disable-features=WebGPU', '--disable-gpu'] },
  ];
  const runs = [];
  for (const mode of modes) {
    const browser = await chromium.launch({ channel: 'chrome', headless: true, args: mode.args });
    const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });
    const page = await context.newPage();
    const diagnostics = attachDiagnostics(page);
    try {
      await waitForApp(page);
      await page.evaluate(() => document.querySelector('button[data-action="start"]')?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
      await waitForPlaying(page);
      const keyboardCanvas = page.locator('canvas[data-role="human-canvas"]');
      await page.keyboard.down('ArrowLeft');
      await page.waitForTimeout(120);
      const keyboardAction = await page.locator('[data-role="human-input"]').textContent();
      await page.keyboard.up('ArrowLeft');
      await page.locator('button[data-action="pause"]').click();
      await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PAUSED', undefined, { timeout: 30_000 });
      await page.locator('select[data-action="input-mode"]').selectOption('mouse');
      const box = await keyboardCanvas.boundingBox();
      if (!box) throw new Error('human canvas has no bounding box');
      await page.locator('button[data-action="start"]').click();
      await waitForPlaying(page);
      await keyboardCanvas.hover({ position: { x: box.width * 0.12, y: box.height * 0.5 } });
      await page.waitForFunction(() => document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === false, undefined, { timeout: 10_000 });
      await page.waitForTimeout(120);
      const mouseAction = await page.locator('[data-role="human-input"]').textContent();
      await page.locator('select[data-action="difficulty"]').selectOption('easy');
      const difficulty = await page.locator('[data-role="ai-difficulty-label"]').textContent();
      await page.evaluate(() => document.querySelector('button[data-action="reset"]')?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
      await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'READY', undefined, { timeout: 30_000 });
      const productState = await page.evaluate(() => ({
        humanStage: document.querySelector('[data-role="human-stage-state"]')?.textContent,
        agentStage: document.querySelector('[data-role="agent-stage-state"]')?.textContent,
        dualCanvasCount: document.querySelectorAll('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').length,
        crossOriginIsolated: window.crossOriginIsolated,
        pointerLockElement: document.pointerLockElement,
        targetMarkerVisible: document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === false,
      }));
      const productScreenshot = mode.name === 'webgpu-preferred'
        ? 'production-smoke-final.png'
        : 'production-smoke-wasm-fallback.png';
      await page.screenshot({ path: artifactPath(productScreenshot), fullPage: true });

      const debugPage = await context.newPage();
      const debugDiagnostics = attachDiagnostics(debugPage);
      await debugPage.goto(withDebug(url), { waitUntil: 'domcontentloaded', timeout: 120_000 });
      await debugPage.locator('button[data-action="start"]').click();
      await waitForPlaying(debugPage);
      const actualBackend = await debugPage.locator('[data-role="gameplay-backend"]').textContent();
      const environmentDiagnostics = await debugPage.evaluate(() => window.__day30EnvironmentDiagnostics ?? null);
      await debugPage.locator('details[data-role="technical-details"]').evaluate((details) => { details.open = true; });
      if (mode.name === 'wasm-fallback' && actualBackend !== 'WASM') throw new Error(`fallback smoke expected WASM, got ${actualBackend}`);
      if (mode.name === 'webgpu-preferred' && actualBackend !== 'WEBGPU' && actualBackend !== 'WASM') throw new Error(`unexpected preferred gameplay backend: ${actualBackend}`);
      const debugScreenshot = mode.name === 'webgpu-preferred'
        ? 'debug-1366x768.png'
        : 'debug-1366x768-wasm-fallback.png';
      await debugPage.screenshot({ path: artifactPath(debugScreenshot), fullPage: true });
      assertDiagnostics(diagnostics, `${mode.name} product`);
      assertDiagnostics(debugDiagnostics, `${mode.name} debug`);
      runs.push({
        mode: mode.name,
        observed: { keyboardAction, mouseAction, difficulty, pauseResumeRestart: true, productState, actualBackend, environmentDiagnostics },
        screenshots: { product: repoRelative(productScreenshot), debug: repoRelative(debugScreenshot) },
        diagnostics: { product: diagnostics, debug: debugDiagnostics },
      });
      await debugPage.close();
    } finally {
      await context.close();
      await browser.close();
    }
  }
  const result = {
    schemaVersion: 1,
    artifactType: 'issue68_production_smoke',
    timestamp: new Date().toISOString(),
    url,
    customDomain: new URL(url).hostname === 'breakout.tommypan.dev',
    runs,
    screenshots: {
      product: repoRelative('production-smoke-final.png'),
      debug: repoRelative('debug-1366x768.png'),
      wasmFallbackProduct: repoRelative('production-smoke-wasm-fallback.png'),
      wasmFallbackDebug: repoRelative('debug-1366x768-wasm-fallback.png'),
    },
  };
  await writeFile(artifactPath('production-smoke.json'), `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  return result;
}

const layouts = await captureLayouts();
const mouse = await runMouseQa();
const difficulty = await runDifficultyQa();
const productionSmoke = await runProductionSmoke();
console.log(JSON.stringify({ url, layouts, mouse, difficulty, productionSmoke }, null, 2));
