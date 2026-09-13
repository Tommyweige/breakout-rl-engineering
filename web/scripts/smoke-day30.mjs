import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import process from 'node:process';

import { chromium } from 'playwright-core';

const url = String(process.argv[2] ?? 'http://127.0.0.1:5173/');
const debugUrl = withDebug(url);
const output = resolve(process.argv[3] ?? '../assets/day30/production-smoke.json');
const forceWasm = process.argv.includes('--force-wasm');
await mkdir(resolve(output, '..'), { recursive: true });

function withDebug(value) {
  const parsed = new URL(value);
  parsed.searchParams.set('debug', '1');
  return parsed.toString();
}

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: forceWasm ? ['--disable-features=WebGPU', '--disable-gpu'] : [],
});
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

try {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.locator('button[data-action="start"]').click();
  await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PLAYING', undefined, { timeout: 180_000 });
  await page.waitForTimeout(3000);
  const keyboardCanvas = page.locator('canvas[data-role="human-canvas"]');
  await page.evaluate(() => document.activeElement?.blur());
  await page.screenshot({ path: resolve(output, '..', 'final-human-vs-rl.png'), fullPage: true });
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
  await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PLAYING', undefined, { timeout: 30_000 });
  await keyboardCanvas.hover({ position: { x: box.width * 0.12, y: box.height * 0.5 } });
  await page.waitForFunction(() => document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === false, undefined, { timeout: 10_000 });
  await page.waitForTimeout(120);
  const mouseAction = await page.locator('[data-role="human-input"]').textContent();
  await page.evaluate(() => document.querySelector('button[data-action="reset"]')?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'READY', undefined, { timeout: 30_000 });
  const productCrossOriginIsolated = await page.evaluate(() => window.crossOriginIsolated);
  const debugPage = await context.newPage();
  debugPage.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
  debugPage.on('pageerror', (error) => pageErrors.push(String(error)));
  debugPage.on('requestfailed', (request) => requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));
  debugPage.on('response', (response) => { if (response.status() >= 400 && !response.url().includes('/favicon')) badResponses.push(`${response.status()} ${response.url()}`); });
  await debugPage.goto(debugUrl, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await debugPage.locator('details[data-role="technical-details"]').evaluate((details) => { details.open = true; });
  await debugPage.locator('button[data-action="start"]').click();
  await debugPage.waitForFunction(() => Boolean(window.__day30Ready), undefined, { timeout: 180_000 });
  await debugPage.waitForFunction(() => document.querySelector('[data-role="runtime-status-label"]')?.textContent === 'running', undefined, { timeout: 30_000 });
  const runtimeDiagnostics = await debugPage.evaluate(() => window.__day30EnvironmentDiagnostics ?? null);
  const actualBackend = await debugPage.locator('[data-role="gameplay-backend"]').textContent();
  const expectedBackend = forceWasm ? 'WASM' : 'WEBGPU';
  if (actualBackend !== expectedBackend) throw new Error(`expected ${expectedBackend} gameplay backend, got ${actualBackend}`);
  await debugPage.close();
  const result = {
    schemaVersion: 1,
    artifactType: 'day30_production_smoke',
    timestamp: new Date().toISOString(),
    url,
    debugUrl,
    mode: forceWasm ? 'forced-wasm-fallback' : 'production-webgpu',
    state: await page.evaluate(() => ({
      humanStage: document.querySelector('[data-role="human-stage-state"]')?.textContent,
      agentStage: document.querySelector('[data-role="agent-stage-state"]')?.textContent,
      agentScore: document.querySelector('[data-role="agent-score"]')?.textContent,
      agentLives: document.querySelector('[data-role="agent-lives"]')?.textContent,
    })),
    productValidation: {
      dualCanvasCount: await page.locator('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').count(),
      keyboardAction,
      mouseAction,
      pauseResumeRestart: true,
      crossOriginIsolated: productCrossOriginIsolated,
      actualBackend,
      fallbackPath: {
        tested: forceWasm,
        expectedBackend,
        observedBackend: actualBackend,
        passed: forceWasm ? actualBackend === 'WASM' : actualBackend === 'WEBGPU',
      },
      runtimeDiagnostics,
    },
    diagnostics: { consoleErrors, pageErrors, requestFailures, badResponses },
  };
  await writeFile(output, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  if (pageErrors.length || requestFailures.length || badResponses.length || consoleErrors.length) throw new Error(`browser diagnostics were not clean: ${JSON.stringify(result.diagnostics)}`);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await context.close().catch(() => undefined);
  await browser.close();
}
