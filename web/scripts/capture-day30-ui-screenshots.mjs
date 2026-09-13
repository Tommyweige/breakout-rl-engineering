import { mkdir, writeFile } from 'node:fs/promises';
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
const headless = args.get('headed') !== true;
await mkdir(outputDirectory, { recursive: true });

function withDebug(value) {
  const parsed = new URL(value);
  parsed.searchParams.set('debug', '1');
  return parsed.toString();
}

async function capturePage(browser, url, route, screenshotPath) {
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
    await page.locator('button[data-action="start"]').waitFor({ state: 'visible', timeout: 30_000 });
    if (route === 'product') {
      const visibleText = await page.locator('body').innerText();
      if (await page.locator('[data-role="technical-details"]').count() !== 0) throw new Error('production route exposed the technical details disclosure');
      if (/ONNX|WASM|WebGPU|Q-values|backend|parity|Contract|SHA|latency|policy/i.test(visibleText)) {
        throw new Error(`production route exposed engineering wording: ${visibleText}`);
      }
      await page.locator('button[data-action="start"]').click();
      await page.waitForFunction(() => document.querySelector('[data-role="agent-stage-state"]')?.textContent === 'PLAYING', undefined, { timeout: 180_000 });
      await page.waitForTimeout(1200);
    } else {
      const details = page.locator('details[data-role="technical-details"]');
      await details.evaluate((element) => { element.open = true; });
      await page.locator('button[data-action="start"]').click();
      await page.waitForFunction(() => Boolean(window.__day30Ready), undefined, { timeout: 180_000 });
      await page.waitForTimeout(1200);
    }
    await page.evaluate(() => document.activeElement?.blur());
    await page.screenshot({ path: screenshotPath, fullPage: true });
    return {
      url,
      route,
      screenshotPath,
      canvasCount: await page.locator('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]').count(),
      technicalDetailsPresent: await page.locator('[data-role="technical-details"]').count() > 0,
      diagnostics: { consoleErrors, pageErrors, requestFailures, badResponses },
    };
  } finally {
    await context.close().catch(() => undefined);
  }
}

const browser = await chromium.launch({ channel: 'chrome', headless });
try {
  const product = await capturePage(browser, productUrl, 'product', resolve(outputDirectory, 'final-production-ui.png'));
  const debug = await capturePage(browser, debugUrl, 'debug', resolve(outputDirectory, 'final-debug-ui.png'));
  const result = { schemaVersion: 1, artifactType: 'day30_ui_screenshots', timestamp: new Date().toISOString(), product, debug };
  await writeFile(resolve(outputDirectory, 'final-ui-screenshots.json'), `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  const diagnostics = [...product.diagnostics.consoleErrors, ...product.diagnostics.pageErrors, ...product.diagnostics.requestFailures, ...product.diagnostics.badResponses, ...debug.diagnostics.consoleErrors, ...debug.diagnostics.pageErrors, ...debug.diagnostics.requestFailures, ...debug.diagnostics.badResponses];
  if (diagnostics.length) throw new Error(`UI screenshot diagnostics were not clean: ${JSON.stringify(result)}`);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
