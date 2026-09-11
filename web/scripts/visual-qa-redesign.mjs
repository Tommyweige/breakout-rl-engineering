import { mkdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';
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
const outputDir = resolve(String(args.get('output-dir') ?? '../.qa-redesign'));
const headless = args.get('headed') !== true;
const viewports = [
  { width: 1440, height: 1100 },
  { width: 1280, height: 1000 },
  { width: 768, height: 1000 },
  { width: 390, height: 900 },
  { width: 375, height: 900 },
];

await mkdir(outputDir, { recursive: true });

const browser = await chromium.launch({ channel: 'chrome', headless });
const viewportResults = [];
let validation = null;

try {
  for (const viewport of viewports) {
    const context = await browser.newContext({ viewport, deviceScaleFactor: 1 });
    const page = await context.newPage();
    const pageErrors = [];
    const requestFailures = [];
    page.on('pageerror', (error) => pageErrors.push(String(error)));
    page.on('requestfailed', (request) => requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));

    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await page.locator('button[data-action="validate"]').waitFor({ state: 'visible', timeout: 30_000 });

    const layout = await page.evaluate(() => {
      const buttons = [...document.querySelectorAll('button[data-action]')].map((button) => {
        const rect = button.getBoundingClientRect();
        return { action: button.getAttribute('data-action'), width: Math.round(rect.width), height: Math.round(rect.height) };
      });
      const panels = [...document.querySelectorAll('.panel')].map((panel) => {
        const rect = panel.getBoundingClientRect();
        return { left: Math.round(rect.left), top: Math.round(rect.top), width: Math.round(rect.width), height: Math.round(rect.height) };
      });
      return {
        viewport: { width: window.innerWidth, height: window.innerHeight },
        documentWidth: document.documentElement.scrollWidth,
        viewportWidth: document.documentElement.clientWidth,
        buttonSizes: buttons,
        panels,
        hasCanvas: Boolean(document.querySelector('canvas')),
        hasBackendSelector: Boolean(document.querySelector('select[data-action="backend"]')),
        hasGameplaySeam: /ALE gameplay not connected yet/i.test(document.body.innerText),
      };
    });

    if (layout.documentWidth > layout.viewportWidth + 1) {
      throw new Error(`${viewport.width}px viewport overflows horizontally: ${JSON.stringify(layout)}`);
    }
    if (layout.hasCanvas || !layout.hasBackendSelector || !layout.hasGameplaySeam) {
      throw new Error(`${viewport.width}px structural seam check failed: ${JSON.stringify(layout)}`);
    }
    if (layout.buttonSizes.some(({ height }) => height < 44)) {
      throw new Error(`${viewport.width}px control target is below 44px: ${JSON.stringify(layout.buttonSizes)}`);
    }

    const screenshotPath = join(outputDir, `browser-foundation-${viewport.width}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    viewportResults.push({ ...viewport, screenshotPath, layout, pageErrors, requestFailures });

    if (viewport.width === 1440) {
      await page.locator('button[data-action="start"]').click();
      const runningStatus = await page.locator('[data-role="runtime-status-label"]').textContent();
      await page.locator('button[data-action="pause"]').click();
      const pausedStatus = await page.locator('[data-role="runtime-status-label"]').textContent();
      await page.locator('button[data-action="reset"]').click();
      const resetStatus = await page.locator('[data-role="runtime-status-label"]').textContent();
      if (runningStatus !== 'running' || pausedStatus !== 'paused' || resetStatus !== 'ready') {
        throw new Error(`shared control transition failed: ${runningStatus} -> ${pausedStatus} -> ${resetStatus}`);
      }

      await page.keyboard.down('ArrowLeft');
      await page.waitForTimeout(100);
      const activeInput = await page.locator('[data-role="human-input"]').textContent();
      await page.keyboard.up('ArrowLeft');
      if (activeInput !== 'LEFT') throw new Error(`keyboard seam failed: ${activeInput}`);

      await page.locator('button[data-action="validate"]').click();
      await page.waitForFunction(() => Boolean(window.__day27Validation), undefined, { timeout: 180_000 });
      validation = await page.evaluate(() => ({
        artifact: window.__day27Validation,
        runtimeStatus: document.querySelector('[data-role="runtime-status-label"]')?.textContent,
        validationStatus: document.querySelector('[data-role="validation-status"]')?.textContent,
      }));
      const artifact = validation.artifact;
      if (!artifact?.passed || artifact.requestedBackend !== 'wasm' || artifact.actualBackend !== 'wasm') {
        throw new Error(`WASM validation gate failed: ${JSON.stringify(validation)}`);
      }
      if (validation.runtimeStatus !== 'ready' || validation.validationStatus !== 'PASS') {
        throw new Error(`runtime state did not remain ready/PASS: ${JSON.stringify(validation)}`);
      }
      await page.screenshot({ path: join(outputDir, 'browser-wasm-validation-1440.png'), fullPage: true });
    }

    await context.close();
  }
} finally {
  await browser.close();
}

console.log(JSON.stringify({
  url,
  browser: { name: 'Chrome', version: browser.version() },
  headless,
  viewports: viewportResults,
  validation,
}, null, 2));
