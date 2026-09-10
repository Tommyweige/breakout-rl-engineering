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
    } else {
      values.set(key, true);
    }
  }
  return values;
}

const args = parseArgs(process.argv.slice(2));
const url = String(args.get('url') ?? 'http://127.0.0.1:5173/');
const output = resolve(String(args.get('output') ?? '../assets/day27/browser-wasm-validation.json'));
const foundationScreenshot = resolve(String(args.get('foundation-screenshot') ?? '../assets/day27/dual-panel-browser-foundation.png'));
const validationScreenshot = resolve(String(args.get('validation-screenshot') ?? '../assets/day27/browser-wasm-validation.png'));
const headless = args.get('headless') === true;

await mkdir(dirname(output), { recursive: true });
await mkdir(dirname(foundationScreenshot), { recursive: true });
await mkdir(dirname(validationScreenshot), { recursive: true });

const consoleErrors = [];
const pageErrors = [];
const requestFailures = [];
const badRuntimeResponses = [];
const browser = await chromium.launch({
  channel: 'chrome',
  headless,
});

try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
    if (message.type() === 'error' || message.type() === 'warning') console.log(`[browser:${message.type()}] ${message.text()}`);
  });
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('requestfailed', (request) => requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));
  page.on('response', (response) => {
    if (response.status() >= 400 && !response.url().includes('/favicon')) {
      badRuntimeResponses.push(`${response.status()} ${response.url()}`);
    }
  });

  console.log(`Opening ${url}`);
  // Cloudflare may keep analytics/beacon requests open after the document is
  // usable. DOM readiness plus the app's visible control is the reliable gate.
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  console.log('Document loaded');
  await page.waitForLoadState('networkidle', { timeout: 15_000 }).catch(() => undefined);
  console.log('Waiting for app controls');
  await page.locator('button[data-action="validate"]').waitFor({ state: 'visible', timeout: 30_000 });
  console.log('Capturing foundation screenshot');
  await page.screenshot({ path: foundationScreenshot, fullPage: true });

  console.log('Starting WASM validation');
  await page.locator('button[data-action="validate"]').click();
  console.log('Waiting for validation artifact');
  const waitStartedAt = Date.now();
  while (Date.now() - waitStartedAt < 180_000) {
    const progress = await page.evaluate(() => ({
      runtimeStatus: document.querySelector('[data-role="runtime-status"]')?.textContent,
      validationStatus: document.querySelector('[data-role="validation-status"]')?.textContent,
      message: document.querySelector('[data-role="validation-message"]')?.textContent,
      artifact: window.__day27Validation ?? null,
      resources: performance.getEntriesByType('resource')
        .map((entry) => ({ name: entry.name, duration: Math.round(entry.duration), size: entry.transferSize }))
        .filter(({ name }) => /model\.onnx|fixtures|wasm|inference_spec|web-model-manifest/i.test(name)),
    }));
    console.log(JSON.stringify({
      elapsedSeconds: Math.round((Date.now() - waitStartedAt) / 1000),
      runtimeStatus: progress.runtimeStatus,
      validationStatus: progress.validationStatus,
      message: progress.message,
      hasArtifact: Boolean(progress.artifact),
      resources: progress.resources,
    }));
    if (progress.artifact) break;
    await new Promise((resolve) => setTimeout(resolve, 2_000));
  }
  const artifact = await page.evaluate(() => window.__day27Validation);
  if (!artifact) throw new Error('browser did not publish __day27Validation');
  if (!artifact.passed || artifact.requestedBackend !== 'wasm' || artifact.actualBackend !== 'wasm') {
    throw new Error(`WASM validation did not pass: ${JSON.stringify(artifact)}`);
  }
  if (pageErrors.length > 0) throw new Error(`browser page errors: ${pageErrors.join(' | ')}`);
  if (requestFailures.length > 0) throw new Error(`browser request failures: ${requestFailures.join(' | ')}`);
  if (badRuntimeResponses.length > 0) throw new Error(`runtime resources returned errors: ${badRuntimeResponses.join(' | ')}`);

  await page.screenshot({ path: validationScreenshot, fullPage: true });
  await writeFile(output, `${JSON.stringify(artifact, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({
    url: page.url(),
    browser: artifact.browser,
    playwrightBrowserVersion: browser.version(),
    wasm: {
      passed: artifact.passed,
      sampleCount: artifact.sampleCount,
      maxAbsoluteError: artifact.maxAbsoluteError,
      meanAbsoluteError: artifact.meanAbsoluteError,
      actionAgreementRate: artifact.actionAgreementRate,
      disagreementIndices: artifact.disagreementIndices,
      requestedBackend: artifact.requestedBackend,
      actualBackend: artifact.actualBackend,
    },
    modelSha256: artifact.modelSha256,
    ortWebVersion: artifact.ortWebVersion,
    output,
    foundationScreenshot,
    validationScreenshot,
    consoleErrors,
  }, null, 2));
} finally {
  await browser.close();
}
