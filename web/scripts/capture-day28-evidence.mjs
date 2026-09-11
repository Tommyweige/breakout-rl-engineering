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
const validationOutput = resolve(String(args.get('validation') ?? '../assets/day28/webgpu-validation.json'));
const benchmarkOutput = resolve(String(args.get('benchmark') ?? '../assets/day28/web-benchmark.json'));
const dualPanelScreenshot = resolve(String(args.get('dual-panel-screenshot') ?? '../assets/day28/dual-panel-webgpu.png'));
const validationScreenshot = resolve(String(args.get('validation-screenshot') ?? '../assets/day28/webgpu-validation.png'));
const headless = args.get('headless') === true;

for (const path of [validationOutput, benchmarkOutput, dualPanelScreenshot, validationScreenshot]) {
  await mkdir(dirname(path), { recursive: true });
}

const consoleErrors = [];
const pageErrors = [];
const requestFailures = [];
const badResponses = [];
const browser = await chromium.launch({ channel: 'chrome', headless });
let context;

try {
  context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
    if (message.type() === 'error' || message.type() === 'warning') console.log(`[browser:${message.type()}] ${message.text()}`);
  });
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('requestfailed', (request) => requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));
  page.on('response', (response) => {
    if (response.status() >= 400 && !response.url().includes('/favicon')) badResponses.push(`${response.status()} ${response.url()}`);
  });

  console.log(`Opening ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.waitForLoadState('networkidle', { timeout: 15_000 }).catch(() => undefined);
  await page.locator('button[data-action="validate"]').waitFor({ state: 'visible', timeout: 30_000 });

  console.log('Starting WASM regression validation');
  await page.locator('button[data-action="validate"]').click();
  await page.waitForFunction(() => Boolean(window.__day27Validation), undefined, { timeout: 180_000 });
  const wasmValidation = await page.evaluate(() => window.__day27Validation);
  if (!wasmValidation || !wasmValidation.passed || wasmValidation.requestedBackend !== 'wasm' || wasmValidation.actualBackend !== 'wasm') {
    throw new Error(`WASM regression did not pass before backend switch: ${JSON.stringify(wasmValidation)}`);
  }

  console.log('Switching the same Agent Panel to WebGPU');
  await page.locator('select[data-action="backend"]').selectOption('webgpu');
  await page.locator('button[data-action="validate"]').click();
  await page.waitForFunction(() => Boolean(window.__day28Validation), undefined, { timeout: 180_000 });
  const validation = await page.evaluate(() => window.__day28Validation);
  if (!validation || !validation.passed || validation.requestedBackend !== 'webgpu' || validation.actualBackend !== 'webgpu') {
    throw new Error(`WebGPU validation did not pass without fallback: ${JSON.stringify(validation)}`);
  }
  if (!validation.webgpuSupport?.supported) {
    throw new Error(`WebGPU support was not available in the formal capture browser: ${JSON.stringify(validation)}`);
  }
  if (validation.backendEvidence !== 'webgpu_session_exposes_env_webgpu_device') {
    throw new Error(`WebGPU active backend was not independently observed: ${JSON.stringify(validation)}`);
  }
  if (validation.environmentContract?.status !== 'partial') {
    throw new Error(`Browser environment parity must be explicit and partial: ${JSON.stringify(validation.environmentContract)}`);
  }

  await page.screenshot({ path: dualPanelScreenshot, fullPage: true });
  await writeFile(validationOutput, `${JSON.stringify(validation, null, 2)}\n`, 'utf8');

  await page.locator('button[data-action="benchmark"]').click();
  await page.waitForFunction(() => Boolean(window.__day28Benchmark), undefined, { timeout: 180_000 });
  const benchmark = await page.evaluate(() => window.__day28Benchmark);
  if (!benchmark || benchmark.results.wasm.actualBackend !== 'wasm' || benchmark.results.webgpu.actualBackend !== 'webgpu') {
    throw new Error(`backend benchmark did not retain requested providers: ${JSON.stringify(benchmark)}`);
  }
  if (!benchmark.validations.wasm.passed || !benchmark.validations.webgpu.passed) {
    throw new Error(`fixed-state parity failed in benchmark: ${JSON.stringify(benchmark.validations)}`);
  }
  if (!benchmark.webgpuSupport?.supported) {
    throw new Error(`benchmark did not record WebGPU support: ${JSON.stringify(benchmark.webgpuSupport)}`);
  }
  if (benchmark.backendEvidence.wasm !== 'wasm_session_created_with_explicit_provider' || benchmark.backendEvidence.webgpu !== 'webgpu_session_exposes_env_webgpu_device') {
    throw new Error(`benchmark backend evidence was not truthful: ${JSON.stringify(benchmark.backendEvidence)}`);
  }
  await page.screenshot({ path: validationScreenshot, fullPage: true });
  await writeFile(benchmarkOutput, `${JSON.stringify(benchmark, null, 2)}\n`, 'utf8');

  if (pageErrors.length > 0) throw new Error(`browser page errors: ${pageErrors.join(' | ')}`);
  if (requestFailures.length > 0) throw new Error(`browser request failures: ${requestFailures.join(' | ')}`);
  if (badResponses.length > 0) throw new Error(`runtime resources returned errors: ${badResponses.join(' | ')}`);

  console.log(JSON.stringify({
    url: page.url(),
    browser: validation.browser,
    playwrightBrowserVersion: browser.version(),
    validation: {
      requestedBackend: validation.requestedBackend,
      actualBackend: validation.actualBackend,
      passed: validation.passed,
      sampleCount: validation.sampleCount,
      maxAbsoluteError: validation.maxAbsoluteError,
      meanAbsoluteError: validation.meanAbsoluteError,
      actionAgreementRate: validation.actionAgreementRate,
      qMargin: validation.qMargin,
    },
    wasmRegression: {
      requestedBackend: wasmValidation.requestedBackend,
      actualBackend: wasmValidation.actualBackend,
      passed: wasmValidation.passed,
      sampleCount: wasmValidation.sampleCount,
    },
    benchmark: {
      sampleCount: benchmark.sampleCount,
      warmupCount: benchmark.warmupCount,
      wasm: benchmark.results.wasm.summary,
      webgpu: benchmark.results.webgpu.summary,
    },
    modelSha256: validation.modelSha256,
    ortWebVersion: validation.ortWebVersion,
    validationOutput,
    benchmarkOutput,
    dualPanelScreenshot,
    validationScreenshot,
    consoleErrors,
  }, null, 2));
} finally {
  if (context) await context.close().catch(() => undefined);
  await browser.close();
}
