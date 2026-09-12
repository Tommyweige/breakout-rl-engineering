import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
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
const url = String(args.get('url') ?? 'http://127.0.0.1:5174/');
const output = resolve(String(args.get('output') ?? '../assets/day29/browser-overlap-evaluation.json'));
const seeds = [
  ...Array.from({ length: 5 }, (_, index) => 101 + index),
  ...Array.from({ length: 5 }, (_, index) => 202 + index),
  ...Array.from({ length: 5 }, (_, index) => 303 + index),
  ...Array.from({ length: 15 }, (_, index) => 308 + index),
];
const selectedSeeds = seeds.slice(0, Number(args.get('count') ?? seeds.length));
const headless = args.get('headed') !== true;

const browser = await chromium.launch({ channel: 'chrome', headless });
const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
async function validate(page, backend) {
  await page.locator('select[data-action="backend"]').selectOption(backend);
  await page.locator('button[data-action="validate"]').click();
  const artifactName = backend === 'wasm' ? '__day27Validation' : '__day28Validation';
  await page.waitForFunction((name) => Boolean(window[name]), artifactName, { timeout: 180_000 });
  const artifact = await page.evaluate((name) => window[name], artifactName);
  if (!artifact?.passed || artifact.actualBackend !== backend) throw new Error(`${backend} validation failed: ${JSON.stringify(artifact)}`);
  return artifact;
}

async function evaluate(page, backend) {
  await page.locator('select[data-action="backend"]').selectOption(backend);
  const validation = await validate(page, backend);
  const evaluation = await page.evaluate((requestedSeeds) => window.__day29Evaluate?.(requestedSeeds) ?? null, selectedSeeds);
  if (!evaluation?.completed || evaluation.actualBackend !== backend) throw new Error(`${backend} overlap evaluation failed: ${JSON.stringify(evaluation)}`);
  return { validation, evaluation };
}

try {
  const pages = await Promise.all([context.newPage(), context.newPage()]);
  const diagnostics = pages.map((page) => ({ consoleErrors: [], pageErrors: [], requestFailures: [] }));
  pages.forEach((page, index) => {
    page.on('console', (message) => {
      if (message.type() === 'error') diagnostics[index].consoleErrors.push(message.text());
    });
    page.on('pageerror', (error) => diagnostics[index].pageErrors.push(String(error)));
    page.on('requestfailed', (request) => diagnostics[index].requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));
  });
  await Promise.all(pages.map(async (page) => {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await page.locator('button[data-action="validate"]').waitFor({ state: 'visible', timeout: 30_000 });
  }));
  const [wasmResult, webgpuResult] = await Promise.all([evaluate(pages[0], 'wasm'), evaluate(pages[1], 'webgpu')]);
  const wasm = wasmResult.evaluation;
  const webgpu = webgpuResult.evaluation;
  const artifact = {
    schemaVersion: 1,
    artifactType: 'day29_browser_overlap_evaluation',
    url,
    seeds: selectedSeeds,
    wasm,
    webgpu,
    validation: { wasm: wasmResult.validation, webgpu: webgpuResult.validation },
    diagnostics,
  };
  await writeFile(output, `${JSON.stringify(artifact, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({ output, wasm: wasm.aggregate, webgpu: webgpu.aggregate }, null, 2));
} finally {
  await context.close().catch(() => undefined);
  await browser.close();
}
