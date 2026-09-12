import { readFile, mkdir, writeFile } from 'node:fs/promises';
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
    } else values.set(key, true);
  }
  return values;
}

const args = parseArgs(process.argv.slice(2));
const url = String(args.get('url') ?? 'http://127.0.0.1:5173/');
const fixturePath = resolve(String(args.get('fixture') ?? '../assets/day29/native-preprocessing-fixture.json'));
const outputPath = resolve(String(args.get('output') ?? '../assets/day29/browser-preprocessing-trace.json'));
const fixture = JSON.parse(await readFile(fixturePath, 'utf8'));
const actions = fixture.actions;

await mkdir(dirname(outputPath), { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: args.get('headed') !== true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
const page = await context.newPage();
const consoleErrors = [];
const pageErrors = [];
const requestFailures = [];
page.on('console', (message) => {
  if (message.type() === 'error') consoleErrors.push(message.text());
});
page.on('pageerror', (error) => pageErrors.push(String(error)));
page.on('requestfailed', (request) => requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));

try {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.waitForFunction(() => typeof window.__day29CapturePreprocessingTrace === 'function', undefined, { timeout: 30_000 });
  const samples = [];
  for (const seed of fixture.seeds) {
    console.log(`Capturing Browser preprocessing trace seed=${seed}`);
    const trace = await page.evaluate(({ seed: requestedSeed, actions: requestedActions }) => (
      window.__day29CapturePreprocessingTrace?.(requestedSeed, requestedActions) ?? null
    ), { seed, actions });
    if (!trace) throw new Error(`Browser did not return a preprocessing trace for seed ${seed}`);
    samples.push(trace);
  }
  if (pageErrors.length || requestFailures.length || consoleErrors.length) {
    throw new Error(`Browser diagnostics were not clean: ${JSON.stringify({ consoleErrors, pageErrors, requestFailures })}`);
  }
  const artifact = {
    schemaVersion: 1,
    artifactType: 'day29_browser_preprocessing_trace',
    source: url,
    actions,
    seeds: fixture.seeds,
    samples,
    diagnostics: { consoleErrors, pageErrors, requestFailures },
  };
  await writeFile(outputPath, `${JSON.stringify(artifact)}\n`, 'utf8');
  console.log(JSON.stringify({ output: outputPath, seeds: fixture.seeds, actions }));
} finally {
  await context.close().catch(() => undefined);
  await browser.close();
}
