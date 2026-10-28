import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright-core';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(scriptDirectory, '..');

const { values } = parseArgs({
  options: {
    url: { type: 'string', default: 'http://localhost:4173' },
    output: { type: 'string', default: resolve(webRoot, '..', '.amp/in/case-study') },
    seconds: { type: 'string', default: '14' },
  },
});

const baseUrl = new URL(values.url);
const outputDirectory = resolve(values.output);
const seconds = Number(values.seconds);
if (!Number.isFinite(seconds) || seconds < 5) {
  throw new Error('--seconds must be a number of at least 5');
}

const rawVideoDirectory = resolve(outputDirectory, 'raw');
await mkdir(rawVideoDirectory, { recursive: true });

const browser = await chromium.launch({ headless: true });
const errors = [];

try {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: rawVideoDirectory, size: { width: 1280, height: 800 } },
  });
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });

  await page.goto(new URL('/', baseUrl).href, { waitUntil: 'networkidle' });
  await page.screenshot({ path: resolve(outputDirectory, 'game-before.png'), fullPage: true });
  await page.click('[data-action="start"]');
  await page.waitForFunction(
    () => document.querySelector('[data-role="agent-state"]')?.textContent === 'Playing',
    undefined,
    { timeout: 30000 },
  );
  await page.waitForTimeout(2500);
  await page.locator('#play-area').scrollIntoViewIfNeeded();
  await page.waitForTimeout(5000);
  await page.screenshot({ path: resolve(outputDirectory, 'gameplay-poster.png') });
  await page.waitForTimeout(Math.max(0, seconds - 5) * 1000);

  const gameplay = await page.evaluate(() => ({
    agentState: document.querySelector('[data-role="agent-state"]')?.textContent ?? null,
    backend: document.querySelector('[data-role="gameplay-backend"]')?.textContent ?? null,
    agentScore: document.querySelector('[data-role="agent-score"]')?.textContent ?? null,
    humanScore: document.querySelector('[data-role="human-score"]')?.textContent ?? null,
  }));
  assert.equal(gameplay.agentState, 'Playing', 'the AI game must be running');
  const video = page.video();
  assert.ok(video, 'Playwright did not expose a video recording');
  await context.close();
  await video.saveAs(resolve(outputDirectory, 'gameplay.webm'));

  const evidenceContext = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    reducedMotion: 'reduce',
  });
  const evidencePage = await evidenceContext.newPage();
  evidencePage.on('pageerror', (error) => errors.push(error.message));
  await evidencePage.goto(new URL('/case-study.html', baseUrl).href, {
    waitUntil: 'networkidle',
  });
  await evidencePage.waitForFunction(
    () => document.querySelector('#app')?.getAttribute('data-summary-state') === 'loaded',
    undefined,
    { timeout: 15000 },
  );
  await evidencePage.screenshot({
    path: resolve(outputDirectory, 'case-study-desktop.png'),
    fullPage: false,
  });
  await evidenceContext.close();

  const mobileContext = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
    deviceScaleFactor: 2,
    reducedMotion: 'reduce',
  });
  const mobilePage = await mobileContext.newPage();
  await mobilePage.goto(new URL('/case-study.html', baseUrl).href, { waitUntil: 'networkidle' });
  await mobilePage.waitForFunction(
    () => document.querySelector('#app')?.getAttribute('data-summary-state') === 'loaded',
    undefined,
    { timeout: 15000 },
  );
  await mobilePage.screenshot({
    path: resolve(outputDirectory, 'case-study-mobile.png'),
    fullPage: false,
  });
  await mobileContext.close();

  console.log(
    JSON.stringify(
      {
        outputDirectory,
        gameplay,
        recording: resolve(outputDirectory, 'gameplay.webm'),
        poster: resolve(outputDirectory, 'gameplay-poster.png'),
        screenshots: [
          resolve(outputDirectory, 'game-before.png'),
          resolve(outputDirectory, 'case-study-desktop.png'),
          resolve(outputDirectory, 'case-study-mobile.png'),
        ],
        errors,
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
