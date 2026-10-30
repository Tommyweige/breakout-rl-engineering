// Records the case-study hero clip (the tracker playing in the web demo) and
// takes desktop and mobile screenshots of the case study pages.
//
//   npm run build && npx vite preview --port 4173 &
//   node scripts/capture-case-study.mjs --chromium /usr/bin/chromium
//
// The clip is recorded straight from the AI game canvas, so it contains only
// game pixels. ffmpeg then speeds it up, scales it with nearest-neighbour
// sampling, and writes tracker.mp4 and tracker-poster.png for web/public/case-study.

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
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
    chromium: { type: 'string' },
    seconds: { type: 'string', default: '150' },
    speed: { type: 'string', default: '6' },
  },
});

const baseUrl = new URL(values.url);
const outputDirectory = resolve(values.output);
const seconds = Number(values.seconds);
const speed = Number(values.speed);
if (!Number.isFinite(seconds) || seconds < 5) throw new Error('--seconds must be at least 5');
if (!Number.isFinite(speed) || speed < 1) throw new Error('--speed must be at least 1');

await mkdir(outputDirectory, { recursive: true });

const browser = await chromium.launch({ headless: true, executablePath: values.chromium });
const errors = [];

try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));

  await page.goto(new URL('/', baseUrl).href, { waitUntil: 'networkidle' });
  await page.selectOption('#difficulty-select', 'unbeatable');
  await page.evaluate(() => {
    const canvas = document.querySelectorAll('canvas')[1];
    const recorder = new MediaRecorder(canvas.captureStream(30), {
      mimeType: 'video/webm;codecs=vp9',
      videoBitsPerSecond: 2_000_000,
    });
    const chunks = [];
    recorder.ondataavailable = (event) => chunks.push(event.data);
    recorder.start(1000);
    window.__trackerRecording = { recorder, chunks };
  });
  await page.click('[data-action="start"]');
  await page.waitForTimeout(seconds * 1000);

  const gameplay = await page.evaluate(() => ({
    agentState: document.querySelector('[data-role="agent-state"]')?.textContent ?? null,
    agentScore: document.querySelector('[data-role="agent-score"]')?.textContent ?? null,
  }));
  const webmBase64 = await page.evaluate(async () => {
    const { recorder, chunks } = window.__trackerRecording;
    await new Promise((done) => {
      recorder.onstop = done;
      recorder.stop();
    });
    const bytes = new Uint8Array(await new Blob(chunks, { type: 'video/webm' }).arrayBuffer());
    let binary = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
  });
  await context.close();

  const rawClip = resolve(outputDirectory, 'tracker-raw.webm');
  const clip = resolve(outputDirectory, 'tracker.mp4');
  const poster = resolve(outputDirectory, 'tracker-poster.png');
  await writeFile(rawClip, Buffer.from(webmBase64, 'base64'));
  assert.ok(webmBase64.length > 0, 'the canvas recording is empty');

  const scale = 'scale=480:630:flags=neighbor';
  execFileSync('ffmpeg', [
    '-y', '-loglevel', 'error', '-i', rawClip,
    '-vf', `setpts=PTS/${speed},fps=30,${scale}`,
    '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26', '-movflags', '+faststart',
    clip,
  ]);
  // The poster shows the tracker partway through digging into the wall.
  const posterAt = ((seconds / speed) * 0.6).toFixed(2);
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', posterAt, '-i', clip, '-frames:v', '1', poster]);

  const screenshots = [];
  for (const [name, options] of [
    ['desktop', { viewport: { width: 1440, height: 1000 } }],
    ['mobile', { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 }],
  ]) {
    for (const path of ['/case-study.html', '/case-study.zh-TW.html']) {
      const pageContext = await browser.newContext({ ...options, reducedMotion: 'reduce' });
      const casePage = await pageContext.newPage();
      casePage.on('pageerror', (error) => errors.push(error.message));
      await casePage.goto(new URL(path, baseUrl).href, { waitUntil: 'networkidle' });
      await casePage.waitForFunction(
        () => document.querySelector('#app')?.getAttribute('data-summary-state') === 'loaded',
        undefined,
        { timeout: 15000 },
      );
      const file = resolve(outputDirectory, `case-study-${name}${path.includes('zh-TW') ? '.zh-TW' : ''}.png`);
      await casePage.screenshot({ path: file, fullPage: true });
      screenshots.push(file);
      await pageContext.close();
    }
  }

  console.log(JSON.stringify({ outputDirectory, gameplay, clip, poster, screenshots, errors }, null, 2));
} finally {
  await browser.close();
}
