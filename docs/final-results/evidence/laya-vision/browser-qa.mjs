/**
 * Laya browser QA harness (PR #74 / issue #73 closeout).
 *
 * Runs the built-or-dev browser app in a real Chrome on Windows against the
 * local CUDA bridge, selects the Decision Model controller, records one
 * gameplay session, then exercises controller switching and pause/resume.
 *
 * Prerequisites:
 *   - Laya CUDA bridge on 127.0.0.1:8766 (`python -m scripts.inference.serve_laya_vision`)
 *   - Web dev server on 0.0.0.0:5180 (`npm run dev -- --host 0.0.0.0 --port 5180`)
 *   - Chrome installed at the path in CHROME below
 *   - `npm install playwright-core@1.63.0` in the directory this file lives in
 *
 * Usage (from a Windows shell):
 *   set QA_OUT=E:\path\to\output
 *   node browser-qa.mjs
 *
 * The script writes one JSON record. It never restarts or reconfigures the
 * services it measures.
 */
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright-core';

const URL = process.env.QA_URL ?? 'http://localhost:5180/?debug=1';
const CHROME = process.env.QA_CHROME ?? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const OUT = process.env.QA_OUT ?? 'E:/Learning/pr74-qa/out';
const RECORD = process.env.QA_RECORD ?? 'laya-browser-qa.json';
const GAME_SECONDS = Number(process.env.QA_GAME_SECONDS ?? 40);
mkdirSync(OUT, { recursive: true });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const gpuSample = () => {
  try {
    const line = execFileSync('nvidia-smi', ['--query-gpu=name,utilization.gpu,memory.used,power.draw',
      '--format=csv,noheader,nounits'], { encoding: 'utf8' }).trim().split('\n')[0];
    const [name, util, mem, power] = line.split(',').map((v) => v.trim());
    return { name, utilizationPercent: Number(util), memoryMiB: Number(mem), powerW: Number(power) };
  } catch {
    return null;
  }
};

const snapshot = (page) => page.evaluate(() => {
  const loop = window.__humanInteractiveDiagnostics?.runtime ?? {};
  const env = window.__day30EnvironmentDiagnostics ?? {};
  const txt = (role) => document.querySelector(`[data-role="${role}"]`)?.textContent?.trim() ?? null;
  return {
    tMs: Math.round(performance.now()),
    state: txt('agent-state'),
    score: txt('agent-score'),
    lives: txt('agent-lives'),
    backend: txt('gameplay-backend'),
    evidence: txt('backend-evidence'),
    difficulty: txt('ai-difficulty-label'),
    decisions: loop.agentDecisionCount ?? null,
    decisionsPerSecond: loop.agentDecisionsPerSecond ?? null,
    decisionP50Ms: loop.agentDecisionP50Ms ?? null,
    decisionP95Ms: loop.agentDecisionP95Ms ?? null,
    nativeFramesPerSecond: loop.agentNativeFramesPerSecond ?? null,
    presentationFramesConsumed: loop.agentPresentationFramesConsumed ?? null,
    presentationBufferedFrames: loop.agentPresentationBufferedFrames ?? null,
    renderCallbackP50Ms: loop.renderCallbackP50Ms ?? null,
    scheduleWaitP50Ms: loop.agentScheduleWaitP50Ms ?? null,
    lastDecisionTimings: env.lastDecisionTimings ?? null,
  };
});

const browser = await chromium.launch({ executablePath: CHROME, headless: false,
  args: ['--window-size=1940,1000', '--disable-features=CalculateNativeWinOcclusion', '--no-first-run'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 960 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${String(e)}`));
page.on('console', (m) => { if (m.type() === 'error') errors.push(`console: ${m.text()}`); });

const record = { schema_version: 1, artifact_type: 'laya_browser_qa_session', recorded_at: new Date().toISOString(),
  url: URL, environment: {}, runtime: null, gpu: [], decision_model_session: {}, controller_switch: {}, pause_resume: {}, errors };

try {
  await page.goto(URL, { waitUntil: 'load', timeout: 120_000 });
  await page.waitForSelector('button[data-action="start"]', { timeout: 60_000 });
  record.environment = await page.evaluate(() => ({
    userAgent: navigator.userAgent, platform: navigator.platform, dpr: window.devicePixelRatio,
    viewport: { width: window.innerWidth, height: window.innerHeight },
    crossOriginIsolated: window.crossOriginIsolated,
  }));
  record.runtime = await page.evaluate(async () => await (await fetch('/api/laya/health')).json());
  record.environment.displayHz = await page.evaluate(async () => await new Promise((resolve) => {
    const t = []; const start = performance.now();
    const tick = (now) => { t.push(now); if (now - start < 2000) requestAnimationFrame(tick);
      else { const g = t.slice(1).map((v, i) => v - t[i]).sort((a, b) => a - b); resolve(+(1000 / g[Math.floor(g.length / 2)]).toFixed(1)); } };
    requestAnimationFrame(tick);
  }));
  log('loaded', record.runtime?.gpu, record.environment.displayHz, 'hz');

  await page.selectOption('#difficulty-select', 'decision-model');
  await page.waitForFunction(() => document.querySelector('[data-role="ai-difficulty-label"]')?.textContent === 'DECISION MODEL', null, { timeout: 120_000 });
  await page.click('button[data-action="start"]');
  await page.waitForFunction(() => window.__day30Ready === true, null, { timeout: 180_000 });
  const startedAt = Date.now();
  const sessionStartMs = await page.evaluate(() => Math.round(performance.now()));
  log('session started at page time', sessionStartMs);

  const samples = [];
  const latency = [];
  let lastLatency = '';
  let gameOver = null;
  while (Date.now() - startedAt < GAME_SECONDS * 1000) {
    const s = await snapshot(page);
    samples.push(s);
    const l = s.lastDecisionTimings;
    if (l && JSON.stringify(l) !== lastLatency) { latency.push(l); lastLatency = JSON.stringify(l); }
    const g = gpuSample(); if (g) record.gpu.push({ tMs: s.tMs, ...g });
    if (!gameOver && s.state === 'Game over') { gameOver = { tMs: s.tMs, playedMs: s.tMs - sessionStartMs, score: s.score, lives: s.lives }; log('game over', s.score); break; }
    await page.waitForTimeout(1000);
  }
  record.decision_model_session = { startedAt: new Date(startedAt).toISOString(), sessionStartMs,
    samples, latencySamples: latency, gameOver, finalSample: samples.at(-1) };
  log('recorded', samples.length, 'samples', latency.length, 'latency rows', 'gameOver', JSON.stringify(gameOver));

  const before = await snapshot(page);
  await page.selectOption('#difficulty-select', 'hard');
  await page.waitForFunction(() => document.querySelector('[data-role="ai-difficulty-label"]')?.textContent === 'HARD', null, { timeout: 60_000 });
  await page.waitForTimeout(6000);
  const dqn = await snapshot(page);
  await page.selectOption('#difficulty-select', 'decision-model');
  await page.waitForFunction(() => document.querySelector('[data-role="ai-difficulty-label"]')?.textContent === 'DECISION MODEL', null, { timeout: 120_000 });
  await page.waitForTimeout(8000);
  const back = await snapshot(page);
  record.controller_switch = { beforeSwitch: before, dqn, backToDecisionModel: back };
  log('switch', dqn.decisionsPerSecond, back.decisionsPerSecond);

  await page.click('button[data-action="pause"]');
  await page.waitForTimeout(2000);
  const paused = await snapshot(page);
  await page.waitForTimeout(2500);
  const pausedLater = await snapshot(page);
  await page.click('button[data-action="start"]');
  await page.waitForTimeout(4000);
  const resumed = await snapshot(page);
  record.pause_resume = { paused, pausedLater, resumed };
  log('pause/resume', paused.state, paused.decisions, pausedLater.decisions, resumed.decisions);
} catch (error) {
  record.failure = String(error?.stack ?? error);
  log('FAILED', record.failure);
} finally {
  record.errors = errors;
  writeFileSync(`${OUT}/${RECORD}`, JSON.stringify(record, null, 2));
  await browser.close();
  log('written');
}
