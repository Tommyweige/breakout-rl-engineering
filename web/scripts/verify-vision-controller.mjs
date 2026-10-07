import assert from 'node:assert/strict';
import { chromium } from 'playwright-core';

const url = new URL(process.argv[2] ?? 'http://127.0.0.1:5173');
url.searchParams.set('debug', '1');
const browser = await chromium.launch({ channel: 'msedge', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  const modelRequests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.url().endsWith('/model.onnx')) modelRequests.push(request.url()); });
  await page.goto(url.href);
  await page.selectOption('[data-action="difficulty"]', 'unbeatable');
  await page.click('[data-action="start"]');
  await page.waitForFunction(() => window.__day30Ready === true);
  await page.waitForFunction(() => Number(document.querySelector('[data-role="agent-score"]').textContent) > 0, undefined, { timeout: 30000 });
  await page.click('[data-action="pause"]');
  const vision = await page.evaluate(() => ({
    backend: document.querySelector('[data-role="gameplay-backend"]').textContent,
    model: document.querySelector('[data-role="model-loaded"]').textContent,
    score: Number(document.querySelector('[data-role="agent-score"]').textContent),
    runtime: window.__humanInteractiveDiagnostics.runtime,
    environment: window.__day30EnvironmentDiagnostics.agentRuntime,
    humanFrame: window.__mouseControlV3Diagnostics.rawFrameNumber,
    canvasLit: [...document.querySelector('[data-role="agent-canvas"]').getContext('2d').getImageData(0, 0, 160, 210).data].some((v, i) => i % 4 !== 3 && v > 0),
  }));
  assert.equal(vision.backend, 'PIXEL VISION / JS');
  assert.equal(vision.model, 'Predictive Vision Controller v1');
  assert.equal(vision.runtime.agentOuterActionRepeat, 1);
  assert.equal(vision.runtime.agentDecisionCount, vision.runtime.agentRawFrameDelta);
  assert.equal(vision.environment.controlContractId, 'breakout-evaluation-v3-frame-skip-1');
  assert.equal(vision.environment.resetNoopCount, 0);
  assert.equal(vision.environment.stickyActionProbability, 0.25);
  assert.equal(modelRequests.length, 0, 'Vision gameplay must not load the DQN');
  assert.ok(vision.canvasLit);
  const humanScore = await page.locator('[data-role="human-score"]').textContent();

  // Switch at a paused boundary, then while playing inside a DQN repeat window.
  await page.selectOption('[data-action="difficulty"]', 'hard');
  await page.waitForFunction(() => document.querySelector('[data-role="ai-difficulty-label"]').textContent === 'HARD');
  assert.equal(await page.locator('[data-role="human-score"]').textContent(), humanScore);
  assert.equal(await page.evaluate(() => window.__mouseControlV3Diagnostics.rawFrameNumber), vision.humanFrame);
  assert.ok(modelRequests.length > 0, 'Hard must still load the DQN');
  await page.click('[data-action="start"]');
  await page.waitForFunction(() => document.querySelector('[data-role="gameplay-backend"]').textContent === 'WASM');
  assert.equal(await page.evaluate(() => window.__humanInteractiveDiagnostics.runtime.agentOuterActionRepeat), 4);
  await page.selectOption('[data-action="difficulty"]', 'unbeatable');
  await page.waitForFunction(() => document.querySelector('[data-role="gameplay-backend"]').textContent === 'PIXEL VISION / JS');
  await page.waitForFunction(() => window.__humanInteractiveDiagnostics.runtime.agentDecisionCount >= 10);
  assert.equal(await page.evaluate(() => window.__humanInteractiveDiagnostics.runtime.agentOuterActionRepeat), 1);
  await page.click('[data-action="reset"]');
  await page.waitForFunction(() => window.__humanInteractiveDiagnostics.runtime === null);
  await page.click('[data-action="start"]');
  await page.waitForFunction(() => window.__humanInteractiveDiagnostics.runtime?.agentDecisionCount >= 10);
  await page.click('[data-action="pause"]');
  assert.equal(await page.locator('[data-role="gameplay-backend"]').textContent(), 'PIXEL VISION / JS');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ url: url.origin, visionScore: vision.score, visionDecisions: vision.runtime.agentDecisionCount,
    rawFrames: vision.runtime.agentRawFrameDelta, switches: 'vision → DQN → vision', reset: 'passed', errors }, null, 2));
} finally {
  await browser.close();
}
