import { mkdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import process from 'node:process';

import { chromium } from 'playwright-core';

const args = new Map();
for (let index = 2; index < process.argv.length; index += 1) {
  const value = process.argv[index];
  if (value?.startsWith('--')) args.set(value.slice(2), process.argv[index + 1] ?? true);
}

const url = String(args.get('url') ?? 'http://127.0.0.1:5173/');
const outputDirectory = resolve(String(args.get('output-dir') ?? '../assets/gameplay-ux/visual-qa'));
const viewports = [
  { width: 1920, height: 1080 },
  { width: 1536, height: 864 },
  { width: 1366, height: 768 },
  { width: 1280, height: 720 },
  { width: 768, height: 1000 },
  { width: 390, height: 900 },
];

await mkdir(outputDirectory, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const results = [];

try {
  for (const viewport of viewports) {
    const context = await browser.newContext({ viewport, deviceScaleFactor: 1 });
    const page = await context.newPage();
    const pageErrors = [];
    const requestFailures = [];
    page.on('pageerror', (error) => pageErrors.push(String(error)));
    page.on('requestfailed', (request) => requestFailures.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown'}`));
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    await page.locator('button[data-action="start"]').waitFor({ state: 'visible', timeout: 30_000 });

    const layout = await page.evaluate(() => {
      const rect = (selector) => {
        const element = document.querySelector(selector);
        if (!element) return null;
        const box = element.getBoundingClientRect();
        return { left: Math.round(box.left), top: Math.round(box.top), width: Math.round(box.width), height: Math.round(box.height) };
      };
      const buttons = [...document.querySelectorAll('.game-controls button')].map((button) => {
        const box = button.getBoundingClientRect();
        return { action: button.getAttribute('data-action'), width: Math.round(box.width), height: Math.round(box.height) };
      });
      const details = document.querySelector('[data-role="technical-details"]');
      const technicalContent = document.querySelector('.technical-content');
      return {
        viewport: { width: innerWidth, height: innerHeight },
        documentWidth: document.documentElement.scrollWidth,
        viewportWidth: document.documentElement.clientWidth,
        documentHeight: document.documentElement.scrollHeight,
        viewportHeight: innerHeight,
        panelColumns: getComputedStyle(document.querySelector('.panel-grid')).gridTemplateColumns,
        gamePanels: [...document.querySelectorAll('.game-card')].map((panel) => { const box = panel.getBoundingClientRect(); return { left: Math.round(box.left), top: Math.round(box.top), width: Math.round(box.width), height: Math.round(box.height) }; }),
        canvases: [...document.querySelectorAll('canvas[data-role="human-canvas"], canvas[data-role="agent-canvas"]')].map((canvas) => {
          const box = canvas.getBoundingClientRect();
          return { width: Math.round(box.width), height: Math.round(box.height) };
        }),
        buttons,
        difficultyOptions: [...document.querySelectorAll('select[data-action="difficulty"] option')].map((option) => option.value),
        inputModeCount: document.querySelectorAll('select[data-action="input-mode"]').length,
        difficultyCount: document.querySelectorAll('select[data-action="difficulty"]').length,
        scoreCount: document.querySelectorAll('[data-role="human-score"], [data-role="human-lives"], [data-role="agent-score"], [data-role="agent-lives"]').length,
        hasTechnicalDetails: Boolean(details),
        visibleText: document.body.innerText,
        technicalOpen: details?.hasAttribute('open') ?? false,
        technicalContentHeight: technicalContent ? Math.round(technicalContent.getBoundingClientRect().height) : -1,
        visibleEngineeringWorkbench: Boolean(document.querySelector('.technical-content')?.getClientRects().length && details?.open),
      };
    });

    if (layout.documentWidth > layout.viewportWidth + 1) throw new Error(`${viewport.width}px viewport overflows horizontally: ${JSON.stringify(layout)}`);
    if (viewport.width >= 1280 && layout.documentHeight > layout.viewportHeight + 1) throw new Error(`${viewport.width}x${viewport.height}px production route scrolls vertically: ${JSON.stringify(layout)}`);
    if (layout.canvases.length !== 2) throw new Error(`${viewport.width}px does not expose both game canvases`);
    if (viewport.width >= 1280 && layout.canvases.some(({ height }) => height < 300)) throw new Error(`${viewport.width}px game canvas is below 300px: ${JSON.stringify(layout.canvases)}`);
    if (layout.buttons.some(({ height }) => height < 44)) throw new Error(`${viewport.width}px game control is below 44px: ${JSON.stringify(layout.buttons)}`);
    if (layout.inputModeCount !== 1 || layout.difficultyCount !== 1 || layout.scoreCount !== 4 || layout.difficultyOptions.join(',') !== 'easy,medium,hard,unbeatable') throw new Error(`${viewport.width}px gameplay controls are incomplete: ${JSON.stringify(layout)}`);
    if (layout.hasTechnicalDetails) throw new Error(`${viewport.width}px production route exposed technical controls`);
    if (/ONNX|WASM|WebGPU|Q-values|backend|parity|Contract|SHA|latency|policy/i.test(layout.visibleText)) throw new Error(`${viewport.width}px production route exposed engineering wording`);
    if (layout.technicalOpen || layout.visibleEngineeringWorkbench) throw new Error(`${viewport.width}px exposed technical details by default: ${JSON.stringify(layout)}`);

    if (viewport.width === 1366) {
      await page.locator('select[data-action="input-mode"]').selectOption('mouse');
      const humanCanvas = page.locator('canvas[data-role="human-canvas"]');
      const box = await humanCanvas.boundingBox();
      if (!box) throw new Error('human canvas has no bounding box');
      await humanCanvas.hover({ position: { x: box.width * 0.12, y: box.height * 0.5 } });
      await page.waitForFunction(() => document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === false, undefined, { timeout: 10_000 });
      const mouseState = await page.evaluate(() => ({
        markerVisible: document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === false,
        pointerLockElement: document.pointerLockElement,
        cursor: getComputedStyle(document.querySelector('canvas[data-role="human-canvas"]')).cursor,
      }));
      await page.locator('select[data-action="input-mode"]').selectOption('keyboard');
      await page.waitForFunction(() => document.querySelector('[data-role="mouse-target-marker"]')?.hasAttribute('hidden') === true, undefined, { timeout: 10_000 });
      await page.keyboard.down('ArrowRight');
      await page.waitForTimeout(120);
      const keyboardAction = await page.locator('[data-role="human-input"]').textContent();
      await page.keyboard.up('ArrowRight');
      if (!mouseState.markerVisible || mouseState.pointerLockElement !== null || mouseState.cursor === 'none' || keyboardAction !== 'RIGHT') throw new Error(`input mode switch failed: mouse=${JSON.stringify(mouseState)}, keyboard=${keyboardAction}`);
      await page.locator('select[data-action="difficulty"]').selectOption('medium');
      const difficultyMessage = await page.locator('[data-role="user-message"]').textContent();
      if (!difficultyMessage?.includes('MEDIUM')) throw new Error(`difficulty switch failed: ${difficultyMessage}`);
    }

    const screenshotPath = join(outputDirectory, `production-${viewport.width}x${viewport.height}.png`);
    await page.evaluate(() => document.activeElement?.blur());
    await page.screenshot({ path: screenshotPath, fullPage: true });
    results.push({ ...viewport, screenshotPath, layout, pageErrors, requestFailures });
    await context.close();
  }
} finally {
  await browser.close();
}

console.log(JSON.stringify({ url, results }, null, 2));
