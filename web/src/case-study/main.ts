import './case-study.css';

import {
  hydrateCaseStudy,
  renderCaseStudyShell,
  type FinalResultsSummary,
} from './renderCaseStudy';

const root = document.querySelector<HTMLElement>('#app');
if (!root) {
  throw new Error('missing #app root');
}

const appRoot: HTMLElement = root;
appRoot.innerHTML = renderCaseStudyShell();

const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)');
const video = appRoot.querySelector<HTMLVideoElement>('[data-role="gameplay-video"]');
if (video) {
  if (reduceMotion?.matches) {
    video.controls = true;
    video.pause();
  } else {
    void video.play().catch(() => {
      video.controls = true;
    });
  }
}

function showSummaryError(error: unknown): void {
  appRoot.dataset.summaryState = 'error';
  const status = appRoot.querySelector<HTMLElement>('[data-role="summary-status"]');
  if (!status) return;
  status.dataset.state = 'error';
  status.textContent =
    'The machine-readable summary could not be loaded. The evidence index is still available on GitHub. ';
  const link = document.createElement('a');
  link.href = 'https://github.com/Tommyweige/breakout-rl-engineering/blob/main/docs/final-results/README.md';
  link.target = '_blank';
  link.rel = 'noreferrer';
  link.textContent = 'Open the results index';
  status.append(link);
  console.error('Failed to load final-results summary', error);
}

async function loadSummary(): Promise<void> {
  const response = await fetch('/final-results/summary.json', {
    headers: { Accept: 'application/json' },
    cache: 'no-cache',
  });
  if (!response.ok) {
    throw new Error(`summary request failed: ${response.status}`);
  }
  const summary = (await response.json()) as FinalResultsSummary;
  if (!summary || typeof summary !== 'object' || !Array.isArray(summary.metrics)) {
    throw new Error('summary payload does not match the expected schema');
  }
  hydrateCaseStudy(appRoot, summary);
}

void loadSummary().catch(showSummaryError);
