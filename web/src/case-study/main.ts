import './case-study.css';

import { CASE_STUDY_COPY, resolveLocale } from './content';
import {
  hydrateCaseStudy,
  renderCaseStudyShell,
  type FinalResultsSummary,
} from './renderCaseStudy';

const locale = resolveLocale(document.documentElement.lang);
const copy = CASE_STUDY_COPY[locale];

const root = document.querySelector<HTMLElement>('#app');
if (!root) {
  throw new Error('missing #app root');
}

const appRoot: HTMLElement = root;
appRoot.innerHTML = renderCaseStudyShell(locale);

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

// The table of contents is a sticky side list on wide screens and a collapsed
// panel above the story on phones, the same way the writing pages on tommypan.com work.
const tocDetails = appRoot.querySelector<HTMLDetailsElement>('[data-role="toc-details"]');
const wideScreen = window.matchMedia?.('(min-width: 1024px)');
if (tocDetails && wideScreen) {
  const syncToc = () => {
    tocDetails.open = wideScreen.matches;
  };
  syncToc();
  wideScreen.addEventListener?.('change', syncToc);
}

// Highlight the chapter being read in the side table of contents.
const tocLinks = new Map(
  [...appRoot.querySelectorAll<HTMLAnchorElement>('[data-toc-target]')].map((link) => [link.dataset.tocTarget, link]),
);
if ('IntersectionObserver' in window && tocLinks.size > 0) {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        tocLinks.forEach((link, id) => {
          if (id === entry.target.id) link.setAttribute('aria-current', 'true');
          else link.removeAttribute('aria-current');
        });
      }
    },
    { rootMargin: '-30% 0px -60% 0px' },
  );
  tocLinks.forEach((_link, id) => {
    const section = id ? document.getElementById(id) : null;
    if (section) observer.observe(section);
  });
}

function showSummaryError(error: unknown): void {
  appRoot.dataset.summaryState = 'error';
  const status = appRoot.querySelector<HTMLElement>('[data-role="summary-status"]');
  if (!status) return;
  status.hidden = false;
  status.dataset.state = 'error';
  status.textContent = copy.hero.summaryError;
  const link = document.createElement('a');
  link.href = 'https://github.com/Tommyweige/breakout-rl-engineering/blob/main/docs/final-results/README.md';
  link.target = '_blank';
  link.rel = 'noreferrer';
  link.textContent = copy.hero.summaryErrorLink;
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
  hydrateCaseStudy(appRoot, summary, locale);
}

void loadSummary().catch(showSummaryError);
