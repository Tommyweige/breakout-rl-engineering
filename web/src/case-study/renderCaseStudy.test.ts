// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import summaryJson from '../../../docs/final-results/summary.json';
import { CASE_STUDY_COPY, type Locale } from './content';
import { FACTS, TRACKER_EPISODES } from './facts';
import {
  formatMetricValue,
  hydrateCaseStudy,
  renderCaseStudyShell,
  type FinalResultsSummary,
} from './renderCaseStudy';

const summary = summaryJson as unknown as FinalResultsSummary;
const locales: Locale[] = ['en', 'zh-TW'];

function mount(locale: Locale): HTMLElement {
  document.body.innerHTML = `<div id="app">${renderCaseStudyShell(locale)}</div>`;
  return document.querySelector<HTMLElement>('#app')!;
}

function text(root: ParentNode, selector: string): string {
  return root.querySelector(selector)?.textContent?.trim() ?? '';
}

function metricValue(id: string): number {
  const value = summary.metrics.find((metric) => metric.id === id)?.value;
  if (typeof value !== 'number') throw new Error(`${id} is not numeric`);
  return value;
}

describe.each(locales)('Case study (%s)', (locale) => {
  const copy = CASE_STUDY_COPY[locale];

  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('renders the hero, timeline, and every chapter with a table of contents entry', () => {
    const root = mount(locale);
    expect(root.querySelectorAll('h1')).toHaveLength(1);
    expect(text(root, 'h1')).toBe(copy.hero.title);
    expect(root.querySelector('video source')?.getAttribute('src')).toBe('/case-study/tracker.mp4');
    expect(root.querySelector('video')?.getAttribute('poster')).toBe('/case-study/tracker-poster.png');
    expect(root.querySelectorAll('[data-role="hero-stats"] > div')).toHaveLength(3);
    expect(root.querySelectorAll('[data-role="timeline"] > li')).toHaveLength(copy.timeline.items.length);

    const tocTargets = [...root.querySelectorAll<HTMLElement>('[data-toc-target]')].map((link) => link.dataset.tocTarget);
    expect(tocTargets).toEqual(['timeline', ...copy.chapters.map((chapter) => chapter.id)]);
    for (const id of tocTargets) {
      expect(root.querySelector(`section#${id}[aria-labelledby]`), `section ${id}`).not.toBeNull();
    }
  });

  it('links to the other locale in the top-right switcher', () => {
    const root = mount(locale);
    const switcher = root.querySelector<HTMLAnchorElement>('[data-role="language-switch"]');
    expect(switcher?.getAttribute('href')).toBe(locale === 'en' ? '/case-study.zh-TW.html' : '/case-study.html');
    expect(switcher?.getAttribute('hreflang')).toBe(locale === 'en' ? 'zh-TW' : 'en');
    expect(switcher?.getAttribute('lang')).toBe(locale === 'en' ? 'zh-Hant-TW' : 'en');
  });

  it('hydrates every summary value and leaves nothing unresolved', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);

    expect(text(root, '[data-metric-id="rgb_controller_a8445df_canonical_clear_detections"]')).toBe('4');
    expect(text(root, '[data-seed-count-id="rgb_controller_a8445df_canonical_clear_detections"]')).toBe('15');
    expect(text(root, '[data-metric-id="rgb_controller_a8445df_mean_score"]')).toBe('721.8');
    expect(text(root, '[data-metric-id="dqn_v3_2m_mean_score"]')).toBe('6.93');
    expect(text(root, '[data-metric-id="dqn_final_holdout_mean"]')).toBe('30.933');
    expect(text(root, '[data-metric-id="clear_score_threshold"]')).toBe('864');
    expect(text(root, '[data-metric-id="laya_browser_session_score"]')).toBe('4');
    expect(text(root, '[data-metric-id="laya_browser_session_played_s"]')).toBe('22.7');
    expect(text(root, '[data-fact-id="day15_gap_steps"]')).toBe('16,395');

    const unresolved = [
      ...root.querySelectorAll<HTMLElement>('[data-metric-id], [data-seed-count-id], [data-fact-id]'),
    ]
      .filter((element) => (element.textContent ?? '').trim() === '—')
      .map((element) => element.dataset.metricId ?? element.dataset.seedCountId ?? element.dataset.factId);
    expect(unresolved).toEqual([]);
    expect(root.textContent).not.toContain('{{');
  });

  it('scales chart bars against their stated maximum', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);

    const shares = (id: string) =>
      [...root.querySelectorAll<HTMLElement>(`[data-value-ref="metric:${id}"]`)].map((fill) =>
        Number(fill.style.getPropertyValue('--share')),
      );
    for (const value of shares('rgb_controller_a8445df_mean_score')) expect(value).toBeCloseTo(721.8 / 864, 6);
    for (const value of shares('dqn_v3_2m_mean_score')) expect(value).toBeCloseTo(6.93 / 864, 6);
    // The training chart is scaled to the selection score, so the 5M column sits just below the top.
    const fiveMillion = root.querySelector<HTMLElement>('[data-value-ref="fact:dueling_5m"]');
    expect(Number(fiveMillion?.style.getPropertyValue('--share'))).toBeCloseTo(49.933 / 51.4, 6);
  });

  it('draws all 15 tracker games and marks exactly the cleared ones', () => {
    const root = mount(locale);
    const games = root.querySelectorAll<HTMLElement>('[data-role="episode-chart"] .cs-episode');
    expect(games).toHaveLength(15);
    expect([...games].filter((game) => game.dataset.cleared === 'true').map((game) => game.textContent?.trim().match(/\d+$/)?.[0])).toEqual([
      '6',
      '12',
      '14',
      '15',
    ]);
  });

  it('tells the story without the removed mix-up or researcher jargon', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);
    const rendered = (root.textContent ?? '').toLowerCase();

    for (const banned of ['137', '50ee352', 'a8445df', 'canonical', 'seed', 'frame_skip', 'contract', '契約', '種子', 'summary.json', '.py', 'python -m', 'npm ', 'onnx runtime web']) {
      expect(rendered, `${locale} mentions ${banned}`).not.toContain(banned);
    }
    for (const contract of [summary.contracts.v2, summary.contracts.v3]) {
      expect(rendered).not.toContain(contract.contract_id);
      expect(rendered).not.toContain(contract.sha256);
    }
    expect(rendered).not.toMatch(/#\d+\b/);
  });

  it('leaves the summary status line hidden until loading actually fails', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);
    const status = root.querySelector<HTMLElement>('[data-role="summary-status"]');
    expect(status?.hasAttribute('hidden')).toBe(true);
    expect(text(root, '[data-role="frozen-date"]')).toBe(summary.frozen_at);
  });

  it('formats null, boolean, integer, and string metric values without inventing data', () => {
    const metric = summary.metrics[0]!;
    expect(formatMetricValue({ ...metric, value: null }, locale)).toBe(copy.common.notRun);
    expect(formatMetricValue({ ...metric, value: true }, locale)).toBe(copy.common.yes);
    expect(formatMetricValue({ ...metric, value: false }, locale)).toBe(copy.common.no);
    expect(formatMetricValue({ ...metric, value: 1234567 }, locale)).toBe('1,234,567');
    expect(formatMetricValue({ ...metric, value: '68.36-69.48' }, locale)).toBe('68.36-69.48');
  });
});

describe('Cross-locale parity', () => {
  it('shows the same numbers, from the same sources, in both locales', () => {
    const en = mount('en');
    hydrateCaseStudy(en, summary, 'en');
    const enValues = [...en.querySelectorAll<HTMLElement>('[data-metric-id], [data-fact-id]')].map(
      (element) => `${element.dataset.metricId ?? element.dataset.factId}=${element.textContent}`,
    );
    const zh = mount('zh-TW');
    hydrateCaseStudy(zh, summary, 'zh-TW');
    const zhValues = [...zh.querySelectorAll<HTMLElement>('[data-metric-id], [data-fact-id]')].map(
      (element) => `${element.dataset.metricId ?? element.dataset.factId}=${element.textContent}`,
    );
    expect(new Set(zhValues)).toEqual(new Set(enValues));
  });

  it('keeps the two locales structurally in step', () => {
    const [en, zh] = [CASE_STUDY_COPY.en, CASE_STUDY_COPY['zh-TW']];
    expect(zh.chapters.map((chapter) => chapter.id)).toEqual(en.chapters.map((chapter) => chapter.id));
    expect(zh.timeline.items.map((item) => item.days.replace('30 天之後', 'After day 30'))).toEqual(
      en.timeline.items.map((item) => item.days),
    );
    zh.chapters.forEach((chapter, index) => {
      expect(chapter.blocks.map((block) => block.kind)).toEqual(en.chapters[index]!.blocks.map((block) => block.kind));
    });
  });

  it('keeps every literal `#` in the copy, so issue and PR numbers reach the DOM intact', () => {
    // The page itself prints no issue or PR numbers, so the guarantee is pinned
    // by rendering a kicker that does contain them.
    const kicker = '首次通關探測 · #41、#43、#45、#48–#61';
    const chapter = CASE_STUDY_COPY['zh-TW'].chapters[0]!;
    const originalKicker = chapter.kicker;
    chapter.kicker = kicker;
    try {
      const root = mount('zh-TW');
      const rendered = root.querySelector(`#${chapter.id} .cs-kicker`)?.textContent?.replace(/^\d+/, '') ?? '';
      expect(rendered).toBe(kicker);
    } finally {
      chapter.kicker = originalKicker;
    }
  });

  it('keeps measured values out of locale copy so both pages hydrate them from one source', () => {
    const serialized = JSON.stringify(CASE_STUDY_COPY);
    for (const value of ['30.933', '137.47', '721.8', '6.93', '864', '2,500,000', '201M', '108,000', '27,000', '16,395']) {
      expect(serialized).not.toContain(value);
    }
  });
});

describe('Story facts', () => {
  it('agree with the summary wherever the summary has the same number', () => {
    const cleared = TRACKER_EPISODES.filter((episode) => episode.cleared);
    expect(cleared).toHaveLength(metricValue('rgb_controller_a8445df_canonical_clear_detections'));
    expect(TRACKER_EPISODES).toHaveLength(15);
    const mean = TRACKER_EPISODES.reduce((total, episode) => total + episode.score, 0) / TRACKER_EPISODES.length;
    expect(mean).toBeCloseTo(metricValue('rgb_controller_a8445df_mean_score'), 1);
    expect(Math.max(...TRACKER_EPISODES.map((episode) => episode.score))).toBe(metricValue('clear_score_threshold'));
    expect(cleared.every((episode) => episode.score === metricValue('clear_score_threshold'))).toBe(true);
    expect(FACTS.family_dueling_500k?.value).toBe('20.178');
    // Day 15: the gap between losing a life and the next serve.
    expect(Number(FACTS.day15_first_fire_step!.value.replace(',', '')) - Number(FACTS.day15_life_lost_step!.value)).toBe(
      Number(FACTS.day15_gap_steps!.value.replace(',', '')),
    );
  });
});
