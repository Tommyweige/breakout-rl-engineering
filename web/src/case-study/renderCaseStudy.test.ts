// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import summaryJson from '../../../docs/final-results/summary.json';
import { CASE_STUDY_COPY, type Locale } from './content';
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

describe.each(locales)('Research results case study (%s)', (locale) => {
  const copy = CASE_STUDY_COPY[locale];

  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('renders an accessible hero with a real gameplay recording', () => {
    const root = mount(locale);
    expect(root.querySelectorAll('h1')).toHaveLength(1);
    expect(text(root, 'h1')).toContain(copy.hero.titleLead);
    expect(text(root, 'h1')).toContain(copy.hero.titleTail);
    expect(root.querySelector('video[data-role="gameplay-video"]')).not.toBeNull();
    expect(root.querySelector('video source')?.getAttribute('src')).toBe('/case-study/gameplay.mp4');
    expect(root.querySelector('video')?.getAttribute('poster')).toBe('/case-study/gameplay-poster.png');
    expect(root.querySelectorAll('section[aria-labelledby]').length).toBeGreaterThanOrEqual(6);
  });

  it('links to the other locale in the top-right switcher', () => {
    const root = mount(locale);
    const switcher = root.querySelector<HTMLAnchorElement>('[data-role="language-switch"]');
    expect(switcher).not.toBeNull();
    expect(switcher?.getAttribute('href')).toBe(
      locale === 'en' ? '/case-study.zh-TW.html' : '/case-study.html',
    );
    expect(switcher?.getAttribute('hreflang')).toBe(locale === 'en' ? 'zh-TW' : 'en');
    expect(switcher?.getAttribute('lang')).toBe(locale === 'en' ? 'zh-Hant-TW' : 'en');
    expect(switcher?.textContent?.trim()).toBe(copy.header.languageSwitch.label);
  });

  it('hydrates every headline value from the machine-readable summary', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);

    expect(text(root, '[data-metric-id="rgb_controller_a8445df_canonical_clear_detections"]')).toBe('4');
    expect(text(root, '[data-seed-count-id="rgb_controller_a8445df_canonical_clear_detections"]')).toBe('15');
    expect(text(root, '[data-metric-id="rgb_controller_a8445df_mean_score"]')).toBe('721.8');
    expect(text(root, '[data-metric-id="rgb_controller_pr26_mean_score"]')).toBe('137.47');
    expect(text(root, '[data-metric-id="dqn_v3_2m_mean_score"]')).toBe('6.93');
    expect(text(root, '[data-metric-id="dqn_v3_2m_clears"]')).toBe('0');
    expect(text(root, '[data-metric-id="laya_complete_episodes"]')).toBe('0');
    expect(text(root, '[data-metric-id="clear_score_threshold"]')).toBe('864');
    expect(text(root, '[data-metric-id="dqn_training_transitions"]')).toBe('2,500,000');

    const unresolved = [...root.querySelectorAll<HTMLElement>('[data-metric-id], [data-seed-count-id]')]
      .filter((element) => (element.textContent ?? '').trim() === '—')
      .map((element) => element.dataset.metricId ?? element.dataset.seedCountId);
    expect(unresolved).toEqual([]);
    expect(root.textContent).not.toContain('{{');
  });

  it('renders the three-way comparison as a four-column table', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);

    const table = root.querySelector('[data-role="results-table"]');
    expect(table).not.toBeNull();
    const rows = table!.querySelectorAll('tbody tr');
    expect(rows).toHaveLength(3);
    expect(table!.querySelectorAll('thead th')).toHaveLength(4);
    expect(text(table!, 'thead')).toContain(copy.results.headers.method);
    expect(text(table!, 'thead')).toContain(copy.results.headers.trained);

    expect(rows[0]!.querySelector('th')?.textContent?.trim()).toBe(copy.results.rows[0]!.method);
    expect(rows[0]!.querySelector('[data-metric-id="rgb_controller_a8445df_mean_score"]')?.textContent).toBe(
      '721.8',
    );
    expect(rows[1]!.querySelector('[data-metric-id="dqn_v3_2m_mean_score"]')?.textContent).toBe('6.93');
  });

  it('keeps the machine-readable ledger and contract identifiers off the page', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);
    const rendered = root.textContent ?? '';

    for (const contract of [summary.contracts.v2, summary.contracts.v3]) {
      expect(rendered).not.toContain(contract.contract_id);
      expect(rendered).not.toContain(contract.sha256);
    }
    for (const pattern of [/\ba8445df\b/, /\b50ee352\b/, /#\d+\b/, /summary\.json/]) {
      expect(rendered, `${locale} leaked ${pattern}`).not.toMatch(pattern);
    }
    expect(root.querySelector('[data-role="experiment-list"]')).toBeNull();
    expect(root.querySelector('[data-role="contract-table"]')).toBeNull();
  });

  it('leaves the summary status line hidden until loading actually fails', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);
    const status = root.querySelector<HTMLElement>('[data-role="summary-status"]');
    expect(status?.hasAttribute('hidden')).toBe(true);
    expect(status?.textContent?.trim()).toBe('');
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
  it('shows identical metric values in both locales', () => {
    const en = mount('en');
    hydrateCaseStudy(en, summary, 'en');
    const zh = mount('zh-TW');
    hydrateCaseStudy(zh, summary, 'zh-TW');

    for (const element of en.querySelectorAll<HTMLElement>('[data-metric-id]')) {
      const id = element.dataset.metricId!;
      expect(text(zh, `[data-metric-id="${id}"]`)).toBe(element.textContent);
    }
  });

  it('keeps every literal `#` in the copy, so issue and PR numbers reach the DOM intact', () => {
    // Templates are split at their hydration tokens and inline HTML tags are
    // dropped, so what remains must appear verbatim in the rendered text. A
    // markdown-style pass that treated `#` as a heading marker would delete
    // these fragments, which is exactly what this test pins down.
    const fragmentsOf = (copy: unknown): string[] => {
      const fragments: string[] = [];
      const walk = (value: unknown): void => {
        if (typeof value === 'string') {
          for (const chunk of value.replaceAll(/<[^>]+>/g, '').split(/\{\{[^}]+\}\}/)) {
            if (chunk.includes('#')) fragments.push(chunk);
          }
          return;
        }
        if (value && typeof value === 'object') Object.values(value).forEach(walk);
      };
      walk(copy);
      return fragments;
    };

    // The page itself no longer prints issue or PR numbers, so the guarantee is
    // pinned by rendering a kicker that does contain them.
    const kicker = '首次通關探測 · #41、#43、#45、#48–#61';
    const copy = CASE_STUDY_COPY['zh-TW'];
    const originalKicker = copy.results.kicker;
    copy.results.kicker = kicker;
    try {
      const root = mount('zh-TW');
      hydrateCaseStudy(root, summary, 'zh-TW');
      const heading = root.querySelector('#results-title');
      expect(heading?.textContent).toBe(copy.results.title);
      const renderedKicker = heading?.closest('div')?.querySelector('.cs-kicker')?.textContent ?? '';
      expect(renderedKicker).toBe(kicker);
      expect(renderedKicker).not.toContain('#41、、、–');
      expect(fragmentsOf(copy)).toContain(kicker);
    } finally {
      copy.results.kicker = originalKicker;
    }
  });

  it('keeps measured values out of locale copy so both pages hydrate from summary.json', () => {
    const serialized = JSON.stringify(CASE_STUDY_COPY);
    for (const value of ['30.933', '137.47', '721.8', '6.93', '864', '2,500,000', '201M', '108,000', '27,000']) {
      expect(serialized).not.toContain(value);
    }
  });
});
