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
    expect(root.querySelectorAll('section[aria-labelledby]').length).toBeGreaterThanOrEqual(7);
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
    expect(text(root, '[data-metric-id="rgb_controller_pr26_max_score"]')).toBe('341');
    expect(text(root, '[data-metric-id="dqn_v3_2m_mean_score"]')).toBe('6.93');
    expect(text(root, '[data-metric-id="dqn_v3_2m_clears"]')).toBe('0');
    expect(text(root, '[data-metric-id="dqn_final_holdout_mean"]')).toBe('30.933');
    expect(text(root, '[data-metric-id="laya_complete_episodes"]')).toBe('0');
    expect(text(root, '[data-metric-id="clear_score_threshold"]')).toBe('864');
    expect(text(root, '[data-metric-id="dqn_training_transitions"]')).toBe('2,500,000');
    expect(text(root, '[data-seed-count-id="reward_shaping_mean_effect"]')).toBe('3');
    expect(text(root, '[data-contract-field="v2.frame_skip"]')).toBe('4');
    expect(text(root, '[data-contract-field="v2.agent_step_limit"]')).toBe('27,000');
    expect(text(root, '[data-contract-field="v2.max_num_frames_per_episode"]')).toBe('108,000');

    const unresolved = [...root.querySelectorAll<HTMLElement>('[data-metric-id]')].filter(
      (element) => (element.textContent ?? '').trim() === '—',
    );
    expect(unresolved.map((element) => element.dataset.metricId)).toEqual([]);
    expect(root.textContent).not.toContain('{{');
  });

  it('renders the results and contract tables from data, not hand-typed rows', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);

    const resultRows = root.querySelectorAll('[data-role="results-table"] tbody tr');
    expect(resultRows).toHaveLength(summary.table_metric_ids.length);
    expect(text(root, '[data-role="results-table"] thead')).toContain(copy.results.headers.metric);
    expect(text(root, '[data-role="results-table"] thead')).toContain(copy.results.headers.scope);
    expect(resultRows[0]?.querySelector('.cs-status')?.getAttribute('data-status')).toBe('verified');

    const contractRows = root.querySelectorAll('[data-role="contract-table"] tbody tr');
    expect(contractRows).toHaveLength(12);
    expect(root.querySelector('[data-role="contract-table"]')?.textContent).toContain(
      summary.contracts.v2.contract_id,
    );
    expect(root.querySelector('[data-role="contract-table"]')?.textContent).toContain(
      summary.contracts.v3.contract_id,
    );
  });

  it('marks the DQN versus controller comparison as not directly comparable', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);
    const note = text(root, '[data-role="comparison-note"]');
    expect(note).toBe(copy.fairness.comparisonNote.replace('{{contract:v2.frame_skip}}', '4'));
    expect(note).not.toContain('{{');
  });

  it('renders every experiment family with its verification status', () => {
    const root = mount(locale);
    hydrateCaseStudy(root, summary, locale);
    const items = root.querySelectorAll('.cs-ledger-item');
    expect(items).toHaveLength(summary.experiments.length);
    expect(items[0]?.querySelector('.cs-status')?.textContent?.trim()).toBe(
      copy.status.labels[summary.experiments[0]!.verification_status],
    );
    expect(text(root, '[data-role="summary-status"]')).toBe(
      copy.hero.summaryLoaded
        .replace('{metrics}', String(summary.metrics.length))
        .replace('{experiments}', String(summary.experiments.length)),
    );
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
  it('shows identical metric and contract values in both locales', () => {
    const en = mount('en');
    hydrateCaseStudy(en, summary, 'en');
    const zh = mount('zh-TW');
    hydrateCaseStudy(zh, summary, 'zh-TW');

    for (const element of en.querySelectorAll<HTMLElement>('[data-metric-id]')) {
      const id = element.dataset.metricId!;
      expect(text(zh, `[data-metric-id="${id}"]`)).toBe(element.textContent);
    }
    for (const element of en.querySelectorAll<HTMLElement>('[data-contract-field]')) {
      const field = element.dataset.contractField!;
      expect(text(zh, `[data-contract-field="${field}"]`)).toBe(element.textContent);
    }
  });

  it('uses localized table labels without duplicating numeric values', () => {
    const root = mount('zh-TW');
    hydrateCaseStudy(root, summary, 'zh-TW');
    expect(root.querySelector('[data-role="results-table"] tbody')?.textContent).toContain(
      '打穿兩面牆的回合數（a8445df）',
    );
    expect(root.querySelector('[data-role="results-table"]')?.textContent).toContain('30.933');
    expect(root.querySelectorAll('[data-role="contract-table"] th').length).toBeGreaterThan(0);
    expect(root.querySelector('[data-role="contract-table"] tbody')?.textContent).toContain('每 4 個影格 1 次');
  });

  it('falls back to the summary text for units and seed roles the locale does not translate', () => {
    const root = mount('zh-TW');
    const first = summary.metrics.find((metric) => metric.id === summary.table_metric_ids[0])!;
    const unknownUnit = 'quux per widget';
    const unknownSeedRole = 'seeds nobody translated';
    hydrateCaseStudy(
      root,
      {
        ...summary,
        metrics: summary.metrics.map((metric) =>
          metric.id === first.id
            ? { ...metric, unit: unknownUnit, seed_role: unknownSeedRole }
            : metric,
        ),
      },
      'zh-TW',
    );

    const table = root.querySelector('[data-role="results-table"]')?.textContent ?? '';
    expect(table).toContain(unknownUnit);
    expect(table).toContain(unknownSeedRole);
    expect(table).not.toContain('{{');
  });

  it('keeps measured values out of locale copy so both pages hydrate from summary.json', () => {
    const serialized = JSON.stringify(CASE_STUDY_COPY);
    for (const value of ['30.933', '137.47', '721.8', '6.93', '864', '2,500,000', '201M', '108,000', '27,000']) {
      expect(serialized).not.toContain(value);
    }
  });
});
