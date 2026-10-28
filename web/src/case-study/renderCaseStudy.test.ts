// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import summaryJson from '../../../docs/final-results/summary.json';
import {
  formatMetricValue,
  hydrateCaseStudy,
  renderCaseStudyShell,
  type FinalResultsSummary,
} from './renderCaseStudy';

const summary = summaryJson as unknown as FinalResultsSummary;

function mount(): HTMLElement {
  document.body.innerHTML = `<div id="app">${renderCaseStudyShell()}</div>`;
  return document.querySelector<HTMLElement>('#app')!;
}

function text(root: ParentNode, selector: string): string {
  return root.querySelector(selector)?.textContent?.trim() ?? '';
}

describe('Research results case study', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('renders an accessible hero with a real gameplay recording', () => {
    const root = mount();
    expect(root.querySelectorAll('h1')).toHaveLength(1);
    expect(text(root, 'h1')).toContain('Does Breakout');
    expect(root.querySelector('video[data-role="gameplay-video"]')).not.toBeNull();
    expect(root.querySelector('video source')?.getAttribute('src')).toBe('/case-study/gameplay.mp4');
    expect(root.querySelector('video')?.getAttribute('poster')).toBe('/case-study/gameplay-poster.png');
    expect(root.querySelectorAll('section[aria-labelledby]').length).toBeGreaterThanOrEqual(7);
  });

  it('hydrates every headline value from the machine-readable summary', () => {
    const root = mount();
    hydrateCaseStudy(root, summary);

    expect(text(root, '[data-metric-id="verified_clears_recorded"]')).toBe('0');
    expect(text(root, '[data-metric-id="dqn_final_holdout_mean"]')).toBe('30.933');
    expect(text(root, '[data-metric-id="rgb_controller_mean_score"]')).toBe('137.47');
    expect(text(root, '[data-metric-id="laya_complete_episodes"]')).toBe('0');
    expect(text(root, '[data-metric-id="clear_score_threshold"]')).toBe('864');
    expect(text(root, '[data-metric-id="dqn_training_transitions"]')).toBe('2,500,000');

    const unresolved = [...root.querySelectorAll<HTMLElement>('[data-metric-id]')].filter(
      (element) => (element.textContent ?? '').trim() === '—',
    );
    expect(unresolved.map((element) => element.dataset.metricId)).toEqual([]);
  });

  it('renders the results and contract tables from data, not hand-typed rows', () => {
    const root = mount();
    hydrateCaseStudy(root, summary);

    const resultRows = root.querySelectorAll('[data-role="results-table"] tbody tr');
    expect(resultRows).toHaveLength(summary.table_metric_ids.length);
    expect(resultRows[0]?.textContent).toContain('Verified full clears');
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
    const root = mount();
    hydrateCaseStudy(root, summary);
    const note = text(root, '[data-role="comparison-note"]');
    expect(note).toContain('not directly comparable');
    expect(note).toContain('Contract v2 makes one decision per four emulator frames');
  });

  it('renders every experiment family with its verification status', () => {
    const root = mount();
    hydrateCaseStudy(root, summary);
    const items = root.querySelectorAll('.cs-ledger-item');
    expect(items).toHaveLength(summary.experiments.length);
    expect(items[0]?.querySelector('.cs-status')?.textContent).toBeTruthy();
    expect(text(root, '[data-role="summary-status"]')).toContain(
      `${summary.metrics.length} metrics`,
    );
  });

  it('formats null, integer, and string metric values without inventing data', () => {
    expect(
      formatMetricValue({
        ...summary.metrics[0]!,
        value: null,
      }),
    ).toBe('Not run');
    expect(
      formatMetricValue({
        ...summary.metrics[0]!,
        value: 1234567,
      }),
    ).toBe('1,234,567');
    expect(
      formatMetricValue({
        ...summary.metrics[0]!,
        value: '68.36-69.48',
      }),
    ).toBe('68.36-69.48');
  });
});
