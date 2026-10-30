import { CASE_STUDY_COPY, type CaseStudyCopy, type Locale } from './content';
import type { Chapter, NumberRef } from './content/types';
import { FACTS, TRACKER_EPISODES, TRACKER_EPISODE_FIRST_WALL, TRACKER_EPISODE_MAX } from './facts';

export type VerificationStatus =
  | 'verified'
  | 'historical_reported'
  | 'missing_raw_artifacts'
  | 'not_yet_evaluated';

export type RawArtifactStatus = 'in_repo' | 'external_branch' | 'missing' | 'not_applicable';

export interface SummaryContractRef {
  id: string;
  path: string;
  frame_skip: number | 'mixed';
}

export interface SummaryMetric {
  id: string;
  label: string;
  value: number | string | boolean | null;
  unit: string;
  contract: SummaryContractRef;
  seed_count: number;
  seed_role?: string;
  source: string;
  ref: string;
  verification_status: VerificationStatus;
  raw_artifact_status?: RawArtifactStatus;
  notes?: string;
}

export interface SummaryContract {
  contract_id: string;
  path: string;
  sha256: string;
  environment_id: string;
  frame_skip: number;
  frame_stack: number;
  sticky_action_probability: number;
  fire_reset: boolean;
  terminal_on_life_loss: boolean;
  max_num_frames_per_episode: number;
  agent_step_limit: number;
  evaluation_epsilon: number;
  evaluation_seeds: number[];
  raw_reward_rule: string;
}

export interface SummaryComparison {
  id: string;
  title: string;
  left: string;
  right: string;
  comparability: string;
  reason: string;
  evidence: string;
}

export interface SummaryExperiment {
  id: string;
  title: string;
  method: string;
  contract: string;
  seed_summary: string;
  headline: string;
  verification_status: VerificationStatus;
  raw_artifact_status: RawArtifactStatus;
  source: string;
  notes: string;
}

export interface FinalResultsSummary {
  schema_version: number;
  frozen_at: string;
  project_status: string;
  research_question: string;
  contracts: { v2: SummaryContract; v3: SummaryContract };
  headline: {
    canonical_clear_detections: number;
    canonical_clear_episodes: number;
    statement: string;
    scope: string;
    clear_definition: string;
    max_observed_probe_score: number;
  };
  metrics: SummaryMetric[];
  table_metric_ids: string[];
  comparisons: SummaryComparison[];
  experiments: SummaryExperiment[];
  non_claims: string[];
  links: {
    demo: string;
    case_study: string;
    repository: string;
    results_index: string;
    handoff: string;
    evidence_inventory: string;
    issues: Record<string, string>;
  };
}

const TOKEN_PATTERN = /\{\{(metric|seeds|fact):([A-Za-z0-9_.-]+)\}\}/g;

const REPOSITORY_URL = 'https://github.com/Tommyweige/breakout-rl-engineering';
const RESULTS_URL = `${REPOSITORY_URL}/blob/main/docs/final-results/README.md`;

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function metricSpan(id: string, fallback = '—'): string {
  return `<span data-metric-id="${escapeHtml(id)}">${escapeHtml(fallback)}</span>`;
}

function factSpan(id: string, copy: CaseStudyCopy): string {
  const fact = FACTS[id];
  return `<span data-fact-id="${escapeHtml(id)}"${fact ? ` title="${escapeHtml(fact.source)}"` : ''}>${escapeHtml(
    fact?.value ?? copy.common.missingMetric,
  )}</span>`;
}

/** Copy templates are trusted markup; only their tokens are replaced. */
export function renderCopy(template: string, copy: CaseStudyCopy): string {
  return template.replace(TOKEN_PATTERN, (_match, kind: string, id: string) => {
    if (kind === 'metric') return metricSpan(id, copy.common.missingMetric);
    if (kind === 'fact') return factSpan(id, copy);
    return `<span data-seed-count-id="${escapeHtml(id)}">${escapeHtml(copy.common.missingMetric)}</span>`;
  });
}

function renderRef(ref: NumberRef, copy: CaseStudyCopy): string {
  const [kind, id] = ref.split(':') as [string, string];
  return kind === 'fact' ? factSpan(id, copy) : metricSpan(id, copy.common.missingMetric);
}

function refNumber(ref: string, metrics?: Map<string, SummaryMetric>): number | null {
  const [kind, id = ''] = ref.split(':');
  if (kind === 'fact') {
    const value = Number(FACTS[id]?.value.replaceAll(',', ''));
    return Number.isFinite(value) ? value : null;
  }
  const value = metrics?.get(id)?.value;
  return typeof value === 'number' ? value : null;
}

function share(valueRef: string, maxRef: string, metrics?: Map<string, SummaryMetric>): number | null {
  const value = refNumber(valueRef, metrics);
  const max = refNumber(maxRef, metrics);
  if (value === null || max === null || max <= 0) return null;
  return Math.min(1, Math.max(0, value / max));
}

function shareStyle(valueRef: string, maxRef: string): string {
  const value = share(valueRef, maxRef);
  return value === null ? '' : ` style="--share: ${value}"`;
}

const tone = (value?: string) => escapeHtml(value ?? 'plain');

type Block = Chapter['blocks'][number];

function figure(className: string, caption: string, body: string, copy: CaseStudyCopy): string {
  return `
    <figure class="cs-figure ${className}">
      ${body}
      <figcaption>${renderCopy(caption, copy)}</figcaption>
    </figure>`;
}

function renderBlock(block: Block, copy: CaseStudyCopy): string {
  switch (block.kind) {
    case 'p':
      return `<p class="cs-p">${renderCopy(block.text, copy)}</p>`;

    case 'callout':
      return `
        <aside class="cs-callout" data-tone="${tone(block.tone)}">
          <p class="cs-callout-title">${escapeHtml(block.title)}</p>
          <p>${renderCopy(block.text, copy)}</p>
        </aside>`;

    case 'flow':
      return figure(
        'cs-flow-figure',
        block.caption,
        `<ol class="cs-flow">${block.steps
          .map(
            (step, index) => `
          <li class="cs-flow-step">
            <span class="cs-flow-index">${index + 1}</span>
            <span class="cs-flow-label">${escapeHtml(step.label)}</span>
            <span class="cs-flow-detail">${renderCopy(step.detail, copy)}</span>
          </li>`,
          )
          .join('')}</ol>`,
        copy,
      );

    case 'table': {
      const numeric = new Set(block.numeric ?? []);
      const cell = (value: string, index: number, tag: 'td' | 'th') =>
        `<${tag}${numeric.has(index) ? ' class="cs-num"' : ''}${tag === 'th' && index === 0 ? ' scope="row"' : ''}>${renderCopy(value, copy)}</${tag}>`;
      return figure(
        'cs-table-figure',
        block.caption,
        `<div class="cs-table-wrap"><table class="cs-table" data-cols="${block.headers.length}">
          <thead><tr>${block.headers
            .map((header, index) => `<th scope="col"${numeric.has(index) ? ' class="cs-num"' : ''}>${escapeHtml(header)}</th>`)
            .join('')}</tr></thead>
          <tbody>${block.rows
            .map((row) => `<tr>${row.map((value, index) => cell(value, index, index === 0 ? 'th' : 'td')).join('')}</tr>`)
            .join('')}</tbody>
        </table></div>`,
        copy,
      );
    }

    case 'bars':
      return figure(
        'cs-bars-figure',
        block.caption,
        `<div class="cs-bars" data-role="bar-chart" data-max-ref="${escapeHtml(block.max)}">
          ${block.items
            .map(
              (item) => `
            <div class="cs-bar" data-tone="${tone(item.tone)}">
              <p class="cs-bar-label">${escapeHtml(item.label)}${
                item.note ? `<small>${escapeHtml(item.note)}</small>` : ''
              }</p>
              <p class="cs-bar-value">${renderRef(item.value, copy)}</p>
              <div class="cs-bar-track" aria-hidden="true">
                <div class="cs-bar-fill" data-value-ref="${escapeHtml(item.value)}"${shareStyle(item.value, block.max)}></div>
              </div>
            </div>`,
            )
            .join('')}
          ${block.maxLabel ? `<p class="cs-bar-scale" aria-hidden="true">${escapeHtml(block.maxLabel)} ${renderRef(block.max, copy)}</p>` : ''}
        </div>`,
        copy,
      );

    case 'columns':
      return figure(
        'cs-columns-figure',
        block.caption,
        `<div class="cs-columns" data-role="column-chart" data-max-ref="${escapeHtml(block.max)}">
          ${block.items
            .map(
              (item) => `
            <div class="cs-column" data-tone="${tone(item.tone)}">
              <div class="cs-column-track">
                <p class="cs-column-value">${renderRef(item.value, copy)}</p>
                <div class="cs-column-fill" aria-hidden="true" data-value-ref="${escapeHtml(item.value)}"${shareStyle(item.value, block.max)}></div>
              </div>
              <p class="cs-column-label">${escapeHtml(item.label)}</p>
            </div>`,
            )
            .join('')}
        </div>
        ${block.note ? `<p class="cs-figure-note">${escapeHtml(block.note)}</p>` : ''}`,
        copy,
      );

    case 'episodes': {
      const firstWall = TRACKER_EPISODE_FIRST_WALL / TRACKER_EPISODE_MAX;
      return figure(
        'cs-episodes-figure',
        block.caption,
        `<div class="cs-episodes" data-role="episode-chart">
          <div class="cs-episodes-plot">
          <div class="cs-episodes-guide" style="--at: 1"><span>${escapeHtml(block.bothWallsLabel)} ${TRACKER_EPISODE_MAX}</span></div>
          <div class="cs-episodes-guide" style="--at: ${firstWall}"><span>${escapeHtml(block.firstWallLabel)} ${TRACKER_EPISODE_FIRST_WALL}</span></div>
          <ol class="cs-episodes-list">
            ${TRACKER_EPISODES.map((episode, index) => {
              const label = block.gameLabel.replace('{n}', String(index + 1));
              return `
              <li class="cs-episode" data-cleared="${episode.cleared}" style="--share: ${episode.score / TRACKER_EPISODE_MAX}">
                <span class="cs-episode-bar" aria-hidden="true"></span>
                <span class="cs-episode-score">${episode.score}</span>
                <span class="cs-episode-label"><span class="cs-sr">${escapeHtml(label)}${
                  episode.cleared ? `, ${escapeHtml(block.clearedLabel)}` : ''
                }</span><span aria-hidden="true">${index + 1}</span></span>
              </li>`;
            }).join('')}
          </ol>
          </div>
          <p class="cs-episodes-legend"><span class="cs-swatch" aria-hidden="true"></span>${escapeHtml(block.clearedLabel)}</p>
        </div>`,
        copy,
      );
    }

    case 'stats':
      return `
        <dl class="cs-stats" data-count="${block.items.length}">
          ${block.items
            .map(
              (item) => `
            <div class="cs-stat" data-tone="${tone(item.tone)}">
              <dt>${escapeHtml(item.label)}</dt>
              <dd>${renderCopy(item.value, copy)}</dd>
            </div>`,
            )
            .join('')}
        </dl>`;

    case 'code':
      return figure('cs-code-figure', block.caption, `<pre class="cs-code"><code>${escapeHtml(block.code)}</code></pre>`, copy);

    case 'ledger':
      return `
        <div class="cs-ledger">
          ${block.items
            .map(
              (item, index) => `
            <article class="cs-ledger-item">
              <p class="cs-ledger-index">${String(index + 1).padStart(2, '0')}</p>
              <h3>${escapeHtml(item.name)}</h3>
              <p class="cs-ledger-idea">${renderCopy(item.idea, copy)}</p>
              <p class="cs-ledger-result">${escapeHtml(item.result)}</p>
              <p class="cs-ledger-detail">${renderCopy(item.detail, copy)}</p>
            </article>`,
            )
            .join('')}
        </div>`;

    case 'links':
      return `
        <ul class="cs-links" data-role="evidence-links">
          ${block.items
            .map((item) => {
              const external = item.href.startsWith('http');
              return `
            <li><a href="${escapeHtml(item.href)}"${external ? ' target="_blank" rel="noreferrer"' : ''}>
              <span class="cs-link-label">${escapeHtml(item.label)} <span aria-hidden="true">→</span></span>
              <span class="cs-link-detail">${escapeHtml(item.detail)}</span>
            </a></li>`;
            })
            .join('')}
        </ul>`;
  }
}

function renderChapter(chapter: Chapter, index: number, copy: CaseStudyCopy): string {
  const titleId = `${chapter.id}-title`;
  return `
    <section class="cs-chapter" id="${escapeHtml(chapter.id)}" aria-labelledby="${escapeHtml(titleId)}">
      <header class="cs-chapter-head">
        <p class="cs-kicker"><span class="cs-index">${String(index + 1).padStart(2, '0')}</span>${escapeHtml(chapter.kicker)}</p>
        <h2 id="${escapeHtml(titleId)}">${escapeHtml(chapter.title)}</h2>
      </header>
      ${chapter.blocks.map((block) => renderBlock(block, copy)).join('')}
    </section>`;
}

export function renderCaseStudyShell(locale: Locale): string {
  const copy = CASE_STUDY_COPY[locale];
  const languageSwitch = `
    <a
      class="cs-lang"
      data-role="language-switch"
      href="${escapeHtml(copy.header.languageSwitch.href)}"
      hreflang="${escapeHtml(copy.header.languageSwitch.hreflang)}"
      lang="${escapeHtml(copy.header.languageSwitch.lang)}"
      aria-label="${escapeHtml(copy.header.languageSwitch.ariaLabel)}"
    >${escapeHtml(copy.header.languageSwitch.label)}</a>`;
  const toc = [
    { id: 'timeline', nav: copy.timeline.nav },
    ...copy.chapters.map((chapter) => ({ id: chapter.id, nav: chapter.nav })),
  ];

  return `
    <header class="cs-header">
      <a class="cs-brand" href="/" aria-label="${escapeHtml(copy.header.brandHomeLabel)}">${escapeHtml(copy.header.brandMark)}</a>
      <nav class="cs-nav" aria-label="${escapeHtml(copy.header.navLabel)}">
        <span class="cs-nav-track">
          <a href="/">${escapeHtml(copy.header.playDemo)}</a>
          <a href="${REPOSITORY_URL}" target="_blank" rel="noreferrer">${escapeHtml(copy.header.github)}</a>
        </span>
        ${languageSwitch}
      </nav>
    </header>

    <main id="main">
      <section class="cs-hero" aria-labelledby="hero-title">
        <div class="cs-hero-copy">
          <p class="cs-kicker cs-hero-kicker">${escapeHtml(copy.hero.kicker)}</p>
          <h1 id="hero-title">${escapeHtml(copy.hero.title)}</h1>
          <p class="cs-lede">${escapeHtml(copy.hero.lede)}</p>
          <dl class="cs-hero-stats" data-role="hero-stats">
            ${copy.hero.stats
              .map(
                (stat) => `
              <div class="cs-hero-stat" data-tone="${tone(stat.tone)}">
                <dt>${escapeHtml(stat.label)}</dt>
                <dd>${renderCopy(stat.value, copy)}</dd>
              </div>`,
              )
              .join('')}
          </dl>
          <div class="cs-actions">
            <a class="cs-button" href="/">${escapeHtml(copy.hero.playCta)} <span aria-hidden="true">→</span></a>
            <a class="cs-button cs-button-ghost" href="#timeline">${escapeHtml(copy.hero.storyCta)} <span aria-hidden="true">↓</span></a>
          </div>
          <p class="cs-summary-status" data-role="summary-status" role="status" hidden></p>
        </div>
        <figure class="cs-hero-media">
          <video
            class="cs-video"
            data-role="gameplay-video"
            poster="/case-study/tracker-poster.png"
            muted
            loop
            playsinline
            preload="metadata"
            aria-label="${escapeHtml(copy.hero.videoAria)}"
          >
            <source src="/case-study/tracker.mp4" type="video/mp4" />
            ${escapeHtml(copy.hero.videoFallback)}
          </video>
          <figcaption>${escapeHtml(copy.hero.caption)}</figcaption>
        </figure>
      </section>

      <div class="cs-layout">
        <nav class="cs-toc" data-role="toc" aria-label="${escapeHtml(copy.header.tocLabel)}">
          <details class="cs-toc-details" data-role="toc-details" open>
            <summary class="cs-toc-title">${escapeHtml(copy.header.tocLabel)}</summary>
            <ol>
              ${toc.map((item) => `<li><a href="#${escapeHtml(item.id)}" data-toc-target="${escapeHtml(item.id)}">${escapeHtml(item.nav)}</a></li>`).join('')}
            </ol>
          </details>
        </nav>

        <div class="cs-content">
          <section class="cs-chapter cs-timeline-section" id="timeline" aria-labelledby="timeline-title">
            <header class="cs-chapter-head">
              <p class="cs-kicker">${escapeHtml(copy.timeline.kicker)}</p>
              <h2 id="timeline-title">${escapeHtml(copy.timeline.title)}</h2>
            </header>
            <ol class="cs-timeline" data-role="timeline">
              ${copy.timeline.items
                .map(
                  (item) => `
                <li class="cs-milestone" data-tone="${tone(item.tone)}">
                  <p class="cs-milestone-days">${escapeHtml(item.days)}</p>
                  <h3>${escapeHtml(item.title)}</h3>
                  <p>${renderCopy(item.text, copy)}</p>
                </li>`,
                )
                .join('')}
            </ol>
          </section>

          ${copy.chapters.map((chapter, index) => renderChapter(chapter, index, copy)).join('')}
        </div>
      </div>
    </main>

    <footer class="cs-footer">
      <p>${escapeHtml(copy.footer.updated)} <span data-role="frozen-date">—</span> · <a href="${RESULTS_URL}" target="_blank" rel="noreferrer">GitHub</a></p>
      ${languageSwitch}
    </footer>
  `;
}

export function formatMetricValue(metric: SummaryMetric, locale: Locale = 'en'): string {
  const copy = CASE_STUDY_COPY[locale];
  if (metric.value === null) return copy.common.notRun;
  if (typeof metric.value === 'boolean') return metric.value ? copy.common.yes : copy.common.no;
  if (typeof metric.value === 'number') {
    if (Number.isInteger(metric.value)) return metric.value.toLocaleString('en-US');
    return String(metric.value);
  }
  return metric.value;
}

export function hydrateCaseStudy(
  root: HTMLElement,
  summary: FinalResultsSummary,
  locale: Locale = 'en',
): void {
  const copy = CASE_STUDY_COPY[locale];
  const metrics = new Map(summary.metrics.map((metric) => [metric.id, metric]));

  root.querySelectorAll<HTMLElement>('[data-metric-id]').forEach((element) => {
    const metric = metrics.get(element.dataset.metricId ?? '');
    if (!metric) {
      element.textContent = copy.common.missingMetric;
      return;
    }
    element.textContent = formatMetricValue(metric, locale);
    element.title = `${metric.label}: ${metric.source}`;
  });

  root.querySelectorAll<HTMLElement>('[data-seed-count-id]').forEach((element) => {
    const metric = metrics.get(element.dataset.seedCountId ?? '');
    if (!metric) {
      element.textContent = copy.common.unknown;
      return;
    }
    element.textContent = metric.seed_count.toLocaleString('en-US');
  });

  root.querySelectorAll<HTMLElement>('[data-max-ref]').forEach((group) => {
    const maxRef = group.dataset.maxRef ?? '';
    group.querySelectorAll<HTMLElement>('[data-value-ref]').forEach((fill) => {
      const value = share(fill.dataset.valueRef ?? '', maxRef, metrics);
      fill.style.setProperty('--share', String(value ?? 0));
    });
  });

  root.querySelectorAll<HTMLElement>('[data-role="frozen-date"]').forEach((element) => {
    element.textContent = summary.frozen_at;
  });

  root.dataset.summaryState = 'loaded';
}
