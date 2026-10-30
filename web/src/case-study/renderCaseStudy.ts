import { CASE_STUDY_COPY, type CaseStudyCopy, type Locale } from './content';

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

const TOKEN_PATTERN = /\{\{(metric|seeds):([A-Za-z0-9_.-]+)\}\}/g;

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

export function renderCopy(template: string, copy: CaseStudyCopy): string {
  return template.replace(TOKEN_PATTERN, (_match, kind: string, id: string) => {
    if (kind === 'metric') return metricSpan(id, copy.common.missingMetric);
    return `<span data-seed-count-id="${escapeHtml(id)}">${escapeHtml(copy.common.missingMetric)}</span>`;
  });
}

function cardMetric(card: CaseStudyCopy['metricCards'][number], copy: CaseStudyCopy): string {
  return `
    <article class="cs-metric-card">
      <p class="cs-metric-label">${escapeHtml(card.label)}</p>
      <p class="cs-metric-value">
        ${metricSpan(card.metricId)}${
          card.suffix ? `<small class="cs-metric-suffix">${renderCopy(card.suffix, copy)}</small>` : ''
        }
      </p>
      <p class="cs-metric-note">${renderCopy(card.note, copy)}</p>
    </article>
  `;
}

function sectionHeading(
  section: { index: string; kicker: string; title: string },
  titleId: string,
): string {
  return `
    <div class="cs-section-heading">
      <p class="cs-index">${escapeHtml(section.index)}</p>
      <div>
        <p class="cs-kicker">${escapeHtml(section.kicker)}</p>
        <h2 id="${escapeHtml(titleId)}">${escapeHtml(section.title)}</h2>
      </div>
    </div>
  `;
}

function proseSection(
  section: CaseStudyCopy['problem'],
  titleId: string,
  copy: CaseStudyCopy,
  sectionId = '',
): string {
  return `
    <section class="cs-section"${sectionId ? ` id="${escapeHtml(sectionId)}"` : ''} aria-labelledby="${escapeHtml(titleId)}">
      ${sectionHeading(section, titleId)}
      <div class="cs-prose">
        ${section.paragraphs.map((paragraph) => `<p>${renderCopy(paragraph, copy)}</p>`).join('')}
      </div>
    </section>
  `;
}

export function renderCaseStudyShell(locale: Locale): string {
  const copy = CASE_STUDY_COPY[locale];

  return `
    <header class="cs-header">
      <a class="cs-brand" href="/" aria-label="${escapeHtml(copy.header.brandHomeLabel)}">
        <span class="cs-brand-mark">${escapeHtml(copy.header.brandMark)}</span>
        <span class="cs-brand-sub">${escapeHtml(copy.header.brandSub)}</span>
      </a>
      <div class="cs-header-right">
        <nav class="cs-nav" aria-label="${escapeHtml(copy.header.navLabel)}">
          <a href="/">${escapeHtml(copy.header.playDemo)}</a>
          <a href="#evidence">${escapeHtml(copy.header.evidence)}</a>
          <a href="https://github.com/Tommyweige/breakout-rl-engineering" target="_blank" rel="noreferrer">${escapeHtml(copy.header.github)}</a>
        </nav>
        <a
          class="cs-lang"
          data-role="language-switch"
          href="${escapeHtml(copy.header.languageSwitch.href)}"
          hreflang="${escapeHtml(copy.header.languageSwitch.hreflang)}"
          lang="${escapeHtml(copy.header.languageSwitch.lang)}"
          aria-label="${escapeHtml(copy.header.languageSwitch.ariaLabel)}"
        >${escapeHtml(copy.header.languageSwitch.label)}</a>
      </div>
    </header>

    <main id="main">
      <section class="cs-hero" aria-labelledby="hero-title">
        <div class="cs-hero-copy">
          <p class="cs-eyebrow">${escapeHtml(copy.hero.eyebrow)} <span data-role="frozen-date">—</span></p>
          <h1 id="hero-title">${escapeHtml(copy.hero.titleLead)}<br /><span>${escapeHtml(copy.hero.titleTail)}</span></h1>
          <p class="cs-lede">${escapeHtml(copy.hero.lede)}</p>
          <p class="cs-hook">${escapeHtml(copy.hero.hook)}</p>
          <div class="cs-hero-actions">
            <a class="cs-button" href="/">${escapeHtml(copy.hero.playCta)}</a>
            <a class="cs-button cs-button-ghost" href="#evidence">${escapeHtml(copy.hero.evidenceCta)}</a>
          </div>
          <p class="cs-summary-status" data-role="summary-status" role="status" hidden></p>
        </div>
        <figure class="cs-media">
          <video
            class="cs-video"
            data-role="gameplay-video"
            poster="/case-study/gameplay-poster.png"
            muted
            loop
            playsinline
            controls
            preload="metadata"
            aria-label="${escapeHtml(copy.hero.videoAria)}"
          >
            <source src="/case-study/gameplay.mp4" type="video/mp4" />
            ${escapeHtml(copy.hero.videoFallback)}
            <a href="/case-study/gameplay-poster.png">${escapeHtml(copy.hero.posterLink)}</a>
            ${escapeHtml(copy.hero.videoFallbackAfter)}
          </video>
          <figcaption>${escapeHtml(copy.hero.figcaption)}</figcaption>
        </figure>
      </section>

      <section class="cs-metrics" aria-label="${escapeHtml(copy.metricsAria)}">
        ${copy.metricCards.map((card) => cardMetric(card, copy)).join('')}
      </section>

      ${proseSection(copy.problem, 'problem-title', copy)}

      <section class="cs-section" aria-labelledby="routes-title">
        ${sectionHeading(copy.routes, 'routes-title')}
        <div class="cs-routes">
          ${copy.routes.items
            .map(
              (item) => `
            <article class="cs-route">
              <h3>${escapeHtml(item.lead)}</h3>
              <p>${renderCopy(item.body, copy)}</p>
            </article>
          `,
            )
            .join('')}
        </div>
      </section>

      <section class="cs-section" id="evidence" aria-labelledby="results-title">
        ${sectionHeading(copy.results, 'results-title')}
        <p class="cs-section-intro">${renderCopy(copy.results.intro, copy)}</p>
        <div class="cs-table-wrap">
          <table class="cs-table" data-role="results-table">
            <thead>
              <tr>
                <th scope="col">${escapeHtml(copy.results.headers.method)}</th>
                <th scope="col">${escapeHtml(copy.results.headers.mean)}</th>
                <th scope="col">${escapeHtml(copy.results.headers.clears)}</th>
                <th scope="col">${escapeHtml(copy.results.headers.trained)}</th>
              </tr>
            </thead>
            <tbody>
              ${copy.results.rows
                .map(
                  (row) => `
                <tr>
                  <th scope="row">${escapeHtml(row.method)}</th>
                  <td class="cs-table-value" data-label="${escapeHtml(copy.results.headers.mean)}">${renderCopy(row.mean, copy)}</td>
                  <td class="cs-table-value" data-label="${escapeHtml(copy.results.headers.clears)}">${renderCopy(row.clears, copy)}</td>
                  <td data-label="${escapeHtml(copy.results.headers.trained)}">${escapeHtml(row.trained)}</td>
                </tr>
              `,
                )
                .join('')}
            </tbody>
          </table>
        </div>
      </section>

      ${proseSection(copy.mistake, 'mistake-title', copy)}
      ${proseSection(copy.laya, 'laya-title', copy)}
      ${proseSection(copy.negative, 'negative-title', copy)}
    </main>

    <footer class="cs-footer">
      <div>
        <p class="cs-brand-mark">${escapeHtml(copy.footer.brand)}</p>
        <p>${escapeHtml(copy.footer.updated)} <span data-role="frozen-date">—</span>.</p>
      </div>
      <div class="cs-footer-note">
        <p><strong>${escapeHtml(copy.footer.limitsHeading)}</strong></p>
        ${copy.footer.limits.map((limit) => `<p>${escapeHtml(limit)}</p>`).join('')}
        <p>
          <a href="https://github.com/Tommyweige/breakout-rl-engineering/blob/main/docs/final-results/README.md" target="_blank" rel="noreferrer">${escapeHtml(copy.footer.evidenceLine)}</a>
        </p>
      </div>
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

  root.querySelectorAll<HTMLElement>('[data-role="frozen-date"]').forEach((element) => {
    element.textContent = summary.frozen_at;
  });

  root.dataset.summaryState = 'loaded';
}
