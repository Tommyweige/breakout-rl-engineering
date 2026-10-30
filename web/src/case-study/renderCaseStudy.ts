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

const REPOSITORY_URL = 'https://github.com/Tommyweige/breakout-rl-engineering';
const RESULTS_URL = `${REPOSITORY_URL}/blob/main/docs/final-results/README.md`;

type Section = CaseStudyCopy['sections'][number];

function renderSection(section: Section, index: number, copy: CaseStudyCopy): string {
  const titleId = `${section.id}-title`;
  const number = String(index + 1).padStart(2, '0');
  const parts: string[] = [
    `<p class="cs-kicker"><span class="cs-index">${number}</span>${escapeHtml(section.kicker)}</p>`,
  ];

  if (section.stat) {
    parts.push(`
      <p class="cs-stat">
        <span class="cs-stat-value">${renderCopy(section.stat.value, copy)}</span>${
          section.stat.unit ? `<span class="cs-stat-unit">${renderCopy(section.stat.unit, copy)}</span>` : ''
        }
      </p>`);
  }

  parts.push(`<h2 id="${escapeHtml(titleId)}">${escapeHtml(section.title)}</h2>`);
  if (section.lead) {
    parts.push(`<p class="cs-lead">${renderCopy(section.lead, copy)}</p>`);
  }

  if (section.stats) {
    parts.push(`
      <dl class="cs-stats">
        ${section.stats
          .map(
            (stat) => `
          <div class="cs-stats-item">
            <dt>${escapeHtml(stat.label)}</dt>
            <dd><span class="cs-stats-value">${renderCopy(stat.value, copy)}</span>${
              stat.unit ? `<span class="cs-stat-unit">${escapeHtml(stat.unit)}</span>` : ''
            }</dd>
          </div>`,
          )
          .join('')}
      </dl>`);
  }

  if (section.bars) {
    parts.push(`
      <div class="cs-bars" data-role="score-bars" data-bar-max-id="${escapeHtml(section.barMaxMetricId ?? '')}">
        ${section.bars
          .map(
            (bar) => `
          <div class="cs-bar">
            <p class="cs-bar-name">${escapeHtml(bar.name)}</p>
            <p class="cs-bar-value">${metricSpan(bar.metricId, copy.common.missingMetric)}</p>
            <div class="cs-bar-track" aria-hidden="true">
              <div class="cs-bar-fill" data-bar-metric-id="${escapeHtml(bar.metricId)}"></div>
            </div>
          </div>`,
          )
          .join('')}
        ${section.barMaxLabel ? `<p class="cs-bar-scale" aria-hidden="true">${escapeHtml(section.barMaxLabel)}</p>` : ''}
      </div>`);
  }

  if (section.cards) {
    parts.push(`
      <div class="cs-cards">
        ${section.cards
          .map(
            (card) => `
          <article class="cs-card">
            <h3>${escapeHtml(card.name)}</h3>
            ${card.tagline ? `<p class="cs-card-tagline">${renderCopy(card.tagline, copy)}</p>` : ''}
            <p class="cs-card-body">${renderCopy(card.body, copy)}</p>
            ${card.verdict ? `<p class="cs-card-verdict">${escapeHtml(card.verdict)}</p>` : ''}
            ${card.detail ? `<p class="cs-card-detail">${renderCopy(card.detail, copy)}</p>` : ''}
          </article>`,
          )
          .join('')}
      </div>`);
  }

  if (section.body) {
    parts.push(
      `<div class="cs-body">${section.body.map((paragraph) => `<p>${renderCopy(paragraph, copy)}</p>`).join('')}</div>`,
    );
  }

  if (section.note) {
    parts.push(`<p class="cs-note">${renderCopy(section.note, copy)}</p>`);
  }

  return `
    <section class="cs-section" id="${escapeHtml(section.id)}" data-tone="${escapeHtml(section.tone ?? 'plain')}" aria-labelledby="${escapeHtml(titleId)}">
      <div class="cs-section-inner">${parts.join('')}</div>
    </section>
  `;
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

  return `
    <header class="cs-header">
      <a class="cs-brand" href="/" aria-label="${escapeHtml(copy.header.brandHomeLabel)}">${escapeHtml(copy.header.brandMark)}</a>
      <nav class="cs-nav" aria-label="${escapeHtml(copy.header.navLabel)}">
        <a href="/">${escapeHtml(copy.header.playDemo)}</a>
        <a href="${REPOSITORY_URL}" target="_blank" rel="noreferrer">${escapeHtml(copy.header.github)}</a>
        ${languageSwitch}
      </nav>
    </header>

    <main id="main">
      <section class="cs-hero" aria-labelledby="hero-title">
        <div class="cs-hero-copy">
          <p class="cs-kicker">${escapeHtml(copy.hero.kicker)}</p>
          <h1 id="hero-title">${escapeHtml(copy.hero.title)}</h1>
          <p class="cs-answer">${escapeHtml(copy.hero.answer)}</p>
          <p class="cs-answer-body">${escapeHtml(copy.hero.answerBody)}</p>
          <a class="cs-scroll" href="#${escapeHtml(copy.sections[0]?.id ?? 'main')}">${escapeHtml(copy.hero.scrollCta)} <span aria-hidden="true">↓</span></a>
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

      ${copy.sections.map((section, index) => renderSection(section, index, copy)).join('')}

      <section class="cs-section cs-evidence" id="evidence" aria-labelledby="evidence-title">
        <div class="cs-section-inner">
          <p class="cs-kicker">${escapeHtml(copy.evidence.kicker)}</p>
          <h2 id="evidence-title">${escapeHtml(copy.evidence.title)}</h2>
          <div class="cs-body"><p>${escapeHtml(copy.evidence.body)}</p></div>
          <div class="cs-actions">
            <a class="cs-button" data-role="evidence-link" href="${RESULTS_URL}" target="_blank" rel="noreferrer">${escapeHtml(copy.evidence.cta)} <span aria-hidden="true">→</span></a>
            <a class="cs-button cs-button-ghost" href="/">${escapeHtml(copy.evidence.playCta)}</a>
          </div>
        </div>
      </section>
    </main>

    <footer class="cs-footer">
      <p>${escapeHtml(copy.footer.updated)} <span data-role="frozen-date">—</span></p>
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

  root.querySelectorAll<HTMLElement>('[data-bar-max-id]').forEach((group) => {
    const max = metrics.get(group.dataset.barMaxId ?? '')?.value;
    group.querySelectorAll<HTMLElement>('[data-bar-metric-id]').forEach((fill) => {
      const value = metrics.get(fill.dataset.barMetricId ?? '')?.value;
      const share = typeof value === 'number' && typeof max === 'number' && max > 0 ? value / max : 0;
      fill.style.setProperty('--share', String(Math.min(1, Math.max(0, share))));
    });
  });

  root.querySelectorAll<HTMLElement>('[data-role="frozen-date"]').forEach((element) => {
    element.textContent = summary.frozen_at;
  });

  root.dataset.summaryState = 'loaded';
}
