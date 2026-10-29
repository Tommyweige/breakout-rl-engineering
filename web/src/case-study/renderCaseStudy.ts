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
    verified_clears_recorded: number;
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

const GITHUB_BLOB = 'https://github.com/Tommyweige/breakout-rl-engineering/blob/main/';
const TOKEN_PATTERN = /\{\{(metric|contract|seeds):([A-Za-z0-9_.-]+)\}\}/g;

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

function statusChip(status: VerificationStatus, copy: CaseStudyCopy): string {
  return `<span class="cs-status" data-status="${status}">${escapeHtml(copy.status.labels[status])}</span>`;
}

function statusForKey(id: string, copy: CaseStudyCopy): string {
  return `<span class="cs-status" data-status-for="${escapeHtml(id)}">${escapeHtml(copy.common.loading)}</span>`;
}

function externalLink(href: string, label: string, className = ''): string {
  const resolved = href.startsWith('http') ? href : `${GITHUB_BLOB}${href.replace(/^\/+/, '')}`;
  return `<a class="${className}" href="${escapeHtml(resolved)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
}

function cardMetric(
  card: CaseStudyCopy['metricCards'][number],
  copy: CaseStudyCopy,
): string {
  return `
    <article class="cs-metric-card">
      <p class="cs-metric-label">${escapeHtml(card.label)}</p>
      <p class="cs-metric-value">${metricSpan(card.metricId)}</p>
      <p class="cs-metric-note">${renderCopy(card.note, copy)}</p>
      ${statusForKey(card.metricId, copy)}
    </article>
  `;
}

export function renderCopy(template: string, copy: CaseStudyCopy): string {
  return template.replace(TOKEN_PATTERN, (_match, kind: string, id: string) => {
    if (kind === 'metric') return metricSpan(id, copy.common.missingMetric);
    if (kind === 'seeds') {
      return `<span data-seed-count-id="${escapeHtml(id)}">${escapeHtml(copy.common.missingMetric)}</span>`;
    }
    return `<span data-contract-field="${escapeHtml(id)}">${escapeHtml(copy.common.missingMetric)}</span>`;
  });
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
          <p class="cs-summary-status" data-role="summary-status" role="status">
            ${escapeHtml(copy.hero.summaryLoading)}
          </p>
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

      <div class="cs-reading-key" role="note" aria-label="${escapeHtml(copy.readingKey.ariaLabel)}">
        <p><strong>${escapeHtml(copy.readingKey.title)}</strong></p>
        <ul>
          ${(Object.keys(copy.status.labels) as VerificationStatus[])
            .map(
              (status) =>
                `<li>${statusChip(status, copy)} <span>${escapeHtml(copy.status.descriptions[status])}</span></li>`,
            )
            .join('')}
        </ul>
      </div>

      <section class="cs-section" aria-labelledby="problem-title">
        <div class="cs-section-heading">
          <p class="cs-index">${escapeHtml(copy.problem.index)}</p>
          <div>
            <p class="cs-kicker">${escapeHtml(copy.problem.kicker)}</p>
            <h2 id="problem-title">${escapeHtml(copy.problem.title)}</h2>
          </div>
        </div>
        <div class="cs-two-column">
          <div class="cs-prose">
            ${copy.problem.paragraphs.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join('')}
            <p class="cs-callout">${renderCopy(copy.problem.callout, copy)}</p>
          </div>
          <div class="cs-diagram" aria-label="${escapeHtml(copy.problem.diagramAria)}">
            ${copy.problem.diagrams
              .map(
                (diagram) => `
              <div class="cs-diagram-row${diagram.accent ? ' cs-diagram-row-accent' : ''}">
                <p class="cs-diagram-label">${escapeHtml(diagram.label)}</p>
                <p class="cs-pipeline">
                  ${diagram.steps
                    .map((step) => `<span>${renderCopy(step, copy)}</span>`)
                    .join('<b>→</b>')}
                </p>
                <p class="cs-diagram-note">${renderCopy(diagram.note, copy)}</p>
              </div>
            `,
              )
              .join('')}
          </div>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="method-title">
        <div class="cs-section-heading">
          <p class="cs-index">${escapeHtml(copy.method.index)}</p>
          <div>
            <p class="cs-kicker">${escapeHtml(copy.method.kicker)}</p>
            <h2 id="method-title">${escapeHtml(copy.method.title)}</h2>
          </div>
        </div>
        <ol class="cs-timeline">
          ${copy.method.items
            .map(
              (item) => `
            <li>
              <span class="cs-timeline-year">${escapeHtml(item.step)}</span>
              <div>
                <h3>${escapeHtml(item.title)}</h3>
                <p>${renderCopy(item.body, copy)}</p>
                ${statusChip(item.status, copy)}
              </div>
            </li>
          `,
            )
            .join('')}
        </ol>
      </section>

      <section class="cs-section" aria-labelledby="results-title" id="evidence">
        <div class="cs-section-heading">
          <p class="cs-index">${escapeHtml(copy.results.index)}</p>
          <div>
            <p class="cs-kicker">${escapeHtml(copy.results.kicker)}</p>
            <h2 id="results-title">${escapeHtml(copy.results.title)}</h2>
          </div>
        </div>
        <p class="cs-section-intro">${renderCopy(copy.results.intro, copy)}</p>
        <div class="cs-table-wrap">
          <table class="cs-table" data-role="results-table">
            <thead>
              <tr>
                <th scope="col">${escapeHtml(copy.results.headers.metric)}</th>
                <th scope="col">${escapeHtml(copy.results.headers.value)}</th>
                <th scope="col">${escapeHtml(copy.results.headers.scope)}</th>
                <th scope="col">${escapeHtml(copy.results.headers.evidence)}</th>
                <th scope="col">${escapeHtml(copy.results.headers.source)}</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="fairness-title">
        <div class="cs-section-heading">
          <p class="cs-index">${escapeHtml(copy.fairness.index)}</p>
          <div>
            <p class="cs-kicker">${escapeHtml(copy.fairness.kicker)}</p>
            <h2 id="fairness-title">${escapeHtml(copy.fairness.title)}</h2>
          </div>
        </div>
        <p class="cs-section-intro">${renderCopy(copy.fairness.intro, copy)}</p>
        <div class="cs-table-wrap">
          <table class="cs-table cs-contract-table" data-role="contract-table">
            <thead>
              <tr>
                <th scope="col">${escapeHtml(copy.fairness.headers.property)}</th>
                <th scope="col">${escapeHtml(copy.fairness.headers.v2)}</th>
                <th scope="col">${escapeHtml(copy.fairness.headers.v3)}</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
        <p class="cs-callout cs-callout-warn" data-role="comparison-note"></p>
        <ul class="cs-notes">
          ${copy.fairness.notes.map((note) => `<li>${renderCopy(note, copy)}</li>`).join('')}
        </ul>
      </section>

      <section class="cs-section cs-section-laya" aria-labelledby="laya-title">
        <div class="cs-section-heading">
          <p class="cs-index">${escapeHtml(copy.laya.index)}</p>
          <div>
            <p class="cs-kicker">${escapeHtml(copy.laya.kicker)}</p>
            <h2 id="laya-title">${escapeHtml(copy.laya.title)}</h2>
          </div>
        </div>
        <div class="cs-laya-grid">
          <div class="cs-prose">
            ${copy.laya.paragraphs
              .map(
                (paragraph) =>
                  `<p><strong>${escapeHtml(paragraph.lead)}</strong> ${renderCopy(paragraph.body, copy)}</p>`,
              )
              .join('')}
            <p class="cs-callout cs-callout-warn">${renderCopy(copy.laya.callout, copy)}</p>
          </div>
          <aside class="cs-fact-panel" aria-label="${escapeHtml(copy.laya.aside.ariaLabel)}">
            <p class="cs-fact-heading">${escapeHtml(copy.laya.aside.heading)}</p>
            <dl>
              ${copy.laya.aside.facts
                .map(
                  (fact) =>
                    `<div><dt>${escapeHtml(fact.label)}</dt><dd>${renderCopy(fact.value, copy)}${
                      fact.unit ? `<small>${escapeHtml(fact.unit)}</small>` : ''
                    }</dd></div>`,
                )
                .join('')}
            </dl>
            <p class="cs-fact-note">${escapeHtml(copy.laya.aside.note)}</p>
            <p class="cs-fact-links">
              ${externalLink('https://github.com/Tommyweige/breakout-rl-engineering/issues/69', copy.laya.aside.researchLinkLabel)}
              ${externalLink('https://github.com/Tommyweige/breakout-rl-engineering/pull/74', copy.laya.aside.prLinkLabel)}
            </p>
          </aside>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="negative-title">
        <div class="cs-section-heading">
          <p class="cs-index">${escapeHtml(copy.negative.index)}</p>
          <div>
            <p class="cs-kicker">${escapeHtml(copy.negative.kicker)}</p>
            <h2 id="negative-title">${escapeHtml(copy.negative.title)}</h2>
          </div>
        </div>
        <div class="cs-negative-grid">
          ${copy.negative.cards
            .map(
              (card) => `
            <article class="cs-negative-card">
              <p class="cs-card-kicker">${escapeHtml(card.kicker)}</p>
              <p class="cs-negative-value">${renderCopy(card.value, copy)}<small>${escapeHtml(card.unit)}</small></p>
              <p>${renderCopy(card.body, copy)}</p>
              ${statusChip(card.status, copy)}
            </article>
          `,
            )
            .join('')}
        </div>
      </section>

      <section class="cs-section" aria-labelledby="ledger-title">
        <div class="cs-section-heading">
          <p class="cs-index">${escapeHtml(copy.ledger.index)}</p>
          <div>
            <p class="cs-kicker">${escapeHtml(copy.ledger.kicker)}</p>
            <h2 id="ledger-title">${escapeHtml(copy.ledger.title)}</h2>
          </div>
        </div>
        <p class="cs-section-intro">${renderCopy(copy.ledger.intro, copy)}</p>
        <div class="cs-ledger" data-role="experiment-list"></div>
        <div class="cs-downloads">
          <a class="cs-button cs-button-ghost" href="/final-results/summary.json">summary.json</a>
          ${copy.ledger.downloads
            .map((download) => externalLink(download.href, download.label, 'cs-button cs-button-ghost'))
            .join('')}
        </div>
      </section>

      <section class="cs-cta" aria-labelledby="cta-title">
        <p class="cs-kicker">${escapeHtml(copy.cta.kicker)}</p>
        <h2 id="cta-title">${escapeHtml(copy.cta.title)}</h2>
        <p>${escapeHtml(copy.cta.body)}</p>
        <div class="cs-hero-actions">
          <a class="cs-button" href="/">${escapeHtml(copy.cta.playCta)}</a>
          <a class="cs-button cs-button-ghost" href="https://github.com/Tommyweige/breakout-rl-engineering" target="_blank" rel="noreferrer">${escapeHtml(copy.cta.githubCta)}</a>
        </div>
      </section>
    </main>

    <footer class="cs-footer">
      <div>
        <p class="cs-brand-mark">${escapeHtml(copy.footer.brand)}</p>
        <p>${escapeHtml(copy.footer.updated)} <span data-role="frozen-date">—</span>.</p>
      </div>
      <div class="cs-footer-note">
        <p><strong>${escapeHtml(copy.footer.notClaimedLabel)}</strong> ${escapeHtml(copy.footer.notClaimed)}</p>
        <p><a href="/">${escapeHtml(copy.footer.playDemo)}</a> · <a href="#evidence">${escapeHtml(copy.footer.evidence)}</a> · <a href="https://github.com/Tommyweige/breakout-rl-engineering" target="_blank" rel="noreferrer">${escapeHtml(copy.footer.github)}</a></p>
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

function sourceHref(ref: string): string {
  if (ref.startsWith('http')) return ref;
  return `${GITHUB_BLOB}${ref.replace(/^\/+/, '')}`;
}

function formatContractNumber(value: number): string {
  return Number.isInteger(value) ? value.toLocaleString('en-US') : String(value);
}

function renderResultsTable(
  summary: FinalResultsSummary,
  copy: CaseStudyCopy,
  locale: Locale,
): string {
  const metrics = new Map(summary.metrics.map((metric) => [metric.id, metric]));
  return summary.table_metric_ids
    .map((id) => {
      const metric = metrics.get(id);
      if (!metric) return '';
      const label = copy.results.metricLabels?.[id] ?? metric.label;
      const scope = metric.seed_role
        ? `${metric.seed_count} · ${metric.seed_role}`
        : String(metric.seed_count);
      return `
        <tr>
          <th scope="row">
            <span class="cs-table-label">${escapeHtml(label)}</span>
            <small>${escapeHtml(metric.contract.id)}${metric.contract.frame_skip === 'mixed' ? '' : ` · frame_skip=${metric.contract.frame_skip}`}</small>
          </th>
          <td class="cs-table-value">
            <span data-metric-id="${escapeHtml(metric.id)}">${escapeHtml(formatMetricValue(metric, locale))}</span>
            <small>${escapeHtml(metric.unit)}</small>
          </td>
          <td>${escapeHtml(scope)}</td>
          <td>${statusChip(metric.verification_status, copy)}</td>
          <td><a href="${escapeHtml(sourceHref(metric.ref))}" target="_blank" rel="noreferrer">${escapeHtml(copy.common.source)}</a></td>
        </tr>
      `;
    })
    .join('');
}

function contractRows(summary: FinalResultsSummary, copy: CaseStudyCopy): string {
  const v2 = summary.contracts.v2;
  const v3 = summary.contracts.v3;
  const phrases = copy.fairness.valuePhrases;
  const decision = (frameSkip: number): string =>
    frameSkip === 1
      ? phrases.decisionPerFrame
      : phrases.decisionPerFrames.replace('{n}', formatContractNumber(frameSkip));
  const rows: Array<[string, string, string]> = [
    [copy.fairness.rows[0]!, v2.contract_id, v3.contract_id],
    [copy.fairness.rows[1]!, v2.path, v3.path],
    [
      copy.fairness.rows[2]!,
      `${v2.sha256.slice(0, 12)}…${v2.sha256.slice(-8)}`,
      `${v3.sha256.slice(0, 12)}…${v3.sha256.slice(-8)}`,
    ],
    [copy.fairness.rows[3]!, decision(v2.frame_skip), decision(v3.frame_skip)],
    [
      copy.fairness.rows[4]!,
      formatContractNumber(v2.agent_step_limit),
      formatContractNumber(v3.agent_step_limit),
    ],
    [
      copy.fairness.rows[5]!,
      formatContractNumber(v2.max_num_frames_per_episode),
      formatContractNumber(v3.max_num_frames_per_episode),
    ],
    [copy.fairness.rows[6]!, String(v2.frame_stack), String(v3.frame_stack)],
    [
      copy.fairness.rows[7]!,
      String(v2.sticky_action_probability),
      String(v3.sticky_action_probability),
    ],
    [
      copy.fairness.rows[8]!,
      v2.fire_reset ? phrases.yes : phrases.no,
      v3.fire_reset ? phrases.yes : phrases.no,
    ],
    [
      copy.fairness.rows[9]!,
      v2.terminal_on_life_loss ? phrases.yes : phrases.no,
      v3.terminal_on_life_loss ? phrases.yes : phrases.no,
    ],
    [
      copy.fairness.rows[10]!,
      phrases.seedCount.replace('{n}', String(v2.evaluation_seeds.length)),
      phrases.seedCount.replace('{n}', String(v3.evaluation_seeds.length)),
    ],
    [copy.fairness.rows[11]!, String(v2.evaluation_epsilon), String(v3.evaluation_epsilon)],
  ];
  return rows
    .map(
      ([property, left, right]) =>
        `<tr><th scope="row">${escapeHtml(property)}</th><td data-label="${escapeHtml(copy.fairness.headers.v2)}">${escapeHtml(left)}</td><td data-label="${escapeHtml(copy.fairness.headers.v3)}">${escapeHtml(right)}</td></tr>`,
    )
    .join('');
}

function experimentList(summary: FinalResultsSummary, copy: CaseStudyCopy): string {
  return summary.experiments
    .map(
      (experiment) => `
        <article class="cs-ledger-item">
          <div class="cs-ledger-main">
            <p class="cs-card-kicker">${escapeHtml(experiment.contract.toUpperCase())} · ${escapeHtml(experiment.id)}</p>
            <h3>${escapeHtml(experiment.title)}</h3>
            <p>${escapeHtml(experiment.headline)}</p>
            <p class="cs-ledger-note">${escapeHtml(experiment.notes)}</p>
          </div>
          <div class="cs-ledger-meta">
            ${statusChip(experiment.verification_status, copy)}
            <a href="${escapeHtml(sourceHref(experiment.source))}" target="_blank" rel="noreferrer">${escapeHtml(copy.common.source)}</a>
          </div>
        </article>
      `,
    )
    .join('');
}

function contractFieldValue(
  summary: FinalResultsSummary,
  path: string,
  copy: CaseStudyCopy,
): string | null {
  const [version, field] = path.split('.');
  if ((version !== 'v2' && version !== 'v3') || !field) return null;
  const contract = summary.contracts[version] as unknown as Record<string, unknown>;
  const value = contract[field];
  if (typeof value === 'number') return formatContractNumber(value);
  if (typeof value === 'boolean') return value ? copy.common.yes : copy.common.no;
  if (typeof value === 'string') return value;
  return null;
}

function hydrateContractFields(
  root: HTMLElement,
  summary: FinalResultsSummary,
  copy: CaseStudyCopy,
): void {
  root.querySelectorAll<HTMLElement>('[data-contract-field]').forEach((element) => {
    const value = contractFieldValue(summary, element.dataset.contractField ?? '', copy);
    element.textContent = value ?? copy.common.unknown;
  });
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
    element.dataset.status = metric.verification_status;
    element.title = `${copy.results.metricLabels?.[metric.id] ?? metric.label}: ${metric.source}`;
  });

  root.querySelectorAll<HTMLElement>('[data-seed-count-id]').forEach((element) => {
    const metric = metrics.get(element.dataset.seedCountId ?? '');
    if (!metric) {
      element.textContent = copy.common.unknown;
      return;
    }
    element.textContent = metric.seed_count.toLocaleString('en-US');
  });

  hydrateContractFields(root, summary, copy);

  root.querySelectorAll<HTMLElement>('[data-status-for]').forEach((element) => {
    const metric = metrics.get(element.dataset.statusFor ?? '');
    if (!metric) {
      element.textContent = copy.common.unknown;
      return;
    }
    element.dataset.status = metric.verification_status;
    element.textContent = copy.status.labels[metric.verification_status];
  });

  const tableBody = root.querySelector<HTMLElement>('[data-role="results-table"] tbody');
  if (tableBody) tableBody.innerHTML = renderResultsTable(summary, copy, locale);

  const contractBody = root.querySelector<HTMLElement>('[data-role="contract-table"] tbody');
  if (contractBody) contractBody.innerHTML = contractRows(summary, copy);

  const comparison = root.querySelector<HTMLElement>('[data-role="comparison-note"]');
  if (comparison) {
    const item = summary.comparisons.find((entry) => entry.id === 'dqn_vs_rgb_controller');
    comparison.innerHTML =
      item && item.comparability === 'not_directly_comparable'
        ? renderCopy(copy.fairness.comparisonNote, copy)
        : renderCopy(copy.fairness.comparisonFallback, copy);
    hydrateContractFields(comparison, summary, copy);
  }

  const experimentContainer = root.querySelector<HTMLElement>('[data-role="experiment-list"]');
  if (experimentContainer) experimentContainer.innerHTML = experimentList(summary, copy);

  root.querySelectorAll<HTMLElement>('[data-role="frozen-date"]').forEach((element) => {
    element.textContent = summary.frozen_at;
  });

  const status = root.querySelector<HTMLElement>('[data-role="summary-status"]');
  if (status) {
    status.textContent = copy.hero.summaryLoaded
      .replace('{metrics}', String(summary.metrics.length))
      .replace('{experiments}', String(summary.experiments.length));
    status.dataset.state = 'loaded';
  }

  root.dataset.summaryState = 'loaded';
}
