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
  value: number | string | null;
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

const STATUS_LABELS: Record<VerificationStatus, string> = {
  verified: 'Verified',
  historical_reported: 'Historical reported',
  missing_raw_artifacts: 'Missing raw artifacts',
  not_yet_evaluated: 'Not yet evaluated',
};

const STATUS_DESCRIPTIONS: Record<VerificationStatus, string> = {
  verified: 'Checked against an artifact in this repository or a rerunnable command.',
  historical_reported: 'Reported in a merged PR, issue, or archived article; raw run output is not in main.',
  missing_raw_artifacts: 'Described, but the raw episodes, logs, or weights needed to audit it are unavailable.',
  not_yet_evaluated: 'No completed evaluation exists for this claim.',
};

const GITHUB_BLOB = 'https://github.com/Tommyweige/breakout-rl-engineering/blob/main/';
const GITHUB_TREE = 'https://github.com/Tommyweige/breakout-rl-engineering/tree/main/';

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function metricSpan(id: string, fallback = '—'): string {
  return `<span data-metric-id="${escapeHtml(id)}">${fallback}</span>`;
}

function statusChip(status: VerificationStatus, label = STATUS_LABELS[status]): string {
  return `<span class="cs-status" data-status="${status}">${escapeHtml(label)}</span>`;
}

function statusForKey(key: string): string {
  return `<span class="cs-status" data-status-for="${escapeHtml(key)}">Loading…</span>`;
}

function externalLink(href: string, label: string, className = ''): string {
  const resolved = href.startsWith('http') ? href : `${GITHUB_BLOB}${href.replace(/^\/+/, '')}`;
  return `<a class="${className}" href="${escapeHtml(resolved)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
}

function cardMetric(
  id: string,
  label: string,
  note: string,
  suffix = '',
): string {
  return `
    <article class="cs-metric-card">
      <p class="cs-metric-label">${escapeHtml(label)}</p>
      <p class="cs-metric-value">${metricSpan(id)}${suffix ? `<small>${escapeHtml(suffix)}</small>` : ''}</p>
      <p class="cs-metric-note">${escapeHtml(note)}</p>
      ${statusForKey(id)}
    </article>
  `;
}

export function renderCaseStudyShell(): string {
  return `
    <header class="cs-header">
      <a class="cs-brand" href="/" aria-label="Breakout RL home">
        <span class="cs-brand-mark">BREAKOUT</span>
        <span class="cs-brand-sub">RESEARCH RESULTS</span>
      </a>
      <nav class="cs-nav" aria-label="Primary">
        <a href="/">Play demo</a>
        <a href="#evidence">Evidence</a>
        <a href="https://github.com/Tommyweige/breakout-rl-engineering" target="_blank" rel="noreferrer">GitHub</a>
      </nav>
    </header>

    <main id="main">
      <section class="cs-hero" aria-labelledby="hero-title">
        <div class="cs-hero-copy">
          <p class="cs-eyebrow">FROZEN EVIDENCE · <span data-role="frozen-date">2026-10-28</span></p>
          <h1 id="hero-title">Does Breakout<br /><span>really need training?</span></h1>
          <p class="cs-lede">
            From a DQN family to an explainable pixel tracker and predictive controller.
            Two decision paradigms, one open question — and no verified full clear yet.
          </p>
          <p class="cs-hook">
            This is the evidence-led closeout of an end-to-end RL engineering project.
            The interesting result is not a high score. It is how much the answer changes
            when the control contract changes, and how careful you have to be before calling
            a game “solved”.
          </p>
          <div class="cs-hero-actions">
            <a class="cs-button" href="/">Play the demo</a>
            <a class="cs-button cs-button-ghost" href="#evidence">Read the evidence</a>
          </div>
          <p class="cs-summary-status" data-role="summary-status" role="status">
            Loading the machine-readable summary…
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
            aria-label="Recorded Breakout gameplay: the human player on the left and the AI policy on the right."
          >
            <source src="/case-study/gameplay.mp4" type="video/mp4" />
            Your browser cannot play this recording. The
            <a href="/case-study/gameplay-poster.png">poster frame</a>
            shows the same real gameplay.
          </video>
          <figcaption>
            Real capture from the built app. The left panel is the human surface; the right
            panel is the AI policy. No mock frames or synthesized gameplay.
          </figcaption>
        </figure>
      </section>

      <section class="cs-metrics" aria-label="Headline results">
        ${cardMetric(
          'verified_clears_recorded',
          'Verified full clears',
          'Audited 864-point detector; absence of record through the freeze date.',
        )}
        ${cardMetric(
          'dqn_final_holdout_mean',
          'DQN final holdout mean',
          'Dueling Double DQN, 2.5M transitions, Contract v2.',
        )}
        ${cardMetric(
          'rgb_controller_mean_score',
          'RGB controller mean',
          'Deterministic pixel controller, Contract v3.',
        )}
        ${cardMetric(
          'laya_complete_episodes',
          'Laya complete episodes',
          'Frozen 201M vision model; latency was measured, gameplay was not completed.',
        )}
      </section>

      <div class="cs-reading-key" role="note" aria-label="How to read the evidence status">
        <p><strong>How to read every claim</strong></p>
        <ul>
          ${(Object.keys(STATUS_LABELS) as VerificationStatus[])
            .map(
              (status) =>
                `<li>${statusChip(status)} <span>${escapeHtml(STATUS_DESCRIPTIONS[status])}</span></li>`,
            )
            .join('')}
        </ul>
      </div>

      <section class="cs-section" aria-labelledby="problem-title">
        <div class="cs-section-heading">
          <p class="cs-index">01</p>
          <div>
            <p class="cs-kicker">THE PROBLEM</p>
            <h2 id="problem-title">Breakout is a control problem disguised as a score.</h2>
          </div>
        </div>
        <div class="cs-two-column">
          <div class="cs-prose">
            <p>
              The ball and paddle are visually separable. Between collisions the motion is close
              to piecewise linear, and a controller can re-plan after every observation. That makes
              Breakout a useful test of a simple question: how much of the task actually needs a
              learned policy?
            </p>
            <p>
              The catch is the environment contract. Frame skip, sticky actions, FIRE handling,
              life-loss handling, and the episode limit all change the control problem. A score
              without its contract is not a result; it is a number looking for a footnote.
            </p>
            <p class="cs-callout">
              <strong>The clear condition is explicit.</strong> Destroy both brick walls in
              ALE/Breakout-v5 one-player Game 1. The audited maximum is
              ${metricSpan('clear_score_threshold')} points, read from the ROM scoreboard in ALE RAM
              and cross-checked against raw reward. An episode that ends is not automatically a clear.
            </p>
          </div>
          <div class="cs-diagram" aria-label="Two decision pipelines compared">
            <div class="cs-diagram-row">
              <p class="cs-diagram-label">LEARNED POLICY</p>
              <p class="cs-pipeline">
                <span>4 stacked frames</span><b>→</b><span>CNN</span><b>→</b><span>Q-values</span><b>→</b><span>argmax</span>
              </p>
              <p class="cs-diagram-note">Contract v2: one decision per four emulator frames.</p>
            </div>
            <div class="cs-diagram-row cs-diagram-row-accent">
              <p class="cs-diagram-label">EXPLICIT CONTROLLER</p>
              <p class="cs-pipeline">
                <span>RGB frame</span><b>→</b><span>ball/paddle</span><b>→</b><span>velocity</span><b>→</b><span>intercept</span><b>→</b><span>action</span>
              </p>
              <p class="cs-diagram-note">Contract v3: one decision per emulator frame. No training.</p>
            </div>
          </div>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="method-title">
        <div class="cs-section-heading">
          <p class="cs-index">02</p>
          <div>
            <p class="cs-kicker">METHOD EVOLUTION</p>
            <h2 id="method-title">Build the baseline, then remove the parts that do not earn their keep.</h2>
          </div>
        </div>
        <ol class="cs-timeline">
          <li>
            <span class="cs-timeline-year">01</span>
            <div>
              <h3>DQN family under Contract v2</h3>
              <p>
                DQN → Double DQN → Dueling Double DQN. The family screen selected Dueling Double DQN
                at 500K transitions and kept it through 1M. The final policy trained for
                ${metricSpan('dqn_training_transitions')} transitions.
              </p>
              ${statusChip('historical_reported')}
            </div>
          </li>
          <li>
            <span class="cs-timeline-year">02</span>
            <div>
              <h3>Controlled experiments that did not promote</h3>
              <p>
                Life-loss reward shaping produced a positive but inconsistent point estimate;
                prioritized replay was descriptive at best. Neither replaced the baseline. These
                are results, not footnotes.
              </p>
              ${statusChip('historical_reported')}
            </div>
          </li>
          <li>
            <span class="cs-timeline-year">03</span>
            <div>
              <h3>A deterministic pixel controller under Contract v3</h3>
              <p>
                Detect the ball and paddle, estimate velocity, predict the paddle-plane intercept
                with wall reflections, and close the loop on observed paddle position. The
                controller runs at about ${metricSpan('rgb_controller_latency_ms_per_frame')} ms per
                controller frame — but it did not clear the game.
              </p>
              ${statusChip('historical_reported')}
            </div>
          </li>
          <li>
            <span class="cs-timeline-year">04</span>
            <div>
              <h3>A frozen vision model as a side experiment</h3>
              <p>
                A ${metricSpan('laya_parameter_count')}-parameter SmolVLM-based checkpoint,
                already exposed to some Breakout expert data, was tested as a frozen policy. The
                latency work is real; the gameplay evidence is not complete.
              </p>
              ${statusChip('missing_raw_artifacts')}
            </div>
          </li>
        </ol>
      </section>

      <section class="cs-section" aria-labelledby="results-title" id="evidence">
        <div class="cs-section-heading">
          <p class="cs-index">03</p>
          <div>
            <p class="cs-kicker">RESULTS AT A GLANCE</p>
            <h2 id="results-title">Every number carries its contract and its evidence status.</h2>
          </div>
        </div>
        <p class="cs-section-intro">
          The table is generated from
          <a href="/final-results/summary.json">summary.json</a>. The page does not keep a second
          hand-typed copy of these values. “Historical reported” means the raw run is not in
          <code>main</code>; it does not mean the number was reproduced here.
        </p>
        <div class="cs-table-wrap">
          <table class="cs-table" data-role="results-table">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">Value</th>
                <th scope="col">Seeds / scope</th>
                <th scope="col">Evidence</th>
                <th scope="col">Source</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="fairness-title">
        <div class="cs-section-heading">
          <p class="cs-index">04</p>
          <div>
            <p class="cs-kicker">CONTRACTS AND FAIRNESS</p>
            <h2 id="fairness-title">The two scores answer different questions.</h2>
          </div>
        </div>
        <p class="cs-section-intro">
          Contract v2 gives the policy one decision per four emulator frames. Contract v3 gives it
          one decision per frame. Both keep the same 108,000-frame ALE episode limit, but the
          agent-step budgets are 27,000 and 108,000. A raw-score comparison across them is
          descriptive, not a ranking.
        </p>
        <div class="cs-table-wrap">
          <table class="cs-table cs-contract-table" data-role="contract-table">
            <thead>
              <tr>
                <th scope="col">Property</th>
                <th scope="col">Contract v2</th>
                <th scope="col">Contract v3</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
        <p class="cs-callout cs-callout-warn" data-role="comparison-note"></p>
        <ul class="cs-notes">
          <li><strong>No fair winner claim.</strong> The DQN holdout and the RGB controller were not run under one matched contract, seed set, episode budget, and command.</li>
          <li><strong>No method ranking from raw score.</strong> A higher score under one cadence does not establish a better method under another.</li>
          <li><strong>No post-hoc promotion.</strong> A matched comparison would have to be predeclared before looking at its results.</li>
        </ul>
      </section>

      <section class="cs-section cs-section-laya" aria-labelledby="laya-title">
        <div class="cs-section-heading">
          <p class="cs-index">05</p>
          <div>
            <p class="cs-kicker">SIDE EXPERIMENT</p>
            <h2 id="laya-title">Laya Vision: real latency, a measured gameplay session, no complete evaluation.</h2>
          </div>
        </div>
        <div class="cs-laya-grid">
          <div class="cs-prose">
            <p>
              <strong>Motivation.</strong> Test whether a small, frozen, non-generative vision
              decision model can choose legal Breakout actions from pixels at a practical inference
              cost.
            </p>
            <p>
              <strong>Frozen hypothesis.</strong> A 201M-parameter SmolVLM-based checkpoint can act
              as a frozen policy without project-specific training. It was already trained on some
              Breakout expert-action data, so this is <em>not</em> a zero-shot claim.
            </p>
            <p>
              <strong>What was measured.</strong> Warm decision latency, a bounded control
              diagnostic, and a live browser session. Warm p50 was ${metricSpan('laya_warm_latency_p50_ms')} ms on one fixed RGB
              fixture; the optimized local proxy p50 was
              ${metricSpan('laya_optimized_proxy_p50_ms')} ms across three rounds. Immediate reversal
              commands went from ${metricSpan('laya_control_diagnostic_raw_reversals')} to
              ${metricSpan('laya_control_diagnostic_reversals')} in a 3-seed × 80-decision
              diagnostic. In the browser session the Decision Model scored
              ${metricSpan('laya_browser_session_score')} and lost every life after
              ${metricSpan('laya_browser_session_played_s')} s of play, presenting
              ${metricSpan('laya_browser_presented_fps')} frames per second.
            </p>
            <p>
              <strong>What is still not measured.</strong> No complete evaluator episode,
              per-episode artifact, or clear exists, so the browser session is not comparable with
              evaluated episodes. Latency does not imply gameplay quality.
            </p>
            <p class="cs-callout cs-callout-warn">
              <strong>Decision:</strong> keep Laya as a research lead. Draft PR #74 stays draft
              until a complete evaluated episode exists. #70–#72 stay open.
            </p>
          </div>
          <aside class="cs-fact-panel" aria-label="Laya evidence summary">
            <p class="cs-fact-heading">Laya at freeze time</p>
            <dl>
              <div><dt>Complete episodes</dt><dd>${metricSpan('laya_complete_episodes')}</dd></div>
              <div><dt>Warm p50</dt><dd>${metricSpan('laya_warm_latency_p50_ms')}<small>ms/decision</small></dd></div>
              <div><dt>Optimized proxy p50</dt><dd>${metricSpan('laya_optimized_proxy_p50_ms')}<small>ms/request</small></dd></div>
              <div><dt>Reversals, raw → braked</dt><dd>${metricSpan('laya_control_diagnostic_raw_reversals')} → ${metricSpan('laya_control_diagnostic_reversals')}</dd></div>
            </dl>
            <p class="cs-fact-note">Weights are CC BY-NC-SA 4.0 and are not in the repository.</p>
            <p class="cs-fact-links">
              ${externalLink('https://github.com/Tommyweige/breakout-rl-engineering/issues/69', 'Research issue')}
              ${externalLink('https://github.com/Tommyweige/breakout-rl-engineering/pull/74', 'Draft PR')}
            </p>
          </aside>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="negative-title">
        <div class="cs-section-heading">
          <p class="cs-index">06</p>
          <div>
            <p class="cs-kicker">FAILURES AND NEGATIVE RESULTS</p>
            <h2 id="negative-title">The experiments that did not promote are part of the result.</h2>
          </div>
        </div>
        <div class="cs-negative-grid">
          <article class="cs-negative-card">
            <p class="cs-card-kicker">REWARD SHAPING · ISSUE #9</p>
            <p class="cs-negative-value">${metricSpan('reward_shaping_mean_effect')}<small>mean effect</small></p>
            <p>
              95% interval ${metricSpan('reward_shaping_ci_low')} to
              ${metricSpan('reward_shaping_ci_high')}. Two of three training seeds were positive.
              The promotion gate did not pass, so the 2.5M comparison was not run.
            </p>
            ${statusChip('historical_reported')}
          </article>
          <article class="cs-negative-card">
            <p class="cs-card-kicker">PRIORITIZED REPLAY · ISSUE #12</p>
            <p class="cs-negative-value">${metricSpan('per_500k_mean_score')}<small>PER mean at 500K</small></p>
            <p>
              Uniform replay was ${metricSpan('uniform_500k_mean_score')} at the same milestone.
              PER cost ${metricSpan('per_training_time_ratio')}× training time. With three seeds,
              this is descriptive, not statistical superiority.
            </p>
            ${statusChip('historical_reported')}
          </article>
          <article class="cs-negative-card">
            <p class="cs-card-kicker">FIRST-CLEAR PROBES · #41, #43, #45, #48–#61</p>
            <p class="cs-negative-value">0<small>verified clears</small></p>
            <p>
              The frozen probes ended without a verified clear. Several were explicitly
              INCONCLUSIVE or BLOCKED_ENVIRONMENT. They identify failure modes; they do not prove
              that a method cannot clear.
            </p>
            ${statusChip('historical_reported')}
          </article>
        </div>
      </section>

      <section class="cs-section" aria-labelledby="ledger-title">
        <div class="cs-section-heading">
          <p class="cs-index">07</p>
          <div>
            <p class="cs-kicker">PUBLIC EVIDENCE</p>
            <h2 id="ledger-title">The ledger, not the highlight reel.</h2>
          </div>
        </div>
        <p class="cs-section-intro">
          Every experiment family in the closeout, with its status and source. The CSV and JSON are
          the machine-readable versions of this page.
        </p>
        <div class="cs-ledger" data-role="experiment-list"></div>
        <div class="cs-downloads">
          <a class="cs-button cs-button-ghost" href="/final-results/summary.json">summary.json</a>
          ${externalLink('docs/final-results/evidence-inventory.csv', 'evidence-inventory.csv', 'cs-button cs-button-ghost')}
          ${externalLink('docs/final-results/README.md', 'Full results index', 'cs-button cs-button-ghost')}
          ${externalLink('docs/final-results/handoff.md', 'Handoff and blockers', 'cs-button cs-button-ghost')}
        </div>
      </section>

      <section class="cs-cta" aria-labelledby="cta-title">
        <p class="cs-kicker">DEMO + SOURCE</p>
        <h2 id="cta-title">Play the current build, then inspect the evidence behind it.</h2>
        <p>
          The demo runs ALE WASM and the ONNX policy entirely in the browser. “Unbeatable” is a
          UI difficulty label: it injects no deliberate mistakes. It is not a claim that the policy
          cannot lose.
        </p>
        <div class="cs-hero-actions">
          <a class="cs-button" href="/">Play the demo</a>
          <a class="cs-button cs-button-ghost" href="https://github.com/Tommyweige/breakout-rl-engineering" target="_blank" rel="noreferrer">GitHub repository</a>
        </div>
      </section>
    </main>

    <footer class="cs-footer">
      <div>
        <p class="cs-brand-mark">BREAKOUT · RESEARCH RESULTS</p>
        <p>Frozen evidence closeout. Last updated <span data-role="frozen-date">2026-10-28</span>.</p>
      </div>
      <div class="cs-footer-note">
        <p><strong>Not claimed:</strong> solved game, verified clear, reliable clear, world record, fair method winner, or a complete evaluated Laya episode.</p>
        <p><a href="/">Play demo</a> · <a href="#evidence">Evidence</a> · <a href="https://github.com/Tommyweige/breakout-rl-engineering" target="_blank" rel="noreferrer">GitHub</a></p>
      </div>
    </footer>
  `;
}

export function formatMetricValue(metric: SummaryMetric): string {
  if (metric.value === null) return 'Not run';
  if (typeof metric.value === 'boolean') return metric.value ? 'Yes' : 'No';
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

function renderResultsTable(summary: FinalResultsSummary): string {
  const metrics = new Map(summary.metrics.map((metric) => [metric.id, metric]));
  return summary.table_metric_ids
    .map((id) => {
      const metric = metrics.get(id);
      if (!metric) return '';
      const scope = metric.seed_role
        ? `${metric.seed_count} · ${metric.seed_role}`
        : String(metric.seed_count);
      return `
        <tr>
          <th scope="row">
            <span class="cs-table-label">${escapeHtml(metric.label)}</span>
            <small>${escapeHtml(metric.contract.id)}${metric.contract.frame_skip === 'mixed' ? '' : ` · frame_skip=${metric.contract.frame_skip}`}</small>
          </th>
          <td class="cs-table-value">
            <span data-metric-id="${escapeHtml(metric.id)}">${escapeHtml(formatMetricValue(metric))}</span>
            <small>${escapeHtml(metric.unit)}</small>
          </td>
          <td>${escapeHtml(scope)}</td>
          <td>${statusChip(metric.verification_status)}</td>
          <td><a href="${escapeHtml(sourceHref(metric.ref))}" target="_blank" rel="noreferrer">source</a></td>
        </tr>
      `;
    })
    .join('');
}

function contractRows(summary: FinalResultsSummary): string {
  const v2 = summary.contracts.v2;
  const v3 = summary.contracts.v3;
  const rows: Array<[string, string, string]> = [
    ['Contract ID', v2.contract_id, v3.contract_id],
    ['File', v2.path, v3.path],
    ['SHA-256', `${v2.sha256.slice(0, 12)}…${v2.sha256.slice(-8)}`, `${v3.sha256.slice(0, 12)}…${v3.sha256.slice(-8)}`],
    ['Decision cadence', `1 per ${v2.frame_skip} emulator frames`, `1 per ${v3.frame_skip} emulator frame`],
    ['Agent step limit', String(v2.agent_step_limit), String(v3.agent_step_limit)],
    ['ALE episode-frame limit', String(v2.max_num_frames_per_episode), String(v3.max_num_frames_per_episode)],
    ['Frame stack', String(v2.frame_stack), String(v3.frame_stack)],
    ['Sticky actions', String(v2.sticky_action_probability), String(v3.sticky_action_probability)],
    ['FIRE reset', v2.fire_reset ? 'yes' : 'no', v3.fire_reset ? 'yes' : 'no'],
    ['Terminal on life loss', v2.terminal_on_life_loss ? 'yes' : 'no', v3.terminal_on_life_loss ? 'yes' : 'no'],
    ['Evaluation seeds', `${v2.evaluation_seeds.length} seeds`, `${v3.evaluation_seeds.length} seeds`],
    ['Evaluation epsilon', String(v2.evaluation_epsilon), String(v3.evaluation_epsilon)],
  ];
  return rows
    .map(
      ([property, left, right]) =>
        `<tr><th scope="row">${escapeHtml(property)}</th><td data-label="Contract v2">${escapeHtml(left)}</td><td data-label="Contract v3">${escapeHtml(right)}</td></tr>`,
    )
    .join('');
}

function experimentList(summary: FinalResultsSummary): string {
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
            ${statusChip(experiment.verification_status)}
            <a href="${escapeHtml(sourceHref(experiment.source))}" target="_blank" rel="noreferrer">source</a>
          </div>
        </article>
      `,
    )
    .join('');
}

export function hydrateCaseStudy(root: HTMLElement, summary: FinalResultsSummary): void {
  const metrics = new Map(summary.metrics.map((metric) => [metric.id, metric]));

  root.querySelectorAll<HTMLElement>('[data-metric-id]').forEach((element) => {
    const metric = metrics.get(element.dataset.metricId ?? '');
    if (!metric) {
      element.textContent = '—';
      return;
    }
    element.textContent = formatMetricValue(metric);
    element.dataset.status = metric.verification_status;
    element.title = `${metric.label}: ${metric.source}`;
  });

  root.querySelectorAll<HTMLElement>('[data-status-for]').forEach((element) => {
    const metric = metrics.get(element.dataset.statusFor ?? '');
    if (!metric) {
      element.textContent = 'Unknown';
      return;
    }
    element.dataset.status = metric.verification_status;
    element.textContent = STATUS_LABELS[metric.verification_status];
  });

  const tableBody = root.querySelector<HTMLElement>('[data-role="results-table"] tbody');
  if (tableBody) tableBody.innerHTML = renderResultsTable(summary);

  const contractBody = root.querySelector<HTMLElement>('[data-role="contract-table"] tbody');
  if (contractBody) contractBody.innerHTML = contractRows(summary);

  const comparison = root.querySelector<HTMLElement>('[data-role="comparison-note"]');
  if (comparison) {
    const item = summary.comparisons.find((entry) => entry.id === 'dqn_vs_rgb_controller');
    comparison.textContent = item
      ? `${item.title}: ${item.comparability.replaceAll('_', ' ')}. ${item.reason}`
      : 'Cross-contract comparison is not directly comparable.';
  }

  const experimentContainer = root.querySelector<HTMLElement>('[data-role="experiment-list"]');
  if (experimentContainer) experimentContainer.innerHTML = experimentList(summary);

  root.querySelectorAll<HTMLElement>('[data-role="frozen-date"]').forEach((element) => {
    element.textContent = summary.frozen_at;
  });

  const status = root.querySelector<HTMLElement>('[data-role="summary-status"]');
  if (status) {
    status.textContent = `Summary loaded: ${summary.metrics.length} metrics, ${summary.experiments.length} experiment families.`;
    status.dataset.state = 'loaded';
  }

  root.dataset.summaryState = 'loaded';
}
