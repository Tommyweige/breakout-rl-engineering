import { ACTION_MEANINGS, type ActionMeaning, type FixtureValidationResult } from '../inference/types';

export function renderAppShell(debug = false): string {
  return [
    renderMasthead(),
    renderCommandDeck(),
    renderPanelGrid(),
    debug ? renderTechnicalDetails() : '',
    renderFooter(),
  ].join('');
}

function renderMasthead(): string {
  return `
    <header class="site-header">
      <div class="hero">
        <div class="hero-copy">
          <h1>BREAKOUT</h1>
        </div>
        <p class="subtitle">YOU vs AI</p>
      </div>
    </header>
  `;
}

function renderCommandDeck(): string {
  return `
    <section class="game-controls" aria-label="game controls">
      <div class="control-intro">
        <strong>READY?</strong>
        <span data-role="user-message" role="status" aria-live="polite">Start both games whenever you are ready.</span>
      </div>
      <div class="toolbar-options">
        <label class="input-picker toolbar-picker" for="input-mode-select">Control
          <select id="input-mode-select" data-action="input-mode" aria-label="Choose human control mode">
            <option value="keyboard" selected>Keyboard</option>
            <option value="mouse">Mouse</option>
          </select>
        </label>
        <div class="controls">
        <button data-action="start" type="button" class="primary">Start</button>
        <button data-action="pause" type="button">Pause</button>
        <button data-action="reset" type="button">Restart</button>
        </div>
      </div>
    </section>
  `;
}

function renderPanelGrid(): string {
  return `
    <section id="play-area" class="panel-grid" aria-label="Human and AI Breakout games">
      ${renderHumanPanel()}
      ${renderAgentPanel()}
    </section>
  `;
}

function renderHumanPanel(): string {
  return `
    <article class="game-card panel panel-human" aria-labelledby="human-title">
      <div class="panel-heading">
        <div>
          <p class="panel-kicker"><span class="panel-index">YOU</span></p>
          <h2 id="human-title">YOU</h2>
        </div>
        <span class="status-chip status-chip-muted" data-role="human-state">Ready</span>
      </div>
      <dl class="score-row score-row-top" aria-label="Your score and lives">
        <div><dt>Score</dt><dd data-role="human-score">0</dd></div>
        <div><dt>Lives</dt><dd data-role="human-lives">—</dd></div>
      </dl>
      <div class="game-stage human-game-stage">
        <div class="stage-topline"><span>PLAYER 01</span><span class="stage-state" data-role="human-stage-state">READY</span></div>
        <canvas data-role="human-canvas" width="160" height="210" aria-label="Your Breakout game canvas"></canvas>
        <span class="mouse-target-marker" data-role="mouse-target-marker" aria-hidden="true" hidden></span>
      </div>
      <div class="input-status">
        <strong data-role="human-input-mode">Keyboard</strong>
        <span class="input-hint" data-role="input-hint">← / → move · Space serves</span>
        <div class="live-line"><span>INPUT</span><strong data-role="human-input">none</strong></div>
      </div>
    </article>
  `;
}

function renderAgentPanel(): string {
  return `
    <article class="game-card panel panel-agent" aria-labelledby="agent-title">
      <div class="panel-heading">
        <div>
          <p class="panel-kicker"><span class="panel-index">AI</span></p>
          <h2 id="agent-title">AI</h2>
        </div>
        <div class="panel-heading-tools">
          <label class="difficulty-picker" for="difficulty-select">AI Difficulty
            <select id="difficulty-select" data-action="difficulty" aria-label="Choose AI difficulty">
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard" selected>Hard</option>
              <option value="unbeatable">Unbeatable</option>
            </select>
          </label>
          <span class="status-chip" data-role="agent-state">Ready</span>
        </div>
      </div>
      <dl class="score-row score-row-top" aria-label="AI score and lives">
        <div><dt>Score</dt><dd data-role="agent-score">0</dd></div>
        <div><dt>Lives</dt><dd data-role="agent-lives">—</dd></div>
      </dl>
      <div class="game-stage agent-game-stage">
        <div class="stage-topline"><span>AI</span><span class="stage-state" data-role="agent-stage-state">READY</span></div>
        <canvas data-role="agent-canvas" width="160" height="210" aria-label="AI Breakout game canvas"></canvas>
      </div>
      <div class="agent-copy">
        <span class="agent-signal" aria-hidden="true"></span>
        <p><strong data-role="ai-difficulty-label">HARD</strong><br>AI is playing automatically.</p>
      </div>
    </article>
  `;
}

function renderTechnicalDetails(): string {
  return `
    <details class="technical-details" data-role="technical-details">
      <summary><span>Technical details</span><small>Validation, backend evidence, and raw evaluation</small></summary>
      <div class="technical-content">
        <section class="technical-controls" aria-label="technical controls">
          <div class="technical-controls-copy">
            <p class="eyebrow">ENGINEERING WORKBENCH</p>
            <h2>Keep the evidence close, not in the way.</h2>
            <p data-role="validation-message">尚未執行。這裡的驗證會對真實 Browser inference 使用明確的 execution provider。</p>
          </div>
          <div class="technical-actions">
            <label class="backend-picker" for="backend-select">Formal backend
              <select id="backend-select" data-action="backend" aria-label="Formal evaluation backend">
                <option value="wasm" selected>WASM / CPU</option>
                <option value="webgpu">WebGPU / GPU</option>
              </select>
            </label>
            <div class="technical-button-row">
              <button data-action="validate" type="button" class="secondary">Validate</button>
              <button data-action="benchmark" type="button" class="secondary">Benchmark</button>
              <button data-action="evaluate" type="button" class="secondary">Run 50 episodes</button>
            </div>
            <div class="deck-downloads">
              <a data-role="download-validation" class="download-link" download hidden>Download validation JSON <span>↓</span></a>
              <a data-role="download-benchmark" class="download-link" download="web-benchmark.json" hidden>Download benchmark JSON <span>↓</span></a>
              <a data-role="download-evaluation" class="download-link" download="final-browser-policy-evaluation.json" hidden>Download score evidence <span>↓</span></a>
            </div>
          </div>
        </section>
        <section class="technical-evidence" aria-label="runtime evidence">
          <div class="technical-heading"><div><p class="eyebrow">RUNTIME EVIDENCE</p><h2>What the browser actually ran</h2></div><div class="technical-heading-status"><span class="runtime-badge" data-role="runtime-status" data-status="idle"><span class="status-dot"></span><span data-role="runtime-status-label">ready</span></span><span class="status-chip" data-role="validation-status">Not validated</span></div></div>
          <div class="technical-facts">
            <div><span>Model</span><strong data-role="model-loaded">not loaded</strong></div>
            <div><span>Requested / actual</span><strong><span data-role="requested-backend">WASM</span><span class="fact-slash">/</span><span data-role="actual-backend">—</span></strong></div>
            <div><span>WebGPU support</span><strong data-role="webgpu-support" data-status="checking">checking…</strong></div>
            <div><span>ORT Web</span><strong data-role="ort-version">—</strong></div>
            <div><span>Browser / platform</span><strong data-role="browser-info">—</strong></div>
            <div><span>Environment parity</span><strong data-role="environment-parity">—</strong></div>
            <div><span>Gameplay backend</span><strong data-role="gameplay-backend">—</strong></div>
            <div><span>Scheduler</span><strong data-role="scheduler-status">idle</strong></div>
            <div><span>Current action</span><strong data-role="current-action">NOOP</strong></div>
            <div><span>Cursor target X</span><strong data-role="cursor-target-x">—</strong></div>
            <div><span>Paddle center X</span><strong data-role="paddle-center-x">—</strong></div>
            <div><span>Motion state</span><strong data-role="motion-state">STOPPED</strong></div>
            <div><span>Position error</span><strong data-role="position-error">—</strong></div>
            <div><span>Mouse dead zone</span><strong data-role="mouse-deadzone-raw-px">6</strong><small>Atari px</small></div>
            <div><span>Dead zone normalized</span><strong data-role="start-threshold">0.03750</strong></div>
            <div><span>Dead zone normalized</span><strong data-role="stop-threshold">0.03750</strong></div>
            <div><span>Human requested</span><strong data-role="human-requested-action">NOOP</strong></div>
            <div><span>Human executed</span><strong data-role="human-executed-action">NOOP</strong></div>
            <div><span>Requested paddle strength</span><strong data-role="requested-paddle-strength">0.000</strong></div>
            <div><span>Executed paddle strength</span><strong data-role="executed-paddle-strength">—</strong></div>
            <div><span>Raw frame number</span><strong data-role="human-raw-frame-number">0</strong></div>
            <div><span>Actual emulator frames</span><strong data-role="human-actual-emulator-frames">0</strong></div>
            <div><span>Human action</span><strong data-role="executed-human-action">NOOP</strong></div>
            <div><span>Human raw FPS</span><strong data-role="human-raw-fps">0</strong></div>
            <div><span>Human tick count</span><strong data-role="human-tick-count">0</strong></div>
            <div><span>Human late / dropped</span><strong><span data-role="human-late-ticks">0</span><span class="fact-slash">/</span><span data-role="human-dropped-ticks">0</span></strong></div>
            <div><span>Human catch-up bursts</span><strong data-role="human-catch-up-bursts">0</strong></div>
            <div><span>Human tick P50 / P95</span><strong data-role="human-tick-p50-p95">0 / 0 ms</strong></div>
            <div><span>Human response latency</span><strong data-role="human-response-latency">—</strong></div>
            <div><span>Human frame repeat / sticky</span><strong><span data-role="human-frame-repeat">1</span><span class="fact-slash">/</span><span data-role="human-sticky">0</span></strong></div>
            <div><span>Agent decisions / sec</span><strong data-role="agent-decision-rate">0</strong></div>
            <div><span>Agent decision count</span><strong data-role="agent-decision-count">0</strong></div>
            <div><span>Agent decision P50 / P95</span><strong data-role="agent-decision-p50-p95">0 / 0 ms</strong></div>
            <div><span>Agent raw frame delta</span><strong data-role="agent-raw-frame-delta">0</strong></div>
            <div><span>Difficulty</span><strong data-role="debug-difficulty">HARD</strong></div>
            <div><span>Mistake rate</span><strong data-role="difficulty-rate">0%</strong></div>
            <div><span>Greedy action</span><strong data-role="greedy-action">NOOP</strong></div>
            <div><span>Mistake injected</span><strong data-role="mistake-injected">no</strong></div>
            <div><span>Episode return</span><strong data-role="episode-return">0</strong></div>
            <div><span>Inference latency</span><strong data-role="inference-latency">—</strong></div>
            <div><span>Frame / step</span><strong data-role="agent-frame">0 / 0</strong></div>
            <div><span>Auto-FIRE</span><strong data-role="auto-fire">0</strong></div>
            <div><span>Backend evidence</span><strong data-role="backend-evidence">—</strong></div>
            <div><span>Action agreement</span><strong data-role="action-agreement">—</strong></div>
            <div><span>Max / mean Q error</span><strong><span data-role="max-error">—</span><span class="fact-slash">/</span><span data-role="mean-error">—</span></strong></div>
            <div><span>Q-margin min / mean</span><strong data-role="q-margin">—</strong></div>
            <div><span>Max margin error</span><strong data-role="margin-error">—</strong></div>
            <div><span>Disagreements</span><strong data-role="disagreements">—</strong></div>
            <div><span>Evidence artifact / page</span><strong><span data-role="evidence-artifact">—</span><span class="fact-slash">/</span><span data-role="evidence-page">—</span></strong></div>
            <div><span>Fixed samples</span><strong data-role="sample-count">0 samples</strong></div>
          </div>
          <div class="q-values gameplay-q-values" data-role="gameplay-q-values" aria-label="current agent Q-values"><p class="muted">Q-values appear after the first real agent decision.</p></div>
          <div class="preprocess-debug" data-role="preprocess-debug">
            <div class="debug-heading"><span>MODEL INPUT / PREPROCESSING</span><strong data-role="preprocess-shape">uint8 (4, 84, 84) → float32 / 255</strong></div>
            <div class="debug-frame-row">
              <div><canvas data-role="preprocess-frame" width="84" height="84" aria-label="Latest processed grayscale frame"></canvas><span>latest 84×84</span></div>
              <div class="stack-grid" aria-label="four-frame stack">
                ${[0, 1, 2, 3].map((index) => `<div><canvas data-stack-index="${index}" width="84" height="84" aria-label="Frame stack ${index + 1}"></canvas><span>t-${3 - index}</span></div>`).join('')}
              </div>
            </div>
          </div>
          <p class="model-hash">Model SHA256: <code data-role="model-sha">—</code></p>
        </section>
        <section class="evidence-card" aria-live="polite">
          <div class="evidence-heading"><div><p class="eyebrow">VALIDATION / EVALUATION</p><h2>Raw evidence</h2></div><div class="evidence-count"><span data-role="evaluation-status">idle</span><span class="count-label">status</span></div></div>
          <div class="evidence-rule"><span>GAMEPLAY RULE</span><strong>Score is raw ALE reward; fixed-state parity and multi-episode evaluation are separate evidence.</strong></div>
          <div class="benchmark-summary" data-role="benchmark-summary" hidden><div class="benchmark-heading"><span>BENCHMARK / BATCH=1</span><strong data-role="benchmark-status">idle</strong></div></div>
          <div class="evaluation-summary" data-role="evaluation-summary" hidden>
            <div class="benchmark-heading"><span>50 EPISODES / BROWSER POLICY</span><strong data-role="evaluation-result">idle</strong></div>
            <div class="benchmark-grid"><div><span>MEAN / MEDIAN</span><strong data-role="evaluation-mean">—</strong></div><div><span>P10 / P90</span><strong data-role="evaluation-p10-p90">—</strong></div><div><span>MIN / MAX</span><strong data-role="evaluation-min-max">—</strong></div><div><span>SUCCESS / CRASH</span><strong data-role="evaluation-success">—</strong></div></div>
            <p data-role="evaluation-message">尚未執行多局 evaluation。</p>
          </div>
        </section>
      </div>
    </details>
  `;
}

function renderFooter(): string {
  return '<footer class="page-foot"><span>BREAKOUT</span></footer>';
}

export function renderQValuesMarkup(result: FixtureValidationResult): string {
  return renderPolicyQValuesMarkup(result.representative.qValues, result.representative.actionIndex, `REPRESENTATIVE FIXTURE #${result.representative.sampleIndex}`, 'Browser / Reference', result.representative.referenceQValues);
}

export function renderPolicyQValuesMarkup(
  values: readonly number[],
  actionIndex: number,
  caption = 'LIVE AGENT DECISION',
  comparisonLabel = 'Q-value',
  referenceValues?: readonly number[],
): string {
  const scale = Math.max(1, ...values.map((value) => Math.abs(value)), ...(referenceValues ?? []).map((value) => Math.abs(value)));
  const hasReference = referenceValues !== undefined;
  return `
    <div class="q-caption"><span>${caption}</span><span>${comparisonLabel}</span></div>
    <div class="q-grid" role="table">
      <div class="q-header"><span>ACTION</span><span>Q-VALUE</span>${hasReference ? '<span>REFERENCE</span>' : ''}</div>
      ${ACTION_MEANINGS.map((action, index) => renderQRow(action, index, values[index] ?? 0, scale, index === actionIndex, hasReference ? referenceValues?.[index] : undefined)).join('')}
    </div>
    <div class="q-legend"><span><i class="legend-swatch legend-swatch-browser"></i>${hasReference ? 'Browser output' : 'Current policy output'}</span>${hasReference ? '<span><i class="legend-swatch legend-swatch-reference"></i>Reference</span>' : ''}</div>
  `;
}

function renderQRow(action: ActionMeaning, index: number, value: number, scale: number, selected: boolean, referenceValue?: number): string {
  const reference = referenceValue === undefined ? '' : `<span class="q-measure" role="cell"><span class="q-track q-track-reference"><span class="q-fill q-fill-reference" style="width:${Math.min(100, (Math.abs(referenceValue) / scale) * 100).toFixed(2)}%"></span></span><code>${referenceValue.toFixed(6)}</code></span>`;
  return `
    <div class="q-row${selected ? ' q-row-selected' : ''}" role="row">
      <span class="q-action" role="cell">${action}${selected ? ' ◀' : ''}</span>
      <span class="q-measure" role="cell"><span class="q-track"><span class="q-fill q-fill-browser" style="width:${Math.min(100, (Math.abs(value) / scale) * 100).toFixed(2)}%"></span></span><code>${value.toFixed(6)}</code></span>
      ${reference}
    </div>
  `;
}
