import { ACTION_MEANINGS, type ActionMeaning, type FixtureValidationResult } from '../inference/types';

export function renderAppShell(): string {
  return [renderMasthead(), renderCommandDeck(), renderPanelGrid(), renderEvidence(), renderFooter()].join('');
}

function renderMasthead(): string {
  return `
    <header class="masthead">
      <div class="brand-line">
        <div class="brand-lockup"><span class="brand-mark">BRK / 29</span><span class="brand-name">Human vs RL / ALE Browser</span></div>
        <span class="brand-note">client-side dual environment</span>
      </div>
      <div class="hero">
        <div class="hero-copy">
          <h1>Two games.<br><em>One policy.</em></h1>
          <p class="subtitle">同一個 Browser 頁面同時執行兩個真正的 ALE Breakout：左側由鍵盤控制，右側把 Day 21 Final Model 經過 ONNX Runtime Web 推論後交回 Atari 2600 emulator。</p>
        </div>
        <div class="hero-facts" aria-label="runtime brief">
          <div><span>ENVIRONMENT</span><strong>ALE / Breakout-v5</strong></div>
          <div><span>MODEL</span><strong>Day 21 FP32 ONNX</strong></div>
          <div><span>RUNTIME</span><strong>ALE WASM / ORT Web</strong></div>
        </div>
      </div>
      <div class="masthead-foot">
        <span>DAY 29 / HUMAN VS RL DUAL BREAKOUT</span>
        <div class="runtime-badge" data-role="runtime-status" data-status="idle"><span class="status-dot"></span><span data-role="runtime-status-label">idle</span></div>
      </div>
    </header>
  `;
}

function renderCommandDeck(): string {
  return `
    <section class="command-deck" aria-label="shared controls">
      <div class="deck-label"><span class="deck-index">01</span><div><strong>Shared controls</strong><small>One clock. Two independent ALE states.</small></div></div>
      <div class="controls">
        <button data-action="validate" type="button" class="secondary"><span class="button-key" aria-hidden="true">↳</span>Validate backend</button>
        <button data-action="benchmark" type="button"><span class="button-key" aria-hidden="true">◫</span>Backend benchmark</button>
        <button data-action="evaluate" type="button"><span class="button-key" aria-hidden="true">◎</span>Run 30-episode evaluation</button>
        <button data-action="start" type="button" class="primary"><span class="button-key" aria-hidden="true">↗</span>Start Both</button>
        <button data-action="pause" type="button"><span class="button-key" aria-hidden="true">Ⅱ</span>Pause Both</button>
        <button data-action="reset" type="button"><span class="button-key" aria-hidden="true">↺</span>Reset Both</button>
      </div>
      <div class="deck-downloads">
        <a data-role="download-validation" class="download-link" download hidden>Download validation JSON <span>↓</span></a>
        <a data-role="download-benchmark" class="download-link" download="web-benchmark.json" hidden>Download benchmark JSON <span>↓</span></a>
        <a data-role="download-evaluation" class="download-link" download="browser-policy-score-comparison.json" hidden>Download score evidence <span>↓</span></a>
      </div>
    </section>
  `;
}

function renderPanelGrid(): string {
  return `
    <section class="panel-grid" aria-label="human and agent breakout panels">
      ${renderHumanPanel()}
      ${renderAgentPanel()}
    </section>
  `;
}

function renderHumanPanel(): string {
  return `
    <article class="panel panel-human" aria-labelledby="human-title">
      <div class="panel-heading">
        <div>
          <p class="panel-kicker"><span class="panel-index">02</span> YOU / HUMAN</p>
          <h2 id="human-title">Human Panel</h2>
        </div>
        <span class="status-chip status-chip-muted" data-role="human-state">waiting</span>
      </div>
      <div class="game-stage human-game-stage">
        <div class="stage-topline"><span>ALE / HUMAN_01</span><span class="stage-state" data-role="human-stage-state">READY</span></div>
        <canvas data-role="human-canvas" width="160" height="210" aria-label="Human Breakout game canvas"></canvas>
      </div>
      <dl class="game-facts human-facts">
        <div><dt>Score / return</dt><dd data-role="human-score">0</dd></div>
        <div><dt>Lives</dt><dd data-role="human-lives">—</dd></div>
        <div><dt>Frame / step</dt><dd data-role="human-frame">0 / 0</dd></div>
        <div><dt>Action</dt><dd data-role="human-action">NOOP</dd></div>
      </dl>
      <div class="key-grid" aria-label="keyboard controls">
        <div class="key-card"><kbd>←</kbd><span>ArrowLeft</span></div>
        <div class="key-card"><kbd>→</kbd><span>ArrowRight</span></div>
        <div class="key-card key-card-wide"><kbd>SPACE</kbd><span>FIRE / serve</span></div>
      </div>
      <div class="live-line"><span>LIVE INPUT</span><strong data-role="human-input">none</strong></div>
    </article>
  `;
}

function renderAgentPanel(): string {
  return `
    <article class="panel panel-agent" aria-labelledby="agent-title">
      <div class="panel-heading">
        <div>
          <p class="panel-kicker"><span class="panel-index">03</span> RL AGENT</p>
          <h2 id="agent-title">Agent Panel</h2>
        </div>
        <div class="panel-actions">
          <label class="backend-picker" for="backend-select"><span>AGENT BACKEND</span><select id="backend-select" data-action="backend" aria-label="Agent inference backend"><option value="wasm" selected>WASM / CPU baseline</option><option value="webgpu">WebGPU / GPU</option></select></label>
          <span class="status-chip" data-role="validation-status">Not validated</span>
        </div>
      </div>
      <div class="game-stage agent-game-stage">
        <div class="stage-topline"><span>ALE / RL_AGENT_01</span><span class="stage-state" data-role="agent-stage-state">READY</span></div>
        <canvas data-role="agent-canvas" width="160" height="210" aria-label="RL Agent Breakout game canvas"></canvas>
        <div class="agent-overlay"><span>ACTUAL BACKEND</span><strong data-role="gameplay-backend">—</strong></div>
      </div>
      <div class="agent-banner">
        <div><span class="banner-label">MODEL CHANNEL</span><strong>DAY 21 FINAL MODEL / FP32 ONNX</strong></div>
        <span class="banner-mark">ORT</span>
      </div>
      <dl class="facts compact-facts">
        <div><dt>Model</dt><dd data-role="model-loaded">not loaded</dd></div>
        <div><dt>Requested / actual</dt><dd><span data-role="requested-backend">WASM</span><span class="fact-slash">/</span><span data-role="actual-backend">—</span></dd></div>
        <div><dt>WebGPU support</dt><dd data-role="webgpu-support" data-status="checking">checking…</dd></div>
        <div><dt>ORT Web</dt><dd data-role="ort-version">—</dd></div>
        <div><dt>Browser / platform</dt><dd data-role="browser-info">—</dd></div>
        <div><dt>Environment parity</dt><dd data-role="environment-parity">—</dd></div>
        <div><dt>Current action</dt><dd data-role="current-action">NOOP</dd></div>
        <div><dt>Episode return</dt><dd data-role="episode-return">0</dd></div>
        <div><dt>Inference latency</dt><dd data-role="inference-latency">—</dd></div>
        <div><dt>Frame / step</dt><dd data-role="agent-frame">0 / 0</dd></div>
        <div><dt>Auto-FIRE</dt><dd data-role="auto-fire">0</dd></div>
        <div><dt>Action mapping</dt><dd data-role="action-mapping">0/1/2/3 → 0/1/3/4</dd></div>
        <div><dt>Backend evidence</dt><dd data-role="backend-evidence">—</dd></div>
        <div><dt>Action agreement</dt><dd data-role="action-agreement">—</dd></div>
        <div><dt>Max / mean Q error</dt><dd><span data-role="max-error">—</span><span class="fact-slash">/</span><span data-role="mean-error">—</span></dd></div>
        <div><dt>Q-margin min / mean</dt><dd data-role="q-margin">—</dd></div>
        <div><dt>Max margin error</dt><dd data-role="margin-error">—</dd></div>
        <div><dt>Disagreements</dt><dd data-role="disagreements">—</dd></div>
        <div><dt>Inference scheduler</dt><dd data-role="scheduler-status">idle</dd></div>
        <div><dt>Evidence artifact / page</dt><dd><span data-role="evidence-artifact">—</span><span class="fact-slash">/</span><span data-role="evidence-page">—</span></dd></div>
      </dl>
      <div class="q-values gameplay-q-values" data-role="gameplay-q-values" aria-label="current agent Q-values">
        <p class="muted">Q-values appear after the first real agent decision.</p>
      </div>
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
    </article>
  `;
}

function renderEvidence(): string {
  return `
    <section class="evidence-card" aria-live="polite">
      <div class="evidence-heading">
        <div>
          <p class="panel-kicker"><span class="panel-index">04</span> RUNTIME EVIDENCE</p>
          <h2>Validation / evaluation</h2>
        </div>
        <div class="evidence-count"><span data-role="sample-count">0 samples</span><span class="count-label">fixed states</span></div>
      </div>
      <p data-role="validation-message">尚未執行。先驗證 selected backend，再用 Start Both 啟動兩個獨立的 ALE instance。</p>
      <div class="evidence-rule"><span>GAMEPLAY RULE</span><strong>Score is raw ALE reward; fixed-state parity and multi-episode evaluation are separate evidence.</strong><span class="rule-mark" data-role="evaluation-status">idle</span></div>
      <div class="benchmark-summary" data-role="benchmark-summary" hidden>
        <div class="benchmark-heading"><span>BENCHMARK / BATCH=1</span><strong data-role="benchmark-status">idle</strong></div>
      </div>
      <div class="evaluation-summary" data-role="evaluation-summary" hidden>
        <div class="benchmark-heading"><span>30 EPISODES / BROWSER POLICY</span><strong data-role="evaluation-result">idle</strong></div>
        <div class="benchmark-grid">
          <div><span>MEAN / MEDIAN</span><strong data-role="evaluation-mean">—</strong></div>
          <div><span>P10 / P90</span><strong data-role="evaluation-p10-p90">—</strong></div>
          <div><span>MIN / MAX</span><strong data-role="evaluation-min-max">—</strong></div>
          <div><span>SUCCESS / CRASH</span><strong data-role="evaluation-success">—</strong></div>
        </div>
        <p data-role="evaluation-message">尚未執行多局 evaluation。</p>
      </div>
    </section>
  `;
}

function renderFooter(): string {
  return '<footer class="page-foot"><span>BREAKOUT RL ENGINEERING</span><span>ALE WASM + ORT WEB / NO SERVER INFERENCE</span><span>DAY 29 / 2026</span></footer>';
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
