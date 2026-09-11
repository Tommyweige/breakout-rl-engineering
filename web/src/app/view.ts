import { ACTION_MEANINGS, type FixtureValidationResult } from '../inference/types';

export function renderAppShell(): string {
  return [renderMasthead(), renderCommandDeck(), renderPanelGrid(), renderEvidence(), renderFooter()].join('');
}

function renderMasthead(): string {
  return `
    <header class="masthead">
      <div class="brand-line">
        <div class="brand-lockup"><span class="brand-mark">BRK / 28</span><span class="brand-name">Browser Backend Lab</span></div>
        <span class="brand-note">client-side policy lab</span>
      </div>
      <div class="hero">
        <div class="hero-copy">
          <h1>Teach the browser <em>what the agent sees.</em></h1>
          <p class="subtitle">同一份 FP32 ONNX，在同一個 Browser 裡比較 WASM 與 WebGPU。今天讀的是 correctness 與 latency signal，不是假裝已經接上 ALE gameplay。</p>
        </div>
        <div class="hero-facts" aria-label="runtime brief">
          <div><span>DEPLOYMENT</span><strong>client-side</strong></div>
          <div><span>MODEL</span><strong>FP32 ONNX</strong></div>
          <div><span>RUNTIME</span><strong>ORT Web / WASM + WebGPU</strong></div>
        </div>
      </div>
      <div class="masthead-foot">
        <span>DAY 28 / WEBGPU INFERENCE</span>
        <div class="runtime-badge" data-role="runtime-status" data-status="idle"><span class="status-dot"></span><span data-role="runtime-status-label">idle</span></div>
      </div>
    </header>
  `;
}

function renderCommandDeck(): string {
  return `
    <section class="command-deck" aria-label="shared controls">
      <div class="deck-label"><span class="deck-index">01</span><div><strong>Shared controls</strong><small>One state. Two panels.</small></div></div>
      <div class="controls">
        <button data-action="validate" type="button" class="primary"><span class="button-key" aria-hidden="true">↳</span>Load + Validate selected backend</button>
        <button data-action="benchmark" type="button"><span class="button-key" aria-hidden="true">◫</span>Run WASM / WebGPU benchmark</button>
        <button data-action="start" type="button"><span class="button-key" aria-hidden="true">↗</span>Start Both</button>
        <button data-action="pause" type="button"><span class="button-key" aria-hidden="true">Ⅱ</span>Pause Both</button>
        <button data-action="reset" type="button"><span class="button-key" aria-hidden="true">↺</span>Reset Both</button>
      </div>
      <div class="deck-downloads">
        <a data-role="download-validation" class="download-link" download hidden>Download validation JSON <span>↓</span></a>
        <a data-role="download-benchmark" class="download-link" download="web-benchmark.json" hidden>Download benchmark JSON <span>↓</span></a>
      </div>
    </section>
  `;
}

function renderPanelGrid(): string {
  return `
    <section class="panel-grid" aria-label="dual browser foundation panels">
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
        <span class="status-chip status-chip-muted">Shell only</span>
      </div>
      <div class="stage-placeholder signal-stage">
        <div class="stage-topline"><span>INPUT CHANNEL / HUMAN_01</span><span class="stage-state">NO ALE FEED</span></div>
        <div class="stage-center">
          <span class="signal-crosshair" aria-hidden="true"></span>
          <span class="stage-code">WAIT / 000</span>
          <strong>ALE gameplay not connected yet</strong>
          <span>今天只觀察鍵盤輸入與 shared panel state。</span>
        </div>
        <div class="stage-corner">keyboard ready<br><span>3 mapped inputs</span></div>
      </div>
      <div class="key-grid" aria-label="keyboard controls">
        <div class="key-card"><kbd>←</kbd><span>ArrowLeft</span></div>
        <div class="key-card"><kbd>→</kbd><span>ArrowRight</span></div>
        <div class="key-card key-card-wide"><kbd>SPACE</kbd><span>Space</span></div>
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
      <div class="agent-banner">
        <div><span class="banner-label">MODEL CHANNEL</span><strong>FP32 ONNX / Q-VALUE HEAD</strong></div>
        <span class="banner-mark">ORT</span>
      </div>
      <dl class="facts compact-facts">
        <div><dt>Model</dt><dd data-role="model-loaded">not loaded</dd></div>
        <div><dt>Requested / actual</dt><dd><span data-role="requested-backend">WASM</span><span class="fact-slash">/</span><span data-role="actual-backend">—</span></dd></div>
        <div><dt>WebGPU support</dt><dd data-role="webgpu-support" data-status="checking">checking…</dd></div>
        <div><dt>ORT Web</dt><dd data-role="ort-version">—</dd></div>
        <div><dt>Browser / platform</dt><dd data-role="browser-info">—</dd></div>
        <div><dt>Environment parity</dt><dd data-role="environment-parity">—</dd></div>
        <div><dt>Backend evidence</dt><dd data-role="backend-evidence">—</dd></div>
        <div><dt>Selected action</dt><dd data-role="selected-action">—</dd></div>
        <div><dt>Action agreement</dt><dd data-role="action-agreement">—</dd></div>
        <div><dt>Max / mean Q error</dt><dd><span data-role="max-error">—</span><span class="fact-slash">/</span><span data-role="mean-error">—</span></dd></div>
        <div><dt>Q-margin min / mean</dt><dd data-role="q-margin">—</dd></div>
        <div><dt>Max margin error</dt><dd data-role="margin-error">—</dd></div>
        <div><dt>Disagreements</dt><dd data-role="disagreements">—</dd></div>
        <div><dt>Inference scheduler</dt><dd data-role="scheduler-status">idle</dd></div>
        <div><dt>Evidence artifact / page</dt><dd><span data-role="evidence-artifact">—</span><span class="fact-slash">/</span><span data-role="evidence-page">—</span></dd></div>
      </dl>
      <div class="q-values" data-role="q-values" aria-label="representative fixture Q-values">
        <p class="muted">Q-values appear after a real fixture inference.</p>
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
          <h2>Browser validation</h2>
        </div>
        <div class="evidence-count"><span data-role="sample-count">0 samples</span><span class="count-label">fixed states</span></div>
      </div>
      <p data-role="validation-message">尚未執行。結果必須來自 Browser 的真實 ORT Web WASM inference。</p>
      <div class="evidence-rule"><span>FIXED STATE RULE</span><strong>Browser output stays within the Day 22 reference envelope.</strong><span class="rule-mark">/ 60</span></div>
      <div class="benchmark-summary" data-role="benchmark-summary" hidden>
        <div class="benchmark-heading"><span>BENCHMARK / BATCH=1</span><strong data-role="benchmark-status">idle</strong></div>
      </div>
    </section>
  `;
}

function renderFooter(): string {
  return '<footer class="page-foot"><span>BREAKOUT RL ENGINEERING</span><span>NO GAMEPLAY SIMULATION IN THIS BUILD</span><span>DAY 28 / 2026</span></footer>';
}

export function renderQValuesMarkup(result: FixtureValidationResult): string {
  return `
    <div class="q-caption"><span>REPRESENTATIVE FIXTURE #${result.representative.sampleIndex}</span><span>Browser / Reference</span></div>
    <div class="q-grid" role="table">
      <div class="q-header"><span>ACTION</span><span>BROWSER</span><span>REFERENCE</span></div>
      ${ACTION_MEANINGS.map((action, index) => `
        <div class="q-row" role="row">
          <span class="q-action" role="cell">${action}</span>
          <span class="q-measure" role="cell"><span class="q-track"><span class="q-fill q-fill-browser" data-q-kind="actual-bar" data-q-index="${index}"></span></span><code data-q-kind="actual" data-q-index="${index}">—</code></span>
          <span class="q-measure" role="cell"><span class="q-track q-track-reference"><span class="q-fill q-fill-reference" data-q-kind="reference-bar" data-q-index="${index}"></span></span><code data-q-kind="reference" data-q-index="${index}">—</code></span>
        </div>
      `).join('')}
    </div>
    <div class="q-legend"><span><i class="legend-swatch legend-swatch-browser"></i>Browser output</span><span><i class="legend-swatch legend-swatch-reference"></i>Reference</span></div>
  `;
}
