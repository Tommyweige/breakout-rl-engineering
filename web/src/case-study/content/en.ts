import type { CaseStudyCopy } from './types';

const GITHUB = 'https://github.com/Tommyweige/breakout-rl-engineering';

export const enCopy: CaseStudyCopy = {
  header: {
    brandHomeLabel: 'Breakout RL home',
    brandMark: 'BREAKOUT',
    brandSub: 'RESEARCH RESULTS',
    navLabel: 'Primary',
    playDemo: 'Play demo',
    evidence: 'Evidence',
    github: 'GitHub',
    languageSwitch: {
      href: '/case-study.zh-TW.html',
      hreflang: 'zh-TW',
      lang: 'zh-Hant-TW',
      label: '中文',
      ariaLabel: 'Switch to Chinese',
    },
  },
  hero: {
    eyebrow: 'FROZEN EVIDENCE ·',
    titleLead: 'Does Breakout',
    titleTail: 'really need training?',
    lede: 'From a DQN family to an explainable pixel tracker and predictive controller. Two decision paradigms, one open question — and no verified full clear yet.',
    hook: 'This is the evidence-led closeout of an end-to-end RL engineering project. The interesting result is not a high score. It is how much the answer changes when the control contract changes, and how careful you have to be before calling a game “solved”.',
    playCta: 'Play the demo',
    evidenceCta: 'Read the evidence',
    summaryLoading: 'Loading the machine-readable summary…',
    summaryLoaded: 'Summary loaded: {metrics} metrics, {experiments} experiment families.',
    summaryError:
      'The machine-readable summary could not be loaded. The evidence index is still available on GitHub. ',
    summaryErrorLink: 'Open the results index',
    videoAria:
      'Recorded Breakout gameplay: the human player on the left and the AI policy on the right.',
    videoFallback: 'Your browser cannot play this recording. The',
    posterLink: 'poster frame',
    videoFallbackAfter: 'shows the same real gameplay.',
    figcaption:
      'Real capture from the built app. The left panel is the human surface; the right panel is the AI policy. No mock frames or synthesized gameplay.',
  },
  common: {
    notRun: 'Not run',
    yes: 'Yes',
    no: 'No',
    unknown: 'Unknown',
    loading: 'Loading…',
    source: 'source',
    missingMetric: '—',
  },
  status: {
    labels: {
      verified: 'Verified',
      historical_reported: 'Historical reported',
      missing_raw_artifacts: 'Missing raw artifacts',
      not_yet_evaluated: 'Not yet evaluated',
    },
    descriptions: {
      verified:
        'Checked against an artifact in this repository or a rerunnable command.',
      historical_reported:
        'Reported in a merged PR, issue, or archived article; raw run output is not in main.',
      missing_raw_artifacts:
        'Described, but the raw episodes, logs, or weights needed to audit it are unavailable.',
      not_yet_evaluated: 'No completed evaluation exists for this claim.',
    },
  },
  metricCards: [
    {
      metricId: 'verified_clears_recorded',
      label: 'Verified full clears',
      note: 'Audited {{metric:clear_score_threshold}}-point detector; absence of record through the freeze date.',
    },
    {
      metricId: 'dqn_final_holdout_mean',
      label: 'DQN final holdout mean',
      note: 'Dueling Double DQN, {{metric:dqn_training_transitions}} transitions, Contract v2.',
    },
    {
      metricId: 'rgb_controller_mean_score',
      label: 'RGB controller mean',
      note: 'Deterministic pixel controller, Contract v3.',
    },
    {
      metricId: 'laya_complete_episodes',
      label: 'Laya complete episodes',
      note: 'Frozen {{metric:laya_parameter_count}} vision model; latency was measured, gameplay was not completed.',
    },
  ],
  metricsAria: 'Headline results',
  readingKey: {
    ariaLabel: 'How to read the evidence status',
    title: 'How to read every claim',
  },
  problem: {
    index: '01',
    kicker: 'THE PROBLEM',
    title: 'Breakout is a control problem disguised as a score.',
    paragraphs: [
      'The ball and paddle are visually separable. Between collisions the motion is close to piecewise linear, and a controller can re-plan after every observation. That makes Breakout a useful test of a simple question: how much of the task actually needs a learned policy?',
      'The catch is the environment contract. Frame skip, sticky actions, FIRE handling, life-loss handling, and the episode limit all change the control problem. A score without its contract is not a result; it is a number looking for a footnote.',
    ],
    callout:
      '<strong>The clear condition is explicit.</strong> Destroy both brick walls in ALE/Breakout-v5 one-player Game 1. The audited maximum is {{metric:clear_score_threshold}} points, read from the ROM scoreboard in ALE RAM and cross-checked against raw reward. An episode that ends is not automatically a clear.',
    diagramAria: 'Two decision pipelines compared',
    diagrams: [
      {
        label: 'LEARNED POLICY',
        steps: ['{{contract:v2.frame_stack}} stacked frames', 'CNN', 'Q-values', 'argmax'],
        note: 'Contract v2: one decision per {{contract:v2.frame_skip}} emulator frames.',
      },
      {
        label: 'EXPLICIT CONTROLLER',
        steps: ['RGB frame', 'ball/paddle', 'velocity', 'intercept', 'action'],
        note: 'Contract v3: one decision per emulator frame. No training.',
        accent: true,
      },
    ],
  },
  method: {
    index: '02',
    kicker: 'METHOD EVOLUTION',
    title: 'Build the baseline, then remove the parts that do not earn their keep.',
    items: [
      {
        step: '01',
        title: 'DQN family under Contract v2',
        body: 'DQN → Double DQN → Dueling Double DQN. The family screen selected Dueling Double DQN at an early checkpoint and kept it. The final policy trained for {{metric:dqn_training_transitions}} transitions.',
        status: 'historical_reported',
      },
      {
        step: '02',
        title: 'Controlled experiments that did not promote',
        body: 'Life-loss reward shaping produced a positive but inconsistent point estimate; prioritized replay was descriptive at best. Neither replaced the baseline. These are results, not footnotes.',
        status: 'historical_reported',
      },
      {
        step: '03',
        title: 'A deterministic pixel controller under Contract v3',
        body: 'Detect the ball and paddle, estimate velocity, predict the paddle-plane intercept with wall reflections, and close the loop on observed paddle position. The controller runs at about {{metric:rgb_controller_latency_ms_per_frame}} ms per controller frame — but it did not clear the game.',
        status: 'historical_reported',
      },
      {
        step: '04',
        title: 'A frozen vision model as a side experiment',
        body: 'A {{metric:laya_parameter_count}}-parameter SmolVLM-based checkpoint, already exposed to some Breakout expert data, was tested as a frozen policy. The latency work is real; the gameplay evidence is not complete.',
        status: 'missing_raw_artifacts',
      },
    ],
  },
  results: {
    index: '03',
    kicker: 'RESULTS AT A GLANCE',
    title: 'Every number carries its contract and its evidence status.',
    intro:
      'The table is generated from <a href="/final-results/summary.json">summary.json</a>. The page does not keep a second hand-typed copy of these values. “Historical reported” means the raw run is not in <code>main</code>; it does not mean the number was reproduced here.',
    headers: {
      metric: 'Metric',
      value: 'Value',
      scope: 'Seeds / scope',
      evidence: 'Evidence',
      source: 'Source',
    },
  },
  fairness: {
    index: '04',
    kicker: 'CONTRACTS AND FAIRNESS',
    title: 'The two scores answer different questions.',
    intro:
      'Contract v2 gives the policy one decision per {{contract:v2.frame_skip}} emulator frames. Contract v3 gives it one decision per frame. Both keep the same {{contract:v2.max_num_frames_per_episode}}-frame ALE episode limit, but the agent-step budgets are {{contract:v2.agent_step_limit}} and {{contract:v3.agent_step_limit}}. A raw-score comparison across them is descriptive, not a ranking.',
    headers: {
      property: 'Property',
      v2: 'Contract v2',
      v3: 'Contract v3',
    },
    rows: [
      'Contract ID',
      'File',
      'SHA-256',
      'Decision cadence',
      'Agent step limit',
      'ALE episode-frame limit',
      'Frame stack',
      'Sticky actions',
      'FIRE reset',
      'Terminal on life loss',
      'Evaluation seeds',
      'Evaluation epsilon',
    ],
    valuePhrases: {
      decisionPerFrames: '1 per {n} emulator frames',
      decisionPerFrame: '1 per emulator frame',
      seedCount: '{n} seeds',
      yes: 'yes',
      no: 'no',
    },
    comparisonNote:
      'DQN (v2) vs RGB predictive controller (v3): not directly comparable. Contract v2 makes one decision per {{contract:v2.frame_skip}} emulator frames; v3 makes one per emulator frame. The evaluation seed roles and episode sets also differ. The raw scores are descriptive, not a fair ranking or a method win.',
    comparisonFallback:
      'Cross-contract comparison is not directly comparable because the decision cadence differs.',
    notes: [
      '<strong>No fair winner claim.</strong> The DQN holdout and the RGB controller were not run under one matched contract, seed set, episode budget, and command.',
      '<strong>No method ranking from raw score.</strong> A higher score under one cadence does not establish a better method under another.',
      '<strong>No post-hoc promotion.</strong> A matched comparison would have to be predeclared before looking at its results.',
    ],
  },
  laya: {
    index: '05',
    kicker: 'SIDE EXPERIMENT',
    title: 'Laya Vision: real latency, a measured gameplay session, no complete evaluation.',
    paragraphs: [
      {
        lead: 'Motivation.',
        body: 'Test whether a small, frozen, non-generative vision decision model can choose legal Breakout actions from pixels at a practical inference cost.',
      },
      {
        lead: 'Frozen hypothesis.',
        body: 'A {{metric:laya_parameter_count}}-parameter SmolVLM-based checkpoint can act as a frozen policy without project-specific training. It was already trained on some Breakout expert-action data, so this is <em>not</em> a zero-shot claim.',
      },
      {
        lead: 'What was measured.',
        body: 'Warm decision latency, a bounded control diagnostic, and a live browser session. Warm p50 was {{metric:laya_warm_latency_p50_ms}} ms on one fixed RGB fixture; the optimized local proxy p50 was {{metric:laya_optimized_proxy_p50_ms}} ms across repeated rounds. Immediate reversal commands went from {{metric:laya_control_diagnostic_raw_reversals}} to {{metric:laya_control_diagnostic_reversals}} in a {{seeds:laya_control_diagnostic_reversals}}-seed control diagnostic. In the browser session the Decision Model scored {{metric:laya_browser_session_score}} and lost every life after {{metric:laya_browser_session_played_s}} s of play, presenting {{metric:laya_browser_presented_fps}} frames per second.',
      },
      {
        lead: 'What is still not measured.',
        body: 'No complete evaluator episode, per-episode artifact, or clear exists, so the browser session is not comparable with evaluated episodes. Latency does not imply gameplay quality.',
      },
    ],
    callout:
      '<strong>Decision:</strong> keep Laya as a research lead. Draft PR #74 stays draft until a complete evaluated episode exists. #70–#72 stay open.',
    aside: {
      ariaLabel: 'Laya evidence summary',
      heading: 'Laya at freeze time',
      facts: [
        { label: 'Complete episodes', value: '{{metric:laya_complete_episodes}}', unit: 'complete episodes' },
        { label: 'Warm p50', value: '{{metric:laya_warm_latency_p50_ms}}', unit: 'ms/decision' },
        { label: 'Optimized proxy p50', value: '{{metric:laya_optimized_proxy_p50_ms}}', unit: 'ms/request' },
        {
          label: 'Reversals, raw → braked',
          value:
            '{{metric:laya_control_diagnostic_raw_reversals}} → {{metric:laya_control_diagnostic_reversals}}',
        },
      ],
      note: 'Weights are CC BY-NC-SA 4.0 and are not in the repository.',
      researchLinkLabel: 'Research issue',
      prLinkLabel: 'Draft PR',
    },
  },
  negative: {
    index: '06',
    kicker: 'FAILURES AND NEGATIVE RESULTS',
    title: 'The experiments that did not promote are part of the result.',
    cards: [
      {
        kicker: 'REWARD SHAPING · ISSUE #9',
        value: '{{metric:reward_shaping_mean_effect}}',
        unit: 'mean effect',
        body: '95% interval {{metric:reward_shaping_ci_low}} to {{metric:reward_shaping_ci_high}}. {{metric:reward_shaping_positive_training_seeds}} of {{seeds:reward_shaping_mean_effect}} training seeds were positive. The promotion gate did not pass, so the longer comparison was not run.',
        status: 'historical_reported',
      },
      {
        kicker: 'PRIORITIZED REPLAY · ISSUE #12',
        value: '{{metric:per_500k_mean_score}}',
        unit: 'PER mean at checkpoint',
        body: 'Uniform replay was {{metric:uniform_500k_mean_score}} at the same milestone. PER cost {{metric:per_training_time_ratio}}× training time. With {{seeds:per_500k_mean_score}} seeds, this is descriptive, not statistical superiority.',
        status: 'historical_reported',
      },
      {
        kicker: 'FIRST-CLEAR PROBES · #41, #43, #45, #48–#61',
        value: '{{metric:verified_clears_recorded}}',
        unit: 'verified clears',
        body: 'The frozen probes ended without a verified clear. Several were explicitly INCONCLUSIVE or BLOCKED_ENVIRONMENT. They identify failure modes; they do not prove that a method cannot clear.',
        status: 'historical_reported',
      },
    ],
  },
  ledger: {
    index: '07',
    kicker: 'PUBLIC EVIDENCE',
    title: 'The ledger, not the highlight reel.',
    intro:
      'Every experiment family in the closeout, with its status and source. Titles, headlines, and notes are quoted verbatim from <code>summary.json</code> in the original language; the CSV and JSON are the machine-readable versions of this page.',
    downloads: [
      { href: `${GITHUB}/blob/main/docs/final-results/evidence-inventory.csv`, label: 'evidence-inventory.csv' },
      { href: `${GITHUB}/blob/main/docs/final-results/README.md`, label: 'Full results index' },
      { href: `${GITHUB}/blob/main/docs/final-results/handoff.md`, label: 'Handoff and blockers' },
    ],
  },
  cta: {
    kicker: 'DEMO + SOURCE',
    title: 'Play the current build, then inspect the evidence behind it.',
    body: 'The demo runs ALE WASM and the ONNX policy entirely in the browser. “Unbeatable” is a UI difficulty label: it injects no deliberate mistakes. It is not a claim that the policy cannot lose.',
    playCta: 'Play the demo',
    githubCta: 'GitHub repository',
  },
  footer: {
    brand: 'BREAKOUT · RESEARCH RESULTS',
    updated: 'Frozen evidence closeout. Last updated',
    notClaimedLabel: 'Not claimed:',
    notClaimed:
      'solved game, verified clear, reliable clear, world record, fair method winner, or a complete evaluated Laya episode.',
    playDemo: 'Play demo',
    evidence: 'Evidence',
    github: 'GitHub',
  },
};
