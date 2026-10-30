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
    titleLead: 'Can Breakout',
    titleTail: 'be played without training?',
    lede: 'A hand-written controller with no training reads the screen, works out where the ball will arrive, and puts the paddle there. In official episodes it has cleared both brick walls more than once, counted by code rather than by eye.',
    hook: 'The whole project turns on one question: how much of this game actually needs something that learns? What is more useful than a high score is how far the answer moves when one rule changes.',
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
      verified: 'Checked against a file in this repository, or reproducible with one command.',
      historical_reported:
        'Reported in a merged PR, issue, or archived article; the raw run output is not in main.',
      missing_raw_artifacts:
        'Described in words, but the raw episodes, logs, or weights needed to audit it are gone.',
      not_yet_evaluated: 'No finished evaluation exists for this claim.',
    },
  },
  metricCards: [
    {
      metricId: 'rgb_controller_a8445df_canonical_clear_detections',
      label: 'Episodes that cleared both walls',
      note: 'In {{seeds:rgb_controller_a8445df_canonical_clear_detections}} official episodes the pixel controller emptied both brick walls {{metric:rgb_controller_a8445df_canonical_clear_detections}} times, for {{metric:clear_score_threshold}} points. The runs used native ALE under Contract v3, the rulebook that decides on every frame, at a8445df. The verdict is read from the scoreboard in the game’s memory, not from watching; when it cannot tell, the episode is recorded as unknown, not as zero.',
    },
    {
      metricId: 'rgb_controller_a8445df_mean_score',
      label: 'Mean score, screen-only controller',
      note: 'No training at all. The older PR #26 rule (50ee352) averaged {{metric:rgb_controller_pr26_mean_score}} and peaked at {{metric:rgb_controller_pr26_max_score}}, short of the 432 points the first wall takes.',
    },
    {
      metricId: 'dqn_v3_2m_mean_score',
      label: 'Mean score, model that learns, same rules',
      note: 'A Dueling Double DQN, one training seed, two million frames: {{metric:dqn_v3_2m_clears}} clears. The {{metric:dqn_final_holdout_mean}} holdout for the browser DQN came from the older v2 rules.',
    },
    {
      metricId: 'laya_complete_episodes',
      label: 'Laya episodes played to the end',
      note: 'A frozen {{metric:laya_parameter_count}}-parameter vision model. Latency was measured; the game was never finished.',
    },
  ],
  metricsAria: 'Headline results',
  readingKey: {
    ariaLabel: 'How to read the evidence status',
    title: 'Who checked this number?',
  },
  problem: {
    index: '01',
    kicker: 'THE PROBLEM',
    title: 'So what is actually hard about Breakout?',
    paragraphs: [
      'The ball and the paddle are two separate shapes on screen, and between bounces the ball travels close to a straight line, so a controller can work it out again from every frame. That makes Breakout a good place to ask how much of the task really needs a model that learns.',
      'The hard part is the rulebook. How many frames pass between decisions, whether an action gets repeated, who presses FIRE, whether losing a life ends the episode, and how long an episode may run all change the question.',
      'A score without that rulebook is a number waiting for a footnote, not a result.',
    ],
    callout:
      '<strong>What counts as clearing both walls?</strong> Emptying both brick walls in ALE/Breakout-v5 one-player Game 1, which is {{metric:clear_score_threshold}} points. That number is read from the scoreboard in the game’s memory and cross-checked against the reward the game returns; an episode ending is not a clear. When the runtime is unfamiliar or the two sources disagree, the check reports unknown rather than a zero. It has been validated on real episodes that clear both walls, and six deliberately broken versions of it were all caught.',
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
    title: 'What survives once you take away everything that does not pay for itself?',
    items: [
      {
        step: '01',
        title: 'First rulebook: let the model learn',
        body: 'The DQN approach looks only at frames and learns roughly what each action is worth. Three variants were tried; the early checkpoint already picked Dueling Double DQN, nothing replaced it, and the final policy trained for {{metric:dqn_training_transitions}} transitions. Under the first rulebook it decided once every {{contract:v2.frame_skip}} frames.',
        status: 'historical_reported',
      },
      {
        step: '02',
        title: 'Two detours that were not adopted',
        body: 'One charged extra for losing a life, hoping the model would stop dropping the ball; the direction was right but the result was unstable. The other replayed surprising frames more often, which at best described a run. Neither replaced the baseline.',
        status: 'historical_reported',
      },
      {
        step: '03',
        title: 'Second rulebook: aim the paddle instead of learning the score',
        body: 'This one does not learn. It finds the ball and the paddle, estimates the ball’s speed, folds in the wall bounces to predict where the ball meets the paddle plane, and corrects with the paddle it actually sees. The earlier version of this rule (PR #26, 50ee352) averaged {{metric:rgb_controller_pr26_mean_score}} with a best of {{metric:rgb_controller_pr26_max_score}}, never finishing the first wall, which is how the {{metric:rgb_controller_pr26_canonical_clear_detections}} was measured. #68 added a velocity anchor and released the paddle early; at a8445df on native ALE with the official {{seeds:rgb_controller_a8445df_mean_score}} seeds it averages {{metric:rgb_controller_a8445df_mean_score}}, clears both walls {{metric:rgb_controller_a8445df_canonical_clear_detections}} times, and decides in about {{metric:rgb_controller_a8445df_latency_ms_mean}} ms per frame.',
        status: 'verified',
      },
      {
        step: '04',
        title: 'A side experiment: a frozen vision model',
        body: 'A {{metric:laya_parameter_count}}-parameter SmolVLM-based checkpoint was frozen and asked to pick actions. It had already seen some Breakout expert play. Latency is measurable; the game has one browser session behind it.',
        status: 'missing_raw_artifacts',
      },
    ],
  },
  results: {
    index: '03',
    kicker: 'RESULTS AT A GLANCE',
    title: 'What conditions was each of these numbers measured under?',
    intro:
      'The table is built from <a href="/final-results/summary.json">summary.json</a>; the page keeps no second hand-typed copy. Every row carries its own evidence status and source alongside the value.',
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
    title: 'Can scores from two rulebooks be compared?',
    intro:
      'The two rulebooks differ mainly in how often a decision is allowed: the older one (v2) decides once every {{contract:v2.frame_skip}} frames, the newer one (v3) decides on every frame. Both cap an episode at {{contract:v2.max_num_frames_per_episode}} frames, but allow {{contract:v2.agent_step_limit}} and {{contract:v3.agent_step_limit}} decisions respectively. The comparison below therefore uses only numbers from the same rulebook.',
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
      'The DQN holdout was measured under the older rulebook and the pixel controller under the newer one: the older rule decides once every {{contract:v2.frame_skip}} frames and the newer one on every frame, and the seeds and episode sets play different roles. The comparison below uses only numbers from the same rulebook.',
    comparisonFallback:
      'The two rulebooks decide at different rates, so the comparison below uses only numbers from the same rulebook.',
    notes: [
      '<strong>One comparison shares a rulebook.</strong> Under the newer rules the DQN baseline, one seed at two million frames, averaged {{metric:dqn_v3_2m_mean_score}} with {{metric:dqn_v3_2m_clears}} clears; the controller averaged {{metric:rgb_controller_a8445df_mean_score}} with {{metric:rgb_controller_a8445df_canonical_clear_detections}} clear detections. The DQN’s raw episodes are not in main, so the table marks it historical.',
    ],
  },
  laya: {
    index: '05',
    kicker: 'SIDE EXPERIMENT',
    title: 'Can the Laya vision model actually play?',
    paragraphs: [
      {
        lead: 'Why try it?',
        body: 'Whether a small, frozen, non-generative vision model can pick legal Breakout actions straight from the screen, and be fast enough to use.',
      },
      {
        lead: 'What is the assumption?',
        body: 'A {{metric:laya_parameter_count}}-parameter SmolVLM-based checkpoint acts as a fixed policy, with no training for this project. It had already seen some Breakout expert play.',
      },
      {
        lead: 'What was measured?',
        body: 'Warm decision latency, a bounded control diagnostic, and one live browser session. On a single fixed frame the warm median was {{metric:laya_warm_latency_p50_ms}} ms; the optimized local proxy came in at {{metric:laya_optimized_proxy_p50_ms}} ms across repeated rounds. In a {{seeds:laya_control_diagnostic_reversals}}-seed diagnostic, immediate reversals dropped from {{metric:laya_control_diagnostic_raw_reversals}} to {{metric:laya_control_diagnostic_reversals}}.',
      },
      {
        lead: 'How did the browser session go?',
        body: 'It scored {{metric:laya_browser_session_score}}, ran out of lives after {{metric:laya_browser_session_played_s}} s, and presented {{metric:laya_browser_presented_fps}} frames per second. That was a real browser session.',
      },
      {
        lead: 'What is still missing?',
        body: 'Complete evaluated episodes: {{metric:laya_complete_episodes}}. There is no per-episode record either, so that score stands on its own. It is fast; it has not finished the game.',
      },
    ],
    callout:
      '<strong>Decision:</strong> Laya stays a research lead. Draft PR #74 stays a draft until a complete evaluated episode exists, and #70–#72 stay open.',
    aside: {
      ariaLabel: 'Laya evidence summary',
      heading: 'Laya at freeze time',
      facts: [
        { label: 'Episodes played to the end', value: '{{metric:laya_complete_episodes}}', unit: 'complete episodes' },
        { label: 'Warm median', value: '{{metric:laya_warm_latency_p50_ms}}', unit: 'ms/decision' },
        { label: 'Optimized proxy median', value: '{{metric:laya_optimized_proxy_p50_ms}}', unit: 'ms/request' },
        {
          label: 'Immediate reversals, raw → braked',
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
    title: 'Do failed experiments count as results?',
    cards: [
      {
        kicker: 'REWARD SHAPING · ISSUE #9',
        value: '{{metric:reward_shaping_mean_effect}}',
        unit: 'mean effect',
        body: 'The 95% interval ran from {{metric:reward_shaping_ci_low}} to {{metric:reward_shaping_ci_high}}, and {{metric:reward_shaping_positive_training_seeds}} of {{seeds:reward_shaping_mean_effect}} training seeds came out positive. It missed the bar for adoption, so the longer comparison was never run.',
        status: 'historical_reported',
      },
      {
        kicker: 'PRIORITIZED REPLAY · ISSUE #12',
        value: '{{metric:per_500k_mean_score}}',
        unit: 'PER mean at checkpoint',
        body: 'Uniform replay sat at {{metric:uniform_500k_mean_score}} at the same milestone. Prioritized replay cost {{metric:per_training_time_ratio}}× the training time. With {{seeds:per_500k_mean_score}} seeds this describes a run; a statistical edge would need more.',
        status: 'historical_reported',
      },
      {
        kicker: 'FIRST-CLEAR PROBES · #41, #43, #45, #48–#61',
        value: '0',
        unit: 'clears counted by code',
        body: 'The frozen DQN probes never produced a single clear of both walls. Several were marked INCONCLUSIVE or BLOCKED_ENVIRONMENT outright, meaning no verdict or a stuck environment. What these records show is a set of failure modes.',
        status: 'historical_reported',
      },
    ],
  },
  ledger: {
    index: '07',
    kicker: 'PUBLIC EVIDENCE',
    title: 'What can you check besides the video?',
    intro:
      'Every experiment family in the closeout, with its status and source. Titles, headlines, and notes are quoted from <code>summary.json</code> in their original language so translation cannot shift a number; the CSV and JSON are the machine-readable version of this page.',
    downloads: [
      { href: `${GITHUB}/blob/main/docs/final-results/evidence-inventory.csv`, label: 'evidence-inventory.csv' },
      { href: `${GITHUB}/blob/main/docs/final-results/README.md`, label: 'Full results index' },
      { href: `${GITHUB}/blob/main/docs/final-results/handoff.md`, label: 'Handoff and blockers' },
    ],
  },
  cta: {
    kicker: 'DEMO + SOURCE',
    title: 'Want to play a round before you trust the numbers?',
    body: 'The demo runs the whole game in the browser: Easy through Hard are the trained ONNX DQN, and “Unbeatable” is the screen-only controller. “Unbeatable” is a label on the difficulty menu; most audited browser episodes still ended before both walls came down.',
    playCta: 'Play the demo',
    githubCta: 'GitHub repository',
  },
  footer: {
    brand: 'BREAKOUT · RESEARCH RESULTS',
    updated: 'Frozen evidence closeout. Last updated',
    limitsHeading: 'Known limits',
    limits: [
      'Clears are counted as canonical clear detections: the detector reads the scoreboard in the game’s memory to decide whether an episode took down both walls. A verified clear would also check each episode’s raw record, which this batch did not do, so the page uses the narrower term throughout.',
      'Some numbers are marked “Historical reported”: the raw run output is not in main, and what survives is the PR, issue, or archived write-up. Each row carries its status and source next to the value.',
    ],
    playDemo: 'Play demo',
    evidence: 'Evidence',
    github: 'GitHub',
  },
};
