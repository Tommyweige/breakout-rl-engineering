import type { CaseStudyCopy } from './types';

export const enCopy: CaseStudyCopy = {
  header: {
    brandHomeLabel: 'Breakout RL home',
    brandMark: 'BREAKOUT',
    brandSub: 'RESEARCH RESULTS',
    navLabel: 'Primary',
    playDemo: 'Play demo',
    evidence: 'Numbers',
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
    lede: 'A hand-written controller with no training reads the screen, works out where the ball will arrive, and puts the paddle there. It has cleared both brick walls more than once, counted by code rather than by eye.',
    hook: 'The whole project turns on one question: how much of this game actually needs something that learns?',
    playCta: 'Play the demo',
    evidenceCta: 'See the numbers',
    summaryError:
      'The machine-readable summary could not be loaded, so the numbers on this page are empty for now. The full evidence index is still on GitHub.',
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
    missingMetric: '—',
  },
  metricCards: [
    {
      metricId: 'rgb_controller_a8445df_canonical_clear_detections',
      label: 'Episodes that cleared both walls',
      suffix: ' of {{seeds:rgb_controller_a8445df_canonical_clear_detections}}',
      note: 'Both walls emptied, for {{metric:clear_score_threshold}} points. The verdict is read from the scoreboard in the game’s memory, not from watching.',
    },
    {
      metricId: 'rgb_controller_a8445df_mean_score',
      label: 'Mean score, screen-only controller',
      note: 'No training at all; it only reads the screen.',
    },
    {
      metricId: 'dqn_v3_2m_mean_score',
      label: 'Mean score, model that learns',
      note: 'A Dueling Double DQN, one training seed.',
    },
  ],
  metricsAria: 'Headline results',
  problem: {
    index: '01',
    kicker: 'THE PROBLEM',
    title: 'So what is actually hard about Breakout?',
    paragraphs: [
      'The ball and the paddle are two separate shapes on screen, and between bounces the ball travels close to a straight line, so a controller can work it out again from every frame.',
      'The variable is the rulebook: how often a decision is allowed, whether losing a life ends the episode, how long an episode may run. Change one and the question changes.',
      'That is why “clearing both walls” is a verdict the code reads from the scoreboard, not something an episode ending implies.',
    ],
  },
  routes: {
    index: '02',
    kicker: 'TWO ROUTES',
    title: 'Train it, or work it out by hand?',
    items: [
      {
        lead: 'The model that learns',
        body: 'It looks only at frames and learns roughly what each action is worth. Three variants were tried; the winner was Dueling Double DQN, trained for {{metric:dqn_training_transitions}} transitions. It needs no hand-written rules, and the price is a lot of simulated play.',
      },
      {
        lead: 'The controller that does not learn',
        body: 'It finds the ball and the paddle, estimates the ball’s speed, folds in the wall bounces to predict where the ball meets the paddle plane, and corrects with the paddle it actually sees. Not one parameter in it was learned.',
      },
    ],
  },
  results: {
    index: '03',
    kicker: 'RESULTS',
    title: 'What did each route actually get?',
    intro: 'The two rulebooks differ in how often a decision is allowed, so every number below comes from the same rulebook.',
    headers: {
      method: 'Method',
      mean: 'Mean score',
      clears: 'Walls cleared',
      trained: 'Training needed',
    },
    rows: [
      {
        method: 'Screen-only controller',
        mean: '{{metric:rgb_controller_a8445df_mean_score}}',
        clears:
          '{{metric:rgb_controller_a8445df_canonical_clear_detections}} of {{seeds:rgb_controller_a8445df_canonical_clear_detections}}',
        trained: 'No',
      },
      {
        method: 'Model that learns',
        mean: '{{metric:dqn_v3_2m_mean_score}}',
        clears: '{{metric:dqn_v3_2m_clears}} of {{seeds:dqn_v3_2m_clears}}',
        trained: 'Yes',
      },
      {
        method: 'Laya vision model',
        mean: '—',
        clears: '—',
        trained: 'Not for this project',
      },
    ],
  },
  mistake: {
    index: '04',
    kicker: 'A MIX-UP',
    title: 'What was the 137-point mix-up?',
    paragraphs: [
      'The older controller averaged {{metric:rgb_controller_pr26_mean_score}}, short of the 432 points the first wall takes.',
      'It left the ball’s speed out of the calculation, and the paddle did not let go before the ball arrived.',
      'With both fixed, the same route averaged {{metric:rgb_controller_a8445df_mean_score}} and cleared both walls in {{metric:rgb_controller_a8445df_canonical_clear_detections}} of {{seeds:rgb_controller_a8445df_mean_score}} official episodes.',
    ],
  },
  laya: {
    index: '05',
    kicker: 'SIDE EXPERIMENT',
    title: 'Can a frozen vision model just play?',
    paragraphs: [
      'Laya is a {{metric:laya_parameter_count}}-parameter SmolVLM-based model, frozen and asked to pick actions, with no training for this project; it had already seen some Breakout expert play, so it does not start from nothing.',
      'The speed is real: the warm median is {{metric:laya_warm_latency_p50_ms}} ms per decision.',
      'It played one browser session, scored {{metric:laya_browser_session_score}}, and ran out of lives after {{metric:laya_browser_session_played_s}} s, with {{metric:laya_complete_episodes}} complete episodes.',
    ],
  },
  negative: {
    index: '06',
    kicker: 'FAILED EXPERIMENTS',
    title: 'Are failed experiments worth anything?',
    paragraphs: [
      'Charging extra for a lost life moved the mean by {{metric:reward_shaping_mean_effect}} points, with a 95% interval from {{metric:reward_shaping_ci_low}} to {{metric:reward_shaping_ci_high}}, and missed the bar for adoption.',
      'Prioritized replay cost {{metric:per_training_time_ratio}}× the training time for {{metric:per_500k_mean_score}} against {{metric:uniform_500k_mean_score}}, which is a description, not an edge.',
    ],
  },
  footer: {
    brand: 'BREAKOUT · RESEARCH RESULTS',
    updated: 'Frozen evidence closeout. Last updated',
    limitsHeading: 'Known limits',
    limits: [
      'Clears are counted as canonical clear detections: the detector reads the scoreboard in the game’s memory to decide whether an episode took down both walls. A verified clear would also check each episode’s raw record, which this batch did not do, so the page uses the narrower term throughout.',
    ],
    evidenceLine: 'The full evidence and raw records are on GitHub →',
  },
};
