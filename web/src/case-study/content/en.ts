import type { CaseStudyCopy } from './types';

export const enCopy: CaseStudyCopy = {
  header: {
    brandHomeLabel: 'Breakout RL home',
    brandMark: 'BREAKOUT',
    navLabel: 'Primary',
    playDemo: 'Play',
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
    kicker: 'Atari Breakout × AI research',
    title: 'Does an AI have to learn to play Breakout?',
    answer: 'Not really.',
    answerBody:
      'A program that simply calculates where the ball is going, with no training at all, plays far better than a trained neural network.',
    scrollCta: 'See the results',
    summaryError: 'The numbers could not be loaded, so they are blank for now.',
    summaryErrorLink: 'See the full results on GitHub',
    videoAria: 'Recording of the tracker playing Breakout in the web version.',
    videoFallback: 'Your browser cannot play this recording.',
    caption: 'The tracker, actually playing. Recorded in the web version at 6× speed.',
  },
  common: {
    notRun: 'Not run',
    yes: 'Yes',
    no: 'No',
    unknown: 'Unknown',
    missingMetric: '—',
  },
  sections: [
    {
      id: 'goal',
      kicker: 'How you win',
      title: 'Knock down both brick walls',
      stat: { value: '{{metric:clear_score_threshold}}', unit: 'points' },
      body: [
        'Breakout has two walls of bricks. Clear both and the game ends. That is the highest score there is.',
      ],
      tone: 'lime',
    },
    {
      id: 'players',
      kicker: 'Two players',
      title: 'One calculates. One learns.',
      cards: [
        {
          name: 'The tracker',
          tagline: 'Calculates where the ball will land',
          body: 'It finds the ball and the paddle on screen, works out the ball’s speed, includes the wall bounces, predicts where the ball will come down, and moves the paddle there first. No training at all.',
        },
        {
          name: 'The neural network',
          tagline: 'Taught itself from 2 million frames',
          body: 'Nobody told it the rules. It played, and remembered which moves tended to score.',
        },
      ],
      note: 'Both see only the game screen, the same as a person would.',
    },
    {
      id: 'clears',
      kicker: 'Result 1',
      title: 'The tracker really did clear both walls',
      stat: {
        value: '{{metric:rgb_controller_a8445df_canonical_clear_detections}}',
        unit: 'of {{seeds:rgb_controller_a8445df_canonical_clear_detections}} games',
      },
      body: ['Not every game. In the rest, it ran out of lives before the walls were gone.'],
      note: 'How a clear is counted: a program reads the score from the game’s memory and only counts a perfect score. Nobody judges it by eye. These games have not yet been rechecked against their raw records one by one.',
      tone: 'lime',
    },
    {
      id: 'scores',
      kicker: 'Result 2',
      title: 'Average scores, more than 100× apart',
      bars: [
        { name: 'The tracker', metricId: 'rgb_controller_a8445df_mean_score' },
        { name: 'The neural network', metricId: 'dqn_v3_2m_mean_score' },
      ],
      barMaxMetricId: 'clear_score_threshold',
      barMaxLabel: 'perfect score',
      body: [
        'Both played by the same game rules, for the same number of games. The neural network never cleared the walls once.',
      ],
      note: 'The neural network was fully trained only once. This is that one run, not proof that a neural network can’t learn the game.',
    },
    {
      id: 'laya',
      kicker: 'Side quest',
      title: 'What about an off-the-shelf AI model?',
      stats: [
        { value: '{{metric:laya_browser_session_score}}', unit: 'points', label: 'for the whole game' },
        { value: '{{metric:laya_browser_session_played_s}}', unit: 'sec', label: 'to lose all five lives' },
      ],
      lead: 'Laya is an existing image-reading model, not trained for this project. It looked at the screen, picked moves directly, and played one game in the browser.',
      tone: 'blue',
    },
    {
      id: 'failed',
      kicker: 'What didn’t work',
      title: 'Two tricks to make the neural network better. Neither helped.',
      cards: [
        {
          name: 'Extra penalty for losing a life',
          body: 'Take away extra points every time it drops the ball, to push it to be careful.',
          verdict: 'No clear gain',
          detail:
            'Average +{{metric:reward_shaping_mean_effect}}, but the error range runs from {{metric:reward_shaping_ci_low}} to +{{metric:reward_shaping_ci_high}}, so it may be no change at all',
        },
        {
          name: 'Prioritized replay',
          body: 'During training, re-study the moments it got most wrong first.',
          verdict: 'No clear gain',
          detail:
            '{{metric:per_training_time_ratio}}× the training time; average {{metric:per_500k_mean_score}} vs {{metric:uniform_500k_mean_score}}',
        },
      ],
      tone: 'coral',
    },
    {
      id: 'takeaway',
      kicker: 'So',
      title: 'If you can calculate it, you may not need to learn it',
      body: [
        'Breakout is visually simple: the ball and paddle are easy to spot, and the ball travels almost straight between bounces. Here, calculating beat letting an AI figure it out by a wide margin.',
        'The question isn’t closed. The tracker can’t clear every game yet, and the neural network was only seriously trained once.',
      ],
    },
  ],
  evidence: {
    kicker: 'Evidence',
    title: 'Every number can be checked',
    body: 'The raw records, evaluation code, and full report are on GitHub.',
    cta: 'See the evidence on GitHub',
    playCta: 'Play it yourself',
  },
  footer: {
    updated: 'Last updated',
  },
};
