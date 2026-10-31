import type { CaseStudyCopy } from './types';

const REPO = 'https://github.com/Tommyweige/breakout-rl-engineering';

export const enCopy: CaseStudyCopy = {
  header: {
    brandHomeLabel: 'Breakout RL home',
    brandMark: 'BREAKOUT RL',
    navLabel: 'Primary',
    tocLabel: 'Contents',
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
    kicker: '30-day engineering case study · Atari Breakout',
    title: '30 days of training a Breakout AI from scratch, and the untrained program that beat it',
    lede: 'This project went from the game environment to a neural network, training, debugging, and finally live play in the browser, leaving a record you can check at every step. The ending was a surprise: a tracker that simply calculates the ball’s path plays far better than a neural network trained on millions of frames.',
    stats: [
      {
        value: '{{metric:rgb_controller_a8445df_canonical_clear_detections}}<small> of {{seeds:rgb_controller_a8445df_canonical_clear_detections}}</small>',
        label: 'games where the tracker knocked down both brick walls',
        tone: 'lime',
      },
      {
        value: '{{metric:rgb_controller_a8445df_mean_score}}<small> vs </small>{{metric:dqn_v3_2m_mean_score}}',
        label: 'average score under the same rules: tracker vs neural network',
      },
      {
        value: '{{metric:browser_fixture_actions_match}}<small> of {{seeds:browser_fixture_actions_match}}</small>',
        label: 'fixed test frames where the network picks the same move after moving into the browser',
        tone: 'blue',
      },
    ],
    playCta: 'Play it in the browser',
    storyCta: 'Read the story',
    summaryError: 'Some numbers could not be loaded, so they are blank for now.',
    summaryErrorLink: 'See the full results on GitHub',
    videoAria: 'Recording of the tracker playing Breakout in the web version.',
    videoFallback: 'Your browser cannot play this recording.',
    caption: 'The tracker, actually playing. Recorded in the web version at 6× speed.',
  },
  timeline: {
    nav: '30-day timeline',
    kicker: 'Timeline',
    title: '30 days, plus the closeout',
    items: [
      {
        days: 'Day 1–4',
        title: 'Wire up the game',
        text: 'Hook up the Atari emulator, shrink the screen to 84×84 grayscale, stack 4 frames.',
      },
      {
        days: 'Day 5–12',
        title: 'Write DQN from scratch',
        text: 'Q-values, a convolutional network, experience replay, a target network: one training loop.',
      },
      {
        days: 'Day 13–15',
        title: 'The first evaluation finds a bug',
        text: 'It beat random play on average, but game by game it never served after losing a life.',
        tone: 'coral',
      },
      {
        days: 'Day 16–21',
        title: 'Speed up, compare, train long',
        text: '1.6× faster training, three DQN variants compared, training to 5 million steps.',
      },
      {
        days: 'Day 22–29',
        title: 'Move into the browser',
        text: 'Export to ONNX, run and check it in the browser, build a human-vs-AI page.',
        tone: 'blue',
      },
      {
        days: 'Day 30',
        title: 'First review',
        text: 'The neural network can play, but it is nowhere near clearing both walls.',
      },
      {
        days: 'After day 30',
        title: 'The tracker, and proving the clears',
        text: 'The tracker clears 4 of 15 games; the detector is calibrated; Laya is tried; evidence frozen.',
        tone: 'lime',
      },
    ],
  },
  common: {
    notRun: 'Not run',
    yes: 'Yes',
    no: 'No',
    unknown: 'Unknown',
    missingMetric: '—',
  },
  chapters: [
    {
      id: 'dqn',
      nav: 'Start: a neural network',
      kicker: 'Day 1–12 · The start',
      title: 'First, let a neural network teach itself Breakout',
      blocks: [
        {
          kind: 'p',
          text: 'DQN (Deep Q-Network) is the method DeepMind used to play Atari games. It does not know the rules. It looks at the screen, presses a button, sees whether the score went up, and slowly learns which button is worth pressing in which situation. That “worth” is called a Q-value.',
        },
        {
          kind: 'flow',
          caption: 'How the neural network turns a screen into a move',
          steps: [
            { label: 'Game screen', detail: '210×160 color' },
            { label: 'Shrink to grayscale', detail: '84×84' },
            { label: 'Stack the last 4', detail: 'shows the ball’s direction and speed' },
            { label: '3 convolution layers', detail: '32 → 64 → 64 filters' },
            { label: 'Split in two', detail: 'how good the position is + each move’s bonus' },
            { label: '4 Q-values', detail: 'stay / serve / right / left' },
            { label: 'Pick the biggest', detail: 'no random presses when evaluating' },
          ],
        },
        {
          kind: 'p',
          text: 'The learning itself is simple. After every step it stores what it saw, what it pressed, what it scored, and what the next screen looked like in a large memory, called experience replay. Training draws a random batch from that memory, uses the formula below to work out a “more correct” Q-value, and nudges the network toward it:',
        },
        {
          kind: 'code',
          caption: 'What every DQN update does',
          code: 'target Q = score from this step + 0.99 × the biggest Q-value on the next screen\nerror    = target Q − the Q the network guesses now   →  make the error smaller',
        },
        {
          kind: 'p',
          text: 'The “next screen” Q-value is also the network’s own guess, so the target would wobble along with it. A second copy, synced only now and then, computes the targets instead: the target network. Wiring and checking these parts took the first 12 days; the deployed model has {{metric:dqn_parameter_count}} parameters.',
        },
      ],
    },
    {
      id: 'deadlock',
      nav: 'Debugging: the stuck serve',
      kicker: 'Day 13–15 · Debugging',
      title: 'It won on average, but it was stuck on the waiting-to-serve screen',
      blocks: [
        {
          kind: 'p',
          text: 'At 100 thousand training steps I froze the model, removed all randomness, and played the same 15 games against an AI that presses buttons at random. At first glance it looked great:',
        },
        {
          kind: 'stats',
          items: [
            { value: '{{fact:day15_dqn_mean}}', label: 'neural network average', tone: 'lime' },
            { value: '{{fact:day15_random_mean}}', label: 'random AI average', tone: 'muted' },
          ],
        },
        {
          kind: 'p',
          text: 'But another number was strange. The random AI lasted {{fact:day15_random_steps}} steps a game on average; the neural network lasted {{fact:day15_dqn_steps}}. In {{fact:day15_timelimit_games}} of the 15 games it stopped near step {{fact:day15_step_limit}}, which is exactly the game’s time limit. It was not surviving well. The game simply was not being played.',
        },
        {
          kind: 'p',
          text: 'Opening one of those games: it lost a life at step {{fact:day15_life_lost_step}}, and its next serve came at step {{fact:day15_first_fire_step}}, {{fact:day15_gap_steps}} steps later. In Breakout you must press serve to bring the ball back after losing a life. The model did not, so the screen sat in the waiting state. Seeing nearly the same screen over and over, it kept choosing the same move, all the way to the time limit.',
        },
        {
          kind: 'p',
          text: 'Why did training not catch it? During training there is a 5% chance of a random press, which sometimes hits serve by accident and rescues the stuck game. The evaluation removed that randomness and exposed the flaw. To confirm the cause, I ran two controls:',
        },
        {
          kind: 'table',
          caption: 'Separating out the serve (15 games each)',
          headers: ['Evaluation', 'Who serves', 'Average', 'Hit the time limit', 'Steps per game'],
          numeric: [2, 3, 4],
          rows: [
            ['Original evaluation', 'The model', '{{fact:day15_dqn_mean}}', '{{fact:day15_timelimit_games}} / 15', '{{fact:day15_dqn_steps}}'],
            ['Environment serves', 'The environment', '{{fact:day15_assist_mean}}', '0 / 15', '{{fact:day15_assist_steps}}'],
            ['Keep 5% randomness', 'The model', '{{fact:day15_epsilon_mean}}', '0 / 15', '{{fact:day15_epsilon_steps}}'],
          ],
        },
        {
          kind: 'callout',
          tone: 'coral',
          title: 'The most important lesson of the project',
          text: 'From then on the environment serves at the start and after every lost life, and training and evaluation use the same rules, so every later comparison measures only the ability to play. The habit matters more: averages can lie, and when a number looks odd you open the games one by one.',
        },
      ],
    },
    {
      id: 'training',
      nav: 'Speed and model choice',
      kicker: 'Day 16–21 · Training',
      title: 'Does training longer always help? No',
      blocks: [
        {
          kind: 'p',
          text: 'Speed came first. Running 2 games at once raised the experience produced per second from {{fact:vec_n1_throughput}} to {{fact:vec_n2_throughput}}, about 1.6× faster. Going to 4 was no faster, so it stayed at 2.',
        },
        {
          kind: 'p',
          text: 'Next, three DQN variants. Double DQN fixes the original’s habit of overestimating Q-values. Dueling also splits the end of the network into “how good is this position” and “how much each move adds or takes away”. Each was trained three separate times, for 500 thousand steps each:',
        },
        {
          kind: 'bars',
          caption: 'Average score of the three DQN variants at 500 thousand steps (mean of three training runs)',
          max: 'fact:family_dueling_500k',
          items: [
            { label: 'DQN', value: 'fact:family_dqn_500k', tone: 'muted' },
            { label: 'Double DQN', value: 'fact:family_double_500k', tone: 'muted' },
            { label: 'Dueling Double DQN', value: 'fact:family_dueling_500k', tone: 'lime' },
          ],
        },
        {
          kind: 'p',
          text: 'Dueling Double DQN won and kept training. Here is how it went:',
        },
        {
          kind: 'columns',
          caption: 'Dueling Double DQN’s average score as training went on',
          max: 'metric:dqn_selection_mean',
          items: [
            { label: '500K steps', value: 'fact:family_dueling_500k' },
            { label: '1M steps', value: 'fact:dueling_1m' },
            { label: '2.5M steps', value: 'metric:dqn_selection_mean', tone: 'lime' },
            { label: '5M steps', value: 'fact:dueling_5m' },
            { label: '2.5M · final check', value: 'metric:dqn_final_holdout_mean', tone: 'coral' },
          ],
          note: '500K and 1M are the mean of three training runs; 2.5M and 5M are the one run that was finally chosen.',
        },
        {
          kind: 'p',
          text: 'Doubling the training to 5 million steps did not raise the score, and the difference sits inside the normal swing between evaluations. Since the extra training bought no progress, the 2.5 million step model is the one deployed. The last column is the real warning: checked again on games never used to pick the model, its average fell from {{metric:dqn_selection_mean}} to {{metric:dqn_final_holdout_mean}}. A score used to choose a model is inflated by the choosing itself, and cannot be reported as the final result.',
        },
      ],
    },
    {
      id: 'deploy',
      nav: 'Into the browser',
      kicker: 'Day 22–29 · Deployment',
      title: 'Move the model into the browser, and prove it is still the same AI',
      blocks: [
        {
          kind: 'p',
          text: 'The trained model was in PyTorch format, which only runs in Python. I exported it to ONNX, a general format for neural networks, and ran it directly in the browser. Anyone who opens the page can play against the same AI live, with no server.',
        },
        {
          kind: 'p',
          text: 'The risk in a move like this is “it runs, but it changed”. So I fixed {{seeds:browser_fixture_actions_match}} test frames, ran both the original and the moved model on each one, and compared them frame by frame:',
        },
        {
          kind: 'stats',
          items: [
            { value: '{{metric:browser_fixture_actions_match}} / {{seeds:browser_fixture_actions_match}}', label: 'frames with the same move', tone: 'lime' },
            { value: '{{fact:web_wasm_p50_ms}}', label: 'median time per decision in the browser (WASM)', tone: 'blue' },
          ],
        },
        {
          kind: 'p',
          text: 'Speed was measured too: the GPU build (WebGPU) takes {{fact:web_webgpu_p50_ms}} ms per decision instead. For a network this small, moving data to the GPU costs more than the math, so the page uses WASM by default.',
        },
        {
          kind: 'table',
          caption: 'The same model and the same starts give identical scores in three environments',
          headers: ['Start', 'Native Python', 'Browser WASM', 'Browser WebGPU'],
          numeric: [1, 2, 3],
          rows: [
            ['Start 1', '73', '73', '73'],
            ['Start 2', '45', '45', '45'],
            ['Start 3', '58', '58', '58'],
            ['Start 4', '47', '47', '47'],
          ],
        },
        {
          kind: 'p',
          text: 'Identical scores mean the whole path, from screen processing to picking a move, survived the trip intact. On the home page, Easy, Medium and Hard all run this network; Unbeatable switches to the tracker and never touches it.',
        },
      ],
    },
    {
      id: 'tracker',
      nav: 'Turn: the tracker',
      kicker: 'After day 30 · The turn',
      title: 'A tracker with no training leaves the neural network far behind',
      blocks: [
        {
          kind: 'p',
          text: 'At the end of the 30 days the neural network was nowhere near clearing both walls. But Breakout’s screen is simple: the ball and paddle are distinct colors, and the ball flies almost straight between bounces. Does this really need to be learned?',
        },
        {
          kind: 'p',
          text: 'So I wrote a tracker. It sees the same screen as the neural network but learns nothing. On every frame it runs the same steps:',
        },
        {
          kind: 'flow',
          caption: 'How the tracker turns a screen into a move',
          steps: [
            { label: 'Color screen', detail: 'the same one the network sees' },
            { label: 'Find ball and paddle', detail: 'by color and position' },
            { label: 'Estimate ball speed', detail: 'compare the last few frames' },
            { label: 'Predict the landing spot', detail: 'including wall bounces' },
            { label: 'Choose a direction', detail: 'only move if far enough off, to avoid jitter' },
            { label: 'Left / right / stay', detail: 'about {{metric:rgb_controller_a8445df_latency_ms_mean}} ms per frame' },
          ],
        },
        {
          kind: 'code',
          caption: 'The two sums that predict the landing spot',
          code: 'frames to paddle = (paddle height − ball height) ÷ falling speed\nlanding spot     = ball now + sideways speed × frames  →  fold back at walls',
        },
        {
          kind: 'p',
          text: 'A fair comparison needs the same rules. The project has two rule sets that differ in how often a decision is made: the neural network always decided once every 4 frames. The tracker has to stop the paddle exactly on the landing spot, which takes a small correction on every frame, so it decides on every frame. To keep it fair, I trained another neural network under the every-frame rules, for 2 million frames, and both played the same 15 games:',
        },
        {
          kind: 'bars',
          caption: 'Average score under the same rules over the same 15 games (scale is the perfect two-wall score)',
          max: 'metric:clear_score_threshold',
          maxLabel: 'perfect score',
          items: [
            { label: 'The tracker', value: 'metric:rgb_controller_a8445df_mean_score', tone: 'lime' },
            { label: 'The neural network', value: 'metric:dqn_v3_2m_mean_score', tone: 'coral' },
          ],
        },
        {
          kind: 'p',
          text: 'That network averaged {{fact:dqn_v3_400k}} at 400 thousand frames, {{fact:dqn_v3_1m}} at 1 million, and {{metric:dqn_v3_2m_mean_score}} at 2 million. Its best game was {{metric:dqn_v3_2m_max_score}} points, and it never cleared the walls. The tracker’s median game was {{metric:rgb_controller_a8445df_median_score}}.',
        },
        {
          kind: 'callout',
          tone: 'muted',
          title: 'My guess, and the next experiment',
          text: 'I have two guesses. One is that the rules got harder: deciding on every frame means four times as many decisions in the same stretch of game time, so a button press and the points it earns sit further apart, and the network struggles to credit the right move. The other is too little training: this network was trained once, for only 2 million frames. The next step pulls the two apart. First, put the same network back under the once-every-4-frames rules and play the same 15 games, to see whether the score comes back; that separates a rules problem from a model problem. Then train three independent networks under the every-frame rules, 5 million frames each, so no single run decides the answer.',
        },
      ],
    },
    {
      id: 'clears',
      nav: 'Proving the clears',
      kicker: 'Proving the clears',
      title: '4 of 15 games cleared, and every one judged by code',
      blocks: [
        {
          kind: 'p',
          text: 'Breakout has two brick walls. Clearing the first is {{fact:first_wall_score}} points; clearing both is {{metric:clear_score_threshold}}, and there is no next level. These are the tracker’s final scores over its 15 games:',
        },
        {
          kind: 'episodes',
          caption: 'The tracker’s final score in each of its 15 games',
          gameLabel: 'Game {n}',
          clearedLabel: 'both walls cleared',
          firstWallLabel: 'first wall',
          bothWallsLabel: 'both walls',
        },
        {
          kind: 'p',
          text: 'The 4 clears are games 6, 12, 14 and 15, each with 1 or 2 lives left at the moment of clearing. The fastest cleared on frame {{metric:rgb_controller_a8445df_best_clear_ale_frame}}, about 6 minutes 18 seconds of game time. Of the 11 games that did not clear, the best reached {{fact:best_unclear_score}}, just a few bricks short.',
        },
        {
          kind: 'p',
          text: 'A clear can’t be judged by looking: the screen goes briefly empty after the first wall too, and the ball keeps bouncing after a clear. So the verdict comes from a detector that reads the game’s memory. Only when the scoreboard in memory reaches {{metric:clear_score_threshold}} and agrees with the score the game reports does it count; if the two disagree the game is marked “unknown”, never 0.',
        },
        {
          kind: 'p',
          text: 'The detector itself had to be tested. I recorded three real trajectories: a full clear, the first wall only, and all but the last brick. It may say “clear” only for the first, on exactly the frame the second wall empties. I also broke it on purpose in {{fact:detector_mutants}} different ways, and the tests caught every one.',
        },
        {
          kind: 'links',
          items: [
            { label: 'The six broken detectors', detail: 'the mistake each one makes, and the test that catches it', href: `${REPO}/blob/main/docs/clear-semantics.md` },
          ],
        },
        {
          kind: 'p',
          text: 'The same tracking logic in the web version (ported to TypeScript) also played 120 games: {{metric:browser_wasm_120_canonical_clear_detections}} cleared both walls, with an average of {{metric:browser_wasm_120_mean_score}}.',
        },
        {
          kind: 'callout',
          tone: 'muted',
          title: 'What the clear rate means',
          text: '{{metric:rgb_controller_a8445df_canonical_clear_detections}} of the {{seeds:rgb_controller_a8445df_canonical_clear_detections}} official games and {{metric:browser_wasm_120_canonical_clear_detections}} of the {{seeds:browser_wasm_120_canonical_clear_detections}} web games were cleared, and the two rates fall in the same range: the tracker clears roughly one game in three or four. It proves both walls can be cleared without any training, but it does not clear every game yet.',
        },
      ],
    },
    {
      id: 'failures',
      nav: 'What didn’t work',
      kicker: 'Failures',
      title: 'Three attempts that didn’t work, each with its numbers',
      blocks: [
        {
          kind: 'p',
          text: 'Failures go into the record too; “it didn’t help” is a result.',
        },
        {
          kind: 'ledger',
          items: [
            {
              name: 'Extra penalty for losing a life',
              idea: 'Take away one extra point for every lost life, to push the network to be careful. Trained three separate times, to 1 million steps each.',
              result: 'No consistent gain',
              detail: 'Average +{{metric:reward_shaping_mean_effect}}, but the 95% interval runs from {{metric:reward_shaping_ci_low}} to +{{metric:reward_shaping_ci_high}}; only {{metric:reward_shaping_positive_training_seeds}} of the three runs improved, short of the bar for adoption.',
            },
            {
              name: 'Prioritized replay',
              idea: 'During training, revisit the experiences it guessed worst first, instead of drawing evenly from memory.',
              result: 'Too small to be worth it',
              detail: 'At 500 thousand steps it averaged {{metric:per_500k_mean_score}} against {{metric:uniform_500k_mean_score}}, for {{metric:per_training_time_ratio}}× the training time.',
            },
            {
              name: 'Laya, an off-the-shelf model',
              idea: 'Take the existing image model Laya ({{metric:laya_parameter_count}} parameters, built on SmolVLM), keep it frozen, and let it pick moves from the screen. It had already seen some expert Breakout play, so it did not start from nothing.',
              result: 'Couldn’t play',
              detail: 'It played the same opening twice in the browser, scored {{metric:laya_browser_session_score}} both times, and lost all five lives in about {{metric:laya_browser_session_played_s}} s. Complete official evaluation games: {{metric:laya_complete_episodes}}.',
            },
          ],
        },
        {
          kind: 'p',
          text: 'Speed was not the problem: the page delivered one decision every {{metric:laya_browser_decision_interval_p50_ms}} ms, almost exactly the pace 60 frames a second needs, and the screen showed {{metric:laya_browser_presented_fps}} frames a second. The problem was decision quality: it went left one moment and right the next, so the paddle jittered in place. In a separate short test I added a brake: when it wanted to reverse straight away, it waited one more decision first. Back-to-back reversals fell from {{metric:laya_control_diagnostic_raw_reversals}} to {{metric:laya_control_diagnostic_reversals}}, and the paddle still missed the ball. Less jitter, same score, so jitter was not the cause; what it lacked was a sense of where the ball would land.',
        },
      ],
    },
    {
      id: 'conclusion',
      nav: 'Conclusion',
      kicker: 'Conclusion',
      title: 'What needs learning, and what doesn’t',
      blocks: [
        {
          kind: 'bars',
          caption: 'Average score of the three approaches (scale is the perfect score)',
          max: 'metric:clear_score_threshold',
          maxLabel: 'perfect score',
          items: [
            { label: 'The tracker', value: 'metric:rgb_controller_a8445df_mean_score', tone: 'lime' },
            { label: 'The neural network', value: 'metric:dqn_v3_2m_mean_score', tone: 'coral' },
            { label: 'Laya', value: 'metric:laya_browser_session_score', tone: 'blue', note: 'different rules; only the same opening, played in the browser' },
          ],
        },
        {
          kind: 'table',
          caption: 'The three approaches side by side',
          headers: ['', 'The tracker', 'The neural network', 'Laya'],
          rows: [
            ['Needs training?', 'No', 'Yes, 2 million frames', 'Not trained for this project'],
            ['Average score', '{{metric:rgb_controller_a8445df_mean_score}}', '{{metric:dqn_v3_2m_mean_score}}', 'Same opening twice: {{metric:laya_browser_session_score}} both times'],
            ['Both walls cleared', '{{metric:rgb_controller_a8445df_canonical_clear_detections}} of {{seeds:rgb_controller_a8445df_canonical_clear_detections}}', '{{metric:dqn_v3_2m_clears}} of {{seeds:dqn_v3_2m_clears}}', 'Not fully evaluated'],
            ['Per decision', 'about {{metric:rgb_controller_a8445df_latency_ms_mean}} ms', 'about {{fact:web_wasm_p50_ms}} ms (browser)', 'about {{metric:laya_browser_decision_interval_p50_ms}} ms'],
            ['Can you see why it moves?', 'Yes, every step can be worked out by hand', 'Hardly; you only see four Q-values', 'Hardly'],
          ],
        },
        {
          kind: 'p',
          text: 'What needs learning is the part you can’t write down as rules. Breakout’s physics is simple: the ball flies straight, bounces off walls, and its angle is predictable, so a few lines of math describe it. Making a neural network discover that from millions of frames is expensive, and it may not get there.',
        },
        {
          kind: 'p',
          text: 'What doesn’t need learning, but does need engineering, is everything around it: fix the serve rule, read evaluations game by game, verify the clear detector, compare the deployed model frame by frame. That is where most of the time went, and it is the part most worth taking away.',
        },
        {
          kind: 'p',
          text: 'Still open: how far can several independently trained networks get under the same rules? Can the tracker clear every game? Both are written up in the handoff notes on GitHub, as the next step.',
        },
      ],
    },
    {
      id: 'evidence',
      nav: 'Evidence and reruns',
      kicker: 'Evidence and reruns',
      title: 'Every number has a source you can check',
      blocks: [
        {
          kind: 'p',
          text: 'The results on this page are not copied by hand: as it loads, the page reads the project’s results file, the same file published with the results report on GitHub, so the two always match. The numbers from the 30 days come from that day’s article. You can rerun the tracker’s 15 games yourself; the steps are under “Reproduction steps” below:',
        },
        {
          kind: 'links',
          items: [
            { label: 'Full results report', detail: 'where every number comes from, how it was checked, and its limits', href: `${REPO}/blob/main/docs/final-results/README.md` },
            { label: 'Clear detection and its validation', detail: 'how the detector works and how it was calibrated', href: `${REPO}/blob/main/docs/clear-semantics.md` },
            { label: 'The tracker’s 15 games', detail: 'per-game score, clear time, and lives left', href: `${REPO}/blob/main/audits/browser-clear/native-15/report.md` },
            { label: 'Reproduction steps', detail: 'from installing the environment to the results', href: `${REPO}/blob/main/docs/repro/README.md` },
            { label: 'The 30 daily articles', detail: 'what each day did and what went wrong (in Chinese)', href: `${REPO}/tree/docs-ironman-series` },
            { label: 'Play online', detail: 'play against the AI in your browser', href: '/' },
          ],
        },
      ],
    },
  ],
  footer: {
    updated: 'Last updated',
  },
};
