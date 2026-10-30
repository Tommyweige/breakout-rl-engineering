import type { CaseStudyCopy } from './types';

export const zhTwCopy: CaseStudyCopy = {
  header: {
    brandHomeLabel: 'Breakout RL 首頁',
    brandMark: 'BREAKOUT',
    navLabel: '主要導覽',
    playDemo: '試玩',
    github: 'GitHub',
    languageSwitch: {
      href: '/case-study.html',
      hreflang: 'en',
      lang: 'en',
      label: 'EN',
      ariaLabel: '切換至英文版',
    },
  },
  hero: {
    kicker: 'Atari 打磚塊 × AI 研究',
    title: '打磚塊，一定要讓 AI 自己學嗎？',
    answer: '不一定。',
    answerBody: '一支直接算球路的程式，完全沒有訓練，打得比訓練過的神經網路好得多。',
    scrollCta: '往下看結果',
    summaryError: '數字載入失敗，頁面上的數字暫時是空的。',
    summaryErrorLink: '到 GitHub 看完整成果',
    videoAria: '追蹤程式在網頁版實際玩打磚塊的錄影。',
    videoFallback: '你的瀏覽器無法播放這段錄影。',
    caption: '追蹤程式實際在玩。網頁版錄影，6 倍速。',
  },
  common: {
    notRun: '未執行',
    yes: '是',
    no: '否',
    unknown: '未知',
    missingMetric: '—',
  },
  sections: [
    {
      id: 'goal',
      kicker: '怎樣算贏',
      title: '把兩面磚牆全部打光',
      stat: { value: '{{metric:clear_score_threshold}}', unit: '分' },
      body: ['打磚塊有上下兩面牆。兩面都打光，遊戲就結束，這也是能拿到的最高分。'],
      tone: 'lime',
    },
    {
      id: 'players',
      kicker: '兩位選手',
      title: '一個用算的，一個用學的',
      cards: [
        {
          name: '追蹤程式',
          tagline: '直接算出球會飛到哪',
          body: '從畫面找出球和球拍，算出球速，連牆壁反彈一起算進去，預測球會落在哪裡，先把球拍移過去。沒有任何訓練。',
        },
        {
          name: '神經網路',
          tagline: '看過 200 萬格畫面後自己學會',
          body: '沒人告訴它規則。它一邊玩，一邊記住哪個動作比較會得分。',
        },
      ],
      note: '兩個都只看遊戲畫面，跟人看到的一樣。',
    },
    {
      id: 'clears',
      kicker: '結果一',
      title: '追蹤程式真的把兩面牆打光了',
      stat: {
        value: '{{metric:rgb_controller_a8445df_canonical_clear_detections}}',
        unit: '／{{seeds:rgb_controller_a8445df_canonical_clear_detections}} 局',
      },
      body: ['不是每一局都行。其他局都在打光之前，就把命用完了。'],
      note: '怎麼算「打光」：由程式讀遊戲記憶體裡的分數，到滿分才算，不靠人看畫面。這幾局還沒有逐筆翻原始紀錄複查。',
      tone: 'lime',
    },
    {
      id: 'scores',
      kicker: '結果二',
      title: '平均分，差了一百多倍',
      bars: [
        { name: '追蹤程式', metricId: 'rgb_controller_a8445df_mean_score' },
        { name: '神經網路', metricId: 'dqn_v3_2m_mean_score' },
      ],
      barMaxMetricId: 'clear_score_threshold',
      barMaxLabel: '滿分',
      body: ['兩邊用同一套遊戲規則，玩一樣多局。神經網路一局都沒打光。'],
      note: '神經網路只完整訓練過一次。這是那一次的成績，不代表神經網路學不會。',
    },
    {
      id: 'laya',
      kicker: '支線',
      title: '拿現成的大模型直接玩呢？',
      stats: [
        { value: '{{metric:laya_browser_session_score}}', unit: '分', label: '整局得分' },
        { value: '{{metric:laya_browser_session_played_s}}', unit: '秒', label: '五條命全部用完' },
      ],
      lead: 'Laya 是一個現成的看圖模型，沒有為這個專案另外訓練。讓它看著畫面直接選動作，在瀏覽器裡玩了一局。',
      tone: 'blue',
    },
    {
      id: 'failed',
      kicker: '沒成功的嘗試',
      title: '兩個想讓神經網路變強的方法，都沒用',
      cards: [
        {
          name: '掉命多扣分',
          body: '掉一條命就多扣它分數，想逼它更小心。',
          verdict: '沒有明顯改善',
          detail:
            '平均 +{{metric:reward_shaping_mean_effect}} 分，但誤差範圍從 {{metric:reward_shaping_ci_low}} 到 +{{metric:reward_shaping_ci_high}}，可能根本沒差',
        },
        {
          name: '優先回放',
          body: '訓練時優先重看它最常猜錯的畫面。',
          verdict: '沒有明顯改善',
          detail:
            '訓練時間 ×{{metric:per_training_time_ratio}}，平均分 {{metric:per_500k_mean_score}} 對 {{metric:uniform_500k_mean_score}}',
        },
      ],
      tone: 'coral',
    },
    {
      id: 'takeaway',
      kicker: '所以',
      title: '能直接算的，不一定要學',
      body: [
        '打磚塊的畫面很單純：球和球拍一眼就找得到，球在兩次反彈之間幾乎走直線。這種時候，直接算比讓 AI 自己摸索有效得多。',
        '但這題還沒結束。追蹤程式還不能每局都打光，神經網路也只認真訓練過一次。',
      ],
    },
  ],
  evidence: {
    kicker: '證據',
    title: '每個數字都查得到',
    body: '原始紀錄、評估程式和完整報告都放在 GitHub。',
    cta: '到 GitHub 看證據',
    playCta: '自己玩玩看',
  },
  footer: {
    updated: '最後更新',
  },
};
