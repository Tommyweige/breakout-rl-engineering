import type { CaseStudyCopy } from './types';

export const zhTwCopy: CaseStudyCopy = {
  header: {
    brandHomeLabel: 'Breakout RL 首頁',
    brandMark: 'BREAKOUT',
    brandSub: '研究成果',
    navLabel: '主要導覽',
    playDemo: '試玩',
    evidence: '數字',
    github: 'GitHub',
    languageSwitch: {
      href: '/case-study.html',
      hreflang: 'en',
      lang: 'en',
      label: 'English',
      ariaLabel: '切換至英文版',
    },
  },
  hero: {
    eyebrow: '凍結證據 ·',
    titleLead: '打磚塊，',
    titleTail: '非學不可嗎？',
    lede: '一個完全不訓練的程式，只讀畫面就能算出球會飛到哪裡，再把球拍移過去。它已經不只一次把兩面牆都打穿——判定是程式做的，不是人用眼睛看的。',
    hook: '整個專案其實只在追一個問題：這款遊戲有多少部分，真的需要一個會學習的模型？',
    playCta: '開始試玩',
    evidenceCta: '看數字',
    summaryError: '機器可讀的摘要載入失敗，畫面上的數字暫時是空的。完整的證據索引仍可在 GitHub 上取得。',
    summaryErrorLink: '開啟成果索引',
    videoAria: '實際錄製的 Breakout 遊戲畫面：左側是人類玩家，右側是 AI 策略。',
    videoFallback: '你的瀏覽器無法播放這段錄影。可以改看',
    posterLink: '封面截圖',
    videoFallbackAfter: '，畫面同樣來自真實遊玩。',
    figcaption: '這是從實際建置版本錄下的畫面。左邊是人類操作區，右邊是 AI 策略；沒有假畫面，也沒有合成對局。',
  },
  common: {
    notRun: '未執行',
    yes: '是',
    no: '否',
    unknown: '未知',
    missingMetric: '—',
  },
  metricCards: [
    {
      metricId: 'rgb_controller_a8445df_canonical_clear_detections',
      label: '打穿兩面牆的回合',
      suffix: '／{{seeds:rgb_controller_a8445df_canonical_clear_detections}} 局',
      note: '兩面牆都清空，也就是拿到 {{metric:clear_score_threshold}} 分。判定是程式讀遊戲記憶體裡的計分板，不是用眼睛看。',
    },
    {
      metricId: 'rgb_controller_a8445df_mean_score',
      label: '只看畫面的控制器，平均分',
      note: '完全不用訓練，只讀畫面。',
    },
    {
      metricId: 'dqn_v3_2m_mean_score',
      label: '會學習的模型，平均分',
      note: 'Dueling Double DQN，只練一個種子。',
    },
  ],
  metricsAria: '主要成果',
  problem: {
    index: '01',
    kicker: '問題',
    title: '打磚塊到底難在哪裡？',
    paragraphs: [
      '球和球拍是畫面上分開的兩團顏色，兩次碰撞之間球幾乎走直線，所以只看畫面就能重算一次。',
      '真正的變數是規則：多久決定一次動作、掉命算不算結束、一局能跑多久，每一項都會換掉題目。',
      '所以「打穿兩面牆」是程式讀計分板判定的，不是回合結束就算。',
    ],
  },
  routes: {
    index: '02',
    kicker: '兩條路線',
    title: '訓練，還是不訓練？',
    items: [
      {
        lead: '會學習的模型',
        body: '只看畫面，自己學會每個動作大概值多少分；三種變體試下來，最後停在 Dueling Double DQN，一共練了 {{metric:dqn_training_transitions}} 次轉移。它不需要人寫規則，代價是要大量的模擬。',
      },
      {
        lead: '不訓練的控制器',
        body: '直接從畫面找出球和球拍、估出球速，把牆面反彈算進去，預測球會飛到球拍平面的哪裡，再用實際看到的球拍位置修正。整條路線沒有一個參數是學來的。',
      },
    ],
  },
  results: {
    index: '03',
    kicker: '結果',
    title: '三條路線，各自拿到什麼？',
    intro: '兩套規則的差別是多久決定一次動作，所以下面的數字都用同一套規則量。',
    headers: {
      method: '方法',
      mean: '平均分',
      clears: '打穿兩面牆',
      trained: '需要訓練嗎',
    },
    rows: [
      {
        method: '只看畫面的控制器',
        mean: '{{metric:rgb_controller_a8445df_mean_score}}',
        clears:
          '{{metric:rgb_controller_a8445df_canonical_clear_detections}}／{{seeds:rgb_controller_a8445df_canonical_clear_detections}}',
        trained: '不用',
      },
      {
        method: '會學習的模型',
        mean: '{{metric:dqn_v3_2m_mean_score}}',
        clears: '{{metric:dqn_v3_2m_clears}}／{{seeds:dqn_v3_2m_clears}}',
        trained: '要',
      },
      {
        method: 'Laya 視覺模型',
        mean: '—',
        clears: '—',
        trained: '沒有為這個專案訓練',
      },
    ],
  },
  mistake: {
    index: '04',
    kicker: '一場誤會',
    title: '137 分那場誤會，是怎麼回事？',
    paragraphs: [
      '舊版的控制器平均只有 {{metric:rgb_controller_pr26_mean_score}} 分，連第一面牆要的 432 分都打不到。',
      '原因是它沒把球速算進去，球拍也不會在接到球前先放開。',
      '修好之後，同一條路線平均 {{metric:rgb_controller_a8445df_mean_score}} 分，{{seeds:rgb_controller_a8445df_mean_score}} 局裡有 {{metric:rgb_controller_a8445df_canonical_clear_detections}} 局打穿兩面牆。',
    ],
  },
  laya: {
    index: '05',
    kicker: '支線',
    title: '一個凍結的視覺模型，能直接玩嗎？',
    paragraphs: [
      'Laya 是一個以 SmolVLM 為底、{{metric:laya_parameter_count}} 參數的模型，凍結之後直接叫它選動作，沒有為這個專案訓練；它先前看過一些 Breakout 專家操作，所以不算從零開始。',
      '速度是真的：暖機後每次決策的中位數是 {{metric:laya_warm_latency_p50_ms}} 毫秒。',
      '遊戲只跑過一次瀏覽器工作階段，拿了 {{metric:laya_browser_session_score}} 分、{{metric:laya_browser_session_played_s}} 秒後生命耗盡，完整回合是 {{metric:laya_complete_episodes}} 個。',
    ],
  },
  negative: {
    index: '06',
    kicker: '失敗的實驗',
    title: '失敗的實驗，值不值錢？',
    paragraphs: [
      '掉命多扣分那條路，平均只多 {{metric:reward_shaping_mean_effect}} 分，95% 區間從 {{metric:reward_shaping_ci_low}} 到 {{metric:reward_shaping_ci_high}}，沒過採用門檻。',
      '優先回放多花了 {{metric:per_training_time_ratio}} 倍的訓練時間，換到 {{metric:per_500k_mean_score}} 對 {{metric:uniform_500k_mean_score}}，只夠當現象看。',
    ],
  },
  footer: {
    brand: 'BREAKOUT · 研究成果',
    updated: '凍結證據收官。最後更新',
    limitsHeading: '已知限制',
    limits: [
      '通關用的是 canonical clear detection：偵測器讀遊戲記憶體裡的計分板，判斷一局有沒有打穿兩面牆。verified clear 還要逐筆核對該局的原始紀錄，這批回合沒有做那一步，所以全頁只採前者的說法。',
    ],
    evidenceLine: '想看完整證據與原始紀錄，去 GitHub →',
  },
};
