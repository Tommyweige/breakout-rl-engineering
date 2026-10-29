import type { CaseStudyCopy } from './types';

const GITHUB = 'https://github.com/Tommyweige/breakout-rl-engineering';

export const zhTwCopy: CaseStudyCopy = {
  header: {
    brandHomeLabel: 'Breakout RL 首頁',
    brandMark: 'BREAKOUT',
    brandSub: '研究成果',
    navLabel: '主要導覽',
    playDemo: '試玩',
    evidence: '證據',
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
    titleTail: '真的非得靠訓練嗎？',
    lede: '從 DQN 家族，到可解釋的像素追蹤器與預測控制器。兩種決策範式、一個還沒關掉的問題，以及至今仍沒有的已驗證完整通關。',
    hook: '這是一個端到端強化學習工程專案的證據式收官。真正有意思的不是高分，而是：當控制契約一改，答案會變多少；以及在說一款遊戲「被解掉」之前，你得有多小心。',
    playCta: '開始試玩',
    evidenceCta: '查看證據',
    summaryLoading: '正在載入機器可讀的摘要…',
    summaryLoaded: '摘要已載入：{metrics} 項指標、{experiments} 個實驗家族。',
    summaryError: '機器可讀的摘要載入失敗。完整的證據索引仍可在 GitHub 上取得。',
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
    loading: '載入中…',
    source: '來源',
    missingMetric: '—',
  },
  status: {
    labels: {
      verified: '已驗證',
      historical_reported: '歷史紀錄',
      missing_raw_artifacts: '缺少原始紀錄',
      not_yet_evaluated: '尚未評估',
    },
    descriptions: {
      verified: '已對照倉庫裡的產物，或可用同一道指令重跑。',
      historical_reported: '來自已合併的 PR、issue 或封存文章；原始執行輸出不在 main。',
      missing_raw_artifacts: '有文字描述，但稽核所需的原始回合、日誌或權重已無法取得。',
      not_yet_evaluated: '這項說法還沒有任何完成的評估。',
    },
  },
  metricCards: [
    {
      metricId: 'verified_clears_recorded',
      label: '已驗證的完整通關',
      note: '以 {{metric:clear_score_threshold}} 分通關偵測器稽核；截至凍結日沒有任何通關紀錄。',
    },
    {
      metricId: 'dqn_final_holdout_mean',
      label: 'DQN 最終保留集平均',
      note: 'Dueling Double DQN，{{metric:dqn_training_transitions}} 次轉移，評測契約 v2。',
    },
    {
      metricId: 'rgb_controller_mean_score',
      label: 'RGB 控制器平均',
      note: '決定性像素控制器，評測契約 v3。',
    },
    {
      metricId: 'laya_complete_episodes',
      label: 'Laya 完整回合',
      note: '凍結的 {{metric:laya_parameter_count}} 參數視覺模型；量到的是延遲，遊戲並沒有打完。',
    },
  ],
  metricsAria: '主要成果',
  readingKey: {
    ariaLabel: '證據狀態怎麼讀',
    title: '每一條說法都要這樣讀',
  },
  problem: {
    index: '01',
    kicker: '問題',
    title: '打磚塊表面上是分數，本質上是控制問題。',
    paragraphs: [
      '球和球拍在畫面上可以清楚分開。兩次碰撞之間，運動接近分段線性，控制器每次看到畫面都能重新規劃。所以打磚塊很適合拿來問一個簡單問題：這個任務到底有多少部分真的需要學習式策略？',
      '麻煩的是環境契約。影格跳躍、黏性動作、FIRE 處理、失去生命怎麼算、回合上限，每一個都會改變控制問題。沒有契約的分數不是成果，只是一個等著被補註腳的數字。',
    ],
    callout:
      '<strong>通關條件寫得很清楚。</strong>在 ALE/Breakout-v5 單人 Game 1 打掉兩面磚牆。經稽核的上限是 {{metric:clear_score_threshold}} 分，讀自 ALE RAM 裡的 ROM 計分板，並與原始獎勵交叉比對。回合結束不代表通關。',
    diagramAria: '兩條決策流程對照',
    diagrams: [
      {
        label: '學習式策略',
        steps: ['{{contract:v2.frame_stack}} 張堆疊影格', 'CNN', 'Q 值', 'argmax'],
        note: '評測契約 v2：每 {{contract:v2.frame_skip}} 個模擬器影格做一次決策。',
      },
      {
        label: '顯式控制器',
        steps: ['RGB 影格', '球／球拍', '速度', '攔截點', '動作'],
        note: '評測契約 v3：每個模擬器影格做一次決策，不需訓練。',
        accent: true,
      },
    ],
  },
  method: {
    index: '02',
    kicker: '方法演進',
    title: '先把基線做出來，再拿掉沒賺回成本的零件。',
    items: [
      {
        step: '01',
        title: '評測契約 v2 下的 DQN 家族',
        body: 'DQN → Double DQN → Dueling Double DQN。家族篩選在早期檢查點就選定 Dueling Double DQN，之後一路沿用。最終策略總共訓練了 {{metric:dqn_training_transitions}} 次轉移。',
        status: 'historical_reported',
      },
      {
        step: '02',
        title: '沒有被採用的對照實驗',
        body: '失去生命的獎勵塑形給出正向但不穩定的點估計；優先經驗回放最多只能算描述性結果。兩者都沒有取代基線。這些是結果，不是註腳。',
        status: 'historical_reported',
      },
      {
        step: '03',
        title: '評測契約 v3 下的決定性像素控制器',
        body: '偵測球與球拍、估計速度、把牆面反射算進去預測球拍平面上的攔截點，再用實際觀測到的球拍位置閉環修正。控制器每幀約 {{metric:rgb_controller_latency_ms_per_frame}} 毫秒——但它沒有打下這一局。',
        status: 'historical_reported',
      },
      {
        step: '04',
        title: '凍結視覺模型：一項支線實驗',
        body: '一個以 SmolVLM 為基礎、{{metric:laya_parameter_count}} 參數的檢查點，先前已看過一些 Breakout 專家動作資料，這次被當成凍結策略來測試。延遲的部分是真的；遊戲證據則不完整。',
        status: 'missing_raw_artifacts',
      },
    ],
  },
  results: {
    index: '03',
    kicker: '成果速覽',
    title: '每個數字都帶著它的契約與證據狀態。',
    intro:
      '表格由 <a href="/final-results/summary.json">summary.json</a> 產生，頁面不另外手抄一份數值。「歷史紀錄」表示原始執行輸出不在 <code>main</code>，不代表這個數字在這裡被重現過。',
    headers: {
      metric: '指標',
      value: '數值',
      scope: '種子數／範圍',
      evidence: '證據狀態',
      source: '來源',
    },
    metricLabels: {
      verified_clears_recorded: '已驗證的完整通關次數',
      dqn_final_holdout_mean: 'DQN 最終保留集平均',
      dqn_selection_mean: 'DQN 模型選擇平均',
      rgb_controller_mean_score: 'RGB 預測控制器平均',
      rgb_controller_canonical_clear_detections: 'RGB 控制器正式通關偵測次數',
      rgb_controller_latency_ms_per_frame: 'RGB 控制器延遲',
      reward_shaping_mean_effect: '獎勵塑形平均效果（與基線之差）',
      per_500k_mean_score: 'PER 檢查點平均分',
      laya_warm_latency_p50_ms: 'Laya 暖機決策延遲 p50',
      laya_complete_episodes: 'Laya 完整回合數',
    },
  },
  fairness: {
    index: '04',
    kicker: '評測契約與公平性',
    title: '兩個分數，回答的是不同問題。',
    intro:
      '評測契約 v2 讓策略每 {{contract:v2.frame_skip}} 個模擬器影格做一次決策，v3 則是每個影格一次。兩者沿用相同的 {{contract:v2.max_num_frames_per_episode}} 影格 ALE 回合上限，但代理步數預算分別是 {{contract:v2.agent_step_limit}} 與 {{contract:v3.agent_step_limit}}。跨契約直接比原始分數只能描述現象，不是排名。',
    headers: {
      property: '項目',
      v2: '評測契約 v2',
      v3: '評測契約 v3',
    },
    rows: [
      '契約 ID',
      '檔案',
      'SHA-256',
      '決策頻率',
      '代理步數上限',
      'ALE 回合影格上限',
      '影格堆疊',
      '黏性動作',
      'FIRE 重置',
      '失去生命即結束',
      '評估種子',
      '評估 epsilon',
    ],
    valuePhrases: {
      decisionPerFrames: '每 {n} 個影格 1 次',
      decisionPerFrame: '每個影格 1 次',
      seedCount: '{n} 組種子',
      yes: '是',
      no: '否',
    },
    comparisonNote:
      'DQN（v2）與 RGB 預測控制器（v3）不能直接比較。評測契約 v2 每 {{contract:v2.frame_skip}} 個模擬器影格才做一次決策，v3 每個影格都做一次；評估種子的角色與回合集合也不同。原始分數只能描述現象，不能當作公平排名，也不能當作方法勝負。',
    comparisonFallback: '兩份評測契約的決策頻率不同，跨契約比較不能直接視為可比。',
    notes: [
      '<strong>不宣稱公平勝出。</strong>DQN 保留集與 RGB 控制器並不是在同一份契約、同一組種子、相同回合預算與同一道指令下跑出來的。',
      '<strong>不用原始分數替方法排名。</strong>在某一種決策頻率下分數較高，不代表換一種頻率後方法更好。',
      '<strong>不做事後升格。</strong>真正匹配的比較，必須在看到結果之前就先講好。',
    ],
  },
  laya: {
    index: '05',
    kicker: '支線實驗',
    title: 'Laya Vision：延遲量到了，遊戲也真的跑過，但還沒有完整評估。',
    paragraphs: [
      {
        lead: '動機。',
        body: '測試一個小型、凍結、非生成式的視覺決策模型，能不能用可接受的推論成本，直接從像素選出合法的 Breakout 動作。',
      },
      {
        lead: '凍結假設。',
        body: '一個以 SmolVLM 為基礎、{{metric:laya_parameter_count}} 參數的檢查點，可以在不做專案專屬訓練的情況下當成凍結策略。它先前已經學過一些 Breakout 專家動作資料，所以這<em>不是</em>零樣本宣稱。',
      },
      {
        lead: '量到了什麼。',
        body: '暖機後的決策延遲、一個有界線的控制診斷，以及一次實際的瀏覽器工作階段。暖機 p50 在單一固定 RGB fixture 上是 {{metric:laya_warm_latency_p50_ms}} 毫秒；最佳化後的本機代理 p50 在多輪測試中是 {{metric:laya_optimized_proxy_p50_ms}} 毫秒。在 {{seeds:laya_control_diagnostic_reversals}} 組種子的控制診斷裡，立即反轉指令從 {{metric:laya_control_diagnostic_raw_reversals}} 次降到 {{metric:laya_control_diagnostic_reversals}} 次。瀏覽器工作階段中，決策模型拿到 {{metric:laya_browser_session_score}} 分，並在遊玩 {{metric:laya_browser_session_played_s}} 秒後失去所有生命，畫面呈現 {{metric:laya_browser_presented_fps}} fps。',
      },
      {
        lead: '還沒量到什麼。',
        body: '沒有完整評估回合、逐回合原始產物，也沒有通關，所以這次瀏覽器工作階段不能和已評估的回合相比。延遲不代表遊戲品質。',
      },
    ],
    callout:
      '<strong>決定：</strong>Laya 保留為研究方向。在出現一個完整評估回合之前，PR #74 維持草稿；#70–#72 繼續開著。',
    aside: {
      ariaLabel: 'Laya 證據摘要',
      heading: '凍結當下的 Laya',
      facts: [
        { label: '完整回合', value: '{{metric:laya_complete_episodes}}', unit: '個完整回合' },
        { label: '暖機 p50', value: '{{metric:laya_warm_latency_p50_ms}}', unit: '毫秒／次決策' },
        { label: '最佳化代理 p50', value: '{{metric:laya_optimized_proxy_p50_ms}}', unit: '毫秒／次請求' },
        {
          label: '反轉次數，原始 → 煞車後',
          value:
            '{{metric:laya_control_diagnostic_raw_reversals}} → {{metric:laya_control_diagnostic_reversals}}',
        },
      ],
      note: '權重授權為 CC BY-NC-SA 4.0，並未放在倉庫裡。',
      researchLinkLabel: '研究 issue',
      prLinkLabel: '草稿 PR',
    },
  },
  negative: {
    index: '06',
    kicker: '失敗與負面結果',
    title: '沒有被採用的實驗，也是成果的一部分。',
    cards: [
      {
        kicker: '獎勵塑形 · ISSUE #9',
        value: '{{metric:reward_shaping_mean_effect}}',
        unit: '平均效果',
        body: '95% 區間為 {{metric:reward_shaping_ci_low}} 到 {{metric:reward_shaping_ci_high}}。{{seeds:reward_shaping_mean_effect}} 個訓練種子中有 {{metric:reward_shaping_positive_training_seeds}} 個為正。採用門檻沒有通過，因此沒有再跑更長的比較。',
        status: 'historical_reported',
      },
      {
        kicker: '優先經驗回放 · ISSUE #12',
        value: '{{metric:per_500k_mean_score}}',
        unit: 'PER 檢查點平均',
        body: '同一個里程碑上，均勻回放是 {{metric:uniform_500k_mean_score}}。PER 花了 {{metric:per_training_time_ratio}} 倍的訓練時間。只有 {{seeds:per_500k_mean_score}} 組種子，這是描述性結果，不是統計上的優勢。',
        status: 'historical_reported',
      },
      {
        kicker: '首次通關探測 · #41、#43、#45、#48–#61',
        value: '{{metric:verified_clears_recorded}}',
        unit: '次已驗證通關',
        body: '凍結的探測最後都沒有出現已驗證通關。其中幾次明確標記為 INCONCLUSIVE 或 BLOCKED_ENVIRONMENT。它們指出失敗模式，但沒有證明某個方法永遠無法通關。',
        status: 'historical_reported',
      },
    ],
  },
  ledger: {
    index: '07',
    kicker: '公開證據',
    title: '看帳本，不看精華片段。',
    intro:
      '以下是收官時納入的每一個實驗家族，附上狀態與來源。標題、結論與備註直接引自 <code>summary.json</code> 原文，保留原語言以免翻譯造成數字落差；CSV 與 JSON 才是這一頁的機器可讀版本。',
    downloads: [
      { href: `${GITHUB}/blob/main/docs/final-results/evidence-inventory.csv`, label: 'evidence-inventory.csv' },
      { href: `${GITHUB}/blob/main/docs/final-results/README.md`, label: '完整成果索引' },
      { href: `${GITHUB}/blob/main/docs/final-results/handoff.md`, label: '交接與待解問題' },
    ],
  },
  cta: {
    kicker: '展示與原始碼',
    title: '先玩現在的版本，再回頭看它背後的證據。',
    body: '展示頁把 ALE WASM 與 ONNX 策略全部跑在瀏覽器裡。「Unbeatable」只是介面上的難度名稱：它不會刻意注入失誤，但也不代表這個策略不會輸。',
    playCta: '開始試玩',
    githubCta: 'GitHub 專案',
  },
  footer: {
    brand: 'BREAKOUT · 研究成果',
    updated: '凍結證據收官。最後更新',
    notClaimedLabel: '不宣稱：',
    notClaimed: '遊戲已解、已驗證通關、穩定通關、世界紀錄、在公平比較下勝出，或完整的 Laya 評估回合。',
    playDemo: '試玩',
    evidence: '證據',
    github: 'GitHub',
  },
};
