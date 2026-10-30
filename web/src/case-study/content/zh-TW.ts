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
    titleTail: '非學不可嗎？',
    lede: '一個完全不訓練的程式，只讀畫面就能算出球會飛到哪裡，再把球拍移過去。在官方回合裡，它已經不只一次把兩面牆都打穿——這個判定是程式做的，不是人用眼睛看的。',
    hook: '整個專案其實只在追一個問題：這款遊戲有多少部分，真的需要一個會學習的模型？比高分更值得看的，是規則一換，答案會跟著變多少。',
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
      verified: '對照過倉庫裡的檔案，或可以用同一道指令重跑一次。',
      historical_reported: '出自已合併的 PR、issue 或封存文章；當時的執行輸出不在 main 上。',
      missing_raw_artifacts: '有文字說明，但要重新查核所需的原始回合、日誌或權重已經拿不到。',
      not_yet_evaluated: '這項說法還沒有任何做完的評估。',
    },
  },
  metricCards: [
    {
      metricId: 'rgb_controller_a8445df_canonical_clear_detections',
      label: '打穿兩面牆的回合數',
      note: '{{seeds:rgb_controller_a8445df_canonical_clear_detections}} 局官方回合裡有 {{metric:rgb_controller_a8445df_canonical_clear_detections}} 局清空兩面磚牆，也就是拿到 {{metric:clear_score_threshold}} 分。回合在 native ALE、每格都決定的那套規則（v3）、程式版本 a8445df 下跑。判定方式是程式去讀遊戲記憶體裡的計分板，不是用眼睛看；讀不出來就記「未知」，不會記成 0。這個判定沒有再往上追每一局的原始紀錄，所以不宣稱超過程式判定的通關。',
    },
    {
      metricId: 'rgb_controller_a8445df_mean_score',
      label: '只看畫面的控制器，平均得分',
      note: '這條路完全不用訓練。舊版規則（PR #26，50ee352）平均 {{metric:rgb_controller_pr26_mean_score}} 分、最高 {{metric:rgb_controller_pr26_max_score}} 分，連第一面牆要的 432 分都不到。',
    },
    {
      metricId: 'dqn_v3_2m_mean_score',
      label: '會學習的模型，同一套規則下的平均',
      note: 'Dueling Double DQN 是會自己學的版本：只練一個種子、兩百萬格畫面，{{metric:dqn_v3_2m_clears}} 次清空兩面牆。瀏覽器上那個 DQN 的 {{metric:dqn_final_holdout_mean}} 分用的是舊規則（v2），不能直接拿來比。',
    },
    {
      metricId: 'laya_complete_episodes',
      label: 'Laya 打完的完整回合',
      note: '一個凍結不動的 {{metric:laya_parameter_count}} 參數視覺模型。這次只量到反應速度，遊戲並沒有打完。',
    },
  ],
  metricsAria: '主要成果',
  readingKey: {
    ariaLabel: '證據狀態怎麼讀',
    title: '這個數字是誰驗的？',
  },
  problem: {
    index: '01',
    kicker: '問題',
    title: '打磚塊到底難在哪裡？',
    paragraphs: [
      '球和球拍在畫面上是分開的兩團顏色，兩次碰撞之間也幾乎走直線，所以每次看到畫面都能重算一次。這讓打磚塊很適合問一個問題：這款遊戲有多少部分，真的需要一個會學習的模型？',
      '麻煩的是規則。遊戲每走幾格才決定一次動作、動作會不會被沿用、發球鍵誰按、掉命算不算結束、一局最多幾格，每一項都會改變題目。',
      '所以一個分數如果沒說清楚用哪套規則，就只是等著被補註腳的數字，不能算成果。',
    ],
    callout:
      '<strong>什麼算「打穿兩面牆」？</strong>在 ALE/Breakout-v5 單人 Game 1 把兩面磚牆清空，滿分是 {{metric:clear_score_threshold}} 分。這個分數是從遊戲記憶體裡的計分板讀出來的，再跟遊戲給的獎勵對過一次；回合結束不代表打穿。判斷程式遇到不認識的環境、或兩個來源對不上時，會直接標成「未知」，不會偷偷記成 0。它拿真實打穿兩面牆的回合驗證過，也故意弄壞六個地方，六個都被抓到。',
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
    title: '先做一個能比的基準，再一項一項拿掉，最後剩下什麼？',
    items: [
      {
        step: '01',
        title: '第一版規則下，讓模型自己學',
        body: 'DQN 的做法是只看畫面，自己學會每個動作大概值多少分。三種變體試下來，早期檢查點就選定 Dueling Double DQN，之後沒再換，總共練了 {{metric:dqn_training_transitions}} 次轉移。第一版規則每 {{contract:v2.frame_skip}} 格畫面才決定一次動作。',
        status: 'historical_reported',
      },
      {
        step: '02',
        title: '試過、但沒有採用的兩條路',
        body: '一條是掉命時多扣分，希望模型學會別漏球；方向對，但結果不穩定。另一條是優先回放，讓模型多練那些出乎意料的畫面；結果最多只能算描述性。兩條都沒有取代原本的做法。',
        status: 'historical_reported',
      },
      {
        step: '03',
        title: '第二版規則：不學分數，直接把球拍算到球下面',
        body: '這次完全不學：從畫面找出球和球拍、估出球速，把牆面反彈算進去，預測球會飛到球拍平面的哪裡，再用實際看到的球拍位置修正。舊版這條規則（PR #26，50ee352）平均 {{metric:rgb_controller_pr26_mean_score}} 分、最高 {{metric:rgb_controller_pr26_max_score}} 分，連第一面牆都打不完，所以當時記到的 {{metric:rgb_controller_pr26_canonical_clear_detections}} 次是量對的，不是漏判。#68 補上速度錨點、並讓球拍在接到球前先放開；在 a8445df、native ALE、{{seeds:rgb_controller_a8445df_mean_score}} 組官方種子下，平均 {{metric:rgb_controller_a8445df_mean_score}} 分，{{metric:rgb_controller_a8445df_canonical_clear_detections}} 次打穿兩面牆，每格約 {{metric:rgb_controller_a8445df_latency_ms_mean}} 毫秒。',
        status: 'verified',
      },
      {
        step: '04',
        title: '支線：拿一個凍結的視覺模型來試',
        body: '把一個以 SmolVLM 為底、{{metric:laya_parameter_count}} 參數的模型凍結起來，直接叫它選動作。它先前看過一些 Breakout 專家操作，所以不算「沒看過也能玩」。速度的部分是真的，遊戲的證據還不完整。',
        status: 'missing_raw_artifacts',
      },
    ],
  },
  results: {
    index: '03',
    kicker: '成果速覽',
    title: '這些數字，各自是在什麼條件下量到的？',
    intro:
      '表格由 <a href="/final-results/summary.json">summary.json</a> 直接產生，頁面沒有另外手抄一份。標成「歷史紀錄」表示當時的執行輸出不在 <code>main</code> 上，不代表這個數字在這裡重跑過。',
    headers: {
      metric: '指標',
      value: '數值',
      scope: '種子數／範圍',
      evidence: '證據狀態',
      source: '來源',
    },
    metricLabels: {
      rgb_controller_a8445df_canonical_clear_detections: 'RGB 預測控制器 canonical clear detection 次數（a8445df）',
      rgb_controller_a8445df_mean_score: 'RGB 預測控制器平均（a8445df）',
      rgb_controller_pr26_mean_score: 'RGB 控制器平均，PR #26 控制律（50ee352）',
      dqn_v3_2m_mean_score: 'DQN 評測契約 v3 平均（種子 11、2M 影格）',
      dqn_final_holdout_mean: 'DQN 最終保留集平均',
      dqn_selection_mean: 'DQN 模型選擇平均',
      browser_wasm_120_canonical_clear_detections: '瀏覽器控制器 canonical clear detection 次數（120 局 WASM 稽核）',
      rgb_controller_a8445df_latency_ms_mean: 'RGB 預測控制器決策延遲（a8445df）',
      reward_shaping_mean_effect: '獎勵塑形平均效果（與基線之差）',
      per_500k_mean_score: 'PER 檢查點平均分',
      laya_warm_latency_p50_ms: 'Laya 暖機決策延遲 p50',
      laya_complete_episodes: 'Laya 完整回合數',
    },
  },
  fairness: {
    index: '04',
    kicker: '評測契約與公平性',
    title: '兩套規則下的分數，可以放在一起比嗎？',
    intro:
      '兩套規則最大的差別是「多久決定一次動作」：舊的（v2）每 {{contract:v2.frame_skip}} 格畫面才決定一次，新的（v3）每一格都決定一次。一局最多幾格，兩邊一樣，都是 {{contract:v2.max_num_frames_per_episode}} 格，但能做的決定次數分別是 {{contract:v2.agent_step_limit}} 與 {{contract:v3.agent_step_limit}}。所以跨規則直接比原始分數，只能描述現象，不能排名。',
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
      'DQN 的舊規則分數和只看畫面的控制器不能直接比。舊規則每 {{contract:v2.frame_skip}} 格畫面才決定一次，新規則每一格都決定，種子的用途和回合集合也不一樣。所以原始分數只能描述現象，不能當公平排名，也不能當成誰贏了。',
    comparisonFallback: '兩套規則決定動作的頻率不同，跨規則比較不能直接當成可比。',
    notes: [
      '<strong>同一套規則下，只有一組能比。</strong>新規則下，DQN 只練一個種子、兩百萬格畫面，平均 {{metric:dqn_v3_2m_mean_score}} 分、{{metric:dqn_v3_2m_clears}} 次打穿；控制器平均 {{metric:rgb_controller_a8445df_mean_score}} 分、{{metric:rgb_controller_a8445df_canonical_clear_detections}} 次打穿。DQN 只有一個訓練種子，原始回合也不在 main，所以不宣稱誰公平勝出。',
      '<strong>不拿原始分數替方法排名。</strong>在一種決定頻率下分數高，不代表換一種頻率還是一樣好。',
      '<strong>不做事後改判。</strong>真正公平的比較，要在看到結果之前就講好要比什麼。',
    ],
  },
  laya: {
    index: '05',
    kicker: '支線實驗',
    title: 'Laya 這個視覺模型，真的能玩嗎？',
    paragraphs: [
      {
        lead: '為什麼試這個？',
        body: '想知道一個小型、凍結、不生成文字的視覺模型，能不能直接用畫面選出合法的動作，而且快到能實際使用。',
      },
      {
        lead: '假設是什麼？',
        body: '一個以 SmolVLM 為底、{{metric:laya_parameter_count}} 參數的模型，不對這個專案另外訓練，直接當成固定策略來用。它先前已經看過一些 Breakout 專家操作，所以這<em>不是</em>「沒看過也能玩」。',
      },
      {
        lead: '量到什麼？',
        body: '暖機後每次決定的延遲、一個有上限的控制診斷，還有一次真的在瀏覽器裡玩的紀錄。單一張固定畫面上，暖機後中位數是 {{metric:laya_warm_latency_p50_ms}} 毫秒；本機最佳化過的代理，多輪測試中位數是 {{metric:laya_optimized_proxy_p50_ms}} 毫秒。{{seeds:laya_control_diagnostic_reversals}} 組種子的控制診斷裡，立刻反向的指令從 {{metric:laya_control_diagnostic_raw_reversals}} 次降到 {{metric:laya_control_diagnostic_reversals}} 次。',
      },
      {
        lead: '瀏覽器裡跑起來如何？',
        body: '那次拿了 {{metric:laya_browser_session_score}} 分，玩了 {{metric:laya_browser_session_played_s}} 秒後生命耗盡，畫面每秒 {{metric:laya_browser_presented_fps}} 格。這是一次真實的工作階段，不是評估回合。',
      },
      {
        lead: '還缺什麼？',
        body: '沒有完整評估回合、沒有逐回合原始紀錄，也沒有打穿兩面牆，所以那次瀏覽器紀錄不能跟已評估的回合相比。延遲快，不代表玩得好。',
      },
    ],
    callout:
      '<strong>決定：</strong>Laya 保留當研究方向。在出現一個完整評估回合之前，PR #74 維持草稿，#70–#72 繼續開著。',
    aside: {
      ariaLabel: 'Laya 證據摘要',
      heading: '凍結當下的 Laya',
      facts: [
        { label: '打完的完整回合', value: '{{metric:laya_complete_episodes}}', unit: '個完整回合' },
        { label: '暖機後中位數', value: '{{metric:laya_warm_latency_p50_ms}}', unit: '毫秒／次決策' },
        { label: '最佳化代理中位數', value: '{{metric:laya_optimized_proxy_p50_ms}}', unit: '毫秒／次請求' },
        {
          label: '立刻反向次數（原始 → 煞車後）',
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
    title: '失敗的實驗，算不算成果？',
    cards: [
      {
        kicker: '獎勵塑形 · ISSUE #9',
        value: '{{metric:reward_shaping_mean_effect}}',
        unit: '平均效果',
        body: '95% 區間是 {{metric:reward_shaping_ci_low}} 到 {{metric:reward_shaping_ci_high}}；{{seeds:reward_shaping_mean_effect}} 組訓練種子裡有 {{metric:reward_shaping_positive_training_seeds}} 組是正的。沒過採用門檻，所以沒有再往下跑更長的比較。',
        status: 'historical_reported',
      },
      {
        kicker: '優先經驗回放 · ISSUE #12',
        value: '{{metric:per_500k_mean_score}}',
        unit: 'PER 檢查點平均',
        body: '同一個進度點上，均勻回放是 {{metric:uniform_500k_mean_score}} 分。優先回放多花了 {{metric:per_training_time_ratio}} 倍的訓練時間。只有 {{seeds:per_500k_mean_score}} 組種子，這只能描述現象，算不上統計上的優勢。',
        status: 'historical_reported',
      },
      {
        kicker: '首次通關探測 · #41、#43、#45、#48–#61',
        value: '0',
        unit: '次由程式判定打穿',
        body: '凍結的 DQN 一路探測，最後都沒有出現一次打穿兩面牆。其中幾次直接標成 INCONCLUSIVE 或 BLOCKED_ENVIRONMENT，也就是無法判定或環境卡住。這些結果指出失敗模式，但沒有證明某個方法永遠打不穿。',
        status: 'historical_reported',
      },
    ],
  },
  ledger: {
    index: '07',
    kicker: '公開證據',
    title: '除了影片，還有什麼可以查？',
    intro:
      '收官時納入的每一個實驗家族，都附上狀態和來源。標題、結論和備註是 <code>summary.json</code> 的原文，保留原語言，以免翻譯讓數字走樣；CSV 和 JSON 才是這一頁的機器可讀版本。',
    downloads: [
      { href: `${GITHUB}/blob/main/docs/final-results/evidence-inventory.csv`, label: 'evidence-inventory.csv' },
      { href: `${GITHUB}/blob/main/docs/final-results/README.md`, label: '完整成果索引' },
      { href: `${GITHUB}/blob/main/docs/final-results/handoff.md`, label: '交接與待解問題' },
    ],
  },
  cta: {
    kicker: '展示與原始碼',
    title: '想自己打一局，再回頭看證據嗎？',
    body: '展示頁把整個遊戲跑在瀏覽器裡：Easy 到 Hard 是訓練出來的 ONNX DQN，Unbeatable 是那個只看畫面的控制器。「Unbeatable」只是難度選單上的名字，它不會故意放水，也不代表控制器不會輸；查核過的瀏覽器回合裡，多數還是在打完兩面牆之前就結束。',
    playCta: '開始試玩',
    githubCta: 'GitHub 專案',
  },
  footer: {
    brand: 'BREAKOUT · 研究成果',
    updated: '凍結證據收官。最後更新',
    notClaimedLabel: '不宣稱：',
    notClaimed: '遊戲已解、超出 canonical clear detection 的通關說法、穩定通關、世界紀錄、在公平比較下勝出，或完整的 Laya 評估回合。',
    playDemo: '試玩',
    evidence: '證據',
    github: 'GitHub',
  },
};
