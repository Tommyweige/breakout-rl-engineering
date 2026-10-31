import type { CaseStudyCopy } from './types';

const REPO = 'https://github.com/Tommyweige/breakout-rl-engineering';

export const zhTwCopy: CaseStudyCopy = {
  header: {
    brandHomeLabel: 'Breakout RL 首頁',
    brandMark: 'BREAKOUT RL',
    navLabel: '主要導覽',
    tocLabel: '目錄',
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
    kicker: '30 天工程案例研究 · Atari 打磚塊',
    title: '從零訓練打磚塊 AI 的 30 天，以及一支不用訓練就贏過它的程式',
    lede: '從遊戲環境、神經網路、訓練、除錯，一路做到瀏覽器裡即時對戰，每一步都留下查得到的紀錄。結果出乎意料：一支直接算球路的追蹤程式，打得比練了上百萬格畫面的神經網路好得多。',
    stats: [
      {
        value: '{{metric:rgb_controller_a8445df_canonical_clear_detections}}<small>／{{seeds:rgb_controller_a8445df_canonical_clear_detections}}</small>',
        label: '追蹤程式打光兩面牆的局數',
        tone: 'lime',
      },
      {
        value: '{{metric:rgb_controller_a8445df_mean_score}}<small> 對 </small>{{metric:dqn_v3_2m_mean_score}}',
        label: '同一套規則下的平均分：追蹤程式 vs 神經網路',
      },
      {
        value: '{{metric:browser_fixture_actions_match}}<small>／{{seeds:browser_fixture_actions_match}}</small>',
        label: '搬進瀏覽器後，測試畫面上動作一致',
        tone: 'blue',
      },
    ],
    playCta: '在瀏覽器裡試玩',
    storyCta: '從頭看這個故事',
    summaryError: '數字載入失敗，部分數字暫時是空的。',
    summaryErrorLink: '到 GitHub 看完整成果',
    videoAria: '追蹤程式在網頁版實際玩打磚塊的錄影。',
    videoFallback: '你的瀏覽器無法播放這段錄影。',
    caption: '追蹤程式實際在玩。網頁版錄影，6 倍速。',
  },
  timeline: {
    nav: '30 天路線圖',
    kicker: '路線圖',
    title: '30 天，加上收尾',
    items: [
      {
        days: 'Day 1–4',
        title: '把遊戲接上程式',
        text: '接上 Atari 模擬器，把畫面縮成 84×84 灰階、疊 4 張。',
      },
      {
        days: 'Day 5–12',
        title: '從零寫出 DQN',
        text: 'Q 值、卷積網路、經驗回放、目標網路，拼成訓練迴圈。',
      },
      {
        days: 'Day 13–15',
        title: '第一次評估就抓到 bug',
        text: '平均分贏過隨機，逐局看才發現 AI 掉命後不發球。',
        tone: 'coral',
      },
      {
        days: 'Day 16–21',
        title: '加速、比較、長訓練',
        text: '訓練加速 1.6 倍、比較三種 DQN、練到 500 萬步。',
      },
      {
        days: 'Day 22–29',
        title: '搬進瀏覽器',
        text: '匯出 ONNX，在瀏覽器執行並逐張比對，做成人機對戰網頁。',
        tone: 'blue',
      },
      {
        days: 'Day 30',
        title: '第一次總結',
        text: '神經網路能玩，但離打光兩面牆還很遠。',
      },
      {
        days: '30 天之後',
        title: '追蹤程式與破關驗證',
        text: '追蹤程式 15 局打光 4 局；校準判定器；試 Laya；凍結證據。',
        tone: 'lime',
      },
    ],
  },
  common: {
    notRun: '未執行',
    yes: '是',
    no: '否',
    unknown: '未知',
    missingMetric: '—',
  },
  chapters: [
    {
      id: 'dqn',
      nav: '起點：訓練神經網路',
      kicker: 'Day 1–12 · 起點',
      title: '先讓一個神經網路自己學會打磚塊',
      blocks: [
        {
          kind: 'p',
          text: 'DQN（Deep Q-Network）是 DeepMind 用來玩 Atari 遊戲的方法。它不知道遊戲規則，只會看畫面、按按鈕、看分數有沒有增加，然後慢慢學會「在這個畫面下，哪個按鈕比較值得按」。這個「值得」的程度就叫 Q 值。',
        },
        {
          kind: 'flow',
          caption: '神經網路從畫面到動作的流程',
          steps: [
            { label: '遊戲畫面', detail: '210×160 彩色' },
            { label: '縮小並轉灰階', detail: '84×84' },
            { label: '疊最近 4 張', detail: '看得出球的方向與速度' },
            { label: '3 層卷積', detail: '32 → 64 → 64 個濾鏡' },
            { label: '拆成兩支', detail: '局面好不好＋每個動作加減幾分' },
            { label: '4 個 Q 值', detail: '不動／發球／右／左' },
            { label: '選最大的', detail: '評估時不再隨機亂按' },
          ],
        },
        {
          kind: 'p',
          text: '每玩一步，就把「看到什麼、按了什麼、得幾分、下一個畫面」存進記憶庫，叫經驗回放。訓練時隨機抽一批，用下面的式子算出「比較正確的 Q 值」，把網路的猜測往那邊修：',
        },
        {
          kind: 'code',
          caption: 'DQN 每一次更新在做的事',
          code: '目標 Q = 這一步拿到的分數 + 0.99 × 下一個畫面裡最大的 Q 值\n誤差   = 目標 Q − 網路現在猜的 Q   →  讓誤差變小',
        },
        {
          kind: 'p',
          text: '式子裡的「下一個畫面的 Q 值」也是網路自己猜的，目標會跟著自己晃。所以另外保留一份隔一段時間才同步的副本專門算目標，叫目標網路。把這些零件接對、驗證，花了前 12 天；最後部署的模型有 {{metric:dqn_parameter_count}} 個參數。',
        },
      ],
    },
    {
      id: 'deadlock',
      nav: '除錯：卡住的發球',
      kicker: 'Day 13–15 · 除錯',
      title: '平均分贏了，但它其實卡在等待發球的畫面',
      blocks: [
        {
          kind: 'p',
          text: '練到 10 萬步時，我把模型凍結、拿掉所有隨機性，和一個亂按按鈕的隨機 AI 各玩同樣的 15 局。第一眼看起來很漂亮：',
        },
        {
          kind: 'stats',
          items: [
            { value: '{{fact:day15_dqn_mean}}', label: '神經網路平均分', tone: 'lime' },
            { value: '{{fact:day15_random_mean}}', label: '隨機 AI 平均分', tone: 'muted' },
          ],
        },
        {
          kind: 'p',
          text: '但另一個數字很奇怪。隨機 AI 每局平均只撐 {{fact:day15_random_steps}} 步，神經網路卻平均走了 {{fact:day15_dqn_steps}} 步。15 局裡有 {{fact:day15_timelimit_games}} 局都停在 {{fact:day15_step_limit}} 步附近，那正好是遊戲的時間上限。它不是很會活，而是遊戲根本沒在進行。',
        },
        {
          kind: 'p',
          text: '把其中一局攤開來看：第 {{fact:day15_life_lost_step}} 步掉了一條命，下一次按發球卻是第 {{fact:day15_first_fire_step}} 步，中間隔了 {{fact:day15_gap_steps}} 步。打磚塊掉命後要按發球鍵才會重新開始；模型沒按，畫面就停在等待狀態。它一直看到幾乎一樣的畫面，就一直選同一個動作，於是一路卡到時間用完。',
        },
        {
          kind: 'p',
          text: '訓練時沒發現，是因為訓練中有 5% 機率隨機按鍵，偶爾亂中發球，把卡住的局救回來。評估拿掉隨機性，缺陷才露出來。我用兩組對照確認原因：',
        },
        {
          kind: 'table',
          caption: '把發球的影響拆開來看（各 15 局）',
          headers: ['評估方式', '誰負責發球', '平均分', '卡到時間上限', '平均每局步數'],
          numeric: [2, 3, 4],
          rows: [
            ['原本的評估', '模型自己', '{{fact:day15_dqn_mean}}', '{{fact:day15_timelimit_games}}／15', '{{fact:day15_dqn_steps}}'],
            ['環境自動發球', '遊戲環境', '{{fact:day15_assist_mean}}', '0／15', '{{fact:day15_assist_steps}}'],
            ['保留 5% 隨機', '模型自己', '{{fact:day15_epsilon_mean}}', '0／15', '{{fact:day15_epsilon_steps}}'],
          ],
        },
        {
          kind: 'callout',
          tone: 'coral',
          title: '這是整個專案最重要的一課',
          text: '從這之後，開局和掉命後的發球都交給環境，訓練和評估用同一套規則，後面所有比較都只量「打磚塊的能力」。更重要的是習慣：平均分會騙人，數字怪的時候要逐局攤開看。',
        },
      ],
    },
    {
      id: 'training',
      nav: '加速與選模型',
      kicker: 'Day 16–21 · 訓練',
      title: '練得越久越好嗎？不一定',
      blocks: [
        {
          kind: 'p',
          text: '先處理速度：同時跑 2 個遊戲，每秒經驗從 {{fact:vec_n1_throughput}} 筆提高到 {{fact:vec_n2_throughput}} 筆，約快 1.6 倍；加到 4 個反而沒更快。',
        },
        {
          kind: 'p',
          text: '接著比較三種 DQN。Double DQN 修正原版容易高估 Q 值的毛病；Dueling 再把網路最後一段拆成「這個局面本身好不好」和「每個動作加減幾分」。每種都獨立訓練三次、各練 50 萬步：',
        },
        {
          kind: 'bars',
          caption: '50 萬步時三種 DQN 的平均分（三次訓練的平均）',
          max: 'fact:family_dueling_500k',
          items: [
            { label: 'DQN', value: 'fact:family_dqn_500k', tone: 'muted' },
            { label: 'Double DQN', value: 'fact:family_double_500k', tone: 'muted' },
            { label: 'Dueling Double DQN', value: 'fact:family_dueling_500k', tone: 'lime' },
          ],
        },
        {
          kind: 'p',
          text: 'Dueling Double DQN 勝出，繼續練下去。下面是它一路的成績：',
        },
        {
          kind: 'columns',
          caption: 'Dueling Double DQN 的平均分隨訓練步數變化',
          max: 'metric:dqn_selection_mean',
          items: [
            { label: '50 萬步', value: 'fact:family_dueling_500k' },
            { label: '100 萬步', value: 'fact:dueling_1m' },
            { label: '250 萬步', value: 'metric:dqn_selection_mean', tone: 'lime' },
            { label: '500 萬步', value: 'fact:dueling_5m' },
            { label: '250 萬步・最後驗收', value: 'metric:dqn_final_holdout_mean', tone: 'coral' },
          ],
          note: '50 萬、100 萬步是三次訓練的平均；250 萬、500 萬步是最後選定的那一次訓練。',
        },
        {
          kind: 'p',
          text: '多練一倍到 500 萬步，分數沒有更高，差距也在每次評估的正常起伏之內；既然多花的訓練換不到進步，最後部署的是 250 萬步的模型。但最有警示意義的是最後一欄：改用一組從沒拿來挑模型的局重新驗收，平均分從 {{metric:dqn_selection_mean}} 掉到 {{metric:dqn_final_holdout_mean}}。挑模型時的分數，會因為「挑」這個動作本身而偏高，不能直接當成最終成績。',
        },
      ],
    },
    {
      id: 'deploy',
      nav: '部署到瀏覽器',
      kicker: 'Day 22–29 · 部署',
      title: '把模型搬進瀏覽器，而且證明它還是同一個 AI',
      blocks: [
        {
          kind: 'p',
          text: '訓練好的模型是 PyTorch 格式，只能在 Python 裡跑。我把它匯出成 ONNX，一種通用的神經網路格式，再直接在瀏覽器裡執行。這樣任何人打開網頁，就能和同一個 AI 即時對戰，不需要伺服器。',
        },
        {
          kind: 'p',
          text: '搬家最怕「看起來能跑，其實變了」。所以我固定了 {{seeds:browser_fixture_actions_match}} 張測試畫面，讓原本的模型和搬過去的模型各算一次，逐張比對：',
        },
        {
          kind: 'stats',
          items: [
            { value: '{{metric:browser_fixture_actions_match}}／{{seeds:browser_fixture_actions_match}}', label: '選到相同動作的畫面', tone: 'lime' },
            { value: '{{fact:web_wasm_p50_ms}}', label: '瀏覽器裡每次決策的中位數（WASM）', tone: 'blue' },
          ],
        },
        {
          kind: 'p',
          text: '速度也量了：GPU 版（WebGPU）每次決策反而要 {{fact:web_webgpu_p50_ms}} 毫秒。這麼小的網路，把資料搬上 GPU 的成本比計算本身還貴，所以網頁預設用 WASM。',
        },
        {
          kind: 'table',
          caption: '同一個模型、同樣的開局，在三個環境裡打出一模一樣的分數',
          headers: ['開局', 'Python 原生', '瀏覽器 WASM', '瀏覽器 WebGPU'],
          numeric: [1, 2, 3],
          rows: [
            ['開局 1', '73', '73', '73'],
            ['開局 2', '45', '45', '45'],
            ['開局 3', '58', '58', '58'],
            ['開局 4', '47', '47', '47'],
          ],
        },
        {
          kind: 'p',
          text: '分數完全一樣，代表從畫面處理到選動作，整條路在瀏覽器裡都沒走樣。首頁對戰裡的 Easy、Medium、Hard 三檔跑的就是這個神經網路；切到 Unbeatable 會換成追蹤程式，完全不經過網路。',
        },
      ],
    },
    {
      id: 'tracker',
      nav: '轉折：追蹤程式',
      kicker: '30 天之後 · 轉折',
      title: '一支不用訓練的追蹤程式，把神經網路遠遠甩開',
      blocks: [
        {
          kind: 'p',
          text: '30 天結束時，神經網路離打光兩面牆還很遠。但打磚塊的畫面很單純：球和球拍顏色分明，球在兩次反彈之間幾乎走直線。真的需要「學」嗎？',
        },
        {
          kind: 'p',
          text: '於是我寫了一支追蹤程式。它跟神經網路看同樣的遊戲畫面，但完全不學，每一格畫面都照同一套步驟計算：',
        },
        {
          kind: 'flow',
          caption: '追蹤程式從畫面到動作的流程',
          steps: [
            { label: '彩色畫面', detail: '和神經網路看的一樣' },
            { label: '找出球與球拍', detail: '靠顏色與位置' },
            { label: '估出球速', detail: '比較前後幾格畫面' },
            { label: '預測落點', detail: '把撞牆反彈算進去' },
            { label: '決定方向', detail: '差距夠大才動，避免抖動' },
            { label: '左／右／不動', detail: '每格約 {{metric:rgb_controller_a8445df_latency_ms_mean}} 毫秒' },
          ],
        },
        {
          kind: 'code',
          caption: '預測落點的兩個算式',
          code: '還要幾格才落到球拍 = (球拍的高度 − 球的高度) ÷ 球往下的速度\n落點               = 球現在的位置 + 橫向速度 × 格數   →  撞到左右牆就折回來',
        },
        {
          kind: 'p',
          text: '公平比較要用同一套規則。專案有兩套規則，差別在多久做一次決定：神經網路一路是「每 4 格畫面決定一次」；追蹤程式要把球拍精準停在落點上，需要每一格都能微調位置，所以用的是「每格都決定」。為了公平，我在「每格都決定」的規則下另外訓練了一個神經網路，練了 200 萬格畫面，兩邊各玩同樣的 15 局：',
        },
        {
          kind: 'bars',
          caption: '同一套規則、同樣 15 局的平均分（刻度是兩面牆全清的滿分）',
          max: 'metric:clear_score_threshold',
          maxLabel: '滿分',
          items: [
            { label: '追蹤程式', value: 'metric:rgb_controller_a8445df_mean_score', tone: 'lime' },
            { label: '神經網路', value: 'metric:dqn_v3_2m_mean_score', tone: 'coral' },
          ],
        },
        {
          kind: 'p',
          text: '這個神經網路練到 40 萬格時平均 {{fact:dqn_v3_400k}} 分、100 萬格 {{fact:dqn_v3_1m}} 分、200 萬格 {{metric:dqn_v3_2m_mean_score}} 分，最好的一局 {{metric:dqn_v3_2m_max_score}} 分，一局都沒打光。追蹤程式的中位數是 {{metric:rgb_controller_a8445df_median_score}} 分。',
        },
        {
          kind: 'callout',
          tone: 'muted',
          title: '我的猜測，與下一步',
          text: '我的猜測有兩個。一是規則變難了：每格都要決定，同一段遊戲時間裡要做四倍的決定，按下按鈕到看見得分之間隔得更遠，神經網路更難把功勞算到對的那一步。二是練得不夠：這個網路只練過一次、只有 200 萬格。下一步把兩者拆開：先把同樣的網路放回「每 4 格決定一次」的規則、玩同樣 15 局，看分數回不回得來，分清是規則的問題還是模型的問題；再在「每格都決定」的規則下獨立訓練三次、各練 500 萬格，不讓一次訓練的運氣定生死。',
        },
      ],
    },
    {
      id: 'clears',
      nav: '破關驗證',
      kicker: '破關驗證',
      title: '15 局裡打光 4 局，而且每一局都由程式判定',
      blocks: [
        {
          kind: 'p',
          text: '打磚塊有上下兩面磚牆。第一面打光是 {{fact:first_wall_score}} 分，兩面都打光是 {{metric:clear_score_threshold}} 分，之後沒有下一關。下面是追蹤程式 15 局的最終分數：',
        },
        {
          kind: 'episodes',
          caption: '追蹤程式 15 局的最終分數',
          gameLabel: '第 {n} 局',
          clearedLabel: '兩面牆打光',
          firstWallLabel: '第一面牆',
          bothWallsLabel: '兩面牆',
        },
        {
          kind: 'p',
          text: '打光的 4 局是第 6、12、14、15 局，打光當下還剩 1 到 2 條命。最快的一局在第 {{metric:rgb_controller_a8445df_best_clear_ale_frame}} 格畫面打光，大約是 6 分 18 秒的遊戲時間。沒打光的 11 局裡，最高的一局 {{fact:best_unclear_score}} 分，只差最後幾塊磚。',
        },
        {
          kind: 'p',
          text: '「打光」不能靠看畫面：打完第一面牆時畫面也會短暫全空，打光後球還會繼續彈。所以判定交給讀遊戲記憶體的判定器：記憶體裡的計分板到 {{metric:clear_score_threshold}} 分，而且和遊戲回傳的分數一致，才算打光；兩邊對不上就標成「不確定」，絕不當成 0。',
        },
        {
          kind: 'p',
          text: '判定器本身也要驗證。我錄下三種真實軌跡：完整打光、只打完第一面牆、剩最後一塊磚。判定器只能在第一種、剛好第二面牆清空的那一格說「打光」。我也故意把它改壞 {{fact:detector_mutants}} 種方式，測試每一種都抓得到。',
        },
        {
          kind: 'links',
          items: [
            { label: '判定器的六種改壞方式', detail: '每一種會造成什麼錯、由哪個測試抓到', href: `${REPO}/blob/main/docs/clear-semantics.md` },
          ],
        },
        {
          kind: 'p',
          text: '網頁版裡同一套追蹤邏輯（移植成 TypeScript）另外跑了 120 局：{{metric:browser_wasm_120_canonical_clear_detections}} 局打光，平均 {{metric:browser_wasm_120_mean_score}} 分。',
        },
        {
          kind: 'callout',
          tone: 'muted',
          title: '這個打光比例代表什麼',
          text: '正式 {{seeds:rgb_controller_a8445df_canonical_clear_detections}} 局打光 {{metric:rgb_controller_a8445df_canonical_clear_detections}} 局，網頁版 {{seeds:browser_wasm_120_canonical_clear_detections}} 局打光 {{metric:browser_wasm_120_canonical_clear_detections}} 局，兩個比例落在同一個範圍：這支程式大約每三、四局能打光一局。它證明了不用訓練也打得光兩面牆，但還不是每局都打得光。',
        },
      ],
    },
    {
      id: 'failures',
      nav: '沒成功的嘗試',
      kicker: '失敗紀錄',
      title: '三個沒成功的嘗試，也都留下了數字',
      blocks: [
        {
          kind: 'p',
          text: '失敗也寫進紀錄，「沒用」本身就是結果。',
        },
        {
          kind: 'ledger',
          items: [
            {
              name: '掉命多扣分',
              idea: '每掉一條命多扣 1 分，逼它更小心。獨立訓練三次、各練 100 萬步。',
              result: '沒有穩定改善',
              detail: '平均 +{{metric:reward_shaping_mean_effect}} 分，但 95% 區間從 {{metric:reward_shaping_ci_low}} 到 +{{metric:reward_shaping_ci_high}}；三次裡只有 {{metric:reward_shaping_positive_training_seeds}} 次變好，沒過採用門檻。',
            },
            {
              name: '優先回放',
              idea: '優先重看「猜得最離譜」的經驗，而不是平均抽。',
              result: '差距太小，不值得',
              detail: '50 萬步時平均 {{metric:per_500k_mean_score}} 對 {{metric:uniform_500k_mean_score}}，訓練時間卻是 {{metric:per_training_time_ratio}} 倍。',
            },
            {
              name: 'Laya 現成模型',
              idea: '拿現成的看圖模型 Laya（{{metric:laya_parameter_count}} 參數，以 SmolVLM 為底）凍結不訓練，直接看畫面選動作。它先前看過一些打磚塊的專家操作，不算從零開始。',
              result: '玩不起來',
              detail: '在瀏覽器用同一個開局玩了兩次，兩次都只拿 {{metric:laya_browser_session_score}} 分，約 {{metric:laya_browser_session_played_s}} 秒就用完五條命。完整的正式評估局數：{{metric:laya_complete_episodes}}。',
            },
          ],
        },
        {
          kind: 'p',
          text: '問題不在速度：頁面每 {{metric:laya_browser_decision_interval_p50_ms}} 毫秒送出一次決策，幾乎剛好跟上每秒 60 格畫面的節奏，畫面每秒實際呈現 {{metric:laya_browser_presented_fps}} 張。問題在決策品質：它常常前一刻往左、下一刻往右，球拍在原地抖動。我在另一組短測試裡加了一道煞車，它想立刻反向時先多等一次決策；連續反向從 {{metric:laya_control_diagnostic_raw_reversals}} 次降到 {{metric:laya_control_diagnostic_reversals}} 次，球拍還是接不住球。抖動少了、分數沒變，所以抖動不是主因，它真正缺的是判斷球會落在哪。',
        },
      ],
    },
    {
      id: 'conclusion',
      nav: '結論',
      kicker: '結論',
      title: '什麼需要學，什麼不需要',
      blocks: [
        {
          kind: 'bars',
          caption: '三條路線的平均分（刻度是滿分）',
          max: 'metric:clear_score_threshold',
          maxLabel: '滿分',
          items: [
            { label: '追蹤程式', value: 'metric:rgb_controller_a8445df_mean_score', tone: 'lime' },
            { label: '神經網路', value: 'metric:dqn_v3_2m_mean_score', tone: 'coral' },
            { label: 'Laya', value: 'metric:laya_browser_session_score', tone: 'blue', note: '規則不同，只在瀏覽器玩過同一個開局' },
          ],
        },
        {
          kind: 'table',
          caption: '三條路線並排比較',
          headers: ['', '追蹤程式', '神經網路', 'Laya'],
          rows: [
            ['需要訓練嗎', '不用', '要，200 萬格畫面', '沒有為這個專案訓練'],
            ['平均分', '{{metric:rgb_controller_a8445df_mean_score}}', '{{metric:dqn_v3_2m_mean_score}}', '同一個開局玩兩次：都是 {{metric:laya_browser_session_score}}'],
            ['打光兩面牆', '{{metric:rgb_controller_a8445df_canonical_clear_detections}}／{{seeds:rgb_controller_a8445df_canonical_clear_detections}}', '{{metric:dqn_v3_2m_clears}}／{{seeds:dqn_v3_2m_clears}}', '沒有完整評估'],
            ['每次決策', '約 {{metric:rgb_controller_a8445df_latency_ms_mean}} 毫秒', '約 {{fact:web_wasm_p50_ms}} 毫秒（瀏覽器）', '約 {{metric:laya_browser_decision_interval_p50_ms}} 毫秒'],
            ['看得懂它在想什麼嗎', '看得懂，每一步都算得出來', '很難，只有四個 Q 值', '很難'],
          ],
        },
        {
          kind: 'p',
          text: '需要學的，是規則說不清楚的部分。打磚塊的物理很單純：球直線飛、撞牆反彈、角度可以預測，幾行算式就能描述。這時讓神經網路花上百萬格畫面自己摸索，成本高，還不一定學得到。',
        },
        {
          kind: 'p',
          text: '不需要學、但需要大量工程的，是周邊的一切：固定發球規則、逐局看評估、驗證破關判定器、逐張比對部署結果。這個專案的時間大多花在這裡，這也是最值得帶走的部分。',
        },
        {
          kind: 'p',
          text: '還沒回答的問題：同一套規則下多次獨立訓練的神經網路能走到哪裡？追蹤程式能不能每局都打光？這些都寫在 GitHub 的交接文件裡，留給下一步。',
        },
      ],
    },
    {
      id: 'evidence',
      nav: '證據與重現',
      kicker: '證據與重現',
      title: '每個數字都查得到出處',
      blocks: [
        {
          kind: 'p',
          text: '這頁的結果數字不是手抄的：頁面打開時直接讀取專案的成果檔，和 GitHub 上的成果報告用的是同一份，所以兩邊永遠一致；30 天過程中的數字則來自當天的文章。追蹤程式的 15 局可以自己重跑一遍，步驟在下方的「重現步驟」：',
        },
        {
          kind: 'links',
          items: [
            { label: '完整成果報告', detail: '數字來源、驗證狀態與限制', href: `${REPO}/blob/main/docs/final-results/README.md` },
            { label: '破關判定與驗證', detail: '判定器如何運作與校準', href: `${REPO}/blob/main/docs/clear-semantics.md` },
            { label: '追蹤程式 15 局紀錄', detail: '逐局分數與打光時間', href: `${REPO}/blob/main/audits/browser-clear/native-15/report.md` },
            { label: '重現步驟', detail: '從安裝到跑出結果', href: `${REPO}/blob/main/docs/repro/README.md` },
            { label: '30 天的完整文章', detail: '每天做了什麼、遇到什麼', href: `${REPO}/tree/docs-ironman-series` },
            { label: '線上試玩', detail: '在瀏覽器裡和 AI 對戰', href: '/' },
          ],
        },
      ],
    },
  ],
  footer: {
    updated: '最後更新',
  },
};
