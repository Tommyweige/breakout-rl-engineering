# Day 13｜DQN 不學時怎麼查：先證明程式沒壞，再談超參數

Day 12 之後，DQN 已經能完整跑起來了：Agent 會和 Breakout 互動、把經驗存進 Replay Buffer，也會真的更新模型。

但這裡有一個很麻煩的地方：**程式會訓練，不代表 Agent 真的有變強。**

強化學習最難除錯的情況，往往不是程式直接 crash，而是下面這種狀態：

```text
程式正常跑
loss 也有數字
GPU 也一直在算
但遊戲分數就是沒有進步
```

這時候如果第一反應只是「換 learning rate 試試看」，很容易把真正的 bug 蓋掉。

所以 Day 13 的目標不是找出最好的參數，而是先建立一套固定的除錯順序：**先確認模型本身會學，再確認更新過程的數值正常，最後才看 Agent 收集到的資料和遊戲表現。**

## RL 的 bug，常常不會讓程式停止

在一般監督式學習裡，資料和答案通常都是固定的。輸入形狀錯了、標籤對不上，很多問題很快就會報錯。

RL 不一樣。Agent 會自己和環境互動，今天模型做出的 action，又會影響下一批收集到的資料。DQN 的學習目標也不是固定標籤，而是會隨著訓練持續改變。

所以很多問題都能「正常跑完」。例如：

- action 都是合法的，但模型幾乎永遠只選同一個動作；
- Replay Buffer 一直有資料進來，但模型其實沒有真的更新；
- 模型選到錯的 Q-value，形狀卻完全合法，所以 PyTorch 不會報錯；
- loss 看起來很小，但模型只是越來越會貼合一批偏掉的資料；
- Q-value 一路變大，還沒變成 NaN，程式仍然繼續跑。

也因此，Day 13 的第一個原則是：**先確認整條學習流程是對的，再開始懷疑超參數。**

## 先把除錯順序固定下來

我把 DQN 的除錯流程整理成三關：

1. **模型本身會不會學？**
2. **模型更新時，數值有沒有出問題？**
3. **模型能更新，但 Agent 的行為和資料正常嗎？**

只有這三關都沒有明顯問題，才輪到 learning rate、gamma 或其他超參數。

[![從數值檢查、固定 batch 到探索訊號與最後超參數調整的 RL 除錯流程](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/debugging-workflow.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/debugging-workflow.png)

圖裡最重要的不是每一個名詞，而是順序。

如果 loss、Q-value 或梯度已經變成 NaN / infinity，先找數值為什麼爆掉；如果模型連固定的一小批資料都學不會，就先查模型更新流程；這些都正常之後，才有理由把注意力移到探索、Replay Buffer 和遊戲分數。

## 第一關：模型連同一份小考都背不起來嗎？

這一關使用的是 **fixed-batch overfit**。可以把它想成一個很簡單的測試：

> 把同一份小考和答案固定下來，讓模型重複做很多次。它至少應該要把這份題目背起來。

實際做法是固定一小批 transition，也固定它們對應的學習目標，然後連續更新模型很多次。

這個測試刻意拿掉了 RL 最麻煩的部分：資料不再變、目標也不再移動。如果模型在這種情況下還是學不會，就沒必要先怪 Breakout 太難或探索不夠。

這次在 RTX 4060、seed `42` 上重複更新 200 次，loss 從：

```text
0.2029491961
↓
0.000000001456
```

幾乎降到零。

這個結果很重要，因為它至少支持一件事：**模型能完成「做出 Q-value 預測 → 和固定答案比較 → 反向更新參數」這條基本學習流程。**

如果這個測試失敗，優先要查的是 action 對應、target、資料型態、正規化、反向傳播和 optimizer，而不是先調遊戲層面的參數。

不過 fixed-batch overfit 只是一個最小測試。它成功不代表 Agent 會玩 Breakout，因為真正訓練時資料會一直改變，Agent 還得自己探索新的畫面。

既然模型本身至少有能力學，下一步就是確認：**真正訓練時，更新過程有沒有慢慢失控。**

## 第二關：模型在更新，但數值健康嗎？

這次我在 `cuda:0` 上用固定 seed `42` 跑了 10,000 個 environment steps。整個 run 完成 48 局、2,251 次模型更新，以及 21 次 Target Network 同步。

這些數字只能證明「trainer 有在做事」，還不能證明它做得對。所以接下來要一起看 loss、Q-value、target 和 gradient。

這幾條曲線都不是越平越好。DQN 每次是從 Replay Buffer 隨機抽一批不同的經驗來更新，而且 Target Network 還會定期同步，因此曲線本來就可能出現尖峰。真正需要判斷的是：**尖峰為什麼出現、整體基線是否持續抬高，以及不同訊號是不是一起失控。**

### Loss：尖峰通常來自「這一批資料比較難」

loss 可以先理解成「模型目前的 Q-value 和這次學習目標差多少」。DQN 常用的 TD error 也是在描述這個預測和目標之間的差距；loss 則把一整批資料的差距整理成一個可以拿來更新模型的數字。

這次 2,251 次更新裡，loss 都保持為正常有限數值，平均約 `0.00299`，最大約 `0.0463`。

[![10K-step CUDA debug run 的 Huber loss 曲線，尖峰代表部分 batch 的誤差較大](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/loss-curve.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/loss-curve.png)

圖裡的尖峰不是憑空出現的。每次更新抽到的 mini-batch 都不同，有些 batch 可能剛好包含比較少見的畫面、真正拿到 reward 的 transition，或目前 Q-value 和 target 差距特別大的資料。這時同一批資料的平均誤差就會突然變大，loss 也跟著形成波峰。

另外，Target Network 會週期性從 Online Network 複製新參數。同步後，某些 next state 的參考價值可能跟上一段時間不同，因此接下來抽到相同類型資料時，prediction 和 target 的距離也可能暫時變大。這是可能造成 loss 波動的另一個來源，但不能只看到一個尖峰就斷定「一定是 Target sync 造成」；要把尖峰發生的 step 和同步時間真正對上才算證據。

Day 12 使用的 Huber loss 會降低極端誤差對更新的影響，但它不會把這些波峰消掉。因此這種「大部分時間很低、偶爾跳高」的形狀本身並不奇怪。

真正值得警覺的是：loss 的基線和尖峰一起持續往上抬，甚至開始出現 NaN / infinity。反過來也一樣：**loss 很小，不能證明 Agent 已經學會。** 它只代表模型比較能貼近目前這批學習目標。

### Q-value 與 Target：為什麼 Target max 會有尖峰，還逐漸往上？

Q-value 是模型對「在這個畫面做某個 action 有多值得」的估計。Target 則是這次更新時拿來當參考答案的數值。

DQN 的 target 可以簡化成：

```text
target = 這一步拿到的 reward
       + gamma × 下一個 state 的最大 Q-value
```

因此 target 並不是固定答案。只要下一個 state 的估計價值變了，target 就會跟著變。

[![10K-step CUDA debug run 的所有 action Q-values 與 Bellman targets](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/q-values.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/q-values.png)

這張圖裡的 **Target max** 有兩個值得分開看的現象：尖峰，以及整體往上的趨勢。

先看尖峰。`max` 本來就是一個對極端值很敏感的統計量：一個 mini-batch 裡只要有一筆 transition 剛好拿到正 reward，或它的 next state 被 Target Network 評估成特別有價值，那一批的 Target max 就可能突然跳高。這也是為什麼 Target max 通常比 Target mean 更容易出現明顯波峰。

再看整體趨勢。這次 Q mean 大致從接近 `0` 升到約 `0.26`，Target max 的上緣也隨訓練逐漸抬高，最高觀測值約 `1.39`。這其實符合 DQN 的 bootstrapping 特性：Online Network 開始對某些 state 給出比較高的未來價值後，定期同步會把這些較高的估計複製到 Target Network；下一輪計算 target 時，`gamma × 下一步最大 Q-value` 這一項也會跟著提高。

所以 **Target max 往上不一定是壞事**。它可能只是代表模型不再把所有畫面都估成接近零，而是開始拉開「比較有希望的 state」和其他 state 的價值差距。

但這裡也正是 DQN 需要小心的地方：target 是拿另一個 Q-value 算出來的，如果模型只是越估越樂觀，較高的 Q-value 又被複製進 Target Network，下一輪 target 就可能繼續被往上推。這種正回饋如果沒有被真實 reward 和遊戲表現支持，就可能演變成 overestimation，甚至讓數值逐步發散。

因此這張圖不能只問「Target max 有沒有變大」，而要一起問：

- Q-value 和 target 是不是以相近尺度往上？
- prediction 和 target 的差距有沒有越拉越開？
- loss 是否也跟著長期抬升？
- 最重要的是，return 有沒有真的改善？

這次所有值仍維持有限，Q-value 也沒有看到無限制暴增，因此目前比較合理的結論是：**Target max 確實有往上走，值得持續監控，但光靠這條上升趨勢還不能判定訓練已經發散，也不能反過來當成 Agent 正在進步的證據。**

### Gradient norm：尖峰代表這一次「想改得比較多」

反向傳播之後，每個參數都會得到一個「應該往哪裡調」的梯度。**Gradient norm** 就是把所有參數的梯度濃縮成一個總大小，可以把它理解成「這一次模型想把參數推動多強」。

[![10K-step CUDA debug run 的 gradient norm 曲線，數值取 clipping 前的總 norm](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/gradient-norm.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/gradient-norm.png)

它出現尖峰的原因和 loss 有關，但兩者不是完全同一個東西。當某一批資料的 Q-value 和 target 差得比較多時，反向傳播通常會產生更強的修正訊號，因此 gradient norm 也容易突然變大。Target Network 同步後如果參考值產生變化，或 mini-batch 剛好抽到 reward / 高 TD error 的 transition，也可能讓某一次 gradient 明顯高於前後更新。

不過 gradient 的大小不只由 loss 決定。它還取決於神經網路本身對參數有多敏感，所以不能簡化成「loss 大兩倍，gradient 就一定大兩倍」。

而且這張圖不只有尖峰，**後段的 gradient norm 整體也比前段高，確實有逐漸抬升的趨勢。** 一個合理的解釋是：隨著 Q-value 和 target 的尺度提高，某些 batch 需要的修正幅度也變大；Replay Buffer 後期包含的狀態更加多樣，也可能讓更新訊號比訓練初期更強。

這裡要特別注意，圖畫的是 **gradient clipping 之前**的總 norm。這次設定的 clipping 門檻是 `10.0`，而實際觀察到的最大值只有約 `0.1952`，所以這條上升趨勢不是因為 clipping 把數值截成某種形狀；實際上這次 run 根本還遠不到 clipping 門檻。

因此目前的判讀不是「gradient 沒有變大」，而是：**它確實在變大，但目前仍維持在有限而且相對小的尺度。** 這是一個應該追蹤的早期訊號，而不是單憑它就宣布 gradient explosion。

### Target max 和 Gradient norm 一起往上，代表什麼？

把前兩張圖放在一起看會更有意思：Target max 的上緣逐漸提高，Gradient norm 的基線也在後段抬高。

這兩件事在機制上可能有關聯。Target Network 給出的參考值變高之後，如果 Online Network 還沒有完全跟上，prediction 和 target 之間的差距就可能變大；差距變大時，某些 batch 會產生更強的反向更新訊號，因此 gradient norm 也可能提高。

但**「兩條曲線一起往上」不等於已經證明 Target max 導致 Gradient norm 上升。** 它們也可能同時受到 Q-value 尺度、Replay Buffer 資料分布，以及不同 batch 難度影響。要證明直接因果，還需要把 Target sync、TD error、loss 和 gradient 在相同步數上對齊分析。

所以 Day 13 最重要的判讀不是看到上升就立刻下結論，而是建立一組連鎖警訊：

```text
Target / Q-value 持續抬高
        ↓
Prediction 和 Target 的差距越拉越大
        ↓
Loss 基線持續上升
        ↓
Gradient norm 也持續加速變大
        ↓
最後出現極端值或 NaN / infinity
```

如果這幾件事一起發生，就很像真的進入不穩定的 bootstrapping 正回饋；如果只有 Target max 和 Gradient norm 緩慢抬高，但 loss 仍受控、數值仍有限，就比較適合標記成「需要持續觀察」，而不是直接判定訓練壞掉。

到這裡，我們看到的情況比較接近後者：**Target max 和 Gradient norm 確實有上升趨勢，但目前沒有伴隨 loss 長期爆大或非有限數值。** 因此還沒有足夠證據說模型正在數值發散。

下一個可能性就是：模型能更新、數值暫時也沒有失控，但它收集到的經驗或實際行為出了問題。

## 第三關：Agent 收集到的資料正常嗎？

DQN 不只是在學模型，它也一直用目前的策略去產生下一批訓練資料。

如果 action 幾乎永遠只有一種，或者探索根本沒有照設定發生，那麼模型再健康也可能學不到有用的東西。

這次 10K-step run 的 action 分布是：

| Action | 次數 |
| --- | ---: |
| NOOP | 2,512 |
| FIRE | 2,546 |
| RIGHT | 2,841 |
| LEFT | 2,101 |

四個 action 都有實際出現，至少可以排除「某個 action 完全沒被送進環境」這種明顯問題。

同一段訓練裡，隨機選 action 有 `4,754` 次，依照目前 Q-value 選 action 有 `5,246` 次；epsilon 則從 `0.9` 降到約 `0.0501`。

這些資料能支持的是：**探索排程確實有在運作，Agent 也沒有完全卡死在單一 action。**

它們不能支持「現在依 Q-value 選出的 action 已經是好策略」。那要回到真正的遊戲分數才能判斷。

Replay Buffer 也需要一起看。這次容量是 10,000，跑完時也存滿 10,000 筆經驗。這讓我們知道問題不是「Buffer 還停在 warm-up，模型根本沒開始學」。

## 最後才看分數：10K steps 還沒有學習證據

前面幾關都沒有看到明顯的 correctness 問題之後，才比較有意義去看 return，也就是每一局累積的原始遊戲分數。

[![10K-step CUDA debug run 的 raw episode return 曲線](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/return-curve.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/return-curve.png)

這次 48 個 episode 的 return 介於 `0` 到 `5`，平均只有 `1.40`；第一局是 `2`，最後一局是 `1`。

中間偶爾出現 `4` 或 `5`，但這不能直接解讀成 Agent 已經開始穩定進步。反過來，最後一局比第一局低，也不能單靠這點宣稱模型完全沒有學到東西。

在這個短 run 和單一 seed 下，最保守、也最正確的結論是：**目前還看不到可靠的學習趨勢。**

為了知道「完全沒有學習」時大概會長什麼樣子，我也跑了 5 局純隨機 action 的 baseline，分數是：

```text
2, 1, 5, 1, 5
平均 2.8
```

這個結果不能直接拿來宣布 random agent 比 DQN 強。5 局太少，而且這也不是正式 evaluation protocol。

它真正的用途是提醒我們：Breakout 的短期分數本來就有很大波動。如果只拿幾局互相比高低，很容易把運氣誤認成學習。

## 一次失敗的訓練，也要能回頭查

RL 除錯還有一個很容易被忽略的問題：今天看到異常曲線，過幾天回頭時，還記得那次到底用了哪一版程式嗎？

所以每次 run 除了曲線，還需要保存足夠的背景資料，例如：

- seed；
- 使用 CPU 還是 GPU；
- Python / PyTorch / Gymnasium / ALE 版本；
- 當時的 Git commit；
- 訓練設定和環境設定。

這些資料不需要全部塞進文章，但一定要留在 run artifact 裡。

這樣之後看到問題時，才能從「我記得那次好像怪怪的」，變成「在這個 seed、這個 commit、這個環境版本下，哪個訊號最先開始異常」。

## 先排除 bug，再開始調參

Day 13 最重要的成果不是多了幾張曲線，而是把除錯順序固定下來：

```text
模型連固定資料都學不會？
→ 先查基本學習流程

模型會學，但數值開始失控？
→ 查 loss、Q-value、gradient、target

數值正常，但 Agent 行為怪？
→ 查 action、epsilon、Replay Buffer

這些都正常，但 return 還是不升？
→ 才開始做受控的超參數實驗
```

這次真實 CUDA debug run 可以證明：模型能在固定 batch 上學習；正式 training loop 也確實有更新、Target Network 有同步；loss 沒有出現長期爆大或非有限值，而 Target max 和 Gradient norm 雖然有逐漸抬高的趨勢，目前仍在有限尺度內；探索和 action 分布也都有在運作。

但 10,000 steps 的 return 仍然沒有提供穩定的學習證據。

這個結果其實很有價值，因為它讓下一步不再是「隨便換個參數試試看」。Day 14 才會在固定條件下，真正開始比較 hyperparameters 對訓練結果的影響。

下一篇：[Day 14｜超參數實驗：用受控比較取代「改一個數字試試看」](day14-hyperparameter-experiments.md)
