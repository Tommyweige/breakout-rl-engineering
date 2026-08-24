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

這些數字只能證明「trainer 有在做事」，還不能證明它做得對。所以接下來要一起看 loss、Q-value 和 gradient。

### Loss：有數字，不等於學得好

loss 可以先理解成「模型目前的 Q-value 和這次學習目標差多少」。DQN 常用的 TD error 也是在描述這個預測和目標之間的差距；loss 則把這些差距整理成一個可以拿來更新模型的數字。

這次 2,251 次更新裡，loss 都保持為正常有限數值，平均約 `0.00299`，最大約 `0.0463`。

[![10K-step CUDA debug run 的 Huber loss 曲線，尖峰代表部分 batch 的誤差較大](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/loss-curve.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/loss-curve.png)

圖裡偶爾有尖峰，表示某些抽到的資料和目前模型預測差得比較多。這本身不一定是 bug。

真正值得警覺的是：loss 開始持續爆大，或直接變成 NaN / infinity。如果那種情況發生，後面再看遊戲分數就沒有太大意義了。

但反過來也一樣：**loss 很小，不能證明 Agent 已經學會。** 它只代表模型比較能貼近目前這批學習目標。

### Q-value：看趨勢，不要硬設一條通用門檻

Q-value 是模型對「在這個畫面做某個 action 有多值得」的估計。

這次所有 action 的 Q-value 都有一起被記錄，而不是只看最後真的被選中的 action。

[![10K-step CUDA debug run 的所有 action Q-values 與 Bellman targets](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/q-values.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/q-values.png)

這個 run 裡，Q mean 大致從接近 `0` 升到約 `0.26`，而 target max 的最高觀測值約 `1.39`。目前沒有看到數值一路無限制暴增，也沒有出現 NaN 或 infinity。

這不代表 Q-value 一定「正確」。比較合理的判讀方式，是看它相對 reward 和 target 的尺度是否逐漸失控，而不是訂一條「Q > 某個數字就一定錯」的通用規則。

### Gradient：模型有沒有收到更新訊號

反向傳播之後，每個參數都會得到一個「應該往哪裡調」的梯度。**Gradient norm** 就是把這些梯度濃縮成一個大小，方便觀察這次更新到底有多強。

[![10K-step CUDA debug run 的 gradient norm 曲線，數值取 clipping 前的總 norm](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/gradient-norm.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/39ae567c4ee01e11b4a9405ba1dd1c1e4af5a6d5/assets/day13/gradient-norm.png)

這次 gradient norm 最大約 `0.1952`。它有波動，但沒有持續變大，也沒有出現非正常數值。

單一尖峰不一定有問題；比較危險的是 gradient、loss 和 Q-value 同時越來越大。那時候就該先查 reward、target、輸入正規化或 optimizer，而不是直接換演算法。

到這裡，至少沒有看到明顯的數值爆炸。那下一個可能性就是：**模型能正常更新，但它收集到的經驗或實際行為出了問題。**

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

這次真實 CUDA debug run 可以證明：模型能在固定 batch 上學習；正式 training loop 也確實有更新、Target Network 有同步，loss、Q-value 和 gradient 沒有出現明顯數值爆炸；探索和 action 分布也都有在運作。

但 10,000 steps 的 return 仍然沒有提供穩定的學習證據。

這個結果其實很有價值，因為它讓下一步不再是「隨便換個參數試試看」。Day 14 才會在固定條件下，真正開始比較 hyperparameters 對訓練結果的影響。
