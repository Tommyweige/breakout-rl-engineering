# Day 27｜把 Breakout 模型真的搬進瀏覽器

前幾天，Breakout Agent 雖然已經訓練完成，也能替四個動作做決定，但大部分事情都還是在 Python 裡完成。

如果最後想把它做成一個「打開網頁就能用」的作品，模型就不能永遠躲在 Python 後面。

所以 Day 27 要做的事情很直接：

> **把已經訓練好的模型放進網頁，讓瀏覽器自己算答案。**

但光是「成功載入模型」還不夠。

我還要確認一件更重要的事：

> **同一個畫面交給 Python 和瀏覽器，最後會不會選到不同動作？**

今天還沒有真的讓 Agent 玩完整的 Breakout。先把模型搬進網頁，而且確認它沒有在搬家的過程中跑掉。

---

## 瀏覽器怎麼直接跑 AI 模型？

Day 22 已經把模型轉成 ONNX。

可以先把 ONNX 想成一種比較通用的模型檔案格式。模型不再只能交給 PyTorch，而是可以換其他工具來讀。

到了網頁裡，我使用 **ONNX Runtime Web**。

它的工作可以簡單理解成：

> **讓瀏覽器直接讀取 ONNX 模型，然後自己完成計算。**

也就是說，使用者打開網頁之後，不需要把每一個畫面傳回 Python 伺服器，再等伺服器算完答案送回來。

Day 27 先使用 WASM。

WASM 全名是 WebAssembly，這裡不用先研究它的底層原理，可以先把它理解成：

> **一種讓瀏覽器能有效率執行程式的方式。**

所以今天真正打通的是：

```text
訓練好的模型
→ 轉成 ONNX
→ 網頁載入模型
→ 瀏覽器自己計算
→ 得到 Agent 的動作
```

---

## 昨天不是才測出 TensorRT 比較快嗎？

Day 26 的結果沒有被推翻。

TensorRT FP32 在我的 NVIDIA GPU 上確實跑得更快，但它適合的是 **NVIDIA 電腦上的原生程式**。

Day 27 換了一個目標：

> **讓使用者直接打開 Chrome，就能在網頁裡執行模型。**

瀏覽器不能直接拿 Day 26 產生的 TensorRT engine 來執行，所以這裡改用 ONNX Runtime Web。

兩者的關係可以簡單看成：

```text
PyTorch 訓練好的模型
        ↓
       ONNX
      /    \
     /      \
NVIDIA 電腦   Browser
   ↓           ↓
TensorRT    ONNX Runtime Web
```

所以不是「TensorRT 和 ONNX 二選一」。

ONNX 是兩邊都能使用的模型格式；真正負責執行模型的工具，則依照環境選擇。

---

## 網頁現在可以做什麼？

目前頁面分成 Human 和 Agent 兩個區域。

左邊的 **Human Panel** 先把玩家輸入準備好，可以讀取：

```text
←      ArrowLeft
→      ArrowRight
SPACE  Space
```

按下鍵盤時，頁面也會直接顯示目前按了什麼。

不過這裡還沒有真正的 Atari Breakout 畫面，所以頁面會明確寫出：

```text
ALE gameplay not connected yet
```

右邊的 **Agent Panel** 則負責顯示模型的計算結果。

還沒執行驗證以前，模型尚未載入；按下：

```text
Load model + Validate WASM
```

瀏覽器才會真正載入 FP32 ONNX 模型，並開始計算。

頁面另外先準備了 `Start Both`、`Pause Both`、`Reset Both`，但現在它們只控制兩邊共用的狀態，還不會真的開始一局 Breakout。

---

## Agent 到底在算什麼？

Breakout Agent 每看到一個畫面，就會替四個動作打分：

```text
NOOP   什麼都不做
FIRE   發球
RIGHT  往右
LEFT   往左
```

這四個分數在強化學習裡叫做 **Q-value**。

可以先把它理解成：

> **模型正在比較「現在做哪個動作比較值得」。**

最後分數最高的動作，就是 Agent 會選的答案。

例如四個分數如果是：

```text
NOOP   2.56
FIRE   2.58
RIGHT  2.63
LEFT   2.53
```

那模型最後就會選 `RIGHT`。

---

## 為什麼要拿 60 個相同畫面來比？

同一個模型從 Python 搬進瀏覽器後，底層的計算方式變了。

即使模型本身完全沒有修改，最後的浮點數也可能出現非常小的差距。

所以我先準備了 **60 個完全固定的 Breakout 畫面狀態**。

每一個狀態都做兩次：

```text
Python 算一次
Browser WASM 再算一次
```

然後比較兩件事：

1. 四個動作的分數差多少？
2. 最後選到的動作是不是一樣？

這樣如果瀏覽器版本真的哪裡算錯，比直接看整局遊戲更容易抓到問題。

因為整局遊戲只要某一步選到不同動作：

```text
這一步動作不同
→ 下一張畫面不同
→ 後面看到的畫面也不同
→ 兩局遊戲越走越遠
```

到那時候，就很難只看最後結果判斷問題到底出在哪裡。

---

## 實際結果：60 個狀態全部選到相同動作

下面這張圖是模型部署到 Cloudflare Pages 後，在 Chrome 裡真的跑完 60 個狀態的結果。

[![Chrome 中的 Day 27 WASM 驗證結果](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/c5153e6/assets/day27/browser-wasm-validation.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/c5153e6/assets/day27/browser-wasm-validation.png)

驗證完成之後，Agent Panel 會顯示 `PASS`，並列出瀏覽器算出的分數和原本的結果。

例如其中一個畫面：

| 動作 | 瀏覽器 | 原本結果 |
| --- | ---: | ---: |
| NOOP | 2.564468 | 2.564434 |
| FIRE | 2.575065 | 2.575027 |
| RIGHT | 2.634134 | 2.634130 |
| LEFT | 2.531219 | 2.531209 |

可以看到兩邊的數字不是每一位都完全相同，但差距非常小，而且最後都選了 `RIGHT`。

完整 60 個畫面的結果是：

| 比較項目 | 結果 |
| --- | ---: |
| 測試畫面 | 60 |
| 最大分數差異 | `0.0001616` |
| 平均分數差異 | `0.0000370` |
| 最後選到相同動作 | **60 / 60** |
| 動作不同 | **0** |

所以這次真正重要的結果是：

> **模型搬進瀏覽器之後，60 個相同畫面全部做出了相同的動作決定。**

---

## Human vs RL 的畫面先把位置留好了

下面是還沒開始驗證時的頁面。

[![Chrome 中的 Human / RL 雙欄網頁](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/c5153e6/assets/day27/dual-panel-browser-foundation.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering-private/blob/c5153e6/assets/day27/dual-panel-browser-foundation.png)

目前已經完成：

- 瀏覽器可以直接載入 FP32 ONNX 模型。
- Chrome 可以使用 WASM 執行模型。
- Human Panel 可以讀取 `←`、`→`、`Space`。
- Agent Panel 可以顯示四個動作的分數與最後選到的動作。
- 可以直接比較瀏覽器與原本模型的計算結果。
- 60 個固定畫面的驗證可以直接在部署後的網頁上完成。

但真正的 Breakout 還沒有出現在頁面裡。

目前仍然沒有：

```text
真的球
真的磚塊
真的 paddle
完整一局遊戲
```

這是刻意留下的界線。

Day 27 先確認模型在瀏覽器裡沒有明顯跑掉；真正的遊戲畫面、遊戲狀態與 Agent 控制會在後面再接上。

---

## Day 27 的答案

Day 27 最重要的成果可以濃縮成一句話：

> **同一個 Breakout Agent，現在已經能離開 Python，直接在 Chrome 裡自己算出動作。**

而且經過 60 個相同畫面的比較：

```text
最大分數差異：約 0.000162
平均分數差異：約 0.000037
動作一致：60 / 60
```

所以目前沒有看到模型搬進網頁之後出現明顯的決策錯誤。

今天回答的是：

> **「瀏覽器能不能正確跑這個模型？」**

答案是：可以。

下一步才會繼續看：

> **「如果改用瀏覽器的 GPU，能不能跑得更快？」**
