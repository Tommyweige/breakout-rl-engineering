# Day 23｜換成 ONNX Runtime 之後，它還是同一個 Agent 嗎？

Day 22，我們已經把訓練好的 PyTorch 模型轉成 ONNX。

乍看之下，好像只差把它丟進瀏覽器就完成了。

但我不太想直接往下衝，因為中間還有一個很基本的問題沒有回答：

> **同一個 Breakout 畫面，交給 PyTorch 和 ONNX Runtime，最後會不會做出不同的決定？**

如果答案是會，那後面不管 Web Demo 做得多漂亮，都沒有意義。

所以 Day 23 先不談速度，也不做網頁。我只做一件事：**確認換了執行模型的方式之後，Agent 的判斷有沒有跑掉。**

---

## ONNX 和 ONNX Runtime，其實是兩個不同的東西

昨天把模型轉成 `.onnx` 時，我們得到的是一個**模型檔案**。

但模型檔案本身不會自己運算。

可以把兩者想成：

- **ONNX**：模型的通用格式；
- **ONNX Runtime**：真正負責把 ONNX 模型跑起來的程式。

有點像影片檔和播放器的關係。

`.mp4` 只是影片檔，真正把畫面播出來的是播放器；同樣地，`.onnx` 只是模型，而真正把輸入送進去、算出結果的是 ONNX Runtime。

這就帶出今天的問題：

> 換了「播放器」之後，模型算出來的結果會不會變？

---

## 為什麼不能只看「模型有成功跑起來」？

Breakout 的 Agent 每看到一個遊戲狀態，都會對四個動作各算出一個 Q-value。

可以先把 Q-value 想成：

> **模型現在有多想做這個動作。**

例如某一刻可能是：

```text
不動   1.2
發球   0.4
往右   3.8
往左   2.1
```

最高的是 `3.8`，所以 Agent 會選「往右」。

如果換成 ONNX Runtime 之後變成：

```text
不動   1.2
發球   0.4
往右   3.7999
往左   2.1001
```

這其實沒什麼問題，最後還是選「往右」。

但如果兩個最高分原本就很接近，一點點誤差也可能讓第一名和第二名交換。

那就會變成：

```text
PyTorch      → 往右
ONNX Runtime → 往左
```

所以今天要看的不是只有「數字差多少」，還要看一件更直接的事：

> **最後選到的動作有沒有改變？**

---

## 我怎麼測？拿同一批遊戲狀態給兩邊算

Day 22 已經先保留了一批固定的 Breakout 遊戲狀態。

這次我沒有重新玩遊戲，也沒有重新挑模型，而是把**完全同一批輸入**分別交給：

- 原本的 PyTorch 模型；
- ONNX Runtime CPU；
- ONNX Runtime CUDA。

然後比較兩件事：

1. 四個 Q-value 差多少；
2. 最後選出的動作是不是一樣。

這樣做的好處很直觀。

如果三邊看到的畫面不一樣，那根本不知道差異是模型造成的，還是遊戲本身造成的；現在輸入固定，就能把問題縮小成：

> **只是換了執行模型的工具，答案有沒有變？**

---

## 結果：60 個遊戲狀態，動作全部一致

下面這張圖是直接由實際比較結果產生的，不是另外手填數字。

[![PyTorch 與 ONNX Runtime 的 Q-value 與動作比較](https://github.com/Tommyweige/breakout-rl-engineering/blob/d8b2bea0de403c7b6addcd08e5f45911bd73d92b/assets/day23/pytorch-vs-onnx-parity.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/d8b2bea0de403c7b6addcd08e5f45911bd73d92b/assets/day23/pytorch-vs-onnx-parity.png)

這張圖可以不用一次看懂所有細節，只要先抓兩件事。

第一，PyTorch 和 ONNX Runtime 算出的 Q-value 非常接近。

第二，也是更重要的：**60 個狀態裡，最後選出的動作 60 / 60 全部一致。**

實際結果如下：

| 執行方式 | 一次送幾個狀態 | 最大 Q-value 差異 | 最後動作一致率 |
| --- | ---: | ---: | ---: |
| ONNX Runtime CPU | 1 | 約 `0.000162` | **100%** |
| ONNX Runtime CPU | 4 | 約 `0.000162` | **100%** |
| ONNX Runtime CUDA | 1 | 約 `0.000162` | **100%** |
| ONNX Runtime CUDA | 4 | 約 `0.000162` | **100%** |

可以看到數字並不是完全一模一樣。

這很正常。不同執行環境在浮點運算上出現非常小的差異，不代表模型壞掉。

真正值得注意的是：

```text
60 個狀態
× CPU / CUDA
× 不同 batch size
↓
最後動作全部一樣
```

也就是說，至少在這批測試裡，**換成 ONNX Runtime 並沒有改變 Agent 的決策。**

---

## CPU 和 GPU 都測，是因為後面會走不同執行環境

ONNX Runtime 可以用 CPU 跑，也可以使用 NVIDIA CUDA 在 GPU 上執行。

今天兩條路都測，是因為後面部署時不一定永遠只跑在同一種硬體上。

這次結果是：

- CPU：通過；
- CUDA：通過。

CUDA 這部分我還特別確認了一件事：不是「要求 GPU，結果偷偷跑 CPU」。

實際檢查顯示，這個 ONNX 模型的 16 個運算節點全部由 CUDA 執行，沒有節點掉回 CPU。

這個細節對一般使用者其實不重要，但對這個專案很重要，因為我不希望最後寫著「CUDA 測試成功」，其實只是程式默默改用 CPU。

---

## 只比較固定畫面還不夠，所以再讓它真的玩一下

離線比較都通過之後，我又把 ONNX Runtime 真的接回 Breakout，讓它跑 256 個步驟。

同一個起始條件下：

```text
PyTorch CUDA
→ 跑完 256 steps
→ 得分 6

ONNX Runtime CUDA
→ 跑完 256 steps
→ 得分 6
```

兩邊在這段短測試裡，要求的動作分布也完全一樣。

這個結果不能拿來證明：

> ONNX Runtime 玩一整局的分數永遠跟 PyTorch 完全相同。

因為強化學習環境本身還可能有隨機性，長時間跑下去也可能因為很小的差異逐漸走到不同狀態。

這個短測試真正想確認的是：

> **ONNX 模型不只是在離線數字比較裡看起來正常，它真的能接回 Breakout 控制遊戲。**

這樣就夠了。

---

## Day 23 到底證明了什麼？

今天可以很有把握地說：

- PyTorch 和 ONNX Runtime 的 Q-value 非常接近；
- 60 個固定遊戲狀態全部選出相同動作；
- CPU 和 CUDA 都得到相同結論；
- ONNX Runtime CUDA 確實使用 GPU 執行；
- ONNX 版本真的能接回 Breakout 跑起來。

但今天**還沒有**證明：

- ONNX Runtime 一定比較快；
- Browser 裡的 WASM / WebGPU 也一定完全相同；
- ONNX 版本的完整遊戲分數一定和 PyTorch 一樣。

這些都是後面要分開回答的問題。

---

## 小結：現在才真的敢往部署走

Day 22 做的是：

> **把 PyTorch 模型轉成 ONNX。**

Day 23 則是在確認：

> **轉完之後，它還是不是原本那個 Agent？**

目前答案是肯定的。

雖然 Q-value 有非常小的浮點差異，但在 60 個固定遊戲狀態裡，PyTorch、ONNX Runtime CPU、ONNX Runtime CUDA 最後全部做出相同決定。

這件事先確認之後，下一步談效能才有意義。

因為如果模型都已經跑偏了，那「每秒快幾毫秒」根本不重要。

所以 Day 24，接下來要問的就是：

> **既然 ONNX Runtime 的答案沒變，那它跑得多快？真正放進即時 Breakout Demo 時，推論延遲會不會成為瓶頸？**
