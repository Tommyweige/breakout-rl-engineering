# Day 26｜TensorRT 真的比較快嗎？RL 部署不能只看「每一步一不一樣」

Day 25 我試過 FP16。模型檔案確實變小了，但部分狀態會選到不同動作，而且一次只處理一個遊戲狀態時也沒有比較快。

Day 26 換一個方向：把原本的 ONNX 模型交給 NVIDIA TensorRT，讓它針對目前這張 GPU 建立一份專用的執行版本。

可以先把 TensorRT 想成：

> **不是重新訓練 Agent，而是想辦法讓同一個模型在 NVIDIA GPU 上跑得更有效率。**

所以今天最直接的問題是：

> **TensorRT 到底有沒有真的讓這個 Breakout Agent 變快？如果變快了，實際玩遊戲的能力有沒有跟著變差？**

---

## 先看速度：TensorRT FP32 這次真的比較快

這次在 RTX 4060 Laptop GPU 上測試，一次只處理一個遊戲狀態。

| 執行方式 | P50 | P95 |
| --- | ---: | ---: |
| PyTorch FP32 | 0.911 ms | 1.081 ms |
| ONNX Runtime FP32 | 0.667 ms | 1.110 ms |
| **TensorRT FP32** | **0.634 ms** | **0.761 ms** |
| TensorRT FP16 | 0.994 ms | 1.162 ms |

P50 可以理解成一般情況下的延遲，P95 則比較接近「偶爾比較慢時會到多少」。

TensorRT FP32 的 P95 從 ONNX Runtime FP32 的約 `1.110 ms` 降到 `0.761 ms`，改善大約 **31%**。

所以至少在這台 GPU、這個模型、這種一次只做一個決策的情況下：

> **TensorRT FP32 的確帶來了實際速度收益。**

這次不是「理論上 GPU 最佳化應該比較快」，而是真的量到了。

---

## 變快之後，先確認模型沒有明顯跑歪

速度快還不夠。

Breakout Agent 每次會替四個動作算出分數，也就是 Q-value，再挑分數最高的動作。

所以我先拿 60 個完全相同的遊戲狀態，分別交給不同版本計算：

| 執行方式 | 最大 Q-value 差異 | 平均 Q-value 差異 | 動作一致率 |
| --- | ---: | ---: | ---: |
| ONNX Runtime FP32 | 0.000162 | 0.0000369 | **100%** |
| TensorRT FP32 | 0.000162 | 0.0000371 | **100%** |
| TensorRT FP16 | 0.003691 | 0.001378 | **93.33%** |

TensorRT FP32 的結果很好：60 個狀態全部選到和原本 PyTorch 相同的動作，而且數值差異非常小。

這代表至少在這批固定輸入上，沒有看到明顯的轉換錯誤或決策偏移。

FP16 就不同了。60 個狀態裡有 4 個改變了動作，所以它仍然需要更小心看待。

下面這張圖就是這次真正的實測結果。

[![Day 26 TensorRT 與 PyTorch、ONNX Runtime 的實測比較](https://github.com/Tommyweige/breakout-rl-engineering/blob/69f0151b3bc95c860a2d7681a3c1e0c936fc5d5c/assets/day26/tensorrt-comparison.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/69f0151b3bc95c860a2d7681a3c1e0c936fc5d5c/assets/day26/tensorrt-comparison.png)

---

## 但「固定畫面一樣」還不是最後答案

60 個固定狀態很適合抓錯誤，卻不能代表整個遊戲。

因為強化學習有一個很重要的特性：**Agent 的動作會改變它下一秒看到的世界。**

假設某一步兩個版本的 Q-value 非常接近，最後剛好選了不同動作：

```text
這一步動作不同
↓
下一張遊戲畫面不同
↓
接下來看到的狀態也不同
↓
之後兩條遊戲路線越走越遠
```

因此，完整遊戲裡的 action trace 沒有 100% 一樣，並不自動代表其中一個版本「壞掉」。

**部署真正該問的，不是它能不能逐步複製 PyTorch，而是它打很多局之後，整體能力有沒有明顯退步。**

---

## 30 局 score distribution 才是正式答案

因此我先在看結果以前固定了 30 個 episode seeds（決定每一局初始隨機狀態的固定數字）。這 30 局保留原本三組測試起點，再把每組從 5 局擴充到 10 局；四個 runtime（實際執行模型的軟體後端）共用完全相同的 seeds。

這次不再把完整 action trace 當成 adoption gate（決定是否採用的必要門檻），而是比較每局 raw Atari score 的 score distribution（把每局分數放在一起觀察的分布）：平均數、中位數、std（標準差，用來描述分數波動）、P10/P90（排序後較差與較好一端的百分位），以及同一個 seed 下兩個 runtime 的分數差。

| 執行方式 | 局數 | 平均 | 中位數 | std | P10 | P90 | 最低 | 最高 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 30 | 42.53 | 37 | 16.12 | 26.0 | 68.2 | 18 | 74 |
| ONNX Runtime FP32 | 30 | 42.03 | 37 | 15.58 | 26.0 | 61.0 | 18 | 74 |
| **TensorRT FP32** | **30** | **42.03** | **37** | **15.58** | **26.0** | **61.0** | **18** | **74** |
| TensorRT FP16 | 30 | 36.80 | 34 | 13.19 | 18.8 | 54.2 | 16 | 64 |

P10 可以想成較差的那一段分數，P90 則是較好的那一段。TensorRT FP32 和 ORT FP32 在 30 局的每一個 score 都相同：平均、中位數、P10、最低分與最高分都沒有差異，也沒有看到 TensorRT FP32 多出一個低分尾端。

同 seed paired score difference（同一個起點下，左邊分數減去右邊分數）的定義是「左邊 runtime 分數 − 右邊 runtime 分數」。bootstrap 95% CI（用固定規則重抽樣，估計平均差可能落在哪個範圍）也一併保存：

| paired comparison | 平均差 | 中位數 | std | P10 | P90 | 最低 | 最高 | mean 的 bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ORT FP32 − PyTorch FP32 | -0.50 | 0 | 4.61 | -0.6 | 0 | -17 | 16 | [-2.17, 1.10] |
| TensorRT FP32 − PyTorch FP32 | -0.50 | 0 | 4.61 | -0.6 | 0 | -17 | 16 | [-2.17, 1.13] |
| **TensorRT FP32 − ORT FP32** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **[0, 0]** |
| TensorRT FP16 − PyTorch FP32 | -5.73 | -3 | 18.94 | -30.7 | 14.3 | -48 | 25 | [-12.83, 1.00] |

下圖左邊是四個 runtime 的真實 score distribution，右邊是同一個 seed 配對後的分數差。trajectory（一局遊戲中一路累積的狀態與動作序列）可以不同，但圖把它和「整體得分能力是否退化」分開呈現：

[![Day 26 30 局 runtime score distribution 與 paired score difference](https://github.com/Tommyweige/breakout-rl-engineering/blob/69f0151b3bc95c860a2d7681a3c1e0c936fc5d5c/assets/day26/runtime-score-distribution.png?raw=1)](https://github.com/Tommyweige/breakout-rl-engineering/blob/69f0151b3bc95c860a2d7681a3c1e0c936fc5d5c/assets/day26/runtime-score-distribution.png)

TensorRT FP32 相對 ORT FP32 的 paired difference 是 30 個零；相對 PyTorch FP32 的平均差是 -0.50，但 ORT FP32 也是 -0.50。這表示這輪觀察到的差異來自既有 runtime 間的行為差別，不是 TensorRT FP32 額外造成的退化。

---

## FP16 目前還是比較不樂觀

FP16 engine 的檔案大小確實只有 FP32 大約一半：

```text
TensorRT FP32：13.39 MB
TensorRT FP16： 6.73 MB
```

但目前看到的證據並不漂亮：

- 60 個固定狀態有 4 個改變動作，動作一致率是 93.33%；
- 30 局平均分數 36.80，低於 FP32 baseline 的 42.03～42.53；
- batch=1 P95 是 `1.162 ms`，反而比 TensorRT FP32 的 `0.761 ms` 慢。

因此如果要把時間花在進一步的多局驗證上，**TensorRT FP32 明顯比 FP16 更值得優先。**

---

## Day 26 現在的答案

目前已經可以確定三件事。

第一：

> **TensorRT FP32 在這台 RTX 4060 上真的比較快。**

這次 batch=1 P95 約改善 **31%**。

第二：

固定狀態測試適合確認模型有沒有明顯轉壞；真正要判斷一個 RL runtime 值不值得部署，應該讓它實際打很多局，再看整體得分能力是否維持。

第三：

> **TensorRT FP32 在 30 個固定 seeds 上與 ORT FP32 的 score distribution 完全一致，而且 batch=1 P95 快約 31%。**

所以 Day 26 的結果可以整理成：

```text
速度        → batch=1 P95 改善約 31%
固定狀態    → 60 / 60 動作一致
30 局 score → TensorRT FP32 與 ORT FP32 完全一致
正式決定    → TensorRT FP32 = native deployment-worthy
```

因此這次正式結論是：**TensorRT FP32 = native deployment-worthy**。這個結論只適用於目前的模型、GPU、runtime 與 workload；FP16 仍因固定狀態動作一致率較低、多局分數也較差而不列入採用。
