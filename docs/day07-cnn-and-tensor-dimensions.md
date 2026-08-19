# Day 7｜CNN 到底怎麼把 `(4, 84, 84)` 變成 features？

Day 6 已經回答了：Breakout 的 state 太複雜，不能為每一種畫面各放一格 Q-table，因此需要讓 neural network 根據畫面估計 Q-values。

但「用 neural network」還不是一個可以執行的 input contract。Day 4 的環境會回傳：

~~~text
(4, 84, 84) uint8
~~~

這四個維度要怎麼送進 PyTorch？卷積層之後，畫面又會變成什麼？今天只處理這段：

~~~text
(4, 84, 84) uint8
      ↓
(1, 4, 84, 84) float32 / 255
      ↓
Conv1 → Conv2 → Conv3
      ↓
Flatten
      ↓
feature vector
~~~

最後的四個 action Q-values 留到 Day 8。

## 先讓真實程式回答問題

先在本專案的 Conda 環境中執行 dimension inspection：

~~~powershell
python .\inspect_cnn_dimensions.py --device cpu --seed 42
~~~

這個程式不是拿一個手動建立的黑色陣列當例子，而是建立專案真正的 `ALE/Breakout-v5` 預處理環境，取得一次 seeded observation，再把它送過 `AtariFeatureExtractor`。這次實際輸出的 shape 是：

~~~text
Device                  : cpu
Environment observation : (4, 84, 84) uint8, range=0..148
Model input             : (1, 4, 84, 84) torch.float32, range=0.0000..0.5804
Conv1                   : (1, 32, 20, 20)
Conv2                   : (1, 64, 9, 9)
Conv3                   : (1, 64, 7, 7)
Flatten                 : (1, 3136)
Feature dimension       : 3136
~~~

接下來每個概念，都用這份結果解釋，而不是先背一串 CNN 名詞。

## `(4, 84, 84)` 的 4 不是 batch size

先看最容易混淆的地方。Day 4 的 observation shape 是：

~~~text
(4, 84, 84)
~~~

這裡的三個數字分別是：

- `4`：連續四張灰階畫面，也就是四個 stacked frames；
- 第一個 `84`：畫面高度；
- 第二個 `84`：畫面寬度。

四張畫面疊在一起，是為了讓 Agent 不只看到球現在在哪裡，也有機會從連續畫面推測球的移動方向。這個 `4` 在 PyTorch 的 `Conv2d` 裡扮演的是 **channel**，不是四筆資料。

PyTorch 的影像卷積介面使用 `NCHW` 順序：

~~~text
N = batch size
C = channels
H = height
W = width
~~~

因此，一個 state 要補上一個 batch dimension，真正送進模型的是：

~~~text
1 × 4 × 84 × 84
~~~

如果一次處理 32 個 state，才會是：

~~~text
32 × 4 × 84 × 84
~~~

這也是為什麼 `(84, 84, 4)` 不能直接當成這個模型的 input：那是把 channel 放到最後，順序不符合 `Conv2d` 的 NCHW contract。

## 為什麼要在模型前才做 `/255`？

環境回傳的 `uint8` 是 8-bit 整數，每個 pixel 的範圍是 `0..255`。它很適合放在 replay buffer：同樣的畫面，用 1 byte 儲存一個 pixel，比永久保存成 `float32` 更省空間。

但 neural network 的計算通常使用 `float32`。所以本專案把轉換集中在唯一的入口 [`observation_to_tensor`](../breakout_rl/tensors.py)：

~~~python
contiguous_observation = np.ascontiguousarray(observation)
tensor = torch.from_numpy(contiguous_observation).to(
    device=device,
    dtype=torch.float32,
)
tensor = tensor.div(255.0)

if observation.ndim == 3 and add_batch_dim:
    tensor = tensor.unsqueeze(0)
~~~

這段程式做了三件事：

1. 確認輸入是 `(4, 84, 84)` 或 `(B, 4, 84, 84)` 的 `uint8` observation；
2. 轉成模型計算使用的 `float32`；
3. 除以 `255`，把 pixel 映射到 `0..1`，單一 state 再補上 batch dimension。

這個選擇保留了兩個清楚的邊界：環境與未來的 replay buffer 保存 `uint8`；模型的 input 則是已經正規化的 `float32`。模型的 `forward()` 不會偷偷猜測輸入 dtype 或偷偷再做一次 normalization。

## CNN 為什麼適合這種畫面？

如果把 84 × 84 的每個 pixel 都當成互不相關的數字，模型需要自己學會「哪些相鄰 pixel 組合可能是球、球拍或磚塊」。這會浪費很多參數，也不容易利用畫面上重複出現的局部結構。

Convolutional Neural Network（CNN）處理的方式是：用一個小的 **kernel** 在畫面上滑動，每次只看一小塊鄰近 pixel，再把相同的計算套用到不同位置。這讓模型可以先找到局部亮暗或邊緣，再把較小的局部線索組合成更大的空間特徵。

Day 7 的 feature extractor 使用經典 Atari DQN 的三層卷積 trunk：

~~~text
Conv2d(4,  32, kernel_size=8, stride=4)
ReLU
Conv2d(32, 64, kernel_size=4, stride=2)
ReLU
Conv2d(64, 64, kernel_size=3, stride=1)
ReLU
Flatten
~~~

第一層的 `in_channels=4`，不是 RGB 常見的 3，因為這裡的 input 是四張灰階 frame。每一個 output channel 可以理解成一個不同的 filter 結果；channel 從 4 增加到 32，再增加到 64，不是增加 batch，而是讓模型保留更多種類的局部特徵。

`ReLU` 是一個簡單的非線性函數：負值變成 0，正值保留。它讓多層卷積不只是把 input 做一連串線性變換。

## 每一層的高度與寬度怎麼來？

卷積層的空間輸出尺寸可以用這個公式計算：

~~~text
out = floor((in + 2p - d × (k - 1) - 1) / stride + 1)
~~~

其中 `k` 是 kernel size，`stride` 是每次滑動幾格，`p` 是 padding，`d` 是 dilation。這個 baseline 沒有 padding，也沒有 dilation，所以三層可以直接算成：

~~~text
Conv1: floor((84 - 8) / 4 + 1) = 20
Conv2: floor((20 - 4) / 2 + 1) = 9
Conv3: floor((9 - 3) / 1 + 1) = 7
~~~

因此，包含 batch dimension 的實際輸出是：

~~~text
Model input : (1, 4, 84, 84)
Conv1       : (1, 32, 20, 20)
Conv2       : (1, 64, 9, 9)
Conv3       : (1, 64, 7, 7)
~~~

空間尺寸從 `84 × 84` 逐步縮小，但 channel 數增加。這不是把資料「壓成一張更小的圖片」而已；每一層都同時改變了空間解析度與 feature map 的種類。

## 為什麼 Flatten 後是 3,136？

Conv3 的每一筆 sample 有：

~~~text
64 × 7 × 7 = 3,136
~~~

`Flatten` 只是把這三個非 batch 維度攤平成一列，因此：

~~~text
(1, 64, 7, 7) → (1, 3,136)
~~~

這個 3,136 是 feature vector 的長度，不是四個 action 的 Q-values。Day 7 刻意沒有接 `Linear(3136, 4)`；下一天才會把 feature vector 交給 fully connected head，產生每個 action 的 Q-value。

程式也沒有把 `3136` 散落在 model 裡當成 magic number。`AtariFeatureExtractor` 初始化時會用 dummy tensor 實際跑過 convolutional trunk，從 runtime feature map 推導 `feature_dim`。這讓程式的公開 property 和真的 flatten shape 有同一個來源。

## Figure：從真實 state 到 feature vector

前面的文字可以算出 shape，但還有一個工程問題：我們是否真的用專案環境和模型跑過這條路？以下圖片回答的技術問題是：**一個真實 `(4, 84, 84)` state 經過 batch 維度、三層卷積後，實際 shape 與元素數量如何變化？**

圖片由 [`visualize_cnn_dimensions.py`](../visualize_cnn_dimensions.py) 產生：

~~~powershell
python .\visualize_cnn_dimensions.py --device cpu --seed 42
~~~

![由真實 Breakout observation 與 CNN forward 產生的 tensor dimension evidence figure](../artifacts/day07/cnn-dimensions.png)

圖的左上角是同一次 seeded environment reset 得到的真實第一張灰階 frame；它不是手動填入的示意圖。右上角是同一個未訓練 `AtariFeatureExtractor` 的 Conv3 第 0 個 channel，shape 是 `(1, 64, 7, 7)`。這個 heatmap 證明模型的 forward 確實產生了中間 feature map，但因為模型還沒有訓練，不能把它解讀成「模型已經認出球或球拍」。

下方的 bars 把 runtime shape 和元素數量放在一起：

- `Environment state` 是 `(4, 84, 84)`；
- `Model input` 是補上 batch 後的 `(1, 4, 84, 84)`，元素數量沒有增加，只是多了一個維度；
- Conv1、Conv2、Conv3 依序把空間尺寸降到 `20 × 20`、`9 × 9`、`7 × 7`，同時把 channel 提升到 32、64、64；
- Conv3 與 Flatten 的元素數量相同，因為 Flatten 改變的是排列方式，不是數量。

這張圖能支持的是「input contract 和 shape transformation 已經由真實程式驗證」。它不能支持「CNN 已經學會 Breakout」，因為這一天沒有 training、loss 或 checkpoint。

PNG 旁邊也保存了同一次執行的 machine-readable metadata：[`cnn-dimensions.json`](../artifacts/day07/cnn-dimensions.json)。其中包含 seed、device、觀察值範圍和每一層實際 shape，方便日後追查圖片是不是由不同設定產生。

## CPU 與 CUDA 的邊界

模型和 input tensor 必須在同一個 device。這次的環境實際驗證結果是：

~~~text
torch 2.13.0+cpu
cuda_available False
~~~

所以 CPU inspection 是正常路徑。CLI 也提供三種選擇：

~~~powershell
python .\inspect_cnn_dimensions.py --device auto --seed 42
python .\inspect_cnn_dimensions.py --device cpu --seed 42
python .\inspect_cnn_dimensions.py --device cuda --seed 42
~~~

`auto` 會在 CUDA 可用時選 CUDA，否則選 CPU；明確指定 `cuda` 但當前環境沒有 CUDA 時，程式會報清楚的錯誤，而不是偷偷把測試改跑 CPU。這個差異很重要：否則你以為自己驗證了 GPU path，實際上可能只是驗證了 CPU。

## 測試固定了哪些 contract？

Day 7 新增的測試在兩個主要 public seams 上檢查行為：

- `tests/test_tensors.py`：驗證單一 observation 的 batch 維度、batch input 不重複加維度、`uint8 → float32 / 255`、shape/dtype 檢查，以及不可用 CUDA 不會被靜默忽略；
- `tests/test_atari_cnn.py`：驗證 batch size 1 和 2、runtime flatten shape、三層 baseline convolution shape，以及 backward 能到達 feature extractor parameters；
- `tests/test_inspect_cnn_dimensions.py`：再用真實 Breakout environment 驗證 `(4, 84, 84) uint8` 能一路到 `(1, 3136)` features。

這些測試不檢查某個 private layer 的寫法，而是檢查後續 Day 8 network 可以依賴的 input/output contract。將來如果內部改用另一種等價的 module 組合，只要 contract 不變，測試就不需要跟著重寫。

## 今天刻意沒有做什麼？

Day 7 的終點是：

~~~text
real Breakout observation
      ↓
uint8 (4, 84, 84)
      ↓
float32 (1, 4, 84, 84)
      ↓
Atari CNN
      ↓
feature vector (1, 3136)
~~~

這一天沒有加入四-action Q-value head、replay buffer、optimizer、TD loss、target network 或 Breakout training。這些東西都需要先知道「CNN 已經輸出什麼」，而不是和 input shape 一起混在同一個問題裡。

下一個自然問題是：**3136 個 features 要怎麼接成 `Q(NOOP), Q(FIRE), Q(RIGHT), Q(LEFT)` 四個輸出？** 這就是 Day 8 的 DQN network。
