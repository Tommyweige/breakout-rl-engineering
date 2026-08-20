# Day 7｜`(4, 84, 84)` 進入 CNN 之後，究竟變成了什麼？

Day 6 留下一個很具體的問題：Breakout 的畫面太複雜，不能像 toy environment 一樣，為每一個 state 建立一格 Q-table。既然要讓 neural network 從畫面估計價值，第一步不是立刻開始訓練，而是先確認一件事：

> **四張 `84 × 84` 的灰階畫面送進 PyTorch 後，CNN 如何把它變成後續可以使用的 features？**

今天只追蹤這條資料流：

~~~text
(4, 84, 84) uint8
      ↓
(1, 4, 84, 84) float32，pixel / 255
      ↓
Conv1 → Conv2 → Conv3
      ↓
Flatten
      ↓
feature vector
~~~

最後接上四個 action Q-values，留到 Day 8。

## 先看一次真的 forward

如果只看 `Conv2d` 的 API，很容易把 batch、channel 和畫面尺寸混在一起。先讓專案真正建立 `ALE/Breakout-v5` 預處理環境，取得一個 seeded observation，再執行 CNN inspection：

~~~powershell
python .\inspect_cnn_dimensions.py --device cpu --seed 42
~~~

這次執行的關鍵輸出是：

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

這不是手算後印出的假結果；每一層的 shape 都來自同一次 model forward。接下來要解釋的，是這些數字為什麼會這樣排列。

## `(4, 84, 84)` 裡的 4 到底代表什麼？

Day 4 的 observation 不是單張 RGB 圖片，而是四張連續的灰階畫面。把最近幾張畫面疊在一起，Agent 才有機會從畫面變化推測球的移動方向；這種把連續畫面放在同一個 state 裡的做法叫做 frame stacking。

因此：

~~~text
(4, 84, 84)
 ^   ^   ^
 |   |   └─ width：畫面寬度
 |   └──── height：畫面高度
 └──────── channels：四張 stacked frames
~~~

這個 `4` 是 channel，不是 batch size。PyTorch 的 `Conv2d` 會把影像輸入視為 `NCHW`：

~~~text
N = batch size：一次處理幾個 state
C = channels：一個 state 有幾個輸入平面
H = height
W = width
~~~

環境一次只給一個 state，所以必須在最前面補一個 `N=1`：

~~~text
environment state : (4, 84, 84)
model input       : (1, 4, 84, 84)
~~~

如果一次處理 32 個 state，才會是 `(32, 4, 84, 84)`。這也說明了為什麼 `(84, 84, 4)` 不能直接交給這個 CNN：channel 放在最後，順序不是 PyTorch 的 NCHW contract。

## 為什麼 pixel 要在模型前才除以 255？

環境回傳的資料型別是 `uint8`。它用 8 個 bits 儲存一個 pixel，範圍是 `0..255`，很適合留在未來的 replay buffer 裡。若 replay buffer 永久保存 `float32`，同一個 pixel 會需要更多記憶體。

另一方面，CNN 的數值運算使用 `float32`，而且通常希望輸入落在較小的 `0..1` 範圍。因此本專案把轉換集中在唯一的 model boundary：

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

這段程式的重點不是語法，而是責任分界：

- storage 端保留 `(4, 84, 84) uint8`；
- 進入 model 前才轉成 `(1, 4, 84, 84) float32`；
- `/255` 只發生在這個入口，不讓 model `forward()` 猜測輸入是否需要 normalization。

同一個入口也支援 batch。已經是 `(B, 4, 84, 84)` 的資料不會再被錯誤地多包一層 batch。

## CNN 在畫面上做了什麼？

如果每一個 pixel 都要和整張畫面的所有 pixel 互相連接，模型會很快變得龐大，也很難利用「相鄰 pixel 通常比較有關係」這個特性。CNN 的解法是用一個小區域反覆掃過畫面：這個小區域叫 kernel，每次移動的距離叫 stride。

同一個 kernel 會在不同位置重複使用，所以模型可以在畫面各處尋找相似的局部結構。第一層可能先保留邊緣或亮暗變化，後面的層再把較小的線索組合成較大的 feature。

Day 7 實際使用的 convolutional trunk 是：

~~~python
self.features = nn.Sequential(
    nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU(),
)
~~~

第一層的 `input_channels` 是 4，因為輸入是四張 stacked frames，不是 RGB 的三個 channel。`32`、`64`、`64` 是架構設計選擇：它們決定每層要保留多少種 feature map，不是由公式唯一推導出來的答案。

`ReLU` 則提供非線性：它把負值變成 0，正值保留。沒有這類非線性，多層卷積疊起來仍然可以被看成一個較大的線性變換，表達能力會受限。

## 20、9、7 是怎麼算出來的？

現在已經知道每層使用的 kernel 和 stride，才適合看輸出尺寸公式。對高度或寬度而言：

~~~text
out = floor((in + 2p - d × (k - 1) - 1) / stride + 1)
~~~

這裡的 `in` 是輸入尺寸，`k` 是 kernel size，`p` 是 padding，`d` 是 dilation，也就是 kernel 取樣間隔。這個 baseline 沒有 padding，dilation 是 1，所以實際三層可以這樣算：

~~~text
Conv1: floor((84 - 8) / 4 + 1) = 20
Conv2: floor((20 - 4) / 2 + 1) = 9
Conv3: floor((9 - 3) / 1 + 1) = 7
~~~

所以 runtime 看到的 tensor shape 會是：

~~~text
Model input : (1, 4, 84, 84)
Conv1       : (1, 32, 20, 20)
Conv2       : (1, 64, 9, 9)
Conv3       : (1, 64, 7, 7)
~~~

這裡有兩個同時發生的變化：空間尺寸變小，channel 數增加。空間尺寸變小是 stride 和 kernel 的結果；channel 從 4 增加到 32、再到 64，則是網路架構希望保留更多種類 feature 的設計選擇。

## Flatten 後的 3,136 是什麼？

Conv3 每個 sample 的 feature map 有：

~~~text
64 × 7 × 7 = 3,136
~~~

`Flatten` 只把這三個非 batch 維度攤平成一列，所以：

~~~text
(1, 64, 7, 7) → (1, 3,136)
~~~

這個 3,136 是 feature vector 的長度，不是四個 action 的 Q-values。它代表 CNN 已經把空間排列轉成一串數值，下一個 fully connected layer 可以再使用這串數值。

程式沒有直接把 `3136` 寫成一個必須永遠成立的 magic number。`AtariFeatureExtractor` 初始化時會用 dummy tensor 跑過 convolutional trunk，從實際 feature map 推導 `feature_dim`。因此 `feature_dim` 和真正 forward 後的 flatten shape 有同一個來源；如果未來改變 input shape 或 convolution 結構，錯誤會比較早被看見。

## 這次的真實圖像看到了什麼？

到這裡，公式已經可以解釋 shape，但仍有一個重要問題：這條資料流是否真的在專案的 Breakout environment 上跑過？下面這張圖要回答的是：

> **一個真實 `(4, 84, 84)` state，經過 batch 維度和三層卷積後，實際的 shape 與元素數量如何變化？**

用同一個命令可以重建圖片：

~~~powershell
python .\visualize_cnn_dimensions.py --device cpu --seed 42
~~~

![由真實 Breakout observation 與 CNN forward 產生的 tensor dimension evidence figure](../assets/day07/cnn-dimensions.png)

左上角是同一次 seeded environment reset 得到的第一張真實灰階 frame；它不是手動畫出的示意圖。右上角是同一個未訓練 `AtariFeatureExtractor` 的 Conv3 第 0 個 channel，shape 是 `(1, 64, 7, 7)`。熱圖能讓讀者看到中間 feature map 確實存在，但這些權重尚未經過訓練，所以不能把它解讀成「模型已經認出球或球拍」。

下方的 shape evidence 把兩件容易混淆的事情放在同一個視野裡：

- `Environment state` 是 `(4, 84, 84)`；
- `Model input` 是 `(1, 4, 84, 84)`，元素數量沒有增加，只是補上 batch 維度；
- Conv1、Conv2、Conv3 把空間尺寸依序變成 `20 × 20`、`9 × 9`、`7 × 7`，同時增加 channel；
- Conv3 和 Flatten 的元素數量相同，因為 Flatten 改變排列，不會創造或刪除數值。

圖旁邊的 [`cnn-dimensions.json`](../assets/day07/cnn-dimensions.json) 保存了同一次執行的 seed、device、observation 範圍與 runtime shapes。這些 evidence 支持的是「model input contract 和 shape transformation 已由真實 forward 驗證」，不是「CNN 已經學會玩 Breakout」。

## CPU、CUDA 與 feature extractor 的邊界

模型參數和 input tensor 必須放在同一個 device。這次開發環境實際使用 `torch 2.13.0+cpu`，`torch.cuda.is_available()` 是 `False`，所以 CPU 是目前可重現的路徑。inspection CLI 仍提供：

~~~powershell
python .\inspect_cnn_dimensions.py --device auto --seed 42
python .\inspect_cnn_dimensions.py --device cpu --seed 42
python .\inspect_cnn_dimensions.py --device cuda --seed 42
~~~

`auto` 會在 CUDA 可用時選 CUDA，否則選 CPU；明確指定 `cuda` 但當前環境不可用時會報錯，而不是靜默換成 CPU。這一天不做效能 benchmark，也不宣稱 GPU 一定比較快。

Day 7 的 `AtariFeatureExtractor` 到 feature vector 就停止。它還沒有 action-value head、replay buffer、optimizer、TD loss 或 target network。下一天真正要回答的問題是：**這串 3,136 個 features 要怎麼接成 `Q(NOOP)、Q(FIRE)、Q(RIGHT)、Q(LEFT)` 四個輸出？**

現在可以把今天的心智模型濃縮成一句話：**四張 stacked frames 是四個 input channels；batch 是另一個維度；CNN 改變空間解析度與 feature channels；Flatten 只把已經算好的 feature map 排成 vector。** 這正是 Day 8 接上 DQN head 的輸入。
