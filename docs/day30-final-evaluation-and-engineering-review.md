# Day 30｜30 天後，我終於把 Breakout 做完了

30天 ！！我終於完成了！

30 天前，我只是看到有人用強化學習讓 AI 玩遊戲，覺得很有趣，所以也想自己做一次。

一開始想得很簡單：把 Breakout 跑起來、寫 DQN、訓練模型，最後讓 AI 會玩就好。

結果真正做下去之後才發現，事情完全不是這麼簡單。

## 一開始只是想把 DQN 搞懂

前幾天都在理解 Atari、ALE、Gymnasium、reward，還有 DQN 到底怎麼選動作。

後來一路做到 Double DQN、Dueling DQN，模型真的開始會玩之後，我本來以為已經差不多完成了。

但這時才慢慢發現：

> 模型會玩，只是整個專案的其中一部分。

模型換個 seed 還穩不穩？匯出成 ONNX 之後是不是還做一樣的決定？搬到 Browser 後，環境、畫面處理和 action 有沒有真的對齊？

這些問題後來花的時間，甚至比單純把模型訓練出來還多。

## 最後做出了什麼

最後的版本很簡單：左邊是玩家，右邊是 AI，兩邊都真的在跑 Breakout。

玩家可以用鍵盤或滑鼠操作，AI 則用訓練好的模型自己玩。

**[https://breakout.tommypan.dev](https://breakout.tommypan.dev)**

[![Breakout Human vs AI](https://github.com/Tommyweige/breakout-rl-engineering/blob/06e6e2d/assets/human-interactive-runtime/production-human-runtime.png?raw=1)](https://breakout.tommypan.dev)

## 30 天後最大的收穫

如果要說這 30 天最大的收穫，我覺得不是學會某一個模型，也不是學會某一個部署工具。

而是開始習慣一件事：

> 不要停在「它可以跑」，而是繼續問「它真的做對了嗎？」

一開始我只是想學怎麼讓 AI 打磚塊。

最後學到的，反而是怎麼把一個實驗裡會跑的模型，一路做成一個真的能打開、能操作、能驗證的作品。

這大概就是這 30 天最值得留下來的東西。
