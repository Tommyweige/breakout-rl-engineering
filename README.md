# Breakout RL Engineering｜30 天鐵人賽文章系列

這是 Breakout RL Engineering 的 30 天鐵人賽文章系列，記錄從 Atari Breakout、DQN 到瀏覽器部署的完整學習與工程過程。

## 文章目錄

1. [Day 1｜Project Introduction](articles/day01-project-introduction.md)
2. [Day 2｜當 Agent 走進 Atari Breakout：ALE 與 Gymnasium](articles/day02-breakout-ale-gymnasium.md)
3. [Day 3｜當 Agent 按下一個動作之後，環境到底回傳了什麼？](articles/day03-state-action-reward-data.md)
4. [Day 4｜AI 看到的，不再只是一張畫面](articles/day04-atari-preprocessing-frame-stacking.md)
5. [Day 5｜一步沒有得分，為什麼仍可能是好選擇？](articles/day05-mdp-bellman-equation.md)
6. [Day 6｜Q-value 一開始不知道，Agent 怎麼把它學出來？](articles/day06-q-learning-to-deep-q-learning.md)
7. [Day 7｜`(4, 84, 84)` 進入 CNN 之後，究竟變成了什麼？](articles/day07-cnn-and-tensor-dimensions.md)
8. [Day 8｜從 CNN Features 到四個 Q-values：完成 DQN Network](articles/day08-dqn-network.md)
9. [Day 9｜Experience Replay：把遊戲經驗存起來，再隨機拿回來學](articles/day09-experience-replay.md)
10. [Day 10｜探索與利用：讓 Agent 不只相信目前最大的 Q-value](articles/day10-exploration-vs-exploitation.md)
11. [Day 11｜Target Network：別讓 DQN 的學習目標每一步都跟著自己跑](articles/day11-target-network.md)
12. [Day 12｜完整 DQN Training Loop：一筆遊戲經驗怎麼真的變成學習](articles/day12-complete-dqn-training-loop.md)
13. [Day 13｜DQN 不學時怎麼查：先證明程式沒壞，再談超參數](articles/day13-debugging-unstable-rl-training.md)
14. [Day 14｜先讓訓練值得跑久，再看 100K 的 DQN 到底學到什麼](articles/day14-hyperparameter-experiments.md)
15. [Day 15｜DQN 真的變強了嗎？一次評估反而抓出了 FIRE deadlock](articles/day15-dqn-milestone-and-evaluation.md)
16. [Day 16｜GPU 不是放著就會變快：一次讓多個 Breakout 一起跑](articles/day16-vectorized-dqn-training.md)
17. [Day 17｜DQN 為什麼會把自己看得太樂觀？從 `max` 的陷阱到 Double DQN](articles/day17-q-overestimation-and-double-dqn.md)
18. [Day 18｜DQN vs Double DQN：100K 看得到學習，不代表已經能選模型](articles/day18-dqn-vs-double-dqn.md)
19. [Day 19｜Dueling Network：先判斷「局面好不好」，再看「哪個動作比較好」](articles/day19-dueling-network-architecture.md)
20. [Day 20｜DQN、Double DQN、Dueling Double DQN：到底該選誰繼續訓練？](articles/day20-dqn-family-comparison.md)
21. [Day 21｜訓練越久越好嗎？跑到 5M 後，我最後反而選了 2.5M 模型](articles/day21-final-long-training.md)
22. [Day 22｜模型訓練完了，接下來呢？把 PyTorch 模型帶進 ONNX](articles/day22-pytorch-to-onnx.md)
23. [Day 23｜換成 ONNX Runtime 之後，它還是同一個 Agent 嗎？](articles/day23-onnx-runtime-inference.md)
24. [Day 24｜模型一次決策要多久？我才發現 GPU Benchmark 其實很容易量錯](articles/day24-correct-inference-benchmarking.md)
25. [Day 25｜FP16 真的比較快嗎？模型縮小一半，我最後卻決定不用它](articles/day25-fp32-vs-fp16.md)
26. [Day 26｜TensorRT 真的比較快嗎？RL 部署不能只看「每一步一不一樣」](articles/day26-tensorrt-optimization-experiment.md)
27. [Day 27｜把 Breakout 模型真的搬進瀏覽器](articles/day27-onnx-runtime-web.md)
28. [Day 28｜GPU 一定比 CPU 快嗎？實測 WebGPU 之後，答案是：不一定](articles/day28-webgpu-inference.md)
29. [Day 29｜讓 RL Agent 真的在瀏覽器裡玩 Breakout](articles/day29-interactive-browser-demo.md)
30. [Day 30｜30 天後，我終於把 Breakout 做完了](articles/day30-final-evaluation-and-engineering-review.md)

Live Demo：[https://breakout.tommypan.dev](https://breakout.tommypan.dev)

主專案：[Tommyweige/breakout-rl-engineering](https://github.com/Tommyweige/breakout-rl-engineering)

實際維護中的程式碼位於 [`main`](https://github.com/Tommyweige/breakout-rl-engineering/tree/main)。
