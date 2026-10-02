# Issue #39 RGB Brick Removal Observability

Classification: **REJECTED**

Event F1: 0.1440 (TP 9, FP 8, FN 99)

Label events: 108 across 3 episodes; labels consistent: True.
Frames: 15000 formal + at most 42 prior smoke/test frames (<= 308 reserved frames remain) = at most 15350 native frames.
Wall time: 83.78s total (20.00s setup/validation/smoke + 63.78s formal runner and finalization). Formal budget: 560.00s; finalization reserve: 20.00s; seeds completed: [707, 808, 909].

Command: `python -m scripts.evaluation.run_issue39_rgb_brick_observability --config configs/eval/issue39_rgb_brick_observability_v1.json --output-dir outputs/issue-39-rgb-brick-observability --pre-run-wall-seconds 20.00 --wall-budget-seconds 560.00`.
Source revision: `219e8a448a64ec2381ffce4746a0fa10d109a6cd`; branch: `codex/issue-39-rgb-brick-observability`.
Config SHA-256: `5e1ab1c8a47027f04cdedfa5a6df56128d169bd5e9d6f10e095901b8393db25b`; Contract v3 SHA-256: `d97b7fcb5758cba495857b3972a119ed3eb7db74fb0145d151e9d82725c5a786`; controller config SHA-256: `25ddff6756747a6539ddd6a1eb570e7ef4c274f3e700ae011e68c87cae37c724`.
Lossless captured RGB crops: `rgb_brick_crops.npz` (NPZ compressed, uint8; SHA-256 `52b2162544cc8b7b71510b4a001b9fa680c6c9c79646eebe8c5c2bbf3b56bb41`). Detector input is these crops only. The existing canonical completion evaluator reads score/RAM only for the frozen clear-stop rule; evaluator label rows are joined after all action selection. Neither value enters the controller, detector, or action choice.

No clear probability or controller-benefit claim is made.
