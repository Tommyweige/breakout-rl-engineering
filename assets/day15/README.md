# Day 15 evaluation evidence

The quantitative figure `random-vs-dqn-returns.png` answers one question:
under the same fixed environment, reset seeds, raw reward accumulation, and
episode harness, how do the Random and frozen Day 14 DQN episode returns
compare? The individual points are read from the two evaluation JSON files;
the `x` markers identify episodes stopped by the evaluator safety cap.

The machine-readable source artifacts are:

- `evaluations/day15-random-baseline/results.json` and `episodes.csv`;
- `evaluations/day15-dqn-cuda/results.json` and `episodes.csv`;
- `random-vs-dqn-returns.json`, which records source hashes, protocol, and the
  plotting command;
- `evaluation-contract.mmd` / `evaluation-contract.png`;
- `evaluation-episode-loop.mmd` / `evaluation-episode-loop.png`.

The formal DQN result used the final Day 14 checkpoint at
`100000` environment steps with requested `cuda` and resolved `cuda:0`.
The checkpoint is intentionally not tracked because local training runs are
ignored; its relative path and SHA-256 are recorded in the DQN result and
report.

Recreate the evaluation artifacts from a local copy of that checkpoint:

```powershell
python evaluate_dqn.py --policy random --config configs/eval/breakout_eval.json
python evaluate_dqn.py --checkpoint assets/day14/final-runs/day14-final-frozen-100k/day14-final-vanilla-dqn-seed42/checkpoints/step-00100000.pt --config configs/eval/breakout_eval.json --device cuda
```

Recreate the plot and report:

```powershell
python visualize_day15_evaluation.py evaluations/day15-random-baseline/results.json evaluations/day15-dqn-cuda/results.json --output assets/day15/random-vs-dqn-returns.png --metadata-output assets/day15/random-vs-dqn-returns.json
python generate_dqn_milestone_report.py --random-results evaluations/day15-random-baseline/results.json --dqn-results evaluations/day15-dqn-cuda/results.json --output reports/day15-dqn-milestone.md
```

The Mermaid figures were rendered with:

```powershell
python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day15/evaluation-contract.mmd assets/day15/evaluation-contract.png
python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day15/evaluation-episode-loop.mmd assets/day15/evaluation-episode-loop.png
```
