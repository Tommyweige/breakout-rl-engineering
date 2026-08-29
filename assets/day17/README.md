# Day 17 evidence

These artifacts are generated from the Day 17 code and are kept as the
machine-readable sources for the article figures.

## Reproduction

```powershell
python -m scripts.demos.overestimation_demo --actions 4 --trials 100000 --noise-stds 0.1 0.5 1.0 --seed 42 --output assets/day17/overestimation-bias.json --plot-output assets/day17/overestimation-bias.png
python -m scripts.analysis.inspect_double_dqn_targets --seed 42 --output assets/day17/dqn-vs-double-targets.json
python -m scripts.visualization.visualize_double_dqn_targets --input assets/day17/dqn-vs-double-targets.json --output assets/day17/dqn-vs-double-targets.png
python -m scripts.analysis.generate_probe_states --contract configs/eval/breakout_contract_v2.json --output assets/day17/probe_states.npz --states-per-seed 4 --stride 32 --max-steps 256
python -m scripts.training.train_vectorized_dqn --config configs/double_dqn_baseline.json --run-id day17-double-dqn-smoke-seed42-final3 --output assets/day17/double-dqn-smoke-summary.json
python -m scripts.analysis.analyze_q_values --checkpoint runs/day17-double-dqn-smoke-seed42-final3/checkpoints/step-00010000.pt --probe-states assets/day17/probe_states.npz --device cuda --output assets/day17/q-probe-summary.json
python -m scripts.visualization.visualize_q_values --input assets/day17/q-probe-summary.json --output assets/day17/q-probe-summary.png
python -m scripts.training.train_vectorized_dqn --config configs/double_dqn_baseline.json --algorithm dqn --run-id day17-vanilla-dqn-control-seed42-final3 --output assets/day17/vanilla-dqn-smoke-summary.json
python -m scripts.analysis.summarize_day17_smoke --vanilla-summary assets/day17/vanilla-dqn-smoke-summary.json --double-summary assets/day17/double-dqn-smoke-summary.json --output assets/day17/smoke-performance.json --plot-output assets/day17/smoke-performance.png
python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day17/double-dqn-target-flow.mmd assets/day17/double-dqn-target-flow.png
```

The CUDA smoke runs use the canonical Day 16 handoff: Contract v2, N=2,
strict action-selection parity, GPU Replay, float32, batch size 32, and two
configured CPU threads. The two summary files are paired for performance
overhead only; they are not a model-quality comparison.

`probe_states.npz` is generated from Contract v2 with environment-owned serve
FIRE enabled. Its metadata records the contract hash, concrete seed, episode,
step, shape, and dtype for every observation. Probe states are diagnostics only
and are not inserted into Replay.
