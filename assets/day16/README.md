# Day 16 evidence

The evidence uses real `ALE/Breakout-v5`, NVIDIA CUDA training, and the fixed
`configs/eval/breakout_contract_v2.json` semantics. The finalization artifacts
separate the 10K systems screening from the fresh 100K validation.

## Machine-readable sources

- `vectorized-training.json`: fresh 10K screening for N=1/2/4/8, including
  stage timings, strict-parity metadata, and fixed-interval runtime samples.
- `vectorized-training-100k-n2.json`: selected fresh 100K N=2 validation under
  strict action-selection parity.
- `vectorized-training-100k.json`: supplemental fresh 100K N=1/N=4 validation
  under strict action-selection parity.
- `replay-insertion.json`: real-observation `add_batch` benchmark for batch
  sizes 1/2/4/8/16, with code commit lineage.
- `fire-sticky-diagnostic.json` and `fire-sticky-trace.jsonl`: real-ALE FIRE
  confirmation, life-loss, observation activity, and action provenance data.
- `q-value-diagnostics.json`: exploratory Q-values from a real checkpoint on
  CUDA, including GPU/runtime/checkpoint metadata.
- `overestimation-bias.json` and `overestimation-bias.png`: reproducible CPU
  NumPy toy simulation and its measured plot; it is not a Breakout trajectory.
- `evaluation-summary.json`: 10K screening, 100K validation, Contract v2
  Random baseline, provenance fields, and hashes for the diagnostics.

## Reproduce

```powershell
conda run --name breakout-rl-engineering python benchmark_vectorized_training.py --device cuda --replay-backend gpu --environment-counts 1 2 4 8 --total-steps 10000 --seed 42 --batch-size 32 --learning-rate 0.0001 --replay-capacity 10000 --learning-starts 1000 --train-frequency 4 --target-update-interval 500 --epsilon-decay-steps 10000 --cpu-threads 2 --profile-stages --no-strict-action-selection-parity --samples-root assets/day16/runtime-samples-final-10k-v2 --run-root runs/day16-finalization-10k-v2 --output assets/day16/vectorized-training.json
conda run --name breakout-rl-engineering python benchmark_vectorized_training.py --device cuda --replay-backend gpu --environment-counts 1 4 --total-steps 100000 --seed 42 --batch-size 32 --learning-rate 0.0001 --replay-capacity 10000 --learning-starts 1000 --train-frequency 4 --target-update-interval 500 --epsilon-decay-steps 10000 --cpu-threads 2 --profile-stages --strict-action-selection-parity --samples-root assets/day16/runtime-samples-final-100k-v2 --run-root runs/day16-finalization-100k-v2 --output assets/day16/vectorized-training-100k.json
conda run --name breakout-rl-engineering python benchmark_vectorized_training.py --device cuda --replay-backend gpu --environment-counts 2 --total-steps 100000 --seed 42 --batch-size 32 --learning-rate 0.0001 --replay-capacity 10000 --learning-starts 1000 --train-frequency 4 --target-update-interval 500 --epsilon-decay-steps 10000 --cpu-threads 2 --profile-stages --strict-action-selection-parity --samples-root assets/day16/runtime-samples-final-100k-v2-n2 --run-root runs/day16-finalization-100k-v2-n2 --output assets/day16/vectorized-training-100k-n2.json
conda run --name breakout-rl-engineering python benchmark_replay_insertion.py --device cuda --iterations 500 --seed 42 --batch-sizes 1 2 4 8 16 --output assets/day16/replay-insertion.json
conda run --name breakout-rl-engineering python evaluate_dqn.py --policy random --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cpu --output-dir evaluations/day16-contract-v2-random --evaluation-id day16-contract-v2-random
conda run --name breakout-rl-engineering python evaluate_vectorized_dqn.py --checkpoint runs/day16-finalization-10k-v2/envs-1/checkpoints/step-00010000.pt --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cuda --output-dir evaluations/day16-final-10k-envs1-contract-v2 --evaluation-id day16-final-10k-envs1-contract-v2
conda run --name breakout-rl-engineering python evaluate_vectorized_dqn.py --checkpoint runs/day16-finalization-10k-v2/envs-2/checkpoints/step-00010000.pt --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cuda --output-dir evaluations/day16-final-10k-envs2-contract-v2 --evaluation-id day16-final-10k-envs2-contract-v2
conda run --name breakout-rl-engineering python evaluate_vectorized_dqn.py --checkpoint runs/day16-finalization-10k-v2/envs-4/checkpoints/step-00010000.pt --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cuda --output-dir evaluations/day16-final-10k-envs4-contract-v2 --evaluation-id day16-final-10k-envs4-contract-v2
conda run --name breakout-rl-engineering python evaluate_vectorized_dqn.py --checkpoint runs/day16-finalization-10k-v2/envs-8/checkpoints/step-00010000.pt --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cuda --output-dir evaluations/day16-final-10k-envs8-contract-v2 --evaluation-id day16-final-10k-envs8-contract-v2
conda run --name breakout-rl-engineering python evaluate_vectorized_dqn.py --checkpoint runs/day16-finalization-100k-v2/envs-1/checkpoints/step-00100000.pt --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cuda --output-dir evaluations/day16-final-100k-envs1-contract-v2 --evaluation-id day16-final-100k-envs1-contract-v2
conda run --name breakout-rl-engineering python evaluate_vectorized_dqn.py --checkpoint runs/day16-finalization-100k-v2-n2/envs-2/checkpoints/step-00100000.pt --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cuda --output-dir evaluations/day16-final-100k-envs2-contract-v2 --evaluation-id day16-final-100k-envs2-contract-v2
conda run --name breakout-rl-engineering python evaluate_vectorized_dqn.py --checkpoint runs/day16-finalization-100k-v2/envs-4/checkpoints/step-00100000.pt --config configs/eval/breakout_eval.json --contract configs/eval/breakout_contract_v2.json --device cuda --output-dir evaluations/day16-final-100k-envs4-contract-v2 --evaluation-id day16-final-100k-envs4-contract-v2
conda run --name breakout-rl-engineering python diagnose_fire_sticky.py --checkpoint runs/day16-finalization-100k-v2-n2/envs-2/checkpoints/step-00100000.pt --contract configs/eval/breakout_contract_v2.json --device cuda --output assets/day16/fire-sticky-diagnostic.json --trace-output assets/day16/fire-sticky-trace.jsonl
conda run --name breakout-rl-engineering python analyze_q_values.py --checkpoint runs/day16-finalization-100k-v2-n2/envs-2/checkpoints/step-00100000.pt --contract configs/eval/breakout_contract_v2.json --device cuda --seeds 101 102 103 104 105 --steps-per-seed 16 --output assets/day16/q-value-diagnostics.json
conda run --name breakout-rl-engineering python overestimation_demo.py --seed 42 --trials 500000 --chunk-size 100000 --output assets/day16/overestimation-bias.json
conda run --name breakout-rl-engineering python visualize_overestimation.py assets/day16/overestimation-bias.json --output assets/day16/overestimation-bias.png
conda run --name breakout-rl-engineering python summarize_day16_evaluation.py --reference-results evaluations/day16-final-10k-envs1-contract-v2/results.json --reference-checkpoint runs/day16-finalization-10k-v2/envs-1/checkpoints/step-00010000.pt --candidate-results evaluations/day16-final-10k-envs2-contract-v2/results.json --candidate-checkpoint runs/day16-finalization-10k-v2/envs-2/checkpoints/step-00010000.pt --candidate-environment-count 2 --screening-training-report assets/day16/vectorized-training.json --validation-reference-results evaluations/day16-final-100k-envs1-contract-v2/results.json --validation-reference-checkpoint runs/day16-finalization-100k-v2/envs-1/checkpoints/step-00100000.pt --validation-candidate-results evaluations/day16-final-100k-envs2-contract-v2/results.json --validation-candidate-checkpoint runs/day16-finalization-100k-v2-n2/envs-2/checkpoints/step-00100000.pt --validation-candidate-environment-count 2 --validation-training-report assets/day16/vectorized-training-100k-n2.json --random-results evaluations/day16-contract-v2-random/results.json --fire-diagnostic assets/day16/fire-sticky-diagnostic.json --q-value-diagnostic assets/day16/q-value-diagnostics.json --overestimation-report assets/day16/overestimation-bias.json --output assets/day16/evaluation-summary.json
conda run --name breakout-rl-engineering python visualize_vectorized_training.py assets/day16/vectorized-training.json --insertion-report assets/day16/replay-insertion.json --output-dir assets/day16
conda run --name breakout-rl-engineering python visualize_vectorized_training.py assets/day16/vectorized-training-100k.json --output-dir assets/day16 --file-prefix vectorized-100k
conda run --name breakout-rl-engineering python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day16/vectorized-pipeline.mmd assets/day16/vectorized-pipeline.png
```

The checkpoint files and run directories are intentionally ignored local
training outputs. The JSON artifacts record their paths, runtime metadata, and
SHA-256 values where a checkpoint or result is referenced.
