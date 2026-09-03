# Day 20 evidence

Day 20 compares DQN, Double DQN, and Dueling Double DQN under the same
Contract v2 task definition and Day 16 canonical CUDA backend. The comparison
manifest records every family/seed/stage entry, reuse audit, runtime metadata,
fixed evaluation, and Q-value probe.

## Reproduction

Run the staged comparison from the repository root. The runner executes one
CUDA job at a time and resumes completed checkpoints when `--resume` is set:

```powershell
conda run --name breakout-rl-engineering python -m scripts.training.run_dqn_family_comparison --stage pilot
conda run --name breakout-rl-engineering python -m scripts.training.run_dqn_family_comparison --stage main --resume
```

The runner first audits `assets/day18/evidence-manifest.json`. Only a fully
compatible Day 18 audit can populate the DQN and Double DQN entries; otherwise
the manifest records the mismatch and requires fresh evidence rather than
mixing protocols. If the 500K aggregate rule finds the top two families
unresolved, run the optional extension:

```powershell
conda run --name breakout-rl-engineering python -m scripts.training.run_dqn_family_comparison --stage extension --resume
```

Generate the report and source-backed figures after the selected runs finish:

```powershell
conda run --name breakout-rl-engineering python -m scripts.analysis.generate_day20_comparison_report --require-formal
conda run --name breakout-rl-engineering python -m scripts.visualization.visualize_dqn_family_comparison
python C:\Users\tommy\.codex\skills\technical-blog-writer\scripts\render_mermaid.py assets/day20/family-comparison-flow.mmd assets/day20/family-comparison-flow.png
```

The formal decision uses complete three-seed fixed evaluation at 500K actual
environment transitions. A completed 1M extension replaces that decision only
when all selected top-two family/seed entries also have complete CUDA
evaluation and Q-probe evidence. Full checkpoints remain in the ignored
`runs/` tree; compact metrics, evaluation artifacts, probes, manifests, and
figures are committed here for review.
