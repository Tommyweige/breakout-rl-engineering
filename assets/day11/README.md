# Day 11 assets

The evidence plot is generated from real `DQNNetwork` outputs and one real
optimizer update on a deterministic synthetic batch. The JSON file beside the
PNG preserves the values, seed, device, and command used for the run.

From the repository root:

```text
conda run --name breakout-rl-engineering python visualize_target_network_sync.py --device cpu --seed 42
```

The structural flow diagram is rendered from `target-network-flow.mmd` with
the Mermaid renderer supplied by the `technical-blog-writer` skill. Its source
and rendered PNG intentionally share the same basename.
