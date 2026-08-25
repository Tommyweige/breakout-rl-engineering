# Day 14 frozen Vanilla DQN config

`day14-vanilla-dqn.json` freezes the Day 14 development candidate after:

- learning-rate comparison selected `2e-4` as the single-seed candidate;
- the batch-size short profiling showed no end-to-end SPS gain for 64 or 128;
- the fixed-interval CPU thread profile selected `2` threads from `1/2/4`;
- the final run remains sequential CUDA, `float32`, and `batch_size=32`.

Checkpoint selection rule for the next day: keep the 25K/50K/75K/100K
checkpoints for diagnosis, but use the latest checkpoint only when the run is
complete, all finite-value guards pass, and the comparison report records the
expected step budget and device metadata. Do not select a checkpoint from a
single unusually high episode return.
