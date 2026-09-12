"""Generate exact Gymnasium AtariPreprocessing NOOP counts for fixed seeds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from gymnasium.wrappers import AtariPreprocessing

from breakout_env import make_breakout_raw_env
from breakout_rl.atari_trace import RawFrameLogger


SEEDS = tuple(range(101, 106)) + tuple(range(202, 207)) + tuple(range(303, 323))


def native_seed_config(seed: int) -> dict[str, int]:
    raw = make_breakout_raw_env(sticky_action_probability=0.25)
    logger = RawFrameLogger(raw)
    env = AtariPreprocessing(
        logger,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    try:
        env.reset(seed=seed)
        logger.finish_reset()
        np_seed, ale_seed = (int(value) for value in np.random.SeedSequence(seed).generate_state(n_words=2))
        return {
            "noop_count": len(logger.reset_frames) - 1,
            "np_seed": np_seed,
            "ale_seed": ale_seed,
        }
    finally:
        env.close()


def main() -> None:
    output = Path("assets/day29/native-noop-reset-manifest.json")
    manifest = {
        "schema_version": 2,
        "artifact_type": "day29_native_noop_reset_manifest",
        "source": "Gymnasium AtariPreprocessing reset / env.unwrapped.np_random.integers(1, noop_max+1), captured through ALE 0.12.0",
        "noop_max": 30,
        "seeds": {str(seed): native_seed_config(seed) for seed in SEEDS},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
