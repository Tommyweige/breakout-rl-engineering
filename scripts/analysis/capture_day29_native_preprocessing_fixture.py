"""Capture native Gymnasium preprocessing frames for Browser parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from breakout_env import BreakoutFireResetWrapper, make_breakout_raw_env
from breakout_rl.atari_trace import RawFrameLogger


DEFAULT_SEEDS = (101, 202, 303)
DEFAULT_ACTIONS = (1, 1, 2)


def capture_seed_clean(seed: int, actions: tuple[int, ...]) -> dict[str, Any]:
    """Capture one seed while retaining the reset stack before stepping."""

    raw = make_breakout_raw_env(sticky_action_probability=0.25)
    logger = RawFrameLogger(raw)
    preprocessed = AtariPreprocessing(
        logger,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    stacked = FrameStackObservation(preprocessed, stack_size=4)
    env = BreakoutFireResetWrapper(stacked)
    try:
        observation, reset_info = env.reset(seed=seed)
        logger.finish_reset()
        reset_observation = np.asarray(observation, dtype=np.uint8).copy()
        reset_frames = [frame.copy() for frame in logger.reset_frames]
        steps: list[dict[str, Any]] = []
        for action_index in actions:
            # Gymnasium's ALE wrapper exposes the minimal action set as a
            # Discrete action index; the browser calls ale.act() with the
            # corresponding raw ALE code directly.
            observation, reward, terminated, truncated, info = env.step(action_index)
            raw_frames = logger.consume_step_frames()
            pooled = np.maximum(raw_frames[-2], raw_frames[-1]) if len(raw_frames) >= 2 else raw_frames[-1]
            stacked_observation = np.asarray(observation, dtype=np.uint8)
            steps.append(
                {
                    "requestedModelAction": action_index,
                    "executedModelAction": int(info["fire_reset_executed_action"]),
                    "autoFire": bool(info["fire_reset_auto"]),
                    "autoFireReason": info["fire_reset_reason"],
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "rawGrayscaleFrames": [frame.reshape(-1).tolist() for frame in raw_frames],
                    "pooledGrayscale": pooled.reshape(-1).tolist(),
                    "processedFrame": stacked_observation[-1].reshape(-1).tolist(),
                    "observation": stacked_observation.reshape(-1).tolist(),
                }
            )
            if terminated or truncated:
                break
        return {
            "seed": seed,
            "noopMax": 30,
            "noopCount": len(reset_frames) - 1,
            "resetNoopGrayscaleFrames": [frame.reshape(-1).tolist() for frame in reset_frames[1:]],
            "resetGrayscaleFrame": reset_frames[-1].reshape(-1).tolist(),
            "resetProcessedFrame": reset_observation[-1].reshape(-1).tolist(),
            "resetObservation": reset_observation.reshape(-1).tolist(),
            "resetInfo": {key: value for key, value in reset_info.items() if isinstance(value, (str, int, float, bool, type(None)))},
            "steps": steps,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("assets/day29/native-preprocessing-fixture.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--actions", type=int, nargs="+", default=list(DEFAULT_ACTIONS))
    args = parser.parse_args()
    actions = tuple(args.actions)
    if any(action not in {0, 1, 2, 3} for action in actions):
        raise SystemExit("actions must be model indices 0..3")
    artifact = {
        "schemaVersion": 1,
        "artifactType": "day29_native_preprocessing_fixture",
        "source": "breakout_env.make_breakout_raw_env + Gymnasium AtariPreprocessing + FrameStackObservation + BreakoutFireResetWrapper",
        "atariPreprocessing": {
            "noop_max": 30,
            "frame_skip": 4,
            "screen_size": 84,
            "terminal_on_life_loss": False,
            "grayscale_obs": True,
            "grayscale_newaxis": False,
            "scale_obs": False,
            "max_pool_space": "grayscale",
            "resize_interpolation": "cv2.INTER_AREA",
            "seed_semantics": {
                "episode_seed": "Gymnasium AtariEnv.reset(seed)",
                "seed_sequence": "np.random.SeedSequence(seed).generate_state(n_words=2)",
                "np_seed": "first uint32 -> env.unwrapped.np_random for noop draw",
                "ale_seed": "second uint32 -> ALE random_seed for game/sticky RNG",
            },
        },
        "seeds": list(args.seeds),
        "actions": list(actions),
        "samples": [capture_seed_clean(seed, actions) for seed in args.seeds],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": args.seeds, "actions": list(actions)}))


if __name__ == "__main__":
    main()
