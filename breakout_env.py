"""Reusable Atari Breakout environment construction for later RL stages."""

from __future__ import annotations

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


gym.register_envs(ale_py)

ENVIRONMENT_ID = "ALE/Breakout-v5"


def make_breakout_raw_env(*, render_mode: str | None = None) -> gym.Env:
    """Create the raw Breakout environment used by the preprocessing chain."""

    return gym.make(
        ENVIRONMENT_ID,
        render_mode=render_mode,
        # AtariPreprocessing must be the only component that skips frames.
        frameskip=1,
        repeat_action_probability=0.25,
    )


def make_breakout_preprocessed_env(*, render_mode: str | None = None) -> gym.Env:
    """Create Breakout after Atari preprocessing but before frame stacking."""

    env = make_breakout_raw_env(render_mode=render_mode)

    # Keep the raw pixels compact while preserving the spatial information
    # needed by Breakout. The wrapper also performs the official max-pooling.
    return AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )


def make_breakout_env(
    *,
    render_mode: str | None = None,
    stack_size: int = 4,
) -> gym.Env:
    """Create the project's baseline preprocessed Breakout environment.

    The base ALE environment disables its own frame skipping so that
    ``AtariPreprocessing`` is the single owner of frame skip and max-pooling.
    The final observation is ``(stack_size, 84, 84)`` with ``uint8`` values.
    """

    if stack_size < 1:
        raise ValueError("stack_size must be at least 1")

    env = make_breakout_preprocessed_env(render_mode=render_mode)

    # Frame skip controls action frequency; stacking supplies short-term
    # history so a policy can infer motion from successive observations.
    return FrameStackObservation(env, stack_size=stack_size)
