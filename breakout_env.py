"""Reusable Atari Breakout environment construction for later RL stages."""

from __future__ import annotations

import operator
from typing import Any

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


gym.register_envs(ale_py)

ENVIRONMENT_ID = "ALE/Breakout-v5"


class BreakoutFireResetWrapper(gym.Wrapper):
    """Insert FIRE only for the initial serve or after an observed life loss."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        meanings = getattr(self.unwrapped, "get_action_meanings", None)
        action_names = tuple(str(value) for value in meanings()) if callable(meanings) else ()
        try:
            self.fire_action = action_names.index("FIRE")
        except ValueError as error:
            raise ValueError("BreakoutFireResetWrapper requires a FIRE action") from error
        self._needs_fire = False
        self._pending_fire_reason: str | None = None
        self._last_lives: int | None = None
        self._auto_fire_count = 0
        self.last_requested_action: int | None = None
        self.last_executed_action: int | None = None
        self.last_action_was_auto_fire = False

    def _lives(self) -> int | None:
        lives = getattr(getattr(self.unwrapped, "ale", None), "lives", None)
        if not callable(lives):
            return None
        try:
            return int(lives())
        except (TypeError, ValueError, RuntimeError):
            return None

    @property
    def auto_fire_count(self) -> int:
        return self._auto_fire_count

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._last_lives = self._lives()
        self._needs_fire = True
        self._pending_fire_reason = "initial_serve"
        self._auto_fire_count = 0
        self.last_requested_action = None
        self.last_executed_action = None
        self.last_action_was_auto_fire = False
        return observation, dict(info)

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        requested_action = int(action)
        auto_fire = self._needs_fire
        fire_reason = self._pending_fire_reason if auto_fire else None
        executed_action = self.fire_action if auto_fire else requested_action
        observation, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        self.last_requested_action = requested_action
        self.last_executed_action = executed_action
        self.last_action_was_auto_fire = auto_fire
        if auto_fire:
            self._auto_fire_count += 1
            self._needs_fire = False
            self._pending_fire_reason = None
        current_lives = self._lives()
        life_loss = (
            self._last_lives is not None
            and current_lives is not None
            and current_lives < self._last_lives
        )
        if life_loss:
            self._needs_fire = True
            self._pending_fire_reason = "after_life_loss"
        self._last_lives = current_lives
        info.update(
            {
                "fire_reset_auto": auto_fire,
                "fire_reset_reason": fire_reason,
                "fire_reset_life_loss": life_loss,
                "fire_reset_requested_action": requested_action,
                "fire_reset_executed_action": executed_action,
            }
        )
        return observation, float(reward), bool(terminated), bool(truncated), info


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
    fire_reset: bool = False,
) -> gym.Env:
    """Create the project's baseline preprocessed Breakout environment.

    The base ALE environment disables its own frame skipping so that
    ``AtariPreprocessing`` is the single owner of frame skip and max-pooling.
    The final observation is ``(stack_size, 84, 84)`` with ``uint8`` values.
    When ``fire_reset`` is true, the wrapper owns only the initial serve and
    the FIRE action immediately following an observed life loss.
    """

    if stack_size < 1:
        raise ValueError("stack_size must be at least 1")

    env = make_breakout_preprocessed_env(render_mode=render_mode)

    # Frame skip controls action frequency; stacking supplies short-term
    # history so a policy can infer motion from successive observations.
    stacked = FrameStackObservation(env, stack_size=stack_size)
    return BreakoutFireResetWrapper(stacked) if fire_reset else stacked


def make_breakout_vector_env(
    num_envs: int,
    *,
    render_mode: str | None = None,
    stack_size: int = 4,
    fire_reset: bool = False,
) -> gym.vector.SyncVectorEnv:
    """Create independent Breakout environments with explicit manual reset.

    ``DISABLED`` autoreset preserves the terminal transition's final
    observation. The vectorized trainer resets only the environments whose
    termination flags are true after it has inserted those transitions.
    """

    if isinstance(num_envs, bool):
        raise TypeError("num_envs must be a positive integer")
    try:
        parsed_num_envs = operator.index(num_envs)
    except TypeError as error:
        raise TypeError("num_envs must be a positive integer") from error
    if parsed_num_envs < 1:
        raise ValueError("num_envs must be a positive integer")

    def make_one() -> gym.Env:
        return make_breakout_env(
            render_mode=render_mode,
            stack_size=stack_size,
            fire_reset=fire_reset,
        )

    return gym.vector.SyncVectorEnv(
        [make_one for _ in range(parsed_num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
