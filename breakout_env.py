"""Reusable Atari Breakout environment construction for later RL stages."""

from __future__ import annotations

import operator
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


gym.register_envs(ale_py)

ENVIRONMENT_ID = "ALE/Breakout-v5"


class BreakoutFireResetWrapper(gym.Wrapper):
    """Insert FIRE only for the initial serve or after an observed life loss."""

    def __init__(
        self,
        env: gym.Env,
        *,
        max_fire_attempts: int = 8,
        confirmation_steps: int = 2,
        min_observation_change_fraction: float = 1e-4,
    ) -> None:
        super().__init__(env)
        if isinstance(max_fire_attempts, bool):
            raise TypeError("max_fire_attempts must be a positive integer")
        try:
            parsed_max_fire_attempts = operator.index(max_fire_attempts)
        except TypeError as error:
            raise TypeError("max_fire_attempts must be a positive integer") from error
        if parsed_max_fire_attempts < 1:
            raise ValueError("max_fire_attempts must be a positive integer")
        if isinstance(confirmation_steps, bool):
            raise TypeError("confirmation_steps must be a positive integer")
        try:
            parsed_confirmation_steps = operator.index(confirmation_steps)
        except TypeError as error:
            raise TypeError("confirmation_steps must be a positive integer") from error
        if parsed_confirmation_steps < 1:
            raise ValueError("confirmation_steps must be a positive integer")
        try:
            parsed_change_fraction = float(min_observation_change_fraction)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "min_observation_change_fraction must be finite and non-negative"
            ) from error
        if not np.isfinite(parsed_change_fraction) or parsed_change_fraction < 0.0:
            raise ValueError(
                "min_observation_change_fraction must be finite and non-negative"
            )
        meanings = getattr(self.unwrapped, "get_action_meanings", None)
        action_names = tuple(str(value) for value in meanings()) if callable(meanings) else ()
        try:
            self.fire_action = action_names.index("FIRE")
        except ValueError as error:
            raise ValueError("BreakoutFireResetWrapper requires a FIRE action") from error
        self.max_fire_attempts = int(parsed_max_fire_attempts)
        self.confirmation_steps = int(parsed_confirmation_steps)
        self.min_observation_change_fraction = parsed_change_fraction
        self._needs_fire = False
        self._pending_fire_reason: str | None = None
        self._fire_attempts = 0
        self._fire_activity_streak = 0
        self._last_lives: int | None = None
        self._last_observation: np.ndarray | None = None
        self._auto_fire_count = 0
        self.last_requested_action: int | None = None
        self.last_executed_action: int | None = None
        self.last_action_was_auto_fire = False
        self.last_fire_reset_confirmed: bool | None = None
        self.last_fire_reset_confirmation: str | None = None

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

    @staticmethod
    def _observation_changed_fraction(
        previous: np.ndarray | None,
        current: Any,
    ) -> float:
        if previous is None:
            return 0.0
        current_array = np.asarray(current)
        if previous.shape != current_array.shape or previous.size == 0:
            return 0.0
        return float(np.count_nonzero(previous != current_array) / previous.size)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._last_lives = self._lives()
        self._needs_fire = True
        self._pending_fire_reason = "initial_serve"
        self._fire_attempts = 0
        self._fire_activity_streak = 0
        self._last_observation = np.array(observation, copy=True)
        self._auto_fire_count = 0
        self.last_requested_action = None
        self.last_executed_action = None
        self.last_action_was_auto_fire = False
        self.last_fire_reset_confirmed = None
        self.last_fire_reset_confirmation = None
        reset_info = dict(info)
        reset_info.update(
            {
                "fire_reset_needs_fire": True,
                "fire_reset_pending_reason": "initial_serve",
                "fire_reset_max_attempts": self.max_fire_attempts,
                "fire_reset_confirmation_steps": self.confirmation_steps,
                "fire_reset_observation_change_threshold": self.min_observation_change_fraction,
            }
        )
        return observation, reset_info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        requested_action = int(action)
        auto_fire = self._needs_fire
        fire_reason = self._pending_fire_reason if auto_fire else None
        previous_observation = self._last_observation
        executed_action = self.fire_action if auto_fire else requested_action
        observation, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        observation_changed_fraction = self._observation_changed_fraction(
            previous_observation,
            observation,
        )
        fire_attempt = 0
        fire_confirmed: bool | None = False if auto_fire else None
        fire_confirmation: str | None = None
        self.last_requested_action = requested_action
        self.last_executed_action = executed_action
        self.last_action_was_auto_fire = auto_fire
        if auto_fire:
            self._fire_attempts += 1
            fire_attempt = self._fire_attempts
            self._auto_fire_count += 1
            if float(reward) != 0.0:
                fire_confirmed = True
                fire_confirmation = "reward"
            elif (
                observation_changed_fraction
                >= self.min_observation_change_fraction
            ):
                self._fire_activity_streak += 1
                if self._fire_activity_streak >= self.confirmation_steps:
                    fire_confirmed = True
                    fire_confirmation = "observation_activity_streak"
            else:
                self._fire_activity_streak = 0
            if fire_confirmed or bool(terminated) or bool(truncated):
                self._needs_fire = False
                self._pending_fire_reason = None
                self._fire_attempts = 0
                self._fire_activity_streak = 0
            elif fire_attempt >= self.max_fire_attempts:
                self._last_observation = np.array(observation, copy=True)
                raise RuntimeError(
                    "FIRE serve was not confirmed after "
                    f"{self.max_fire_attempts} attempts for {fire_reason}; "
                    "the wrapper refuses to continue with a possible serve deadlock"
                )
        current_lives = self._lives()
        life_loss = (
            self._last_lives is not None
            and current_lives is not None
            and current_lives < self._last_lives
        )
        if life_loss:
            self._needs_fire = True
            self._pending_fire_reason = "after_life_loss"
            self._fire_attempts = 0
            self._fire_activity_streak = 0
        elif not auto_fire:
            self._fire_activity_streak = 0
        self._last_lives = current_lives
        self._last_observation = np.array(observation, copy=True)
        self.last_fire_reset_confirmed = fire_confirmed
        self.last_fire_reset_confirmation = fire_confirmation
        info.update(
            {
                "fire_reset_auto": auto_fire,
                "fire_reset_reason": fire_reason,
                "fire_reset_life_loss": life_loss,
                "fire_reset_requested_action": requested_action,
                "fire_reset_executed_action": executed_action,
                "fire_reset_attempt": fire_attempt,
                "fire_reset_confirmed": fire_confirmed,
                "fire_reset_confirmation": fire_confirmation,
                "fire_reset_activity_streak": self._fire_activity_streak,
                "fire_reset_confirmation_steps": self.confirmation_steps,
                "fire_reset_observation_change_threshold": self.min_observation_change_fraction,
                "fire_reset_observation_changed_fraction": observation_changed_fraction,
                "fire_reset_auto_fire_count": self._auto_fire_count,
                "fire_reset_needs_fire": self._needs_fire,
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
    fire_reset_max_attempts: int = 8,
    fire_confirmation_steps: int = 2,
    fire_confirmation_change_fraction: float = 1e-4,
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
    return (
        BreakoutFireResetWrapper(
            stacked,
            max_fire_attempts=fire_reset_max_attempts,
            confirmation_steps=fire_confirmation_steps,
            min_observation_change_fraction=fire_confirmation_change_fraction,
        )
        if fire_reset
        else stacked
    )


def make_breakout_vector_env(
    num_envs: int,
    *,
    render_mode: str | None = None,
    stack_size: int = 4,
    fire_reset: bool = True,
    fire_reset_max_attempts: int = 8,
    fire_confirmation_steps: int = 2,
    fire_confirmation_change_fraction: float = 1e-4,
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
            fire_reset_max_attempts=fire_reset_max_attempts,
            fire_confirmation_steps=fire_confirmation_steps,
            fire_confirmation_change_fraction=fire_confirmation_change_fraction,
        )

    return gym.vector.SyncVectorEnv(
        [make_one for _ in range(parsed_num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
