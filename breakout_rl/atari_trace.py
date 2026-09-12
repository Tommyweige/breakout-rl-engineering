"""Small reusable wrappers for auditing raw Atari observations."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class RawFrameLogger(gym.Wrapper):
    """Record the grayscale frame immediately after every raw ALE step."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reset_frames: list[np.ndarray] = []
        self.step_frames: list[np.ndarray] = []
        self._capturing_reset = False

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.reset_frames = [self._screen_grayscale()]
        self.step_frames = []
        self._capturing_reset = True
        return observation, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.env.step(action)
        frame = self._screen_grayscale()
        (self.reset_frames if self._capturing_reset else self.step_frames).append(frame)
        return result

    def finish_reset(self) -> None:
        self._capturing_reset = False

    def consume_step_frames(self) -> list[np.ndarray]:
        frames = self.step_frames
        self.step_frames = []
        return frames

    def _screen_grayscale(self) -> np.ndarray:
        return np.asarray(self.unwrapped.ale.getScreenGrayscale(), dtype=np.uint8).copy()
