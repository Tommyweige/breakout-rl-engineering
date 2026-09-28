"""Canonical full-clear detection for the repository's ALE Breakout ROM."""

from __future__ import annotations

import hashlib
import math
import operator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


COMPLETION_SCHEMA_VERSION = 1
BREAKOUT_COMPLETION_DETECTOR_ID = "ale-breakout-two-wall-score-864-v1"
BREAKOUT_ENVIRONMENT_ID = "ALE/Breakout-v5"
BREAKOUT_ALE_PY_VERSION = "0.12.0"
BREAKOUT_ROM_SHA256 = "376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5"
BREAKOUT_MODE = 0
BREAKOUT_DIFFICULTY = 0
BREAKOUT_FULL_CLEAR_SCORE = 864
BREAKOUT_COMPLETION_SOURCE = "ale_raw_reward_exact_max_score"


@dataclass(frozen=True)
class CompletionSupport:
    """Whether this environment matches the audited completion contract."""

    supported: bool
    environment_id: str | None
    game: str | None
    mode: int | None
    difficulty: int | None
    ale_py_version: str | None
    rom_sha256: str | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_id": BREAKOUT_COMPLETION_DETECTOR_ID,
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "supported": self.supported,
            "environment_id": self.environment_id,
            "game": self.game,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "ale_py_version": self.ale_py_version,
            "rom_sha256": self.rom_sha256,
            "completion_rule": (
                "destroy both brick walls; exact documented maximum score"
            ),
            "score_threshold": BREAKOUT_FULL_CLEAR_SCORE,
            "completion_detection_source": (
                BREAKOUT_COMPLETION_SOURCE if self.supported else None
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CompletionState:
    """The latched completion state observed during one episode."""

    cleared: bool | None
    clear_agent_step: int | None = None
    clear_emulator_frame: int | None = None
    clear_score: float | None = None
    lives_remaining_at_clear: int | None = None
    completion_detection_source: str | None = None


def _bundled_breakout_rom_sha256() -> str:
    from ale_py import roms

    rom_path = Path(roms.get_rom_path("breakout"))
    return hashlib.sha256(rom_path.read_bytes()).hexdigest()


def inspect_breakout_completion_support(env: Any) -> CompletionSupport:
    """Fail closed unless the loaded ALE environment matches the audited ROM."""

    base = getattr(env, "unwrapped", env)
    spec = getattr(env, "spec", None)
    spec_kwargs = getattr(spec, "kwargs", {})
    if not isinstance(spec_kwargs, dict):
        spec_kwargs = {}
    environment_id = getattr(spec, "id", None)
    game = getattr(base, "_game", spec_kwargs.get("game"))
    mode = getattr(base, "_game_mode", spec_kwargs.get("mode"))
    difficulty = getattr(base, "_game_difficulty", spec_kwargs.get("difficulty"))
    if mode is None:
        mode = spec_kwargs.get("mode")
    if difficulty is None:
        difficulty = spec_kwargs.get("difficulty")

    def _optional_integer(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(operator.index(value))
        except TypeError:
            return None

    parsed_mode = _optional_integer(mode)
    parsed_difficulty = _optional_integer(difficulty)

    try:
        ale_py_version = version("ale-py")
    except PackageNotFoundError:
        ale_py_version = None
    try:
        rom_sha256 = _bundled_breakout_rom_sha256()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        rom_sha256 = None

    ale = getattr(base, "ale", None)
    required_ale_methods = (
        "getRAMSize",
        "getEpisodeFrameNumber",
        "lives",
        "game_over",
        "game_truncated",
    )
    has_required_ale_methods = all(
        callable(getattr(ale, name, None)) for name in required_ale_methods
    )
    try:
        ram_size = ale.getRAMSize() if has_required_ale_methods else None
    except (AttributeError, RuntimeError, TypeError):
        ram_size = None

    reason = None
    if environment_id != BREAKOUT_ENVIRONMENT_ID:
        reason = "environment id does not match the audited ALE/Breakout-v5 environment"
    elif game != "breakout":
        reason = "loaded game is not the packaged Breakout ROM"
    elif parsed_mode != BREAKOUT_MODE or parsed_difficulty != BREAKOUT_DIFFICULTY:
        reason = "game mode or difficulty does not match the audited default game"
    elif ale_py_version != BREAKOUT_ALE_PY_VERSION:
        reason = "ale-py version does not match the audited runtime"
    elif rom_sha256 != BREAKOUT_ROM_SHA256:
        reason = "Breakout ROM checksum does not match the audited ROM"
    elif not has_required_ale_methods or ram_size != 128:
        reason = "ALE does not expose the audited RAM and episode-state interface"

    return CompletionSupport(
        supported=reason is None,
        environment_id=environment_id if isinstance(environment_id, str) else None,
        game=game if isinstance(game, str) else None,
        mode=parsed_mode,
        difficulty=parsed_difficulty,
        ale_py_version=ale_py_version,
        rom_sha256=rom_sha256,
        reason=reason,
    )


class BreakoutCompletionDetector:
    """Detect a verified two-wall clear from the audited raw score signal."""

    def __init__(self, support: CompletionSupport) -> None:
        if not isinstance(support, CompletionSupport):
            raise TypeError("support must be a CompletionSupport")
        self.support = support
        self.reset()

    @property
    def state(self) -> CompletionState:
        return self._state

    def reset(self) -> CompletionState:
        """Start a new episode without inferring a clear from prior results."""

        self._state = CompletionState(
            cleared=False if self.support.supported else None,
            completion_detection_source=(
                BREAKOUT_COMPLETION_SOURCE if self.support.supported else None
            ),
        )
        return self._state

    def observe(
        self,
        *,
        cumulative_score: float,
        agent_step: int,
        emulator_frame: int | None,
        lives_remaining: int | None,
    ) -> CompletionState:
        """Latch the first exact maximum-score observation for this episode."""

        if not self.support.supported or self._state.cleared is True:
            return self._state
        if isinstance(cumulative_score, bool):
            raise TypeError("cumulative_score must be a finite number")
        try:
            score = float(cumulative_score)
        except (TypeError, ValueError) as error:
            raise TypeError("cumulative_score must be a finite number") from error
        if not math.isfinite(score):
            raise ValueError("cumulative_score must be a finite number")
        if isinstance(agent_step, bool):
            raise ValueError("agent_step must be a positive integer")
        try:
            parsed_agent_step = int(operator.index(agent_step))
        except TypeError as error:
            raise ValueError("agent_step must be a positive integer") from error
        if parsed_agent_step < 1:
            raise ValueError("agent_step must be a positive integer")
        if emulator_frame is not None:
            if isinstance(emulator_frame, bool):
                raise ValueError(
                    "emulator_frame must be a non-negative integer or None"
                )
            try:
                emulator_frame = int(operator.index(emulator_frame))
            except TypeError as error:
                raise ValueError(
                    "emulator_frame must be a non-negative integer or None"
                ) from error
            if emulator_frame < 0:
                raise ValueError(
                    "emulator_frame must be a non-negative integer or None"
                )
        if lives_remaining is not None:
            if isinstance(lives_remaining, bool):
                raise ValueError(
                    "lives_remaining must be a non-negative integer or None"
                )
            try:
                lives_remaining = int(operator.index(lives_remaining))
            except TypeError as error:
                raise ValueError(
                    "lives_remaining must be a non-negative integer or None"
            ) from error
            if lives_remaining < 0:
                raise ValueError(
                    "lives_remaining must be a non-negative integer or None"
                )

        if math.isclose(score, BREAKOUT_FULL_CLEAR_SCORE, rel_tol=0.0, abs_tol=1e-6):
            self._state = CompletionState(
                cleared=True,
                clear_agent_step=parsed_agent_step,
                clear_emulator_frame=emulator_frame,
                clear_score=score,
                lives_remaining_at_clear=lives_remaining,
                completion_detection_source=BREAKOUT_COMPLETION_SOURCE,
            )
        return self._state


def read_ale_episode_frame(env: Any) -> int | None:
    base = getattr(env, "unwrapped", env)
    ale = getattr(base, "ale", None)
    getter = getattr(ale, "getEpisodeFrameNumber", None)
    if not callable(getter):
        return None
    try:
        value = int(getter())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def read_ale_lives(env: Any) -> int | None:
    base = getattr(env, "unwrapped", env)
    ale = getattr(base, "ale", None)
    getter = getattr(ale, "lives", None)
    if not callable(getter):
        return None
    try:
        value = int(getter())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


__all__ = [
    "BREAKOUT_COMPLETION_DETECTOR_ID",
    "BREAKOUT_COMPLETION_SOURCE",
    "BREAKOUT_DIFFICULTY",
    "BREAKOUT_ENVIRONMENT_ID",
    "BREAKOUT_FULL_CLEAR_SCORE",
    "BREAKOUT_MODE",
    "BREAKOUT_ROM_SHA256",
    "COMPLETION_SCHEMA_VERSION",
    "BreakoutCompletionDetector",
    "CompletionState",
    "CompletionSupport",
    "inspect_breakout_completion_support",
    "read_ale_episode_frame",
    "read_ale_lives",
]
