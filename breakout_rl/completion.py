"""Canonical full-clear detection for the repository's ALE Breakout ROM."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping


COMPLETION_SCHEMA_VERSION = 1
BREAKOUT_AUDIT_CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "configs/eval/breakout_completion_audit_v1.json"
)


def _load_audit_config() -> Mapping[str, Any]:
    try:
        payload = json.loads(BREAKOUT_AUDIT_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


_AUDIT_CONFIG = _load_audit_config()
_AUDIT_ENVIRONMENT = _AUDIT_CONFIG.get("environment", {})
_AUDIT_RUNTIME = _AUDIT_CONFIG.get("runtime", {})
_AUDIT_COMPLETION = _AUDIT_CONFIG.get("completion", {})
if not isinstance(_AUDIT_ENVIRONMENT, Mapping):
    _AUDIT_ENVIRONMENT = {}
if not isinstance(_AUDIT_RUNTIME, Mapping):
    _AUDIT_RUNTIME = {}
if not isinstance(_AUDIT_COMPLETION, Mapping):
    _AUDIT_COMPLETION = {}


def _audit_integer(values: Mapping[str, Any], field: str, fallback: int) -> int:
    value = values.get(field)
    if isinstance(value, bool):
        return fallback
    try:
        return int(operator.index(value))
    except TypeError:
        return fallback


def _audit_string(values: Mapping[str, Any], field: str, fallback: str) -> str:
    value = values.get(field)
    return value if isinstance(value, str) and value else fallback


def _audit_addresses(values: Mapping[str, Any]) -> tuple[int, ...]:
    raw_addresses = values.get("addresses")
    if not isinstance(raw_addresses, list):
        return ()
    addresses: list[int] = []
    for value in raw_addresses:
        if isinstance(value, bool):
            return ()
        try:
            address = int(operator.index(value))
        except TypeError:
            return ()
        if not 0 <= address < 128:
            return ()
        addresses.append(address)
    return tuple(addresses)


BREAKOUT_COMPLETION_DETECTOR_ID = _audit_string(
    _AUDIT_CONFIG, "detector_id", "unsupported"
)
BREAKOUT_ENVIRONMENT_ID = _audit_string(_AUDIT_ENVIRONMENT, "id", "unsupported")
BREAKOUT_ALE_PY_VERSION = _audit_string(
    _AUDIT_RUNTIME, "ale_py_version", "unsupported"
)
BREAKOUT_ROM_SHA256 = _audit_string(_AUDIT_RUNTIME, "rom_sha256", "unsupported")
BREAKOUT_MODE = _audit_integer(_AUDIT_ENVIRONMENT, "mode", -1)
BREAKOUT_DIFFICULTY = _audit_integer(_AUDIT_ENVIRONMENT, "difficulty", -1)
BREAKOUT_FULL_CLEAR_SCORE = _audit_integer(_AUDIT_COMPLETION, "score_threshold", -1)
BREAKOUT_EXPECTED_MAXIMUM_SCORE = _audit_integer(
    _AUDIT_COMPLETION, "expected_maximum_score", -1
)
BREAKOUT_COMPLETION_SOURCE = _audit_string(
    _AUDIT_COMPLETION, "detection_source", "unsupported"
)
_AUDIT_SCORE_RAM_FIELD = _AUDIT_COMPLETION.get("ram_score_field", {})
if not isinstance(_AUDIT_SCORE_RAM_FIELD, Mapping):
    _AUDIT_SCORE_RAM_FIELD = {}
BREAKOUT_SCORE_RAM_ADDRESSES = _audit_addresses(_AUDIT_SCORE_RAM_FIELD)
BREAKOUT_SCORE_RAM_ENCODING = _audit_string(
    _AUDIT_SCORE_RAM_FIELD, "encoding", "unsupported"
)
BREAKOUT_SCORE_RAM_BYTE_ORDER = _audit_string(
    _AUDIT_SCORE_RAM_FIELD, "byte_order", "unsupported"
)
BREAKOUT_SCORE_RAM_DECODING_RULE = _audit_string(
    _AUDIT_SCORE_RAM_FIELD, "decoding_rule", "unsupported"
)
_COMPLETION_AUDIT_VALID = (
    _AUDIT_CONFIG.get("schema_version") == COMPLETION_SCHEMA_VERSION
    and BREAKOUT_COMPLETION_DETECTOR_ID != "unsupported"
    and BREAKOUT_ENVIRONMENT_ID == "ALE/Breakout-v5"
    and BREAKOUT_ALE_PY_VERSION != "unsupported"
    and BREAKOUT_ROM_SHA256 != "unsupported"
    and BREAKOUT_MODE >= 0
    and BREAKOUT_DIFFICULTY >= 0
    and BREAKOUT_FULL_CLEAR_SCORE > 0
    and BREAKOUT_EXPECTED_MAXIMUM_SCORE == BREAKOUT_FULL_CLEAR_SCORE
    and BREAKOUT_COMPLETION_SOURCE != "unsupported"
    and len(BREAKOUT_SCORE_RAM_ADDRESSES) == 2
    and BREAKOUT_SCORE_RAM_ENCODING == "packed_bcd_four_digit"
    and BREAKOUT_SCORE_RAM_BYTE_ORDER == "high_pair_then_low_pair"
    and BREAKOUT_SCORE_RAM_DECODING_RULE != "unsupported"
)


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
            "audit_config_path": "configs/eval/breakout_completion_audit_v1.json",
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
            "score_ram_addresses": list(BREAKOUT_SCORE_RAM_ADDRESSES),
            "score_ram_encoding": BREAKOUT_SCORE_RAM_ENCODING,
            "score_ram_byte_order": BREAKOUT_SCORE_RAM_BYTE_ORDER,
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
    unavailable_reason: str | None = None


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
        "getRAM",
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
    if not _COMPLETION_AUDIT_VALID:
        reason = "completion audit config is missing or invalid"
    elif environment_id != BREAKOUT_ENVIRONMENT_ID:
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

        self._unavailable = not self.support.supported
        self._state = CompletionState(
            cleared=False if self.support.supported else None,
            completion_detection_source=(
                BREAKOUT_COMPLETION_SOURCE if self.support.supported else None
            ),
            unavailable_reason=self.support.reason,
        )
        return self._state

    def _mark_unavailable(self, reason: str) -> CompletionState:
        self._unavailable = True
        self._state = CompletionState(
            cleared=None,
            unavailable_reason=reason,
        )
        return self._state

    def observe(
        self,
        *,
        cumulative_score: float,
        ram_score: int | None,
        agent_step: int,
        emulator_frame: int | None,
        lives_remaining: int | None,
    ) -> CompletionState:
        """Latch the first exact maximum-score observation for this episode."""

        if not self.support.supported or self._unavailable or self._state.cleared is True:
            return self._state
        if isinstance(cumulative_score, bool):
            raise TypeError("cumulative_score must be a finite number")
        try:
            score = float(cumulative_score)
        except (TypeError, ValueError) as error:
            raise TypeError("cumulative_score must be a finite number") from error
        if not math.isfinite(score):
            raise ValueError("cumulative_score must be a finite number")
        if isinstance(ram_score, bool):
            return self._mark_unavailable("ALE scoreboard RAM value is invalid")
        try:
            parsed_ram_score = int(operator.index(ram_score))
        except TypeError:
            return self._mark_unavailable("ALE scoreboard RAM value is unavailable")
        if parsed_ram_score < 0:
            return self._mark_unavailable("ALE scoreboard RAM value is invalid")
        if not math.isclose(score, parsed_ram_score, rel_tol=0.0, abs_tol=1e-6):
            return self._mark_unavailable(
                "ALE RAM score disagrees with cumulative raw reward"
            )
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

        if parsed_ram_score == BREAKOUT_FULL_CLEAR_SCORE:
            self._state = CompletionState(
                cleared=True,
                clear_agent_step=parsed_agent_step,
                clear_emulator_frame=emulator_frame,
                clear_score=float(parsed_ram_score),
                lives_remaining_at_clear=lives_remaining,
                completion_detection_source=BREAKOUT_COMPLETION_SOURCE,
            )
        return self._state


def _read_ale_integer(env: Any, getter_name: str) -> int | None:
    base = getattr(env, "unwrapped", env)
    ale = getattr(base, "ale", None)
    getter = getattr(ale, getter_name, None)
    if not callable(getter):
        return None
    try:
        raw_value = getter()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if isinstance(raw_value, bool):
        return None
    try:
        value = int(operator.index(raw_value))
    except TypeError:
        return None
    return value if value >= 0 else None


def read_ale_episode_frame(env: Any) -> int | None:
    return _read_ale_integer(env, "getEpisodeFrameNumber")


def read_ale_lives(env: Any) -> int | None:
    return _read_ale_integer(env, "lives")


def read_breakout_score(env: Any) -> int | None:
    """Read the audited four-digit packed-BCD score from the Breakout ROM."""

    base = getattr(env, "unwrapped", env)
    ale = getattr(base, "ale", None)
    getter = getattr(ale, "getRAM", None)
    if not callable(getter) or len(BREAKOUT_SCORE_RAM_ADDRESSES) != 2:
        return None
    try:
        ram = bytes(getter())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if not ram or max(BREAKOUT_SCORE_RAM_ADDRESSES) >= len(ram):
        return None
    digits: list[int] = []
    for address in BREAKOUT_SCORE_RAM_ADDRESSES:
        value = ram[address]
        high_digit, low_digit = value >> 4, value & 0x0F
        if high_digit > 9 or low_digit > 9:
            return None
        digits.extend((high_digit, low_digit))
    score = 0
    for digit in digits:
        score = score * 10 + digit
    return score


__all__ = [
    "BREAKOUT_COMPLETION_DETECTOR_ID",
    "BREAKOUT_COMPLETION_SOURCE",
    "BREAKOUT_DIFFICULTY",
    "BREAKOUT_ENVIRONMENT_ID",
    "BREAKOUT_FULL_CLEAR_SCORE",
    "BREAKOUT_MODE",
    "BREAKOUT_ROM_SHA256",
    "BREAKOUT_SCORE_RAM_ADDRESSES",
    "COMPLETION_SCHEMA_VERSION",
    "BreakoutCompletionDetector",
    "CompletionState",
    "CompletionSupport",
    "inspect_breakout_completion_support",
    "read_ale_episode_frame",
    "read_ale_lives",
    "read_breakout_score",
]
