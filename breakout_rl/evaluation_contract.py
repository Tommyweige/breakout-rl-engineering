"""Machine-readable Breakout evaluation semantics shared with later days."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA_VERSION = 2
BREAKOUT_ENVIRONMENT_ID = "ALE/Breakout-v5"
BREAKOUT_FRAME_SKIP = 4
BREAKOUT_STICKY_ACTION_PROBABILITY = 0.25


def _positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return int(parsed)


def _non_negative_int(value: Any, *, name: str) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"{name} must not be negative")
    return int(parsed)


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number between 0 and 1")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1")
    return parsed


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def expand_concrete_episode_seeds(
    evaluation_seeds: Sequence[int],
    *,
    episodes_per_seed: int,
) -> tuple[int, ...]:
    """Expand each evaluation group into its deterministic reset seeds."""

    if isinstance(evaluation_seeds, (str, bytes)):
        raise TypeError("evaluation_seeds must be a non-empty sequence")
    if not evaluation_seeds:
        raise ValueError("evaluation_seeds must be non-empty")
    parsed_seeds = tuple(
        _non_negative_int(seed, name="evaluation_seed") for seed in evaluation_seeds
    )
    if len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError("evaluation_seeds must be unique")
    count = _positive_int(episodes_per_seed, name="episodes_per_seed")
    return tuple(
        seed + offset
        for seed in parsed_seeds
        for offset in range(count)
    )


@dataclass(frozen=True)
class BreakoutEvaluationContractV2:
    """Environment and scoring semantics for Day 15 diagnostics and Day 16."""

    schema_version: int
    contract_id: str
    environment_id: str
    frame_skip: int
    frame_stack: int
    sticky_action_probability: float
    fire_reset: bool
    terminal_on_life_loss: bool
    time_limit_semantics: Mapping[str, Any]
    concrete_episode_seeds: tuple[int, ...]
    evaluation_epsilon: float
    raw_reward_rule: str

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CONTRACT_SCHEMA_VERSION}"
            )
        _non_empty_string(self.contract_id, name="contract_id")
        _non_empty_string(self.environment_id, name="environment_id")
        _positive_int(self.frame_skip, name="frame_skip")
        _positive_int(self.frame_stack, name="frame_stack")
        _probability(
            self.sticky_action_probability,
            name="sticky_action_probability",
        )
        if not isinstance(self.fire_reset, bool):
            raise TypeError("fire_reset must be a boolean")
        if not isinstance(self.terminal_on_life_loss, bool):
            raise TypeError("terminal_on_life_loss must be a boolean")
        if not isinstance(self.time_limit_semantics, Mapping):
            raise TypeError("time_limit_semantics must be an object")
        for name in (
            "source",
            "max_num_frames_per_episode",
            "agent_step_limit",
            "truncated_is_finished",
        ):
            if name not in self.time_limit_semantics:
                raise ValueError(f"time_limit_semantics is missing {name}")
        _non_empty_string(
            self.time_limit_semantics["source"],
            name="time_limit_semantics.source",
        )
        _positive_int(
            self.time_limit_semantics["max_num_frames_per_episode"],
            name="time_limit_semantics.max_num_frames_per_episode",
        )
        _positive_int(
            self.time_limit_semantics["agent_step_limit"],
            name="time_limit_semantics.agent_step_limit",
        )
        if not isinstance(self.time_limit_semantics["truncated_is_finished"], bool):
            raise TypeError("time_limit_semantics.truncated_is_finished must be a boolean")
        if not self.concrete_episode_seeds:
            raise ValueError("concrete_episode_seeds must be non-empty")
        parsed_seeds = tuple(
            _non_negative_int(seed, name="concrete_episode_seed")
            for seed in self.concrete_episode_seeds
        )
        if len(set(parsed_seeds)) != len(parsed_seeds):
            raise ValueError("concrete_episode_seeds must be unique")
        _probability(self.evaluation_epsilon, name="evaluation_epsilon")
        _non_empty_string(self.raw_reward_rule, name="raw_reward_rule")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BreakoutEvaluationContractV2":
        if not isinstance(values, Mapping):
            raise TypeError("evaluation contract must be a JSON object")
        raw_seeds = values.get("concrete_episode_seeds")
        if isinstance(raw_seeds, (str, bytes)) or not isinstance(raw_seeds, Sequence):
            raise TypeError("concrete_episode_seeds must be a non-empty sequence")
        time_limit = values.get("time_limit_semantics")
        if not isinstance(time_limit, Mapping):
            raise TypeError("time_limit_semantics must be an object")
        return cls(
            schema_version=values.get("schema_version"),
            contract_id=values.get("contract_id"),
            environment_id=values.get("environment_id"),
            frame_skip=values.get("frame_skip"),
            frame_stack=values.get("frame_stack"),
            sticky_action_probability=values.get("sticky_action_probability"),
            fire_reset=values.get("fire_reset"),
            terminal_on_life_loss=values.get("terminal_on_life_loss"),
            time_limit_semantics=dict(time_limit),
            concrete_episode_seeds=tuple(raw_seeds),
            evaluation_epsilon=values.get("evaluation_epsilon"),
            raw_reward_rule=values.get("raw_reward_rule"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "environment_id": self.environment_id,
            "frame_skip": self.frame_skip,
            "frame_stack": self.frame_stack,
            "sticky_action_probability": self.sticky_action_probability,
            "fire_reset": self.fire_reset,
            "terminal_on_life_loss": self.terminal_on_life_loss,
            "time_limit_semantics": dict(self.time_limit_semantics),
            "concrete_episode_seeds": list(self.concrete_episode_seeds),
            "evaluation_epsilon": self.evaluation_epsilon,
            "raw_reward_rule": self.raw_reward_rule,
        }


def load_evaluation_contract(path: str | Path) -> BreakoutEvaluationContractV2:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: evaluation contract must be a JSON object")
    return BreakoutEvaluationContractV2.from_mapping(payload)


def validate_breakout_runtime_contract(
    contract: BreakoutEvaluationContractV2,
) -> None:
    """Reject Contract v2 values that the shared Breakout constructor cannot honor."""

    if not isinstance(contract, BreakoutEvaluationContractV2):
        raise TypeError("contract must be a BreakoutEvaluationContractV2")
    if contract.environment_id != BREAKOUT_ENVIRONMENT_ID:
        raise ValueError(
            f"unsupported contract environment: {contract.environment_id}"
        )
    if contract.frame_skip != BREAKOUT_FRAME_SKIP:
        raise ValueError(
            f"Breakout runtime requires frame_skip={BREAKOUT_FRAME_SKIP}"
        )
    if contract.frame_stack != 4:
        raise ValueError("Breakout runtime requires frame_stack=4")
    if contract.sticky_action_probability != BREAKOUT_STICKY_ACTION_PROBABILITY:
        raise ValueError(
            "Breakout runtime requires sticky_action_probability="
            f"{BREAKOUT_STICKY_ACTION_PROBABILITY}"
        )
    if not contract.fire_reset:
        raise ValueError("Breakout runtime requires fire_reset=true")
    if contract.terminal_on_life_loss:
        raise ValueError(
            "Breakout runtime requires terminal_on_life_loss=false"
        )
    time_limit = contract.time_limit_semantics
    if time_limit["source"] != "ale.game_truncated":
        raise ValueError(
            "Breakout runtime requires TimeLimit source ale.game_truncated"
        )
    if time_limit["max_num_frames_per_episode"] != 108000:
        raise ValueError(
            "Breakout runtime requires max_num_frames_per_episode=108000"
        )
    if time_limit["agent_step_limit"] != 27000:
        raise ValueError("Breakout runtime requires agent_step_limit=27000")
    if not time_limit["truncated_is_finished"]:
        raise ValueError("Breakout runtime treats truncated episodes as finished")


__all__ = [
    "BreakoutEvaluationContractV2",
    "BREAKOUT_ENVIRONMENT_ID",
    "BREAKOUT_FRAME_SKIP",
    "BREAKOUT_STICKY_ACTION_PROBABILITY",
    "CONTRACT_SCHEMA_VERSION",
    "expand_concrete_episode_seeds",
    "load_evaluation_contract",
    "validate_breakout_runtime_contract",
]
