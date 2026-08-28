"""FIRE and TimeLimit diagnostics for the frozen Day 15 DQN."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import operator
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from breakout_rl.evaluation_artifacts import summary_from_episode_rows
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.training.dqn_trainer import resolve_device


DiagnosticMode = Literal["v1", "fire_assist", "epsilon005"]
EnvironmentFactory = Callable[[], Any]


@dataclass(frozen=True)
class EpisodeSpec:
    """One concrete environment reset with its evaluation-group identity."""

    evaluation_seed: int
    episode_index: int
    episode_seed: int


def _positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return int(parsed)


def _mode_settings(mode: DiagnosticMode) -> tuple[float, bool, str]:
    if mode == "v1":
        return 0.0, False, "Evaluation Contract v1: policy-responsible FIRE"
    if mode == "fire_assist":
        return 0.0, True, "Diagnostic A: greedy policy with environment-side FIRE assist"
    if mode == "epsilon005":
        return 0.05, False, "Diagnostic B: epsilon=0.05 with policy-responsible FIRE"
    raise ValueError(f"unsupported diagnostic mode: {mode}")


def _wrapper_chain(env: Any) -> list[Any]:
    chain: list[Any] = []
    current = env
    while current is not None and current not in chain:
        chain.append(current)
        current = getattr(current, "env", None)
    return chain


def _environment_limits(env: Any) -> dict[str, Any]:
    frame_skip = None
    for wrapper in _wrapper_chain(env):
        candidate = getattr(wrapper, "frame_skip", None)
        if isinstance(candidate, int) and candidate > 0:
            frame_skip = int(candidate)
            break
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    kwargs = getattr(spec, "kwargs", {})
    max_frames = kwargs.get("max_num_frames_per_episode") if isinstance(kwargs, Mapping) else None
    if isinstance(max_frames, int) and max_frames > 0 and frame_skip:
        agent_step_limit = max_frames // frame_skip
    else:
        agent_step_limit = None
    return {
        "frame_skip": frame_skip,
        "max_num_frames_per_episode": max_frames,
        "agent_step_limit": agent_step_limit,
    }


def _action_names(env: Any, action_count: int) -> tuple[str, ...]:
    unwrapped = getattr(env, "unwrapped", env)
    meanings = getattr(unwrapped, "get_action_meanings", None)
    if callable(meanings):
        values = tuple(str(value) for value in meanings())
        if len(values) == action_count and all(values):
            return values
    return tuple(f"ACTION_{index}" for index in range(action_count))


def _find_action(action_names: Sequence[str], name: str) -> int | None:
    for index, action_name in enumerate(action_names):
        if action_name.upper() == name:
            return index
    return None


def _ale(env: Any) -> Any:
    return getattr(getattr(env, "unwrapped", env), "ale", None)


def _read_lives(ale: Any) -> tuple[int | None, bool]:
    method = getattr(ale, "lives", None)
    if not callable(method):
        return None, False
    try:
        value = method()
        if isinstance(value, bool):
            return None, False
        return int(value), True
    except (TypeError, ValueError, RuntimeError):
        return None, False


def _time_limit_signal(ale: Any, truncated: bool, info: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not truncated:
        return False, None
    game_truncated = getattr(ale, "game_truncated", None)
    if callable(game_truncated):
        try:
            if bool(game_truncated()):
                return True, "ale.game_truncated"
        except RuntimeError:
            pass
    if bool(info.get("TimeLimit.truncated", False)):
        return True, "info.TimeLimit.truncated"
    if bool(info.get("time_limit", False)):
        return True, "info.time_limit"
    return False, None


def _observation_signal(previous: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    difference = np.not_equal(previous, current)
    changed_fraction = float(np.count_nonzero(difference) / difference.size)
    return {
        "changed": bool(changed_fraction > 0.0),
        "changed_fraction": changed_fraction,
        "checksum": hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest()[:16],
    }


def _max_consecutive(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _policy_action(
    model: nn.Module,
    observation: np.ndarray,
    *,
    device: torch.device,
    epsilon: float,
    rng: np.random.Generator,
    action_count: int,
) -> tuple[int, int, list[float], str]:
    state = observation_to_tensor(observation, device=device)
    with torch.no_grad():
        q_tensor = model(state)
    if not isinstance(q_tensor, torch.Tensor) or tuple(q_tensor.shape) != (1, action_count):
        raise ValueError(
            f"diagnostic DQN must return shape (1, {action_count}); got {getattr(q_tensor, 'shape', None)}"
        )
    if not torch.isfinite(q_tensor).all().item():
        raise ValueError("diagnostic DQN returned non-finite Q-values")
    q_values = [float(value) for value in q_tensor[0].detach().cpu().tolist()]
    greedy_action = int(torch.argmax(q_tensor[0]).item())
    if rng.random() < epsilon:
        return int(rng.integers(0, action_count)), greedy_action, q_values, "epsilon_random"
    return greedy_action, greedy_action, q_values, "greedy"


def _run_episode(
    model: nn.Module,
    *,
    env_factory: EnvironmentFactory,
    device: torch.device,
    episode_spec: EpisodeSpec,
    mode: DiagnosticMode,
    record_trace: bool,
    environment_fire_assist: bool | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    epsilon, mode_fire_assist, _ = _mode_settings(mode)
    fire_assist = mode_fire_assist if environment_fire_assist is None else False
    env = env_factory()
    trace: list[dict[str, Any]] = []
    try:
        action_count = _positive_int(
            getattr(getattr(env, "action_space", None), "n", None),
            name="env.action_space.n",
        )
        action_names = _action_names(env, action_count)
        fire_action = _find_action(action_names, "FIRE")
        if fire_assist and fire_action is None:
            raise ValueError("FIRE assist requires an action named FIRE")
        limits = _environment_limits(env)
        observation, _ = env.reset(seed=episode_spec.episode_seed)
        observation = np.asarray(observation)
        seed_method = getattr(getattr(env, "action_space", None), "seed", None)
        if callable(seed_method):
            seed_method(episode_spec.episode_seed)
        rng = np.random.default_rng(episode_spec.episode_seed)
        ale = _ale(env)
        lives_before_reset, lives_reliable = _read_lives(ale)
        current_lives = lives_before_reset
        initial_lives = lives_before_reset
        needs_fire = bool(fire_assist)
        pending_life_loss_indices: list[int] = []
        life_loss_events: list[dict[str, Any]] = []
        steps_since_last_reward = 0
        steps_since_life_loss = 0
        max_steps_since_last_reward = 0
        max_steps_since_life_loss = 0
        unchanged_flags: list[bool] = []
        changed_fractions: list[float] = []
        action_values: list[int] = []
        greedy_values: list[int] = []
        auto_fire_count = 0
        auto_fire_steps: list[int] = []
        episode_return = 0.0
        terminated = False
        truncated = False
        time_limit = False
        time_limit_source: str | None = None
        max_steps = (
            int(limits["agent_step_limit"] * 2)
            if isinstance(limits["agent_step_limit"], int)
            else 100_000
        )

        for agent_step in range(1, max_steps + 1):
            policy_action, greedy_action, q_values, policy_source = _policy_action(
                model,
                observation,
                device=device,
                epsilon=epsilon,
                rng=rng,
                action_count=action_count,
            )
            executed_action = policy_action
            action_source = policy_source
            auto_fire = False
            fire_reason = None
            if fire_assist and needs_fire:
                executed_action = int(fire_action)
                action_source = "auto_fire_after_life_loss" if pending_life_loss_indices else "auto_fire_initial_serve"
                fire_reason = action_source
                auto_fire = True
                needs_fire = False
                auto_fire_count += 1
                auto_fire_steps.append(agent_step)
            if not 0 <= executed_action < action_count:
                raise ValueError(f"diagnostic policy returned illegal action {executed_action}")
            next_observation, reward, terminated_raw, truncated_raw, info = env.step(executed_action)
            next_observation = np.asarray(next_observation)
            if not isinstance(info, Mapping):
                info = {}
            environment_executed_action = info.get(
                "fire_reset_executed_action",
                executed_action,
            )
            try:
                environment_executed_action = operator.index(environment_executed_action)
            except TypeError as error:
                raise ValueError("fire_reset_executed_action must be an integer") from error
            environment_auto_fire = bool(info.get("fire_reset_auto", False))
            if environment_auto_fire and not auto_fire:
                auto_fire_count += 1
                auto_fire_steps.append(agent_step)
            if environment_auto_fire:
                executed_action = int(environment_executed_action)
                auto_fire = True
                action_source = "environment_fire_reset"
                fire_reason = info.get("fire_reset_reason")
                if fire_reason not in {"initial_serve", "after_life_loss"}:
                    fire_reason = "environment_fire_reset"
            reward_value = float(reward)
            if not math.isfinite(reward_value):
                raise ValueError("environment reward must be finite")
            terminated = bool(terminated_raw)
            truncated = bool(truncated_raw)
            if terminated and truncated:
                raise ValueError("environment returned terminated and truncated together")
            time_limit, time_limit_source = _time_limit_signal(ale, truncated, info)
            next_lives, next_lives_reliable = _read_lives(ale)
            lives_reliable = lives_reliable and next_lives_reliable
            pending_before_step = list(pending_life_loss_indices)
            life_loss_event = (
                current_lives is not None
                and next_lives is not None
                and next_lives < current_lives
            )
            if life_loss_event:
                life_loss_events.append(
                    {
                        "step": agent_step,
                        "lives_before": current_lives,
                        "lives_after": next_lives,
                        "first_fire_step": None,
                        "fire_latency_steps": None,
                    }
                )
                pending_life_loss_indices.append(len(life_loss_events) - 1)
                if fire_assist:
                    needs_fire = True
                steps_since_life_loss = 0
            else:
                steps_since_life_loss += 1
            max_steps_since_life_loss = max(
                max_steps_since_life_loss,
                steps_since_life_loss,
            )

            first_fire_after_life_loss = bool(
                executed_action == fire_action and pending_before_step
            )
            if first_fire_after_life_loss:
                for event_index in pending_before_step:
                    event = life_loss_events[event_index]
                    event["first_fire_step"] = agent_step
                    event["fire_latency_steps"] = agent_step - int(event["step"])
                pending_life_loss_indices = [
                    event_index
                    for event_index in pending_life_loss_indices
                    if event_index not in pending_before_step
                ]

            steps_since_last_reward = 0 if reward_value != 0.0 else steps_since_last_reward + 1
            max_steps_since_last_reward = max(
                max_steps_since_last_reward,
                steps_since_last_reward,
            )
            episode_return += reward_value
            observation_info = _observation_signal(observation, next_observation)
            unchanged_flags.append(not observation_info["changed"])
            changed_fractions.append(float(observation_info["changed_fraction"]))
            action_values.append(executed_action)
            greedy_values.append(greedy_action)
            if record_trace:
                trace.append(
                    {
                        "evaluation_seed": episode_spec.evaluation_seed,
                        "episode_index": episode_spec.episode_index,
                        "episode_seed": episode_spec.episode_seed,
                        "agent_step": agent_step,
                        "raw_reward": reward_value,
                        "action": executed_action,
                        "policy_action": policy_action,
                        "action_source": action_source,
                        "auto_fire": auto_fire,
                        "fire_reason": fire_reason,
                        "q_values": q_values,
                        "greedy_action": greedy_action,
                        "lives": next_lives,
                        "lives_before": current_lives,
                        "lives_after": next_lives,
                        "lives_observation_reliable": lives_reliable,
                        "life_loss_event": life_loss_event,
                        "steps_since_last_reward": steps_since_last_reward,
                        "steps_since_life_loss": steps_since_life_loss,
                        "first_fire_after_life_loss": first_fire_after_life_loss,
                        "terminated": terminated,
                        "truncated": truncated,
                        "time_limit": time_limit,
                        "time_limit_source": time_limit_source,
                        "observation_changed": observation_info["changed"],
                        "observation_changed_fraction": observation_info["changed_fraction"],
                        "observation_checksum": observation_info["checksum"],
                    }
                )
            current_lives = next_lives
            observation = next_observation
            if terminated or truncated:
                break
        else:
            raise RuntimeError(
                f"diagnostic episode {episode_spec.episode_seed} did not finish within safety bound {max_steps}"
            )

        action_counter = Counter(action_values)
        greedy_counter = Counter(greedy_values)
        action_distribution = {
            name: int(action_counter.get(index, 0))
            for index, name in enumerate(action_names)
        }
        greedy_distribution = {
            name: int(greedy_counter.get(index, 0))
            for index, name in enumerate(action_names)
        }
        total_steps = len(action_values)
        life_loss_fire_latencies = [
            int(event["fire_latency_steps"])
            for event in life_loss_events
            if event["fire_latency_steps"] is not None
        ]
        row = {
            "evaluation_seed": episode_spec.evaluation_seed,
            "episode_index": episode_spec.episode_index,
            "episode_seed": episode_spec.episode_seed,
            "episode_return": float(episode_return),
            "episode_length": total_steps,
            "terminated": terminated,
            "truncated": truncated,
            "time_limit": time_limit,
            "time_limit_source": time_limit_source,
            "complete": terminated or truncated,
            "stop_reason": "time_limit" if time_limit else "terminated" if terminated else "truncated",
            "time_limit_agent_step_limit": limits["agent_step_limit"],
            "lives_observation_reliable": lives_reliable,
            "initial_lives": initial_lives,
            "final_lives": current_lives,
            "life_loss_count": len(life_loss_events),
            "life_loss_events": life_loss_events,
            "first_fire_after_life_loss_steps": [
                int(event["first_fire_step"])
                for event in life_loss_events
                if event["first_fire_step"] is not None
            ],
            "life_loss_to_fire_latencies": life_loss_fire_latencies,
            "life_loss_events_without_fire": len(life_loss_events) - len(life_loss_fire_latencies),
            "max_steps_since_last_reward": max_steps_since_last_reward,
            "max_steps_since_life_loss": max_steps_since_life_loss,
            "observation_changed_step_fraction": float(
                sum(not value for value in unchanged_flags) / max(total_steps, 1)
            ),
            "mean_observation_changed_fraction": float(
                np.mean(changed_fractions) if changed_fractions else 0.0
            ),
            "unchanged_observation_step_fraction": float(
                sum(unchanged_flags) / max(total_steps, 1)
            ),
            "max_consecutive_unchanged_observation": _max_consecutive(unchanged_flags),
            "action_distribution": action_distribution,
            "dominant_action": max(action_distribution, key=action_distribution.get),
            "dominant_action_fraction": float(
                max(action_distribution.values()) / max(total_steps, 1)
            ),
            "max_consecutive_action": _max_consecutive(
                [action == max(action_counter, key=action_counter.get) for action in action_values]
            ),
            "greedy_action_distribution": greedy_distribution,
            "dominant_greedy_action": max(
                greedy_distribution, key=greedy_distribution.get
            ),
            "dominant_greedy_action_fraction": float(
                max(greedy_distribution.values()) / max(total_steps, 1)
            ),
            "max_consecutive_greedy_action": _max_consecutive(
                [greedy == max(greedy_counter, key=greedy_counter.get) for greedy in greedy_values]
            ),
            "auto_fire_count": auto_fire_count,
        }
        row["auto_fire_steps"] = auto_fire_steps
        return row, trace
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _aggregate_action_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = row.get(field, {})
        if isinstance(values, Mapping):
            counts.update({str(name): int(count) for name, count in values.items()})
    return dict(sorted(counts.items()))


def _dominant_fraction(counts: Mapping[str, int]) -> tuple[str | None, float]:
    total = sum(int(value) for value in counts.values())
    if total == 0:
        return None, 0.0
    name = max(counts, key=counts.get)
    return name, float(counts[name] / total)


def summarize_diagnostic_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one diagnostic episode is required")
    summary = summary_from_episode_rows(rows)
    terminated_returns = [float(row["episode_return"]) for row in rows if row["terminated"]]
    truncated_returns = [float(row["episode_return"]) for row in rows if row["truncated"]]
    terminated_lengths = [float(row["episode_length"]) for row in rows if row["terminated"]]
    truncated_lengths = [float(row["episode_length"]) for row in rows if row["truncated"]]
    action_counts = _aggregate_action_counts(rows, "action_distribution")
    greedy_counts = _aggregate_action_counts(rows, "greedy_action_distribution")
    dominant_action, dominant_action_fraction = _dominant_fraction(action_counts)
    dominant_greedy_action, dominant_greedy_fraction = _dominant_fraction(greedy_counts)
    latencies = [
        float(latency)
        for row in rows
        for latency in row.get("life_loss_to_fire_latencies", [])
    ]
    summary.update(
        {
            "dominant_action": dominant_action,
            "dominant_action_fraction": dominant_action_fraction,
            "mean_dominant_action_fraction": float(
                np.mean([float(row["dominant_action_fraction"]) for row in rows])
            ),
            "action_distribution": action_counts,
            "dominant_greedy_action": dominant_greedy_action,
            "dominant_greedy_action_fraction": dominant_greedy_fraction,
            "mean_dominant_greedy_action_fraction": float(
                np.mean([float(row["dominant_greedy_action_fraction"]) for row in rows])
            ),
            "greedy_action_distribution": greedy_counts,
            "max_consecutive_action": max(int(row["max_consecutive_action"]) for row in rows),
            "max_consecutive_greedy_action": max(
                int(row["max_consecutive_greedy_action"]) for row in rows
            ),
            "max_consecutive_unchanged_observation": max(
                int(row["max_consecutive_unchanged_observation"]) for row in rows
            ),
            "mean_observation_changed_fraction": float(
                np.mean([float(row["mean_observation_changed_fraction"]) for row in rows])
            ),
            "unchanged_observation_step_fraction": float(
                np.mean([float(row["unchanged_observation_step_fraction"]) for row in rows])
            ),
            "life_loss_event_count": sum(int(row["life_loss_count"]) for row in rows),
            "life_loss_events_without_fire": sum(
                int(row["life_loss_events_without_fire"]) for row in rows
            ),
            "mean_life_loss_fire_latency": _mean_or_none(latencies),
            "life_loss_fire_latency_observed": bool(latencies),
            "auto_fire_count": sum(int(row["auto_fire_count"]) for row in rows),
            "auto_fire_steps": [
                {"episode_seed": row["episode_seed"], "steps": row.get("auto_fire_steps", [])}
                for row in rows
                if row.get("auto_fire_steps")
            ],
            "mean_return_terminated": _mean_or_none(terminated_returns),
            "mean_return_truncated": _mean_or_none(truncated_returns),
            "mean_length_terminated": _mean_or_none(terminated_lengths),
            "mean_length_truncated": _mean_or_none(truncated_lengths),
        }
    )
    return summary


def run_diagnostic_evaluation(
    model: nn.Module,
    *,
    env_factory: EnvironmentFactory,
    device: torch.device | str,
    episode_specs: Sequence[EpisodeSpec],
    mode: DiagnosticMode,
    trace_seeds: Sequence[int] = (),
    environment_fire_assist: bool | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one diagnostic mode without changing the frozen v1 artifacts."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not callable(env_factory):
        raise TypeError("env_factory must be callable")
    if not episode_specs:
        raise ValueError("episode_specs must be non-empty")
    identities = {(spec.evaluation_seed, spec.episode_index) for spec in episode_specs}
    if len(identities) != len(episode_specs):
        raise ValueError("episode_specs must contain unique evaluation seed/index pairs")
    if any(
        spec.episode_seed != spec.evaluation_seed + spec.episode_index - 1
        for spec in episode_specs
    ):
        raise ValueError("episode_specs episode_seed must follow the concrete seed expansion rule")
    epsilon, fire_assist, description = _mode_settings(mode)
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()
    trace_seed_set = {int(seed) for seed in trace_seeds}
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for episode_spec in episode_specs:
        row, trace = _run_episode(
            model,
            env_factory=env_factory,
            device=resolved_device,
            episode_spec=episode_spec,
            mode=mode,
            record_trace=episode_spec.episode_seed in trace_seed_set,
            environment_fire_assist=environment_fire_assist,
        )
        rows.append(row)
        if trace:
            traces.append(
                {
                    "evaluation_seed": episode_spec.evaluation_seed,
                    "episode_index": episode_spec.episode_index,
                    "episode_seed": episode_spec.episode_seed,
                    "steps": trace,
                }
            )
    evaluation_seeds = sorted({int(spec.evaluation_seed) for spec in episode_specs})
    episodes_per_seed = max(
        sum(spec.evaluation_seed == seed for spec in episode_specs)
        for seed in evaluation_seeds
    )
    return {
        "diagnostic_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "diagnostic_mode": mode,
        "diagnostic_description": description,
        "based_on_evaluation_contract": "v1",
        "policy_type": "dqn",
        "evaluation_seeds": evaluation_seeds,
        "episodes_per_seed": episodes_per_seed,
        "concrete_episode_seeds": [int(spec.episode_seed) for spec in episode_specs],
        "evaluation_epsilon": epsilon,
        "fire_assist": fire_assist,
        "environment_fire_assist": environment_fire_assist is True,
        "requested_device": str(device),
        "resolved_device": str(resolved_device),
        "checkpoint": dict(checkpoint or {}),
        "training": dict(training or {}),
        "per_episode": rows,
        "summary": summarize_diagnostic_rows(rows),
        "metadata": dict(metadata or {}),
        "trace": traces,
    }


def write_diagnostic_artifacts(
    payload: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Write JSON, CSV, and JSONL trace artifacts for one diagnostic mode."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "results.json"
    episodes_path = destination / "episodes.csv"
    trace_path = destination / "trace.jsonl"
    serializable = dict(payload)
    traces = serializable.pop("trace", [])
    serializable["artifacts"] = {
        "results_json": results_path.name,
        "episodes_csv": episodes_path.name,
        "trace_jsonl": trace_path.name,
    }
    results_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = serializable.get("per_episode", [])
    fieldnames = [
        "evaluation_seed",
        "episode_index",
        "episode_seed",
        "episode_return",
        "episode_length",
        "terminated",
        "truncated",
        "time_limit",
        "stop_reason",
        "life_loss_count",
        "life_loss_events_without_fire",
        "auto_fire_count",
        "dominant_action",
        "dominant_action_fraction",
        "dominant_greedy_action",
        "dominant_greedy_action_fraction",
        "mean_observation_changed_fraction",
        "unchanged_observation_step_fraction",
        "life_loss_to_fire_latencies_json",
        "action_distribution_json",
    ]
    with episodes_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{
                        field: row.get(field)
                        for field in fieldnames
                        if field not in {
                            "life_loss_to_fire_latencies_json",
                            "action_distribution_json",
                        }
                    },
                    "life_loss_to_fire_latencies_json": json.dumps(
                        row.get("life_loss_to_fire_latencies", []), ensure_ascii=False
                    ),
                    "action_distribution_json": json.dumps(
                        row.get("action_distribution", {}), ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    with trace_path.open("w", encoding="utf-8") as stream:
        for trace in traces:
            for step in trace.get("steps", []):
                stream.write(json.dumps(step, ensure_ascii=False) + "\n")
    return results_path, episodes_path, trace_path


__all__ = [
    "DiagnosticMode",
    "EpisodeSpec",
    "run_diagnostic_evaluation",
    "summarize_diagnostic_rows",
    "write_diagnostic_artifacts",
]
