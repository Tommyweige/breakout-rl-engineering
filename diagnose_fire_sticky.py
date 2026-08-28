"""Diagnose Contract v2 FIRE serving under real ALE sticky actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from breakout_env import make_breakout_env
from breakout_rl.evaluation import load_dqn_checkpoint
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.training.diagnostics import ATARI_ACTION_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _action_names(env: Any, action_count: int) -> tuple[str, ...]:
    unwrapped = getattr(env, "unwrapped", env)
    meanings = getattr(unwrapped, "get_action_meanings", None)
    if callable(meanings):
        names = tuple(str(value) for value in meanings())
        if len(names) == action_count and all(names):
            return names
    return tuple(ATARI_ACTION_NAMES.get(index, f"ACTION_{index}") for index in range(action_count))


def _lives(env: Any) -> int | None:
    ale = getattr(getattr(env, "unwrapped", env), "ale", None)
    method = getattr(ale, "lives", None)
    if not callable(method):
        return None
    try:
        value = method()
    except (RuntimeError, TypeError, ValueError):
        return None
    if isinstance(value, bool):
        return None
    return int(value)


def _time_limit(env: Any, truncated: bool, info: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not truncated:
        return False, None
    ale = getattr(getattr(env, "unwrapped", env), "ale", None)
    signal = getattr(ale, "game_truncated", None)
    if callable(signal):
        try:
            if bool(signal()):
                return True, "ale.game_truncated"
        except RuntimeError:
            pass
    if bool(info.get("TimeLimit.truncated", False)):
        return True, "info.TimeLimit.truncated"
    return False, None


def _observation_signal(previous: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    difference = np.not_equal(previous, current)
    changed_fraction = float(np.count_nonzero(difference) / max(difference.size, 1))
    return {
        "changed": bool(changed_fraction > 0.0),
        "changed_fraction": changed_fraction,
        "checksum": hashlib.sha256(
            np.ascontiguousarray(current).tobytes()
        ).hexdigest()[:16],
    }


def _max_consecutive(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _policy_action(
    model: torch.nn.Module,
    observation: np.ndarray,
    *,
    device: torch.device,
    action_count: int,
) -> tuple[int, list[float]]:
    state = observation_to_tensor(observation, device=device)
    with torch.no_grad():
        q_values = model(state)
    if not isinstance(q_values, torch.Tensor) or tuple(q_values.shape) != (1, action_count):
        raise ValueError(
            "diagnostic model must return shape "
            f"(1, {action_count}); got {getattr(q_values, 'shape', None)}"
        )
    if not torch.isfinite(q_values).all().item():
        raise ValueError("diagnostic model returned non-finite Q-values")
    return int(torch.argmax(q_values[0]).item()), [
        float(value) for value in q_values[0].detach().cpu().tolist()
    ]


def _resolved_action(
    info: Mapping[str, Any],
    *,
    requested_action: int,
    action_count: int,
) -> tuple[int, bool, str | None]:
    raw_requested = info.get("fire_reset_requested_action", requested_action)
    try:
        resolved_requested = operator.index(raw_requested)
    except TypeError as error:
        raise ValueError("fire_reset_requested_action must be an integer") from error
    if int(resolved_requested) != requested_action:
        raise ValueError("wrapper provenance does not match the policy request")
    auto_fire = bool(info.get("fire_reset_auto", False))
    if auto_fire and "fire_reset_executed_action" not in info:
        raise ValueError("auto FIRE must report fire_reset_executed_action")
    raw_executed = info.get("fire_reset_executed_action", requested_action)
    try:
        executed_action = operator.index(raw_executed)
    except TypeError as error:
        raise ValueError("fire_reset_executed_action must be an integer") from error
    if not 0 <= int(executed_action) < action_count:
        raise ValueError(f"wrapper resolved illegal action {executed_action}")
    reason = info.get("fire_reset_reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("fire_reset_reason must be a string or None")
    return int(executed_action), auto_fire, reason


def _run_episode(
    model: torch.nn.Module,
    *,
    env_factory: Any,
    device: torch.device,
    seed: int,
    max_steps: int,
    trace: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = env_factory()
    started_at = time.perf_counter()
    trace_rows: list[dict[str, Any]] = []
    try:
        action_count = int(env.action_space.n)
        action_names = _action_names(env, action_count)
        fire_action = action_names.index("FIRE")
        observation, _ = env.reset(seed=seed)
        observation = np.asarray(observation)
        previous_lives = _lives(env)
        initial_lives = previous_lives
        requested_counts: Counter[str] = Counter()
        executed_counts: Counter[str] = Counter()
        fire_reasons: Counter[str] = Counter()
        serve_attempts: list[dict[str, Any]] = []
        life_loss_events: list[dict[str, Any]] = []
        changed_flags: list[bool] = []
        changed_fractions: list[float] = []
        requested_actions: list[int] = []
        executed_actions: list[int] = []
        episode_return = 0.0
        terminated = False
        truncated = False
        time_limit = False
        time_limit_source: str | None = None

        for agent_step in range(1, max_steps + 1):
            requested_action, q_values = _policy_action(
                model,
                observation,
                device=device,
                action_count=action_count,
            )
            next_observation, reward, terminated_raw, truncated_raw, raw_info = env.step(
                requested_action
            )
            info = raw_info if isinstance(raw_info, Mapping) else {}
            executed_action, auto_fire, fire_reason = _resolved_action(
                info,
                requested_action=requested_action,
                action_count=action_count,
            )
            next_observation = np.asarray(next_observation)
            signal = _observation_signal(observation, next_observation)
            reward_value = float(reward)
            if not math.isfinite(reward_value):
                raise ValueError("raw reward must be finite")
            terminated = bool(terminated_raw)
            truncated = bool(truncated_raw)
            if terminated and truncated:
                raise ValueError("terminated and truncated cannot both be true")
            time_limit, time_limit_source = _time_limit(env, truncated, info)
            current_lives = _lives(env)
            life_loss = (
                previous_lives is not None
                and current_lives is not None
                and current_lives < previous_lives
            )
            if life_loss:
                life_loss_events.append(
                    {
                        "step": agent_step,
                        "lives_before": previous_lives,
                        "lives_after": current_lives,
                    }
                )
            if auto_fire:
                attempt = int(info.get("fire_reset_attempt", 0))
                confirmed = info.get("fire_reset_confirmed")
                confirmation = info.get("fire_reset_confirmation")
                serve_attempts.append(
                    {
                        "step": agent_step,
                        "reason": fire_reason,
                        "attempt": attempt,
                        "confirmed": confirmed,
                        "confirmation": confirmation,
                        "observation_changed_fraction": info.get(
                            "fire_reset_observation_changed_fraction",
                            signal["changed_fraction"],
                        ),
                        "raw_reward": reward_value,
                    }
                )
                if fire_reason is not None:
                    fire_reasons[fire_reason] += 1

            requested_counts[action_names[requested_action]] += 1
            executed_counts[action_names[executed_action]] += 1
            requested_actions.append(requested_action)
            executed_actions.append(executed_action)
            changed_flags.append(bool(signal["changed"]))
            changed_fractions.append(float(signal["changed_fraction"]))
            episode_return += reward_value
            if trace:
                trace_rows.append(
                    {
                        "episode_seed": seed,
                        "agent_step": agent_step,
                        "requested_policy_action": requested_action,
                        "wrapper_resolved_action": executed_action,
                        "executed_action": executed_action,
                        "fire_reset_auto": auto_fire,
                        "fire_reset_reason": fire_reason,
                        "fire_reset_attempt": info.get("fire_reset_attempt", 0),
                        "fire_reset_confirmed": info.get("fire_reset_confirmed"),
                        "fire_reset_confirmation": info.get(
                            "fire_reset_confirmation"
                        ),
                        "fire_reset_needs_fire": info.get("fire_reset_needs_fire"),
                        "fire_reset_auto_fire_count": info.get(
                            "fire_reset_auto_fire_count"
                        ),
                        "lives_before": previous_lives,
                        "lives_after": current_lives,
                        "life_loss_event": life_loss,
                        "raw_reward": reward_value,
                        "terminated": terminated,
                        "truncated": truncated,
                        "time_limit": time_limit,
                        "time_limit_source": time_limit_source,
                        "observation_changed": signal["changed"],
                        "observation_changed_fraction": signal["changed_fraction"],
                        "observation_checksum": signal["checksum"],
                        "q_values": q_values,
                    }
                )
            previous_lives = current_lives
            observation = next_observation
            if terminated or truncated:
                break
        else:
            raise RuntimeError(
                f"seed {seed} did not finish within the {max_steps}-step diagnostic limit"
            )

        elapsed = max(time.perf_counter() - started_at, 1e-9)
        auto_fire_count = len(serve_attempts)
        row = {
            "episode_seed": seed,
            "episode_return": episode_return,
            "episode_length": len(executed_actions),
            "terminated": terminated,
            "truncated": truncated,
            "time_limit": time_limit,
            "time_limit_source": time_limit_source,
            "initial_lives": initial_lives,
            "final_lives": previous_lives,
            "lives_observation_reliable": initial_lives is not None
            and all(event["lives_before"] is not None for event in life_loss_events),
            "life_loss_count": len(life_loss_events),
            "life_loss_events": life_loss_events,
            "requested_action_distribution": dict(
                sorted(requested_counts.items())
            ),
            "wrapper_resolved_action_distribution": dict(
                sorted(executed_counts.items())
            ),
            "executed_action_distribution": dict(sorted(executed_counts.items())),
            "auto_fire_count": auto_fire_count,
            "auto_fire_reason_counts": dict(sorted(fire_reasons.items())),
            "serve_attempts": serve_attempts,
            "serve_retry_count": sum(
                max(int(attempt["attempt"]) - 1, 0) for attempt in serve_attempts
            ),
            "serve_confirmation_failures": sum(
                1 for attempt in serve_attempts if attempt["confirmed"] is False
            ),
            "observation_changed_step_fraction": float(
                sum(changed_flags) / max(len(changed_flags), 1)
            ),
            "mean_observation_changed_fraction": float(
                np.mean(changed_fractions) if changed_fractions else 0.0
            ),
            "unchanged_observation_step_fraction": float(
                sum(not value for value in changed_flags) / max(len(changed_flags), 1)
            ),
            "max_consecutive_unchanged_observation": _max_consecutive(
                [not value for value in changed_flags]
            ),
            "diagnostic_wall_clock_seconds": elapsed,
        }
        return row, trace_rows
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _runtime(device: torch.device) -> dict[str, Any]:
    values: dict[str, Any] = {
        "pytorch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "resolved_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    index = 0 if device.index is None else int(device.index)
    values.update(
        {
            "cuda_device_index": index,
            "gpu_model": torch.cuda.get_device_name(index),
        }
    )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose Contract v2 FIRE and sticky-action serving on real ALE."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--trace-seeds", nargs="+", type=int, default=[101, 102])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day16/fire-sticky-diagnostic.json"),
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=Path("assets/day16/fire-sticky-trace.jsonl"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract)
    if not torch.cuda.is_available() or not str(args.device).lower().startswith("cuda"):
        raise RuntimeError(
            "real Breakout FIRE diagnostics require an available NVIDIA CUDA device; "
            "refusing CPU fallback"
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("real Breakout FIRE diagnostics require --device cuda")
    seeds = tuple(args.seeds or contract.concrete_episode_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique list")
    expected_seeds = set(contract.concrete_episode_seeds)
    unexpected = sorted(set(seeds) - expected_seeds)
    if unexpected:
        raise ValueError(f"diagnostic seeds are not in Contract v2: {unexpected}")
    trace_seeds = set(args.trace_seeds)
    if not trace_seeds.issubset(set(seeds)):
        raise ValueError("trace-seeds must be included in seeds")
    max_steps = args.max_steps or int(contract.time_limit_semantics["agent_step_limit"])
    if max_steps < 1:
        raise ValueError("max-steps must be positive")

    env_factory = lambda: make_breakout_env(
        stack_size=contract.frame_stack,
        fire_reset=contract.fire_reset,
    )
    loaded = load_dqn_checkpoint(
        args.checkpoint,
        device=device,
        env_factory=env_factory,
    )
    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for seed in seeds:
        row, trace = _run_episode(
            loaded.model,
            env_factory=env_factory,
            device=device,
            seed=int(seed),
            max_steps=max_steps,
            trace=int(seed) in trace_seeds,
        )
        rows.append(row)
        if trace:
            traces.extend(trace)

    requested_counts: Counter[str] = Counter()
    executed_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in rows:
        requested_counts.update(row["requested_action_distribution"])
        executed_counts.update(row["executed_action_distribution"])
        reason_counts.update(row["auto_fire_reason_counts"])
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "purpose": "Contract v2 FIRE/sticky-action root-cause diagnostic",
        "git_commit_sha": _git_commit_sha(),
        "contract_path": args.contract.as_posix(),
        "contract": contract.to_dict(),
        "checkpoint": {
            **dict(loaded.checkpoint_metadata),
            "source_day14_run_id": None,
            "source_day16_run_id": Path(args.checkpoint).resolve().parent.parent.name,
            "sha256": _sha256(Path(args.checkpoint).resolve()),
        },
        "training": {
            **dict(loaded.training_metadata),
            "source_day14_run_id": None,
            "source_day16_run_id": Path(args.checkpoint).resolve().parent.parent.name,
        },
        "protocol": {
            "policy": "greedy DQN, epsilon=0",
            "seeds": list(seeds),
            "trace_seeds": sorted(trace_seeds),
            "max_agent_steps": max_steps,
            "fire_confirmation_rule": (
                "observable raw reward or two consecutive observations whose "
                "change fraction reaches 0.0001 after wrapper-resolved FIRE"
            ),
            "max_fire_attempts": 8,
            "fire_confirmation_steps": 2,
            "fire_confirmation_change_fraction": 1e-4,
            "hidden_ale_action_visibility": (
                "ALE API does not expose the hidden sticky-action draw; "
                "wrapper-resolved action means the action passed downward"
            ),
        },
        "runtime": {
            **_runtime(device),
            "wall_clock_seconds": time.perf_counter() - started_at,
        },
        "per_episode": rows,
        "summary": {
            "episode_count": len(rows),
            "terminated_count": sum(bool(row["terminated"]) for row in rows),
            "truncated_count": sum(bool(row["truncated"]) for row in rows),
            "time_limit_count": sum(bool(row["time_limit"]) for row in rows),
            "requested_action_distribution": dict(sorted(requested_counts.items())),
            "wrapper_resolved_action_distribution": dict(
                sorted(executed_counts.items())
            ),
            "executed_action_distribution": dict(sorted(executed_counts.items())),
            "auto_fire_count": sum(int(row["auto_fire_count"]) for row in rows),
            "auto_fire_reason_counts": dict(sorted(reason_counts.items())),
            "serve_retry_count": sum(int(row["serve_retry_count"]) for row in rows),
            "serve_confirmation_failures": sum(
                int(row["serve_confirmation_failures"]) for row in rows
            ),
            "life_loss_count": sum(int(row["life_loss_count"]) for row in rows),
            "max_consecutive_unchanged_observation": max(
                int(row["max_consecutive_unchanged_observation"]) for row in rows
            ),
        },
        "trace": {
            "path": args.trace_output.as_posix(),
            "row_count": len(traces),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    with args.trace_output.open("w", encoding="utf-8") as stream:
        for row in traces:
            stream.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Diagnostic report written to: {args.output}")
    print(f"Diagnostic trace written to: {args.trace_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
