"""Collect real CUDA Q-value diagnostics from a Breakout checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
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


def _action_names(env: Any, action_count: int) -> tuple[str, ...]:
    meanings = getattr(getattr(env, "unwrapped", env), "get_action_meanings", None)
    if callable(meanings):
        names = tuple(str(value) for value in meanings())
        if len(names) == action_count and all(names):
            return names
    return tuple(ATARI_ACTION_NAMES.get(index, f"ACTION_{index}") for index in range(action_count))


def _observation_checksum(observation: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()[:16]


def _greedy_q_values(
    model: torch.nn.Module,
    observation: np.ndarray,
    *,
    device: torch.device,
    action_count: int,
) -> tuple[int, np.ndarray]:
    state = observation_to_tensor(observation, device=device)
    with torch.no_grad():
        q_tensor = model(state)
    if not isinstance(q_tensor, torch.Tensor) or tuple(q_tensor.shape) != (1, action_count):
        raise ValueError(
            f"model must return shape (1, {action_count}); got {getattr(q_tensor, 'shape', None)}"
        )
    if not torch.isfinite(q_tensor).all().item():
        raise ValueError("model returned non-finite Q-values")
    values = q_tensor[0].detach().cpu().numpy().astype(np.float64, copy=True)
    return int(np.argmax(values)), values


def _resolved_action(
    info: Mapping[str, Any],
    *,
    requested_action: int,
    action_count: int,
) -> tuple[int, bool, str | None]:
    raw_requested = info.get("fire_reset_requested_action", requested_action)
    if operator.index(raw_requested) != requested_action:
        raise ValueError("wrapper requested-action provenance does not match policy action")
    auto_fire = bool(info.get("fire_reset_auto", False))
    if auto_fire and "fire_reset_executed_action" not in info:
        raise ValueError("auto FIRE must report fire_reset_executed_action")
    raw_executed = info.get("fire_reset_executed_action", requested_action)
    executed_action = operator.index(raw_executed)
    if not 0 <= executed_action < action_count:
        raise ValueError(f"wrapper resolved illegal action {executed_action}")
    reason = info.get("fire_reset_reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("fire_reset_reason must be a string or None")
    return int(executed_action), auto_fire, reason


def _collect_probe_states(
    model: torch.nn.Module,
    *,
    env_factory: Any,
    device: torch.device,
    seeds: Sequence[int],
    steps_per_seed: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    rows: list[dict[str, Any]] = []
    action_names: tuple[str, ...] = ()
    for seed in seeds:
        env = env_factory()
        try:
            action_count = int(env.action_space.n)
            action_names = _action_names(env, action_count)
            observation, _ = env.reset(seed=int(seed))
            observation = np.asarray(observation)
            for step in range(1, steps_per_seed + 1):
                requested_action, q_values = _greedy_q_values(
                    model,
                    observation,
                    device=device,
                    action_count=action_count,
                )
                next_observation, reward, terminated, truncated, raw_info = env.step(
                    requested_action
                )
                info = raw_info if isinstance(raw_info, Mapping) else {}
                executed_action, auto_fire, fire_reason = _resolved_action(
                    info,
                    requested_action=requested_action,
                    action_count=action_count,
                )
                top_two = np.sort(q_values)[-2:]
                next_observation = np.asarray(next_observation)
                rows.append(
                    {
                        "seed": int(seed),
                        "step": step,
                        "observation_checksum": _observation_checksum(observation),
                        "q_values": q_values.tolist(),
                        "max_q": float(q_values.max()),
                        "min_q": float(q_values.min()),
                        "action_gap": float(top_two[-1] - top_two[-2]),
                        "greedy_action": requested_action,
                        "greedy_action_name": action_names[requested_action],
                        "requested_action": requested_action,
                        "wrapper_resolved_action": executed_action,
                        "wrapper_resolved_action_name": action_names[executed_action],
                        "fire_reset_auto": auto_fire,
                        "fire_reset_reason": fire_reason,
                        "fire_reset_attempt": info.get("fire_reset_attempt", 0),
                        "raw_reward_after_action": float(reward),
                        "terminated_after_action": bool(terminated),
                        "truncated_after_action": bool(truncated),
                    }
                )
                observation = next_observation
                if terminated or truncated:
                    break
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    return rows, action_names


def _summary(rows: Sequence[Mapping[str, Any]], action_names: Sequence[str]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one probe state is required")
    q_values = np.asarray([row["q_values"] for row in rows], dtype=np.float64)
    max_q = q_values.max(axis=1)
    gaps = np.asarray([float(row["action_gap"]) for row in rows], dtype=np.float64)
    greedy_counts = Counter(
        str(row["greedy_action_name"]) for row in rows
    )
    per_action: dict[str, dict[str, float]] = {}
    for index, name in enumerate(action_names):
        values = q_values[:, index]
        per_action[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "probe_state_count": len(rows),
        "q_value_by_action": per_action,
        "max_q_mean": float(max_q.mean()),
        "max_q_std": float(max_q.std()),
        "max_q_min": float(max_q.min()),
        "max_q_max": float(max_q.max()),
        "action_gap_mean": float(gaps.mean()),
        "action_gap_std": float(gaps.std()),
        "greedy_action_distribution": dict(sorted(greedy_counts.items())),
        "auto_fire_count": sum(bool(row["fire_reset_auto"]) for row in rows),
        "truncated_probe_steps": sum(bool(row["truncated_after_action"]) for row in rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real Breakout Q-value diagnostics on NVIDIA CUDA."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--steps-per-seed", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day16/q-value-diagnostics.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract)
    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise RuntimeError(
            "real checkpoint Q-value diagnostics require --device cuda; "
            "CPU diagnostics are not an accepted replacement"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "real checkpoint Q-value diagnostics require an available NVIDIA CUDA device; "
            "refusing silent CPU fallback"
        )
    if args.steps_per_seed < 1:
        raise ValueError("steps-per-seed must be positive")
    seeds = tuple(args.seeds or contract.concrete_episode_seeds[:5])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique list")
    unexpected = sorted(set(seeds) - set(contract.concrete_episode_seeds))
    if unexpected:
        raise ValueError(f"probe seeds are not in Contract v2: {unexpected}")

    env_factory = lambda: make_breakout_env(
        stack_size=contract.frame_stack,
        fire_reset=contract.fire_reset,
    )
    loaded = load_dqn_checkpoint(
        args.checkpoint,
        device=requested_device,
        env_factory=env_factory,
    )
    rows, action_names = _collect_probe_states(
        loaded.model,
        env_factory=env_factory,
        device=requested_device,
        seeds=seeds,
        steps_per_seed=args.steps_per_seed,
    )
    checkpoint_path = Path(args.checkpoint).resolve()
    source_run_id = checkpoint_path.parent.parent.name
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "purpose": "Exploratory real-checkpoint Q-value diagnostics for Day 16",
        "interpretation_boundary": (
            "These are model outputs on real Breakout observations. Without a "
            "ground-truth Q-star oracle they do not prove that a particular "
            "checkpoint is overestimating value. The CPU toy simulation isolates "
            "the max-selection mechanism separately."
        ),
        "contract_path": args.contract.as_posix(),
        "contract": contract.to_dict(),
        "checkpoint": {
            **dict(loaded.checkpoint_metadata),
            "source_day14_run_id": None,
            "source_day16_run_id": source_run_id,
            "sha256": _sha256(checkpoint_path),
        },
        "training": {
            **dict(loaded.training_metadata),
            "source_day14_run_id": None,
            "source_day16_run_id": source_run_id,
        },
        "runtime": {
            "requested_device": str(requested_device),
            "resolved_device": str(requested_device),
            "cuda_available": True,
            "gpu_model": torch.cuda.get_device_name(requested_device),
            "pytorch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
        },
        "protocol": {
            "seeds": list(seeds),
            "steps_per_seed": args.steps_per_seed,
            "epsilon": 0.0,
            "policy": "greedy DQN",
            "observation_source": "real ALE/Breakout-v5 Contract v2 rollout",
            "inference_context": "torch.no_grad() on NVIDIA CUDA",
        },
        "action_names": list(action_names),
        "summary": _summary(rows, action_names),
        "probe_states": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Q-value diagnostic written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
