"""Generate a frozen Contract v2 Breakout probe-state artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from breakout_env import make_breakout_env
from breakout_rl.analysis.q_values import save_probe_states
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)


def _observation(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or tuple(array.shape) != (4, 84, 84):
        raise ValueError(
            "Breakout Contract v2 observations must have shape (4, 84, 84) "
            f"and dtype uint8; received {tuple(array.shape)} {array.dtype}"
        )
    return np.ascontiguousarray(array)


def collect_probe_states(
    contract: BreakoutEvaluationContractV2,
    *,
    seeds: Sequence[int],
    states_per_seed: int,
    stride: int,
    max_steps: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Collect deterministic observations using a seeded random behavior policy."""

    if states_per_seed < 1 or stride < 1 or max_steps < 1:
        raise ValueError("states_per_seed, stride, and max_steps must be positive")
    if not seeds:
        raise ValueError("seeds must contain at least one value")

    states: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for concrete_seed in seeds:
        env = make_breakout_env(**breakout_environment_kwargs(contract))
        try:
            observation, _ = env.reset(seed=int(concrete_seed))
            current = _observation(observation)
            rng = np.random.default_rng(int(concrete_seed))
            collected = 0
            for step in range(max_steps):
                requested_action = int(rng.integers(0, int(env.action_space.n)))
                next_observation, reward, terminated, truncated, info = env.step(
                    requested_action
                )
                if step % stride == 0 and collected < states_per_seed:
                    executed_action = int(
                        info.get("fire_reset_executed_action", requested_action)
                    )
                    records.append(
                        {
                            "concrete_seed": int(concrete_seed),
                            "episode": 1,
                            "step": int(step),
                            "observation_shape": list(current.shape),
                            "dtype": str(current.dtype),
                            "requested_action": requested_action,
                            "executed_action": executed_action,
                            "action_overridden": executed_action != requested_action,
                            "fire_reset_auto": bool(info.get("fire_reset_auto", False)),
                            "fire_reset_reason": info.get("fire_reset_reason"),
                            "raw_reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    states.append(current.copy())
                    collected += 1
                current = _observation(next_observation)
                if terminated or truncated:
                    break
            if collected != states_per_seed:
                raise RuntimeError(
                    f"seed {concrete_seed} produced only {collected}/"
                    f"{states_per_seed} probe states before episode end"
                )
        finally:
            env.close()
    return np.stack(states, axis=0), records


def generate_probe_artifact(
    *,
    contract_path: str | Path,
    output: str | Path,
    seeds: Sequence[int] | None = None,
    states_per_seed: int = 4,
    stride: int = 32,
    max_steps: int = 256,
) -> Path:
    """Generate and save probes plus their Contract v2 provenance."""

    source = Path(contract_path)
    contract = load_evaluation_contract(source)
    validate_breakout_runtime_contract(contract)
    selected_seeds = tuple(
        int(seed)
        for seed in (contract.concrete_episode_seeds if seeds is None else seeds)
    )
    observations, records = collect_probe_states(
        contract,
        seeds=selected_seeds,
        states_per_seed=states_per_seed,
        stride=stride,
        max_steps=max_steps,
    )
    metadata = {
        "schema_version": 1,
        "contract_id": contract.contract_id,
        "contract_path": source.as_posix(),
        "contract_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "environment_id": contract.environment_id,
        "source_behavior": "seeded_uniform_random_requested_actions",
        "environment_fire_reset": contract.fire_reset,
        "seeds": list(selected_seeds),
        "states_per_seed": states_per_seed,
        "stride": stride,
        "observation_shape": list(observations.shape[1:]),
        "observation_dtype": str(observations.dtype),
        "state_count": int(observations.shape[0]),
        "records": records,
    }
    return save_probe_states(output, observations, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reusable fixed probe states under Contract v2."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day17/probe_states.npz"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--states-per-seed", type=int, default=4)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = generate_probe_artifact(
        contract_path=args.contract,
        output=args.output,
        seeds=args.seeds,
        states_per_seed=args.states_per_seed,
        stride=args.stride,
        max_steps=args.max_steps,
    )
    print(json.dumps({"output": path.as_posix()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
