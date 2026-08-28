"""Profile batched action inference across multiple Breakout environments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch

from breakout_env import make_breakout_vector_env
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.training.dqn_trainer import resolve_device
from profile_batch_size_experiment import RuntimeSampler, _sample_summary


def _build_vector_env(
    count: int,
    *,
    contract: BreakoutEvaluationContractV2 | None = None,
) -> gym.vector.SyncVectorEnv:
    active_contract = contract
    if active_contract is not None:
        return make_breakout_vector_env(
            count,
            **breakout_environment_kwargs(active_contract),
        )
    return make_breakout_vector_env(
        count,
        stack_size=4,
        fire_reset=True,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_count(
    count: int,
    *,
    iterations: int,
    seed: int,
    device: torch.device,
    cpu_threads: int,
    sample_path: Path,
    gpu_index: int,
    sample_interval: float,
    contract: BreakoutEvaluationContractV2 | None = None,
) -> dict[str, Any]:
    torch.set_num_threads(cpu_threads)
    envs = _build_vector_env(count, contract=contract)
    sampler = RuntimeSampler(
        sample_path,
        interval_seconds=sample_interval,
        gpu_index=gpu_index,
    )
    observations, _ = envs.reset(seed=seed)
    action_count = int(envs.single_action_space.n)
    network = DQNNetwork(action_count).to(device).eval()

    try:
        sampler.start()
        state = observation_to_tensor(
            np.ascontiguousarray(observations),
            device=device,
        )
        with torch.no_grad():
            for _ in range(min(50, iterations)):
                actions = network(state).argmax(dim=1).cpu().numpy()
                observations, _, terminated, truncated, _ = envs.step(actions)
                if np.logical_or(terminated, truncated).any():
                    observations, _ = envs.reset(
                        options={
                            "reset_mask": np.logical_or(terminated, truncated),
                        }
                    )
                state = observation_to_tensor(
                    np.ascontiguousarray(observations),
                    device=device,
                )

        _synchronize(device)
        start = time.perf_counter()
        total_steps = 0
        state = observation_to_tensor(
            np.ascontiguousarray(observations),
            device=device,
        )
        with torch.no_grad():
            for _ in range(iterations):
                actions = network(state).argmax(dim=1).cpu().numpy()
                observations, _, terminated, truncated, _ = envs.step(actions)
                total_steps += count
                if np.logical_or(terminated, truncated).any():
                    observations, _ = envs.reset(
                        options={
                            "reset_mask": np.logical_or(terminated, truncated),
                        }
                    )
                state = observation_to_tensor(
                    np.ascontiguousarray(observations),
                    device=device,
                )
        _synchronize(device)
        elapsed = max(time.perf_counter() - start, 1e-9)
    finally:
        sampler.stop()
        envs.close()

    return {
        "environment_count": count,
        "iterations": iterations,
        "total_environment_steps": total_steps,
        "wall_clock_seconds": elapsed,
        "environment_steps_per_second": total_steps / elapsed,
        "gpu_profile": _sample_summary(
            sample_path,
            interval_seconds=sample_interval,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile batched action inference over SyncVectorEnv Breakout instances."
    )
    parser.add_argument("--environment-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("assets/day16/vectorized-rollout-profiling"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.environment_counts or any(count < 1 for count in args.environment_counts):
        raise ValueError("environment counts must be positive")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")

    device = resolve_device(args.device)
    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract)
    results = []
    for count in args.environment_counts:
        sample_path = args.samples_root / f"envs-{count}" / "runtime-samples.csv"
        results.append(
            _run_count(
                count,
                iterations=args.iterations,
                seed=args.seed,
                device=device,
                cpu_threads=args.cpu_threads,
                sample_path=sample_path,
                gpu_index=args.gpu_index,
                sample_interval=args.sample_interval,
                contract=contract,
            )
        )

    report = {
        "schema_version": 1,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cpu_threads": args.cpu_threads,
        "contract_path": args.contract.as_posix(),
        "contract": contract.to_dict(),
        "results": results,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
