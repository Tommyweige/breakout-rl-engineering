"""Measure batched replay insertion with a real Breakout observation source."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from breakout_env import make_breakout_env
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.replay import ReplayBuffer
from breakout_rl.replay_gpu import GPUReplayBuffer
from breakout_rl.training.dqn_trainer import resolve_device


def _batch_from_real_observations(
    batch_size: int,
    *,
    seed: int,
    contract: BreakoutEvaluationContractV2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env = make_breakout_env(
        stack_size=contract.frame_stack,
        fire_reset=contract.fire_reset,
    )
    try:
        state, _ = env.reset(seed=seed)
        next_state, reward, terminated, truncated, _ = env.step(0)
    finally:
        env.close()
    states = np.repeat(np.asarray(state)[None, ...], batch_size, axis=0)
    next_states = np.repeat(np.asarray(next_state)[None, ...], batch_size, axis=0)
    return (
        states,
        np.zeros(batch_size, dtype=np.int64),
        np.full(batch_size, float(reward), dtype=np.float32),
        next_states,
        np.full(batch_size, bool(terminated), dtype=np.bool_),
        np.full(batch_size, bool(truncated), dtype=np.bool_),
    )


def _run_one(
    batch_size: int,
    *,
    iterations: int,
    seed: int,
    device: torch.device,
    capacity: int,
    contract: BreakoutEvaluationContractV2,
) -> dict[str, Any]:
    values = _batch_from_real_observations(
        batch_size,
        seed=seed,
        contract=contract,
    )
    if device.type == "cuda":
        replay: ReplayBuffer | GPUReplayBuffer = GPUReplayBuffer(
            capacity,
            observation_shape=tuple(values[0].shape[1:]),
            device=device,
        )
    else:
        replay = ReplayBuffer(
            capacity,
            observation_shape=tuple(values[0].shape[1:]),
        )

    for _ in range(min(iterations, 20)):
        replay.add_batch(*values)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    for _ in range(iterations):
        replay.add_batch(*values)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "transitions": batch_size * iterations,
        "wall_clock_seconds": elapsed,
        "transitions_per_second": batch_size * iterations / elapsed,
        "latency_ms_per_call": elapsed / iterations * 1000.0,
        "buffer_capacity": capacity,
        "observation_shape": list(values[0].shape[1:]),
        "source": "one real ALE/Breakout-v5 reset and step, repeated for copy timing",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark replay.add_batch")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day16/replay-insertion.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.batch_sizes or any(size < 1 for size in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if args.iterations < 1 or args.capacity < 1:
        raise ValueError("iterations and capacity must be positive")
    device = resolve_device(args.device)
    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract)
    results = [
        _run_one(
            batch_size,
            iterations=args.iterations,
            seed=args.seed,
            device=device,
            capacity=max(args.capacity, batch_size),
            contract=contract,
        )
        for batch_size in args.batch_sizes
    ]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "protocol": {
            "operation": "GPUReplayBuffer.add_batch or ReplayBuffer.add_batch",
            "seed": args.seed,
            "synchronize_cuda_before_and_after_measurement": device.type == "cuda",
            "contract_path": args.contract.as_posix(),
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
