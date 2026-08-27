"""Compare direct NumPy-to-CUDA transfer with pinned host staging."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from breakout_rl.replay import ReplayBuffer
from breakout_rl.replay_tensors import ReplayTensorBatch, replay_batch_to_tensors
from breakout_rl.tensors import OBSERVATION_SHAPE
from breakout_rl.training.dqn_trainer import resolve_device


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _make_replay(capacity: int) -> ReplayBuffer:
    replay = ReplayBuffer(capacity)
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.uint8)
    next_state = np.full(OBSERVATION_SHAPE, 127, dtype=np.uint8)
    for index in range(capacity):
        replay.add(state, index % 4, float(index % 3), next_state, False, False)
    return replay


def _pinned_staging(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "states": torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.uint8, pin_memory=True),
        "next_states": torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.uint8, pin_memory=True),
        "actions": torch.empty((batch_size,), dtype=torch.long, pin_memory=True),
        "rewards": torch.empty((batch_size,), dtype=torch.float32, pin_memory=True),
        "terminated": torch.empty((batch_size,), dtype=torch.bool, pin_memory=True),
        "truncated": torch.empty((batch_size,), dtype=torch.bool, pin_memory=True),
    }


def _stage(staging: dict[str, torch.Tensor], sampled: Any, device: torch.device) -> ReplayTensorBatch:
    staging["states"].copy_(torch.from_numpy(sampled.states))
    staging["next_states"].copy_(torch.from_numpy(sampled.next_states))
    staging["actions"].copy_(torch.from_numpy(sampled.actions))
    staging["rewards"].copy_(torch.from_numpy(sampled.rewards))
    staging["terminated"].copy_(torch.from_numpy(sampled.terminated))
    staging["truncated"].copy_(torch.from_numpy(sampled.truncated))
    return ReplayTensorBatch(
        states=staging["states"].to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0),
        actions=staging["actions"].to(device=device, non_blocking=True),
        rewards=staging["rewards"].to(device=device, non_blocking=True),
        next_states=staging["next_states"].to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0),
        terminated=staging["terminated"].to(device=device, non_blocking=True),
        truncated=staging["truncated"].to(device=device, non_blocking=True),
    )


def _run(batch_size: int, *, replay: ReplayBuffer, device: torch.device, iterations: int) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    staging = _pinned_staging(batch_size)
    direct_seconds = 0.0
    staged_seconds = 0.0
    pinned_transfer_only_seconds = 0.0

    sampled = replay.sample(batch_size, rng)
    staging["states"].copy_(torch.from_numpy(sampled.states))
    staging["next_states"].copy_(torch.from_numpy(sampled.next_states))
    staging["actions"].copy_(torch.from_numpy(sampled.actions))
    staging["rewards"].copy_(torch.from_numpy(sampled.rewards))
    staging["terminated"].copy_(torch.from_numpy(sampled.terminated))
    staging["truncated"].copy_(torch.from_numpy(sampled.truncated))

    for _ in range(3):
        replay_batch_to_tensors(replay.sample(batch_size, rng), device=device)
        _stage(staging, replay.sample(batch_size, rng), device)
    _sync(device)

    for _ in range(iterations):
        sampled = replay.sample(batch_size, rng)
        _sync(device)
        start = time.perf_counter()
        replay_batch_to_tensors(sampled, device=device)
        _sync(device)
        direct_seconds += time.perf_counter() - start

        sampled = replay.sample(batch_size, rng)
        _sync(device)
        start = time.perf_counter()
        _stage(staging, sampled, device)
        _sync(device)
        staged_seconds += time.perf_counter() - start

        _sync(device)
        start = time.perf_counter()
        ReplayTensorBatch(
            states=staging["states"].to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0),
            actions=staging["actions"].to(device=device, non_blocking=True),
            rewards=staging["rewards"].to(device=device, non_blocking=True),
            next_states=staging["next_states"].to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0),
            terminated=staging["terminated"].to(device=device, non_blocking=True),
            truncated=staging["truncated"].to(device=device, non_blocking=True),
        )
        _sync(device)
        pinned_transfer_only_seconds += time.perf_counter() - start

    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "direct_numpy_to_cuda_ms": direct_seconds / iterations * 1000.0,
        "pinned_stage_plus_cuda_ms": staged_seconds / iterations * 1000.0,
        "pinned_transfer_only_ms": pinned_transfer_only_seconds / iterations * 1000.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    replay = _make_replay(args.replay_capacity)
    report = {
        "schema_version": 1,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "results": [
            _run(batch, replay=replay, device=device, iterations=args.iterations)
            for batch in args.batch_sizes
        ],
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
