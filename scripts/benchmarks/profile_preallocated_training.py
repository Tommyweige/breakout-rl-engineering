"""Measure DQN updates with pinned host staging and reusable GPU buffers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.replay import ReplayBuffer
from breakout_rl.replay_tensors import ReplayTensorBatch
from breakout_rl.tensors import OBSERVATION_SHAPE
from breakout_rl.training.dqn_trainer import dqn_training_step, resolve_device


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _make_replay(capacity: int) -> ReplayBuffer:
    replay = ReplayBuffer(capacity)
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.uint8)
    next_state = np.full(OBSERVATION_SHAPE, 127, dtype=np.uint8)
    for index in range(capacity):
        replay.add(state, index % 4, float(index % 3), next_state, False, False)
    return replay


def _make_host_staging(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "states": torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.uint8, pin_memory=True),
        "next_states": torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.uint8, pin_memory=True),
        "actions": torch.empty((batch_size,), dtype=torch.long, pin_memory=True),
        "rewards": torch.empty((batch_size,), dtype=torch.float32, pin_memory=True),
        "terminated": torch.empty((batch_size,), dtype=torch.bool, pin_memory=True),
        "truncated": torch.empty((batch_size,), dtype=torch.bool, pin_memory=True),
    }


def _make_gpu_batch(batch_size: int, device: torch.device) -> ReplayTensorBatch:
    return ReplayTensorBatch(
        states=torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.float32, device=device),
        actions=torch.empty((batch_size,), dtype=torch.long, device=device),
        rewards=torch.empty((batch_size,), dtype=torch.float32, device=device),
        next_states=torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.float32, device=device),
        terminated=torch.empty((batch_size,), dtype=torch.bool, device=device),
        truncated=torch.empty((batch_size,), dtype=torch.bool, device=device),
    )


def _stage_numpy(host: dict[str, torch.Tensor], sampled: Any) -> None:
    host["states"].copy_(torch.from_numpy(sampled.states))
    host["next_states"].copy_(torch.from_numpy(sampled.next_states))
    host["actions"].copy_(torch.from_numpy(sampled.actions))
    host["rewards"].copy_(torch.from_numpy(sampled.rewards))
    host["terminated"].copy_(torch.from_numpy(sampled.terminated))
    host["truncated"].copy_(torch.from_numpy(sampled.truncated))


def _copy_to_gpu(host: dict[str, torch.Tensor], gpu: ReplayTensorBatch) -> None:
    gpu.states.copy_(host["states"], non_blocking=True).div_(255.0)
    gpu.next_states.copy_(host["next_states"], non_blocking=True).div_(255.0)
    gpu.actions.copy_(host["actions"], non_blocking=True)
    gpu.rewards.copy_(host["rewards"], non_blocking=True)
    gpu.terminated.copy_(host["terminated"], non_blocking=True)
    gpu.truncated.copy_(host["truncated"], non_blocking=True)


def _run(batch_size: int, *, iterations: int, device: torch.device) -> dict[str, Any]:
    torch.set_num_threads(2)
    replay = _make_replay(10000)
    host = _make_host_staging(batch_size)
    gpu_batch = _make_gpu_batch(batch_size, device)
    online = DQNNetwork(4).to(device)
    target = DQNNetwork(4).to(device)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    rng = np.random.default_rng(42)

    def one_update() -> None:
        sampled = replay.sample(batch_size, rng)
        _stage_numpy(host, sampled)
        _sync(device)
        _copy_to_gpu(host, gpu_batch)
        _sync(device)
        dqn_training_step(
            online,
            target,
            optimizer,
            gpu_batch,
            gamma=0.99,
            gradient_clip_norm=10.0,
            collect_diagnostics=False,
        )

    for _ in range(2):
        one_update()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stage_ms = 0.0
    copy_ms = 0.0
    update_ms = 0.0
    sample_ms = 0.0
    for _ in range(iterations):
        start = time.perf_counter()
        sampled = replay.sample(batch_size, rng)
        sample_ms += time.perf_counter() - start

        start = time.perf_counter()
        _stage_numpy(host, sampled)
        stage_ms += time.perf_counter() - start

        _sync(device)
        start = time.perf_counter()
        _copy_to_gpu(host, gpu_batch)
        _sync(device)
        copy_ms += time.perf_counter() - start

        start = time.perf_counter()
        dqn_training_step(
            online,
            target,
            optimizer,
            gpu_batch,
            gamma=0.99,
            gradient_clip_norm=10.0,
            collect_diagnostics=False,
        )
        _sync(device)
        update_ms += time.perf_counter() - start

    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "sample_ms_per_update": sample_ms / iterations * 1000.0,
        "numpy_to_pinned_stage_ms_per_update": stage_ms / iterations * 1000.0,
        "pinned_to_gpu_ms_per_update": copy_ms / iterations * 1000.0,
        "actual_dqn_update_ms_per_update": update_ms / iterations * 1000.0,
        "total_ms_per_update": (sample_ms + stage_ms + copy_ms + update_ms) / iterations * 1000.0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.iterations < 1:
        raise ValueError("batch size and iterations must be positive")
    device = resolve_device(args.device)
    report = {
        "schema_version": 1,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "result": _run(args.batch_size, iterations=args.iterations, device=device),
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
