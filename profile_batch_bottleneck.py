"""Break down replay, host-to-device, and optimizer time by batch size."""

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
from breakout_rl.replay_tensors import replay_batch_to_tensors
from breakout_rl.training.dqn_trainer import dqn_training_step, resolve_device


OBSERVATION_SHAPE = (4, 84, 84)
ACTION_COUNT = 4


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_replay(capacity: int) -> ReplayBuffer:
    replay = ReplayBuffer(capacity, observation_shape=OBSERVATION_SHAPE)
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.uint8)
    next_state = np.full(OBSERVATION_SHAPE, 127, dtype=np.uint8)
    for index in range(capacity):
        replay.add(
            state,
            index % ACTION_COUNT,
            float(index % 3),
            next_state,
            False,
            False,
        )
    return replay


def _run_batch(
    batch_size: int,
    *,
    replay: ReplayBuffer,
    device: torch.device,
    warmup_updates: int,
    measured_updates: int,
    diagnostics_updates: int,
    cpu_threads: int,
) -> dict[str, Any]:
    torch.set_num_threads(cpu_threads)
    online = DQNNetwork(ACTION_COUNT).to(device)
    target = DQNNetwork(ACTION_COUNT).to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    rng = np.random.default_rng(42)

    for _ in range(warmup_updates):
        sampled = replay.sample(batch_size, rng)
        tensor_batch = replay_batch_to_tensors(sampled, device=device)
        dqn_training_step(
            online,
            target,
            optimizer,
            tensor_batch,
            gamma=0.99,
            gradient_clip_norm=10.0,
            collect_diagnostics=False,
        )
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sample_wall = 0.0
    sample_cpu = 0.0
    transfer_wall = 0.0
    transfer_cpu = 0.0
    update_wall = 0.0
    update_cpu = 0.0
    measured_start = time.perf_counter()
    for _ in range(measured_updates):
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        sampled = replay.sample(batch_size, rng)
        sample_wall += time.perf_counter() - wall_start
        sample_cpu += time.process_time() - cpu_start

        _sync(device)
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        tensor_batch = replay_batch_to_tensors(sampled, device=device)
        _sync(device)
        transfer_wall += time.perf_counter() - wall_start
        transfer_cpu += time.process_time() - cpu_start

        _sync(device)
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        dqn_training_step(
            online,
            target,
            optimizer,
            tensor_batch,
            gamma=0.99,
            gradient_clip_norm=10.0,
            collect_diagnostics=False,
        )
        _sync(device)
        update_wall += time.perf_counter() - wall_start
        update_cpu += time.process_time() - cpu_start
    measured_wall = time.perf_counter() - measured_start

    diagnostic_wall = 0.0
    if diagnostics_updates:
        _sync(device)
        start = time.perf_counter()
        for _ in range(diagnostics_updates):
            sampled = replay.sample(batch_size, rng)
            tensor_batch = replay_batch_to_tensors(sampled, device=device)
            dqn_training_step(
                online,
                target,
                optimizer,
                tensor_batch,
                gamma=0.99,
                gradient_clip_norm=10.0,
                collect_diagnostics=True,
            )
        _sync(device)
        diagnostic_wall = time.perf_counter() - start

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    def average(value: float) -> float:
        return value / measured_updates

    return {
        "batch_size": batch_size,
        "measured_updates": measured_updates,
        "warmup_updates": warmup_updates,
        "sample_wall_ms_per_update": average(sample_wall) * 1000.0,
        "sample_cpu_ms_per_update": average(sample_cpu) * 1000.0,
        "h2d_wall_ms_per_update": average(transfer_wall) * 1000.0,
        "h2d_cpu_ms_per_update": average(transfer_cpu) * 1000.0,
        "pure_update_wall_ms_per_update": average(update_wall) * 1000.0,
        "pure_update_cpu_ms_per_update": average(update_cpu) * 1000.0,
        "measured_total_ms_per_update": measured_wall / measured_updates * 1000.0,
        "pure_update_per_second": measured_updates / update_wall,
        "diagnostic_update_ms_per_update": (
            diagnostic_wall / diagnostics_updates * 1000.0
            if diagnostics_updates
            else None
        ),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "sampled_states_bytes_per_update": batch_size
        * int(np.prod(OBSERVATION_SHAPE))
        * 2,
        "float32_states_bytes_per_update": batch_size
        * int(np.prod(OBSERVATION_SHAPE))
        * 2
        * 4,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--measured-updates", type=int, default=20)
    parser.add_argument("--diagnostics-updates", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(batch < 1 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if args.measured_updates < 1 or args.warmup_updates < 0 or args.diagnostics_updates < 0:
        raise ValueError("update counts are invalid")

    device = resolve_device(args.device)
    replay = _make_replay(args.replay_capacity)
    results = [
        _run_batch(
            batch,
            replay=replay,
            device=device,
            warmup_updates=args.warmup_updates,
            measured_updates=args.measured_updates,
            diagnostics_updates=args.diagnostics_updates,
            cpu_threads=args.cpu_threads,
        )
        for batch in args.batch_sizes
    ]
    report = {
        "schema_version": 1,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "torch_version": torch.__version__,
        "cpu_threads": args.cpu_threads,
        "replay_capacity": args.replay_capacity,
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
