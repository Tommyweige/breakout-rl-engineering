"""Benchmark double-buffered CUDA-copy-stream prefetch against DQN compute."""

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


def _make_replay(capacity: int) -> ReplayBuffer:
    replay = ReplayBuffer(capacity)
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.uint8)
    next_state = np.full(OBSERVATION_SHAPE, 127, dtype=np.uint8)
    for index in range(capacity):
        replay.add(state, index % 4, float(index % 3), next_state, False, False)
    return replay


def _host_batch(batch_size: int, sampled: Any) -> dict[str, torch.Tensor]:
    host = {
        "states": torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.uint8, pin_memory=True),
        "next_states": torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.uint8, pin_memory=True),
        "actions": torch.empty((batch_size,), dtype=torch.long, pin_memory=True),
        "rewards": torch.empty((batch_size,), dtype=torch.float32, pin_memory=True),
        "terminated": torch.empty((batch_size,), dtype=torch.bool, pin_memory=True),
        "truncated": torch.empty((batch_size,), dtype=torch.bool, pin_memory=True),
    }
    host["states"].copy_(torch.from_numpy(sampled.states))
    host["next_states"].copy_(torch.from_numpy(sampled.next_states))
    host["actions"].copy_(torch.from_numpy(sampled.actions))
    host["rewards"].copy_(torch.from_numpy(sampled.rewards))
    host["terminated"].copy_(torch.from_numpy(sampled.terminated))
    host["truncated"].copy_(torch.from_numpy(sampled.truncated))
    return host


def _device_batch(batch_size: int, device: torch.device) -> ReplayTensorBatch:
    return ReplayTensorBatch(
        states=torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.float32, device=device),
        actions=torch.empty((batch_size,), dtype=torch.long, device=device),
        rewards=torch.empty((batch_size,), dtype=torch.float32, device=device),
        next_states=torch.empty((batch_size, *OBSERVATION_SHAPE), dtype=torch.float32, device=device),
        terminated=torch.empty((batch_size,), dtype=torch.bool, device=device),
        truncated=torch.empty((batch_size,), dtype=torch.bool, device=device),
    )


def _enqueue_copy(
    copy_stream: torch.cuda.Stream,
    host: dict[str, torch.Tensor],
    device_batch: ReplayTensorBatch,
    ready_event: torch.cuda.Event,
    done_event: torch.cuda.Event | None,
) -> None:
    with torch.cuda.stream(copy_stream):
        if done_event is not None:
            copy_stream.wait_event(done_event)
        device_batch.states.copy_(host["states"], non_blocking=True).div_(255.0)
        device_batch.next_states.copy_(host["next_states"], non_blocking=True).div_(255.0)
        device_batch.actions.copy_(host["actions"], non_blocking=True)
        device_batch.rewards.copy_(host["rewards"], non_blocking=True)
        device_batch.terminated.copy_(host["terminated"], non_blocking=True)
        device_batch.truncated.copy_(host["truncated"], non_blocking=True)
        ready_event.record(copy_stream)


def _run(batch_size: int, *, iterations: int, device: torch.device) -> dict[str, Any]:
    torch.set_num_threads(2)
    replay = _make_replay(10000)
    rng = np.random.default_rng(42)
    hosts = [
        _host_batch(batch_size, replay.sample(batch_size, rng)),
        _host_batch(batch_size, replay.sample(batch_size, rng)),
    ]
    device_batches = [_device_batch(batch_size, device), _device_batch(batch_size, device)]
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device)
    ready_events = [torch.cuda.Event(), torch.cuda.Event()]
    done_events: list[torch.cuda.Event | None] = [None, None]
    online = DQNNetwork(4).to(device)
    target = DQNNetwork(4).to(device)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)

    def schedule_updates(count: int) -> None:
        for index in range(count):
            slot = index % 2
            _enqueue_copy(
                copy_stream,
                hosts[slot],
                device_batches[slot],
                ready_events[slot],
                done_events[slot],
            )
            compute_stream.wait_event(ready_events[slot])
            dqn_training_step(
                online,
                target,
                optimizer,
                device_batches[slot],
                gamma=0.99,
                gradient_clip_norm=10.0,
                collect_diagnostics=False,
            )
            done_events[slot] = torch.cuda.Event()
            done_events[slot].record(compute_stream)

    schedule_updates(2)
    compute_stream.synchronize()
    copy_stream.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    schedule_updates(iterations)
    compute_stream.synchronize()
    copy_stream.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "overlapped_ms_per_update": elapsed / iterations * 1000.0,
        "training_samples_per_second": batch_size * iterations / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "cpu_sampling_and_host_staging_included": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=10)
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
