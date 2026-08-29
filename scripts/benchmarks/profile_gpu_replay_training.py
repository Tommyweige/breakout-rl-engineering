"""Benchmark DQN updates with a uint8 replay buffer resident on the GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.replay_tensors import ReplayTensorBatch
from breakout_rl.training.dqn_trainer import dqn_training_step, resolve_device


OBSERVATION_SHAPE = (4, 84, 84)
CAPACITY = 10_000


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _run(batch_size: int, *, iterations: int, device: torch.device) -> dict[str, Any]:
    torch.set_num_threads(2)
    replay_states = torch.randint(
        0,
        256,
        (CAPACITY, *OBSERVATION_SHAPE),
        dtype=torch.uint8,
        device=device,
    )
    replay_next_states = torch.randint(
        0,
        256,
        (CAPACITY, *OBSERVATION_SHAPE),
        dtype=torch.uint8,
        device=device,
    )
    replay_actions = torch.randint(0, 4, (CAPACITY,), dtype=torch.long, device=device)
    replay_rewards = torch.rand((CAPACITY,), dtype=torch.float32, device=device)
    replay_terminated = torch.zeros((CAPACITY,), dtype=torch.bool, device=device)
    replay_truncated = torch.zeros((CAPACITY,), dtype=torch.bool, device=device)

    gathered_states = torch.empty(
        (batch_size, *OBSERVATION_SHAPE), dtype=torch.uint8, device=device
    )
    gathered_next_states = torch.empty_like(gathered_states)
    gathered_actions = torch.empty((batch_size,), dtype=torch.long, device=device)
    gathered_rewards = torch.empty((batch_size,), dtype=torch.float32, device=device)
    gathered_terminated = torch.empty((batch_size,), dtype=torch.bool, device=device)
    gathered_truncated = torch.empty((batch_size,), dtype=torch.bool, device=device)
    batch = ReplayTensorBatch(
        states=torch.empty(
            (batch_size, *OBSERVATION_SHAPE), dtype=torch.float32, device=device
        ),
        actions=torch.empty((batch_size,), dtype=torch.long, device=device),
        rewards=torch.empty((batch_size,), dtype=torch.float32, device=device),
        next_states=torch.empty(
            (batch_size, *OBSERVATION_SHAPE), dtype=torch.float32, device=device
        ),
        terminated=torch.empty((batch_size,), dtype=torch.bool, device=device),
        truncated=torch.empty((batch_size,), dtype=torch.bool, device=device),
    )

    online = DQNNetwork(4).to(device)
    target = DQNNetwork(4).to(device)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)

    def fill_batch() -> None:
        indices = torch.randperm(CAPACITY, device=device)[:batch_size]
        torch.index_select(replay_states, 0, indices, out=gathered_states)
        torch.index_select(replay_next_states, 0, indices, out=gathered_next_states)
        torch.index_select(replay_actions, 0, indices, out=gathered_actions)
        torch.index_select(replay_rewards, 0, indices, out=gathered_rewards)
        torch.index_select(replay_terminated, 0, indices, out=gathered_terminated)
        torch.index_select(replay_truncated, 0, indices, out=gathered_truncated)
        batch.states.copy_(gathered_states).div_(255.0)
        batch.next_states.copy_(gathered_next_states).div_(255.0)
        batch.actions.copy_(gathered_actions)
        batch.rewards.copy_(gathered_rewards)
        batch.terminated.copy_(gathered_terminated)
        batch.truncated.copy_(gathered_truncated)

    def update() -> None:
        dqn_training_step(
            online,
            target,
            optimizer,
            batch,
            gamma=0.99,
            gradient_clip_norm=10.0,
            collect_diagnostics=False,
        )

    for _ in range(3):
        fill_batch()
        update()
    _sync(device)
    torch.cuda.reset_peak_memory_stats(device)

    gather_seconds = 0.0
    update_seconds = 0.0
    for _ in range(iterations):
        start = time.perf_counter()
        fill_batch()
        _sync(device)
        gather_seconds += time.perf_counter() - start

        start = time.perf_counter()
        update()
        _sync(device)
        update_seconds += time.perf_counter() - start

    total_seconds = gather_seconds + update_seconds
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "gpu_replay_gather_cast_ms_per_update": gather_seconds / iterations * 1000.0,
        "actual_dqn_update_ms_per_update": update_seconds / iterations * 1000.0,
        "total_ms_per_update": total_seconds / iterations * 1000.0,
        "training_samples_per_second": batch_size * iterations / total_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
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
