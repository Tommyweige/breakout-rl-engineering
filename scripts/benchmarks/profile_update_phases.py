"""Profile individual GPU phases of one DQN update for a large batch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.replay import ReplayBuffer
from breakout_rl.replay_tensors import replay_batch_to_tensors
from breakout_rl.training.dqn_trainer import dqn_training_step, resolve_device


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _make_replay(capacity: int) -> ReplayBuffer:
    replay = ReplayBuffer(capacity)
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    next_state = np.full((4, 84, 84), 127, dtype=np.uint8)
    for index in range(capacity):
        replay.add(state, index % 4, float(index % 3), next_state, False, False)
    return replay


def _timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    _sync(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    return value, time.perf_counter() - start


def _run(batch_size: int, *, repeats: int, device: torch.device) -> dict[str, Any]:
    replay = _make_replay(10000)
    sampled = replay.sample(batch_size, np.random.default_rng(42))
    batch = replay_batch_to_tensors(sampled, device=device)
    online = DQNNetwork(4).to(device)
    target = DQNNetwork(4).to(device)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)

    phases: dict[str, list[float]] = {
        "online_forward_ms": [],
        "target_forward_ms": [],
        "backward_ms": [],
        "gradient_clip_ms": [],
        "optimizer_step_ms": [],
    }

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        q_values = online(batch.states)
        with torch.no_grad():
            next_values = target(batch.next_states)
        selected = q_values.gather(1, batch.actions[:, None]).squeeze(1)
        targets = batch.rewards + 0.99 * next_values.max(dim=1).values
        loss = torch.nn.functional.smooth_l1_loss(selected, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(online.parameters()), max_norm=10.0)
        optimizer.step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        q_values, elapsed = _timed(device, lambda: online(batch.states))
        phases["online_forward_ms"].append(elapsed * 1000.0)

        next_values, elapsed = _timed(
            device,
            lambda: target(batch.next_states),
        )
        phases["target_forward_ms"].append(elapsed * 1000.0)

        selected = q_values.gather(1, batch.actions[:, None]).squeeze(1)
        with torch.no_grad():
            targets = batch.rewards + 0.99 * next_values.max(dim=1).values
        loss = torch.nn.functional.smooth_l1_loss(selected, targets)

        _, elapsed = _timed(device, loss.backward)
        phases["backward_ms"].append(elapsed * 1000.0)

        _, elapsed = _timed(
            device,
            lambda: torch.nn.utils.clip_grad_norm_(
                list(online.parameters()),
                max_norm=10.0,
            ),
        )
        phases["gradient_clip_ms"].append(elapsed * 1000.0)

        _, elapsed = _timed(device, optimizer.step)
        phases["optimizer_step_ms"].append(elapsed * 1000.0)

    actual_dqn_ms: list[float] = []
    for _ in range(repeats):
        _, elapsed = _timed(
            device,
            lambda: dqn_training_step(
                online,
                target,
                optimizer,
                batch,
                gamma=0.99,
                gradient_clip_norm=10.0,
                collect_diagnostics=False,
            ),
        )
        actual_dqn_ms.append(elapsed * 1000.0)

    result: dict[str, Any] = {
        "batch_size": batch_size,
        "repeats": repeats,
        "phases_ms": {
            name: sum(values) / len(values) for name, values in phases.items()
        },
        "actual_dqn_training_step_ms": sum(actual_dqn_ms) / len(actual_dqn_ms),
    }
    result["peak_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    result["peak_reserved_bytes"] = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.repeats < 1:
        raise ValueError("batch size and repeats must be positive")
    device = resolve_device(args.device)
    report = {
        "schema_version": 1,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "result": _run(args.batch_size, repeats=args.repeats, device=device),
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
