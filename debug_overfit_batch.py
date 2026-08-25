"""Run the fixed-batch DQN sanity check with a real observation contract."""

from __future__ import annotations

import argparse
import json

import torch

from breakout_rl.models.dqn import DQNNetwork
from breakout_rl.replay_tensors import ReplayTensorBatch
from breakout_rl.training.diagnostics import run_fixed_batch_overfit
from breakout_rl.training.dqn_trainer import seed_everything


OBSERVATION_SHAPE = (4, 84, 84)
ACTION_COUNT = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overfit one fixed DQN batch with frozen target values."
    )
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    return device


def run(
    *,
    updates: int,
    device: str,
    seed: int,
    batch_size: int,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    resolved_device = _resolve_device(device)
    seed_everything(seed)

    model = DQNNetwork(
        ACTION_COUNT,
        input_shape=OBSERVATION_SHAPE,
        hidden_dim=64,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    states = torch.rand(
        batch_size,
        *OBSERVATION_SHAPE,
        dtype=torch.float32,
        device=resolved_device,
    )
    actions = torch.arange(batch_size, device=resolved_device, dtype=torch.long)
    actions = actions.remainder(ACTION_COUNT)
    batch = ReplayTensorBatch(
        states=states,
        actions=actions,
        rewards=torch.zeros(batch_size, device=resolved_device),
        next_states=torch.zeros_like(states),
        terminated=torch.ones(batch_size, dtype=torch.bool, device=resolved_device),
        truncated=torch.zeros(batch_size, dtype=torch.bool, device=resolved_device),
    )
    # These values are intentionally independent of the model and never
    # recomputed inside the update loop: moving targets would hide the sanity
    # check's answer.
    fixed_targets = torch.linspace(
        -1.0,
        1.0,
        batch_size,
        dtype=torch.float32,
        device=resolved_device,
    )
    result = run_fixed_batch_overfit(
        model,
        optimizer,
        batch,
        fixed_targets,
        updates=updates,
    )
    return {
        "device": str(resolved_device),
        "seed": seed,
        "batch_size": batch_size,
        "target_values_fixed": True,
        **result.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(
            updates=args.updates,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"Fixed-batch sanity check failed to start: {error}")
        return 2

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
