"""Crafted, machine-readable target-rule comparisons for Day 17."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from breakout_rl.targets import compute_double_dqn_targets, compute_dqn_targets


class FixedQNetwork(nn.Module):
    """Return one fixed Q-vector for every input row."""

    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(states.shape[0], -1)


def build_target_comparison(*, seed: int = 42) -> dict[str, Any]:
    """Compute both target rules on deliberately different estimators."""

    torch.manual_seed(seed)
    online_values = [1.0, 5.0, 2.0, 0.0]
    target_values = [4.0, 3.0, 2.0, 1.0]
    rewards = torch.tensor([1.0], dtype=torch.float32)
    next_states = torch.zeros((1, 1), dtype=torch.float32)
    terminated = torch.tensor([False])
    online = FixedQNetwork(online_values)
    target = FixedQNetwork(target_values)

    vanilla_target = compute_dqn_targets(
        rewards,
        next_states,
        terminated,
        target,
        gamma=0.5,
    )
    double_target = compute_double_dqn_targets(
        rewards,
        next_states,
        terminated,
        online,
        target,
        gamma=0.5,
    )
    selected_action = int(torch.tensor(online_values).argmax().item())
    return {
        "schema_version": 1,
        "seed": int(seed),
        "fixture": {
            "rewards": [1.0],
            "terminated": [False],
            "gamma": 0.5,
            "online_next_q": online_values,
            "target_next_q": target_values,
        },
        "vanilla": {
            "selected_action": int(torch.tensor(target_values).argmax().item()),
            "evaluated_next_value": float(max(target_values)),
            "final_target": float(vanilla_target.item()),
        },
        "double_dqn": {
            "selected_action": selected_action,
            "evaluated_next_value": float(target_values[selected_action]),
            "final_target": float(double_target.item()),
        },
    }


__all__ = ["FixedQNetwork", "build_target_comparison"]
