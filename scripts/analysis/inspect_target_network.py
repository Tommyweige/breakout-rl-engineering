"""Inspect hard target synchronization with a real DQN forward pass."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from breakout_rl.models import DQNNetwork
from breakout_rl.targets import hard_update


DEFAULT_NUM_ACTIONS = 4
DEFAULT_BATCH_SIZE = 2
DEFAULT_INPUT_SHAPE = (4, 84, 84)


@dataclass(frozen=True)
class TargetNetworkInspection:
    """Runtime evidence for one online-update and target-sync sequence."""

    device: torch.device
    seed: int
    batch_shape: tuple[int, ...]
    num_actions: int
    phase_names: tuple[str, ...]
    online_sample_outputs: tuple[tuple[float, ...], ...]
    target_sample_outputs: tuple[tuple[float, ...], ...]
    max_abs_diffs: tuple[float, ...]
    update_loss: float
    online_parameter_delta_max_abs: float
    target_parameter_delta_after_online_update: float


def resolve_device(device_name: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` without hiding CUDA errors."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but it is not available in this environment."
            )
        return torch.device("cuda")
    raise ValueError("device_name must be one of: auto, cpu, cuda")


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the maximum absolute elementwise difference as a Python float."""

    return float((left - right).abs().max().item())


def _sample_output(values: torch.Tensor) -> tuple[float, ...]:
    """Return the first sample's Q-values for compact inspection output."""

    return tuple(float(value) for value in values[0].detach().cpu().tolist())


def collect_target_network_inspection(
    *,
    seed: int = 42,
    device_name: str = "auto",
) -> TargetNetworkInspection:
    """Run a deterministic synthetic batch through independent DQN instances."""

    device = resolve_device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    batch_shape = (DEFAULT_BATCH_SIZE, *DEFAULT_INPUT_SHAPE)
    states = torch.linspace(
        0.0,
        1.0,
        steps=DEFAULT_BATCH_SIZE * 4 * 84 * 84,
        dtype=torch.float32,
        device=device,
    ).reshape(batch_shape)

    online_network = DQNNetwork(num_actions=DEFAULT_NUM_ACTIONS).to(device)
    target_network = DQNNetwork(num_actions=DEFAULT_NUM_ACTIONS).to(device)
    online_network.train()
    target_network.eval()
    hard_update(target_network, online_network)

    with torch.inference_mode():
        online_before = online_network(states)
        target_before = target_network(states)

    online_parameters_before = [
        parameter.detach().clone() for parameter in online_network.parameters()
    ]
    target_parameters_before = [
        parameter.detach().clone() for parameter in target_network.parameters()
    ]

    optimizer = torch.optim.SGD(online_network.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss = online_network(states).mean()
    loss.backward()
    optimizer.step()

    with torch.inference_mode():
        online_after_update = online_network(states)
        target_after_update = target_network(states)

    online_parameter_delta = max(
        (parameter.detach() - before).abs().max().item()
        for parameter, before in zip(
            online_network.parameters(),
            online_parameters_before,
            strict=True,
        )
    )
    target_parameter_delta = max(
        (parameter.detach() - before).abs().max().item()
        for parameter, before in zip(
            target_network.parameters(),
            target_parameters_before,
            strict=True,
        )
    )

    hard_update(target_network, online_network)
    with torch.inference_mode():
        online_after_sync = online_network(states)
        target_after_sync = target_network(states)

    phase_names = (
        "after initial hard sync",
        "after online optimizer step",
        "after hard sync",
    )
    online_outputs = (online_before, online_after_update, online_after_sync)
    target_outputs = (target_before, target_after_update, target_after_sync)

    return TargetNetworkInspection(
        device=device,
        seed=seed,
        batch_shape=batch_shape,
        num_actions=DEFAULT_NUM_ACTIONS,
        phase_names=phase_names,
        online_sample_outputs=tuple(_sample_output(values) for values in online_outputs),
        target_sample_outputs=tuple(_sample_output(values) for values in target_outputs),
        max_abs_diffs=tuple(
            _max_abs_difference(online_values, target_values)
            for online_values, target_values in zip(
                online_outputs,
                target_outputs,
                strict=True,
            )
        ),
        update_loss=float(loss.detach().cpu().item()),
        online_parameter_delta_max_abs=float(online_parameter_delta),
        target_parameter_delta_after_online_update=float(target_parameter_delta),
    )


def parse_args() -> argparse.Namespace:
    """Parse inspection CLI options."""

    parser = argparse.ArgumentParser(
        description="Inspect online and target DQN synchronization behavior."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="forward device; auto selects CUDA when available (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for model initialization and synthetic input (default: 42)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="optional path for machine-readable inspection output",
    )
    return parser.parse_args()


def print_inspection(inspection: TargetNetworkInspection) -> None:
    """Print the evidence used by the Day 11 article and visualization."""

    print(f"Device              : {inspection.device}")
    print(f"Seed                : {inspection.seed}")
    print(f"Synthetic batch     : {inspection.batch_shape} float32")
    print(f"Action count        : {inspection.num_actions}")
    print(f"Optimizer loss      : {inspection.update_loss:+.8f}")
    print(
        "Online parameter Δ  : "
        f"{inspection.online_parameter_delta_max_abs:.8f}"
    )
    print(
        "Target parameter Δ  : "
        f"{inspection.target_parameter_delta_after_online_update:.8f}"
    )
    for phase, online_values, target_values in zip(
        inspection.phase_names,
        inspection.online_sample_outputs,
        inspection.target_sample_outputs,
        strict=True,
    ):
        print(f"{phase} online Q[0] : " + "[" + ", ".join(f"{value:+.6f}" for value in online_values) + "]")
        print(f"{phase} target Q[0] : " + "[" + ", ".join(f"{value:+.6f}" for value in target_values) + "]")

    print(f"before online update: max abs diff = {inspection.max_abs_diffs[0]:.8f}")
    print(f"after online update : max abs diff = {inspection.max_abs_diffs[1]:.8f}")
    print(f"after target sync   : max abs diff = {inspection.max_abs_diffs[2]:.8f}")


def inspection_to_dict(inspection: TargetNetworkInspection) -> dict[str, object]:
    """Serialize runtime evidence for the plotting script."""

    return {
        "device": str(inspection.device),
        "seed": inspection.seed,
        "batch_shape": list(inspection.batch_shape),
        "num_actions": inspection.num_actions,
        "phase_names": list(inspection.phase_names),
        "online_sample_outputs": [
            list(values) for values in inspection.online_sample_outputs
        ],
        "target_sample_outputs": [
            list(values) for values in inspection.target_sample_outputs
        ],
        "max_abs_diffs": list(inspection.max_abs_diffs),
        "update_loss": inspection.update_loss,
        "online_parameter_delta_max_abs": inspection.online_parameter_delta_max_abs,
        "target_parameter_delta_after_online_update": (
            inspection.target_parameter_delta_after_online_update
        ),
    }


def write_inspection_json(
    output: Path,
    inspection: TargetNetworkInspection,
) -> None:
    """Write runtime evidence without rounding the values used by a plot."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(inspection_to_dict(inspection), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    try:
        inspection = collect_target_network_inspection(
            seed=args.seed,
            device_name=args.device,
        )
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error
    print_inspection(inspection)
    if args.json_output is not None:
        write_inspection_json(args.json_output, inspection)
        print(f"Saved JSON: {args.json_output}")


if __name__ == "__main__":
    main()
