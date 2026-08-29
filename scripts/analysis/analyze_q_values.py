"""Analyze a DQN checkpoint on frozen Contract v2 probe states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from breakout_rl.analysis.q_values import (
    analyze_q_values as analyze_model_q_values,
    load_probe_states,
)
from breakout_rl.models.dqn import DQNNetwork


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    return payload


def load_checkpoint_model(
    path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a standard DQN checkpoint without changing its training state."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _load_payload(source)
    state_dict = payload.get("online_network")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint does not contain online_network")
    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    model_config = payload.get("model_config", {})
    if not isinstance(model_config, Mapping):
        model_config = {}
    raw_shape = model_config.get("input_shape", (4, 84, 84))
    input_shape = tuple(int(value) for value in raw_shape)
    num_actions = int(model_config.get("num_actions", 4))
    hidden_dim_value = model_config.get("hidden_dim", 512)
    hidden_dim = 512 if hidden_dim_value is None else int(hidden_dim_value)
    model = DQNNetwork(
        num_actions,
        input_shape=input_shape,  # type: ignore[arg-type]
        hidden_dim=hidden_dim,
    ).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint does not match the standard DQN architecture") from error
    model.eval()
    metadata = {
        "path": source.as_posix(),
        "sha256": _sha256(source),
        "algorithm": payload.get("algorithm", config.get("algorithm", "dqn")),
        "architecture": payload.get("architecture", model_config.get("architecture", "standard")),
        "num_envs": payload.get("num_envs", config.get("num_envs", 1)),
        "replay_backend": payload.get("replay_backend", config.get("replay_backend", "cpu")),
        "training_steps": payload.get("training_steps", payload.get("global_step")),
        "model_config": {
            "num_actions": num_actions,
            "input_shape": list(input_shape),
            "hidden_dim": hidden_dim,
        },
        "environment_contract": (
            dict(payload["environment_contract"])
            if isinstance(payload.get("environment_contract"), Mapping)
            else (
                dict(payload["metadata"]["environment_contract"])
                if isinstance(payload.get("metadata"), Mapping)
                and isinstance(payload["metadata"].get("environment_contract"), Mapping)
                else None
            )
        ),
    }
    return model, metadata


def analyze_checkpoint(
    checkpoint: str | Path,
    probe_states: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Return Q statistics plus provenance for one checkpoint/probe pair."""

    model, checkpoint_metadata = load_checkpoint_model(checkpoint, device=device)
    observations, probe_metadata = load_probe_states(probe_states)
    analysis = analyze_model_q_values(model, observations, device=device)
    return {
        "schema_version": 1,
        "checkpoint": checkpoint_metadata,
        "probe_states": {
            "path": Path(probe_states).as_posix(),
            "sha256": _sha256(Path(probe_states)),
            "metadata": probe_metadata,
        },
        "analysis": analysis,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Q-values for a fixed Breakout probe artifact."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-states", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day17/q-probe-summary.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_checkpoint(
        args.checkpoint,
        args.probe_states,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
