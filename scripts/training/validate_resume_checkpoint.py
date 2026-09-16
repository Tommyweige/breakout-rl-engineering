"""Validate whether DQN checkpoints support exact training continuation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


RESUME_CONTRACT_VERSION = 1
# The current trainers intentionally start a fresh environment and do not
# restore replay arrays or ALE state.  Keep this explicit so a future payload
# cannot make this validator overclaim exact continuation before the loaders
# implement the corresponding restore contract.
EXACT_RESUME_LOADER_SUPPORTED = False


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint must contain a mapping")
    return payload


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    payload = _load(path)
    resume_contract_version = payload.get("resume_contract_version")
    replay_state = payload.get("replay_state")
    has_replay_state = isinstance(replay_state, Mapping) and all(
        key in replay_state
        for key in (
            "states",
            "next_states",
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "capacity",
            "size",
            "write_index",
        )
    )
    has_environment_state = bool(
        isinstance(payload.get("environment_state"), Mapping)
        and payload["environment_state"]
    )
    rng_state = payload.get("rng_state")
    required_rng_keys = {"python", "numpy_global", "torch_cpu", "action_rng"}
    has_rng_state = isinstance(rng_state, Mapping) and required_rng_keys <= set(
        rng_state
    )
    has_model_state = all(
        bool(isinstance(payload.get(key), Mapping) and payload[key])
        for key in ("online_network", "target_network", "optimizer")
    )
    has_step_state = all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (
            payload.get("global_step"),
            payload.get("training_steps", payload.get("global_step")),
        )
    )
    replay_saved = bool(payload.get("replay_saved", False))
    blockers: list[str] = []
    if resume_contract_version != RESUME_CONTRACT_VERSION:
        blockers.append(
            f"resume contract version must be {RESUME_CONTRACT_VERSION}"
        )
    if not replay_saved or not has_replay_state:
        blockers.append("replay buffer contents are not checkpointed")
    if not has_environment_state:
        blockers.append("environment/ALE state is not checkpointed")
    if not has_rng_state:
        blockers.append("complete RNG state is not checkpointed")
    if not has_model_state:
        blockers.append("model and optimizer state is incomplete")
    if not has_step_state:
        blockers.append("global/training step state is incomplete")
    if not EXACT_RESUME_LOADER_SUPPORTED:
        blockers.append(
            "trainer does not restore replay contents and environment/ALE state"
        )
    return {
        "path": str(path),
        "format_version": payload.get("format_version"),
        "training_steps": payload.get("training_steps", payload.get("global_step")),
        "algorithm": payload.get("algorithm"),
        "architecture": payload.get("architecture"),
        "training_seed": (
            payload.get("config", {}).get("seed")
            if isinstance(payload.get("config"), dict)
            else None
        ),
        "resume_contract_version": resume_contract_version,
        "replay_saved": replay_saved,
        "has_replay_state": has_replay_state,
        "has_rng_state": has_rng_state,
        "has_environment_state": has_environment_state,
        "has_model_state": has_model_state,
        "has_step_state": has_step_state,
        "exact_continuation_available": not blockers,
        "blockers": blockers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether checkpoints provide exact continuation state."
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reports = [inspect_checkpoint(path) for path in args.checkpoints]
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"Resume validation failed: {error}", file=sys.stderr)
        return 2
    payload = {
        "schema_version": 1,
        "artifact_type": "issue9_reward_shaping_resume_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "exact_continuation_available": all(
            bool(report["exact_continuation_available"]) for report in reports
        ),
        "checkpoints": reports,
        "decision": (
            "resume_from_checkpoint"
            if all(bool(report["exact_continuation_available"]) for report in reports)
            else "train_stage3_from_scratch"
        ),
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
