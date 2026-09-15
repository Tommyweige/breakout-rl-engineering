"""Validate whether DQN checkpoints support exact training continuation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


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
    replay_saved = bool(payload.get("replay_saved", False))
    has_replay_arrays = all(
        key in payload
        for key in (
            "replay_states",
            "replay_next_states",
            "replay_actions",
            "replay_rewards",
            "replay_terminated",
            "replay_truncated",
        )
    )
    has_environment_state = any(
        key in payload
        for key in ("environment_state", "environment_states", "ale_state")
    )
    blockers: list[str] = []
    if not replay_saved or not has_replay_arrays:
        blockers.append("replay buffer contents are not checkpointed")
    if not has_environment_state:
        blockers.append("environment/ALE state is not checkpointed")
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
        "replay_saved": replay_saved,
        "has_replay_arrays": has_replay_arrays,
        "has_rng_state": isinstance(payload.get("rng_state"), dict),
        "has_environment_state": has_environment_state,
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
