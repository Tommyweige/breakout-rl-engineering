"""Command-line entry point for the Day 12 Breakout DQN trainer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from breakout_env import make_breakout_env
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.dqn_trainer import DQNTrainer, NonFiniteTrainingError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the baseline vanilla DQN on preprocessed Breakout."
    )
    parser.add_argument(
        "--preset",
        choices=("development", "smoke", "debug"),
        default=None,
        help="validated configuration preset (default: development)",
    )
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="torch device, for example cpu or cuda")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint to restore; replay data is warmed up again",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--train-frequency", type=int, default=None)
    parser.add_argument("--target-update-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--no-reward-clip",
        action="store_true",
        help="store raw rewards for training instead of sign-clipped rewards",
    )
    return parser


def _load_checkpoint_config(path: Path) -> DQNConfig:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("checkpoint does not contain a DQN config")
    return DQNConfig.from_dict(payload["config"])


def _config_from_args(args: argparse.Namespace) -> DQNConfig:
    if args.resume is not None:
        base = _load_checkpoint_config(args.resume)
    elif args.preset == "debug":
        base = DQNConfig.debug(device=args.device or "cuda")
    elif args.preset == "smoke":
        base = DQNConfig.smoke(device=args.device or "cpu")
    else:
        base = DQNConfig(device=args.device or "cpu")

    overrides: dict[str, Any] = {}
    for name in (
        "total_steps",
        "seed",
        "device",
        "batch_size",
        "replay_capacity",
        "learning_starts",
        "train_frequency",
        "target_update_interval",
        "checkpoint_interval",
    ):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.no_reward_clip:
        overrides["reward_clip"] = False
    return base.with_overrides(**overrides)


def _run_path(args: argparse.Namespace, config: DQNConfig) -> Path:
    if args.resume is not None and args.run_dir is None:
        return args.resume.resolve().parent.parent

    root = args.run_dir or Path("runs")
    run_id = args.run_id
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        preset_name = args.preset or "development"
        day_prefix = "day13" if preset_name == "debug" else "day12"
        run_id = f"{day_prefix}-{preset_name}-seed{config.seed}-{timestamp}"
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run-id must be a single directory name")
    return root / run_id


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _config_from_args(args)
        run_path = _run_path(args, config)
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"Invalid training configuration: {error}")
        return 2

    env = make_breakout_env()
    try:
        trainer = DQNTrainer(
            env,
            config,
            run_dir=run_path,
            resume_from=args.resume,
        )
        summary = trainer.train()
    except NonFiniteTrainingError as error:
        print(f"Training stopped because a non-finite value was detected: {error}")
        return 1
    except RuntimeError as error:
        print(f"Training could not start or was stopped: {error}")
        return 2
    finally:
        env.close()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
