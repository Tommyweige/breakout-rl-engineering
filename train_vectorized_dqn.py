"""Command-line entry point for transition-counted vectorized DQN training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from breakout_env import make_breakout_vector_env
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.vectorized import VectorizedDQNTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train vanilla DQN with batched Breakout environments."
    )
    parser.add_argument(
        "--preset",
        choices=("development", "smoke", "debug"),
        default=None,
        help="validated configuration preset (default: development)",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:<index>, or auto")
    parser.add_argument("--replay-backend", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--train-frequency", type=int, default=None)
    parser.add_argument("--target-update-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument(
        "--strict-action-selection-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "require each vector batch to fit within one optimizer interval "
            "(default: enabled)"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
        help="load a machine-readable Breakout environment contract",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> DQNConfig:
    if args.preset == "debug":
        base = DQNConfig.debug(device=args.device or "cuda")
    elif args.preset == "smoke":
        base = DQNConfig.smoke(device=args.device or "cpu")
    else:
        base = DQNConfig(device=args.device or "cpu")

    overrides: dict[str, Any] = {"num_envs": args.num_envs}
    for name in (
        "total_steps",
        "seed",
        "device",
        "replay_backend",
        "batch_size",
        "replay_capacity",
        "learning_starts",
        "train_frequency",
        "target_update_interval",
        "checkpoint_interval",
        "cpu_threads",
    ):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.profile_stages:
        overrides["profile_stages"] = True
    overrides["strict_action_selection_parity"] = args.strict_action_selection_parity
    return base.with_overrides(**overrides)


def _run_path(args: argparse.Namespace, config: DQNConfig) -> Path:
    root = args.run_dir or Path("runs")
    if args.run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        preset_name = args.preset or "development"
        run_id = (
            f"day16-{preset_name}-envs{config.num_envs}-seed{config.seed}-"
            f"{timestamp}"
        )
    else:
        run_id = args.run_id
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run-id must be a single directory name")
    return root / run_id


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _config_from_args(args)
        run_path = _run_path(args, config)
        contract: BreakoutEvaluationContractV2 | None = (
            load_evaluation_contract(args.contract) if args.contract is not None else None
        )
        validate_breakout_runtime_contract(contract)
    except (TypeError, ValueError, FileNotFoundError) as error:
        print(f"Invalid vectorized training configuration: {error}")
        return 2

    env = make_breakout_vector_env(
        config.num_envs,
        **breakout_environment_kwargs(contract),
    )
    try:
        trainer = VectorizedDQNTrainer(env, config, run_dir=run_path)
        summary = trainer.train()
    except (RuntimeError, ValueError) as error:
        print(f"Vectorized training could not start or was stopped: {error}")
        return 2
    finally:
        env.close()

    serialized = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    print(f"Artifacts written to: {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
