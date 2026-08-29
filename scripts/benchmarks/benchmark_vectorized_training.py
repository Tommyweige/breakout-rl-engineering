"""Run a real single-env versus vectorized DQN systems screening."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from breakout_env import make_breakout_vector_env
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.training.config import DQNConfig
from breakout_rl.training.dqn_trainer import resolve_device
from breakout_rl.training.vectorized import VectorizedDQNTrainer
from scripts.benchmarks.profile_batch_size_experiment import RuntimeSampler, _sample_summary


def _git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare transition-counted vectorized DQN training throughput."
    )
    parser.add_argument("--environment-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument("--replay-backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10_000)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--diagnostics-interval", type=int, default=100)
    parser.add_argument("--metrics-flush-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--profile-stages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record stage timings in each run summary (default: enabled)",
    )
    parser.add_argument(
        "--strict-action-selection-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "reject environment counts whose batched actions can span an "
            "optimizer boundary (default: disabled for screening)"
        ),
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--run-root", type=Path, default=Path("runs/day16-benchmark"))
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("assets/day16/runtime-samples"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day16/vectorized-training.json"),
    )
    return parser


def _run_one(
    environment_count: int,
    args: argparse.Namespace,
    *,
    resolved_device: str,
    contract: BreakoutEvaluationContractV2,
) -> dict[str, Any]:
    run_dir = args.run_root / f"envs-{environment_count}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"run directory already contains artifacts: {run_dir}; choose a new --run-root"
        )
    sample_path = args.samples_root / f"envs-{environment_count}" / "runtime-samples.csv"
    env = make_breakout_vector_env(
        environment_count,
        **breakout_environment_kwargs(contract),
    )
    sampler = RuntimeSampler(
        sample_path,
        interval_seconds=args.sample_interval,
        gpu_index=args.gpu_index,
    )
    checkpoint_interval = args.checkpoint_interval or args.total_steps
    config = DQNConfig(
        total_steps=args.total_steps,
        seed=args.seed,
        num_envs=environment_count,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        train_frequency=args.train_frequency,
        target_update_interval=args.target_update_interval,
        epsilon_decay_steps=args.epsilon_decay_steps,
        checkpoint_interval=checkpoint_interval,
        diagnostics_interval=args.diagnostics_interval,
        metrics_flush_interval=args.metrics_flush_interval,
        cpu_threads=args.cpu_threads,
        device=args.device,
        replay_backend=args.replay_backend,
        profile_stages=args.profile_stages,
        strict_action_selection_parity=args.strict_action_selection_parity,
    )
    started_at = time.perf_counter()
    try:
        sampler.start()
        trainer = VectorizedDQNTrainer(env, config, run_dir=run_dir)
        summary = trainer.train()
    finally:
        sampler.stop()
        env.close()

    return {
        "environment_count": environment_count,
        "run_dir": str(run_dir),
        "resolved_device": resolved_device,
        "summary": summary,
        "runtime_profile": _sample_summary(
            sample_path,
            interval_seconds=args.sample_interval,
        ),
        "wall_clock_seconds_from_benchmark": time.perf_counter() - started_at,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.environment_counts or any(count < 1 for count in args.environment_counts):
        raise ValueError("environment counts must be positive")
    if args.total_steps < 1:
        raise ValueError("total steps must be positive")
    if args.total_steps < args.learning_starts:
        raise ValueError("total steps must reach learning_starts for a systems screening")
    if args.sample_interval <= 0:
        raise ValueError("sample interval must be greater than zero")

    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract)
    device = resolve_device(args.device)
    results = [
        _run_one(
            count,
            args,
            resolved_device=str(device),
            contract=contract,
        )
        for count in args.environment_counts
    ]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "replay_backend": args.replay_backend,
        "contract_path": args.contract.as_posix(),
        "contract": contract.to_dict(),
        "protocol": {
            "environment_id": contract.environment_id,
            "algorithm": "vanilla DQN",
            "total_transitions": args.total_steps,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "learning_starts": args.learning_starts,
            "train_frequency": args.train_frequency,
            "target_update_interval": args.target_update_interval,
            "epsilon_decay_steps": args.epsilon_decay_steps,
            "checkpoint_interval": args.checkpoint_interval or args.total_steps,
            "precision": "float32",
            "fire_reset": contract.fire_reset,
            "frame_skip": contract.frame_skip,
            "frame_stack": contract.frame_stack,
            "sticky_action_probability": contract.sticky_action_probability,
            "terminal_on_life_loss": contract.terminal_on_life_loss,
            "schedule_semantics": (
                "exact transition boundaries with replay chunks when a vector "
                "step crosses an event"
            ),
            "action_selection_batch_semantics": (
                "one batched forward selects all N actions before env.step; "
                "later transitions in a crossing batch use the pre-update "
                "online-network snapshot"
            ),
            "strict_action_selection_parity_rule": (
                "strict parity requires num_envs <= train_frequency and "
                "train_frequency % num_envs == 0"
            ),
            "strict_action_selection_parity_requested": args.strict_action_selection_parity,
            "git_commit_sha": _git_commit_sha(),
            "profile_stages": args.profile_stages,
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    print(f"Benchmark report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
