"""Evaluate a vectorized-training checkpoint under the Day 15 v2 contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from breakout_env import make_breakout_env
from breakout_rl.evaluation import (
    evaluate_policy,
    load_dqn_checkpoint,
    load_evaluation_config,
    write_evaluation_artifacts,
)
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from evaluate_dqn import _validate_contract_for_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a vectorized DQN checkpoint with the fixed v2 contract."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/day16-vectorized-contract-v2"),
    )
    parser.add_argument("--evaluation-id", default="day16-vectorized-contract-v2")
    return parser


def _environment_factory(
    contract: BreakoutEvaluationContractV2,
) -> Callable[[], Any]:
    return lambda: make_breakout_env(**breakout_environment_kwargs(contract))


def run_evaluation(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract)
    evaluation_config = load_evaluation_config(args.config)
    _validate_contract_for_config(contract, evaluation_config)
    env_factory = _environment_factory(contract)
    loaded = load_dqn_checkpoint(
        args.checkpoint,
        device=args.device,
        env_factory=env_factory,
    )
    vectorized_run_id = Path(args.checkpoint).resolve().parent.parent.name
    metadata = {
        "evaluation_config_path": args.config.as_posix(),
        "evaluation_config": evaluation_config.to_dict(),
        "evaluation_contract_path": args.contract.as_posix(),
        "evaluation_contract": contract.to_dict(),
        "purpose": "Day 15 Contract v2 learning-regression guardrail for Day 16",
        "policy_protocol": "same frozen seeds, environment-side FIRE, raw reward, and epsilon",
    }
    training_metadata = {
        **dict(loaded.training_metadata),
        "source_day14_run_id": None,
        "source_day16_run_id": vectorized_run_id,
        "day16_vectorized_checkpoint": True,
    }
    checkpoint_metadata = {
        **dict(loaded.checkpoint_metadata),
        "source_day14_run_id": None,
        "source_day16_run_id": vectorized_run_id,
    }
    result = evaluate_policy(
        loaded.model,
        episodes=evaluation_config.episodes_per_seed,
        seeds=evaluation_config.seeds,
        device=args.device,
        epsilon=evaluation_config.epsilon,
        model_id=loaded.model_id,
        training_metadata=training_metadata,
        checkpoint_metadata=checkpoint_metadata,
        evaluation_id=args.evaluation_id,
        env_factory=env_factory,
        metadata=metadata,
    )
    results_path, episodes_path = write_evaluation_artifacts(result, args.output_dir)
    return results_path, episodes_path, result.to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results_path, episodes_path, payload = run_evaluation(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"Vectorized checkpoint evaluation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "results": results_path.as_posix(),
                "episodes": episodes_path.as_posix(),
                "model_id": payload["model_id"],
                "summary": payload["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
