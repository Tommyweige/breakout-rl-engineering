"""Evaluate a vectorized-training checkpoint under an explicit Breakout contract."""

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
    BREAKOUT_CONTRACT_V2_ID,
    BREAKOUT_CONTRACT_V3_ID,
    BreakoutEvaluationContractV2,
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from scripts.evaluation.evaluate_dqn import (
    CONTRACT_V2_EVALUATION_IDS,
    CONTRACT_V2_OUTPUT_DIRS,
    CONTRACT_V3_EVALUATION_IDS,
    CONTRACT_V3_OUTPUT_DIRS,
    _validate_contract_for_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a vectorized DQN checkpoint with an explicit contract."
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
        default=None,
    )
    parser.add_argument("--evaluation-id", default=None)
    return parser


def _environment_factory(
    contract: BreakoutEvaluationContractV2,
) -> Callable[[], Any]:
    return lambda: make_breakout_env(
        **breakout_environment_kwargs(contract, allow_contract_v3=True)
    )


def run_evaluation(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract, allow_contract_v3=True)
    if contract.contract_id == BREAKOUT_CONTRACT_V2_ID:
        default_output_dir = Path("evaluations/day16-vectorized-contract-v2")
        default_evaluation_id = "day16-vectorized-contract-v2"
        protected_output_dirs = {
            Path("evaluations/day20-vectorized-contract-v3"),
            *CONTRACT_V3_OUTPUT_DIRS.values(),
        }
        protected_evaluation_ids = {
            "day20-vectorized-contract-v3",
            *CONTRACT_V3_EVALUATION_IDS.values(),
        }
    elif contract.contract_id == BREAKOUT_CONTRACT_V3_ID:
        default_output_dir = Path("evaluations/day20-vectorized-contract-v3")
        default_evaluation_id = "day20-vectorized-contract-v3"
        protected_output_dirs = {
            Path("evaluations/day16-vectorized-contract-v2"),
            *CONTRACT_V2_OUTPUT_DIRS.values(),
        }
        protected_evaluation_ids = {
            "day16-vectorized-contract-v2",
            *CONTRACT_V2_EVALUATION_IDS.values(),
        }
    else:
        raise ValueError(f"unsupported Breakout contract id: {contract.contract_id}")
    output_dir = args.output_dir or default_output_dir
    evaluation_id = args.evaluation_id or default_evaluation_id
    if output_dir.resolve() in {path.resolve() for path in protected_output_dirs}:
        raise ValueError("evaluation output directory belongs to another contract")
    if evaluation_id in protected_evaluation_ids:
        raise ValueError("evaluation id belongs to another contract")
    evaluation_config = load_evaluation_config(args.config)
    _validate_contract_for_config(contract, evaluation_config)
    env_factory = _environment_factory(contract)
    loaded = load_dqn_checkpoint(
        args.checkpoint,
        device=args.device,
        env_factory=env_factory,
    )
    checkpoint_contract_id = loaded.training_metadata.get("contract_id")
    if contract.contract_id == BREAKOUT_CONTRACT_V3_ID:
        checkpoint_contract_path = loaded.training_metadata.get("contract_path")
        if checkpoint_contract_id != contract.contract_id:
            raise ValueError(
                "Contract v3 checkpoint must record the matching contract id"
            )
        if not isinstance(checkpoint_contract_path, (str, Path)):
            raise ValueError("Contract v3 checkpoint must record its contract path")
        recorded_contract_path = Path(checkpoint_contract_path)
        if not recorded_contract_path.is_absolute():
            recorded_contract_path = Path.cwd() / recorded_contract_path
        if recorded_contract_path.resolve() != Path(args.contract).resolve():
            raise ValueError(
                "Contract v3 checkpoint contract path does not match evaluation"
            )
    elif (
        checkpoint_contract_id is not None
        and checkpoint_contract_id != contract.contract_id
    ):
        raise ValueError(
            "checkpoint contract id does not match the evaluation contract"
        )
    vectorized_run_id = Path(args.checkpoint).resolve().parent.parent.name
    metadata = {
        "evaluation_config_path": args.config.as_posix(),
        "evaluation_config": evaluation_config.to_dict(),
        "evaluation_contract_path": args.contract.as_posix(),
        "evaluation_contract": contract.to_dict(),
        "purpose": (
            "Fixed-seed evaluation of a vectorized DQN checkpoint under "
            f"{contract.contract_id}"
        ),
        "policy_protocol": "same frozen seeds, environment-side FIRE, raw reward, and epsilon",
    }
    training_metadata = {
        **dict(loaded.training_metadata),
        "source_day14_run_id": None,
        "source_vectorized_run_id": vectorized_run_id,
        "training_steps": loaded.checkpoint_metadata.get("training_steps"),
        "vectorized_checkpoint": True,
    }
    checkpoint_metadata = {
        **dict(loaded.checkpoint_metadata),
        "source_day14_run_id": None,
        "source_vectorized_run_id": vectorized_run_id,
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
        evaluation_id=evaluation_id,
        env_factory=env_factory,
        metadata=metadata,
    )
    results_path, episodes_path = write_evaluation_artifacts(result, output_dir)
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
