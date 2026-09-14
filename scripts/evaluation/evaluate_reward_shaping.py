"""Run paired raw-score evaluation for Issue #9 reward shaping."""

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
from breakout_rl.reward_shaping_experiment import write_reward_shaping_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline and life-loss-penalty checkpoints on the same "
            "predeclared raw-score seed set."
        )
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--shaped-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--shaped-run", type=Path, required=True)
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
        default=Path("experiments/issue-9-reward-shaping"),
    )
    parser.add_argument("--recent-window", type=int, default=20)
    return parser


def _environment_factory(contract: BreakoutEvaluationContractV2) -> Callable[[], Any]:
    return lambda: make_breakout_env(**breakout_environment_kwargs(contract))


def _validate_experiment_config(
    evaluation_config: Any,
    contract: BreakoutEvaluationContractV2,
) -> None:
    if evaluation_config.environment_id != contract.environment_id:
        raise ValueError("reward-shaping evaluation config and contract use different environments")
    if evaluation_config.epsilon != contract.evaluation_epsilon:
        raise ValueError("reward-shaping evaluation epsilon does not match the contract")
    if evaluation_config.total_episodes != 50:
        raise ValueError("Issue #9 paired evaluation requires exactly 50 episodes")
    if len(set(evaluation_config.seeds)) != 50:
        raise ValueError("Issue #9 paired evaluation requires 50 unique predeclared seeds")


def _evaluate_checkpoint(
    checkpoint: Path,
    *,
    label: str,
    evaluation_config: Any,
    contract: BreakoutEvaluationContractV2,
    device: str,
    env_factory: Callable[[], Any],
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    loaded = load_dqn_checkpoint(
        checkpoint,
        device=device,
        env_factory=env_factory,
    )
    checkpoint_contract_id = loaded.training_metadata.get("contract_id")
    if checkpoint_contract_id is not None and checkpoint_contract_id != contract.contract_id:
        raise ValueError(
            f"{label} checkpoint Contract v2 id does not match the evaluation contract"
        )
    training_metadata = {
        **dict(loaded.training_metadata),
        "reward_shaping_variant": label,
        "evaluation_score_uses_raw_reward": True,
    }
    checkpoint_metadata = {
        **dict(loaded.checkpoint_metadata),
        "reward_shaping_variant": label,
    }
    result = evaluate_policy(
        loaded.model,
        episodes=evaluation_config.episodes_per_seed,
        seeds=evaluation_config.seeds,
        device=device,
        epsilon=evaluation_config.epsilon,
        model_id=loaded.model_id,
        training_metadata=training_metadata,
        checkpoint_metadata=checkpoint_metadata,
        evaluation_id=f"issue-9-{label}",
        env_factory=env_factory,
        metadata={
            "purpose": "Issue #9 paired raw-score comparison",
            "reward_shaping_variant": label,
            "raw_reward": True,
            "life_loss_penalty_is_not_evaluation_score": True,
            "evaluation_config": evaluation_config.to_dict(),
            "evaluation_contract": contract.to_dict(),
        },
    )
    results_path, episodes_path = write_evaluation_artifacts(result, output_dir)
    return results_path, episodes_path, result.to_dict()


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract)
    evaluation_config = load_evaluation_config(args.config)
    _validate_experiment_config(evaluation_config, contract)
    if args.recent_window < 1:
        raise ValueError("recent-window must be positive")
    env_factory = _environment_factory(contract)
    output_dir = args.output_dir
    baseline_results, baseline_episodes, baseline_payload = _evaluate_checkpoint(
        args.baseline_checkpoint,
        label="baseline",
        evaluation_config=evaluation_config,
        contract=contract,
        device=args.device,
        env_factory=env_factory,
        output_dir=output_dir / "baseline-evaluation",
    )
    shaped_results, shaped_episodes, shaped_payload = _evaluate_checkpoint(
        args.shaped_checkpoint,
        label="life-loss-penalty",
        evaluation_config=evaluation_config,
        contract=contract,
        device=args.device,
        env_factory=env_factory,
        output_dir=output_dir / "life-loss-penalty-evaluation",
    )
    artifact_paths = write_reward_shaping_artifacts(
        baseline_run=args.baseline_run,
        shaped_run=args.shaped_run,
        baseline_evaluation=baseline_payload,
        shaped_evaluation=shaped_payload,
        output_dir=output_dir,
        experiment_config={
            "baseline_run": str(args.baseline_run),
            "shaped_run": str(args.shaped_run),
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "shaped_checkpoint": str(args.shaped_checkpoint),
            "training_configs": {
                "baseline": "configs/dueling_double_dqn_baseline.json",
                "shaped": "configs/dueling_double_dqn_life_loss_penalty.json",
            },
            "evaluation_config": str(args.config),
            "contract": str(args.contract),
            "device": args.device,
            "paired_seed_count": evaluation_config.total_episodes,
            "baseline_evaluation_results": str(baseline_results),
            "baseline_evaluation_episodes": str(baseline_episodes),
            "shaped_evaluation_results": str(shaped_results),
            "shaped_evaluation_episodes": str(shaped_episodes),
            "conditions": (
                "Dueling Double DQN, same seed, transition budget, contract, "
                "optimizer, epsilon schedule, and runtime; only life_loss_penalty differs"
            ),
        },
        recent_window=args.recent_window,
    )
    return {
        "artifact_paths": artifact_paths,
        "baseline_summary": baseline_payload["summary"],
        "shaped_summary": shaped_payload["summary"],
        "paired_evaluation": json.loads(
            (output_dir / "paired-evaluation.json").read_text(encoding="utf-8")
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_evaluation(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"Issue #9 reward-shaping evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
