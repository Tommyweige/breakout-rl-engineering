"""Run the fixed Day 15 evaluation protocol for Random or a frozen DQN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from breakout_env import make_breakout_env
from breakout_rl.evaluation import (
    EvaluationConfig,
    evaluate_policy,
    load_day14_provenance,
    load_dqn_checkpoint,
    load_evaluation_config,
    validate_checkpoint_provenance,
    write_evaluation_artifacts,
)
from breakout_rl.evaluation_contract import (
    BreakoutEvaluationContractV2,
    expand_concrete_episode_seeds,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)


PolicyName = Literal["random", "dqn"]
OUTPUT_DIRS: dict[PolicyName, Path] = {
    "random": Path("evaluations/day15-random-baseline"),
    "dqn": Path("evaluations/day15-dqn-cuda"),
}
FORMAL_DQN_OUTPUT_DIR = OUTPUT_DIRS["dqn"]
DQN_REFERENCE_OUTPUT_DIR = Path("evaluations/day15-dqn-cpu-reference")
EVALUATION_IDS: dict[PolicyName, str] = {
    "random": "day15-random-baseline",
    "dqn": "day15-dqn-cuda",
}
FORMAL_DQN_EVALUATION_ID = EVALUATION_IDS["dqn"]
DQN_REFERENCE_EVALUATION_ID = "day15-dqn-cpu-reference"
CONTRACT_V2_OUTPUT_DIRS: dict[PolicyName, Path] = {
    "random": Path("evaluations/day15-contract-v2-random"),
    "dqn": Path("evaluations/day15-contract-v2-dqn"),
}
CONTRACT_V2_EVALUATION_IDS: dict[PolicyName, str] = {
    "random": "day15-contract-v2-random",
    "dqn": "day15-contract-v2-dqn",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen Breakout policy on fixed seeds."
    )
    parser.add_argument(
        "--policy",
        choices=tuple(OUTPUT_DIRS),
        default=None,
        help="policy to evaluate; inferred from --checkpoint when omitted",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="explicit torch device; the formal DQN run uses cuda",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluation-id", default=None)
    parser.add_argument(
        "--source-day14-manifest",
        type=Path,
        default=None,
        help="override the manifest referenced by the evaluation config",
    )
    parser.add_argument(
        "--source-day14-profiling-report",
        type=Path,
        default=None,
        help="override the profiling report referenced by the evaluation config",
    )
    parser.add_argument(
        "--contract",
        "--evaluation-contract",
        dest="contract",
        type=Path,
        default=None,
        help="load a machine-readable environment contract (v2 uses environment-side FIRE)",
    )
    return parser


def _selected_policy(args: argparse.Namespace) -> PolicyName:
    if args.policy is not None:
        return args.policy
    return "dqn" if args.checkpoint is not None else "random"


def _resolve_manifest(config: EvaluationConfig, config_path: Path) -> Path:
    if config.source_day14_manifest is None:
        return Path("experiments/day14-final-frozen-100k/manifest.json")
    configured = Path(config.source_day14_manifest)
    candidates = (configured, Path.cwd() / configured, config_path.parent / configured)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return configured


def _resolve_profiling_report(
    config: EvaluationConfig,
    config_path: Path,
) -> Path | None:
    if config.source_day14_profiling_report is None:
        return None
    configured = Path(config.source_day14_profiling_report)
    candidates = (configured, Path.cwd() / configured, config_path.parent / configured)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return configured


def _is_cuda_request(device: str | None) -> bool:
    if device is None:
        return False
    normalized = device.strip().lower()
    return normalized == "cuda" or normalized.startswith("cuda:")


def _output_destination(
    policy_name: PolicyName,
    args: argparse.Namespace,
    *,
    contract_id: str | None = None,
) -> tuple[Path, str]:
    requested_output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if contract_id is not None:
        output_dir = requested_output_dir or CONTRACT_V2_OUTPUT_DIRS[policy_name]
        evaluation_id = args.evaluation_id or CONTRACT_V2_EVALUATION_IDS[policy_name]
        legacy_paths = {path.resolve() for path in OUTPUT_DIRS.values()}
        if output_dir.resolve() in legacy_paths:
            raise ValueError(
                "Contract v2 evaluation cannot overwrite Evaluation Contract v1 artifacts"
            )
        if evaluation_id in set(EVALUATION_IDS.values()):
            raise ValueError(
                "Contract v2 evaluation cannot reuse an Evaluation Contract v1 id"
            )
        return output_dir, evaluation_id
    if policy_name != "dqn" or _is_cuda_request(args.device):
        output_dir = requested_output_dir or OUTPUT_DIRS[policy_name]
        evaluation_id = args.evaluation_id or EVALUATION_IDS[policy_name]
        return output_dir, evaluation_id

    output_dir = requested_output_dir or DQN_REFERENCE_OUTPUT_DIR
    evaluation_id = args.evaluation_id or DQN_REFERENCE_EVALUATION_ID
    if output_dir.resolve() == FORMAL_DQN_OUTPUT_DIR.resolve():
        raise ValueError(
            "CPU DQN reference evaluation cannot write the formal CUDA output directory; "
            f"use {DQN_REFERENCE_OUTPUT_DIR.as_posix()} or an explicit separate directory"
        )
    if evaluation_id == FORMAL_DQN_EVALUATION_ID:
        raise ValueError(
            "CPU DQN reference evaluation cannot use the formal CUDA evaluation id"
        )
    return output_dir, evaluation_id


def _validate_contract_for_config(
    contract: BreakoutEvaluationContractV2,
    evaluation_config: EvaluationConfig,
) -> None:
    validate_breakout_runtime_contract(contract)
    if contract.environment_id != evaluation_config.environment_id:
        raise ValueError(
            "evaluation config and contract must use the same environment_id"
        )
    expected_seeds = expand_concrete_episode_seeds(
        evaluation_config.seeds,
        episodes_per_seed=evaluation_config.episodes_per_seed,
    )
    if contract.concrete_episode_seeds != expected_seeds:
        raise ValueError(
            "evaluation config concrete episode seeds do not match the contract"
        )
    if contract.evaluation_epsilon != evaluation_config.epsilon:
        raise ValueError(
            "evaluation config epsilon does not match the contract"
        )


def _contract_environment_factory(
    contract: BreakoutEvaluationContractV2,
) -> Callable[[], Any]:
    return lambda: make_breakout_env(
        stack_size=contract.frame_stack,
        fire_reset=contract.fire_reset,
    )


def _portable_command(
    args: argparse.Namespace,
    *,
    checkpoint_path: str | None,
) -> list[str]:
    command = list(sys.argv)
    if checkpoint_path and args.checkpoint is not None:
        values = {str(args.checkpoint), str(args.checkpoint.resolve())}
        command = [checkpoint_path if value in values else value for value in command]
    return command


def run_evaluation(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    """Load provenance, evaluate one policy, and write both result formats."""

    policy_name = _selected_policy(args)
    if policy_name == "dqn" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for --policy dqn")
    if policy_name == "random" and args.checkpoint is not None:
        raise ValueError("--checkpoint cannot be combined with --policy random")
    if policy_name == "dqn" and args.device is None:
        raise ValueError(
            "DQN evaluation requires an explicit --device; use --device cuda "
            "for the formal Day 15 result"
        )

    evaluation_config = load_evaluation_config(args.config)
    contract_path = getattr(args, "contract", None)
    contract = load_evaluation_contract(contract_path) if contract_path is not None else None
    if contract is not None:
        _validate_contract_for_config(contract, evaluation_config)
    output_dir, evaluation_id = _output_destination(
        policy_name,
        args,
        contract_id=contract.contract_id if contract is not None else None,
    )
    requested_device = args.device or "cpu"
    model = None
    model_id = "random-policy"
    training_metadata: dict[str, Any] = {
        "training_seed": None,
        "training_budget": None,
        "learning_rate": None,
        "batch_size": None,
        "train_frequency": None,
        "replay_backend": None,
        "config_reference": None,
        "source_day14_manifest": None,
    }
    checkpoint_metadata: dict[str, Any] = {}

    if policy_name == "dqn":
        manifest_path = args.source_day14_manifest or _resolve_manifest(
            evaluation_config,
            args.config,
        )
        profiling_path = args.source_day14_profiling_report or _resolve_profiling_report(
            evaluation_config,
            args.config,
        )
        provenance = load_day14_provenance(
            manifest_path,
            profiling_report_path=profiling_path,
        )
        day14_gate = provenance.get("day14_gate", {})
        if not isinstance(day14_gate, Mapping) or day14_gate.get("status") != "passed":
            raise ValueError(
                "Day 14 Gate A is not satisfied; refusing to label this as the "
                f"formal Day 15 DQN evaluation: {day14_gate.get('reasons', []) if isinstance(day14_gate, Mapping) else day14_gate}"
            )
        loaded = load_dqn_checkpoint(
            args.checkpoint,
            device=requested_device,
            source_day14_manifest=manifest_path,
        )
        validate_checkpoint_provenance(
            loaded.checkpoint_metadata,
            loaded.training_metadata,
            provenance,
        )
        model = loaded.model
        model_id = loaded.model_id
        training_metadata = {
            **dict(loaded.training_metadata),
            "source_of_truth": provenance["source_of_truth"],
            "source_day14_manifest": provenance["manifest_path"],
            "config_reference": provenance.get("config_reference"),
            "selection_rule": provenance["selection_rule"],
            "selection_rationale": provenance.get("selection_rationale"),
            "day14_experiment_id": provenance.get("experiment_id"),
            "day14_run_artifact_dir": provenance.get("run_dir"),
            "trainer_runtime": provenance.get("runtime", {}),
            "gpu_profiling_summary": provenance.get("gpu_profiling_summary", {}),
            "source_day14_profiling_report": provenance.get(
                "source_day14_profiling_report"
            ),
            "day14_gate": day14_gate,
        }
        checkpoint_metadata = {
            **dict(loaded.checkpoint_metadata),
            "selection_rule": provenance["selection_rule"],
            "manifest_run_id": provenance.get("run_id"),
        }

    metadata: dict[str, Any] = {
        "evaluation_config_path": args.config.as_posix(),
        "evaluation_config": evaluation_config.to_dict(),
        "command": _portable_command(
            args,
            checkpoint_path=checkpoint_metadata.get("path"),
        ),
        "raw_reward": True,
        "policy_protocol": "shared environment construction, seed handling, episode loop, and schema",
    }
    env_factory = None
    if contract is not None:
        metadata["evaluation_contract_path"] = Path(contract_path).as_posix()
        metadata["evaluation_contract"] = contract.to_dict()
        env_factory = _contract_environment_factory(contract)

    result = evaluate_policy(
        model,
        episodes=evaluation_config.episodes_per_seed,
        seeds=evaluation_config.seeds,
        device=requested_device,
        epsilon=evaluation_config.epsilon,
        model_id=model_id,
        training_metadata=training_metadata,
        checkpoint_metadata=checkpoint_metadata,
        evaluation_id=evaluation_id,
        env_factory=env_factory or make_breakout_env,
        metadata=metadata,
    )
    results_path, episodes_path = write_evaluation_artifacts(result, output_dir)
    return results_path, episodes_path, result.to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results_path, episodes_path, payload = run_evaluation(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "results": results_path.as_posix(),
                "episodes": episodes_path.as_posix(),
                "policy_type": payload["policy_type"],
                "requested_device": payload["requested_device"],
                "resolved_device": payload["resolved_device"],
                "summary": payload["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
