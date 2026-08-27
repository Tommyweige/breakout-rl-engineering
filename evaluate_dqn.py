"""Evaluate Random or a frozen DQN checkpoint with the Day 15 protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.evaluation import (
    EvaluationConfig,
    evaluate_policy,
    load_day14_provenance,
    load_dqn_checkpoint,
    load_evaluation_config,
    validate_checkpoint_provenance,
    write_evaluation_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen Breakout policy on fixed seeds."
    )
    parser.add_argument(
        "--policy",
        choices=("random", "dqn"),
        default=None,
        help="policy to evaluate (default: dqn when --checkpoint is supplied)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "explicit torch device; DQN evaluation requires this flag, and "
            "the formal run uses cuda"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluation-id", default=None)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="override the fixed config value for a portability run",
    )
    parser.add_argument(
        "--source-day14-manifest",
        type=Path,
        default=None,
        help="override the Day 14 final manifest used for checkpoint validation",
    )
    return parser


def _policy_name(args: argparse.Namespace) -> str:
    if args.policy is not None:
        return str(args.policy)
    return "dqn" if args.checkpoint is not None else "random"


def _resolve_manifest_path(config: EvaluationConfig, config_path: Path) -> Path:
    if config.source_day14_manifest is None:
        return Path("experiments/day14-final-frozen-100k/manifest.json")
    configured = Path(config.source_day14_manifest)
    candidates = [configured, Path.cwd() / configured, config_path.parent / configured]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return configured


def _default_output_dir(policy: str) -> Path:
    return Path("evaluations") / (
        "day15-dqn-cuda" if policy == "dqn" else "day15-random-baseline"
    )


def _run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    policy_name = _policy_name(args)
    if policy_name == "dqn" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for --policy dqn")
    if policy_name == "random" and args.checkpoint is not None:
        raise ValueError("--checkpoint cannot be combined with --policy random")

    evaluation_config = load_evaluation_config(args.config)
    if args.max_steps_per_episode is not None:
        evaluation_config = EvaluationConfig(
            seeds=evaluation_config.seeds,
            episodes_per_seed=evaluation_config.episodes_per_seed,
            epsilon=evaluation_config.epsilon,
            max_steps_per_episode=args.max_steps_per_episode,
            environment_id=evaluation_config.environment_id,
            source_day14_manifest=evaluation_config.source_day14_manifest,
        )
    if policy_name == "dqn" and args.device is None:
        raise ValueError(
            "DQN evaluation requires an explicit --device; use --device cuda "
            "for the formal Day 15 result"
        )
    requested_device = args.device or "cpu"

    model = None
    model_id = "random-policy"
    training_metadata: dict[str, Any] = {
        "training_seed": None,
        "training_budget": None,
        "learning_rate": None,
        "batch_size": None,
        "replay_backend": None,
        "config_reference": None,
        "source_day14_manifest": None,
    }
    checkpoint_metadata: dict[str, Any] = {}
    if policy_name == "dqn":
        manifest_path = args.source_day14_manifest or _resolve_manifest_path(
            evaluation_config,
            args.config,
        )
        provenance = load_day14_provenance(manifest_path)
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
            "source_day14_manifest": provenance["manifest_path"],
            "config_reference": provenance.get("config_reference"),
            "selection_rule": provenance["selection_rule"],
            "day14_experiment_id": provenance.get("experiment_id"),
            "day14_run_artifact_dir": provenance.get("run_dir"),
            "trainer_runtime": provenance.get("runtime", {}),
            "selection_rationale": provenance.get("selection_rationale"),
            "gpu_profiling_summary": provenance.get("gpu_profiling_summary", {}),
        }
        checkpoint_metadata = {
            **dict(loaded.checkpoint_metadata),
            "selection_rule": provenance["selection_rule"],
            "manifest_run_id": provenance.get("run_id"),
        }

    result = evaluate_policy(
        model,
        episodes=evaluation_config.episodes_per_seed,
        seeds=evaluation_config.seeds,
        device=requested_device,
        epsilon=evaluation_config.epsilon,
        max_steps_per_episode=evaluation_config.max_steps_per_episode,
        model_id=model_id,
        training_metadata=training_metadata,
        checkpoint_metadata=checkpoint_metadata,
        evaluation_id=args.evaluation_id or (
            "day15-dqn-cuda" if policy_name == "dqn" else "day15-random-baseline"
        ),
        metadata={
            "evaluation_config_path": args.config.as_posix(),
            "evaluation_config": evaluation_config.to_dict(),
            "command": _artifact_command(
                args,
                checkpoint_metadata=checkpoint_metadata,
            ),
            "raw_reward": True,
            "policy_protocol": "shared environment, reset seed, episode loop, and schema",
        },
    )
    output_dir = args.output_dir or _default_output_dir(policy_name)
    results_path, episodes_path = write_evaluation_artifacts(result, output_dir)
    return results_path, episodes_path, result.to_dict()


def _artifact_command(
    args: argparse.Namespace,
    *,
    checkpoint_metadata: Mapping[str, Any],
) -> list[str]:
    """Keep recorded commands reproducible across local worktree paths."""

    command = list(sys.argv)
    checkpoint_path = checkpoint_metadata.get("path")
    if checkpoint_path:
        absolute_checkpoint = str(Path(args.checkpoint).resolve())
        command = [
            checkpoint_path
            if value in {str(args.checkpoint), absolute_checkpoint}
            else value
            for value in command
        ]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results_path, episodes_path, payload = _run(args)
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
