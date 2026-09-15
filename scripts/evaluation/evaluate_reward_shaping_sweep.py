"""Evaluate every Stage 2 penalty checkpoint on the same 50-seed protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
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
from breakout_rl.reward_shaping_experiment import (
    compare_evaluation_payloads,
    summarize_training_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the Issue #9 Stage 2 penalty sweep at every checkpoint."
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=Path("experiments/issue-9-reward-shaping/stage2-250k"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/reward_shaping_eval.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/eval/breakout_contract_v2.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        action="append",
        dest="checkpoint_steps",
        default=None,
        help="repeat for explicit checkpoints; defaults to 62500/125000/187500/250000",
    )
    return parser


def _environment_factory(contract: BreakoutEvaluationContractV2) -> Callable[[], Any]:
    return lambda: make_breakout_env(**breakout_environment_kwargs(contract))


def _validate_eval_config(evaluation_config: Any, contract: BreakoutEvaluationContractV2) -> None:
    if evaluation_config.environment_id != contract.environment_id:
        raise ValueError("evaluation config and contract use different environments")
    if evaluation_config.epsilon != contract.evaluation_epsilon:
        raise ValueError("evaluation epsilon does not match the contract")
    if evaluation_config.total_episodes != 50:
        raise ValueError("Stage 2 requires 50 evaluation episodes")
    if len(set(evaluation_config.seeds)) != 50:
        raise ValueError("Stage 2 requires 50 unique evaluation seeds")


def _load_manifest(stage_dir: Path) -> dict[str, Any]:
    path = stage_dir / "training-sweep.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("variants"), list):
        raise ValueError(f"{path}: invalid training sweep manifest")
    if payload.get("status") != "completed":
        raise ValueError(f"{path}: training sweep is not completed")
    return payload


def _validate_manifest_fairness(variants: Sequence[Mapping[str, Any]]) -> None:
    if not variants:
        raise ValueError("Stage 2 sweep manifest has no variants")
    labels = [str(variant.get("label")) for variant in variants]
    if len(set(labels)) != len(labels):
        raise ValueError("Stage 2 sweep manifest has duplicate variant labels")
    first_config = variants[0].get("config")
    if not isinstance(first_config, Mapping):
        raise ValueError("Stage 2 sweep manifest has an invalid variant config")
    comparable = {
        key: value
        for key, value in first_config.items()
        if key not in {"life_loss_penalty", "contract_id", "contract_path"}
    }
    first_contract_sha256 = variants[0].get("contract_sha256")
    for variant in variants[1:]:
        config = variant.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("Stage 2 sweep manifest has an invalid variant config")
        candidate = {
            key: value
            for key, value in config.items()
            if key not in {"life_loss_penalty", "contract_id", "contract_path"}
        }
        if candidate != comparable:
            raise ValueError(
                "Stage 2 sweep variants differ in a training field other than "
                "life_loss_penalty"
            )
        if (
            first_contract_sha256 is not None
            and variant.get("contract_sha256") != first_contract_sha256
        ):
            raise ValueError("Stage 2 sweep variants use different contracts")


def _variant_checkpoint(
    variant: Mapping[str, Any],
    *,
    step: int,
) -> Path:
    run_dir = Path(str(variant["run_dir"]))
    checkpoint = run_dir / "checkpoints" / f"step-{step:08d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _validate_checkpoint_provenance(
    loaded: Any,
    *,
    variant: Mapping[str, Any],
    label: str,
    step: int,
    contract: BreakoutEvaluationContractV2,
) -> None:
    expected_config = variant.get("config")
    if not isinstance(expected_config, Mapping):
        raise ValueError(f"{label}@{step}: sweep manifest config is invalid")
    training_metadata = loaded.training_metadata
    checkpoint_metadata = loaded.checkpoint_metadata
    observed_config = training_metadata.get("training_config")
    if not isinstance(observed_config, Mapping):
        raise ValueError(f"{label}@{step}: checkpoint is missing training config provenance")
    for field, expected in expected_config.items():
        if field in {"contract_id", "contract_path"}:
            continue
        if observed_config.get(field) != expected:
            raise ValueError(
                f"{label}@{step}: checkpoint config field {field!r} does not "
                f"match the sweep manifest (expected={expected!r}, "
                f"observed={observed_config.get(field)!r})"
            )
    if training_metadata.get("training_seed") != expected_config.get("seed"):
        raise ValueError(f"{label}@{step}: checkpoint training seed does not match")
    if training_metadata.get("training_budget") != expected_config.get("total_steps"):
        raise ValueError(f"{label}@{step}: checkpoint training budget does not match")
    if checkpoint_metadata.get("training_steps") != step:
        raise ValueError(f"{label}@{step}: checkpoint step provenance does not match path")
    if checkpoint_metadata.get("source_day14_run_id") != label:
        raise ValueError(
            f"{label}@{step}: checkpoint run id does not match the sweep variant"
        )
    if training_metadata.get("contract_id") != contract.contract_id:
        raise ValueError(f"{label}@{step}: checkpoint contract id does not match")
    expected_contract_sha256 = variant.get("contract_sha256")
    if expected_contract_sha256 is not None:
        canonical_contract = json.dumps(
            contract.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        observed_contract_sha256 = hashlib.sha256(canonical_contract).hexdigest()
        if observed_contract_sha256 != expected_contract_sha256:
            raise ValueError(
                f"{label}@{step}: checkpoint contract provenance is inconsistent"
            )


def _evaluate_checkpoint(
    checkpoint: Path,
    *,
    label: str,
    step: int,
    variant: Mapping[str, Any],
    evaluation_config: Any,
    contract: BreakoutEvaluationContractV2,
    device: str,
    env_factory: Callable[[], Any],
    output_dir: Path,
) -> dict[str, Any]:
    loaded = load_dqn_checkpoint(
        checkpoint,
        device=device,
        env_factory=env_factory,
    )
    _validate_checkpoint_provenance(
        loaded,
        variant=variant,
        label=label,
        step=step,
        contract=contract,
    )
    checkpoint_contract_id = loaded.training_metadata.get("contract_id")
    if checkpoint_contract_id != contract.contract_id:
        raise ValueError(f"{label}@{step}: checkpoint contract id does not match")
    result = evaluate_policy(
        loaded.model,
        episodes=evaluation_config.episodes_per_seed,
        seeds=evaluation_config.seeds,
        device=device,
        epsilon=evaluation_config.epsilon,
        model_id=loaded.model_id,
        training_metadata={
            **dict(loaded.training_metadata),
            "reward_shaping_variant": label,
            "checkpoint_step": step,
            "evaluation_score_uses_raw_reward": True,
            "evaluation_score_definition": (
                "raw Atari game reward sum; no clipping and no life-loss penalty"
            ),
        },
        checkpoint_metadata={
            **dict(loaded.checkpoint_metadata),
            "reward_shaping_variant": label,
            "checkpoint_step": step,
        },
        evaluation_id=f"issue-9-stage2-{label}-step-{step}",
        env_factory=env_factory,
        metadata={
            "purpose": "Issue #9 Stage 2 250k penalty sweep",
            "reward_shaping_variant": label,
            "checkpoint_step": step,
            "raw_reward": True,
            "score_definition": (
                "raw Atari game reward sum; no clipping and no life-loss penalty"
            ),
            "life_loss_penalty_is_not_evaluation_score": True,
            "evaluation_contract": contract.to_dict(),
            "evaluation_config": evaluation_config.to_dict(),
        },
    )
    results_path, episodes_path = write_evaluation_artifacts(result, output_dir)
    payload = result.to_dict()
    payload["artifacts"] = {
        "results": str(results_path),
        "episodes": str(episodes_path),
    }
    return payload


def _write_results_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "checkpoint_step",
        "penalty",
        "mean_raw_score",
        "median_raw_score",
        "std_raw_score",
        "p10_raw_score",
        "p90_raw_score",
        "min_raw_score",
        "max_raw_score",
        "mean_episode_length",
        "mean_life_loss_count",
        "training_q_mean",
        "training_td_error_mean_abs",
        "score_per_life",
        "frames_between_life_losses",
        "time_to_first_life_loss",
        "life_losses_per_1000_steps",
        "wins",
        "ties",
        "losses",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    stage_dir = args.stage_dir
    output_dir = args.output_dir or stage_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_steps = tuple(args.checkpoint_steps or (62_500, 125_000, 187_500, 250_000))
    if any(step < 1 for step in checkpoint_steps):
        raise ValueError("checkpoint steps must be positive")
    manifest = _load_manifest(stage_dir)
    contract = load_evaluation_contract(args.contract)
    validate_breakout_runtime_contract(contract)
    evaluation_config = load_evaluation_config(args.config)
    _validate_eval_config(evaluation_config, contract)
    variants = [variant for variant in manifest["variants"] if isinstance(variant, Mapping)]
    _validate_manifest_fairness(variants)
    if len(variants) < 2:
        raise ValueError("Stage 2 sweep needs baseline and at least one shaped variant")
    baseline_variant = next(
        (variant for variant in variants if float(variant["config"]["life_loss_penalty"]) == 0.0),
        None,
    )
    if baseline_variant is None:
        raise ValueError("Stage 2 sweep is missing the penalty=0.0 baseline")
    env_factory = _environment_factory(contract)
    payloads: dict[str, dict[int, dict[str, Any]]] = {}
    training_summaries: dict[str, Any] = {}
    for variant in variants:
        label = str(variant["label"])
        penalty = float(variant["config"]["life_loss_penalty"])
        run_dir = Path(str(variant["run_dir"]))
        training_summaries[label] = summarize_training_run(run_dir)
        payloads[label] = {}
        for step in checkpoint_steps:
            checkpoint = _variant_checkpoint(variant, step=step)
            payloads[label][step] = _evaluate_checkpoint(
                checkpoint,
                label=label,
                step=step,
                variant=variant,
                evaluation_config=evaluation_config,
                contract=contract,
                device=args.device,
                env_factory=env_factory,
                output_dir=output_dir / "evaluations" / label / str(step),
            )

    comparisons: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    for step in checkpoint_steps:
        step_key = str(step)
        comparisons[step_key] = {}
        baseline_payload = payloads[str(baseline_variant["label"])][step]
        for variant in variants:
            label = str(variant["label"])
            penalty = float(variant["config"]["life_loss_penalty"])
            if label == baseline_variant["label"]:
                comparison = {
                    "baseline": None,
                    "shaped": payloads[label][step]["summary"],
                    "wins": None,
                    "ties": None,
                    "losses": None,
                }
            else:
                comparison = compare_evaluation_payloads(
                    baseline_payload,
                    payloads[label][step],
                )
            comparisons[step_key][label] = comparison
            summary = payloads[label][step]["summary"]
            milestone_name = {
                62_500: "25_percent",
                125_000: "50_percent",
                187_500: "75_percent",
                250_000: "100_percent",
            }.get(step)
            training_milestone = (
                training_summaries[label]["milestones"].get(milestone_name, {})
                if milestone_name is not None
                else {}
            )
            if label == baseline_variant["label"]:
                wins = ties = losses = None
            else:
                wins = comparison["wins"]
                ties = comparison["ties"]
                losses = comparison["losses"]
            table_rows.append(
                {
                    "checkpoint_step": step,
                    "penalty": penalty,
                    "mean_raw_score": summary["mean_return"],
                    "median_raw_score": summary["median_return"],
                    "std_raw_score": summary["std_return"],
                    "p10_raw_score": summary["p10_return"],
                    "p90_raw_score": summary["p90_return"]
                    if "p90_return" in summary
                    else None,
                    "min_raw_score": summary["min_return"],
                    "max_raw_score": summary["max_return"],
                    "mean_episode_length": summary.get("mean_episode_length"),
                    "mean_life_loss_count": summary.get("mean_life_loss_count"),
                    "training_q_mean": training_milestone.get(
                        "recent_q_value_statistics", {}
                    ).get("q_mean", {}).get("mean"),
                    "training_td_error_mean_abs": training_milestone.get(
                        "recent_td_error_statistics", {}
                    ).get("td_error_mean_abs", {}).get("mean"),
                    "score_per_life": summary.get("mean_score_per_life"),
                    "frames_between_life_losses": summary.get(
                        "mean_frames_between_life_losses"
                    ),
                    "time_to_first_life_loss": summary.get(
                        "mean_time_to_first_life_loss"
                    ),
                    "life_losses_per_1000_steps": summary.get(
                        "life_losses_per_1000_steps"
                    ),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )
        (output_dir / f"paired-evaluation-{step}.json").write_text(
            json.dumps(comparisons[step_key], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    sweep_summary = {
        "schema_version": 1,
        "artifact_type": "issue9_reward_shaping_stage2_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "training_seed": manifest["training_seed"],
        "training_transitions": manifest["training_transitions"],
        "checkpoint_steps": list(checkpoint_steps),
        "evaluation_seed_count": evaluation_config.total_episodes,
        "evaluation_score_definition": (
            "raw Atari game reward sum; no clipping and no life-loss penalty"
        ),
        "selection_status": "candidate screening; not final model promotion",
        "variants": variants,
        "training_summaries": training_summaries,
        "comparisons": comparisons,
        "results_table": table_rows,
    }
    summary_path = output_dir / "sweep-summary.json"
    summary_path.write_text(
        json.dumps(sweep_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _write_results_table(output_dir / "stage2-results.csv", table_rows)
    return {
        "summary": str(summary_path),
        "results_table": str(output_dir / "stage2-results.csv"),
        "evaluation_count": len(variants) * len(checkpoint_steps),
        "paired_comparison_count": (len(variants) - 1) * len(checkpoint_steps),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_sweep(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"Reward-shaping sweep evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
