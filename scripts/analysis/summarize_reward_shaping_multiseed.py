"""Aggregate the Issue #9 Stage 3B training-seed replication.

The evaluator already stores paired raw-score comparisons for each checkpoint.
This module validates their common evaluation contract and then aggregates over
training seeds.  It never pools a training seed with an evaluation seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence


BASELINE_LABEL = "penalty-0.0"
SHAPED_LABEL = "penalty-minus-1.0"
CHECKPOINT_STEPS = (250_000, 500_000, 750_000, 1_000_000)
EXPECTED_SCORE_DEFINITION = "raw Atari game reward sum; no clipping and no life-loss penalty"
MILESTONE_KEYS = {
    250_000: "25_percent",
    500_000: "50_percent",
    750_000: "75_percent",
    1_000_000: "100_percent",
}

DEFAULT_EVALUATION_DIRS = {
    2022: Path("experiments/issue-9-reward-shaping/stage3-1m/evaluation"),
    2023: Path("experiments/issue-9-reward-shaping/stage3b-multiseed/seed2023/evaluation"),
    2024: Path("experiments/issue-9-reward-shaping/stage3b-multiseed/seed2024/evaluation"),
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _number(value: Any, *, path: str) -> float:
    parsed = _finite_float(value)
    if parsed is None:
        raise ValueError(f"{path} must be a finite number")
    return parsed


def _mean(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return median(values) if values else None


def _pstd(values: Sequence[float]) -> float | None:
    return pstdev(values) if values else None


def _stat_mean(block: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(block, Mapping):
        return None
    return _finite_float(block.get(key))


def _nested_stat_mean(
    block: Mapping[str, Any] | None,
    outer_key: str,
    inner_key: str = "mean",
) -> float | None:
    if not isinstance(block, Mapping):
        return None
    nested = block.get(outer_key)
    return _stat_mean(nested if isinstance(nested, Mapping) else None, inner_key)


def _config_from_variant(variant: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    config = variant.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{path}: variant config is missing")
    return dict(config)


def _normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"seed", "life_loss_penalty", "contract_id", "contract_path"}
    }


def _variant_map(payload: Mapping[str, Any], *, path: Path) -> dict[str, dict[str, Any]]:
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise ValueError(f"{path}: variants must be a list")
    result: dict[str, dict[str, Any]] = {}
    for raw_variant in variants:
        if not isinstance(raw_variant, Mapping):
            raise ValueError(f"{path}: every variant must be an object")
        label = raw_variant.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{path}: every variant needs a label")
        if label in result:
            raise ValueError(f"{path}: duplicate variant label {label}")
        result[label] = dict(raw_variant)
    for label in (BASELINE_LABEL, SHAPED_LABEL):
        if label not in result:
            raise ValueError(f"{path}: missing required variant {label}")
    return result


def _condition_metrics(block: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    fields = {
        "mean_raw_score": "mean_raw_score",
        "median_raw_score": "median_raw_score",
        "std_raw_score": "std_raw_score",
        "p10_raw_score": "p10_raw_score",
        "p90_raw_score": "p90_raw_score",
        "min_raw_score": "min_raw_score",
        "max_raw_score": "max_raw_score",
        "mean_episode_length": "mean_episode_length",
        "mean_life_loss_count": "mean_life_loss_count",
        "score_per_life": "mean_score_per_life",
        "frames_between_life_losses": "mean_frames_between_life_losses",
        "time_to_first_life_loss": "mean_time_to_first_life_loss",
        "life_losses_per_1000_steps": "life_losses_per_1000_steps",
    }
    result: dict[str, Any] = {}
    for output_key, input_key in fields.items():
        result[output_key] = _number(block.get(input_key), path=f"{path}.{input_key}")
    return result


def _training_milestone(
    training_summary: Mapping[str, Any],
    *,
    step: int,
    path: str,
) -> dict[str, Any]:
    milestones = training_summary.get("milestones")
    if not isinstance(milestones, Mapping):
        raise ValueError(f"{path}: milestones are missing")
    raw_milestone = milestones.get(MILESTONE_KEYS[step])
    if not isinstance(raw_milestone, Mapping):
        raise ValueError(f"{path}: milestone for {step} is missing")
    q_stats = raw_milestone.get("recent_q_value_statistics")
    td_stats = raw_milestone.get("recent_td_error_statistics")
    return {
        "target_step": step,
        "observed_step": raw_milestone.get("observed_step"),
        "completed_episode_count": raw_milestone.get("completed_episode_count"),
        "recent_raw_episode_return_mean": _nested_stat_mean(
            raw_milestone, "recent_raw_score"
        ),
        "recent_training_episode_return_mean": _nested_stat_mean(
            raw_milestone, "recent_training_return"
        ),
        "recent_episode_length_mean": _nested_stat_mean(
            raw_milestone, "recent_episode_length"
        ),
        "recent_life_loss_count_mean": _nested_stat_mean(
            raw_milestone, "recent_life_loss_count"
        ),
        "recent_score_per_life_mean": _nested_stat_mean(
            raw_milestone, "recent_score_per_life"
        ),
        "recent_frames_between_life_losses_mean": _nested_stat_mean(
            raw_milestone, "recent_frames_between_life_losses"
        ),
        "recent_time_to_first_life_loss_mean": _nested_stat_mean(
            raw_milestone, "recent_time_to_first_life_loss"
        ),
        "recent_life_losses_per_1000_steps_mean": _nested_stat_mean(
            raw_milestone, "recent_life_losses_per_1000_steps"
        ),
        "q_mean": _nested_stat_mean(q_stats if isinstance(q_stats, Mapping) else None, "q_mean"),
        "td_error_mean_abs": _nested_stat_mean(
            td_stats if isinstance(td_stats, Mapping) else None,
            "td_error_mean_abs",
        ),
    }


def _training_overall(training_summary: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    raw_score = training_summary.get("raw_score")
    training_return = training_summary.get("training_return")
    q_stats = training_summary.get("q_value_statistics")
    td_stats = training_summary.get("td_error_statistics")
    if not isinstance(raw_score, Mapping) or not isinstance(training_return, Mapping):
        raise ValueError(f"{path}: overall training return summaries are missing")
    return {
        "raw_episode_return_mean": _number(raw_score.get("mean"), path=f"{path}.raw_score.mean"),
        "training_episode_return_mean": _number(
            training_return.get("mean"), path=f"{path}.training_return.mean"
        ),
        "life_loss_count": _number(
            training_summary.get("life_loss_count"), path=f"{path}.life_loss_count"
        ),
        "life_losses_per_1000_steps": _number(
            training_summary.get("life_losses_per_1000_steps"),
            path=f"{path}.life_losses_per_1000_steps",
        ),
        "q_mean": _nested_stat_mean(
            q_stats if isinstance(q_stats, Mapping) else None,
            "q_mean",
        ),
        "td_error_mean_abs": _nested_stat_mean(
            td_stats if isinstance(td_stats, Mapping) else None,
            "td_error_mean_abs",
        ),
    }


def _training_diagnostics(
    training_summaries: Mapping[str, Any],
    *,
    step: int,
    path: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for label in (BASELINE_LABEL, SHAPED_LABEL):
        summary = training_summaries.get(label)
        if not isinstance(summary, Mapping):
            raise ValueError(f"{path}: missing training summary for {label}")
        diagnostics[label] = {
            "milestone": _training_milestone(
                summary,
                step=step,
                path=f"{path}.{label}",
            ),
            "overall": _training_overall(summary, path=f"{path}.{label}"),
            "raw_reward_source": "environment.step reward; never shaped",
            "training_reward_definition": (
                "sign(raw_reward) when reward_clip=true, plus life_loss_penalty "
                "when info['fire_reset_life_loss']=true"
            ),
        }
    return diagnostics


def _checkpoint_record(payload: Mapping[str, Any], *, step: int, path: Path) -> dict[str, Any]:
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError(f"{path}: comparisons are missing")
    raw_checkpoint = comparisons.get(str(step))
    if not isinstance(raw_checkpoint, Mapping):
        raise ValueError(f"{path}: comparison for checkpoint {step} is missing")
    comparison = raw_checkpoint.get(SHAPED_LABEL)
    if not isinstance(comparison, Mapping):
        raise ValueError(f"{path}: paired comparison for {SHAPED_LABEL} is missing")
    baseline = comparison.get("baseline")
    shaped = comparison.get("shaped")
    delta = comparison.get("score_delta")
    if not isinstance(baseline, Mapping) or not isinstance(shaped, Mapping):
        raise ValueError(f"{path}: paired comparison conditions are incomplete")
    if not isinstance(delta, Mapping):
        raise ValueError(f"{path}: paired score delta is missing")
    wins = comparison.get("wins")
    ties = comparison.get("ties")
    losses = comparison.get("losses")
    for name, value in (("wins", wins), ("ties", ties), ("losses", losses)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{path}: {name} must be a non-negative integer")
    if wins + ties + losses != payload.get("evaluation_seed_count"):
        raise ValueError(f"{path}: W/T/L does not cover all evaluation seeds")

    results_table = payload.get("results_table")
    if not isinstance(results_table, list):
        raise ValueError(f"{path}: results_table is missing")
    table_rows = {
        float(row.get("penalty")): row
        for row in results_table
        if isinstance(row, Mapping)
        and row.get("checkpoint_step") == step
        and _finite_float(row.get("penalty")) is not None
    }
    if 0.0 not in table_rows or -1.0 not in table_rows:
        raise ValueError(f"{path}: results_table lacks baseline/-1.0 at {step}")

    def table_diagnostics(row: Mapping[str, Any], row_path: str) -> dict[str, Any]:
        return {
            "training_q_mean": _number(
                row.get("training_q_mean"), path=f"{row_path}.training_q_mean"
            ),
            "training_td_error_mean_abs": _number(
                row.get("training_td_error_mean_abs"),
                path=f"{row_path}.training_td_error_mean_abs",
            ),
        }

    return {
        "checkpoint_step": step,
        "baseline": {
            **_condition_metrics(baseline, path=f"{path}.baseline"),
            **table_diagnostics(table_rows[0.0], f"{path}.results_table.baseline"),
        },
        "minus1": {
            **_condition_metrics(shaped, path=f"{path}.shaped"),
            **table_diagnostics(table_rows[-1.0], f"{path}.results_table.minus1"),
        },
        "paired": {
            "delta_mean_raw_score": _number(delta.get("mean"), path=f"{path}.score_delta.mean"),
            "delta_median_raw_score": _number(
                delta.get("median"), path=f"{path}.score_delta.median"
            ),
            "delta_std_raw_score": _number(delta.get("std"), path=f"{path}.score_delta.std"),
            "delta_p10_raw_score": _number(delta.get("p10"), path=f"{path}.score_delta.p10"),
            "delta_p90_raw_score": _number(delta.get("p90"), path=f"{path}.score_delta.p90"),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
    }


def _validate_common_evaluation_contract(
    payloads: Mapping[int, Mapping[str, Any]],
    *,
    paths: Mapping[int, Path],
) -> dict[str, Any]:
    first_seed = min(payloads)
    first = payloads[first_seed]
    eval_seeds = first.get("eval_seeds")
    if not isinstance(eval_seeds, list) or len(eval_seeds) != 50:
        raise ValueError(f"{paths[first_seed]}: expected 50 evaluation seeds")
    if len(set(eval_seeds)) != 50:
        raise ValueError(f"{paths[first_seed]}: evaluation seeds are not unique")
    if first.get("evaluation_episodes_per_seed") != 1:
        raise ValueError(f"{paths[first_seed]}: expected one episode per evaluation seed")
    if first.get("evaluation_seed_count") != 50:
        raise ValueError(f"{paths[first_seed]}: evaluation_seed_count must be 50")
    if first.get("contract_id") != "day15-breakout-evaluation-v2-fire-reset":
        raise ValueError(f"{paths[first_seed]}: unexpected contract_id")
    if first.get("evaluation_score_definition") != EXPECTED_SCORE_DEFINITION:
        raise ValueError(f"{paths[first_seed]}: evaluation is not raw-score-only")
    if first.get("checkpoint_steps") != list(CHECKPOINT_STEPS):
        raise ValueError(f"{paths[first_seed]}: checkpoint steps are not 250/500/750/1M")
    runtime_validation: dict[str, Any] = {}
    for seed, payload in payloads.items():
        for field in (
            "eval_seeds",
            "evaluation_episodes_per_seed",
            "evaluation_seed_count",
            "contract_id",
            "evaluation_score_definition",
            "checkpoint_steps",
        ):
            if payload.get(field) != first.get(field):
                raise ValueError(f"{paths[seed]}: {field} differs from seed {first_seed}")
        if payload.get("training_seed") != seed:
            raise ValueError(
                f"{paths[seed]}: summary training_seed does not match requested seed {seed}"
            )
        if payload.get("training_transitions") != 1_000_000:
            raise ValueError(f"{paths[seed]}: training budget must be 1M")
        detail_checks = 0
        for label in (BASELINE_LABEL, SHAPED_LABEL):
            for step in CHECKPOINT_STEPS:
                detail_path = (
                    paths[seed].parent
                    / "evaluations"
                    / label
                    / str(step)
                    / "results.json"
                )
                detail = _read_json(detail_path)
                if detail.get("evaluation_epsilon") != 0.0:
                    raise ValueError(f"{detail_path}: evaluation_epsilon must be 0.0")
                if detail.get("evaluation_seeds") != eval_seeds:
                    raise ValueError(f"{detail_path}: evaluation seeds differ")
                if detail.get("episodes_per_seed") != 1:
                    raise ValueError(f"{detail_path}: episodes_per_seed must be 1")
                if detail.get("total_episodes") != 50:
                    raise ValueError(f"{detail_path}: total_episodes must be 50")
                training_detail = detail.get("training")
                checkpoint_detail = detail.get("checkpoint")
                if not isinstance(training_detail, Mapping) or not isinstance(
                    checkpoint_detail, Mapping
                ):
                    raise ValueError(f"{detail_path}: provenance objects are missing")
                if checkpoint_detail.get("contract_id") != first["contract_id"]:
                    raise ValueError(f"{detail_path}: contract_id differs")
                if (
                    training_detail.get("training_seed") != seed
                    or checkpoint_detail.get("step") != step
                ):
                    raise ValueError(f"{detail_path}: training/checkpoint provenance differs")
                metadata_detail = detail.get("metadata")
                score_definition = training_detail.get("evaluation_score_definition")
                if isinstance(metadata_detail, Mapping):
                    score_definition = score_definition or metadata_detail.get(
                        "score_definition"
                    )
                if score_definition != EXPECTED_SCORE_DEFINITION:
                    raise ValueError(f"{detail_path}: evaluation is not raw-score-only")
                detail_checks += 1
        runtime_validation[str(seed)] = {
            "evaluation_epsilon": 0.0,
            "validated_result_files": detail_checks,
            "expected_result_files": len((BASELINE_LABEL, SHAPED_LABEL))
            * len(CHECKPOINT_STEPS),
        }
    return {
        "evaluation_seed_count": 50,
        "evaluation_episodes_per_seed": 1,
        "eval_seeds": list(eval_seeds),
        "contract_id": first["contract_id"],
        "contract_path": first.get("contract_path"),
        "evaluation_score_definition": first["evaluation_score_definition"],
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "runtime_result_validation": runtime_validation,
    }


def _validate_training_fairness(
    payloads: Mapping[int, Mapping[str, Any]],
    *,
    paths: Mapping[int, Path],
) -> dict[str, Any]:
    reference: dict[str, Any] | None = None
    reference_seed: int | None = None
    config_differences: list[dict[str, Any]] = []
    for seed, payload in payloads.items():
        variants = _variant_map(payload, path=paths[seed])
        baseline_config = _config_from_variant(
            variants[BASELINE_LABEL], path=f"{paths[seed]}.{BASELINE_LABEL}"
        )
        shaped_config = _config_from_variant(
            variants[SHAPED_LABEL], path=f"{paths[seed]}.{SHAPED_LABEL}"
        )
        if _normalized_config(baseline_config) != _normalized_config(shaped_config):
            config_differences.append(
                {
                    "training_seed": seed,
                    "scope": "baseline_vs_minus1",
                    "reason": "configs differ beyond seed/life_loss_penalty",
                }
            )
        if baseline_config.get("seed") != seed or shaped_config.get("seed") != seed:
            raise ValueError(f"{paths[seed]}: nested config seed mismatch")
        if baseline_config.get("life_loss_penalty") != 0.0:
            raise ValueError(f"{paths[seed]}: baseline penalty must be 0.0")
        if shaped_config.get("life_loss_penalty") != -1.0:
            raise ValueError(f"{paths[seed]}: shaped penalty must be -1.0")
        normalized = _normalized_config(baseline_config)
        if reference is None:
            reference = normalized
            reference_seed = seed
        elif normalized != reference:
            config_differences.append(
                {
                    "training_seed": seed,
                    "scope": f"seed_{reference_seed}_vs_seed_{seed}",
                    "reason": "training configs differ beyond seed/life_loss_penalty",
                }
            )
    if config_differences:
        raise ValueError(json.dumps(config_differences, ensure_ascii=False))
    return {
        "training_config_equivalent_except_seed_and_life_loss_penalty": True,
        "training_seeds": sorted(payloads),
        "variants": [BASELINE_LABEL, SHAPED_LABEL],
    }


def _cross_curve(records: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for step in CHECKPOINT_STEPS:
        checkpoints = [records[seed]["checkpoints"][str(step)] for seed in sorted(records)]
        baseline = [item["baseline"] for item in checkpoints]
        shaped = [item["minus1"] for item in checkpoints]
        paired = [item["paired"] for item in checkpoints]
        curve.append(
            {
                "checkpoint_step": step,
                "baseline_mean_raw_score_across_training_seeds": _mean(
                    [item["mean_raw_score"] for item in baseline]
                ),
                "minus1_mean_raw_score_across_training_seeds": _mean(
                    [item["mean_raw_score"] for item in shaped]
                ),
                "baseline_median_raw_score_across_training_seeds": _mean(
                    [item["median_raw_score"] for item in baseline]
                ),
                "minus1_median_raw_score_across_training_seeds": _mean(
                    [item["median_raw_score"] for item in shaped]
                ),
                "mean_of_paired_delta_mean": _mean(
                    [item["delta_mean_raw_score"] for item in paired]
                ),
                "median_of_paired_delta_mean": _median(
                    [item["delta_mean_raw_score"] for item in paired]
                ),
                "training_seed_wins": sum(
                    item["delta_mean_raw_score"] > 0 for item in paired
                ),
                "training_seed_ties": sum(
                    item["delta_mean_raw_score"] == 0 for item in paired
                ),
                "training_seed_losses": sum(
                    item["delta_mean_raw_score"] < 0 for item in paired
                ),
                "pooled_episode_wins": sum(item["wins"] for item in paired),
                "pooled_episode_ties": sum(item["ties"] for item in paired),
                "pooled_episode_losses": sum(item["losses"] for item in paired),
            }
        )
    return curve


def _survival_curve(records: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "mean_episode_length",
        "score_per_life",
        "frames_between_life_losses",
        "time_to_first_life_loss",
        "life_losses_per_1000_steps",
        "mean_life_loss_count",
    )
    result: list[dict[str, Any]] = []
    for step in CHECKPOINT_STEPS:
        checkpoints = [records[seed]["checkpoints"][str(step)] for seed in sorted(records)]
        row: dict[str, Any] = {"checkpoint_step": step}
        for field in fields:
            baseline = [item["baseline"][field] for item in checkpoints]
            shaped = [item["minus1"][field] for item in checkpoints]
            row[f"baseline_{field}_across_training_seeds"] = _mean(baseline)
            row[f"minus1_{field}_across_training_seeds"] = _mean(shaped)
            row[f"minus1_delta_{field}"] = _mean(shaped) - _mean(baseline)
        result.append(row)
    return result


def _training_curve(records: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "recent_raw_episode_return_mean",
        "recent_training_episode_return_mean",
        "q_mean",
        "td_error_mean_abs",
        "recent_episode_length_mean",
        "recent_life_loss_count_mean",
        "recent_score_per_life_mean",
        "recent_frames_between_life_losses_mean",
        "recent_time_to_first_life_loss_mean",
        "recent_life_losses_per_1000_steps_mean",
    )
    result: list[dict[str, Any]] = []
    for step in CHECKPOINT_STEPS:
        row: dict[str, Any] = {"checkpoint_step": step}
        for label, output_label in (
            (BASELINE_LABEL, "baseline"),
            (SHAPED_LABEL, "minus1"),
        ):
            for field in fields:
                values = [
                    records[seed]["training"][str(step)][label]["milestone"].get(field)
                    for seed in sorted(records)
                ]
                finite = [value for value in values if _finite_float(value) is not None]
                row[f"{output_label}_{field}_across_training_seeds"] = _mean(
                    [float(value) for value in finite]
                )
        result.append(row)
    return result


def _learning_curve_interpretation(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_step = {int(row["checkpoint_step"]): row for row in curve}
    early = _number(by_step[250_000]["mean_of_paired_delta_mean"], path="learning_curve.250k")
    mid_values = [
        _number(by_step[step]["mean_of_paired_delta_mean"], path=f"learning_curve.{step}")
        for step in (500_000, 750_000)
    ]
    late = _number(by_step[1_000_000]["mean_of_paired_delta_mean"], path="learning_curve.1m")
    mid = mean(mid_values)
    return {
        "early_250k_delta_mean": early,
        "mid_500k_750k_delta_mean": mid,
        "late_1m_delta_mean": late,
        "early_advantage_observed": early > 0,
        "mid_advantage_observed": mid > 0,
        "late_degradation_observed": late < 0,
        "early_faster_mid_advantage_late_degradation_pattern": (
            early > 0 and mid > 0 and late < 0
        ),
        "late_recovery_after_mid_advantage": early > 0 and mid > 0 and late > 0,
    }


def build_multiseed_summary(
    evaluation_dirs: Mapping[int, Path] = DEFAULT_EVALUATION_DIRS,
) -> dict[str, Any]:
    payloads: dict[int, dict[str, Any]] = {}
    paths: dict[int, Path] = {}
    for seed, evaluation_dir in sorted(evaluation_dirs.items()):
        path = evaluation_dir / "sweep-summary.json"
        payloads[seed] = _read_json(path)
        paths[seed] = path
    common_contract = _validate_common_evaluation_contract(payloads, paths=paths)
    fairness = _validate_training_fairness(payloads, paths=paths)

    records: dict[int, dict[str, Any]] = {}
    final_table: list[dict[str, Any]] = []
    for seed, payload in sorted(payloads.items()):
        variants = _variant_map(payload, path=paths[seed])
        training_summaries = payload.get("training_summaries")
        if not isinstance(training_summaries, Mapping):
            raise ValueError(f"{paths[seed]}: training_summaries are missing")
        checkpoints: dict[str, Any] = {}
        training: dict[str, Any] = {}
        for step in CHECKPOINT_STEPS:
            record = _checkpoint_record(payload, step=step, path=paths[seed])
            checkpoints[str(step)] = record
            training[str(step)] = _training_diagnostics(
                training_summaries,
                step=step,
                path=str(paths[seed]),
            )
        final = checkpoints[str(CHECKPOINT_STEPS[-1])]
        final_table.append(
            {
                "training_seed": seed,
                "baseline_mean_raw_score": final["baseline"]["mean_raw_score"],
                "minus1_mean_raw_score": final["minus1"]["mean_raw_score"],
                "delta_mean_raw_score": final["paired"]["delta_mean_raw_score"],
                "baseline_median_raw_score": final["baseline"]["median_raw_score"],
                "minus1_median_raw_score": final["minus1"]["median_raw_score"],
                "delta_median_raw_score": final["paired"]["delta_median_raw_score"],
                "baseline_std_raw_score": final["baseline"]["std_raw_score"],
                "minus1_std_raw_score": final["minus1"]["std_raw_score"],
                "baseline_p10_raw_score": final["baseline"]["p10_raw_score"],
                "minus1_p10_raw_score": final["minus1"]["p10_raw_score"],
                "baseline_p90_raw_score": final["baseline"]["p90_raw_score"],
                "minus1_p90_raw_score": final["minus1"]["p90_raw_score"],
                "wins": final["paired"]["wins"],
                "ties": final["paired"]["ties"],
                "losses": final["paired"]["losses"],
            }
        )
        records[seed] = {
            "training_seed": seed,
            "evaluation_dir": evaluation_dirs[seed].as_posix(),
            "config_paths": {
                BASELINE_LABEL: variants[BASELINE_LABEL].get("config_path"),
                SHAPED_LABEL: variants[SHAPED_LABEL].get("config_path"),
            },
            "checkpoints": checkpoints,
            "training": training,
            "training_overall": {
                label: _training_overall(
                    training_summaries[label],
                    path=f"{paths[seed]}.training_summaries.{label}",
                )
                for label in (BASELINE_LABEL, SHAPED_LABEL)
            },
        }

    curve = _cross_curve(records)
    survival = _survival_curve(records)
    training_curve = _training_curve(records)
    final_deltas = [row["delta_mean_raw_score"] for row in final_table]
    final_median_deltas = [row["delta_median_raw_score"] for row in final_table]
    pooled_wins = sum(row["wins"] for row in final_table)
    pooled_ties = sum(row["ties"] for row in final_table)
    pooled_losses = sum(row["losses"] for row in final_table)
    final_cross_seed = {
        "baseline_mean_raw_score": _mean(
            [row["baseline_mean_raw_score"] for row in final_table]
        ),
        "minus1_mean_raw_score": _mean(
            [row["minus1_mean_raw_score"] for row in final_table]
        ),
        "mean_effect": _mean(final_deltas),
        "median_effect": _median(final_deltas),
        "std_of_effect": _pstd(final_deltas),
        "baseline_median_raw_score_mean_across_training_seeds": _mean(
            [row["baseline_median_raw_score"] for row in final_table]
        ),
        "minus1_median_raw_score_mean_across_training_seeds": _mean(
            [row["minus1_median_raw_score"] for row in final_table]
        ),
        "mean_of_paired_median_effect": _mean(final_median_deltas),
        "training_seeds_won_by_minus1": sum(delta > 0 for delta in final_deltas),
        "training_seed_ties": sum(delta == 0 for delta in final_deltas),
        "training_seeds_lost_by_minus1": sum(delta < 0 for delta in final_deltas),
        "pooled_episode_wins": pooled_wins,
        "pooled_episode_ties": pooled_ties,
        "pooled_episode_losses": pooled_losses,
        "pooled_episode_win_tie_loss_total": pooled_wins + pooled_ties + pooled_losses,
    }
    final_survival = survival[-1]

    return {
        "schema_version": 1,
        "artifact_type": "issue9_reward_shaping_stage3b_multiseed_summary",
        "training_seeds": sorted(records),
        "training_seed_count": len(records),
        "training_algorithm": "double_dqn",
        "training_architecture": "dueling",
        "training_budget": 1_000_000,
        "variants": [BASELINE_LABEL, SHAPED_LABEL],
        "reward_design": {
            "baseline": "brick reward -> +1; life loss -> no additional penalty",
            "experiment": "brick reward -> +1; life loss -> additional -1.0",
            "life_loss_signal": "info['fire_reset_life_loss']",
            "raw_reward_and_training_reward_separate": True,
            "evaluation_primary_metric": "raw Atari score",
        },
        "common_evaluation_contract": common_contract,
        "training_fairness": fairness,
        "per_training_seed": records,
        "final_comparison_table": final_table,
        "cross_training_seed_final": final_cross_seed,
        "learning_curve": curve,
        "learning_curve_interpretation": _learning_curve_interpretation(curve),
        "survival_comparison": {
            "by_checkpoint": survival,
            "final_checkpoint": final_survival,
        },
        "training_reward_raw_reward_q_td_diagnostics": {
            "by_checkpoint": training_curve,
            "definition": (
                "recent raw episode return is the unshaped environment reward sum; "
                "recent training episode return includes the configured life-loss penalty"
            ),
        },
        "decision": {
            "stage3b_final_raw_score_advantage_reproducible": (
                final_cross_seed["mean_effect"] is not None
                and final_cross_seed["mean_effect"] > 0
                and final_cross_seed["training_seeds_won_by_minus1"]
                > final_cross_seed["training_seeds_lost_by_minus1"]
            ),
            "minus1_learning_dynamics_replicated_as_early_advantage_then_late_degradation": (
                _learning_curve_interpretation(curve)[
                    "early_faster_mid_advantage_late_degradation_pattern"
                ]
            ),
            "candidate_for_2_5m_final_validation": SHAPED_LABEL,
            "2_5m_action": (
                "If proceeding, validate -1.0 against the 0.0 baseline as the paired "
                "control; do not start 2.5M automatically from this artifact."
            ),
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed2022-evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIRS[2022])
    parser.add_argument("--seed2023-evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIRS[2023])
    parser.add_argument("--seed2024-evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIRS[2024])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/issue-9-reward-shaping/stage3b-multiseed"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation_dirs = {
        2022: args.seed2022_evaluation_dir,
        2023: args.seed2023_evaluation_dir,
        2024: args.seed2024_evaluation_dir,
    }
    summary = build_multiseed_summary(evaluation_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "cross-seed-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "cross-seed-summary.csv", summary["final_comparison_table"])
    _write_csv(args.output_dir / "learning-curve.csv", summary["learning_curve"])
    _write_csv(
        args.output_dir / "survival-comparison.csv",
        summary["survival_comparison"]["by_checkpoint"],
    )
    _write_csv(
        args.output_dir / "training-diagnostics.csv",
        summary["training_reward_raw_reward_q_td_diagnostics"]["by_checkpoint"],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
