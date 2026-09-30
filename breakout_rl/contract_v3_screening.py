"""Assess screening evidence from Contract v3 baseline checkpoints."""

from collections.abc import Mapping, Sequence
from typing import Any


def build_screening_assessment(
    milestone_records: Sequence[Mapping[str, Any]],
    *,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not milestone_records:
        raise ValueError("screening assessment requires evaluated milestones")
    clear_checkpoints = [
        int(item["agent_steps"])
        for item in milestone_records
        if int(item["summary"]["clear_count"]) > 0
    ]
    score_progression = []
    for item in milestone_records:
        summary = item["summary"]
        score_progression.append(
            {
                "agent_steps": int(item["agent_steps"]),
                "mean_raw_score": float(summary["mean_return"]),
                "median_raw_score": float(summary["median_return"]),
                "max_raw_score": float(summary["max_return"]),
                "clear_count": int(summary["clear_count"]),
                "clear_rate": float(summary["clear_rate"]),
                "survival_metrics": {
                    key: float(summary[key])
                    for key in (
                        "mean_life_loss_count",
                        "life_losses_per_1000_steps",
                        "mean_score_per_life",
                        "mean_frames_between_life_losses",
                        "mean_time_to_first_life_loss",
                    )
                },
            }
        )
    reference = score_progression[0]
    latest = score_progression[-1]
    score_metric_keys = ("mean_raw_score", "median_raw_score", "max_raw_score")
    consistent_score_progression = len(score_progression) > 1 and all(
        all(
            earlier[key] < later[key]
            for earlier, later in zip(score_progression, score_progression[1:])
        )
        for key in score_metric_keys
    )
    higher_clear_count_or_rate = any(
        later["clear_count"] > reference["clear_count"]
        or later["clear_rate"] > reference["clear_rate"]
        for later in score_progression[1:]
    )
    best_later = max(
        score_progression[1:] or [reference], key=lambda item: item["max_raw_score"]
    )
    higher_max_score = best_later["max_raw_score"] > reference["max_raw_score"]
    survival_directions = {
        "mean_life_loss_count": "lower",
        "life_losses_per_1000_steps": "lower",
        "mean_score_per_life": "higher",
        "mean_frames_between_life_losses": "higher",
        "mean_time_to_first_life_loss": "higher",
    }
    survival_comparison = {
        key: {
            "reference": reference["survival_metrics"][key],
            "latest": latest["survival_metrics"][key],
            "favorable_at_latest": (
                latest["survival_metrics"][key] < reference["survival_metrics"][key]
                if direction == "lower"
                else latest["survival_metrics"][key]
                > reference["survival_metrics"][key]
            ),
        }
        for key, direction in survival_directions.items()
    }
    favorable_survival_metrics = sum(
        item["favorable_at_latest"] for item in survival_comparison.values()
    )
    if favorable_survival_metrics == len(survival_comparison):
        survival_status = "positive_signal"
    elif favorable_survival_metrics:
        survival_status = "mixed"
    else:
        survival_status = "not_observed"
    high_score_samples = (
        controls.get("high_score", {}).get("sample_count")
        if controls is not None
        else None
    )
    first_clear = {
        "status": "met" if clear_checkpoints else "not_met",
        "clear_checkpoints": clear_checkpoints,
    }
    criteria = {
        "first_verified_clear": first_clear,
        "higher_clear_count_or_rate": {
            "status": "observed" if higher_clear_count_or_rate else "not_observed",
            "checkpoint_values": [
                {
                    "agent_steps": item["agent_steps"],
                    "clear_count": item["clear_count"],
                    "clear_rate": item["clear_rate"],
                }
                for item in score_progression
            ],
        },
        "materially_higher_max_score": {
            "status": "observed_increase" if higher_max_score else "not_observed",
            "reference_checkpoint_agent_steps": reference["agent_steps"],
            "reference_max_raw_score": reference["max_raw_score"],
            "best_later_checkpoint_agent_steps": best_later["agent_steps"],
            "best_later_max_raw_score": best_later["max_raw_score"],
            "increase_points": best_later["max_raw_score"] - reference["max_raw_score"],
            "interpretation": (
                "within-run checkpoint change; materiality is judged from the full "
                "raw-score progression and is not an independent control comparison"
            ),
        },
        "consistent_raw_score_progression": {
            "status": "observed" if consistent_score_progression else "not_observed",
            "required_metrics": list(score_metric_keys),
            "checkpoints": score_progression,
        },
        "materially_better_late_game_survival": {
            "status": survival_status,
            "reference_checkpoint_agent_steps": reference["agent_steps"],
            "latest_checkpoint_agent_steps": latest["agent_steps"],
            "latest_vs_reference": survival_comparison,
            "interpretation": (
                "descriptive endpoint checkpoint comparison; one survival metric "
                "moving against the reference keeps this signal mixed"
                if survival_status == "mixed"
                else "descriptive endpoint checkpoint comparison"
            ),
        },
        "visibly_improved_high_speed_paddle_control": {
            "status": "not_assessed",
            "reason": (
                "no high-score training transitions were available"
                if high_score_samples == 0
                else "no checkpoint-specific paddle-contact control comparison is available"
            ),
            "high_score_training_sample_count": high_score_samples,
        },
        "better_performance_at_matched_emulator_frame_budget": {
            "status": "not_assessed",
            "reason": "no historical matched-frame comparator is included in this run",
        },
        "independent_matched_contract_v3_control": {
            "status": "not_assessed",
            "reason": "this issue runs one training lifecycle and no separate control",
        },
    }
    promising = bool(clear_checkpoints) or consistent_score_progression
    if clear_checkpoints:
        basis = "at least one canonical Contract v3 evaluation episode recorded a verified clear"
    elif consistent_score_progression:
        baseline_values = " / ".join(
            f"{reference[key]:.2f}" for key in score_metric_keys
        )
        latest_values = " / ".join(f"{latest[key]:.2f}" for key in score_metric_keys)
        basis = (
            "raw score mean / median / max improved at each of the three canonical "
            "Contract v3 checkpoints on the same 15 evaluation seeds "
            f"({reference['agent_steps']:,} steps: {baseline_values}; "
            f"{latest['agent_steps']:,} steps: {latest_values}); no verified clear "
            "was recorded"
        )
    else:
        basis = (
            "no verified clear or consistent raw-score progression across the "
            "canonical checkpoints was observed; the single-run results do not "
            "establish comparative survival, control, or matched-frame evidence"
        )
    if promising:
        return {
            "status": "promising",
            "promising": True,
            "decision": "promote_for_follow_up",
            "basis": basis,
            "criteria": criteria,
            "clear_checkpoints": clear_checkpoints,
            "score_progression": score_progression,
            "second_training_seed_started": False,
            "next_algorithm_started": False,
        }
    return {
        "status": "inconclusive",
        "promising": None,
        "decision": "hold_for_more_evidence",
        "basis": basis,
        "criteria": criteria,
        "clear_checkpoints": [],
        "score_progression": score_progression,
        "second_training_seed_started": False,
        "next_algorithm_started": False,
    }
