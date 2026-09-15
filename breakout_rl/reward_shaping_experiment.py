"""Comparison and artifact helpers for the Issue #9 reward experiment."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Iterable, Mapping, Sequence

from breakout_rl.evaluation_artifacts import validate_episode_rows
from breakout_rl.training.config import DQNConfig


EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_ARTIFACT_TYPE = "issue9_reward_shaping"
MILESTONES: tuple[tuple[str, float], ...] = (
    ("25_percent", 0.25),
    ("50_percent", 0.50),
    ("75_percent", 0.75),
    ("100_percent", 1.00),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def score_statistics(values: Iterable[float]) -> dict[str, Any]:
    """Return the Issue #9 score statistics, including P10/P90."""

    parsed = [float(value) for value in values]
    if not parsed:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError("statistics require finite values")
    ordered = sorted(parsed)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)

    return {
        "count": len(parsed),
        "mean": float(fmean(parsed)),
        "median": float(median(parsed)),
        "std": float(pstdev(parsed)),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
        "min": float(min(parsed)),
        "max": float(max(parsed)),
    }


def _episode_summary(
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scores = [float(row["episode_return"]) for row in rows]
    life_losses = [int(row.get("life_loss_count", 0)) for row in rows]
    lengths = [int(row["episode_length"]) for row in rows]
    score_per_life = [float(row["score_per_life"]) for row in rows]
    frames_between = [
        float(row["frames_between_life_losses"])
        for row in rows
        if row.get("frames_between_life_losses") is not None
    ]
    first_loss_times = [
        float(row["time_to_first_life_loss"])
        for row in rows
        if row.get("time_to_first_life_loss") is not None
    ]
    score_summary = score_statistics(scores)
    return {
        "model_id": payload.get("model_id"),
        "policy_type": payload.get("policy_type"),
        "episode_count": len(rows),
        "raw_score": score_summary,
        "mean_raw_score": score_summary["mean"],
        "median_raw_score": score_summary["median"],
        "std_raw_score": score_summary["std"],
        "p10_raw_score": score_summary["p10"],
        "p90_raw_score": score_summary["p90"],
        "min_raw_score": score_summary["min"],
        "max_raw_score": score_summary["max"],
        "life_loss_count": int(sum(life_losses)),
        "mean_life_loss_count": float(fmean(life_losses)),
        "life_loss_statistics": score_statistics(life_losses),
        "mean_episode_length": float(fmean(lengths)),
        "episode_length_statistics": score_statistics(lengths),
        "score_per_life": score_statistics(score_per_life),
        "mean_score_per_life": float(fmean(score_per_life)),
        "mean_frames_between_life_losses": (
            float(fmean(frames_between)) if frames_between else None
        ),
        "mean_time_to_first_life_loss": (
            float(fmean(first_loss_times)) if first_loss_times else None
        ),
        "life_losses_per_1000_steps": float(
            sum(life_losses) / max(1, sum(lengths)) * 1000.0
        ),
        "score_definition": "raw Atari game reward sum; no training shaping",
    }


def compare_evaluation_payloads(
    baseline_payload: Mapping[str, Any],
    shaped_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Pair two raw-score evaluation artifacts by evaluation seed and episode."""

    baseline_rows = validate_episode_rows(
        baseline_payload,
        source="baseline evaluation",
        require_complete=True,
    )
    shaped_rows = validate_episode_rows(
        shaped_payload,
        source="life-loss-penalty evaluation",
        require_complete=True,
    )
    baseline_by_identity = {
        (int(row["evaluation_seed"]), int(row["episode_index"])): row
        for row in baseline_rows
    }
    shaped_by_identity = {
        (int(row["evaluation_seed"]), int(row["episode_index"])): row
        for row in shaped_rows
    }
    if set(baseline_by_identity) != set(shaped_by_identity):
        raise ValueError("baseline and shaped evaluations do not share the same seed set")

    paired_rows: list[dict[str, Any]] = []
    for identity in sorted(baseline_by_identity):
        baseline = baseline_by_identity[identity]
        shaped = shaped_by_identity[identity]
        baseline_score = float(baseline["episode_return"])
        shaped_score = float(shaped["episode_return"])
        delta = shaped_score - baseline_score
        paired_rows.append(
            {
                "evaluation_seed": identity[0],
                "episode_index": identity[1],
                "baseline_raw_score": baseline_score,
                "shaped_raw_score": shaped_score,
                "score_delta": delta,
                "baseline_life_loss_count": int(baseline.get("life_loss_count", 0)),
                "shaped_life_loss_count": int(shaped.get("life_loss_count", 0)),
                "baseline_score_per_life": float(baseline["score_per_life"]),
                "shaped_score_per_life": float(shaped["score_per_life"]),
                "baseline_frames_between_life_losses": baseline.get(
                    "frames_between_life_losses"
                ),
                "shaped_frames_between_life_losses": shaped.get(
                    "frames_between_life_losses"
                ),
                "baseline_time_to_first_life_loss": baseline.get(
                    "time_to_first_life_loss"
                ),
                "shaped_time_to_first_life_loss": shaped.get(
                    "time_to_first_life_loss"
                ),
                "baseline_life_losses_per_1000_steps": float(
                    baseline["life_losses_per_1000_steps"]
                ),
                "shaped_life_losses_per_1000_steps": float(
                    shaped["life_losses_per_1000_steps"]
                ),
                "baseline_episode_length": int(baseline["episode_length"]),
                "shaped_episode_length": int(shaped["episode_length"]),
            }
        )

    deltas = [float(row["score_delta"]) for row in paired_rows]
    wins = sum(delta > 0.0 for delta in deltas)
    ties = sum(delta == 0.0 for delta in deltas)
    losses = sum(delta < 0.0 for delta in deltas)
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": "issue9_reward_shaping_paired_evaluation",
        "evaluation_seed_count": len(paired_rows),
        "baseline": _episode_summary(baseline_payload, baseline_rows),
        "shaped": _episode_summary(shaped_payload, shaped_rows),
        "score_delta_definition": "shaped_model_score_minus_baseline_model_score",
        "score_delta": score_statistics(deltas),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "paired_rows": paired_rows,
    }


def _read_run(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    path = Path(run_dir)
    config_path = path / "config.json"
    summary_path = path / "summary.json"
    metrics_path = path / "metrics.csv"
    if not config_path.is_file() or not summary_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"training run is missing artifacts: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError(f"training run artifacts must contain JSON objects: {path}")
    with metrics_path.open("r", newline="", encoding="utf-8") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    return config, summary, rows


def _completed_training_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    completed: list[dict[str, float]] = []
    for row in rows:
        step = _float(row.get("global_step"))
        raw_score = _float(row.get("raw_episode_return"))
        if step is None or raw_score is None:
            continue
        training_return = _float(row.get("training_episode_return"))
        episode_length = _float(row.get("episode_length"))
        episode_life_losses = _float(row.get("episode_life_loss_count"))
        score_per_life = _float(row.get("score_per_life"))
        frames_between = _float(row.get("episode_frames_between_life_losses"))
        time_to_first = _float(row.get("episode_time_to_first_life_loss"))
        life_loss_rate = _float(row.get("episode_life_losses_per_1000_steps"))
        parsed_life_losses = 0.0 if episode_life_losses is None else episode_life_losses
        completed.append(
            {
                "global_step": step,
                "raw_episode_return": raw_score,
                "training_episode_return": (
                    raw_score if training_return is None else training_return
                ),
                "episode_length": (
                    0.0 if episode_length is None else episode_length
                ),
                "episode_life_loss_count": (
                    0.0 if episode_life_losses is None else episode_life_losses
                ),
                "score_per_life": (
                    raw_score / max(1.0, parsed_life_losses)
                    if score_per_life is None
                    else score_per_life
                ),
                "frames_between_life_losses": (
                    float("nan") if frames_between is None else frames_between
                ),
                "time_to_first_life_loss": (
                    float("nan") if time_to_first is None else time_to_first
                ),
                "life_losses_per_1000_steps": (
                    (
                        0.0
                        if episode_length in (None, 0)
                        else parsed_life_losses / episode_length * 1000.0
                    )
                    if life_loss_rate is None
                    else life_loss_rate
                ),
            }
        )
    return completed


def _recent_stats(
    rows: Sequence[Mapping[str, float]],
    *,
    field: str,
    window: int,
) -> dict[str, Any]:
    values = [
        float(row[field])
        for row in rows[-window:]
        if math.isfinite(float(row[field]))
    ]
    return score_statistics(values)


def summarize_training_run(
    run_dir: str | Path,
    *,
    recent_window: int = 20,
) -> dict[str, Any]:
    """Extract raw/training returns and life-loss curves from one run."""

    if recent_window < 1:
        raise ValueError("recent_window must be positive")
    config, summary, rows = _read_run(run_dir)
    completed = _completed_training_rows(rows)
    expected_steps = _float(config.get("total_steps"))
    if expected_steps is None:
        expected_steps = _float(summary.get("total_steps"))
    final_life_loss_count = 0
    final_penalty_total = 0.0
    for row in reversed(rows):
        parsed_count = _float(row.get("life_loss_count"))
        parsed_penalty = _float(row.get("life_loss_penalty_total"))
        if parsed_count is not None:
            final_life_loss_count = int(parsed_count)
            break
        if parsed_penalty is not None:
            final_penalty_total = parsed_penalty
    for row in reversed(rows):
        parsed_penalty = _float(row.get("life_loss_penalty_total"))
        if parsed_penalty is not None:
            final_penalty_total = parsed_penalty
            break

    raw_values = [row["raw_episode_return"] for row in completed]
    training_values = [row["training_episode_return"] for row in completed]
    score_per_life_values = [row["score_per_life"] for row in completed]
    recent = completed[-recent_window:]
    curve = [dict(row) for row in completed]
    milestones: dict[str, Any] = {}
    for name, fraction in MILESTONES:
        target = expected_steps * fraction if expected_steps is not None else None
        eligible = [
            row for row in completed if target is None or row["global_step"] <= target
        ]
        recent_eligible = eligible[-recent_window:]
        milestones[name] = {
            "target_step": None if target is None else int(target),
            "observed_step": (
                None if not eligible else int(eligible[-1]["global_step"])
            ),
            "completed_episode_count": len(eligible),
            "recent_raw_score": _recent_stats(
                eligible,
                field="raw_episode_return",
                window=recent_window,
            ),
            "recent_training_return": _recent_stats(
                eligible,
                field="training_episode_return",
                window=recent_window,
            ),
            "recent_life_loss_count": _recent_stats(
                recent_eligible,
                field="episode_life_loss_count",
                window=recent_window,
            ),
            "recent_score_per_life": _recent_stats(
                recent_eligible,
                field="score_per_life",
                window=recent_window,
            ),
            "recent_frames_between_life_losses": _recent_stats(
                recent_eligible,
                field="frames_between_life_losses",
                window=recent_window,
            ),
            "recent_time_to_first_life_loss": _recent_stats(
                recent_eligible,
                field="time_to_first_life_loss",
                window=recent_window,
            ),
            "recent_life_losses_per_1000_steps": _recent_stats(
                recent_eligible,
                field="life_losses_per_1000_steps",
                window=recent_window,
            ),
            "recent_episode_length": _recent_stats(
                recent_eligible,
                field="episode_length",
                window=recent_window,
            ),
        }

    return {
        "run_dir": str(Path(run_dir)),
        "run_id": str(config.get("run_id", Path(run_dir).name)),
        "status": summary.get("status"),
        "algorithm": config.get("algorithm", summary.get("algorithm")),
        "architecture": config.get("architecture", summary.get("architecture")),
        "seed": config.get("seed", summary.get("seed")),
        "total_steps": expected_steps,
        "life_loss_penalty": config.get(
            "life_loss_penalty",
            summary.get("life_loss_penalty", 0.0),
        ),
        "episode_count": len(completed),
        "raw_score": score_statistics(raw_values),
        "training_return": score_statistics(training_values),
        "score_per_life": score_statistics(score_per_life_values),
        "recent_raw_score": _recent_stats(
            recent,
            field="raw_episode_return",
            window=recent_window,
        ),
        "recent_training_return": _recent_stats(
            recent,
            field="training_episode_return",
            window=recent_window,
        ),
        "life_loss_count": final_life_loss_count,
        "life_loss_penalty_total": final_penalty_total,
        "life_losses_per_1000_steps": float(
            final_life_loss_count / max(1, int(expected_steps or 0)) * 1000.0
        ),
        "learning_curve": curve,
        "milestones": milestones,
        "runtime": config.get("runtime", {}),
        "summary": summary,
    }


def compare_training_runs(
    baseline_run: str | Path,
    shaped_run: str | Path,
    *,
    recent_window: int = 20,
) -> dict[str, Any]:
    """Compare training learning curves while checking fair-run conditions."""

    baseline = summarize_training_run(baseline_run, recent_window=recent_window)
    shaped = summarize_training_run(shaped_run, recent_window=recent_window)
    baseline_config, _baseline_summary, _baseline_rows = _read_run(baseline_run)
    shaped_config, _shaped_summary, _shaped_rows = _read_run(shaped_run)
    config_differences = {
        field.name: {
            "baseline": baseline_config.get(field.name),
            "shaped": shaped_config.get(field.name),
        }
        for field in DQNConfig.__dataclass_fields__.values()
        if baseline_config.get(field.name) != shaped_config.get(field.name)
    }
    comparable_fields = (
        "total_steps",
        "seed",
        "algorithm",
        "architecture",
    )
    conditions = {
        field: {
            "baseline": baseline.get(field),
            "shaped": shaped.get(field),
            "match": baseline.get(field) == shaped.get(field),
        }
        for field in comparable_fields
    }
    conditions["only_reward_design_difference_expected"] = (
        set(config_differences) == {"life_loss_penalty"}
        and baseline.get("life_loss_penalty") != shaped.get("life_loss_penalty")
    )
    milestone_comparison: dict[str, Any] = {}
    for name, _fraction in MILESTONES:
        base_milestone = baseline["milestones"][name]
        shaped_milestone = shaped["milestones"][name]
        base_raw = base_milestone["recent_raw_score"]["mean"]
        shaped_raw = shaped_milestone["recent_raw_score"]["mean"]
        base_life = base_milestone["recent_life_loss_count"]["mean"]
        shaped_life = shaped_milestone["recent_life_loss_count"]["mean"]
        base_score_per_life = base_milestone["recent_score_per_life"]["mean"]
        shaped_score_per_life = shaped_milestone["recent_score_per_life"]["mean"]
        base_frames = base_milestone["recent_frames_between_life_losses"]["mean"]
        shaped_frames = shaped_milestone["recent_frames_between_life_losses"]["mean"]
        base_first_loss = base_milestone["recent_time_to_first_life_loss"]["mean"]
        shaped_first_loss = shaped_milestone["recent_time_to_first_life_loss"]["mean"]
        base_rate = base_milestone["recent_life_losses_per_1000_steps"]["mean"]
        shaped_rate = shaped_milestone["recent_life_losses_per_1000_steps"]["mean"]
        base_length = base_milestone["recent_episode_length"]["mean"]
        shaped_length = shaped_milestone["recent_episode_length"]["mean"]
        milestone_comparison[name] = {
            "baseline": base_milestone,
            "shaped": shaped_milestone,
            "raw_score_mean_delta": (
                None if base_raw is None or shaped_raw is None else shaped_raw - base_raw
            ),
            "life_loss_mean_delta": (
                None if base_life is None or shaped_life is None else shaped_life - base_life
            ),
            "score_per_life_mean_delta": (
                None
                if base_score_per_life is None or shaped_score_per_life is None
                else shaped_score_per_life - base_score_per_life
            ),
            "frames_between_life_losses_mean_delta": (
                None
                if base_frames is None or shaped_frames is None
                else shaped_frames - base_frames
            ),
            "time_to_first_life_loss_mean_delta": (
                None
                if base_first_loss is None or shaped_first_loss is None
                else shaped_first_loss - base_first_loss
            ),
            "life_losses_per_1000_steps_mean_delta": (
                None
                if base_rate is None or shaped_rate is None
                else shaped_rate - base_rate
            ),
            "episode_length_mean_delta": (
                None
                if base_length is None or shaped_length is None
                else shaped_length - base_length
            ),
        }
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": "issue9_reward_shaping_training_comparison",
        "baseline": baseline,
        "shaped": shaped,
        "conditions": conditions,
        "config_differences": config_differences,
        "milestones": milestone_comparison,
        "learning_efficiency_note": (
            "Milestone comparisons use the most recent completed episodes up to "
            f"each transition boundary; they are descriptive, not a causal test."
        ),
    }


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _line_plot(
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (1200, 675), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24)
    label_font = _font(15)
    small_font = _font(12)
    left, top, right, bottom = 100, 75, 1150, 575
    non_empty = [(name, list(values)) for name, values in series if values]
    draw.text((left, 24), title, font=title_font, fill=(17, 24, 39))
    draw.text((18, top), ylabel, font=label_font, fill=(55, 65, 81))
    if not non_empty:
        draw.rectangle((left, top, right, bottom), outline=(107, 114, 128), width=2)
        draw.text((left + 300, top + 220), "No finite samples", font=label_font, fill=(75, 85, 99))
        canvas.save(output_path, format="PNG", optimize=True)
        return
    all_points = [point for _name, points in non_empty for point in points]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    padding = max((y_max - y_min) * 0.08, 1e-6)
    y_min -= padding
    y_max += padding
    for index in range(6):
        fraction = index / 5
        y = int(top + fraction * (bottom - top))
        tick = y_max - fraction * (y_max - y_min)
        draw.line((left, y, right, y), fill=(229, 231, 235), width=1)
        draw.text((left - 75, y - 8), f"{tick:.3g}", font=small_font, fill=(107, 114, 128))
    draw.rectangle((left, top, right, bottom), outline=(107, 114, 128), width=2)
    colors = ((37, 99, 235), (220, 38, 38), (22, 163, 74), (124, 58, 237))
    for series_index, (name, points) in enumerate(non_empty):
        scaled = [
            (
                int(left + (x - x_min) / (x_max - x_min) * (right - left)),
                int(bottom - (y - y_min) / (y_max - y_min) * (bottom - top)),
            )
            for x, y in points
        ]
        if len(scaled) == 1:
            x, y = scaled[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[series_index])
        else:
            draw.line(scaled, fill=colors[series_index], width=2, joint="curve")
        legend_x = left + series_index * 230
        draw.rectangle((legend_x, top - 32, legend_x + 14, top - 18), fill=colors[series_index])
        draw.text((legend_x + 20, top - 36), name, font=small_font, fill=(55, 65, 81))
    draw.text((left, bottom + 12), f"{x_min:.0f}", font=small_font, fill=(107, 114, 128))
    draw.text((right - 45, bottom + 12), f"{x_max:.0f}", font=small_font, fill=(107, 114, 128))
    canvas.save(output_path, format="PNG", optimize=True)


def _paired_plot(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 100, 70, 820, 600
    draw.text((left, 24), "Paired raw score comparison", font=_font(24), fill=(17, 24, 39))
    points = [
        (float(row["baseline_raw_score"]), float(row["shaped_raw_score"]))
        for row in rows
    ]
    if not points:
        canvas.save(output_path, format="PNG", optimize=True)
        return
    values = [value for point in points for value in point]
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        minimum -= 1.0
        maximum += 1.0
    padding = max((maximum - minimum) * 0.08, 1.0)
    minimum -= padding
    maximum += padding

    def xy(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return (
            int(left + (x - minimum) / (maximum - minimum) * (right - left)),
            int(bottom - (y - minimum) / (maximum - minimum) * (bottom - top)),
        )

    draw.rectangle((left, top, right, bottom), outline=(107, 114, 128), width=2)
    draw.line((left, bottom, right, top), fill=(156, 163, 175), width=2)
    for point in points:
        x, y = xy(point)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(37, 99, 235))
    draw.text((left, bottom + 16), "Baseline raw score", font=_font(15), fill=(55, 65, 81))
    draw.text((18, top), "Shaped raw score", font=_font(15), fill=(55, 65, 81))
    canvas.save(output_path, format="PNG", optimize=True)


def _histogram_plot(values: Sequence[float], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (900, 650), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((100, 24), "Score delta distribution", font=_font(24), fill=(17, 24, 39))
    left, top, right, bottom = 100, 80, 820, 560
    if not values:
        canvas.save(output_path, format="PNG", optimize=True)
        return
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        minimum -= 1.0
        maximum += 1.0
    bins = min(12, max(5, len(values)))
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - minimum) / (maximum - minimum) * bins))
        counts[index] += 1
    maximum_count = max(counts) or 1
    width = (right - left) / bins
    draw.rectangle((left, top, right, bottom), outline=(107, 114, 128), width=2)
    for index, count in enumerate(counts):
        x0 = int(left + index * width + 2)
        x1 = int(left + (index + 1) * width - 2)
        y = int(bottom - count / maximum_count * (bottom - top))
        draw.rectangle((x0, y, x1, bottom), fill=(220, 38, 38))
    draw.text((left, bottom + 15), f"{minimum:.3g}", font=_font(12), fill=(107, 114, 128))
    draw.text((right - 50, bottom + 15), f"{maximum:.3g}", font=_font(12), fill=(107, 114, 128))
    canvas.save(output_path, format="PNG", optimize=True)


def _summary_plot(comparison: Mapping[str, Any], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((100, 24), "Final raw score summary", font=_font(24), fill=(17, 24, 39))
    baseline = comparison["baseline"]
    shaped = comparison["shaped"]
    values = [
        ("Baseline mean", baseline["mean_raw_score"]),
        ("Shaped mean", shaped["mean_raw_score"]),
        ("Baseline median", baseline["median_raw_score"]),
        ("Shaped median", shaped["median_raw_score"]),
    ]
    finite = [float(value) for _label, value in values if value is not None]
    maximum = max(finite, default=1.0) or 1.0
    left, top, bottom = 130, 90, 520
    width = 120
    gap = 65
    for index, (label, value) in enumerate(values):
        numeric = 0.0 if value is None else float(value)
        x0 = left + index * (width + gap)
        y0 = bottom - numeric / maximum * (bottom - top)
        draw.rectangle((x0, y0, x0 + width, bottom), fill=(37, 99, 235) if index % 2 == 0 else (220, 38, 38))
        draw.text((x0, bottom + 15), label, font=_font(12), fill=(55, 65, 81))
        draw.text((x0, y0 - 22), f"{numeric:.3g}", font=_font(12), fill=(17, 24, 39))
    canvas.save(output_path, format="PNG", optimize=True)


def write_reward_shaping_artifacts(
    *,
    baseline_run: str | Path,
    shaped_run: str | Path,
    baseline_evaluation: Mapping[str, Any],
    shaped_evaluation: Mapping[str, Any],
    output_dir: str | Path,
    experiment_config: Mapping[str, Any],
    recent_window: int = 20,
) -> dict[str, str]:
    """Write durable JSON/PNG evidence for a paired Issue #9 experiment."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training_comparison = compare_training_runs(
        baseline_run,
        shaped_run,
        recent_window=recent_window,
    )
    paired_evaluation = compare_evaluation_payloads(
        baseline_evaluation,
        shaped_evaluation,
    )
    baseline_summary = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": "issue9_reward_shaping_baseline_summary",
        "training": training_comparison["baseline"],
        "evaluation": paired_evaluation["baseline"],
    }
    shaped_summary = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": "issue9_reward_shaping_life_loss_penalty_summary",
        "training": training_comparison["shaped"],
        "evaluation": paired_evaluation["shaped"],
    }
    payloads = {
        "experiment-config.json": {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "artifact_type": EXPERIMENT_ARTIFACT_TYPE,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "score_definition": "raw Atari game reward sum; training shaping is excluded",
            **dict(experiment_config),
        },
        "baseline-summary.json": baseline_summary,
        "life-loss-penalty-summary.json": shaped_summary,
        "paired-evaluation.json": paired_evaluation,
        "training-comparison.json": training_comparison,
    }
    paths: dict[str, str] = {}
    for filename, payload in payloads.items():
        path = destination / filename
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        paths[filename] = str(path)

    baseline_curve = training_comparison["baseline"]["learning_curve"]
    shaped_curve = training_comparison["shaped"]["learning_curve"]
    _line_plot(
        (
            (
                "Baseline raw score",
                [
                    (row["global_step"], row["raw_episode_return"])
                    for row in baseline_curve
                ],
            ),
            (
                "Shaped raw score",
                [
                    (row["global_step"], row["raw_episode_return"])
                    for row in shaped_curve
                ],
            ),
        ),
        title="Training raw-score curve",
        ylabel="Raw episode score",
        output_path=destination / "training-score-curve.png",
    )
    _line_plot(
        (
            (
                "Baseline life losses",
                [
                    (row["global_step"], row["episode_life_loss_count"])
                    for row in baseline_curve
                ],
            ),
            (
                "Shaped life losses",
                [
                    (row["global_step"], row["episode_life_loss_count"])
                    for row in shaped_curve
                ],
            ),
        ),
        title="Life losses over training",
        ylabel="Life losses per completed episode",
        output_path=destination / "life-loss-curve.png",
    )
    paired_rows = paired_evaluation["paired_rows"]
    _paired_plot(paired_rows, destination / "paired-score-comparison.png")
    _histogram_plot(
        [float(row["score_delta"]) for row in paired_rows],
        destination / "score-delta-distribution.png",
    )
    _summary_plot(paired_evaluation, destination / "evaluation-summary.png")
    for filename in (
        "training-score-curve.png",
        "life-loss-curve.png",
        "paired-score-comparison.png",
        "score-delta-distribution.png",
        "evaluation-summary.png",
    ):
        paths[filename] = str(destination / filename)
    return paths


__all__ = [
    "EXPERIMENT_ARTIFACT_TYPE",
    "EXPERIMENT_SCHEMA_VERSION",
    "compare_evaluation_payloads",
    "compare_training_runs",
    "score_statistics",
    "summarize_training_run",
    "write_reward_shaping_artifacts",
]
