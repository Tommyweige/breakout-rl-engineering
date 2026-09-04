"""Render source-backed Day 21 final-training figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from breakout_rl.day21_final_training import (
    DAY21_STAGE_ORDER,
    DAY21_STAGE_TARGETS,
    read_day21_manifest,
    relative_path,
    sha256_file,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot real Day 21 final long-training and holdout evidence."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day21-final-long-training/manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("assets/day21"))
    return parser


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _configure_font() -> None:
    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["axes.unicode_minus"] = False
    for candidate in (
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/mingliu.ttc"),
        Path("C:/Windows/Fonts/NotoSansTC-VF.ttf"),
    ):
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            matplotlib.rcParams["font.family"] = [
                font_manager.FontProperties(fname=str(candidate)).get_name()
            ]
            return


def _entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = [entry for entry in manifest.get("runs", []) if isinstance(entry, Mapping)]
    if not entries:
        raise ValueError("Day 21 manifest contains no run entries")
    return entries


def _stage_record(entry: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    stages = entry.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(stages.get(stage), Mapping):
        raise ValueError(f"entry is missing {stage}")
    return stages[stage]


def _completed_stage_records(
    entries: Iterable[Mapping[str, Any]],
    stage: str,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    completed: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for entry in entries:
        record = _stage_record(entry, stage)
        evaluation = record.get("evaluation")
        if (
            record.get("status") == "completed"
            and record.get("eligible") is True
            and isinstance(evaluation, Mapping)
            and evaluation.get("status") == "completed"
            and isinstance(evaluation.get("summary"), Mapping)
        ):
            completed.append((entry, record))
    return completed


def _read_compact_metrics(
    manifest_path: Path,
    record: Mapping[str, Any],
) -> list[dict[str, str]]:
    training = record.get("training")
    if not isinstance(training, Mapping):
        return []
    raw_path = training.get("compact_metrics_path")
    if not isinstance(raw_path, str):
        return []
    path = (manifest_path.parent / raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_long_training_metrics(
    manifest_path: Path,
    entry: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Use the highest completed immutable snapshot for one training seed."""

    completed = [
        (int(_stage_record(entry, stage).get("target_transitions", 0)), stage)
        for stage in DAY21_STAGE_ORDER
        if _stage_record(entry, stage).get("status") == "completed"
    ]
    if not completed:
        return []
    _target, stage = max(completed)
    return _read_compact_metrics(manifest_path, _stage_record(entry, stage))


def _save_figure(figure: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="png", dpi=180)


def _write_figure_metadata(
    *,
    manifest_path: Path,
    output_dir: Path,
    filename: str,
    question: str,
    outputs: Sequence[Path],
    source_entries: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "command": list(sys.argv),
        "question": question,
        "source_manifest": relative_path(manifest_path, start=Path.cwd()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_entries": [
            {
                "run_id": entry.get("run_id"),
                "training_seed": entry.get("training_seed"),
                "stages": {
                    stage: {
                        "status": _stage_record(entry, stage).get("status"),
                        "target_transitions": _stage_record(entry, stage).get(
                            "target_transitions"
                        ),
                        "checkpoint": _stage_record(entry, stage).get("checkpoint"),
                        "gameplay": _stage_record(entry, stage).get("gameplay"),
                        "metrics_path": _stage_record(entry, stage)
                        .get("training", {})
                        .get("compact_metrics_path")
                        if isinstance(_stage_record(entry, stage).get("training"), Mapping)
                        else None,
                        "evaluation": _stage_record(entry, stage).get("evaluation"),
                    }
                    for stage in DAY21_STAGE_ORDER
                },
            }
            for entry in source_entries
        ],
        "outputs": [path.as_posix() for path in outputs],
    }
    if extra:
        payload.update(dict(extra))
    write_json(output_dir / filename, payload)


def render_long_training_return(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    """Show actual episode returns and a rolling view across all seeds."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_font()
    colors = ("#0f766e", "#7c3aed", "#ea580c")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.2, 8.2),
        sharex=True,
        constrained_layout=True,
    )
    plotted = False
    source_entries: list[Mapping[str, Any]] = []
    for color, entry in zip(colors, sorted(entries, key=lambda item: int(item["training_seed"])), strict=False):
        rows = _read_long_training_metrics(manifest_path, entry)
        if not rows:
            continue
        source_entries.append(entry)
        values: list[tuple[int, float]] = []
        for row in rows:
            step = _number(row.get("global_step"))
            episode_return = _number(row.get("raw_episode_return"))
            if step is not None and episode_return is not None:
                values.append((int(step), episode_return))
        if not values:
            continue
        plotted = True
        label = f"training seed {entry['training_seed']}"
        axes[0].plot(
            [step for step, _value in values],
            [value for _step, value in values],
            marker=".",
            markersize=2.0,
            linewidth=0.7,
            alpha=0.45,
            color=color,
            label=label,
        )
        if len(values) >= 20:
            rolling = [
                (
                    values[index - 1][0],
                    sum(value for _step, value in values[index - 20 : index]) / 20,
                )
                for index in range(20, len(values) + 1)
            ]
            axes[1].plot(
                [step for step, _value in rolling],
                [value for _step, value in rolling],
                linewidth=1.5,
                color=color,
                label=label,
            )
    if not plotted:
        plt.close(figure)
        raise ValueError("completed Day 21 runs contain no episode-return metrics")
    for axis in axes:
        for stage, target in DAY21_STAGE_TARGETS.items():
            axis.axvline(target, color="#94a3b8", linewidth=0.75, linestyle="--", alpha=0.65)
            axis.text(
                target,
                0.97,
                stage,
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=7,
                color="#475569",
            )
        axis.grid(True, alpha=0.23)
        axis.legend(loc="best", fontsize=8)
    axes[0].set_title("Raw episode return — actual training evidence")
    axes[0].set_ylabel("raw Atari return")
    axes[1].set_title("20-episode rolling mean")
    axes[1].set_xlabel("actual environment transitions")
    axes[1].set_ylabel("rolling mean return")
    figure.suptitle("Day 21 final long training across fresh seeds", fontsize=13)
    output = output_dir / "long-training-return.png"
    _save_figure(figure, output)
    plt.close(figure)
    _write_figure_metadata(
        manifest_path=manifest_path,
        output_dir=output_dir,
        filename="long-training-return.json",
        question="Across fresh training seeds, does the raw learning signal change over actual transitions and remain visible through the requested 5M target?",
        outputs=(output,),
        source_entries=source_entries,
        extra={"milestones": DAY21_STAGE_TARGETS, "rolling_window_episodes": 20},
    )
    return output


def render_milestone_evaluation(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    """Compare aggregate fixed-evaluation means at each completed milestone."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_font()
    figure, axis = plt.subplots(figsize=(11.0, 6.0), constrained_layout=True)
    colors = ("#0f766e", "#7c3aed", "#ea580c")
    plotted = False
    source_entries: list[Mapping[str, Any]] = []
    stage_positions = {stage: index for index, stage in enumerate(DAY21_STAGE_ORDER)}
    for color, entry in zip(colors, sorted(entries, key=lambda item: int(item["training_seed"])), strict=False):
        entry_plotted = False
        for stage in DAY21_STAGE_ORDER:
            record = _stage_record(entry, stage)
            evaluation = record.get("evaluation")
            summary = evaluation.get("summary", {}) if isinstance(evaluation, Mapping) else {}
            mean = _number(summary.get("mean_return")) if isinstance(summary, Mapping) else None
            spread = _number(summary.get("std_return")) if isinstance(summary, Mapping) else None
            if mean is None:
                continue
            entry_plotted = True
            plotted = True
            x = stage_positions[stage] + (int(entry["training_seed"]) % 1000) / 10000 - 0.05
            axis.errorbar(
                x,
                mean,
                yerr=0.0 if spread is None else spread,
                fmt="o-",
                color=color,
                capsize=3,
                markersize=6,
                label=f"seed {entry['training_seed']}" if stage == DAY21_STAGE_ORDER[0] else None,
            )
        if entry_plotted:
            source_entries.append(entry)
    if not plotted:
        plt.close(figure)
        raise ValueError("Day 21 manifest contains no milestone evaluation summaries")
    axis.set_xticks(range(len(DAY21_STAGE_ORDER)), ["1M", "2.5M", "5M"])
    axis.set_xlabel("training milestone")
    axis.set_ylabel("mean raw Atari return ± episode std")
    axis.set_title("Fixed Contract v2 selection evaluation at each milestone")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    output = output_dir / "milestone-evaluation.png"
    _save_figure(figure, output)
    plt.close(figure)
    _write_figure_metadata(
        manifest_path=manifest_path,
        output_dir=output_dir,
        filename="milestone-evaluation.json",
        question="How do complete fixed-evaluation means and episode spread change at 1M, 2.5M, and 5M?",
        outputs=(output,),
        source_entries=source_entries,
        extra={"error_bar": "within-evaluation episode standard deviation"},
    )
    return output


def render_training_seed_spread(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    """Show seed-level evaluation variation without collapsing it too early."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_font()
    figure, axis = plt.subplots(figsize=(10.6, 5.8), constrained_layout=True)
    plotted = False
    source_entries: list[Mapping[str, Any]] = []
    for stage_index, stage in enumerate(DAY21_STAGE_ORDER):
        for seed_index, entry in enumerate(
            sorted(entries, key=lambda item: int(item["training_seed"]))
        ):
            record = _stage_record(entry, stage)
            evaluation = record.get("evaluation")
            summary = evaluation.get("summary", {}) if isinstance(evaluation, Mapping) else {}
            mean = _number(summary.get("mean_return")) if isinstance(summary, Mapping) else None
            spread = _number(summary.get("std_return")) if isinstance(summary, Mapping) else None
            if mean is None:
                continue
            plotted = True
            axis.errorbar(
                stage_index + (seed_index - 1) * 0.08,
                mean,
                yerr=0.0 if spread is None else spread,
                fmt="o",
                capsize=3,
                markersize=6,
                label=f"seed {entry['training_seed']}" if stage_index == 0 else None,
            )
            if entry not in source_entries:
                source_entries.append(entry)
    if not plotted:
        plt.close(figure)
        raise ValueError("Day 21 manifest contains no seed-level evaluation values")
    axis.set_xticks(range(len(DAY21_STAGE_ORDER)), ["1M", "2.5M", "5M"])
    axis.set_xlabel("training milestone")
    axis.set_ylabel("mean raw Atari return ± episode std")
    axis.set_title("Fresh-seed spread remains visible at milestone decisions")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    output = output_dir / "training-seed-spread.png"
    _save_figure(figure, output)
    plt.close(figure)
    _write_figure_metadata(
        manifest_path=manifest_path,
        output_dir=output_dir,
        filename="training-seed-spread.json",
        question="At each milestone, how much does fixed-evaluation quality vary across fresh training seeds?",
        outputs=(output,),
        source_entries=source_entries,
        extra={"point_semantics": "one point per completed seed/milestone"},
    )
    return output


def _read_episode_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def render_final_holdout(
    manifest_path: Path,
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Plot the untouched final holdout using per-group real episode rows."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_font()
    holdout = manifest.get("final_holdout")
    if not isinstance(holdout, Mapping) or holdout.get("status") != "completed":
        raise ValueError("final holdout is not complete; refusing to create a placeholder figure")
    episodes_reference = holdout.get("episodes")
    if not isinstance(episodes_reference, str):
        raise ValueError("final holdout episodes artifact is missing")
    episodes_path = (manifest_path.parent / episodes_reference).resolve()
    rows = _read_episode_rows(episodes_path)
    by_group: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        group = _number(row.get("evaluation_seed"))
        value = _number(row.get("episode_return"))
        if group is not None and value is not None:
            by_group[int(group)].append(value)
    if not by_group:
        raise ValueError("final holdout contains no finite episode returns")
    groups = sorted(by_group)
    means = [sum(by_group[group]) / len(by_group[group]) for group in groups]
    spreads = [
        (
            sum((value - means[index]) ** 2 for value in by_group[group])
            / len(by_group[group])
        )
        ** 0.5
        for index, group in enumerate(groups)
    ]
    figure, axis = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    axis.errorbar(
        range(len(groups)),
        means,
        yerr=spreads,
        fmt="o",
        color="#0f766e",
        capsize=4,
        markersize=7,
    )
    axis.set_xticks(range(len(groups)), [str(group) for group in groups])
    axis.set_xlabel("final holdout group seed")
    axis.set_ylabel("mean raw Atari return ± episode std")
    axis.set_title("Untouched Contract v2 final holdout")
    axis.grid(axis="y", alpha=0.25)
    output = output_dir / "final-holdout.png"
    _save_figure(figure, output)
    plt.close(figure)
    _write_figure_metadata(
        manifest_path=manifest_path,
        output_dir=output_dir,
        filename="final-holdout.json",
        question="After the checkpoint was frozen, what did the untouched holdout groups show?",
        outputs=(output,),
        source_entries=_entries(manifest),
        extra={
            "episodes_artifact": relative_path(episodes_path, start=Path.cwd()),
            "canonical_final_model": manifest.get("canonical_final_model"),
            "group_count": len(groups),
            "episodes_per_group": {
                str(group): len(by_group[group]) for group in groups
            },
            "means": {str(group): means[index] for index, group in enumerate(groups)},
            "std": {str(group): spreads[index] for index, group in enumerate(groups)},
            "holdout_metadata": dict(holdout),
        },
    )
    return output


def render_all(manifest_path: str | Path, output_dir: str | Path) -> list[Path]:
    source = Path(manifest_path).resolve()
    manifest = read_day21_manifest(source)
    entries = _entries(manifest)
    destination = Path(output_dir)
    outputs = [render_long_training_return(source, destination, entries)]
    outputs.append(render_milestone_evaluation(source, destination, entries))
    outputs.append(render_training_seed_spread(source, destination, entries))
    outputs.append(render_final_holdout(source, destination, manifest))
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = render_all(args.manifest, args.output_dir)
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Unable to render Day 21 figures: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"outputs": [path.as_posix() for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
