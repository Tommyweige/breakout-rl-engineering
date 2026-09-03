"""Render Day 20 figures from completed family-comparison artifacts."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from breakout_rl.day20_comparison import (
    DAY20_EXTENSION_STAGE,
    DAY20_EXTENSION_TARGET,
    DAY20_FAMILY_IDS,
    DAY20_MILESTONES,
    load_day20_config,
    read_day20_manifest,
    read_metrics,
    sha256_file,
    relative_path,
    write_json,
)
from breakout_rl.models.factory import build_q_network


FAMILY_COLORS = {
    "dqn": "#0f766e",
    "double_dqn": "#7c3aed",
    "dueling_double_dqn": "#ea580c",
}
FAMILY_LABELS = {
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "dueling_double_dqn": "Dueling Double DQN",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot real Day 20 DQN-family comparison artifacts."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day20-dqn-family/manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("assets/day20"))
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


def _completed(entries: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected = [
        entry
        for entry in entries
        if entry.get("status") in {"completed", "reused"}
        and entry.get("eligible") is True
    ]
    if not selected:
        raise ValueError(
            "no completed eligible Day 20 artifacts are available; refusing to create a placeholder figure"
        )
    return selected


def _entry_metrics(
    manifest_path: Path,
    entry: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    inline = entry.get("metrics")
    if isinstance(inline, list):
        return [row for row in inline if isinstance(row, Mapping)]
    return read_metrics(manifest_path, entry)


def _source_metadata(
    manifest_path: Path,
    entries: Sequence[Mapping[str, Any]],
    outputs: Sequence[Path],
    *,
    question: str,
) -> dict[str, Any]:
    return {
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
                "family_id": entry.get("family_id"),
                "training_seed": entry.get("training_seed"),
                "stage": entry.get("stage"),
                "target_transitions": entry.get("target_transitions"),
                "status": entry.get("status"),
                "eligible": entry.get("eligible"),
                "metrics_path": (
                    entry.get("training", {}).get("metrics_path")
                    if isinstance(entry.get("training"), Mapping)
                    else None
                ),
                "evaluation": entry.get("evaluation"),
                "q_probe": entry.get("q_probe"),
                "runtime": entry.get("runtime"),
            }
            for entry in entries
        ],
        "outputs": [path.as_posix() for path in outputs],
    }


def _save_figure(figure: Any, paths: Sequence[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="png", dpi=180)


def _write_metadata(
    manifest_path: Path,
    output_dir: Path,
    filename: str,
    entries: Sequence[Mapping[str, Any]],
    outputs: Sequence[Path],
    *,
    question: str,
) -> None:
    write_json(
        output_dir / filename,
        _source_metadata(manifest_path, entries, outputs, question=question),
    )


def _family_seed_series(
    manifest_path: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[tuple[str, int], list[tuple[int, float]]]:
    grouped: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for entry in entries:
        family_id = str(entry.get("family_id"))
        seed = int(entry["training_seed"])
        for row in _entry_metrics(manifest_path, entry):
            step = _number(row.get("global_step"))
            value = _number(row.get(field))
            if step is None or value is None:
                continue
            grouped[(family_id, seed)][int(step)] = value
    return {
        key: sorted(values.items())
        for key, values in grouped.items()
    }


def _rolling(values: Sequence[tuple[int, float]], window: int = 20) -> list[tuple[int, float]]:
    if len(values) < window:
        return []
    return [
        (
            values[index - 1][0],
            sum(value for _, value in values[index - window : index]) / window,
        )
        for index in range(window, len(values) + 1)
    ]


def _render_training_local(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = _family_seed_series(manifest_path, entries, field="raw_episode_return")
    if not series:
        raise ValueError("completed runs contain no finite raw_episode_return metrics")
    _configure_font()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.2, 8.2),
        sharex=True,
        constrained_layout=True,
    )
    raw_axis, rolling_axis = axes
    has_extension = any(entry.get("stage") == DAY20_EXTENSION_STAGE for entry in entries)
    plotted_milestones = dict(DAY20_MILESTONES)
    if has_extension:
        plotted_milestones[DAY20_EXTENSION_STAGE] = DAY20_EXTENSION_TARGET
    for (family_id, seed), values in sorted(series.items()):
        color = FAMILY_COLORS.get(family_id, "#334155")
        label = f"{FAMILY_LABELS.get(family_id, family_id)} seed {seed}"
        raw_axis.plot(
            [step for step, _ in values],
            [value for _, value in values],
            marker=".",
            markersize=1.9,
            linewidth=0.65,
            alpha=0.34,
            color=color,
            label=label,
        )
        smoothed = _rolling(values)
        if smoothed:
            rolling_axis.plot(
                [step for step, _ in smoothed],
                [value for _, value in smoothed],
                linewidth=1.45,
                alpha=0.88,
                color=color,
                label=label,
            )
    for axis in axes:
        for stage, target in plotted_milestones.items():
            axis.axvline(target, color="#94a3b8", linewidth=0.75, linestyle="--", alpha=0.6)
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
        axis.grid(True, alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize=7, ncol=2)
    raw_axis.set_title("Raw episode return — each line keeps its training seed")
    raw_axis.set_ylabel("Raw Atari episode return")
    rolling_axis.set_title("20-episode rolling mean")
    rolling_axis.set_xlabel("Actual environment transitions")
    rolling_axis.set_ylabel("Rolling mean return")
    figure.suptitle(
        "Day 20 DQN family training curves"
        + (" through 1M extension" if has_extension else " through 500K"),
        fontsize=13,
    )
    outputs = (
        output_dir / "dqn-family-training.png",
        output_dir / "training-return-100k-500k.png",
    )
    _save_figure(figure, outputs)
    plt.close(figure)
    _write_metadata(
        manifest_path,
        output_dir,
        "dqn-family-training.json",
        entries,
        outputs,
        question="在 actual environment transitions 軸上，三個 DQN family 的 learning trend 是否跨 training seed 重複？",
    )
    return outputs


def _render_training_worker(
    manifest_path: str,
    output_dir: str,
    entries: list[dict[str, Any]],
    result_queue: Any,
) -> None:
    try:
        # Isolate Matplotlib from the separate OpenMP runtime loaded by PyTorch
        # on Windows; the plotted values still come from the real CSV files.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        outputs = _render_training_local(
            Path(manifest_path),
            Path(output_dir),
            entries,
        )
        result_queue.put({"ok": True, "outputs": [path.as_posix() for path in outputs]})
    except Exception as error:  # pragma: no cover - exercised in child process
        result_queue.put({"ok": False, "error": f"{type(error).__name__}: {error}"})


def render_training(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Render training curves, isolating Matplotlib from PyTorch DLLs."""

    if "torch" not in sys.modules:
        return _render_training_local(manifest_path, output_dir, entries)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_render_training_worker,
        args=(str(manifest_path), str(output_dir), [dict(entry) for entry in entries], result_queue),
    )
    process.start()
    process.join()
    try:
        result = result_queue.get(timeout=5)
    except queue.Empty:
        result = None
    result_queue.close()
    if process.exitcode != 0 or not isinstance(result, dict) or not result.get("ok"):
        error = result.get("error") if isinstance(result, dict) else "unknown plotting error"
        raise RuntimeError(f"training plot worker failed: {error}")
    return tuple(Path(path) for path in result["outputs"])  # type: ignore[return-value]


def _evaluation_summary(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = entry.get("evaluation")
    if isinstance(evaluation, Mapping) and isinstance(evaluation.get("summary"), Mapping):
        return evaluation["summary"]
    return {}


def _render_evaluation_local(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_font()
    figure, axis = plt.subplots(figsize=(11.0, 6.0), constrained_layout=True)
    plotted = False
    for family_id in DAY20_FAMILY_IDS:
        family_entries = [entry for entry in entries if entry.get("family_id") == family_id]
        for seed in sorted({int(entry["training_seed"]) for entry in family_entries}):
            values = [
                entry
                for entry in family_entries
                if int(entry["training_seed"]) == seed
            ]
            values.sort(key=lambda entry: int(entry["target_transitions"]))
            points = [
                (
                    int(entry["target_transitions"]),
                    _evaluation_summary(entry),
                )
                for entry in values
                if _evaluation_summary(entry).get("mean_return") is not None
            ]
            if not points:
                continue
            plotted = True
            axis.errorbar(
                [step for step, _ in points],
                [float(summary["mean_return"]) for _, summary in points],
                yerr=[float(summary.get("std_return", 0.0)) for _, summary in points],
                marker="o",
                linewidth=1.05,
                markersize=4.8,
                capsize=3,
                alpha=0.88,
                color=FAMILY_COLORS[family_id],
                label=f"{FAMILY_LABELS[family_id]} seed {seed}",
            )
    if not plotted:
        plt.close(figure)
        raise ValueError("completed entries contain no evaluation summaries")
    tick_values = [100_000, 250_000, 500_000]
    tick_labels = ["100K screening", "250K pilot", "500K main"]
    if any(
        int(entry.get("target_transitions", 0)) >= 1_000_000
        and _evaluation_summary(entry).get("mean_return") is not None
        for entry in entries
    ):
        tick_values.append(1_000_000)
        tick_labels.append("1M extension")
    axis.set_xticks(tick_values, tick_labels)
    axis.set_xlabel("Checkpoint transition budget")
    axis.set_ylabel("Mean raw Atari return ± episode std")
    axis.set_title("Contract v2 evaluation — preserve family and training seed")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize=8, ncol=2)
    output = output_dir / "dqn-family-evaluation.png"
    _save_figure(figure, (output,))
    plt.close(figure)
    _write_metadata(
        manifest_path,
        output_dir,
        "dqn-family-evaluation.json",
        entries,
        (output,),
        question="在固定 Contract v2 evaluation protocol 下，三個 family 的每個 training seed 在 100K/250K/500K 如何變化？",
    )
    return output


def _render_seed_spread_local(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_font()
    formal = [entry for entry in entries if entry.get("stage") == "main"]
    if not formal:
        raise ValueError("500K family entries are required for seed-spread figure")
    figure, axis = plt.subplots(figsize=(10.6, 5.8), constrained_layout=True)
    plotted = False
    for family_index, family_id in enumerate(DAY20_FAMILY_IDS):
        family_entries = {
            int(entry["training_seed"]): entry
            for entry in formal
            if entry.get("family_id") == family_id
        }
        for seed_index, seed in enumerate(sorted(family_entries)):
            summary = _evaluation_summary(family_entries[seed])
            mean = _number(summary.get("mean_return"))
            spread = _number(summary.get("std_return"))
            if mean is None:
                continue
            plotted = True
            x = seed_index * len(DAY20_FAMILY_IDS) + family_index
            axis.errorbar(
                x,
                mean,
                yerr=0.0 if spread is None else spread,
                fmt="o",
                color=FAMILY_COLORS[family_id],
                capsize=3,
                markersize=6,
                label=FAMILY_LABELS[family_id] if seed_index == 0 else None,
            )
    if not plotted:
        plt.close(figure)
        raise ValueError("500K entries contain no evaluation means")
    seeds = sorted({int(entry["training_seed"]) for entry in formal})
    centers = [index * len(DAY20_FAMILY_IDS) + 1 for index in range(len(seeds))]
    axis.set_xticks(centers, [f"seed {seed}" for seed in seeds])
    axis.set_ylabel("500K mean raw Atari return ± episode std")
    axis.set_title("500K seed spread — quality evidence stays at seed level")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    output = output_dir / "dqn-family-seed-spread.png"
    _save_figure(figure, (output,))
    plt.close(figure)
    _write_metadata(
        manifest_path,
        output_dir,
        "dqn-family-seed-spread.json",
        formal,
        (output,),
        question="500K 時每個 family 的固定 evaluation mean 與 seed-level spread 是否支持穩健選擇？",
    )
    return output


def _runtime_value(entry: Mapping[str, Any], field: str) -> float | None:
    runtime = entry.get("runtime")
    if not isinstance(runtime, Mapping):
        runtime = {}
    if field == "peak_vram_mib":
        value = runtime.get("cuda_peak_allocated_bytes")
        number = _number(value)
        return None if number is None else number / (1024 * 1024)
    return _number(runtime.get(field))


def _parameter_count(entry: Mapping[str, Any]) -> float | None:
    runtime = entry.get("runtime")
    if isinstance(runtime, Mapping):
        value = _number(runtime.get("parameter_count"))
        if value is not None:
            return value
    summary = entry.get("summary")
    if isinstance(summary, Mapping):
        value = _number(summary.get("parameter_count"))
        if value is not None:
            return value
        model_config = summary.get("model_config")
        if isinstance(model_config, Mapping):
            value = _number(model_config.get("parameter_count"))
            if value is not None:
                return value
    architecture = entry.get("architecture")
    if architecture not in {"standard", "dueling"}:
        return None
    model = build_q_network(
        str(architecture),
        num_actions=4,
        input_shape=(4, 84, 84),
    )
    return float(sum(parameter.numel() for parameter in model.parameters()))


def _render_runtime_cost_local(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_font()
    formal = [entry for entry in entries if entry.get("stage") == "main"]
    if not formal:
        raise ValueError("500K family entries are required for runtime-cost figure")
    fields = (
        ("steps_per_second", "Mean environment SPS", "transitions/s"),
        ("wall_clock_seconds", "Wall-clock", "seconds"),
        ("peak_vram_mib", "Peak allocated CUDA memory", "MiB"),
        ("parameter_count", "Parameter count", "parameters"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    plotted_any = False
    for axis, (field, title, ylabel) in zip(axes.flat, fields, strict=True):
        family_index = {family_id: index for index, family_id in enumerate(DAY20_FAMILY_IDS)}
        for family_id in DAY20_FAMILY_IDS:
            values: list[float] = []
            for entry in formal:
                if entry.get("family_id") != family_id:
                    continue
                if field == "parameter_count":
                    parsed = _parameter_count(entry)
                else:
                    parsed = _runtime_value(entry, field)
                if parsed is not None:
                    values.append(parsed)
            if not values:
                continue
            plotted_any = True
            x_values = [
                family_index[family_id] + (index - (len(values) - 1) / 2) * 0.08
                for index in range(len(values))
            ]
            axis.scatter(
                x_values,
                values,
                color=FAMILY_COLORS[family_id],
                s=42,
                label=FAMILY_LABELS[family_id],
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            range(len(DAY20_FAMILY_IDS)),
            [FAMILY_LABELS[id] for id in DAY20_FAMILY_IDS],
            rotation=18,
        )
        axis.grid(axis="y", alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize=7)
    if not plotted_any:
        plt.close(figure)
        raise ValueError("completed entries contain no reliable runtime metrics")
    figure.suptitle("Day 20 engineering cost at the 500K milestone", fontsize=13)
    output = output_dir / "dqn-family-runtime-cost.png"
    _save_figure(figure, (output,))
    plt.close(figure)
    _write_metadata(
        manifest_path,
        output_dir,
        "dqn-family-runtime-cost.json",
        formal,
        (output,),
        question="在同一 NVIDIA CUDA、float32 與 transition budget 下，三個 family 的吞吐、耗時、VRAM 與參數成本是多少？",
    )
    return output


def render_evaluation(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    """Render an evaluation figure for callers that need one figure only."""

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return _render_evaluation_local(manifest_path, output_dir, entries)


def render_seed_spread(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return _render_seed_spread_local(manifest_path, output_dir, entries)


def render_runtime_cost(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return _render_runtime_cost_local(manifest_path, output_dir, entries)


def _render_all_local(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> list[Path]:
    paths: list[Path] = []
    paths.extend(_render_training_local(manifest_path, output_dir, entries))
    paths.append(_render_evaluation_local(manifest_path, output_dir, entries))
    paths.append(_render_seed_spread_local(manifest_path, output_dir, entries))
    paths.append(_render_runtime_cost_local(manifest_path, output_dir, entries))
    return paths


def _render_all_worker(
    manifest_path: str,
    output_dir: str,
    entries: list[dict[str, Any]],
    result_queue: Any,
) -> None:
    try:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        outputs = _render_all_local(Path(manifest_path), Path(output_dir), entries)
        result_queue.put({"ok": True, "outputs": [path.as_posix() for path in outputs]})
    except Exception as error:  # pragma: no cover - exercised in child process
        result_queue.put({"ok": False, "error": f"{type(error).__name__}: {error}"})


def render_all(manifest_path: str | Path, output_dir: str | Path) -> list[Path]:
    source = Path(manifest_path).resolve()
    manifest = read_day20_manifest(source)
    entries = _completed(manifest["runs"])
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_render_all_worker,
        args=(str(source), str(Path(output_dir)), [dict(entry) for entry in entries], result_queue),
    )
    process.start()
    process.join()
    try:
        result = result_queue.get(timeout=5)
    except queue.Empty:
        result = None
    result_queue.close()
    if process.exitcode != 0 or not isinstance(result, dict) or not result.get("ok"):
        error = result.get("error") if isinstance(result, dict) else "unknown plotting error"
        raise RuntimeError(f"Day 20 plotting worker failed: {error}")
    return [Path(path) for path in result["outputs"]]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = render_all(args.manifest, args.output_dir)
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Unable to render Day 20 family figures: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"outputs": [path.as_posix() for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
