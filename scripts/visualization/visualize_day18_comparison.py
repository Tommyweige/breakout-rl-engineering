"""Render Day 18 figures directly from completed comparison artifacts."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from breakout_rl.day18_comparison import (
    DAY18_ALGORITHMS,
    DAY18_MILESTONES,
    build_day18_report,
    load_evaluation_entries,
    load_q_probe_entries,
    load_training_entries,
    read_day18_manifest,
    relative_path,
    sha256_file,
    write_json,
)


ALGORITHM_COLORS = {"dqn": "#0f766e", "double_dqn": "#7c3aed"}
ALGORITHM_LABELS = {"dqn": "DQN", "double_dqn": "Double DQN"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot real Day 18 DQN versus Double DQN comparison artifacts."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day18-dqn-vs-double/manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/day18"),
    )
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


def _require_completed(
    entries: Iterable[Mapping[str, Any]],
    *,
    name: str,
) -> list[Mapping[str, Any]]:
    selected = [entry for entry in entries if entry.get("eligible")]
    if not selected:
        raise ValueError(
            f"no completed {name} artifacts are available; refusing to create a placeholder figure"
        )
    return selected


def _metrics_series(
    entries: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[tuple[str, int], list[tuple[int, float]]]:
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    ordered = sorted(
        entries,
        key=lambda entry: (
            int(entry["training_seed"]),
            int(entry["target_transitions"]),
        ),
    )
    for entry in ordered:
        key = (str(entry["algorithm"]), int(entry["training_seed"]))
        existing_steps = {step for step, _ in grouped[key]}
        for raw_row in entry.get("metrics", []):
            step = _number(raw_row.get("global_step"))
            value = _number(raw_row.get(field))
            if step is None or value is None:
                continue
            step_int = int(step)
            if step_int in existing_steps:
                continue
            grouped[key].append((step_int, value))
            existing_steps.add(step_int)
        grouped[key].sort(key=lambda item: item[0])
    return grouped


def _rolling(values: Sequence[tuple[int, float]], window: int = 20) -> list[tuple[int, float]]:
    if len(values) < window:
        return []
    return [
        (values[index - 1][0], sum(value for _, value in values[index - window : index]) / window)
        for index in range(window, len(values) + 1)
    ]


def _stage_lines(axis: Any) -> None:
    for stage, target in DAY18_MILESTONES.items():
        axis.axvline(target, color="#94a3b8", linewidth=0.8, linestyle="--", alpha=0.6)
        axis.text(
            target,
            0.98,
            stage,
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=7,
            color="#475569",
        )


def _save_figure(figure: Any, paths: Sequence[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="png", dpi=180)


def _write_metadata(
    *,
    output_dir: Path,
    filename: str,
    question: str,
    manifest_path: Path,
    source_entries: Sequence[Mapping[str, Any]],
    outputs: Sequence[Path],
) -> Path:
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "command": list(sys.argv),
        "question": question,
        "source_manifest": relative_path(manifest_path, start=Path.cwd()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_entries": [
            {
                "run_id": entry.get("run_id"),
                "algorithm": entry.get("algorithm"),
                "training_seed": entry.get("training_seed"),
                "stage": entry.get("stage"),
                "target_transitions": entry.get("target_transitions"),
                "run_dir": entry.get("run_dir"),
                "checkpoint": entry.get("checkpoint"),
                "evaluation": entry.get("results"),
                "q_probe": entry.get("path"),
                "throughput_accounting": entry.get("summary", {}).get(
                    "throughput_accounting"
                ),
                "stage_start_counters": entry.get("summary", {}).get(
                    "stage_start_counters"
                ),
                "stage_counters": entry.get("summary", {}).get("stage_counters"),
                "stage_rates": {
                    field: entry.get("runtime", {}).get(field)
                    for field in (
                        "steps_per_second",
                        "environment_transitions_per_second",
                        "physical_environment_steps_per_second",
                        "vector_iterations_per_second",
                        "action_inference_batches_per_second",
                        "action_inference_transitions_per_second",
                        "replay_insertion_calls_per_second",
                        "replay_insertion_transitions_per_second",
                        "optimizer_updates_per_second",
                        "training_samples_per_second",
                    )
                },
            }
            for entry in source_entries
        ],
        "outputs": [path.as_posix() for path in outputs],
    }
    return write_json(output_dir / filename, payload)


def _render_training_local(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = _metrics_series(entries, field="raw_episode_return")
    if not series:
        raise ValueError("completed runs contain no finite raw_episode_return metrics")
    _configure_font()
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True, constrained_layout=True)
    raw_axis, rolling_axis = axes
    for (algorithm, seed), values in sorted(series.items()):
        color = ALGORITHM_COLORS[algorithm]
        label = f"{ALGORITHM_LABELS[algorithm]} seed {seed}"
        raw_axis.plot(
            [step for step, _ in values],
            [value for _, value in values],
            marker=".",
            markersize=2.2,
            linewidth=0.7,
            alpha=0.42,
            color=color,
            label=label,
        )
        smoothed = _rolling(values)
        if smoothed:
            rolling_axis.plot(
                [step for step, _ in smoothed],
                [value for _, value in smoothed],
                linewidth=1.5,
                color=color,
                alpha=0.85,
                label=label,
            )
    _stage_lines(raw_axis)
    raw_axis.set_title("Day 18 training return: every completed episode per seed")
    raw_axis.set_ylabel("Raw Atari episode return")
    rolling_axis.set_title("Same runs with a 20-episode rolling mean")
    rolling_axis.set_xlabel("Environment transitions (global step)")
    rolling_axis.set_ylabel("Rolling mean return")
    for axis in axes:
        axis.grid(True, alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize=7, ncol=2)
    figure.suptitle(
        "DQN vs Double DQN — seed-level training curves from completed runs",
        fontsize=13,
    )
    outputs = (
        output_dir / "dqn-vs-double-training.png",
        output_dir / "training-return-100k-500k.png",
    )
    _save_figure(figure, outputs)
    plt.close(figure)
    _write_metadata(
        output_dir=output_dir,
        filename="dqn-vs-double-training.json",
        question="在 actual environment transitions 軸上，DQN 與 Double DQN 的 episode return 是否呈現可跨 training seed 重複的趨勢？",
        manifest_path=manifest_path,
        source_entries=entries,
        outputs=outputs,
    )
    return outputs


def _render_training_worker(
    manifest_path: str,
    output_dir: str,
    entries: list[dict[str, Any]],
    result_queue: Any,
) -> None:
    try:
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
    """Render training figures, isolating Matplotlib from PyTorch DLLs."""

    if "torch" not in sys.modules:
        return _render_training_local(manifest_path, output_dir, entries)
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


def _evaluation_points(entries: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[(str(entry["algorithm"]), int(entry["training_seed"]))].append(entry)
    for values in grouped.values():
        values.sort(key=lambda entry: int(entry["target_transitions"]))
    return grouped


def render_evaluation(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = _evaluation_points(entries)
    _configure_font()
    figure, axis = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    for (algorithm, seed), values in sorted(grouped.items()):
        color = ALGORITHM_COLORS[algorithm]
        axis.errorbar(
            [int(value["target_transitions"]) for value in values],
            [float(value["summary"]["mean_return"]) for value in values],
            yerr=[float(value["summary"]["std_return"]) for value in values],
            marker="o",
            linewidth=1.1,
            markersize=5,
            capsize=3,
            alpha=0.85,
            color=color,
            label=f"{ALGORITHM_LABELS[algorithm]} seed {seed}",
        )
    axis.set_xticks([250_000, 500_000], ["250K pilot", "500K main"])
    axis.set_xlabel("Environment transitions at checkpoint")
    axis.set_ylabel("Mean raw Atari return ± episode std")
    axis.set_title("Contract v2 evaluation: preserve each training seed")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize=8, ncol=2)
    outputs = (
        output_dir / "dqn-vs-double-eval.png",
        output_dir / "evaluation-250k-500k.png",
    )
    _save_figure(figure, outputs)
    plt.close(figure)
    _write_metadata(
        output_dir=output_dir,
        filename="dqn-vs-double-eval.json",
        question="在固定 Contract v2 evaluation seeds、epsilon=0 與 raw reward 下，250K/500K 的 per-training-seed evaluation spread 是否分開？",
        manifest_path=manifest_path,
        source_entries=entries,
        outputs=outputs,
    )
    return outputs


def render_paired(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    main = [entry for entry in entries if entry.get("stage") == "main"]
    by_key = {
        (int(entry["training_seed"]), str(entry["algorithm"])): entry
        for entry in main
    }
    seeds = sorted({int(entry["training_seed"]) for entry in main})
    if not seeds or any((seed, algorithm) not in by_key for seed in seeds for algorithm in DAY18_ALGORITHMS):
        raise ValueError("paired figure requires both completed algorithms for every main training seed")
    _configure_font()
    figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    for index, seed in enumerate(seeds):
        dqn = float(by_key[(seed, "dqn")]["summary"]["mean_return"])
        double = float(by_key[(seed, "double_dqn")]["summary"]["mean_return"])
        axis.plot(
            [index - 0.12, index + 0.12],
            [dqn, double],
            color="#94a3b8",
            linewidth=1.0,
            zorder=1,
        )
        axis.scatter(index - 0.12, dqn, color=ALGORITHM_COLORS["dqn"], s=48, zorder=2)
        axis.scatter(index + 0.12, double, color=ALGORITHM_COLORS["double_dqn"], s=48, zorder=2)
    axis.set_xticks(range(len(seeds)), [f"seed {seed}" for seed in seeds])
    axis.set_ylabel("500K mean raw Atari return")
    axis.set_title("500K paired seed comparison — connecting lines preserve the pairing")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=ALGORITHM_COLORS["dqn"],
                label="DQN",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=ALGORITHM_COLORS["double_dqn"],
                label="Double DQN",
            ),
        ],
        loc="best",
    )
    output = output_dir / "paired-seed-comparison.png"
    _save_figure(figure, (output,))
    plt.close(figure)
    _write_metadata(
        output_dir=output_dir,
        filename="paired-seed-comparison.json",
        question="在同一 training seed 配對下，500K 的 DQN 與 Double DQN evaluation mean 差異是否方向一致？",
        manifest_path=manifest_path,
        source_entries=main,
        outputs=(output,),
    )
    return output


def render_q_probe(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = _evaluation_points(entries)
    _configure_font()
    figure, axis = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    for (algorithm, seed), values in sorted(grouped.items()):
        color = ALGORITHM_COLORS[algorithm]
        stats = [value["summary"] for value in values]
        axis.errorbar(
            [int(value["target_transitions"]) for value in values],
            [float(stat["max_q_mean"]) for stat in stats],
            yerr=[float(stat["max_q_std"]) for stat in stats],
            marker="o",
            linewidth=1.1,
            markersize=5,
            capsize=3,
            alpha=0.85,
            color=color,
            label=f"{ALGORITHM_LABELS[algorithm]} seed {seed}",
        )
    axis.set_xlabel("Environment transitions at checkpoint")
    axis.set_ylabel("Mean max Q-value ± probe std")
    axis.set_title("Fixed Contract v2 probe states: Q scale and spread")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize=8, ncol=2)
    output = output_dir / "q-probe-comparison.png"
    _save_figure(figure, (output,))
    plt.close(figure)
    _write_metadata(
        output_dir=output_dir,
        filename="q-probe-comparison.json",
        question="相同 fixed probe states 上，兩個 algorithm 的 max-Q 分布如何隨 actual training transitions 改變？",
        manifest_path=manifest_path,
        source_entries=entries,
        outputs=(output,),
    )
    return output


def render_runtime(
    manifest_path: Path,
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_font()
    algorithms = list(DAY18_ALGORITHMS)
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.8), constrained_layout=True)
    metrics = (
        ("steps_per_second", "Transitions / second"),
        ("wall_clock_seconds", "Wall-clock seconds"),
        ("peak_allocated_vram_bytes", "Peak allocated VRAM (MiB)"),
    )
    plotted_any = False
    seeds = sorted({int(entry["training_seed"]) for entry in entries})
    for axis, (field, title) in zip(axes, metrics, strict=True):
        values_by_algorithm: dict[str, list[tuple[float, float]]] = {
            algorithm: [] for algorithm in algorithms
        }
        by_key = {
            (str(entry["algorithm"]), int(entry["training_seed"])): entry
            for entry in entries
        }
        for algorithm in algorithms:
            for seed_index, seed in enumerate(seeds):
                entry = by_key.get((algorithm, seed))
                if entry is None:
                    continue
                runtime = entry.get("runtime", {})
                if not isinstance(runtime, Mapping):
                    runtime = {}
                if field == "steps_per_second":
                    value = _number(runtime.get("steps_per_second"))
                elif field == "wall_clock_seconds":
                    value = _number(runtime.get("wall_clock_seconds"))
                else:
                    raw = _number(runtime.get("cuda_peak_allocated_bytes"))
                    value = None if raw is None else raw / (1024 * 1024)
                if value is not None:
                    offset = -0.12 if algorithm == "dqn" else 0.12
                    values_by_algorithm[algorithm].append((seed_index + offset, value))
        for algorithm in algorithms:
            algorithm_values = values_by_algorithm[algorithm]
            if not algorithm_values:
                continue
            axis.scatter(
                [x for x, _ in algorithm_values],
                [value for _, value in algorithm_values],
                color=ALGORITHM_COLORS[algorithm],
                label=ALGORITHM_LABELS[algorithm],
                s=38,
            )
            plotted_any = True
        axis.set_xticks(range(len(seeds)), [f"seed {seed}" for seed in seeds])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if field == "peak_allocated_vram_bytes":
            axis.set_ylabel("MiB")
        else:
            axis.set_ylabel("measured value")
        axis.legend(loc="best", fontsize=8)
    if not plotted_any:
        plt.close(figure)
        raise ValueError("completed runs contain no reliable runtime metrics")
    figure.suptitle("Day 18 engineering cost at the 500K milestone", fontsize=13)
    output = output_dir / "runtime-comparison.png"
    _save_figure(figure, (output,))
    plt.close(figure)
    _write_metadata(
        output_dir=output_dir,
        filename="runtime-comparison.json",
        question="在相同 NVIDIA CUDA、precision 與 actual transition budget 下，兩個 algorithm 的 SPS、wall-clock 與 peak VRAM 工程成本是多少？",
        manifest_path=manifest_path,
        source_entries=entries,
        outputs=(output,),
    )
    return output


def render_all(manifest_path: str | Path, output_dir: str | Path) -> list[Path]:
    source = Path(manifest_path).resolve()
    read_day18_manifest(source)
    training = _require_completed(
        load_training_entries(
            source,
            include_metrics=True,
            require_checkpoint=False,
        ),
        name="training",
    )
    all_evaluations = _require_completed(
        load_evaluation_entries(source, training),
        name="evaluation",
    )
    evaluations = [
        entry
        for entry in all_evaluations
        if entry.get("stage") in {"pilot", "main"}
    ]
    if not evaluations:
        raise ValueError("completed 250K/500K evaluation artifacts are required")
    q_probe = _require_completed(load_q_probe_entries(source, training), name="Q probe")
    output = Path(output_dir)
    paths: list[Path] = []
    paths.extend(render_training(source, output, training))
    paths.extend(render_evaluation(source, output, evaluations))
    paths.append(render_paired(source, output, evaluations))
    paths.append(render_q_probe(source, output, q_probe))
    main_runtime = [entry for entry in training if entry.get("stage") == "main"]
    paths.append(render_runtime(source, output, _require_completed(main_runtime, name="main runtime")))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = render_all(args.manifest, args.output_dir)
    except (FileNotFoundError, TypeError, ValueError, OSError, RuntimeError) as error:
        print(f"Unable to render Day 18 comparison figures: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"outputs": [path.as_posix() for path in paths]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
