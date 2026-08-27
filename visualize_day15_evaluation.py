"""Plot real Day 15 Random and frozen-DQN evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import queue
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from breakout_rl.evaluation_artifacts import (
    read_evaluation_results,
    validate_episode_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_results(path: Path) -> dict[str, Any]:
    return read_evaluation_results(path)


def _group_returns(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["evaluation_seed"]), []).append(
            float(row["episode_return"])
        )
    return grouped


def _configure_font() -> None:
    """Use an installed Traditional Chinese font when available on Windows."""

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


def _validate_inputs(
    random_payload: Mapping[str, Any],
    dqn_payload: Mapping[str, Any],
    *,
    random_source: Path,
    dqn_source: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if random_payload.get("policy_type") != "random":
        raise ValueError(f"{random_source}: expected policy_type=random")
    if dqn_payload.get("policy_type") != "dqn":
        raise ValueError(f"{dqn_source}: expected policy_type=dqn")
    if random_payload.get("evaluation_seeds") != dqn_payload.get("evaluation_seeds"):
        raise ValueError("Random and DQN results must use the same evaluation seeds")
    if random_payload.get("episodes_per_seed") != dqn_payload.get("episodes_per_seed"):
        raise ValueError("Random and DQN results must use the same episodes_per_seed")
    raw_seeds = dqn_payload.get("evaluation_seeds")
    try:
        seeds = tuple(int(seed) for seed in raw_seeds)
        episodes_per_seed = int(dqn_payload["episodes_per_seed"])
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("evaluation protocol contains invalid seeds or episode count") from error
    random_rows = validate_episode_rows(
        random_payload,
        source=random_source,
        expected_seeds=seeds,
        expected_episodes_per_seed=episodes_per_seed,
        require_complete=True,
    )
    dqn_rows = validate_episode_rows(
        dqn_payload,
        source=dqn_source,
        expected_seeds=seeds,
        expected_episodes_per_seed=episodes_per_seed,
        require_complete=True,
    )
    return random_rows, dqn_rows


def _render_local(
    random_source: Path,
    dqn_source: Path,
    random_payload: Mapping[str, Any],
    dqn_payload: Mapping[str, Any],
    destination: Path,
    metadata_destination: Path,
) -> None:
    random_rows, dqn_rows = _validate_inputs(
        random_payload,
        dqn_payload,
        random_source=random_source,
        dqn_source=dqn_source,
    )
    random_values = [float(row["episode_return"]) for row in random_rows]
    dqn_values = [float(row["episode_return"]) for row in dqn_rows]
    random_groups = _group_returns(random_rows)
    dqn_groups = _group_returns(dqn_rows)
    seeds = [int(seed) for seed in random_payload["evaluation_seeds"]]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _configure_font()
    colors = {"Random": "#64748b", "DQN": "#0f766e"}
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.9),
        dpi=180,
        gridspec_kw={"width_ratios": (1.05, 1.35)},
    )
    distribution_axis, seed_axis = axes
    labels = ["Random", "DQN"]
    values_by_policy = [random_values, dqn_values]
    positions = [1, 2]
    boxplot = distribution_axis.boxplot(
        values_by_policy,
        positions=positions,
        widths=0.38,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": "#334155", "linewidth": 1.0},
        whiskerprops={"color": "#475569", "linewidth": 1.0},
        capprops={"color": "#475569", "linewidth": 1.0},
        medianprops={"color": "#111827", "linewidth": 1.6},
    )
    for patch, label in zip(boxplot["boxes"], labels):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.3)

    for position, values, rows, label in zip(
        positions,
        values_by_policy,
        (random_rows, dqn_rows),
        labels,
    ):
        jitter = (
            [0.0]
            if len(values) == 1
            else [-0.12 + 0.24 * index / (len(values) - 1) for index in range(len(values))]
        )
        for offset, value, row in zip(jitter, values, rows):
            marker = "s" if row["truncated"] else "o"
            distribution_axis.scatter(
                [position + offset],
                [value],
                s=27 if marker == "o" else 31,
                marker=marker,
                color=colors[label],
                alpha=0.8,
                edgecolors="white" if marker == "o" else colors[label],
                linewidths=0.4,
                zorder=3,
            )
        distribution_axis.scatter(
            [position],
            [fmean(values)],
            marker="D",
            s=38,
            color=colors[label],
            edgecolors="#111827",
            linewidths=0.7,
            zorder=4,
        )

    distribution_axis.set_xticks(positions, labels)
    distribution_axis.set_ylabel("Raw Atari episode return")
    distribution_axis.set_title("每局回報分布")
    distribution_axis.grid(axis="y", alpha=0.25)
    distribution_axis.legend(
        handles=[
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#475569", markeredgecolor="#111827", markersize=6, label="Mean"),
            Line2D([0], [0], color="#111827", linewidth=1.8, label="Median"),
            Line2D([0], [0], marker="o", color="#475569", markerfacecolor="#475569", markersize=5, linestyle="none", label="Terminated episode"),
            Line2D([0], [0], marker="s", color="#475569", markersize=5, linestyle="none", label="Environment truncated"),
        ],
        loc="best",
        fontsize=7.5,
    )

    random_means = [fmean(random_groups[seed]) for seed in seeds]
    dqn_means = [fmean(dqn_groups[seed]) for seed in seeds]
    random_stds = [pstdev(random_groups[seed]) for seed in seeds]
    dqn_stds = [pstdev(dqn_groups[seed]) for seed in seeds]
    seed_positions = list(range(1, len(seeds) + 1))
    seed_axis.errorbar(
        [position - 0.08 for position in seed_positions],
        random_means,
        yerr=random_stds,
        fmt="o",
        capsize=3,
        color=colors["Random"],
        label="Random",
    )
    seed_axis.errorbar(
        [position + 0.08 for position in seed_positions],
        dqn_means,
        yerr=dqn_stds,
        fmt="o",
        capsize=3,
        color=colors["DQN"],
        label="DQN",
    )
    seed_axis.set_xticks(seed_positions, [str(seed) for seed in seeds])
    seed_axis.set_xlabel("Evaluation seed group")
    seed_axis.set_ylabel("Mean raw return ± population std")
    seed_axis.set_title("各 seed group 的平均與 spread")
    seed_axis.grid(axis="y", alpha=0.25)
    seed_axis.legend(loc="best", fontsize=8)

    figure.suptitle("Day 15: Random vs frozen DQN evaluation", fontsize=13, y=0.99)
    figure.text(
        0.5,
        0.01,
        f"每個 policy {min(len(random_values), len(dqn_values))} 局；資料來自 results.json；分數是 raw Atari reward",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="png")
    plt.close(figure)

    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "command": list(sys.argv),
                "question": "固定 evaluation seeds 與 raw reward 下，DQN 的每局分布是否高於 Random？",
                "source_artifacts": {
                    "random_results": random_source.as_posix(),
                    "random_results_sha256": _sha256(random_source),
                    "dqn_results": dqn_source.as_posix(),
                    "dqn_results_sha256": _sha256(dqn_source),
                },
                "dqn_provenance": {
                    "model_id": dqn_payload.get("model_id"),
                    "checkpoint": dqn_payload.get("checkpoint"),
                    "training": dqn_payload.get("training"),
                },
                "evaluation_protocol": {
                    "seeds": seeds,
                    "episodes_per_seed": random_payload.get("episodes_per_seed"),
                    "random_epsilon": random_payload.get("evaluation_epsilon"),
                    "dqn_epsilon": dqn_payload.get("evaluation_epsilon"),
                    "random_resolved_device": random_payload.get("resolved_device"),
                    "dqn_resolved_device": dqn_payload.get("resolved_device"),
                },
                "output": destination.as_posix(),
                "metadata_output": metadata_destination.as_posix(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _render_worker(
    random_source: str,
    dqn_source: str,
    random_payload: dict[str, Any],
    dqn_payload: dict[str, Any],
    destination: str,
    metadata_destination: str,
    result_queue: Any,
) -> None:
    try:
        _render_local(
            Path(random_source),
            Path(dqn_source),
            random_payload,
            dqn_payload,
            Path(destination),
            Path(metadata_destination),
        )
        result_queue.put({"ok": True})
    except Exception as error:  # pragma: no cover - exercised in child process
        result_queue.put({"ok": False, "error": f"{type(error).__name__}: {error}"})


def render_evaluation_comparison(
    random_results_path: str | Path,
    dqn_results_path: str | Path,
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> Path:
    """Render a distribution plot from two raw JSON artifacts."""

    random_source = Path(random_results_path)
    dqn_source = Path(dqn_results_path)
    random_payload = _read_results(random_source)
    dqn_payload = _read_results(dqn_source)
    _validate_inputs(
        random_payload,
        dqn_payload,
        random_source=random_source,
        dqn_source=dqn_source,
    )
    destination = Path(output_path)
    metadata_destination = (
        Path(metadata_path)
        if metadata_path is not None
        else destination.with_suffix(".json")
    )
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_render_worker,
        args=(
            str(random_source),
            str(dqn_source),
            dict(random_payload),
            dict(dqn_payload),
            str(destination),
            str(metadata_destination),
            result_queue,
        ),
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
        raise RuntimeError(f"plot worker failed: {error}")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot real Day 15 Random and DQN evaluation distributions."
    )
    parser.add_argument("random_results", type=Path)
    parser.add_argument("dqn_results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day15/random-vs-dqn-returns.png"),
    )
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = render_evaluation_comparison(
            args.random_results,
            args.dqn_results,
            args.output,
            metadata_path=args.metadata_output,
        )
    except (FileNotFoundError, TypeError, ValueError, OSError, RuntimeError) as error:
        print(f"Unable to render Day 15 evaluation: {error}", file=sys.stderr)
        return 2
    print(f"Wrote Day 15 evaluation figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
