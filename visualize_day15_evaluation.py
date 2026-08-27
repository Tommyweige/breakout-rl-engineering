"""Plot the real Day 15 Random-versus-DQN evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import queue
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Mapping, Sequence

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_evaluation_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: evaluation results must be a JSON object")
    if not isinstance(payload.get("per_episode"), list) or not payload["per_episode"]:
        raise ValueError(f"{path}: per_episode must be a non-empty array")
    return payload


def _episode_rows(payload: Mapping[str, Any], *, source: Path) -> list[dict[str, Any]]:
    raw_rows = payload.get("per_episode")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{source}: per_episode must be a non-empty array")
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{source}: every per_episode item must be an object")
        try:
            raw_return = raw_row.get("episode_return")
            if raw_return is None:
                raw_return = raw_row["return"]
            rows.append(
                {
                    "evaluation_seed": int(raw_row["evaluation_seed"]),
                    "episode_index": int(raw_row["episode_index"]),
                    "episode_return": float(raw_return),
                    "evaluator_truncated": bool(
                        raw_row.get("evaluator_truncated", False)
                    ),
                    "stop_reason": str(raw_row.get("stop_reason", "unknown")),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source}: malformed per_episode item") from error
    return rows


def _group_returns(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        seed = int(row["evaluation_seed"])
        value = float(row["episode_return"])
        grouped.setdefault(seed, []).append(value)
    return grouped


def _configure_font() -> None:
    """Use an installed Traditional Chinese font when running on Windows."""

    import matplotlib
    from matplotlib import font_manager

    for candidate in (
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/mingliu.ttc"),
        Path("C:/Windows/Fonts/NotoSansTC-VF.ttf"),
    ):
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            font = font_manager.FontProperties(fname=str(candidate))
            matplotlib.rcParams["font.family"] = [font.get_name()]
            return


def _validate_payloads(
    random_payload: Mapping[str, Any],
    dqn_payload: Mapping[str, Any],
    *,
    random_source: Path,
    dqn_source: Path,
) -> None:
    if random_payload.get("policy_type") != "random":
        raise ValueError(f"{random_source}: expected policy_type=random")
    if dqn_payload.get("policy_type") != "dqn":
        raise ValueError(f"{dqn_source}: expected policy_type=dqn")
    if random_payload.get("evaluation_seeds") != dqn_payload.get("evaluation_seeds"):
        raise ValueError("Random and DQN results must use the same evaluation seeds")
    if random_payload.get("episodes_per_seed") != dqn_payload.get("episodes_per_seed"):
        raise ValueError(
            "Random and DQN results must use the same episodes_per_seed value"
        )


def _render_local(
    random_source: Path,
    dqn_source: Path,
    random_payload: Mapping[str, Any],
    dqn_payload: Mapping[str, Any],
    destination: Path,
    metadata_destination: Path,
) -> None:
    """Render in a process that has not imported Matplotlib alongside Torch."""

    _validate_payloads(
        random_payload,
        dqn_payload,
        random_source=random_source,
        dqn_source=dqn_source,
    )

    random_rows = _episode_rows(random_payload, source=random_source)
    dqn_rows = _episode_rows(dqn_payload, source=dqn_source)
    random_values = [float(row["episode_return"]) for row in random_rows]
    dqn_values = [float(row["episode_return"]) for row in dqn_rows]
    random_groups = _group_returns(random_rows)
    dqn_groups = _group_returns(dqn_rows)
    seed_values = [int(seed) for seed in random_payload["evaluation_seeds"]]

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
    for position, values, label in zip(positions, values_by_policy, labels):
        policy_rows = random_rows if label == "Random" else dqn_rows
        for marker, rows in (
            ("o", [row for row in policy_rows if not row["evaluator_truncated"]]),
            ("x", [row for row in policy_rows if row["evaluator_truncated"]]),
        ):
            if not rows:
                continue
            plotted_values = [float(row["episode_return"]) for row in rows]
            if len(plotted_values) == 1:
                jitter = [0.0]
            else:
                jitter = [
                    -0.12 + 0.24 * index / (len(plotted_values) - 1)
                    for index in range(len(plotted_values))
                ]
            scatter_kwargs: dict[str, Any] = {
                "s": 26 if marker == "o" else 34,
                "marker": marker,
                "color": colors[label],
                "alpha": 0.8,
                "linewidths": 0.4,
                "zorder": 3,
            }
            if marker == "o":
                scatter_kwargs["edgecolors"] = "white"
            distribution_axis.scatter(
                [position + offset for offset in jitter],
                plotted_values,
                **scatter_kwargs,
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
        distribution_axis.scatter(
            [position],
            [median(values)],
            marker="_",
            s=190,
            color="#111827",
            linewidths=1.4,
            zorder=5,
        )
    distribution_axis.set_xticks(positions, labels)
    distribution_axis.set_ylabel("Raw Atari episode return")
    distribution_axis.set_title("每局回報分布")
    distribution_axis.grid(axis="y", alpha=0.25)
    distribution_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#475569",
                markeredgecolor="#111827",
                markersize=6,
                label="Mean",
            ),
            Line2D(
                [0],
                [0],
                marker="_",
                color="#111827",
                markersize=12,
                label="Median",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#475569",
                markerfacecolor="#475569",
                markersize=5,
                label="Episode",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="#475569",
                markersize=6,
                linestyle="none",
                label="Evaluator cap",
            ),
        ],
        loc="best",
        fontsize=8,
    )

    random_means = [fmean(random_groups[seed]) for seed in seed_values]
    dqn_means = [fmean(dqn_groups[seed]) for seed in seed_values]
    random_stds = [pstdev(random_groups[seed]) for seed in seed_values]
    dqn_stds = [pstdev(dqn_groups[seed]) for seed in seed_values]
    seed_positions = list(range(1, len(seed_values) + 1))
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
    seed_axis.set_xticks(seed_positions, [str(seed) for seed in seed_values])
    seed_axis.set_xlabel("Evaluation seed group")
    seed_axis.set_ylabel("Mean raw return ± population std")
    seed_axis.set_title("各 seed group 的平均與 spread")
    seed_axis.grid(axis="y", alpha=0.25)
    seed_axis.legend(loc="best", fontsize=8)

    episode_count = min(len(random_values), len(dqn_values))
    figure.suptitle(
        "Day 15: Random vs frozen DQN evaluation",
        fontsize=13,
        y=0.99,
    )
    figure.text(
        0.5,
        0.01,
        f"每個 policy {episode_count} 局；x 表示 evaluator cap；資料來自 results.json；分數是 raw Atari reward",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="png")
    plt.close(figure)

    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "command": list(sys.argv),
        "question": "在固定 evaluation seeds 與 raw reward 下，DQN 是否穩定高於 Random？",
        "source_artifacts": {
            "random_results": random_source.as_posix(),
            "random_results_sha256": _sha256(random_source),
            "dqn_results": dqn_source.as_posix(),
            "dqn_results_sha256": _sha256(dqn_source),
        },
        "evaluation_protocol": {
            "seeds": seed_values,
            "episodes_per_seed": random_payload.get("episodes_per_seed"),
            "epsilon": {
                "random": random_payload.get("evaluation_epsilon"),
                "dqn": dqn_payload.get("evaluation_epsilon"),
            },
            "random_resolved_device": random_payload.get("resolved_device"),
            "dqn_resolved_device": dqn_payload.get("resolved_device"),
            "evaluator_truncated_episodes": {
                "random": sum(bool(row["evaluator_truncated"]) for row in random_rows),
                "dqn": sum(bool(row["evaluator_truncated"]) for row in dqn_rows),
            },
        },
        "output": destination.as_posix(),
        "metadata_output": metadata_destination.as_posix(),
    }
    metadata_destination.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
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
    """Render per-episode distributions and per-seed summaries from JSON."""

    random_source = Path(random_results_path)
    dqn_source = Path(dqn_results_path)
    random_payload = _read_evaluation_results(random_source)
    dqn_payload = _read_evaluation_results(dqn_source)
    _validate_payloads(
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
    except (FileNotFoundError, TypeError, ValueError, OSError) as error:
        print(f"Unable to render Day 15 evaluation: {error}", file=sys.stderr)
        return 2
    print(f"Wrote Day 15 evaluation figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
