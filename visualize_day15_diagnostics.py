"""Plot the real Day 15 FIRE/TimeLimit diagnostic comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from breakout_rl.evaluation_artifacts import (
    read_evaluation_results,
    summary_from_episode_rows,
    validate_embedded_summary,
    validate_episode_rows,
)
from breakout_rl.evaluation_contract import load_evaluation_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def _validate_diagnostic_payload(
    payload: Mapping[str, Any],
    *,
    source: Path,
    exact_protocol: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds = payload.get("evaluation_seeds")
    episodes_per_seed = payload.get("episodes_per_seed")
    if not isinstance(seeds, list) or not isinstance(episodes_per_seed, int):
        raise ValueError(f"{source}: diagnostic evaluation protocol is incomplete")
    rows = validate_episode_rows(
        payload,
        source=source,
        expected_seeds=seeds if exact_protocol else None,
        expected_episodes_per_seed=episodes_per_seed if exact_protocol else None,
        require_complete=True,
    )
    computed = summary_from_episode_rows(rows)
    validate_embedded_summary(
        payload,
        computed,
        source=source,
        require_time_limit_fields=True,
    )
    embedded_summary = payload.get("summary")
    if not isinstance(embedded_summary, Mapping):
        raise ValueError(f"{source}: diagnostic summary is required")
    return rows, dict(embedded_summary)


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


def render_diagnostic_figure(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> Path:
    manifest_source = Path(manifest_path)
    manifest = _read_json(manifest_source)
    root = manifest_source.parent.parent.parent
    source_v1 = manifest.get("source_v1_artifacts", {})
    modes = manifest.get("modes", {})
    if not isinstance(source_v1, Mapping) or not isinstance(modes, Mapping):
        raise ValueError(f"{manifest_source}: diagnostic manifest is incomplete")

    v1_summary_path = _resolve(
        root,
        str(source_v1["time_limit_summaries"][1]),
    )
    fire_path = _resolve(root, str(modes["fire_assist"][0]))
    epsilon_path = _resolve(root, str(modes["epsilon005"][0]))
    v1_summary = _read_json(v1_summary_path)
    fire_payload = read_evaluation_results(fire_path)
    epsilon_payload = read_evaluation_results(epsilon_path)
    v1_rows = validate_episode_rows(v1_summary, source=v1_summary_path, require_complete=True)
    fire_rows, fire_summary = _validate_diagnostic_payload(
        fire_payload,
        source=fire_path,
        exact_protocol=True,
    )
    epsilon_rows, epsilon_summary = _validate_diagnostic_payload(
        epsilon_payload,
        source=epsilon_path,
        exact_protocol=True,
    )
    contract_path = root / "configs/eval/breakout_contract_v2.json"
    contract = load_evaluation_contract(contract_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_font()
    labels = ["v1\ngreedy/no assist", "A\ngreedy/FIRE assist", "B\nepsilon=.05"]
    colors = ["#64748b", "#0f766e", "#b45309"]
    row_groups = [v1_rows, fire_rows, epsilon_rows]
    summaries = [
        v1_summary["summary"],
        fire_summary,
        epsilon_summary,
    ]
    returns = [[float(row["episode_return"]) for row in rows] for rows in row_groups]
    positions = [1, 2, 3]

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), dpi=180)
    return_axis, outcome_axis, length_axis, fire_axis = axes.ravel()
    boxplot = return_axis.boxplot(
        returns,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        boxprops={"edgecolor": "#334155", "linewidth": 1.0},
        whiskerprops={"color": "#475569", "linewidth": 1.0},
        capprops={"color": "#475569", "linewidth": 1.0},
        medianprops={"color": "#111827", "linewidth": 1.5},
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for position, values, color in zip(positions, returns, colors):
        jitter = [
            position - 0.16 + 0.32 * index / max(len(values) - 1, 1)
            for index in range(len(values))
        ]
        return_axis.scatter(jitter, values, s=20, color=color, alpha=0.75, zorder=3)
        return_axis.scatter(
            [position],
            [fmean(values)],
            marker="D",
            s=38,
            color=color,
            edgecolors="#111827",
            linewidths=0.7,
            zorder=4,
        )
    return_axis.set_xticks(positions, labels)
    return_axis.set_ylabel("Raw Atari episode return")
    return_axis.set_title("每局回報：v1 與兩個 diagnostic")
    return_axis.grid(axis="y", alpha=0.25)

    terminated = [int(summary["terminated_count"]) for summary in summaries]
    time_limits = [int(summary["time_limit_truncated_count"]) for summary in summaries]
    outcome_axis.bar(positions, terminated, color="#0f766e", label="Terminated")
    outcome_axis.bar(
        positions,
        time_limits,
        bottom=terminated,
        color="#dc2626",
        alpha=0.75,
        label="TimeLimit truncated",
    )
    outcome_axis.set_xticks(positions, labels)
    outcome_axis.set_ylabel("Episode count")
    outcome_axis.set_title("結束原因")
    outcome_axis.legend(fontsize=8)
    outcome_axis.grid(axis="y", alpha=0.25)

    mean_lengths = [float(summary["mean_episode_length"]) for summary in summaries]
    length_axis.bar(positions, mean_lengths, color=colors)
    length_axis.set_yscale("log")
    length_axis.set_xticks(positions, labels)
    length_axis.set_ylabel("Mean agent steps (log scale)")
    length_axis.set_title("平均 episode length")
    length_axis.grid(axis="y", alpha=0.25)

    auto_fire = [int(summary.get("auto_fire_count", 0) or 0) for summary in summaries]
    latency_positions = [2, 3]
    latency = [
        float(fire_summary["mean_life_loss_fire_latency"]),
        float(epsilon_summary["mean_life_loss_fire_latency"]),
    ]
    fire_axis.bar(latency_positions, latency, color=[colors[1], colors[2]])
    fire_axis.set_xticks(latency_positions, ["A FIRE assist", "B epsilon=.05"])
    fire_axis.set_ylabel("Mean life-loss → FIRE steps")
    fire_axis.set_title(f"重發球延遲；auto-FIRE total: {auto_fire[1]} vs {auto_fire[2]}")
    fire_axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Day 15 FIRE/TimeLimit audit: v1 vs diagnostics",
        fontsize=14,
        y=0.99,
    )
    figure.text(
        0.5,
        0.01,
        "真實 results.json / time-limit-summary.json；Option B contract 已由 A/B evidence 選定",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="png")
    plt.close(figure)

    metadata_destination = (
        Path(metadata_path) if metadata_path is not None else destination.with_suffix(".json")
    )
    source_paths = [v1_summary_path, fire_path, epsilon_path, contract_path]
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "command": list(sys.argv),
                "question": "FIRE assist or epsilon exploration removes the v1 TimeLimit/deadlock pattern?",
                "source_artifacts": {
                    path.as_posix(): _sha256(path) for path in source_paths
                },
                "contract_v2": contract.to_dict(),
                "summaries": {
                    "v1": summaries[0],
                    "fire_assist": summaries[1],
                    "epsilon005": summaries[2],
                },
                "output": destination.as_posix(),
                "metadata_output": metadata_destination.as_posix(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Day 15 FIRE/TimeLimit diagnostics.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluations/day15-diagnostics/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day15/fire-time-limit-diagnostics.png"),
    )
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = render_diagnostic_figure(
            args.manifest,
            args.output,
            metadata_path=args.metadata_output,
        )
    except (FileNotFoundError, TypeError, ValueError, OSError) as error:
        print(f"Unable to render Day 15 diagnostic figure: {error}", file=sys.stderr)
        return 2
    print(f"Wrote Day 15 diagnostic figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
