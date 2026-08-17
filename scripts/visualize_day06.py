"""Generate Day 6 figures from a real q_learning_demo.py execution."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


matplotlib.rcParams["font.family"] = "Microsoft JhengHei"
matplotlib.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "day06"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run q_learning_demo.py and plot its real update trace."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for CSV, JSON and PNG artifacts",
    )
    return parser


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["metadata"]


def _float(row: dict[str, str], name: str) -> float:
    return float(row[name])


def _int(row: dict[str, str], name: str) -> int:
    return int(row[name])


def _plot_learning_curve(
    rows: list[dict[str, str]],
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    states = [int(state) for state in metadata["states"]]
    actions = list(metadata["actions"])
    values: dict[tuple[int, str], list[float]] = {
        (state, action): [0.0] for state in states for action in actions
    }
    update_steps = [0]

    positive_reward_updates: list[tuple[int, float]] = []
    for update_index, row in enumerate(rows, start=1):
        key = (_int(row, "state"), row["action"])
        for state_action in values:
            values[state_action].append(values[state_action][-1])
        values[key][-1] = _float(row, "updated_q")
        update_steps.append(update_index)
        if _float(row, "reward") > 0:
            positive_reward_updates.append(
                (update_index, _float(row, "reward"))
            )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#2563eb", "#f97316", "#14b8a6", "#8b5cf6"]
    color_index = 0
    for state in states:
        for action in actions:
            key = (state, action)
            ax.step(
                update_steps,
                values[key],
                where="post",
                linewidth=2.2,
                color=colors[color_index],
                label=f"Q(state {state}, {action})",
            )
            color_index += 1

    for update_index, reward in positive_reward_updates:
        ax.axvline(
            update_index,
            color="#b45309",
            alpha=0.18,
            linewidth=1.4,
        )
        ax.annotate(
            f"reward = {reward:g}",
            xy=(update_index, 0),
            xytext=(6, 12),
            textcoords="offset points",
            fontsize=9,
            color="#92400e",
            rotation=90,
            va="bottom",
        )

    ax.set_title("Q-value 如何隨真實 Q-Learning update 改變", loc="left")
    ax.set_xlabel("Update step（每一筆實際 transition）")
    ax.set_ylabel("Q-value")
    ax.set_xlim(0, max(len(rows), 1))
    ax.grid(axis="y", color="#d1d5db", alpha=0.55)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(
        1.0,
        1.02,
        (
            f"episodes={metadata['episodes']} · "
            f"alpha={metadata['alpha']} · gamma={metadata['gamma']} · "
            f"epsilon={metadata['epsilon']} · seed={metadata['seed']}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    fig.text(
        0.5,
        0.015,
        "資料來源：q_learning_demo.py 的完整 update trace；不是手動填入的曲線。",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_breakdown_row(rows: list[dict[str, str]]) -> dict[str, str]:
    non_terminal = [
        row
        for row in rows
        if row["terminated"].lower() == "false"
        and _float(row, "next_q_max") > 0
    ]
    if non_terminal:
        return non_terminal[0]
    return max(rows, key=lambda row: abs(_float(row, "td_error")))


def _plot_update_breakdown(
    row: dict[str, str],
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    current_q = _float(row, "current_q")
    reward = _float(row, "reward")
    next_q_max = _float(row, "next_q_max")
    target = _float(row, "target")
    td_error = _float(row, "td_error")
    updated_q = _float(row, "updated_q")
    gamma = float(metadata["gamma"])
    alpha = float(metadata["alpha"])

    labels = [
        "current Q",
        "reward",
        "max Q(next)",
        "target",
        "TD error",
        "updated Q",
    ]
    values = [current_q, reward, next_q_max, target, td_error, updated_q]
    colors = ["#64748b", "#f59e0b", "#2563eb", "#14b8a6", "#ef4444", "#8b5cf6"]

    fig = plt.figure(figsize=(11, 7))
    grid = fig.add_gridspec(2, 1, height_ratios=(3.2, 1.5), hspace=0.18)
    ax = fig.add_subplot(grid[0])
    x_positions = list(range(len(labels)))
    bars = ax.bar(x_positions, values, color=colors, width=0.62)
    ax.axhline(0, color="#374151", linewidth=1)
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("value")
    ax.set_title(
        (
            f"一次真實 Q-Learning update：episode {_int(row, 'episode')}, "
            f"step {_int(row, 'step')} · state {_int(row, 'state')} "
            f"action {row['action']}"
        ),
        loc="left",
    )
    ax.grid(axis="y", color="#d1d5db", alpha=0.55)
    for bar, value in zip(bars, values):
        offset = 4 if value >= 0 else -14
        ax.annotate(
            f"{value:.6f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    formula_ax = fig.add_subplot(grid[1])
    formula_ax.axis("off")
    formula_ax.text(
        0.02,
        0.82,
        f"target = reward + gamma × max_next_q = {reward:.6f} + {gamma:.2f} × {next_q_max:.6f} = {target:.6f}",
        fontsize=11,
        family="monospace",
    )
    formula_ax.text(
        0.02,
        0.51,
        f"td_error = target - current_q = {target:.6f} - {current_q:.6f} = {td_error:.6f}",
        fontsize=11,
        family="monospace",
    )
    formula_ax.text(
        0.02,
        0.20,
        f"updated_q = current_q + alpha × td_error = {current_q:.6f} + {alpha:.2f} × {td_error:.6f} = {updated_q:.6f}",
        fontsize=11,
        family="monospace",
    )
    fig.text(
        0.5,
        0.015,
        "每個數字都直接來自同一次 q_learning_demo.py update。",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_csv = output_dir / "q_learning_trace.csv"
    trace_json = output_dir / "q_learning_trace.json"
    learning_curve = output_dir / "q_value_learning_curve.png"
    update_breakdown = output_dir / "q_learning_update_breakdown.png"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "q_learning_demo.py"),
        "--episodes",
        str(args.episodes),
        "--alpha",
        str(args.alpha),
        "--gamma",
        str(args.gamma),
        "--epsilon",
        str(args.epsilon),
        "--seed",
        str(args.seed),
        "--trace-csv",
        str(trace_csv),
        "--trace-json",
        str(trace_json),
        "--quiet",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    rows = _read_rows(trace_csv)
    metadata = _metadata(trace_json)
    if not rows:
        raise RuntimeError("q_learning_demo.py produced an empty update trace")

    _plot_learning_curve(rows, metadata, learning_curve)
    _plot_update_breakdown(
        _select_breakdown_row(rows),
        metadata,
        update_breakdown,
    )

    print(f"Generated {trace_csv}")
    print(f"Generated {trace_json}")
    print(f"Generated {learning_curve}")
    print(f"Generated {update_breakdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
