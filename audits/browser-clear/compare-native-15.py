#!/usr/bin/env python3
"""Compare the native-ALE 15-seed evaluation with the browser WASM 15 seeds.

Run the official evaluation first:

    python -m scripts.evaluation.evaluate_predictive_controller \
        --output-dir outputs/issue25-vision-controller

Then:

    python audits/browser-clear/compare-native-15.py outputs/issue25-vision-controller
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BROWSER_SUMMARY = SCRIPT_DIR / "summary-20.json"
CONTRACT_SEEDS = (101, 102, 103, 104, 105, 202, 203, 204, 205, 206, 303, 304, 305, 306, 307)


def load_native(directory: Path) -> dict[int, dict]:
    rows = {}
    with (directory / "episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows[int(row["episode_seed"])] = {
                "score": float(row["raw_score"]),
                "cleared": row["cleared"] == "True",
                "steps": int(row["total_agent_steps"]),
                "frames": int(row["total_emulator_frames"]),
                "failure": row["failure_reason"],
            }
    return rows


def load_browser() -> dict[int, dict]:
    payload = json.loads(BROWSER_SUMMARY.read_text())
    return {
        episode["seed"]: {
            "score": episode["score"],
            "cleared": episode["canonical_cleared"],
            "steps": episode["steps"],
        }
        for episode in payload["episodes"]
        if episode["seed"] in CONTRACT_SEEDS
    }


def summarize(label: str, rows: dict[int, dict]) -> dict:
    scores = [rows[seed]["score"] for seed in CONTRACT_SEEDS]
    return {
        "label": label,
        "episodes": len(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "max": max(scores),
        "clears": sum(1 for seed in CONTRACT_SEEDS if rows[seed]["cleared"]),
    }


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/issue25-vision-controller")
    native = load_native(directory)
    browser = load_browser()
    print("| seed | native score | native cleared | browser score | browser cleared | native steps | browser steps |")
    print("|---|---|---|---|---|---|---|")
    for seed in CONTRACT_SEEDS:
        n, b = native[seed], browser[seed]
        print(f"| {seed} | {n['score']:.0f} | {'yes' if n['cleared'] else 'no'} | {b['score']} | {'yes' if b['cleared'] else 'no'} | {n['steps']} | {b['steps']} |")
    native_summary = summarize("native", native)
    browser_summary = summarize("browser", browser)
    print()
    print("| runtime | mean | median | max | canonical clears |")
    print("|---|---|---|---|---|")
    for summary in (native_summary, browser_summary):
        print(f"| {summary['label']} | {summary['mean']:.1f} | {summary['median']:.1f} | {summary['max']:.0f} | {summary['clears']}/15 |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
