"""Plot the Q-values emitted by ``analyze_q_values.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from breakout_rl.analysis.q_values import plot_q_probe_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize actual Q-values from a probe analysis artifact."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/day17/q-probe-summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day17/q-probe-summary.png"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit("Q analysis must be a JSON object")
    analysis = payload.get("analysis")
    if not isinstance(analysis, Mapping) or "q_values" not in analysis:
        raise SystemExit("Q analysis is missing analysis.q_values")
    plot_q_probe_summary(analysis["q_values"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
