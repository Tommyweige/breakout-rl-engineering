"""Generate a source-backed Day 18 DQN versus Double DQN report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from breakout_rl.day18_comparison import build_day18_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate completed Day 18 training, evaluation, and Q-probe artifacts."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day18-dqn-vs-double/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day18/comparison-report.json"),
    )
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help="return a non-zero status unless the 500K three-pair CUDA comparison is eligible",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_day18_report(args.manifest, output_path=args.output)
    except (FileNotFoundError, TypeError, ValueError, OSError) as error:
        print(f"Unable to generate Day 18 comparison report: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "conclusion": report["conclusion"],
                "comparison_conditions": report["comparison_conditions"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.require_formal and not report["comparison_conditions"].get(
        "formal_quality_eligible",
        False,
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
