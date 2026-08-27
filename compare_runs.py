"""Compare DQN run artifacts without treating incomplete runs as successes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from breakout_rl.experiments import compare_manifest, compare_run_dirs, write_json_object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DQN run directories or an experiment manifest."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--recent-window", type=int, default=20)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON report path (the report is also printed)",
    )
    return parser


def compare_inputs(
    inputs: Sequence[str | Path],
    *,
    recent_window: int = 20,
    rolling_window: int = 20,
) -> dict:
    if not inputs:
        raise ValueError("at least one manifest or run directory is required")
    paths = [Path(value) for value in inputs]
    if len(paths) == 1 and paths[0].is_file():
        return compare_manifest(
            paths[0],
            recent_window=recent_window,
            rolling_window=rolling_window,
        )
    if any(path.is_file() for path in paths):
        raise ValueError("a manifest cannot be combined with run directories")
    return compare_run_dirs(
        paths,
        recent_window=recent_window,
        rolling_window=rolling_window,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_inputs(
            args.inputs,
            recent_window=args.recent_window,
            rolling_window=args.rolling_window,
        )
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Unable to compare runs: {error}")
        return 2
    if args.output is not None:
        write_json_object(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
