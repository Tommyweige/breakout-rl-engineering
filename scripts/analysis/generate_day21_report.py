"""Generate the Day 21 machine-readable and Markdown reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from breakout_rl.day21_final_training import (
    build_day21_report,
    load_day21_config,
    render_day21_markdown,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Day 21 final long-training report."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day21-final-long-training/manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final-training/manifest.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("assets/day21/final-training-report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/day21-final-long-training.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_day21_config(args.config)
        report = build_day21_report(args.manifest, config=config)
        write_json(args.json_output, report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_day21_markdown(report), encoding="utf-8")
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"Day 21 report generation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "json": args.json_output.as_posix(),
                "report": args.output.as_posix(),
                "status": report.get("status"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
