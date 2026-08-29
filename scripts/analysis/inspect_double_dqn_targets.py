"""Inspect Vanilla and Double DQN targets on a crafted batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from breakout_rl.analysis.target_inspection import build_target_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a target-rule comparison with distinct online/target outputs."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_target_comparison(seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
