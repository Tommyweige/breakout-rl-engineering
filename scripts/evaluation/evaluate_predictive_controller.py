"""Evaluate the deterministic pixel-based Breakout predictive controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from breakout_rl.vision_evaluation import (
    DEFAULT_VISION_CONFIG,
    run_vision_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-seed RGB predictive controller evaluation with the "
            "canonical Breakout clear detector."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_VISION_CONFIG,
        help="versioned deterministic-controller evaluation config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/issue25-vision-controller"),
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="safety guard no greater than the contract limit; partial episodes are never reported",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results_path, episodes_path, diagnostics_path, payload = run_vision_evaluation(
            config_path=args.config,
            output_dir=args.output_dir,
            max_steps_per_episode=args.max_steps_per_episode,
        )
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "results": results_path.as_posix(),
                "episodes": episodes_path.as_posix(),
                "diagnostics": diagnostics_path.as_posix(),
                "report": (args.output_dir / "report.md").as_posix(),
                "policy_type": payload["policy_type"],
                "evaluation_id": payload["evaluation_id"],
                "episode_count": payload["evaluation_protocol"]["episode_count"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
