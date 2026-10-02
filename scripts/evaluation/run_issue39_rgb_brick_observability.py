"""Run the frozen Issue #39 RGB brick-removal observability diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from breakout_rl.issue39_rgb_brick_observability import DEFAULT_CONFIG, DEFAULT_OUTPUT_DIR, run_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pre-run-wall-seconds", type=float, default=0.0)
    parser.add_argument("--wall-budget-seconds", type=float, default=600.0)
    args = parser.parse_args()
    paths, result = run_probe(args.config, args.output_dir,
                              pre_run_wall_seconds=args.pre_run_wall_seconds,
                              wall_budget_seconds=args.wall_budget_seconds)
    print(json.dumps({"artifacts": [path.as_posix() for path in paths],
                      "completed_seeds": result["completed_seeds"],
                      "native_frames": result["native_frames"],
                      "classification": result["classification"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
