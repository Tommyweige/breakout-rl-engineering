"""CLI entry point for the reproducible Issue 21 baseline experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from breakout_rl.contract_v3_baseline import (
    REPOSITORY_ROOT,
    run_formal_experiment,
    run_preflight,
)
from scripts.analysis.control_diagnostics import summarize_training_control_metrics


DEFAULT_CONFIG = Path("configs/training/issue21_contract_v3_baseline.json")
DEFAULT_RUN_DIR = Path("runs/issue21-contract-v3-seed11")


def _resolve_from_repository(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the continuous Issue 21 Contract v3 baseline, evaluate its "
            "400K/1M/2M checkpoints, and write a local provenance report."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--evaluation-device", default="cuda")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the contract and run a short real-ALE optimizer smoke",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve_from_repository(args.config)
    run_dir = _resolve_from_repository(args.run_dir)
    try:
        if args.preflight_only:
            result: Any = run_preflight(config_path)
        else:
            result = run_formal_experiment(
                config_path=config_path,
                run_dir=run_dir,
                evaluation_device=args.evaluation_device,
                control_summarizer=summarize_training_control_metrics,
            )
    except KeyboardInterrupt:
        print(
            "Issue 21 run interrupted; the local manifest preserves its status.",
            file=sys.stderr,
        )
        return 130
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Issue 21 run failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.preflight_only:
        print(f"Issue 21 artifacts written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
