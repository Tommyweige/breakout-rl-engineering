"""Audit whether Day 18 DQN evidence is compatible with Day 20."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from breakout_rl.day20_comparison import (
    audit_day18_evidence_reuse,
    load_day20_config,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check Day 18 DQN/Double DQN evidence against the Day 20 "
            "family-comparison protocol."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/comparisons/dqn-family/manifest.json"),
    )
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/day20/evidence-reuse-audit.json"),
    )
    parser.add_argument(
        "--fail-on-incompatible",
        action="store_true",
        help="return a non-zero status when reuse is not allowed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_day20_config(args.config, require_probe_states=False)
        audit = audit_day18_evidence_reuse(
            config,
            source_manifest=args.source_manifest,
        )
        write_json(args.output, audit)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"Day 20 evidence reuse audit failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.fail_on_incompatible and audit.get("reuse_allowed") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
