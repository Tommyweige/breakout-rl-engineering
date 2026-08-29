"""Compatibility entry point for the staged Day 18 model comparison runner."""

from __future__ import annotations

from scripts.training.run_day18_comparison import (
    build_parser,
    main,
    run_comparison,
)


__all__ = ["build_parser", "main", "run_comparison"]


if __name__ == "__main__":
    raise SystemExit(main())
