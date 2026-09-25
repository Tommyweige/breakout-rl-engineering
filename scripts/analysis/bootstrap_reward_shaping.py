"""Run Issue #10 bootstrap uncertainty analysis on Issue #9 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from breakout_rl.analysis.bootstrap_reward_shaping import (
    ALLOWED_RAW_SCORE_DEFINITIONS,
    BASELINE_LABEL,
    CHECKPOINT_STEP,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONTRACT_PATH,
    DEFAULT_EVALUATION_DIRS,
    DEFAULT_ITERATIONS,
    EXPECTED_EVALUATION_SEEDS,
    EXPECTED_SCORE_DEFINITION,
    PairedRecord,
    REQUIRED_TRAINING_SEEDS,
    SHAPED_LABEL,
    build_analysis,
    hierarchical_bootstrap,
    load_paired_artifacts,
    paired_bootstrap,
    render_analysis_report,
    training_seed_level_bootstrap,
    validate_paired_records,
    validate_paired_rows,
    write_analysis_artifacts,
)


__all__ = [
    "ALLOWED_RAW_SCORE_DEFINITIONS",
    "BASELINE_LABEL",
    "CHECKPOINT_STEP",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONTRACT_PATH",
    "DEFAULT_EVALUATION_DIRS",
    "DEFAULT_ITERATIONS",
    "EXPECTED_EVALUATION_SEEDS",
    "EXPECTED_SCORE_DEFINITION",
    "PairedRecord",
    "REQUIRED_TRAINING_SEEDS",
    "SHAPED_LABEL",
    "build_analysis",
    "hierarchical_bootstrap",
    "load_paired_artifacts",
    "paired_bootstrap",
    "render_analysis_report",
    "training_seed_level_bootstrap",
    "validate_paired_records",
    "validate_paired_rows",
    "write_analysis_artifacts",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed2022-evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIRS[2022],
    )
    parser.add_argument(
        "--seed2023-evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIRS[2023],
    )
    parser.add_argument(
        "--seed2024-evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIRS[2024],
    )
    parser.add_argument(
        "--evaluation",
        action="append",
        default=None,
        metavar="TRAINING_SEED=DIR",
        help=(
            "add or replace an evaluation source; repeat for future training "
            "seeds (for example 2025=path/to/evaluation)"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="bootstrap iterations; Issue #10 requires at least 10000",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--contract-path",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="canonical Contract v2 path used to validate source artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/issue-9-reward-shaping/bootstrap-analysis"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_evaluation_dirs(args: argparse.Namespace) -> dict[int, Path]:
    evaluation_dirs = {
        2022: args.seed2022_evaluation_dir,
        2023: args.seed2023_evaluation_dir,
        2024: args.seed2024_evaluation_dir,
    }
    explicitly_seen: set[int] = set()
    for value in args.evaluation or []:
        seed_text, separator, directory = value.partition("=")
        if not separator or not seed_text or not directory:
            raise ValueError("--evaluation must use TRAINING_SEED=DIR syntax")
        try:
            training_seed = int(seed_text)
        except ValueError as error:
            raise ValueError(
                "--evaluation training seed must be an integer"
            ) from error
        if training_seed in explicitly_seen:
            raise ValueError(
                f"duplicate training seed in --evaluation: {training_seed}"
            )
        explicitly_seen.add(training_seed)
        evaluation_dirs[training_seed] = Path(directory)
    return evaluation_dirs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation_dirs = _resolve_evaluation_dirs(args)
    records, sources, contract_metadata = load_paired_artifacts(
        evaluation_dirs,
        contract_path=args.contract_path,
    )
    summary, distributions = build_analysis(
        records,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
        sources=sources,
        contract_metadata=contract_metadata,
    )
    paths = write_analysis_artifacts(
        args.output_dir,
        summary,
        distributions,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {name: path.as_posix() for name, path in paths.items()},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
