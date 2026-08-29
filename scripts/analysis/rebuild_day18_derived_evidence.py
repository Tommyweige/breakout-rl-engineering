"""Rebuild Day 18 derived evidence from existing run artifacts.

This command never starts a trainer. It reconstructs stage-local throughput
from the completed run summaries, refreshes compact evidence, then regenerates
the report and figures from the unchanged evaluations and Q probes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from breakout_rl.day18_comparison import (
    build_day18_report,
    compact_training_summary,
    day18_source_hashes,
    historical_run_provenance,
    load_training_entries,
    read_day18_manifest,
    write_json,
    utc_timestamp,
)
from scripts.analysis.export_day18_evidence import export_evidence
from scripts.visualization.visualize_day18_comparison import render_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild Day 18 derived evidence without training or changing "
            "evaluation/Q-probe artifacts."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day18-dqn-vs-double/manifest.json"),
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=Path("assets/day18/evidence-manifest.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("assets/day18/comparison-report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/day18"),
    )
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help="require all completed formal 500K evidence before rebuilding",
    )
    return parser


def rebuild_derived_evidence(
    manifest_path: str | Path,
    *,
    evidence_manifest_path: str | Path,
    report_path: str | Path,
    output_dir: str | Path,
    require_formal: bool = False,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = read_day18_manifest(source)
    training = load_training_entries(
        source,
        include_metrics=False,
        require_checkpoint=False,
    )
    if require_formal:
        existing_report = build_day18_report(source)
        if not existing_report["comparison_conditions"].get(
            "formal_quality_eligible",
            False,
        ):
            raise ValueError(
                "the existing evidence is not formal-quality eligible; refusing rebuild"
            )

    by_key = {
        (
            str(entry["algorithm"]),
            int(entry["training_seed"]),
            str(entry["stage"]),
        ): entry
        for entry in training
    }
    for entry in manifest["runs"]:
        key = (
            str(entry["algorithm"]),
            int(entry["training_seed"]),
            str(entry["stage"]),
        )
        normalized = by_key.get(key)
        if normalized is not None and normalized.get("summary"):
            entry["summary"] = compact_training_summary(normalized["summary"])

    repository_root = source.parent.parent.parent
    manifest["provenance"] = {
        "source_hashes": day18_source_hashes(repository_root),
        "historical_run_worktree_provenance": historical_run_provenance(training),
        "derived_artifact_rebuild": {
            "method": "existing cumulative counters + previous stage snapshot + stage wall-clock",
            "training_performed": False,
            "evaluation_artifacts_rewritten": False,
            "q_probe_artifacts_rewritten": False,
        },
    }
    manifest["updated_at_utc"] = utc_timestamp()
    write_json(source, manifest)

    export_evidence(
        source,
        evidence_manifest_path,
        require_formal=require_formal,
    )
    report = build_day18_report(source, output_path=report_path)
    outputs = render_all(source, output_dir)
    return {
        "manifest": source.as_posix(),
        "evidence_manifest": Path(evidence_manifest_path).resolve().as_posix(),
        "report": Path(report_path).resolve().as_posix(),
        "outputs": [path.as_posix() for path in outputs],
        "formal_quality_eligible": report["comparison_conditions"].get(
            "formal_quality_eligible",
            False,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = rebuild_derived_evidence(
            args.manifest,
            evidence_manifest_path=args.evidence_manifest,
            report_path=args.report,
            output_dir=args.output_dir,
            require_formal=args.require_formal,
        )
    except (FileNotFoundError, TypeError, ValueError, OSError, RuntimeError) as error:
        print(f"Unable to rebuild Day 18 derived evidence: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
