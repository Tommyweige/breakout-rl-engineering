"""Export a compact, real-data Day 18 evidence manifest for figure rebuilds."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.day18_comparison import (
    build_day18_report,
    compact_training_summary,
    load_training_entries,
    read_day18_manifest,
    relative_path,
    resolve_manifest_reference,
    sha256_file,
    utc_timestamp,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export compact real Day 18 artifacts used to rebuild figures."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day18-dqn-vs-double/manifest.json"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("assets/day18/evidence-manifest.json"),
    )
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help="require the complete three-seed 500K comparison before exporting",
    )
    return parser


def _copy_metrics(source: Path, destination: Path) -> int:
    """Keep raw episode completions and diagnostic rows from the real CSV."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = []
        diagnostic_fields = (
            "loss",
            "q_mean",
            "q_max",
            "q_min",
            "target_mean",
            "target_max",
            "td_error_mean_abs",
            "td_error_max_abs",
            "gradient_norm",
        )
        for row in reader:
            if row.get("raw_episode_return") not in (None, "") or any(
                row.get(field) not in (None, "") for field in diagnostic_fields
            ):
                rows.append(dict(row))
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_evidence(
    manifest_path: str | Path,
    output_manifest: str | Path,
    *,
    require_formal: bool = False,
) -> Path:
    source = Path(manifest_path).resolve()
    manifest = read_day18_manifest(source)
    training_entries = load_training_entries(
        source,
        include_metrics=False,
        require_checkpoint=False,
    )
    training_by_key = {
        (
            str(entry["algorithm"]),
            int(entry["training_seed"]),
            str(entry["stage"]),
        ): entry
        for entry in training_entries
    }
    if require_formal:
        report = build_day18_report(source)
        if not report["comparison_conditions"].get("formal_quality_eligible"):
            raise ValueError(
                "the source manifest is not formal-quality eligible; refusing compact export"
            )
    destination = Path(output_manifest).resolve()
    output_root = destination.parent
    output_root.mkdir(parents=True, exist_ok=True)
    clone = json.loads(json.dumps(manifest, ensure_ascii=False))
    clone["evidence_mode"] = (
        "compact real artifacts: episode-completion and diagnostic metrics rows; "
        "full checkpoints remain in the local run tree"
    )
    clone["source_manifest"] = {
        "path": relative_path(source, start=output_root),
        "sha256": sha256_file(source),
    }
    copied_metrics: dict[str, int] = {}
    for entry in clone["runs"]:
        if entry.get("status") != "completed":
            continue
        source_run = resolve_manifest_reference(source, entry.get("run_dir"))
        if not source_run.is_dir():
            continue
        compact_name = (
            f"{entry['algorithm']}-seed{int(entry['training_seed'])}-"
            f"{entry['stage']}"
        )
        compact_run = output_root / "evidence-runs" / compact_name
        compact_run.mkdir(parents=True, exist_ok=True)
        training_report = training_by_key.get(
            (
                str(entry["algorithm"]),
                int(entry["training_seed"]),
                str(entry["stage"]),
            )
        )
        for name in ("config.json", "summary.json"):
            source_file = source_run / name
            if source_file.is_file():
                if name == "summary.json" and training_report is not None:
                    write_json(compact_run / name, training_report["summary"])
                elif name == "config.json" and training_report is not None:
                    config_payload = json.loads(
                        source_file.read_text(encoding="utf-8")
                    )
                    if isinstance(config_payload, dict):
                        config_payload["runtime"] = dict(
                            training_report.get("runtime", {})
                        )
                    write_json(compact_run / name, config_payload)
                else:
                    shutil.copy2(source_file, compact_run / name)
        source_metrics = source_run / "metrics.csv"
        if not source_metrics.is_file():
            continue
        copied_metrics[compact_name] = _copy_metrics(
            source_metrics,
            compact_run / "metrics.csv",
        )
        entry["source_run_dir"] = entry.get("run_dir")
        entry["run_dir"] = relative_path(compact_run, start=output_root)
        if training_report is not None:
            entry["summary"] = compact_training_summary(training_report["summary"])

        raw_q_path = entry.get("q_probe")
        if isinstance(raw_q_path, str):
            source_q = resolve_manifest_reference(source, raw_q_path)
            if source_q.is_file():
                compact_q = output_root / "evidence-q" / f"{compact_name}.json"
                compact_q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_q, compact_q)
                entry["q_probe"] = relative_path(compact_q, start=output_root)

        evaluation = entry.get("evaluation")
        if isinstance(evaluation, dict):
            for field in ("directory", "results", "episodes"):
                raw_value = evaluation.get(field)
                if not isinstance(raw_value, str):
                    continue
                original = resolve_manifest_reference(source, raw_value)
                if original.exists():
                    evaluation[field] = relative_path(original, start=output_root)

    clone["compact_metrics_rows"] = copied_metrics
    clone["compact_exported_at_utc"] = utc_timestamp()
    write_json(destination, clone)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = export_evidence(
            args.manifest,
            args.output_manifest,
            require_formal=args.require_formal,
        )
    except (FileNotFoundError, TypeError, ValueError, OSError) as error:
        print(f"Unable to export Day 18 evidence: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output_manifest": output.as_posix()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
