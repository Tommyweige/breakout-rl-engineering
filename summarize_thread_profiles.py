"""Select the CPU thread count from fixed-interval profiling evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from breakout_rl.experiments import write_json_object


def _numeric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def summarize(report_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source = Path(report_path).resolve()
    report = json.loads(source.read_text(encoding="utf-8"))
    runs = list(report.get("runs", []))
    observed = {int(run["cpu_threads"]) for run in runs}
    required = {1, 2, 4}
    if observed != required:
        raise ValueError(f"thread profile must contain exactly 1/2/4; observed {sorted(observed)}")
    completed = [
        run
        for run in runs
        if run.get("status") == "completed"
        and all(int(value) > 0 for value in run.get("finite_metric_counts", {}).values())
    ]
    if len(completed) != len(runs):
        raise ValueError("cannot select CPU threads while a profile is incomplete or non-finite")
    selected = max(completed, key=lambda run: _numeric(run.get("end_to_end_sps")) or float("-inf"))
    result = {
        "schema_version": 1,
        "source_report": str(source),
        "profiles": runs,
        "selection": {
            "rule": "select the completed 1/2/4 setting with the highest end-to-end SPS; retain learning and memory guardrails",
            "selected_cpu_threads": selected["cpu_threads"],
            "selected_run_id": selected["run_id"],
            "selected_end_to_end_sps": selected["end_to_end_sps"],
            "selected_wall_clock_seconds": selected["wall_clock_seconds"],
        },
    }
    write_json_object(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = summarize(args.report, args.output)
    print(json.dumps(result["selection"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
