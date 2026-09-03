"""Build the Day 20 machine-readable and Markdown comparison reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from breakout_rl.day20_comparison import (
    build_day20_report,
    load_day20_config,
    write_json,
)


def _display(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    protocol = report.get("protocol", {})
    selection = report.get("selection", {})
    reuse = report.get("evidence_reuse", {})
    aggregates = report.get("aggregates", [])
    extension = report.get("extension", {})
    extension_aggregates = (
        extension.get("aggregates", [])
        if isinstance(extension, Mapping)
        else []
    )
    lines = [
        "# Day 20 DQN family comparison",
        "",
        "這份 report 回答的問題是：在同一個 Contract v2、Day 16 CUDA backend、paired training seeds 與 500K actual environment transitions 下，哪個 DQN family 值得進入 Final Long Training？",
        "",
        f"- manifest status: `{report.get('manifest_status')}`",
        f"- formal horizon: `{protocol.get('formal_quality_transitions')}` actual environment transitions",
        f"- training seeds: `{protocol.get('training_seeds')}`",
        f"- evaluation: `{protocol.get('evaluation_seeds')}` × `{protocol.get('episodes_per_evaluation_seed')}` episodes, epsilon `{protocol.get('evaluation_epsilon')}`, raw reward",
        f"- runtime requirement: requested `{protocol.get('requested_device')}`, precision `{protocol.get('precision')}`, sequential `{protocol.get('sequential')}`",
        "",
        "## Evidence reuse",
        "",
        f"Day 18 DQN/Double evidence decision: `{reuse.get('decision', reuse.get('status', 'unavailable'))}`.",
        "Reuse is accepted only after the machine-readable audit confirms the Contract v2, backend controls, seeds, milestones, evaluation/Q-probe artifacts, and CUDA runtime conditions. A failed audit must leave the old entries out of the formal aggregate rather than treating them as zero-valued runs.",
        "",
        "## 500K family evidence",
        "",
        "| family | complete seeds | mean evaluation return | seed spread | mean SPS | mean wall-clock (s) | peak VRAM (bytes) | parameters |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(aggregates, list):
        for aggregate in aggregates:
            if not isinstance(aggregate, Mapping):
                continue
            lines.append(
                "| {family} | {complete}/{total} | {quality} | {spread} | {sps} | {wall} | {vram} | {parameters} |".format(
                    family=aggregate.get("label", aggregate.get("family_id")),
                    complete=aggregate.get("formal_entry_count", 0),
                    total=aggregate.get("training_seed_count", 0),
                    quality=_display(aggregate.get("quality_mean")),
                    spread=_display(aggregate.get("quality_seed_spread")),
                    sps=_display(aggregate.get("mean_sps")),
                    wall=_display(aggregate.get("mean_wall_clock_seconds")),
                    vram=_display(aggregate.get("mean_peak_allocated_vram_bytes")),
                    parameters=_display(aggregate.get("parameter_count")),
                )
            )
    lines.extend(
        [
            "",
            "每個 family 的 quality 欄位是三個 training seed 的 fixed-evaluation mean 再取平均；seed spread 保留跨訓練隨機性的可見程度。SPS、wall-clock、VRAM 與 parameter count 是工程成本，不取代相同 transition budget 下的 quality 比較。",
            "",
            "## Optional 1M extension",
            "",
            f"- status: `{extension.get('status') if isinstance(extension, Mapping) else 'unavailable'}`",
            f"- triggered by the 500K rule: `{extension.get('triggered') if isinstance(extension, Mapping) else None}`",
            f"- completed entries: `{extension.get('completed_entry_count') if isinstance(extension, Mapping) else 0}/{extension.get('expected_entry_count') if isinstance(extension, Mapping) else 0}`",
            "",
            "The extension is reported separately from the 500K screening decision. It can replace the final family selection only after every selected top-two family has complete 1M CUDA evaluation evidence.",
            "",
            "| family | complete seeds | mean evaluation return | seed spread | mean SPS | mean wall-clock (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if isinstance(extension_aggregates, list):
        for aggregate in extension_aggregates:
            if not isinstance(aggregate, Mapping):
                continue
            lines.append(
                "| {family} | {complete}/{total} | {quality} | {spread} | {sps} | {wall} |".format(
                    family=aggregate.get("label", aggregate.get("family_id")),
                    complete=aggregate.get("formal_entry_count", 0),
                    total=aggregate.get("training_seed_count", 0),
                    quality=_display(aggregate.get("quality_mean")),
                    spread=_display(aggregate.get("quality_seed_spread")),
                    sps=_display(aggregate.get("mean_sps")),
                    wall=_display(aggregate.get("mean_wall_clock_seconds")),
                )
            )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- final-training family: `{selection.get('final_training_family')}`",
            f"- selection horizon: `{selection.get('selection_horizon')}` actual environment transitions",
            f"- deployment candidate: `{selection.get('deployment_candidate')}`",
            f"- winner above Contract v2 Random baseline: `{selection.get('winner_above_random_baseline')}`",
            f"- winner beats runner-up on every paired seed: `{selection.get('winner_beats_runner_up_on_every_seed')}`",
            f"- 1M extension applied to final selection: `{extension.get('applied') if isinstance(extension, Mapping) else None}`",
            "",
            "這個選擇不使用 best single episode、best single seed、training return 峰值、100K 分數或 GIF 外觀。若正式 evidence 尚未完整，selection 保持 `incomplete`；若所有 family 都沒有可靠超過 Random baseline，deployment candidate 會保持空值，而不是製造 `best.pt`。",
            "",
            "## Reproducible evidence",
            "",
            "- `assets/day20/evidence-reuse-audit.json` — Day 18 reuse checks",
            "- `assets/day20/dqn-family-training.png` — seed-level training curves",
            "- `assets/day20/dqn-family-evaluation.png` — fixed evaluation by milestone",
            "- `assets/day20/dqn-family-seed-spread.png` — 500K seed spread",
            "- `assets/day20/dqn-family-runtime-cost.png` — measured engineering cost",
            "- `assets/day20/family-comparison-flow.png` — staged execution/data-flow diagram",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Day 20 DQN-family comparison report."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/day20-dqn-family/manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/comparisons/dqn-family/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/day20-dqn-family-comparison.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("assets/day20/comparison-report.json"),
    )
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help="fail unless all nine 500K family/seed entries are complete and eligible",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_day20_config(args.config, require_probe_states=False)
        report = build_day20_report(args.manifest, config=config)
        write_json(args.json_output, report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_markdown(report), encoding="utf-8")
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"Day 20 report generation failed: {error}", file=sys.stderr)
        return 2
    conditions = report.get("comparison_conditions", {})
    formal_complete = (
        conditions.get("formal_completed_entry_count")
        == conditions.get("formal_expected_entry_count")
    )
    if args.require_formal and not formal_complete:
        print(
            "Day 20 formal report requires all family/seed entries at 500K; "
            f"got {conditions.get('formal_completed_entry_count')} of "
            f"{conditions.get('formal_expected_entry_count')}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "json": args.json_output.as_posix(),
                "report": args.output.as_posix(),
                "selection": report.get("selection"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
